"""
Target Encoding CV-safe com smoothing bayesiano

Porque nao label encoding(cat.codes) ? 
    Label Encoding trata o codigo como ordinal: gp-code=5 parece estar 'entre' gp-code=4 e gp-code=6, o que nao faz sentido. Target encoding captura a relacao entre a categoria e o target (lap time), sem assumir ordenacao.
    Em GroupKFOld por GP, o modelo nunca viu os codigos de teste no treino, e a ordinalidade inventa relaçoes que nao existem. Testamos empiricamente:
    piorou o RMSE de 1.19 para 1.33 no dataset real.

Porque substituir target encoding ?
    Substitui a categoria pelo valor medio do target nas linhas de treino daquela categoria, com smoothing bayesiano contra a media global.
    O modelo passa a ver 'esse piloto' é o numero 17. Captura o sinal sistematico sem ordinalidade espuria.

Por que smoothing ?
    Categorias com poucas amostras( piloto que correu 2gp) tem media ruidosa. O smoothing puxa essas medias de volta para a media global
    proporcionalmente a quao poucas amostras elas tem:

Por que CV-safe ?
    Calcular target encoding no dataset inteiro vaza o target nas linhas de teste. A unica forma correta é : pra cada fold, calcular o encoding
    usando APENAS as linhas de treino daquele fold, e aplicar nas de teste. Isso é exatamente o que o target encodingfaz.
"""

from __future__ import annotations
import pandas as pd


def compute_target_encoding(
        train_df : pd.DataFrame,
        target_col : str,
    category_col : str,
        smoothing: float = 20.0,
) -> tuple[dict,float]:
    """
    Calcula o target encoding com smoothing bayesiano para uma categoria.

    Args:
        train_df: DataFrame de treino
        target_col: nome da coluna do target (ex: 'LapTime')
        category_col: nome da coluna da categoria a ser codificada (ex: 'gp')
        smoothing: parâmetro de smoothing, controla o peso da média global

    Returns:
        Um dicionário com o encoding e a média global do target
    """
    # Média global do target
    global_mean = train_df[target_col].mean()

    # Estatísticas por categoria
    stats = train_df.groupby(category_col)[target_col].agg(['mean', 'count'])
    smoothed = (
        stats['count'] * stats['mean'] + smoothing * global_mean
    ) / (stats['count'] + smoothing)

    # Dicionário de encoding
    encoding_dict = smoothed.to_dict()
    return encoding_dict, global_mean

def apply_target_encoding(
        df: pd.DataFrame,
        category_col: str,
        encoding_dict: dict,
        global_mean: float,
) -> pd.Series:
    """
    Aplica o target encoding a um DataFrame usando o dicionário de encoding.

    Args:
        df: DataFrame a ser transformado
        category_col: nome da coluna da categoria a ser codificada
        encoding_dict: dicionário com o encoding calculado no treino
        global_mean: média global do target para categorias não vistas

    Returns:
        DataFrame com a coluna codificada
    """
    # Aplica o encoding, usando a média global para categorias não vistas
    return df[category_col].map(encoding_dict).fillna(global_mean).astype(float)

class TargetEncoderCV:
    """
    Target encoder fold-safe para multiplas colunas categoricas

    Uso tipico dentro de um loop do CV:
        enc = TargetEncoderCV(cols = ['driver', 'team', 'gp'])
        train_enc = enc.transform(df.iloc[train_idx])
        test_enc = enc.transform(df.iloc[test_idx])

    Para uma inferencia em producao (modelo final trainado no dataset inteiro),
    basta um fit unico no dataset completo, nesse caso nao ha risco de leakage porque nao ha split train/test,
    e voce aplica em dados novos via transform()
    """
    def __init__(self, cols : list[str], smoothing: float = 20.0):
        self.cols = cols
        self.smoothing = smoothing
        self.encodings : dict[str, dict] = {}
        self.global_means : dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str | None = None,
        *,
        target: str | None = None,
    ) -> "TargetEncoderCV":
        if target_col is None:
            target_col = target
        elif target is not None and target != target_col:
            raise TypeError(
                "Provide either target_col positional OR target keyword, not both."
            )

        if target_col is None:
            raise TypeError("Missing required target column name via target_col or target=")

        for col in self.cols:
            mapping, gm = compute_target_encoding(
                train_df, target_col, col, self.smoothing
            )
            self.encodings[col] = mapping
            self.global_means[col] = gm
        return self
    
    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.cols:
            out[f"{col}_te"] = apply_target_encoding(
                df, col, self.encodings[col], self.global_means[col]
            )
        return out
    

    @property
    def feature_names(self) -> list[str]:
        return [f"{col}_te" for col in self.cols]