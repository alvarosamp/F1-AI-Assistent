# F1 AI Race Engineer

Sistema de simulação e estratégia de corridas de Fórmula 1 com foco em Machine Learning, telemetria, MLOps e arquitetura híbrida de modelos.

## Visão geral

Este projeto evoluiu de um simulador simples por volta para uma plataforma de IA aplicada à Fórmula 1 com:

- coleta de dados reais com FastF1
- extração de telemetria por volta
- feature engineering avançado
- modelo global para generalização entre pistas
- modelos locais por pista para especialização
- comparação global vs local
- rastreamento de experimentos com MLflow
- otimização de hiperparâmetros com Optuna
- base para roteamento de inferência e simulação visual

## Objetivos

O sistema busca apoiar decisões como:

- previsão de tempo de volta
- análise de degradação de pneus
- suporte a decisões de pit stop
- comparação de comportamento entre pistas
- futura explicação por LLM local
- futura integração com simulação visual e estratégia

## Arquitetura do projeto

```text
FastF1
  ↓
Coleta de dados brutos
  ↓
Extração de telemetria por volta
  ↓
Build features avançado
  ↓
Treino global
  ↓
Treino local por pista
  ↓
Comparação global vs local
  ↓
Router de inferência
  ↓
Simulação / explicação / visualização
```

## Estrutura sugerida

```text
F1-AI-Assistent/
│
├── data/
│   ├── cache/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── per_track/
│   ├── per_track_optuna/
│   ├── model.pkl
│   ├── global_model_optuna.pkl
│   ├── fold_results.csv
│   ├── global_fold_results_optuna.csv
│   └── comparison.csv
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   ├── make_dataset_full_season.py
│   │   └── make_dataset_telemetry.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── train_global_optuna.py
│   │   ├── train_local_optuna.py
│   │   ├── train_per_track.py
│   │   ├── compare_global_vs_local.py
│   │   └── predict_router.py
│   │
│   └── simulation/
│       ├── track_real.py
│       └── simulate_full.py
│
├── mlruns/
├── requirements.txt
└── README.md
```

## Pipeline de dados

### 1. Coleta de dados
O projeto usa FastF1 para obter:

- tempos de volta
- setores
- pneus
- stint
- posição
- clima
- track status
- telemetria agregada por volta

Exemplo de execução:

```bash
python src/data/make_dataset_telemetry.py
```

### 2. Build features
O build de features gera variáveis como:

- `CompoundEncoded`
- `lap_time_mean_3`
- `lap_time_delta`
- `speed_delta`
- `throttle_delta`
- `brake_delta`
- `degradation_score`
- `aggression_score`
- `consistency_score`
- `efficiency_score`
- `drs_usage_intensity`
- `tyre_ratio`
- `stint_progress`
- `LapTimeResidual`

Exemplo de execução:

```bash
python src/features/build_features.py
```

## Modelagem

### Modelo global
O modelo global busca generalizar entre pistas.

- validação por `LeaveOneGroupOut`
- grupo: `gp`
- alvo: `LapTimeResidual`

Execução:

```bash
python src/models/train.py
```

### Modelo global com Optuna
Versão com ajuste de hiperparâmetros para o modelo global.

Execução:

```bash
python src/models/train_global_optuna.py
```

### Modelos locais por pista
Os modelos locais são especialistas por circuito.

Execução:

```bash
python src/models/train_per_track.py
```

### Modelos locais com Optuna
Ajuste fino por pista crítica.

Execução:

```bash
python src/models/train_local_optuna.py
```

## Comparação entre modelos

Após treinar o global e os locais, é possível comparar os resultados:

```bash
python src/models/compare_global_vs_local.py
```

Saídas esperadas:

- métricas globais por pista
- métricas locais por pista
- vencedor por pista
- ganho de RMSE, MAE e R²

## Router de inferência

O roteador escolhe automaticamente qual modelo usar:

- modelo local, quando existir para a pista
- modelo global, como fallback

Arquivo principal:

```text
src/models/predict_router.py
```

## MLOps

O projeto já utiliza:

- MLflow para tracking de experimentos
- Optuna para tuning
- organização de artefatos por modelo
- comparação entre estratégias globais e locais

Para abrir a interface do MLflow:

```bash
mlflow ui
```

## Resultados alcançados

O modelo global avançado com telemetria apresentou desempenho consistente na maioria das pistas, enquanto os modelos locais mostraram melhor desempenho em circuitos específicos, especialmente os mais peculiares.

Isso validou a arquitetura híbrida:

- global como fallback
- local como especialista

## Próximos passos

- corrigir e estabilizar o global com Optuna
- expandir modelos locais para mais pistas
- integrar `predict_router.py` à simulação
- adicionar explicação por LLM local
- construir dashboard de corrida
- evoluir a simulação visual em pista real
- estudar estratégia de pit stop mais avançada
- testar ensemble entre global e local

## Tecnologias usadas

- Python
- FastF1
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- MLflow
- Optuna
- Joblib
- Matplotlib

## Sugestão de fluxo de execução

```bash
python src/data/make_dataset_telemetry.py
python src/features/build_features.py
python src/models/train.py
python src/models/train_per_track.py
python src/models/compare_global_vs_local.py
```

## Autor

Projeto em evolução com foco em IA aplicada à Fórmula 1, telemetria, estratégia, simulação e MLOps.
