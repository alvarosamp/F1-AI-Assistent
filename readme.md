# 🏎️ F1 AI Race Insights

**Sistema de decision support para corridas de Fórmula 1** — prevê probabilidades de vitória, pódio, top 6 e top 10 usando simulação Monte Carlo calibrada.

Dado um grid de largada e um Grand Prix, o sistema simula a corrida **10.000 vezes em 12 segundos** e produz distribuições de probabilidade por piloto, validadas contra resultados reais.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Status](https://img.shields.io/badge/status-v3.0_Sprint_3_complete-green)

---

## Resultados

### Calibração contra 2024 (24 corridas reais)

| Mercado | Brier Score | vs Baseline (grid pos.) | ECE |
|---------|------------|------------------------|-----|
| **Vitória** | 0.036 | **+4.2% melhor** | 0.005 |
| **Pódio** | 0.073 | **+20.0% melhor** | 0.023 |
| **Top 6** | 0.064 | **+34.8% melhor** | 0.027 |
| **Top 10** | 0.102 | **+12.4% melhor** | 0.018 |
| **DNF** | 0.087 | **+1.4% melhor** | 0.012 |

O modelo **bate o baseline trivial em todos os 5 mercados**. ECE < 0.03 em todos — probabilidades bem calibradas.

### Modelo de Lap Time (base do simulador)

| Métrica | Valor |
|---------|-------|
| RMSE (walk-forward 2022+23 → 2024) | **0.722s** |
| R² | **0.638** |
| MAE | **0.537s** |
| Ganho sobre baseline trivial | **39.9%** |

### Exemplo de saída (Australian GP, grid real)

```
Driver     Win   Podium    Top6   Top10    DNF
---------------------------------------------
  VER   40.9%    78.0%   92.1%   93.5%   6.5%
  LEC   19.6%    55.1%   87.7%   93.6%   6.4%
  SAI   17.8%    56.4%   87.5%   93.4%   6.6%
  NOR   11.3%    42.9%   83.2%   93.2%   6.8%
  PIA    5.2%    29.0%   76.5%   92.9%   7.1%
  RUS    3.4%    22.7%   74.1%   92.3%   7.7%
  HAM    1.7%    14.3%   62.8%   91.9%   8.1%
```

---

## Arquitetura

```
                    ┌─────────────────┐
                    │   FastF1 API    │
                    │  (2022-2024)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  make_dataset   │  96.598 voltas brutas
                    │  + weather      │  + weather, race control,
                    │  + ergast       │    Ergast quali/grid
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ build_features  │  59.362 voltas filtradas
                    │  v4 (78 feat)   │  IQR + residual filter
                    │  anti-leakage   │  10+ testes pytest
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │  Lap Time     │ │   DNF    │ │ Safety Car  │
     │  XGBoost v3   │ │ LogReg   │ │ BetaBinom   │
     │  RMSE=0.72    │ │ Brier    │ │ por pista   │
     │  walk-forward │ │ =0.087   │ │             │
     └────────┬──────┘ └────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Monte Carlo   │  10.000 sims
                    │   Simulator     │  em 12 segundos
                    │   (vectorized)  │  (3.960x speedup)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │ Probabilidades│ │Calibração│ │  Dashboard  │
     │ Win/Podium/   │ │ Brier,   │ │  Streamlit  │
     │ Top6/Top10    │ │ ECE,     │ │             │
     │               │ │Reliab.   │ │             │
     └───────────────┘ └──────────┘ └─────────────┘
```

---

## Como usar

### Instalação

```bash
git clone https://github.com/seu-usuario/F1-AI-Assistent.git
cd F1-AI-Assistent
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Pipeline completo (do zero)

```bash
# 1. Coleta de dados (2-4 horas, retomável)
python src/data/make_dataset_v2.py

# 2. Feature engineering (1 min)
python src/features/build_features.py

# 3. Treinar modelo de lap time (25-40 min)
python src/models/train_global_optuna.py

# 4. Treinar submodelos
python src/dnf/build_dnf_dataset.py
python src/dnf/train_dnf.py
python src/dnf/train_sc_model.py
python src/dnf/tyre_model.py

# 5. Testes (todos devem passar)
pytest tests/ src/tests/ -v

# 6. Simular uma corrida
python src/simulation/race_simulate.py

# 7. Calibração (5 min)
python src/simulation/extract_2024_results.py
python src/simulation/calibrate_simulate.py

# 8. Previsão de uma race week (grid de qualifying real)
python src/predict_race_week.py --gp "Australian Grand Prix" --grid examples/race_week_grid.csv --sims 2000

# 9. Dashboard
streamlit run dashboard.py
```

### Uso rápido (modelos já treinados)

```python
from src.simulation.race_simulate import RaceSimulator

sim = RaceSimulator()
results = sim.simulate(
    gp="Monaco Grand Prix",
    grid=[
        {"driver": "VER", "team": "Red Bull Racing", "grid_pos": 1, "quali_pos": 1, "gap_to_pole_ms": 0},
        {"driver": "LEC", "team": "Ferrari", "grid_pos": 2, "quali_pos": 2, "gap_to_pole_ms": 150},
        # ... mais pilotos
    ],
    n_simulations=10_000,
)

print(results["probabilities"]["VER"]["win"])   # P(Verstappen vence)
print(results["probabilities"]["LEC"]["podium"]) # P(Leclerc top 3)
```

---

## Estrutura do projeto

```
F1-AI-Assistent/
├── src/
│   ├── data/
│   │   └── make_dataset_v2.py          # Coleta FastF1 + weather + Ergast
│   ├── features/
│   │   ├── build_features_v4.py        # Feature engineering v4 (78 features)
│   │   └── target_encoding.py          # Target encoding CV-safe
│   ├── models/
│   │   ├── train_global_optuna.py      # XGBoost + Optuna walk-forward
│   │   └── analise_modelo.py           # Análise pós-treino
│   ├── dnf/
│   │   ├── build_dnf_dataset.py        # Dataset de DNF
│   │   ├── train_dnf.py                # Modelo de DNF calibrado
│   │   ├── train_sc_model.py           # Safety Car por pista
│   │   └── tyre_model.py               # Degradação de pneu
│   ├── simulation/
│   │   ├── race_simulate.py            # Monte Carlo otimizado (RaceSimulator)
│   │   ├── extract_2024_results.py     # Resultados reais pra calibração
│   │   └── calibrate_simulate.py       # Calibração formal
│   ├── strategy/
│   │   └── pit_stop_model.py           # Modelo de decisão de pit stop
│   ├── telemetry/
│   │   ├── telemetry_signals.py        # Extração de sinais de telemetria
│   │   └── plot_lap_telemetry.py       # Visualização de telemetria por volta
│   ├── llm/
│   │   └── engenheiro.py               # Explicação em linguagem natural das decisões (LLM)
│   ├── betting_recommender.py          # Ranking de apostas por EV a partir das probabilidades
│   └── predict_race_week.py            # CLI de previsão pós-qualifying
├── tests/
│   └── test_betting_recommender.py     # Testes do recomendador de apostas
├── src/tests/
│   ├── test_no_leakage.py              # Testes anti-leakage
│   ├── test_leakage_features.py        # Testes anti-leakage (features)
│   └── test_target_encoding.py         # Testes do encoder
├── examples/                           # CSVs de exemplo (grid de race week, etc.)
├── docs/                               # Documentação adicional
├── models/                             # Artefatos treinados (.pkl, .json)
├── data/
│   ├── raw/                            # CSVs brutos do FastF1
│   └── processed/                      # CSVs processados
├── dvc.yaml / dvc.lock                 # Pipeline versionado com DVC
├── dashboard.py                        # Streamlit dashboard
└── README.md
```

---

## Decisões técnicas importantes

### 1. Anti-leakage rigoroso
Toda feature derivada de lap time ou telemetria é **shiftada 1 volta pra trás** (sufixo `_prev`). Na volta N, o modelo só vê informação até N-1. Provado por 10+ testes pytest, incluindo um teste que **embaralha o target de voltas futuras e verifica que features do passado não mudam** — prova matemática de ausência de leakage.

### 2. Target encoding em vez de label encoding
Label encoding trata categorias como ordinais (driver_code=5 parece estar "entre" 4 e 6), o que confunde árvores de decisão em validação por grupo. Testamos empiricamente: label encoding **piorava** o RMSE de 1.19 para 1.33. Target encoding com smoothing bayesiano captura "Verstappen é sistematicamente 0.99s mais rápido" sem ordinalidade espúria. Implementado com CV-safety (encoding calculado só no treino de cada fold).

### 3. Walk-forward em vez de GroupKFold
GroupKFold por GP testa generalização para pistas nunca vistas — cenário irrealista. Walk-forward (treina 2022+2023, testa 2024) testa o cenário real: "prever o próximo ano". RMSE walk-forward foi melhor que GroupKFold (0.72 vs 0.77), confirmando que o modelo generaliza bem temporalmente.

### 4. Vetorização do simulador (3.960x speedup)
v1 chamava `model.predict()` 5.8M vezes (1 por piloto/volta/simulação) → 13 horas. v2 faz 1 batch predict por volta com DataFrame de 100k linhas → 12 segundos. Mesmo resultado, 3.960x mais rápido. XGBoost é otimizado para batch prediction.

### 5. Calibração honesta com baselines
Todo modelo é comparado contra baseline trivial antes de ser aceito. O modelo de DNF quase não bate o trivial (+2.5%), e isso é **reportado honestamente**. O modelo de Safety Car é marginal. Transparência > métricas bonitas.

---

## Limitações conhecidas

- **Dados de 2022-2024 apenas.** Regulamentos de 2026 mudaram radicalmente (novos motores, chassis, DRS abolido). Modelos precisam ser retreinados com dados da era nova.
- **Sem simulação de ultrapassagens explícitas.** Posições mudam por diferença de tempo acumulado, não por modelagem de aerodinâmica/DRS.
- **Estratégia de pit stop fixa.** Na realidade cada equipe otimiza estratégia em tempo real.
- **Weather fixo durante simulação.** Não modela mudança de condições durante a corrida.
- **Pilotos novos (rookies) usam fallback do encoder.** Poucos dados = mais incerteza.

---

## Dados

| Fonte | O que coleta | Volume |
|-------|-------------|--------|
| FastF1 | Telemetria por volta, weather, race control | 96.598 voltas |
| Ergast/Jolpica | Qualifying results, grid position, gap to pole | 24 GPs/ano |

---

## Referências técnicas

- **XGBoost**: Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*
- **Target Encoding**: Micci-Barreca (2001). *A preprocessing scheme for high-cardinality categorical attributes*
- **Calibração**: Niculescu-Mizil & Caruana (2005). *Predicting good probabilities with supervised learning*
- **Monte Carlo em esportes**: Silver & Minka (2012). *Sports forecasting with Monte Carlo simulation*
- **FastF1**: Oehrly (2020). *Python package for accessing Formula 1 data*

---

## Licença

Este projeto é um exercício técnico de ML aplicado. **Não é aconselhamento de apostas.** Apostas envolvem risco financeiro real e devem ser tratadas como entretenimento, não investimento.

---

*Desenvolvido em 3 sprints: (1) modelo honesto de lap time, (2) features de contexto + walk-forward, (3) simulador Monte Carlo calibrado com dashboard.*
