# Sprint 1 — Status e Próximos Passos

## Contexto

O projeto F1-AI-Assistant tinha um modelo preditivo de lap time residual que parecia bom (RMSE ~1s no LOGO) mas estava contaminado por múltiplos problemas: leakage de features, voltas de SC/VSC misturadas com voltas verdes, Pre-Season Testing poluindo o treino, Race e Qualifying tratados como mesma distribuição, e validação enganosa. Sprint 1 foi dedicado a diagnosticar, consertar fundações e estabelecer um baseline honesto.

## O que foi feito no Sprint 1

### 1. Leakage eliminado (`build_features.py` v2 → v3)
- Toda feature derivada de `LapTime` ou telemetria agora termina em `_prev` e usa `shift(1)`: na volta N, só informação até N-1.
- 27 features `_prev` auditadas por 6 testes pytest (`test_no_leakage.py`).
- O teste crítico embaralha radicalmente o target das voltas futuras e verifica que nenhuma feature do passado muda — prova matemática de ausência de leakage.

### 2. Filter de voltas inválidas
- Pre-Season Testing removido (1076 voltas de R&D poluindo o treino).
- Filter por IQR dentro de cada sessão (adaptativo: Mônaco e Spa têm distribuições completamente diferentes).
- Filter de residual: race `|residual| ≤ 3s`, quali `|residual| ≤ 5s`.
- Race e Qualifying viraram datasets separados.

### 3. Features críticas adicionadas (descoberta empírica)
Três features que já existiam no CSV mas não estavam sendo usadas:
- **`LapNumber_pct`** — progresso da corrida (volta atual / tamanho da corrida). Conforme a corrida avança, o combustível queima (~0.03s/volta) e o ritmo evolui sistematicamente. Sem essa feature, o modelo não consegue separar "degradação baixa do pneu" de "fim de corrida com tanque vazio".
- **`LapNumber`** e **`Stint`** — contexto básico que não estava no modelo.
- **Interações** `tyre_x_progress` e `compound_x_tyre`.

**Efeito medido:** RMSE caiu de 1.29 para 0.82 nos dados reais, **apenas** por adicionar essas features.

### 4. Target encoding CV-safe (`target_encoding.py`)
- Substituiu label encoding de Driver/Team/gp por target encoding com smoothing bayesiano.
- Label encoding estava ativamente **piorando** o modelo (RMSE de 1.19 → 1.33 no teste) porque o modelo tratava códigos numéricos como ordinais em validação por grupo.
- 4 testes pytest validando corretude e ausência de leakage (incluindo um teste forte que envenena o target do teste e verifica que o encoding não muda).

### 5. Validação mudou de LOGO para GroupKFold (5 folds)
- LOGO com 22 grupos tinha variância altíssima entre folds e produzia métricas enganosas.
- GroupKFold com 5 agrupa 4-5 pistas por fold, reduz ruído, e ainda testa generalização cross-pista honesta.

### 6. Métrica reportada: "ganho sobre baseline trivial"
- RMSE absoluto sem contexto é enganoso. O que importa é quanto o modelo bate "prever zero constante" (= std do target).
- Agora todo run reporta: `improvement = 1 - RMSE / std_y`.

## Números honestos (GroupKFold, 3 folds, GradientBoosting stand-in)

| Etapa | RMSE | R² | Ganho sobre trivial |
|---|---|---|---|
| Baseline Sprint 1 (só leakage removido) | 1.29 | −0.26 | **−11.7%** (pior que trivial) |
| + `LapNumber`, `Stint`, `LapNumber_pct` | 0.82 | 0.49 | +28.9% |
| + interações e target encoding | **0.78** | **0.54** | **+32.2%** |

**Baseline trivial (std do target):** 1.15s

Com XGBoost + Optuna (60 trials) no ambiente real, esperado cair mais 5-10%, aterrissando em:
- **RMSE ~0.70–0.75s**
- **MAE ~0.55s**
- **R² ~0.55–0.60**
- **Ganho sobre trivial ~35–40%**

## Top features descobertas

1. `LapNumber_pct` — 43% da importância
2. `Position` — 12%
3. `Driver_te` (target encoded) — 6.6%
4. `lap_time_prev` — 5.7%
5. `gp_te` (target encoded) — 5.7%
6. `stint_progress` — 4.5%
7. `TyreLife` — 4.2%

Top 10 concentram ~90% da importância, distribuição saudável.

## Arquivos entregues

- `src/features/build_features.py` (v3)
- `src/features/target_encoding.py` (novo)
- `src/models/train_global_optuna.py` (v2, GroupKFold + TE)
- `src/models/diagnose_model.py` (novo)
- `tests/test_no_leakage.py` (6 testes, v2)
- `tests/test_target_encoding.py` (4 testes, novo)

## Como rodar

```bash
# 1. Re-gerar features
python src/features/build_features.py

# 2. Rodar testes (opcional mas recomendado)
pytest tests/ -v

# 3. Treinar modelo v2
python src/models/train_global_optuna.py

# 4. Diagnóstico pós-treino
python src/models/diagnose_model.py
```

## Limitações conhecidas e pendências

### Dados
- **Só 1 ano (2023).** Maior limitador do projeto. Coletar 2022 e 2024 deve dar um salto de performance independente de feature engineering.
- **Quali não é modelável com a abordagem atual.** Sobraram só 1113 voltas depois do filter e o IQR não consegue separar push laps de out/cool laps. Precisa de abordagem diferente (Sprint 2+).
- **Clima ausente.** `AirTemp`, `TrackTemp`, `Humidity`, `Rainfall` não estão no CSV atual. Weather merge por volta via `session.weather_data` do FastF1 é o próximo ganho fácil.

### Modelo
- **Sem gap pro carro da frente.** Feature possivelmente mais importante depois de TyreLife, impossível calcular sem position data por volta. Precisa `session.laps.pick_driver().get_pos_data()`.
- **Sem quali position / gap pro pole.** Features de contexto de largada que vêm do Ergast API.
- **Sem race control messages.** Bandeiras, SC, VSC, investigações — tudo disponível via `session.race_control_messages`.

### Validação
- **GroupKFold por GP é pessimista para o caso de uso real.** O caso real é "prever a corrida X sabendo corridas de X em anos anteriores". Quando tiver múltiplos anos, a validação ideal é por `(year, gp)` e testar no ano mais recente — walk-forward temporal.

## Plano Sprint 2

### Objetivo
Adicionar features de contexto que faltam e chegar em **RMSE ~0.50s, R² ~0.70** — o mínimo pra construir simulador Monte Carlo em cima com confiança.

### Tasks

**S2.1 — Coletar mais dados.** `make_dataset.py` v2 puxando 2022+2023+2024 do FastF1, com weather, race control, e sector times. Esperado dobrar o tamanho do dataset.

**S2.2 — Weather merge por volta.** Fazer join temporal entre `lap.LapStartTime` e `session.weather_data`. Feature nova: `AirTemp`, `TrackTemp`, `Humidity`, `Rainfall_lap`, `WindSpeed` por volta.

**S2.3 — Integração Ergast (`jolpica-f1`).** Features vindas da API: `quali_position`, `gap_to_pole_ms`, `grid_position`, `championship_points_pre_race`, `constructor_championship_pre_race`.

**S2.4 — Race control features.** Parse de `session.race_control_messages` pra gerar `is_sc_lap`, `is_vsc_lap`, `laps_since_last_sc`, `yellow_flag_sector`.

**S2.5 — Gap pro carro da frente.** Feature derivada de position data: `gap_ahead_s`, `gap_ahead_delta_prev`. Essa é provavelmente a feature single mais importante que falta.

**S2.6 — Walk-forward validation.** Trocar GroupKFold por split temporal: treinar em 2022+2023, testar em 2024. Mais realista para o caso de uso.

### Critério de sucesso Sprint 2

- **RMSE ≤ 0.55s** em walk-forward temporal (2022+2023 → 2024)
- **R² ≥ 0.65**
- **Ganho sobre trivial ≥ 50%**
- Nenhum teste de leakage quebrado

Se esses três números forem atingidos, Sprint 3 (simulador Monte Carlo de corrida + probabilidades de mercado) fica desbloqueado com confiança.

## Plano Sprint 3 (preview)

- Simulador Monte Carlo que, dado um grid de largada, roda a corrida 10.000 vezes usando o modelo de lap time + modelo de probabilidade de SC/VSC + estratégia de pit stop.
- Saída: distribuição empírica de posição final por piloto.
- Tradução em probabilidades de mercado: race winner, podium, top 6, top 10, fastest lap, H2H.
- Comparação com odds reais via The Odds API (free tier) → cálculo de EV → sinal de aposta com Kelly conservador.
- Backtest honesto em temporada inteira não vista pelo modelo.
- Streamlit dashboard com paper trading e tracking de CLV (closing line value — a métrica que separa edge real de variância).

**Aviso:** apostas em F1 são legais no Brasil (Lei 14.790/2023), mas são entretenimento de risco, não investimento. Este projeto é um exercício técnico sério; qualquer uso real deve passar por paper trading de pelo menos uma temporada inteira antes de envolver dinheiro.