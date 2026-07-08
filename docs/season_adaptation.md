# Season adaptation and regulation-shift control

This project now has a current-season adaptation layer. It is designed for
seasons where the technical rules changed enough that old data may be stale.

## Why

The base simulator learned patterns from previous seasons. In a regulation
change, these can drift:

- team strength
- dirty air and overtaking
- tyre degradation
- DRS/active aero/energy deployment behaviour
- DNF and reliability rates

The adaptation layer does not retrain the model. It adjusts probabilities after
simulation so current-season evidence has more influence.

## Prediction with adaptation

```powershell
.venv\Scripts\python.exe -B src\predict_race_week.py `
  --gp "Australian Grand Prix" `
  --grid examples\race_week_grid.csv `
  --regulation-profile major_2026 `
  --current-form configs\season_adaptation_2026.csv `
  --sims 2000 `
  --quiet
```

Profiles:

- `stable`: use when rules are similar and model is trusted.
- `transition`: mild regulation/team-strength drift.
- `major_2026`: strong drift; shrinks historical confidence and boosts current form.

## Current form CSV

`configs/season_adaptation_2026.csv`:

```csv
driver,pace_delta,confidence,notes
NOR,-0.05,0.60,current-form placeholder
VER,0.00,0.50,neutral until current-season evidence is added
```

`pace_delta` is seconds per lap relative to what the historical model expects.

- Negative means the driver/team is faster than the historical model expects.
- Positive means slower than expected.
- `confidence` controls how strongly this should pull the forecast.

Good sources for `pace_delta`:

- FP2/FP3 long-run pace
- qualifying gap to pole
- recent race pace
- tyre degradation from current-weekend telemetry
- expert/manual adjustment after upgrades

## After each race

When real results are available, append/create a predictions CSV with the same
schema as `models/calibration_predictions_2024.csv`, then run:

```powershell
.venv\Scripts\python.exe -B src\analysis\drift_monitor.py `
  --predictions models\calibration_predictions_2026.csv `
  --season 2026 `
  --out reports\drift_monitor_2026.md
```

If two or more markets are `CRITICAL`, treat the model as out-of-regime until
current-season retraining/calibration is done.

## Full diagnostic study

```powershell
.venv\Scripts\python.exe -B src\analysis\regime_shift_study.py `
  --predictions models\calibration_predictions_2026.csv `
  --season 2026 `
  --out reports\regime_shift_study_2026.md
```

## Practical rule

Early in a new regulation season:

1. Use `major_2026`.
2. Keep recommendations conservative.
3. Update `configs/season_adaptation_2026.csv` after every race weekend.
4. Recalibrate once 4-6 races are available.
5. Retrain current-era encoders/model once enough telemetry exists.
