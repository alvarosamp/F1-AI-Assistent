# Regime-shift and accuracy study - 2026

## Executive summary

- Rows analysed: 198 driver-race predictions across 9 races.
- Top predicted winner hit rate: 66.7%.
- High-confidence win picks (`pred_win >= 40%`): predicted mean 52.2% vs real 66.7%.
- Main risk for a new regulation season: the simulator is calibrated on the previous technical era, so relationships learned from grid, tyre behaviour, DRS/dirty air and team strength can drift.

## Market metrics

| market | n   | base_rate | pred_mean | brier | log_loss | ece   |
| ------ | --- | --------- | --------- | ----- | -------- | ----- |
| win    | 198 | 0.045     | 0.045     | 0.020 | 0.065    | 0.016 |
| podium | 198 | 0.136     | 0.136     | 0.072 | 0.275    | 0.064 |
| top6   | 198 | 0.273     | 0.273     | 0.113 | 0.450    | 0.098 |
| top10  | 198 | 0.455     | 0.455     | 0.202 | 0.758    | 0.154 |
| dnf    | 198 | 0.212     | 0.088     | 0.180 | 0.573    | 0.124 |

## Calibration report comparison

| market | brier_model | brier_baseline | improvement_pct | ece   |
| ------ | ----------- | -------------- | --------------- | ----- |
| win    | 0.035       | 0.038          | 6.165           | 0.011 |
| podium | 0.081       | 0.091          | 11.149          | 0.060 |
| top6   | 0.077       | 0.098          | 21.359          | 0.034 |
| top10  | 0.115       | 0.117          | 1.135           | 0.072 |
| dnf    | 0.087       | 0.089          | 1.381           | 0.012 |

## Worst races by win Brier

| gp                    | n  | win_brier | podium_brier | top6_brier | top10_brier | dnf_brier |
| --------------------- | -- | --------- | ------------ | ---------- | ----------- | --------- |
| Barcelona Grand Prix  | 22 | 0.034     | 0.039        | 0.056      | 0.172       | 0.186     |
| British Grand Prix    | 22 | 0.033     | 0.056        | 0.103      | 0.216       | 0.082     |
| Canadian Grand Prix   | 22 | 0.032     | 0.132        | 0.206      | 0.322       | 0.233     |
| Monaco Grand Prix     | 22 | 0.023     | 0.092        | 0.216      | 0.312       | 0.232     |
| Miami Grand Prix      | 22 | 0.017     | 0.113        | 0.056      | 0.145       | 0.155     |
| Chinese Grand Prix    | 22 | 0.013     | 0.016        | 0.132      | 0.238       | 0.266     |
| Austrian Grand Prix   | 22 | 0.013     | 0.094        | 0.103      | 0.025       | 0.151     |
| Japanese Grand Prix   | 22 | 0.010     | 0.062        | 0.005      | 0.149       | 0.079     |
| Australian Grand Prix | 22 | 0.010     | 0.047        | 0.142      | 0.242       | 0.234     |

## Grid bucket bias

| grid_bucket | n   | win_pred | win_real | win_gap | podium_pred | podium_real | podium_gap | top6_pred | top6_real | top6_gap | top10_pred | top10_real | top10_gap | dnf_pred | dnf_real | dnf_gap |
| ----------- | --- | -------- | -------- | ------- | ----------- | ----------- | ---------- | --------- | --------- | -------- | ---------- | ---------- | --------- | -------- | -------- | ------- |
| P1-P3       | 27  | 0.316    | 0.333    | -0.017  | 0.802       | 0.593       | 0.209      | 0.933     | 0.704     | 0.230    | 0.935      | 0.778      | 0.157     | 0.065    | 0.148    | -0.083  |
| P4-P6       | 27  | 0.017    | 0.000    | 0.017   | 0.195       | 0.333       | -0.139     | 0.857     | 0.741     | 0.116    | 0.930      | 0.778      | 0.152     | 0.070    | 0.148    | -0.078  |
| P7-P10      | 36  | 0.000    | 0.000    | 0.000   | 0.003       | 0.056       | -0.053     | 0.156     | 0.389     | -0.233   | 0.821      | 0.639      | 0.182     | 0.081    | 0.167    | -0.086  |
| P11+        | 108 | 0.000    | 0.000    | 0.000   | 0.000       | 0.000       | 0.000      | 0.001     | 0.009     | -0.009   | 0.093      | 0.231      | -0.138    | 0.101    | 0.259    | -0.159  |

## Top predicted winner misses

| gp                   | model_pick | pick_grid | pick_win_prob | pick_result | real_winner | hit   |
| -------------------- | ---------- | --------- | ------------- | ----------- | ----------- | ----- |
| Barcelona Grand Prix | RUS        | 1         | 0.499         | 2           | HAM         | False |
| British Grand Prix   | ANT        | 1         | 0.477         | 15          | LEC         | False |
| Canadian Grand Prix  | RUS        | 1         | 0.427         | 99          | ANT         | False |

## Largest individual errors

### Win

| gp                   | driver | grid_pos | pred_win | real_win | real_position | real_dnf | abs_error |
| -------------------- | ------ | -------- | -------- | -------- | ------------- | -------- | --------- |
| Canadian Grand Prix  | ANT    | 2        | 0.307    | 1        | 1             | 0        | 0.693     |
| Barcelona Grand Prix | HAM    | 2        | 0.308    | 1        | 1             | 0        | 0.692     |
| British Grand Prix   | LEC    | 2        | 0.315    | 1        | 1             | 0        | 0.685     |
| Monaco Grand Prix    | ANT    | 1        | 0.410    | 1        | 1             | 0        | 0.590     |
| Barcelona Grand Prix | RUS    | 1        | 0.499    | 0        | 2             | 0        | 0.499     |
| Miami Grand Prix     | ANT    | 1        | 0.502    | 1        | 1             | 0        | 0.498     |
| British Grand Prix   | ANT    | 1        | 0.477    | 0        | 15            | 0        | 0.477     |
| Austrian Grand Prix  | RUS    | 1        | 0.551    | 1        | 1             | 0        | 0.449     |
| Chinese Grand Prix   | ANT    | 1        | 0.565    | 1        | 1             | 0        | 0.435     |
| Canadian Grand Prix  | RUS    | 1        | 0.427    | 0        | 99            | 1        | 0.427     |

### Podium

| gp                  | driver | grid_pos | pred_podium | real_podium | real_position | real_dnf | abs_error |
| ------------------- | ------ | -------- | ----------- | ----------- | ------------- | -------- | --------- |
| Monaco Grand Prix   | GAS    | 9        | 0.000       | 1           | 3             | 0        | 1.000     |
| Miami Grand Prix    | PIA    | 7        | 0.003       | 1           | 3             | 0        | 0.997     |
| Canadian Grand Prix | VER    | 6        | 0.097       | 1           | 3             | 0        | 0.903     |
| British Grand Prix  | ANT    | 1        | 0.898       | 0           | 15            | 0        | 0.898     |
| Canadian Grand Prix | RUS    | 1        | 0.884       | 0           | 99            | 1        | 0.884     |
| Japanese Grand Prix | RUS    | 2        | 0.871       | 0           | 4             | 0        | 0.871     |
| Canadian Grand Prix | HAM    | 5        | 0.164       | 1           | 2             | 0        | 0.836     |
| Austrian Grand Prix | VER    | 5        | 0.168       | 1           | 2             | 0        | 0.832     |
| Monaco Grand Prix   | VER    | 2        | 0.823       | 0           | 99            | 1        | 0.823     |
| Miami Grand Prix    | VER    | 2        | 0.802       | 0           | 5             | 0        | 0.802     |

### Top 10

| gp                    | driver | grid_pos | pred_top10 | real_top10 | real_position | real_dnf | abs_error |
| --------------------- | ------ | -------- | ---------- | ---------- | ------------- | -------- | --------- |
| Monaco Grand Prix     | ALO    | 21       | 0.000      | 1          | 10            | 0        | 1.000     |
| Australian Grand Prix | VER    | 20       | 0.001      | 1          | 6             | 0        | 0.999     |
| British Grand Prix    | COL    | 19       | 0.001      | 1          | 9             | 0        | 0.999     |
| Chinese Grand Prix    | SAI    | 17       | 0.003      | 1          | 9             | 0        | 0.997     |
| Miami Grand Prix      | ALB    | 15       | 0.011      | 1          | 10            | 0        | 0.989     |
| Monaco Grand Prix     | OCO    | 17       | 0.014      | 1          | 9             | 0        | 0.986     |
| Canadian Grand Prix   | BEA    | 16       | 0.021      | 1          | 10            | 0        | 0.979     |
| Canadian Grand Prix   | SAI    | 15       | 0.036      | 1          | 9             | 0        | 0.964     |
| Monaco Grand Prix     | LIN    | 15       | 0.038      | 1          | 7             | 0        | 0.962     |
| Chinese Grand Prix    | LAW    | 14       | 0.041      | 1          | 7             | 0        | 0.959     |

## Regulation-change risks for the current season

1. **Power-unit and energy-management drift**: if the season has a different hybrid/ERS behaviour, historical lap-time residuals and straight-line performance features can become stale.
2. **Aero/dirty-air drift**: if active aero or changed car dimensions affect following distance and overtaking, the current traffic and overtake penalties may be miscalibrated.
3. **Tyre degradation drift**: narrower/different tyres or new operating windows can change stint degradation; this directly affects pit-stop strategy and Monte Carlo race pace.
4. **Team-strength reset**: a major regulation change can reshuffle competitive order; target encoding by driver/team can overvalue old dominant teams.
5. **Track-specific bias**: the worst-GP table should be monitored first. If the same circuits stay bad across seasons, track modelling is weak; if new bad races appear, it is likely regime drift.

## Recommended fixes

- Add a `season_weight` or time-decay retraining mode so current-season races matter more than old regulation-era data.
- Add an explicit `regulation_era` flag and avoid mixing eras without interaction features.
- Retrain target encoders using only current-era data once at least 4-6 races exist.
- Calibrate market probabilities per season, not just globally.
- Add a drift dashboard: compare predicted vs real win/podium/top10 after every race.
- For strategy, do not trust historical pit counts directly; estimate tyre degradation from current FP long runs, then simulate strategy windows.

## How to use this report

Use this file as the baseline diagnostic. When you have current-season real results, generate a new `calibration_predictions_<season>.csv` with the same columns and rerun this script.
