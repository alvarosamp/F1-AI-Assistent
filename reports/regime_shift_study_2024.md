# Regime-shift and accuracy study - 2024

## Executive summary

- Rows analysed: 478 driver-race predictions across 24 races.
- Top predicted winner hit rate: 50.0%.
- High-confidence win picks (`pred_win >= 40%`): predicted mean 53.2% vs real 48.0%.
- Main risk for a new regulation season: the simulator is calibrated on the previous technical era, so relationships learned from grid, tyre behaviour, DRS/dirty air and team strength can drift.

## Market metrics

| market | n   | base_rate | pred_mean | brier | log_loss | ece   |
| ------ | --- | --------- | --------- | ----- | -------- | ----- |
| win    | 478 | 0.050     | 0.050     | 0.035 | 0.158    | 0.011 |
| podium | 478 | 0.151     | 0.151     | 0.081 | 0.344    | 0.060 |
| top6   | 478 | 0.301     | 0.301     | 0.077 | 0.357    | 0.034 |
| top10  | 478 | 0.502     | 0.502     | 0.115 | 0.448    | 0.072 |
| dnf    | 478 | 0.098     | 0.086     | 0.087 | 0.314    | 0.012 |

## Calibration report comparison

| market | brier_model | brier_baseline | improvement_pct | ece   |
| ------ | ----------- | -------------- | --------------- | ----- |
| win    | 0.035       | 0.038          | 6.165           | 0.011 |
| podium | 0.081       | 0.091          | 11.149          | 0.060 |
| top6   | 0.077       | 0.098          | 21.359          | 0.034 |
| top10  | 0.115       | 0.117          | 1.135           | 0.072 |
| dnf    | 0.087       | 0.089          | 1.381           | 0.012 |

## Worst races by win Brier

| gp                       | n  | win_brier | podium_brier | top6_brier | top10_brier | dnf_brier |
| ------------------------ | -- | --------- | ------------ | ---------- | ----------- | --------- |
| Belgian Grand Prix       | 20 | 0.085     | 0.207        | 0.066      | 0.041       | 0.046     |
| São Paulo Grand Prix     | 19 | 0.080     | 0.282        | 0.199      | 0.170       | 0.176     |
| Austrian Grand Prix      | 20 | 0.070     | 0.156        | 0.042      | 0.084       | 0.008     |
| United States Grand Prix | 20 | 0.065     | 0.058        | 0.019      | 0.227       | 0.046     |
| Miami Grand Prix         | 20 | 0.065     | 0.070        | 0.079      | 0.175       | 0.047     |
| Italian Grand Prix       | 20 | 0.057     | 0.046        | 0.072      | 0.097       | 0.047     |
| Azerbaijan Grand Prix    | 20 | 0.048     | 0.043        | 0.095      | 0.118       | 0.089     |
| Australian Grand Prix    | 19 | 0.042     | 0.122        | 0.109      | 0.154       | 0.096     |
| British Grand Prix       | 20 | 0.038     | 0.066        | 0.076      | 0.095       | 0.089     |
| Hungarian Grand Prix     | 20 | 0.037     | 0.083        | 0.011      | 0.214       | 0.048     |

## Grid bucket bias

| grid_bucket | n   | win_pred | win_real | win_gap | podium_pred | podium_real | podium_gap | top6_pred | top6_real | top6_gap | top10_pred | top10_real | top10_gap | dnf_pred | dnf_real | dnf_gap |
| ----------- | --- | -------- | -------- | ------- | ----------- | ----------- | ---------- | --------- | --------- | -------- | ---------- | ---------- | --------- | -------- | -------- | ------- |
| P1-P3       | 72  | 0.316    | 0.264    | 0.052   | 0.804       | 0.611       | 0.193      | 0.933     | 0.875     | 0.058    | 0.934      | 0.958      | -0.024    | 0.066    | 0.042    | 0.024   |
| P4-P6       | 72  | 0.018    | 0.042    | -0.024  | 0.193       | 0.306       | -0.112     | 0.855     | 0.792     | 0.064    | 0.928      | 0.958      | -0.030    | 0.072    | 0.000    | 0.072   |
| P7-P10      | 96  | 0.000    | 0.010    | -0.010  | 0.002       | 0.031       | -0.029     | 0.155     | 0.188     | -0.032   | 0.799      | 0.729      | 0.070     | 0.081    | 0.083    | -0.002  |
| P11+        | 238 | 0.000    | 0.004    | -0.004  | 0.000       | 0.013       | -0.013     | 0.001     | 0.025     | -0.024   | 0.123      | 0.134      | -0.012    | 0.099    | 0.151    | -0.052  |

## Top predicted winner misses

| gp                       | model_pick | pick_grid | pick_win_prob | pick_result | real_winner | hit   |
| ------------------------ | ---------- | --------- | ------------- | ----------- | ----------- | ----- |
| Belgian Grand Prix       | VER        | 1         | 0.827         | 5           | RUS         | False |
| Austrian Grand Prix      | VER        | 1         | 0.707         | 5           | RUS         | False |
| Azerbaijan Grand Prix    | LEC        | 1         | 0.607         | 2           | PIA         | False |
| São Paulo Grand Prix     | NOR        | 1         | 0.590         | 6           | VER         | False |
| Australian Grand Prix    | VER        | 1         | 0.540         | 99          | SAI         | False |
| United States Grand Prix | NOR        | 1         | 0.502         | 3           | LEC         | False |
| Spanish Grand Prix       | NOR        | 1         | 0.484         | 2           | VER         | False |
| Miami Grand Prix         | VER        | 1         | 0.451         | 2           | NOR         | False |
| Hungarian Grand Prix     | NOR        | 1         | 0.446         | 2           | PIA         | False |
| British Grand Prix       | RUS        | 1         | 0.441         | 99          | HAM         | False |
| Italian Grand Prix       | NOR        | 1         | 0.423         | 3           | LEC         | False |
| Canadian Grand Prix      | RUS        | 1         | 0.406         | 3           | VER         | False |

## Largest individual errors

### Win

| gp                       | driver | grid_pos | pred_win | real_win | real_position | real_dnf | abs_error |
| ------------------------ | ------ | -------- | -------- | -------- | ------------- | -------- | --------- |
| São Paulo Grand Prix     | VER    | 12       | 0.000    | 1        | 1             | 0        | 1.000     |
| Belgian Grand Prix       | RUS    | 7        | 0.000    | 1        | 1             | 0        | 1.000     |
| Miami Grand Prix         | NOR    | 5        | 0.011    | 1        | 1             | 0        | 0.990     |
| United States Grand Prix | LEC    | 4        | 0.036    | 1        | 1             | 0        | 0.964     |
| Austrian Grand Prix      | RUS    | 3        | 0.078    | 1        | 1             | 0        | 0.922     |
| Italian Grand Prix       | LEC    | 4        | 0.079    | 1        | 1             | 0        | 0.921     |
| Belgian Grand Prix       | VER    | 1        | 0.827    | 0        | 5             | 0        | 0.827     |
| Azerbaijan Grand Prix    | PIA    | 2        | 0.238    | 1        | 1             | 0        | 0.762     |
| British Grand Prix       | HAM    | 2        | 0.276    | 1        | 1             | 0        | 0.724     |
| Canadian Grand Prix      | VER    | 2        | 0.288    | 1        | 1             | 0        | 0.712     |

### Podium

| gp                   | driver | grid_pos | pred_podium | real_podium | real_position | real_dnf | abs_error |
| -------------------- | ------ | -------- | ----------- | ----------- | ------------- | -------- | --------- |
| São Paulo Grand Prix | GAS    | 15       | 0.000       | 1           | 3             | 0        | 1.000     |
| Las Vegas Grand Prix | HAM    | 10       | 0.000       | 1           | 2             | 0        | 1.000     |
| Abu Dhabi Grand Prix | LEC    | 14       | 0.000       | 1           | 3             | 0        | 1.000     |
| São Paulo Grand Prix | VER    | 12       | 0.000       | 1           | 1             | 0        | 1.000     |
| Belgian Grand Prix   | RUS    | 7        | 0.002       | 1           | 1             | 0        | 0.999     |
| Dutch Grand Prix     | LEC    | 6        | 0.004       | 1           | 3             | 0        | 0.997     |
| Austrian Grand Prix  | PIA    | 7        | 0.014       | 1           | 2             | 0        | 0.986     |
| São Paulo Grand Prix | NOR    | 1        | 0.943       | 0           | 6             | 0        | 0.943     |
| Hungarian Grand Prix | HAM    | 5        | 0.060       | 1           | 3             | 0        | 0.940     |
| Belgian Grand Prix   | PIA    | 6        | 0.065       | 1           | 3             | 0        | 0.935     |

### Top 10

| gp                       | driver | grid_pos | pred_top10 | real_top10 | real_position | real_dnf | abs_error |
| ------------------------ | ------ | -------- | ---------- | ---------- | ------------- | -------- | --------- |
| Chinese Grand Prix       | HAM    | 18       | 0.000      | 1          | 9             | 0        | 1.000     |
| Hungarian Grand Prix     | RUS    | 17       | 0.001      | 1          | 8             | 0        | 1.000     |
| Canadian Grand Prix      | OCO    | 18       | 0.001      | 1          | 10            | 0        | 1.000     |
| Hungarian Grand Prix     | PER    | 16       | 0.001      | 1          | 7             | 0        | 0.999     |
| Abu Dhabi Grand Prix     | HAM    | 18       | 0.002      | 1          | 4             | 0        | 0.999     |
| United States Grand Prix | COL    | 17       | 0.004      | 1          | 10            | 0        | 0.996     |
| Azerbaijan Grand Prix    | NOR    | 16       | 0.004      | 1          | 5             | 0        | 0.996     |
| Australian Grand Prix    | HUL    | 16       | 0.005      | 1          | 10            | 0        | 0.995     |
| Mexico City Grand Prix   | PIA    | 17       | 0.005      | 1          | 8             | 0        | 0.995     |
| Las Vegas Grand Prix     | PER    | 16       | 0.009      | 1          | 10            | 0        | 0.992     |

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
