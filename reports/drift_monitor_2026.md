# Drift monitor - 2026

## Status by market

| market | n   | current_brier | baseline_brier | current_ece | baseline_ece | pred_mean | real_rate | status   |
| ------ | --- | ------------- | -------------- | ----------- | ------------ | --------- | --------- | -------- |
| win    | 198 | 0.020         | 0.035          | 0.016       | 0.011        | 0.045     | 0.045     | CRITICAL |
| podium | 198 | 0.072         | 0.081          | 0.064       | 0.060        | 0.136     | 0.136     | OK       |
| top6   | 198 | 0.113         | 0.077          | 0.098       | 0.034        | 0.273     | 0.273     | CRITICAL |
| top10  | 198 | 0.202         | 0.115          | 0.154       | 0.072        | 0.455     | 0.455     | CRITICAL |
| dnf    | 198 | 0.180         | 0.087          | 0.124       | 0.012        | 0.088     | 0.212     | CRITICAL |

## Warnings

- win: CRITICAL - current Brier 0.020 vs baseline 0.035; current ECE 0.016 vs baseline 0.011
- top6: CRITICAL - current Brier 0.113 vs baseline 0.077; current ECE 0.098 vs baseline 0.034
- top10: CRITICAL - current Brier 0.202 vs baseline 0.115; current ECE 0.154 vs baseline 0.072
- dnf: CRITICAL - current Brier 0.180 vs baseline 0.087; current ECE 0.124 vs baseline 0.012

## Recommended action

Use `major_2026` adaptation, increase current-season form weight, and avoid high-stake recommendations.

## Rule of thumb

If two or more markets are `CRITICAL`, treat the model as out-of-regime until current-season retraining/calibration is done.
