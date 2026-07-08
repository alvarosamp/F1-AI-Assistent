# Drift monitor - 2024

## Status by market

| market | n   | current_brier | baseline_brier | current_ece | baseline_ece | pred_mean | real_rate | status |
| ------ | --- | ------------- | -------------- | ----------- | ------------ | --------- | --------- | ------ |
| win    | 478 | 0.035         | 0.035          | 0.011       | 0.011        | 0.050     | 0.050     | OK     |
| podium | 478 | 0.081         | 0.081          | 0.060       | 0.060        | 0.151     | 0.151     | OK     |
| top6   | 478 | 0.077         | 0.077          | 0.034       | 0.034        | 0.301     | 0.301     | OK     |
| top10  | 478 | 0.115         | 0.115          | 0.072       | 0.072        | 0.502     | 0.502     | OK     |
| dnf    | 478 | 0.087         | 0.087          | 0.012       | 0.012        | 0.086     | 0.098     | OK     |

## Warnings

- No warnings.

## Recommended action

Current drift is acceptable; keep monitoring race by race.

## Rule of thumb

If two or more markets are `CRITICAL`, treat the model as out-of-regime until current-season retraining/calibration is done.
