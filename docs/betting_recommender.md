# Betting recommender

This project now has a conservative recommendation layer on top of the race
simulator. It is a decision-support tool: it points out possible value spots,
but the final decision stays with you.

## Run

```powershell
python src\predict_race_week.py `
  --gp "Australian Grand Prix" `
  --grid examples\race_week_grid.csv `
  --odds examples\race_week_odds.csv `
  --sims 2000 `
  --quiet
```

## Grid input

`examples/race_week_grid.csv`:

```csv
driver,team,grid_pos,quali_pos,gap_to_pole_ms,avg_residual_recent
NOR,McLaren,1,1,0,-0.05
VER,Red Bull Racing,2,2,95,-0.02
```

## Odds input

`examples/race_week_odds.csv`:

```csv
driver,market,odds,sportsbook
NOR,win,3.20,manual
VER,podium,1.70,manual
RUS,top6,1.75,manual
```

Supported markets:

- `win`
- `podium`
- `top6`
- `top10`
- `dnf`

Odds must be decimal odds.

## Recommendation logic

For each driver/market, the recommender calculates:

- model probability from the simulator
- implied probability from the odds
- conservative probability after calibration haircut
- expected value
- conservative expected value
- market edge
- market quality from `models/calibration_report.json`

The system only emits `RECOMMEND` when conservative EV and edge pass the
configured thresholds and the market calibration is acceptable.

Default thresholds:

```text
min_edge = 3 percentage points
min_ev   = 5%
```

## Decisions

- `RECOMMEND`: passed conservative filters.
- `WATCH`: raw EV is positive, but calibration haircut removed the edge.
- `PASS`: market is weak, or EV/edge is below threshold.

## Important

This is not a guarantee. F1 has DNFs, safety cars, weather, penalties,
strategy surprises, and model error. Treat recommendations as ranked signals,
not certainty.
