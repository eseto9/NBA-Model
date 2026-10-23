# Tipoff: NBA game predictions

A player-aware NBA model that publishes a win probability for every game, refreshed each
morning after the last game of the night. The site lives in [`site/`](site) and is deployed
with GitHub Pages.

## The model
Features are built only from information available before each game:

- **Elo rating**: FiveThirtyEight-style, adjusted for margin of victory, with 25% regression to the mean between seasons
- **Recent form**: exponentially weighted offensive, defensive and net rating, plus the four factors (shooting, turnovers, offensive rebounding, free throws)
- **Rest**: days since the last game and back-to-backs
- **Who's playing** ([players.py](players.py)): every player gets a minutes-weighted, recency-weighted rating
  on three metrics: PIE, on-court net rating and Game Score per 36, each pulled toward a newcomer's level until there's enough data.
  Lineup strength is the expected-minutes-weighted rating of the top 10 available players.
  - Past games use the players who actually played.
  - Upcoming games use today's roster minus the ESPN injury report. Players listed out are dropped, and day-to-day players count for half their minutes.

A regularized logistic regression combines them. It's trained on every game since 2011-12.

| Backtest, 2020-21 to 2025-26 | Accuracy | Log loss |
|---|---|---|
| **Team + players** | **67.1%** | **0.606** |
| Team stats only | 65.3% | 0.624 |
| Elo only | 64.2% | 0.638 |
| Always pick the home team | 55.4% | 0.689 |

## Usage
```
pip install -r requirements.txt
python nba.py fetch          # team + player game logs since 2010-11 (cached in data/)
python nba.py backtest       # walk-forward evaluation by season
python nba.py predict --days 3 --out "Player Name"   # terminal predictions with an injury override
python export.py             # build site/data/*.json for the next slate
python daily.py              # export, then commit and push (the daily job)
```
Preview the site locally with `python -m http.server -d site` and open http://localhost:8000.

## Daily updates
stats.nba.com blocks cloud servers, including GitHub Actions, so the data refresh runs on a
home PC. Windows Task Scheduler runs `daily.py` at 4:00 AM Eastern. It retries every 20 minutes
until last night's box scores are available, then commits `site/data` and `log/`. That push triggers
[`.github/workflows/pages.yml`](.github/workflows/pages.yml), which redeploys the site.

Picks are frozen in `log/predictions.csv` when they're first published, so the record on the site
only counts predictions made before tipoff.

*A statistical model, not betting advice.*
