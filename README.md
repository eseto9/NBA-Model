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
  on three metrics: PIE, on-court net rating (from plus-minus) and Game Score per 36, all computed from box scores,
  each pulled toward a newcomer's level until there's enough data.
  Lineup strength is the expected-minutes-weighted rating of the top 10 available players.
  - Past games use the players who actually played.
  - Upcoming games use today's roster minus the ESPN injury report. Players listed out are dropped, and day-to-day players count for half their minutes.

A regularized logistic regression combines them. It's trained on every game since 2011-12.

| Backtest, 2020-21 to 2025-26 | Accuracy | Log loss |
|---|---|---|
| **Team + players** | **67.5%** | **0.606** |
| Team stats only | 65.3% | 0.624 |
| Elo only | 64.2% | 0.638 |
| Always pick the home team | 55.4% | 0.689 |

## Data
- **Past seasons (2010-11 to 2025-26)** come from stats.nba.com and are stored in [`data/nba`](data/nba).
  They were fetched once with `python nba.py fetch-history`. stats.nba.com blocks cloud servers, so that
  command only works from a home connection.
- **The season in progress** comes from ESPN's public feed ([espn.py](espn.py)), one file per day in
  [`data/espn`](data/espn). That includes box scores, the schedule, rosters and the injury report. ESPN players
  are matched to NBA player ids by name so their ratings carry over (see `data/espn/ids.csv`).

## Usage
```
pip install -r requirements.txt
python nba.py update         # add any new box scores from ESPN
python nba.py backtest       # walk-forward evaluation by season
python nba.py predict --days 3 --out "Player Name"   # terminal predictions with an injury override
python export.py             # build site/data/*.json for the next slate
python daily.py --no-push    # what the daily job runs, without committing
```
Preview the site locally with `python -m http.server -d site` and open http://localhost:8000.

## Daily updates
[`.github/workflows/daily.yml`](.github/workflows/daily.yml) runs every morning at 09:15 UTC
(5:15 AM EDT / 4:15 AM EST), after the night's last game. It pulls the new box scores, retrains,
predicts the next slate, commits `site/data`, `data/espn` and `log/`, and redeploys the site. If a game
isn't final yet it retries every 20 minutes for up to two hours. You can also run it by hand from the
Actions tab ("Daily update" → Run workflow).

Picks are frozen in `log/predictions.csv` when they're first published, so the record on the site
only counts predictions made before tipoff.

*A statistical model, not betting advice.*
