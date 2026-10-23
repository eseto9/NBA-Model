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
  each pulled toward a newcomer's starting level until there's enough data. That starting level, and a newcomer's expected
  minutes, depend on draft position (picks 1-3, 4-10, 11-20, 21-30, 31-45, 46-60, undrafted), learned from how past
  rookies in each range played.
  Lineup strength is the expected-minutes-weighted rating of the top 10 available players.
  - Past games use the players who actually played.
  - Upcoming games use today's roster minus the ESPN injury report. Players listed out are dropped, and day-to-day players count for half their minutes.

A regularized logistic regression combines them. It's trained on every game since 2011-12. The site's Record page
shows a walk-forward backtest of every season since 2012-13 (each predicted by a model trained only on earlier
seasons), next to the betting market on the same games.

| Model variants, 2020-21 to 2025-26 | Accuracy | Log loss |
|---|---|---|
| **Team + players** | **67.5%** | **0.605** |
| Team stats only | 65.3% | 0.624 |
| Elo only | 64.2% | 0.638 |
| Always pick the home team | 55.4% | 0.689 |

## Betting odds
[odds.py](odds.py) pulls moneylines from [The Odds API](https://the-odds-api.com) (US books, moneyline market, 1 credit
per call). The free plan has 500 credits a month, so [`.github/workflows/odds.yml`](.github/workflows/odds.yml)
runs hourly from 14:05 to 03:05 UTC. With the morning run that's 15 calls a day, about 465 in a 31-day month.
`odds.py` also skips a call when no game starts in the next 24 hours, or when the remaining credits wouldn't cover the
rest of the month. Each book's margin is removed and the books are averaged into a market probability. The last
snapshot before tipoff is the closing line that live picks are compared against (`log/odds.csv`).

The API key is read from the `ODDS_API_KEY` repository secret (Settings → Secrets and variables → Actions).

For past seasons, [tools/hist_odds.py](tools/hist_odds.py) collected the betting lines ESPN recorded for each game
(`data/nba/odds_hist.csv.gz`). The backtest compares the model against them.

Point spreads and totals come free from ESPN's scoreboard on every odds run. They're displayed next to the spread
implied by the model (normal margin, SD 12) but are never a model input.

## Late scratches
Actions → **Late scratch** → Run workflow, then enter the players who won't play (and optionally any listed as out
who will). It records them in `data/overrides.csv`, re-prices today's slate and redeploys the site. Overrides last
through the next day. Picks can be revised until tipoff, then they're frozen.

## Model Lab
The site's Model Lab (`site/lab.js`) refits the model in the browser, in a Web Worker. You can choose L2 logistic
regression (Newton's method, same objective as the default, so it reproduces its backtest) or histogram gradient
boosting, adjust the hyperparameters, and toggle input groups, including pace. LEBRON, VORP and Win Shares are listed
but disabled, with the reason for each. It replays the full walk-forward backtest against the default model and the
market, and can replace the Today page's probabilities with the custom model's. Its data comes from
`site/data/lab_hist.json` (written once per season) and `lab_live.json` (daily).

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
