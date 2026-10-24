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
- **RAPM from play-by-play** ([stints.py](stints.py), [rapm.py](rapm.py)): ESPN's play-by-play is replayed to rebuild
  the ten players on the floor for every stretch of every game since 2010-11 (`data/nba/stints`, about 620,000 stints;
  rebuilt minutes match box scores within a minute). A ridge regression of each stint's margin per 100 possessions on
  who was on the floor gives every player's adjusted plus-minus (lambda 1000, two-year half-life). It's refit on the
  first of each month from earlier games only. Past seasons' fits are cached in `data/nba/rapm_monthly.csv.gz`.
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
[odds.py](odds.py) reads DraftKings' moneyline, spread and total from ESPN's public scoreboard. It's free with no key,
so [`.github/workflows/odds.yml`](.github/workflows/odds.yml) polls hourly through the day and every 30 minutes in the
run-up to tipoff. The margin is removed to give a market probability. Every snapshot goes to `log/odds.csv`, and the
last one before tipoff is the closing line that picks are graded against.

For past seasons, [tools/hist_odds.py](tools/hist_odds.py) collected the betting lines ESPN recorded for each game
(`data/nba/odds_hist.csv.gz`). The backtest compares the model against them.

Point spreads and totals come from the same scoreboard call. They're displayed next to the spread
implied by the model (normal margin, SD 12) but are never a model input.

## Live scores
During games the site polls ESPN's scoreboard from the browser ([site/live.js](site/live.js)): every 30 seconds while a
game is on, every 5 minutes otherwise, and not while the tab is hidden. Cards show the score and clock, and bars slide
to a live win probability: Brownian motion from the current lead and time left, centred on the pregame expected margin.
A running tally shows tonight's record.

## Browsing days
The Today page has a day strip covering this season and the week ahead. Each morning's export writes
`site/data/days/<date>.json` for every unplayed day in the window, and rewrites it until that day's games start, so a
past day's file is the card published before tipoff. Past days show it next to the final scores and closing lines from
the season record. Days before the site launched are rebuilt from the record's backfilled picks. Future days are an
early look.

## Late scratches
On the site, open a game and tap a player in either rotation to mark him out, or back in if he's listed as out. The
browser recomputes both lineups with the model's own formulas and re-prices the game instantly. This is saved in
your browser only. To change the published picks for everyone, run Actions → **Late scratch**, which records the
change in `data/overrides.csv`, re-prices the slate and redeploys.

## Beating the market
[tools/experiments.py](tools/experiments.py) tests new inputs walk-forward against the historical market line:
- **Tipoff Impact**, a custom BPM-style player metric: box-score rates regressed onto on-court plus-minus, fit on 2010-12 only
- **Travel and schedule density**
- **Massey ratings**
- **Market stacking**

RAPM from lineup data is the only one that helped clearly (log loss gap to the market 0.0073 → 0.0058), so it's in
the default model. None beat the closing line on log loss. A two-input stack (`logit p = a + b·logit(market) + c·logit(model)`)
ties it, and the site shows it as "Tipoff × Market". The new inputs are optional in the Model Lab, and the full
write-up is on the site's Methodology page.

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
