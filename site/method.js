/* Tipoff: the methodology page. Formulas are TeX, rendered by KaTeX after the view mounts. */
"use strict";

const FEATURE_LABELS = {
  elo_diff: "Elo difference (home − away + home court)",
  d_net: "Net rating, recent form", d_ortg: "Offensive rating, form", d_drtg: "Defensive rating, form",
  d_efg: "Effective FG%, form", d_opp_efg: "Opponent eFG% allowed, form", d_tov: "Turnover rate, form",
  d_orb: "Offensive rebound rate, form", d_ftr: "Free-throw rate, form", d_win: "Win rate, form",
  rest_h: "Home rest days", rest_a: "Away rest days", b2b_h: "Home back-to-back", b2b_a: "Away back-to-back",
  gp_min: "Games played (fewer of the two)", playoffs: "Playoff game", neutral: "Neutral site",
  d_lu_pie: "Lineup PIE", d_lu_net: "Lineup on-court net rating", d_lu_gmsc: "Lineup Game Score/36",
  d_lu_star: "Best available player", d_lu_min: "Rotation minutes available", d_lu_rapm: "Lineup RAPM (play-by-play)", d_lu_tpi: "Lineup Tipoff Impact",
};

function coefChart() {
  const f = D.meta.model.features.slice().sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
  const W = 760, rowH = 24, L = 280, R = 60, top = 24, H = top + f.length * rowH + 10;
  const max = Math.max(...f.map((d) => Math.abs(d.coef)));
  const x0 = L + (W - L - R) / 2, sc = (W - L - R) / 2 / max;
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Standardized logistic regression coefficients">`;
  s += `<line x1="${x0}" x2="${x0}" y1="${top - 8}" y2="${H - 4}" stroke="var(--line)"/>`;
  s += `<text x="${x0 - 6}" y="12" text-anchor="end">← favors away</text><text x="${x0 + 6}" y="12">favors home →</text>`;
  f.forEach((d, i) => {
    const y = top + i * rowH, w = Math.abs(d.coef) * sc, pos = d.coef >= 0;
    s += `<text x="${L - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" class="lbl-ink" style="font-family:var(--font);font-size:12px">${esc(FEATURE_LABELS[d.name] || d.name)}</text>`;
    s += `<rect class="grow" style="animation-delay:${0.03 * i}s;transform-origin:${pos ? "left" : "right"} center" x="${pos ? x0 : x0 - w}" y="${y + 5}" width="${Math.max(w, 1.5)}" height="${rowH - 10}" rx="3"
      fill="${pos ? "var(--orange)" : "var(--slate)"}" data-tip="<b>${esc(d.name)}</b><br>β = ${d.coef} per SD<br>feature mean ${d.mean}, SD ${d.sd}"/>`;
    s += `<text x="${pos ? x0 + w + 6 : x0 - w - 6}" y="${y + rowH / 2 + 4}" text-anchor="${pos ? "start" : "end"}">${d.coef > 0 ? "+" : ""}${d.coef.toFixed(3)}</text>`;
  });
  return s + "</svg>";
}

function priorTable() {
  const order = ["1-3", "4-10", "11-20", "21-30", "31-45", "46-60", "undrafted"];
  const p = D.meta.priors || {};
  return `<div class="card table-wrap"><table><thead><tr><th>Draft range</th><th class="r">Rookies</th><th class="r">Exp. minutes</th><th class="r">Game Score/36</th><th class="r">PIE</th><th class="r">On-court net</th></tr></thead><tbody>
    ${order.filter((b) => p[b]).map((b) => `<tr><td class="mono">${b}</td><td class="r mono">${p[b].n}</td><td class="r mono">${p[b].min.toFixed(1)}</td>
      <td class="r mono">${p[b].gmsc.toFixed(2)}</td><td class="r mono">${pct(p[b].pie, 1)}</td><td class="r mono">${p[b].net.toFixed(1)}</td></tr>`).join("")}</tbody></table></div>`;
}

function viewMethod() {
  const o = D.backtest.overall, vm = D.backtest.vs_market, m = D.meta.model;
  return String.raw`<section class="hero" style="margin-bottom:8px"><div><span class="pill"><span class="dot"></span>Methodology</span>
    <h1 style="font-size:clamp(36px,5vw,56px)">How Tipoff<br><em>prices a game.</em></h1>
    <p class="lede">A regularized logistic regression on team strength, recent form, rest and a credibility-weighted rating of the players actually available. Every number on this page comes from the live model.</p></div>
    <nav class="toc card">${[["data", "Data"], ["team", "Team features"], ["players", "Player ratings"], ["lineup", "Lineups & availability"], ["model", "The model"], ["valid", "Validation"], ["market", "Betting market"], ["beat", "Can it beat the market?"], ["metrics", "Why these metrics"], ["limits", "Limitations"]]
      .map(([id, t], i) => `<a href="#/how" data-jump="${id}"><span class="mono muted">${String(i + 1).padStart(2, "0")}</span> ${t}</a>`).join("")}</nav></section>

  <article class="prose method">
  <h2 id="data">1 · Data</h2>
  <p><b>Seasons 2010-11 to 2025-26</b> come from stats.nba.com (team and player box scores; 19,000+ games, ~450,000 player-games) and are stored in the repository. <b>The current season</b> comes from ESPN's public feed, appended daily. That covers box scores, schedule, rosters and the injury report. stats.nba.com blocks cloud servers, so this split lets the whole pipeline run on GitHub Actions. ESPN athletes are mapped to NBA player ids by normalized name, restricted to players active in the last four years. Rookies are mapped through the draft table, so a player's rating history is continuous across sources. All advanced metrics are recomputed from the base box score, so both sources are rated identically.</p>

  <h2 id="team">2 · Team features</h2>
  <h3>Elo</h3>
  <p>FiveThirtyEight-style Elo with a margin-of-victory multiplier. With home-court advantage \(H = 100\) (0 at neutral sites):</p>
  $$E_h = \frac{1}{1 + 10^{-(R_h + H - R_a)/400}}, \qquad R_h' = R_h + K\,M\,(S_h - E_h),\quad K = 20$$
  $$M = \frac{(|\text{MOV}| + 3)^{0.8}}{7.5 + 0.006\,D_w}$$
  <p>Here \(D_w\) is the winner's pre-game Elo edge, including home court. That damps the autocorrelation that pure margin-of-victory Elo suffers, where favorites win by more and inflate their own ratings. Between seasons every team regresses a quarter of the way to the mean: \(R \leftarrow 0.75R + 0.25\cdot1505\). The model uses \(R_h + H - R_a\) as a feature, not \(E_h\), and learns the scale itself.</p>
  <h3>Possessions, efficiency and the four factors</h3>
  $$\text{Poss} = \tfrac12\Big[(\text{FGA} - \text{ORB} + \text{TOV} + 0.44\,\text{FTA})_{\text{tm}} + (\cdot)_{\text{opp}}\Big], \qquad \text{ORtg} = 100\,\frac{\text{PTS}}{\text{Poss}}$$
  <p>Net rating is ORtg − DRtg. The four factors are effective FG% \(\frac{\text{FGM} + 0.5\,\text{3PM}}{\text{FGA}}\), turnover rate \(\frac{\text{TOV}}{\text{Poss}}\), offensive rebound rate \(\frac{\text{ORB}}{\text{ORB} + \text{DRB}_{\text{opp}}}\) and free-throw rate \(\frac{\text{FTM}}{\text{FGA}}\), plus the opponent's eFG% (shooting allowed).</p>
  <h3>Recency weighting</h3>
  <p>Each team statistic is an exponentially weighted mean over the team's previous games, carried across seasons, with a half-life of \(h = 10\) games:</p>
  $$f_t = \frac{\sum_{i<t} \lambda^{\,t-1-i} x_i}{\sum_{i<t} \lambda^{\,t-1-i}}, \qquad \lambda = 2^{-1/h}$$
  <p>Every feature enters as a home-minus-away difference, so the model is antisymmetric apart from the intercept, which absorbs home-court advantage. Rest is days since the previous game, capped at 5, plus back-to-back indicators for each side. The smaller of the two teams' games played this season lets the model discount early-season form.</p>

  <h2 id="players">3 · Player ratings</h2>
  <p>Three per-game metrics, all computable from a basic box score:</p>
  $$\begin{aligned}\text{GmSc}_{36} = \frac{36}{\text{MIN}}\big(&\text{PTS} + 0.4\,\text{FGM} - 0.7\,\text{FGA} - 0.4(\text{FTA}-\text{FTM}) + 0.7\,\text{ORB} \ &+ 0.3\,\text{DRB} + \text{STL} + 0.7\,\text{AST} + 0.7\,\text{BLK} - 0.4\,\text{PF} - \text{TOV}\big)\end{aligned}$$
  $$\text{PIE}_i = \frac{e_i}{\sum_{j \in \text{game}} e_j}$$
  $$\begin{aligned}e = {}&\text{PTS} + \text{FGM} + \text{FTM} - \text{FGA} - \text{FTA} + \text{DRB} \ &+ \tfrac12\text{ORB} + \text{AST} + \text{STL} + \tfrac12\text{BLK} - \text{PF} - \text{TOV}\end{aligned}$$
  $$\text{Net}^{\text{on}}_i \approx \frac{100 \cdot (\pm)_i}{\text{Poss}_{\text{game}} \cdot \text{MIN}_i / 48}$$
  <p>The last one estimates on-court net rating from plus-minus. Its minutes-weighted correlation with NBA.com's tracked value is 0.99.</p>
  <h3>Credibility-weighted estimate</h3>
  <p>A player's rating on each metric is a minutes-weighted, exponentially decayed mean, shrunk toward a prior. It is a credibility estimator in the Bühlmann sense:</p>
  $$\hat r_i = Z_i\,\bar x_i + (1 - Z_i)\,\mu_{b(i)}, \qquad Z_i = \frac{M_i}{M_i + K}, \qquad M_i = \sum_k \lambda_p^{\,n-k}\,\text{MIN}_k, \quad \bar x_i = \frac{\sum_k \lambda_p^{\,n-k}\,\text{MIN}_k\,x_k}{M_i}$$
  <p>The half-life is 25 player-games (\(\lambda_p = 2^{-1/25}\)) and \(K = 250\) minutes. So a player with 250 discounted minutes, about seven full games, is half credible. \(K\) plays the role of Bühlmann's \(k = \sigma^2_{\text{process}}/\sigma^2_{\text{hyp}}\). It was set by validation rather than estimated from variance components, which would be a natural refinement. Ratings are computed strictly before each game.</p>
  <h3>Draft-range priors</h3>
  <p>The complement of credibility, \(\mu_b\), depends on the player's draft range. It is the minutes-weighted mean over the first 20 career games of players whose careers began inside the data. Expected minutes for a player with no NBA games start from the same group's average. Current values:</p>
  ${priorTable()}
  <p style="margin-top:10px">Minutes and PIE fall steadily with draft position. Game Score per 36 is flatter, because high picks are often given high-usage, low-efficiency roles early. Draft position mostly tells the model how much a rookie will play. Expected minutes for veterans are an exponentially weighted mean with a 6-game half-life.</p>

  <h2 id="lineup">4 · Lineups & availability</h2>
  <p>For each game, every player on the team gets availability \(a_j\): 0 if listed out, ½ if day-to-day, 1 otherwise. On the site you can tap any player in a game's rotation to mark him out or back in. The browser recomputes both lineups with these formulas and adds the change in the model's index to the published log-odds, so the price updates instantly. The lineup is the top 10 players by \(w_j = \hat m_j\,a_j\), where \(\hat m_j\) is expected minutes, and the lineup features are:</p>
  $$\text{L}^{(x)} = \frac{\sum_{j \in \text{top10}} w_j\,\hat r^{(x)}_j}{\sum_{j \in \text{top10}} w_j}, \qquad \text{Star} = \max_{j:\,w_j \ge 20} \hat r^{\text{GmSc}}_j, \qquad \text{Depth} = \sum_{j \in \text{top10}} w_j$$
  <p>Normalizing by \(\sum w_j\) redistributes a missing player's minutes to the rest of the rotation at their own ratings, so losing a star costs exactly the gap between him and his replacements. In training, the players who actually appeared define the lineup, with \(a = 1\). Restricting to the top 10 by expected minutes keeps garbage-time appearances from leaking the result into the features.</p>

  <h2 id="model">5 · The model</h2>
  <p>Let \(z\) be the vector of 22 standardized features. The model is an L2-penalized logistic regression:</p>
  $$\Pr(\text{home win}) = \sigma\!\Big(\beta_0 + \sum_k \beta_k z_k\Big), \qquad \hat\beta = \arg\min_\beta\; -\sum_g \ell_g(\beta) + \frac{1}{2C}\lVert\beta_{1:}\rVert^2, \quad C = ${m.C}$$
  <p>It is refit every morning on all completed games since 2011-12 (\(N\) = ${D.meta.trained_games.toLocaleString()}). The intercept is \(\beta_0 = ${m.intercept}\), about \(\sigma(\beta_0) \approx\) ${pct(1 / (1 + Math.exp(-m.intercept)), 1)} for two average teams at the home team's arena. Standardized coefficients, as the log-odds change per one-SD change in each feature:</p>
  ${coefChart()}
  <p>Several features are near-collinear by construction. Net rating is exactly offensive minus defensive rating, and the three lineup metrics are correlated. The ridge penalty spreads weight across them, so individual coefficients aren't separately identifiable and shouldn't be read as causal effects. The fitted probability is what matters, and it's stable.</p>
  <h3>Why logistic, and not gradient boosting?</h3>
  <ul>
    <li><b>Structure.</b> Paired comparisons are naturally logistic in a strength difference: Bradley–Terry, and Elo itself. Almost all the signal here is an additive difference in team quality, and there's little interaction for trees to find.</li>
    <li><b>Evidence.</b> A histogram gradient-boosting classifier (300 trees, learning rate 0.03, depth 3, min leaf 100, L2 = 1) on the same features scored log loss 0.608 against 0.605 for the logistic over 2020-26. A 50/50 blend didn't improve on the logistic alone.</li>
    <li><b>Calibration and attribution.</b> The logistic's probabilities are calibrated out of the box (see the Record page). Its log-odds decompose exactly into feature contributions \(\beta_k z_k\). That's what drives the factor chart on each game, which applies grouped contributions in sequence: home court, Elo, form, players, rest. The order of the steps changes how much each one moves the probability, but not the final number.</li>
    <li><b>Variance.</b> About 19,000 rows of very noisy outcomes (a game is close to a coin flip with a tilt) favor a low-variance model.</li>
  </ul>

  <h3>Try other models: the Model Lab</h3>
  <p>The <a href="#/lab">Model Lab</a> refits the model in your browser. You can switch to gradient boosting or plain Elo, change the L2 penalty, and turn input groups on or off, including pace, which the default leaves out. It reruns the same walk-forward backtest against the default model and the market. Its logistic fit uses the same objective as the default, solved by Newton's method, and reproduces the default's backtest log loss to four decimals. You can also tell the Today page to use your model instead.</p>

  <h2 id="valid">6 · Validation</h2>
  <p>The backtest uses an expanding window, season by season. Season \(s\) is predicted by a model fit only on seasons before \(s\), using features that are strictly pre-game. From ${esc(D.backtest.first)} to ${esc(D.backtest.through)} that's ${o.n.toLocaleString()} games at ${pct(o.accuracy, 1)} accuracy and log loss ${o.log_loss}:</p>
  $$\text{LogLoss} = -\frac1N\sum_g \big[y_g \ln p_g + (1 - y_g)\ln(1 - p_g)\big]$$
  <p>For reference, a constant home-win rate scores about 0.689 and Elo alone about 0.638. Calibration is checked by binning predicted favorites at 5-point intervals; the curves are on the Record page.</p>
  <p><b>Known optimistic biases:</b></p>
  <ol>
    <li>In the backtest, the lineup is the players who actually played, while live picks rely on the injury report. That's a look-ahead on availability.</li>
    <li>The half-lives, \(K\), \(C\) and lineup size were chosen with the 2020-26 results in view, a mild selection effect.</li>
    <li>The draft-range priors are fit on the full history.</li>
  </ol>
  <p>The live record, graded against closing lines, is the clean out-of-sample test.</p>

  <h2 id="market">7 · Betting market</h2>
  <p>Live moneylines are DraftKings' lines as published on ESPN's scoreboard, free and keyless. American odds convert to decimal \(d = 1 + \text{ML}/100\) for \(\text{ML} > 0\) and \(d = 1 + 100/|\text{ML}|\) otherwise, with implied probability \(q = 1/d\). The book's margin, or overround \(q_h + q_a - 1\), is removed multiplicatively:</p>
  $$p^{\text{mkt}}_h = \frac{q_h}{q_h + q_a}$$
  <p>Snapshots are taken hourly through the day and every 30 minutes in the run-up to tipoff. The last one before tipoff is the <b>closing line</b>, the standard efficient benchmark that live picks are graded against. For past seasons, the backtest uses the moneylines ESPN recorded for each game, de-vigged the same way. From 2012-13 these are listed under TeamRankings and numberFire, later under ESPN's consensus and DraftKings. Where they overlap, the providers' de-vigged probabilities correlate at 0.996 or higher with the consensus and differ by about 1 point on average, so they're market prices, not those sites' own projections. 2010-12 has no lines. Over the ${vm.n ? vm.n.toLocaleString() : 0} backtest games with a line, the market picked ${vm.n ? pct(vm.market.accuracy, 1) : "—"} of winners against the model's ${vm.n ? pct(vm.model.accuracy, 1) : "—"}, with log loss ${vm.n ? vm.market.log_loss : "—"} against ${vm.n ? vm.model.log_loss : "—"}.</p>
  <p>The <b>best value</b> card ranks every side of every game by expected return per $1 at the posted price, using the model's probability:</p>
  $$\text{EV} = p^{\text{model}} \cdot d - 1$$
  <p>A positive EV means the model rates a side above the price on offer. Since the closing line is usually better calibrated than any public model, read it as a disagreement worth checking, not a bet.</p>
  <p><b>Point spreads</b> come from ESPN's scoreboard (DraftKings) and are shown for reference only. They are never a model input. For comparison, the model's probability is converted to an implied spread, assuming the final margin is normal with standard deviation \(\sigma = 12\) points:</p>
  $$\text{margin}_h = \sigma\,\Phi^{-1}\!\big(p^{\text{model}}_h\big), \qquad \text{spread}_h = -\,\text{margin}_h$$

  <h2 id="beat">8 · Can it beat the market?</h2>
  <p>Short answer: not on its own. Lineup data (RAPM) closed about a sixth of the gap, and a blend with the market ties it. Every candidate went through the same walk-forward backtest and was scored on the 17,577 games from 2012-13 onward that have a market line. The last column is the model's log loss minus the market's, with a paired standard error (positive means the market is better).</p>
  <div class="card table-wrap"><table><thead><tr><th>Variant</th><th class="r">Accuracy</th><th class="r">Market</th><th class="r">Δ log loss vs market</th></tr></thead><tbody>
    <tr><td>Box-score model (before lineup data)</td><td class="r mono">67.5%</td><td class="r mono">68.2%</td><td class="r mono">+0.0073 ± 0.0010</td></tr>
    <tr><td>+ Tipoff Impact (custom player metric)</td><td class="r mono">67.7%</td><td class="r mono">68.2%</td><td class="r mono">+0.0069 ± 0.0011</td></tr>
    <tr><td>+ travel, time zones, 3-in-4 nights, altitude</td><td class="r mono">67.7%</td><td class="r mono">68.2%</td><td class="r mono">+0.0074 ± 0.0011</td></tr>
    <tr><td>+ Massey (opponent-adjusted) ratings</td><td class="r mono">67.4%</td><td class="r mono">68.2%</td><td class="r mono">+0.0071 ± 0.0010</td></tr>
    <tr><td><b>+ RAPM from play-by-play lineups</b> (now the default)</td><td class="r mono"><b>67.8%</b></td><td class="r mono">68.2%</td><td class="r mono"><b>+0.0058 ± 0.0011</b></td></tr>
    <tr><td>Market line + all model features (16,346 games)</td><td class="r mono">68.1%</td><td class="r mono">68.2%</td><td class="r mono">+0.0008 ± 0.0005</td></tr>
    <tr><td><b>Tipoff × Market stack</b> (16,346 games)</td><td class="r mono"><b>${D.backtest.blend ? pct(D.backtest.blend.vs_market.model.accuracy, 1) : "—"}</b></td><td class="r mono">${D.backtest.blend ? pct(D.backtest.blend.vs_market.market.accuracy, 1) : "—"}</td><td class="r mono">+0.0000 ± 0.0002</td></tr>
  </tbody></table></div>
  <h3>RAPM from play-by-play</h3>
  <p>ESPN's play-by-play logs every substitution, so the five players on the floor for each team are rebuilt for every stretch of every game since 2010-11 (about 620,000 <i>stints</i>; reconstructed minutes match the box score to within a minute). Each stint \(s\) gives one observation: \(x_{s,j} = +1\) for the home five, \(-1\) for the away five, response \(y_s\) = home margin per 100 possessions, weight \(w_s\) = possessions × recency decay (half-life two years, three-season window):</p>
  $$\hat\beta = \arg\min_\beta \sum_s w_s\big(y_s - h - x_s^\top \beta\big)^2 + \lambda\lVert\beta\rVert^2, \qquad \lambda = 1000$$
  <p>\(h\) is home-court advantage. Ratings are refit on the first of every month from stints before that date, so every game in the backtest is priced with ratings that never saw it. Players without lineup data count at −2 (replacement level). The lineup feature is the expected-minutes-weighted RAPM of the available rotation, aggregated exactly like the box metrics. \(\lambda\) and the half-life were chosen on a small grid with <code>tools/tune_rapm.py</code>; all settings were within 0.0007 log loss of each other. RAPM now has the largest standardized coefficient of any player input.</p>
  <h3>Tipoff Impact (TPI)</h3>
  <p>A custom, BPM-style player metric. Each player-game's box score is converted to rates per 100 on-court possessions \(z_k\), and the estimated on-court net rating is regressed on them by minutes-weighted ridge regression:</p>
  $$\text{TPI} = \alpha + \sum_k \gamma_k z_k$$
  <p>It was fit only on 2010-11 and 2011-12, so no backtest season informs its coefficients. Fitted per-100 weights: points +1.11, FGM −0.52, FGA −0.83, FTA −0.61, defensive rebounds +0.80, assists +0.93, steals +1.00, blocks +0.75, turnovers −1.03, fouls −0.20, offensive rebounds ≈ 0 (\(\alpha = -9.3\)). TPI then goes through the same credibility weighting and lineup aggregation as the other metrics. It's sensible, and slightly better than leaving it out, but the gain (0.0004) is inside the noise. It's available as an input in the <a href="#/lab">Model Lab</a>, along with Massey ratings and the travel and schedule inputs.</p>
  <h3>Massey ratings</h3>
  <p>Before each game day, a recency-weighted ridge regression of every prior margin (capped at 30, 45-day half-life, 400-day window) on team indicators plus home court:</p>
  $$\min_{r, h}\; \sum_g w_g \big(\text{MOV}_g - (r_{\text{home}(g)} - r_{\text{away}(g)} + h)\big)^2 + \lambda \lVert r \rVert^2$$
  <h3>The stack</h3>
  <p>The only construction that matched the market is a two-input logistic stack, fit walk-forward on earlier seasons' games that had a line:</p>
  $$\operatorname{logit} p = a + b\,\operatorname{logit} p^{\text{mkt}} + c\,\operatorname{logit} p^{\text{model}}$$
  <p>Fit on all games with a line, the weights are \(b = ${D.backtest.blend ? D.backtest.blend.coef.market : "—"}\) and \(c = ${D.backtest.blend ? D.backtest.blend.coef.model : "—"}\). The market carries most of the information. The model's weight rose from about 0.12 to 0.21 once lineup data was added, so RAPM carries information the market doesn't fully price, but not enough to beat it on its own. It's shown on every game as "Tipoff × Market".</p>
  <p><b>Why the market is hard to beat.</b> The closing line aggregates injury news up to tipoff, lineup and rest decisions, tracking-based player value and the money of bettors who have their own models. This model's information is a subset of that: public box scores and the injury report. Raw accuracy also has little statistical power here. On 15,000 games, the standard error of an accuracy figure is about 0.4 points, so a gap under about 1 point between two models is mostly noise. Log loss is the sharper test, and it consistently favors the market. Lineup data was the one addition that moved the needle. What's left is mostly information the market sees and this model doesn't: late availability, minutes restrictions and tracking data.</p>

  <h2 id="metrics">9 · Why these player metrics</h2>
  <p>The player layer needs a rating that (a) updates daily from data free in the cloud, (b) works at the single-game level so it can be decayed and credibility-weighted, (c) is a rate, not a volume measure, since minutes are handled separately, and (d) captures both box-score production and non-box impact. Here's how the common alternatives fare:</p>
  <div class="card table-wrap"><table><thead><tr><th>Metric</th><th>What it is</th><th>Why not (or not yet)</th></tr></thead><tbody>
    <tr><td><b>Win Shares</b></td><td>Credits team wins to players via marginal offense and defense</td><td>A cumulative volume measure. Per-48 helps, but the defensive half is mostly team defense split by minutes and rebounds, so it double-counts what Elo and team form already carry. Needs Basketball-Reference's season-level pipeline.</td></tr>
    <tr><td><b>VORP</b></td><td>BPM above replacement (−2.0) × share of team minutes</td><td>Volume by construction. Minutes are already modeled separately as expected minutes and availability, so VORP would count playing time twice. The rate inside it, BPM, is the interesting part (see below).</td></tr>
    <tr><td><b>BPM</b></td><td>Box-score regression onto RAPM, with team adjustment</td><td>The strongest free box metric, and computable from box scores with published coefficients plus position and role estimates. It's the natural next upgrade, and it would replace Game Score.</td></tr>
    <tr><td><b>LEBRON / EPM / RAPTOR</b></td><td>Ridge-regressed plus-minus with box and tracking priors</td><td>Proprietary or discontinued. Their core, RAPM from lineup stints, is now built here from ESPN play-by-play (section 8), without the tracking-data priors.</td></tr>
    <tr><td><b>Elo</b></td><td>A team rating from results</td><td>Already in the model at team level. There's no standard player Elo, because players don't win or lose head-to-head.</td></tr>
    <tr><td><b>Pace</b></td><td>Possessions per 48 minutes</td><td>Measures tempo, not quality. It's already used to normalize every rating to per-100 possessions. Pace changes the variance of the margin more than who wins.</td></tr>
  </tbody></table></div>
  <p style="margin-top:10px">The three chosen metrics complement each other. Game Score/36 measures scoring and box production per minute. PIE measures share of all box events, which adjusts for game context and tempo. On-court net rating carries the non-box signal (defense, spacing, screening) that box metrics miss, though it's noisy per game. That's the reason for credibility weighting and for letting the regression set its weight. Together they took the backtest log loss from 0.624 (team features only) to 0.605. Candidate metrics are judged on the same walk-forward log loss, so swapping in BPM is a straightforward experiment.</p>

  <h2 id="limits">10 · Limitations</h2>
  <ul>
    <li>Minutes restrictions and in-game injuries are invisible. Day-to-day is a blunt half-weight.</li>
    <li>Coaching, scheme and matchup interactions are absent. There are no player-vs-player effects.</li>
    <li>Players new to the NBA from overseas start from their draft range's average (often "undrafted") regardless of their history abroad.</li>
    <li>There's no travel, altitude, or rest-advantage asymmetry beyond rest days.</li>
    <li>The rating hyperparameters were tuned by hand on validation loss, not chosen by cross-validation.</li>
  </ul>
  <p class="muted">A statistical model, not betting advice. Source: <a href="https://github.com/eseto9/NBA-Model">github.com/eseto9/NBA-Model</a>.</p>
  </article>`;
}

function renderMath(root) {
  if (window.renderMathInElement) {
    renderMathInElement(root, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
  } else {
    setTimeout(() => renderMath(root), 150);   // KaTeX is loaded with defer
  }
}
