# Weekly forward tester pre-activation review — 2026-09-04 ET

**Follow-up:** the [full history-quality review](weekly_forward_test_history_review.md)
reproduced these defects and found additional internally valid source
contradictions. Activation remains disabled; no replacement prices were accepted.

**Release blocked; `FORWARD_TEST_ENABLED=false`.** The implementation passes its local verification gates, but the intended Tradier account returns invalid required SPX and SPY history. The reviewed implementation is being preserved in a scoped review branch. Merge, production migrations, registration, deployment and activation remain conditional on repairing those prerequisites. No production forecasts, observations or synthetic demonstration records were written.

## Cumulative identity and regression evidence

`model_methodology_id` excludes weekly training rows, coefficients, covariance and fit dates. It incorporates the methodology source/configuration/dependency hash, model name and actual selected feature columns. `fit_id` and immutable input snapshots retain each refit's provenance. Model specification, GEX policy and recommendation source changes affect the methodology hash; feature availability changes affect the actual-design hash. The scoreboard groups by study/cohort/model version/ticker/model/tier, never by individual fit.

The new three-week isolated test fits and scores SPY/M2/Point with production code. Weeks 2026-09-07 and 2026-09-14 have different completed training outcomes, coefficients, covariance, training cutoffs and source/input hashes. They produce **one cumulative group with `close_n=2` and `path_n=2`**. The third week removes the actual VIX feature from the fitted design matrix, without manufacturing a different version label, and produces **a separate group with `close_n=1` and `path_n=1`**.

| Fixture week | Actual features | VIX coefficient | Fit hash prefix | Model-version prefix |
|---|---|---:|---|---|
| 2026-09-07 | HAR core + VIX | 0.015757851763463823 | 9134d8661ad6 | a2523f2692e7 |
| 2026-09-14 | HAR core + VIX | 0.014680974399255614 | 12796408f2f6 | a2523f2692e7 |
| 2026-09-21 | HAR core | omitted | fad856381b3a | 2db48f39e810 |

These are synthetic isolated test results, not prospective performance. Full hashes, coefficients, provenance and comparison rows are in the local `forward-test-two-week-evidence.json` artifact.

The exact local project command `python -m pytest -q --junitxml=<local-artifact>` ran through `.venv\Scripts\python.exe` with the explicitly isolated `FORWARD_TEST_TEST_DATABASE_URL`: **553 passed in 60.29 seconds**, no failures or skips. PostgreSQL 17.11 listened only on 127.0.0.1:55441 with a disposable `gammalens_forward_test` database. Tests used separate schemas and verified migrations, triggers, concurrent single-winner capture, atomic rollback, and deadline rejection both before and after database writes. The previous stopped temporary PostgreSQL folder was left alone.

The empty registered-study view now shows its study/start week, usable filters, zero eligible samples and a headers-only Excel download. Streamlit AppTest passed the actual Forward Test route and populated/empty export checks. Cloud route and Excel response certification remain pending deployment.

## Read-only live source checks

Checked beginning 2026-09-05T01:45:50Z (Friday evening ET). Explicit app credentials were loaded from the application's local secrets file, rather than an unrelated inherited database variable. The Tradier production Brokerage API authenticated successfully; account identifiers and credentials were not recorded. Market state was **closed**. Returned quote timestamps were preserved with their observed ages; no opening-session freshness or actual real-time latency was claimed.

| Capability | SPX | SPY | AAPL | AMD |
|---|---|---|---|---|
| Quote identity/open/last | available | available | available | available |
| Weekly history | 317 valid bars | 317 valid bars | 317 valid bars | 317 valid bars |
| Daily history | 1 invalid row | 8 invalid rows | 1,526 valid bars | 1,526 valid bars |
| Listed 2026-09-11 expiry | yes | yes | yes | yes |
| Verified call/put contracts | 273 / 273 SPXW | 195 / 195 SPY | 79 / 79 AAPL | 209 / 209 AMD |
| Sep 4 minute history, 09:45–16:00 | 375 bars | 375 bars | 375 bars | 375 bars |

The history spans 2020-08-10 through 2026-09-04 (weekly labels through 2026-08-31), meeting the requested six years plus warm-up buffer in quantity. Invalid values prevent admission despite sufficient depth. Existing older Gamma Lens weekly history was inspected read-only: 535 SPX, 533 SPY, 316 AAPL and 528 AMD rows. No historical rows were imported as forecasts.

**Repeatable source blockers:**

- SPX 2026-03-19: open/high/low are literal `"NaN"`; close is 6606.49.
- SPY close falls below its reported low on 2020-12-17, 2021-03-18, 2022-09-15, 2022-12-15, 2024-12-19, 2025-06-18, 2025-12-18 and 2026-06-17. Example: 2026-06-17 low 739.22, close 739.05648.
- Individual-date re-requests returned the same values. The supported [Tradier history API](https://docs.tradier.com/reference/brokerage-api-markets-get-history) documents symbol, interval, start and end parameters; no documented adjustment control fixes these responses. No alternate prices, fabricated OHLC, row removal, relaxed validation or paid source was used. Regression tests preserve rejection of these observed defects.

Cboe VIX/VIX9D/VIX3M histories were available through September 4, with 9,266 / 3,941 / 4,267 rows. FRED DGS10/DGS2/DFF were available through September 3, with 1,518 / 1,518 / 2,218 observations. Required FOMC/CPI/NFP lists extend through December 2026. September entries match the official [BLS release calendar](https://www.bls.gov/schedule/2026/09_sched.htm) (NFP Sep 4, CPI Sep 11) and [FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) (Sep 15–16 meeting). PPI/PCE remain banner-only inputs; they were not added to the forecasting rules.

SPX options had standard 100-unit SPXW OCC symbols matching root, expiration, side and strike. PM cash/European interpretation follows [Cboe's SPXW specification](https://www.cboe.com/available_weeklys/). An official settlement value is still unverified under the shipped adapter and remains separate from the regular-session close.

**Session-dependent checks still required:** observed live quote latency, current opening anchors/quote freshness, actual capture completion before the database deadline, and post-close first-session minute completeness. [Tradier's entitlement description](https://docs.tradier.com/docs/market-data) is not a substitute for measuring those. No delay/realtime field appeared in the returned quotes; the existing delay setting was not changed.

## Database, deployment and scheduler

The intended local app configuration reached Neon database `neondb`, schema `public`, with target fingerprint `36d4cccd11918975`. The session reported `transaction_read_only=on`, and the existing Gamma Lens weekly/GEX/model tables matched the application. **Zero `ft_*` tables existed.** Neither reviewed migration was applied to production, and no study was registered, because release prerequisites failed. GitHub secret names exist, but their database value has not yet been compared by a production workflow against this target fingerprint.

Streamlit management access confirmed Python 3.11, main branch and `streamlit_app.py` at [gex-q-dashboard.streamlit.app](https://gex-q-dashboard.streamlit.app/). Its older displayed repository name, `spx-gex-dashboard`, resolves through GitHub to `RedTeamGo69/GammaLens`. No reboot or deployment was performed. Remote main was `ae7930fd7eb95ff3c39a883441c983aef75be2e5`; the running Cloud container's exact revision has not been certified. The new health caption exposes the checked-out revision after deployment.

The prepared forward workflow is separate from GEX. It attempts capture at 09:45, 09:55 and 10:05 ET and observation at 18:10 ET, with paired UTC entries and a New York gate. The exchange calendar restricts admission to the first actual session, 09:45 inclusive through 10:15 exclusive, with holiday/DST/early-close tests. At most three model attempts are admitted. A 25-minute job timeout and bounded provider HTTP retries limit execution; finishing late rolls back capture.

Concurrency group `weekly-forward-test` uses `cancel-in-progress: false`. A running capture is not cancelled by a retry. GitHub may replace an older **pending** attempt with a newer pending attempt; immutable slot identity and database row locks prevent successful forecasts being replaced. See [GitHub concurrency behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

The existing GEX workflow documents its cron-job.org primary at 09:28 ET. Its September 4 [dispatch run](https://github.com/RedTeamGo69/GammaLens/actions/runs/33878217962) was created at 09:28:03 ET. The two [GitHub](https://github.com/RedTeamGo69/GammaLens/actions/runs/33898667432) [scheduled backups](https://github.com/RedTeamGo69/GammaLens/actions/runs/33902458895) were created at 13:05:09 and 13:47:44 ET, outside this tester's admission window. The cron-job.org console ultimately required sign-in and no API key was accessible. No dispatcher job or GEX configuration was changed. The prepared GitHub schedule therefore has an explicit punctuality limitation; it cannot guarantee capture.

If the blockers are resolved and activation completes in time, the next eligible week is **2026-09-07**, whose first session is **Tuesday September 8**. Conditional capture attempts: **09:45 / 09:55 / 10:05 ET** (13:45 / 13:55 / 14:05 UTC); deadline **10:15 ET exclusive**. First observation: **September 8 at 18:10 ET** (22:10 UTC). Final close: **Friday September 11 at 16:00 ET**, with its scheduled observation at **18:10 ET**. These are eligible times, not enabled or guaranteed runs. If registration is delayed beyond that capture window, choose the next still-prospective week.

**Actual activation state: disabled and blocked. First live capture: not performed.** The conditional release sequence remains normal merge, target-database verification, additive migration/schema inspection, prospective registration, Cloud route/filter/Excel verification, prepared GitHub verification and only then enabling the variable. No trades or purchases occurred.
