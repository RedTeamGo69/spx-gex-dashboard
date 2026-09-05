# Weekly Spread Finder forward test

Implemented locally on `codex/weekly-forward-tester`, based on `ae7930fd7eb95ff3c39a883441c983aef75be2e5`. The implementation is observational. It contains no order submission or execution accounting. Production migrations, deployment, GitHub push and schedule activation have **not** been performed.

## Methodology

The study universe is exactly **SPX, SPY, AAPL and AMD**. Each week has 16 model slots and, when all models are available, 64 immutable tier forecasts.

| Model | Production feature specification before the existing availability filter |
|---|---|
| `M1_baseline` | `har_d1`, `har_w`, `har_m` |
| `M2_vix` | HAR core plus `vix_close` |
| `M3_extended` | M2 plus `hv_ratio`, `event_count`, `spx_return_lag1`, `abs_return_lag1` |
| `M4_full` | M3 plus `vix_ts_slope`, `yield_spread` |

The four tiers are **Lower PI**, **Point Estimate**, **80% PI Upper**, and **Effective (+buffer)**. The engine preserves production `PI_ALPHA=0.20`, the six-year fitting window, COVID exclusions, event buffer, side-share quantile, ticker rounding and chain snapping. Conformal correction and GEX influence remain off. No calibration coefficients, subscriptions, GEX research collection or legacy outcome semantics were changed.

The interactive finder and headless capture now call the same recommendation functions. The current UI and master-workbook export display the original `model_*` strikes when an expected-move floor widened the internal ladder. The new study freezes those actual displayed strikes. It does not substitute the wider contextual ladder. Available long strikes, widths and contract symbols are retained; they are not evidence of an executed spread.

Fresh production fits are calculated in memory from completed data. The tester neither loads nor overwrites legacy saved models. An explicit compatibility gate rejects wrong ticker/model/schema, disabled GEX features and target-week training if a saved-fit adapter is added later. The shared production feature availability filter remains in force; actual columns and omitted columns are archived. A missing current value for a selected feature fails the model rather than using the forecasting function's mean fallback. Missing M4 supplemental sources explicitly make M4 unavailable. Different actual feature subsets receive different model-version identities.

The underlying's first-session daily open and VIX open anchor the range. Quote identity, session date, positive prices and freshness are checked. Freshness allows the declared feed delay (0 or 900 seconds) plus 120 seconds; the default is real time. Historical weekly/daily bars come from Tradier, volatility history from Cboe, macro history from the existing FRED collector and known events from the existing event calendars. No primary yfinance path was added. Existing older weekly history is read only to retain the UI's history depth for side-share estimation and HAR lags; the newest history is fetched before generation. The original feature arithmetic is shared through an injected, non-persisting `build_features` path.

The target week's outcomes and later rows cannot enter training. Daily exchange-session gaps, missing prior-week bars, stale prior-final-session VIX history, duplicate labels and invalid OHLC are rejected. M4 detects stale/missing term-structure or macro sources. Stored data are the source snapshots available when this forecast was generated; subsequent historical refreshes cannot replace them. Calendar event coverage must extend through the evaluation week. When the existing event lists expire, extend those calendars through their normal maintenance process; forecasts fail explicitly until coverage is restored.

## Time and admission

All decisions use **America/New_York** and the NYSE exchange-session calendar. Timestamps are stored with UTC offsets. The week is labeled by its Monday date even when Monday is closed.

- Capture opens **15 minutes after the first actual session opens** and ends **45 minutes after that open, exclusive**: normally **09:45 <= availability < 10:15 ET**.
- Scheduled attempts are 09:45, 09:55 and 10:05 ET on weekdays. Only the week's first actual session admits a primary capture.
- `scheduled_at` is the opening-window start. `prepared_at`, actual `captured_at`, `available_at`, anchor time, prior-session feature cutoff, actual last training-outcome close and declared data delay are stored separately. Availability is measured after computation finishes, not at the opening anchor or runner start.
- PostgreSQL checks its own current clock under the slot lock and again after the input/forecast writes, before commit. Crossing the deadline during those writes rolls back the entire capture. An application timestamp inside the window cannot override either check.
- Each model has at most three capture attempts. Its four tiers and inputs commit together. Other models/tickers may succeed independently. Successful slots cannot be reopened by a new fit, version, expiry, duplicate trigger or later deployment.
- A missed window becomes an explicit missed slot with its last error. An outage is reconciled into missed slots when the runner returns. There are no retrospective forecasts, optional late cohorts or spreadsheet-history imports in the primary study.

For example, the week labeled 2026-09-07 opens on Tuesday 2026-09-08. The capture window is Tuesday 09:45–10:15 ET. Thanksgiving week's Friday evaluation uses the calendar's 13:00 close; a Friday-holiday week evaluates on its actual last session.

## Expiration, observations and scoring

The expiration must be both listed by Tradier and equal to the week's last actual session. There is no nearest-date or next-week fallback. SPX requires a standard 100-unit **SPXW PM-settled European cash contract**, including matching OCC symbol, root, date and strike. AM-settled SPX monthlies and adjusted/unknown contracts are rejected. Cboe identifies SPXW as PM settled and describes the holiday adjustment for equity/ETF weeklies. [Cboe weekly options](https://www.cboe.com/available_weeklys/)

The evaluation is the **final regular-session close** of that week. Primary path evidence covers **regular trading sessions from actual forecast availability through that close**. It does not claim overnight or extended-hours coverage. Each completed full session contributes daily OHLC. The partial opening session also requires one-minute bars, collected after the session; all models for a ticker share those observations.

Bars become eligible only when collected at least **90 minutes after the session close**. The scheduled observation run is **18:10 ET**. Morning runs also reconcile eligible previous-session data. Each ticker/session is checked at most once per New York calendar day. Missing/invalid outcome data are retried on subsequent days, up to **14 calendar days after evaluation**. Complete ordinary daily bars are rechecked through age four days; first-session minute evidence and final SPX settlement remain eligible through the longer window. After it expires, incomplete records remain visible and excluded from the affected denominators. Frozen forecasts are never repaired with new strikes.

The adapter requests `session_filter=open`, `interval=1min`; Tradier documents 20 days of regular-session one-minute history. That supports the bounded reconciliation period, but it does not guarantee availability or completeness for any ticker/account. [Tradier history limits](https://docs.tradier.com/reference/brokerage-api-markets-get-timesales)

| Result | Definition |
|---|---|
| Weekly close success | `put <= final close <= call` |
| Put/call strict breach | `low < put` / `high > call` |
| Put/call touch | `low <= put` / `high >= call`; equality alone is not a strict breach |
| Close boundary | Final close equals either short strike |
| Earlier close outside | Any earlier eligible daily close is outside; unknown if missing evidence prevents determining absence |
| Recovery | A verified strict breach followed by an inside final weekly close |
| Width | `(call - put) / opening reference` |

Classifications distinguish inside/no breach, inside after put/call/both-side breach, below put, above call and pending/incomplete/invalid data. An earlier outside close does not override a successful final close. No result represents realized P&L, premium, maximum loss or assignment avoidance.

The minute straddling availability is treated conservatively. An entirely contained bar can establish absence of breach during its partial interval. An outside extreme, including its closing trade, may predate availability and cannot independently establish a prospective breach. Later unambiguous evidence may establish that side's event; otherwise the side remains unknown. Missing or conflicting duplicate minutes prevent path eligibility. First-breach sessions and minute intervals are retained where supported; **exact breach timestamps remain blank** for OHLC evidence. Whole-week daily extremes are separate and include pre-forecast movement. Observed prospective extremes exclude the unresolved partial minute. Tradier documents the OHLC fields as interval values. [Tradier time-and-sales definitions](https://docs.tradier.com/docs/timesale)

Official option settlement is separate from the range test. The scorer can retain a source-identified, matching-root/date verified settlement value. **The shipped Tradier adapter does not supply an official settlement feed and records settlement as unverified.** It never relabels the underlying close as official settlement. Stock/ETF results are explicitly theoretical expiration proxies.

## Persistence, results and usage

Eight additive tables hold studies, admission slots, deduplicated input snapshots, immutable forecasts, observation revisions, observation-check metadata, score revisions and run health: `ft_studies`, `ft_slots`, `ft_inputs`, `ft_forecasts`, `ft_observations`, `ft_observation_checks`, `ft_scores`, `ft_runs`.

Study/cohort identity is immutable. A logical `(study, week, ticker, model)` slot intentionally prohibits replacement even with a different model version or expiry; `(slot, tier)` is unique. Version/expiry/contract details live in the immutable forecast. The cumulative `model_version` hashes methodology sources/configuration/dependencies, model name and the actual selected feature columns. It deliberately excludes weekly training dates, rows, coefficients and covariance. Those belong to separate immutable `fit_id`, `input_id` and `source_inputs_id` identities. Routine refits therefore accumulate under one comparison version; code/policy/rule changes or a different actual feature subset stay separate. Inputs include training rows, forecast feature vector, coefficients/covariance, source times, anchor, ticker configuration and side placement. PostgreSQL triggers reject evidence updates/deletions. Transaction and advisory locks serialize capture and observation/score revisions. A correction from value A to B and back to A produces three ordered versions. Unchanged retries append nothing.

The Forward Test tab reads durable records without requesting a live quote or initializing any schema. It includes filters for study, cohort, ticker, model, tier, version and week; independent close/path/recovery denominators; width, failure and pending/missed/incomplete counts; latest run/capture/scoring times; and errors. Comparison rows retain separate versions/cohorts. Registered studies, filters and a headers-only Excel export remain available before the first capture, without inventing forecasts. The health area reports the checked-out app revision when Git metadata is available. The Excel button exports that filtered snapshot into a Scoreboard, weekly four-tier tabs, Breach details and definitions. The two supplied master workbooks were inspected as layout/comparison references and remain unchanged; none of their historical rows were imported.

Expected usage, excluding retries and vendor corrections:

- Opening capture: about **20 Tradier GETs** (two histories, one batch quote, expirations and one chain per ticker), **3 Cboe history requests** and **3 FRED series requests**. Common supplemental inputs are reused across the four tickers in one invocation. Failed attempts are capped at three per model; the study's Tradier history/quote/timesales requests have 20-second timeouts and at most two HTTP retries. Existing chain/source clients are reused; the overall GitHub job has a 25-minute limit.
- A five-session week initially needs **20 daily-bar requests plus 4 minute-history requests** for observations. Daily reconciliation adds bounded rechecks; plan for roughly **150–250 observation GETs per study week** with ordinary weekday scheduling, before HTTP retries. Outages, source corrections and manual invocations affect that estimate. No continuous polling or full-chain archives are stored.
- A runner uses **one database connection**, shared observations and four-tier transactions. Typical primary growth is 64 forecast rows, 16 model snapshots, 4 shared ticker-input snapshots and 20 first observation revisions per full week, plus score revisions and small run/check metadata. JSON input sizes depend on history depth. The 160-week deterministic fixture used **1,435,503 input JSON bytes** and **141,652 observation JSON bytes**; these are measured fixture sizes, not production storage estimates. Production six-year/deeper snapshots will be larger, ordinarily a few MB per week before indexes/backups.
- The dashboard caches reads for ten minutes and opens/closes one connection per cache miss. Historic finalized forecasts age out of runner work; old incomplete outcomes are terminalized after reconciliation expires. No dollar billing estimate is claimed.

## Verification and reproduction

The original full suite passed **486 tests** before implementation, with no pre-existing failures. The integrated suite with real isolated PostgreSQL passed **548 tests in 52.45 seconds** before the final UI/export and duplicate-minute hardening; the final validation result is recorded in `weekly_forward_test_verification.md`.

Coverage includes all 16 ticker/model combinations and four tiers, shared UI/headless parity, GEX invariance, target-week exclusion, saved-fit incompatibility, admission deadlines, concurrent/duplicate writes, atomic rollback, partial failures/retry limits, delayed outcomes, calendar holidays/early closes/DST, actual expiry/root selection, every breach/recovery direction, boundaries, missing/stale/wrong-identity bars, settlement unavailable/verified separation, denominator/version logic, Excel literals and the real app's Forward Test route. PostgreSQL tests use only `FORWARD_TEST_TEST_DATABASE_URL`, reject non-local/non-`*_test` databases and create/drop private temporary schemas. They never use `DATABASE_URL`.

```powershell
.venv\Scripts\python.exe --version
# Configure FORWARD_TEST_TEST_DATABASE_URL for a disposable local *_test DB.
.venv\Scripts\python.exe -m pytest -q
# Without that explicit test URL only the two PostgreSQL tests are skipped.
.venv\Scripts\python.exe scripts/demo_forward_test.py --output-dir C:\Temp\gamma-forward-fixture
.venv\Scripts\python.exe -m streamlit run scripts/view_forward_test_fixture.py --server.address 127.0.0.1 --server.headless true -- --database C:\Temp\gamma-forward-fixture\fixture.sqlite
```

The demonstration requires a new output directory and produces fixture-only SQLite, JSON and Excel evidence. Its separate `DEMO_ONLY` study and `deterministic_fixture` cohort captured 64 forecasts, rejected a replacement-version trigger, then scored 48/64 close successes, 48/64 breached records and 32/48 breach recoveries. This is synthetic evidence, not model performance. Streamlit AppTest verifies 64 records, 16 after AMD filtering and 4 after selecting M2. A local browser additionally verifies the rendered fixture view and served Excel content.

## Activation instructions — blocked by live history prerequisites

The final pre-activation review is in [weekly_forward_test_release_review.md](weekly_forward_test_release_review.md). Keep `FORWARD_TEST_ENABLED=false` until the recorded source blockers are resolved and the release gates below pass. Read-only prerequisites can be repeated with `scripts/check_forward_test_prerequisites.py --secrets-file .streamlit/secrets.toml --output <local-evidence.json>` using the project interpreter. This command never fits models, writes forecasts or migrates tables.

1. Merge the scoped implementation through the repository's normal process only after its checks and live prerequisites pass. Keep the repository variable `FORWARD_TEST_ENABLED=false` throughout preparation. The new workflow is independent of the existing GEX snapshot and cron-job.org schedules.
2. Choose the production database and explicitly set `FORWARD_TEST_DATABASE_URL` or `DATABASE_URL` in the operator's shell. Do not use the isolated test URL. Run `.venv\Scripts\python.exe forward_test.py migrate` once. This applies only the two additive `ft_*` migrations. Neither the runner nor Streamlit applies them automatically.
3. Register the first still-prospective week: `.venv\Scripts\python.exe forward_test.py register --start-week YYYY-MM-DD`. Use its Monday label. For 2026-09-07 this is only valid before Tuesday 2026-09-08 at 10:15 ET; if that window has ended, choose a later week. The study ID defaults to `spread-finder-weekly-v1`; its start/configuration cannot be rewritten.
4. Confirm the existing GitHub secrets `DATABASE_URL`, `TRADIER_TOKEN` and `FRED_API_KEY` refer to the intended account/database. The workflow uses `DATABASE_URL`; the optional CLI override is for explicit operator/test environments. Leave `FORWARD_TEST_DATA_DELAY_SECONDS` at `0` unless the actual feed is known to be delayed, in which case set it to `900`. This does not buy or change any entitlement.
5. Deploy the reviewed code to Streamlit Cloud and verify the **Forward Test** tab. If the container still serves old code, use **Reboot app**. The results tab should work before any live market quote is requested. Point its existing `DATABASE_URL` secret at the same database.
6. Run the prepared **Forward Test Verification** workflow, which uses its own disposable PostgreSQL service. Then set the repository variable **`FORWARD_TEST_ENABLED=true`** to activate the capture/reconciliation workflow. The workflow must be on the default branch for scheduled events. A manual dispatch is also gated by this variable; it cannot bypass the engine's prospective window.
7. After the first eligible opening run, check the dashboard and workflow health: 16 captured models / 64 tiers, or explicit unavailable/missed reasons. After the final-session collection, check independent close and path sample counts. Missing minute history may reduce path coverage while valid close results remain available. Verify the real account's SPX, quote and history availability during this first run; no live production certification was attempted here.

The cron contains paired UTC times for DST and a New York time gate. Its separate `weekly-forward-test` concurrency group has `cancel-in-progress: false`: overlapping attempts wait without cancelling a healthy capture. GitHub retains at most one pending run, so a newer retry can replace an older pending attempt, but cannot replace captured database slots. GitHub may delay or drop scheduled jobs, so this architecture cannot guarantee every week's capture. Late computation is rejected and reported as missed, never backdated. An additional cron-job.org trigger is authorized by the release request, but the management console requires sign-in and no API credential was available during review. No existing GEX job was changed and no extra dispatcher job was created. Retain the prepared GitHub schedule with this punctuality limitation. [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

To pause after activation, set `FORWARD_TEST_ENABLED=false`. Leave the immutable study tables intact. Resuming later marks elapsed unobserved capture windows as missed. A new methodology may create distinct future model versions, but it cannot replace existing forecasts.

## Changed files and preserved work

- Entry point and UI: `forward_test.py`, `ui_forward_test.py`, `streamlit_app.py`, `ui_theme.py`.
- Shared production extraction: `range_finder/recommendations.py`, `range_finder/trading_week.py`, `range_finder/har_model.py`, `range_finder/feature_builder.py`, `range_finder/event_calendars.py`, `ui_spread_finder.py`. The expiry helper now requires the actual listed final-session date instead of its old +/-3-day fallback.
- Contract audit metadata: `phase1/data_client.py`.
- New study package: `range_finder/forward_test/{__init__,config,provider,capture,store,runner,scoring,results}.py`.
- Schema: `migrations/001_forward_test.sql`, `migrations/002_forward_test_immutability.sql`.
- Inactive execution and test workflows: `.github/workflows/forward_test.yml`, `.github/workflows/forward_test_checks.yml`.
- Evidence/tests: `range_finder/tests/forward_fixtures.py`, `test_forward_test.py`, `test_forward_provider.py`, `test_forward_postgres.py`, `tests/test_forward_dashboard.py`, `scripts/demo_forward_test.py`, `scripts/view_forward_test_fixture.py`, `scripts/check_forward_test_prerequisites.py`, and these documentation files.

Pre-existing `.gitignore`, untracked `AGENTS.md`, candidate/gap/target-form research files and `test_candidate_features.py` were preserved. No changes were made to legacy `spread_log`, its SPX high/low loss scorer, its calibration consumers, `saved_models`, existing scheduled workflows or GEX collection universes. The implementation is prepared on `codex/weekly-forward-tester`; the release review records the conditional activation status.
