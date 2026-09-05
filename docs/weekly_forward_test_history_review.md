# History quality follow-up — September 4, 2026 ET

**Release remains blocked and `FORWARD_TEST_ENABLED=false`.** No replacement
prices were accepted. No production migration, registration, deployment, or
capture was performed. This review found additional discrepancies beyond the
nine rows rejected by the original OHLC gate. A complete compatible alternative
has not been certified from the accessible sources; a paid subscription is not
established as necessary.

## Reproduced source defects

The authenticated Tradier production history responses were archived before
normalization, with request parameters, UTC retrieval times, and raw/sanitized
SHA-256 hashes. Full-range and individual-date requests returned identical
values on all nine original defects. The eight SPY cases are:

| Date | Open | High | Low | Tradier close | Yahoo quote close, rounded to cents |
|---|---:|---:|---:|---:|---:|
| 2020-12-17 | 371.94 | 372.46 | 371.05 | 370.66 | 372.24 |
| 2021-03-18 | 394.475 | 396.72 | 390.75 | 390.20221 | 391.48 |
| 2022-09-15 | 392.96 | 395.96 | 388.78 | 388.5236 | 390.12 |
| 2022-12-15 | 394.30 | 395.25 | 387.885 | 387.8486 | 389.63 |
| 2024-12-19 | 591.36 | 593.00 | 585.85 | 584.13445 | 586.10 |
| 2025-06-18 | 598.44 | 601.22 | 596.47 | 595.67888 | 597.44 |
| 2025-12-18 | 677.60 | 680.74 | 674.90 | 674.47663 | 676.47 |
| 2026-06-17 | 751.29 | 752.15 | 739.22 | 739.05648 | 740.96 |

SPX 2026-03-19 returns literal `"NaN"` for open/high/low and 6606.49 for
close. Yahoo `^GSPC` reports OHLC approximately
6583.12 / 6636.74 / 6557.82 / 6606.49. FRED SP500 independently corroborates
the 6606.49 close. It does not corroborate the missing daily O/H/L.

All **24** SPY cash dividends in the required window match the difference
between Yahoo's raw quote close and Tradier's close on the preceding exchange
session. Massive's dividend metadata provides an independent cash amount;
Massive's accessible daily prices corroborate the more recent close differences.
Sixteen affected closes remain inside their bars and would escape the original
inequality check. The June 20, 2025 ex-date correctly maps to June 18, before
Juneteenth. This is strong evidence of a local pre-ex-date close adjustment in
Tradier, not a date-normalization bug or a coherent adjustment of all historical
OHLC. The exact upstream implementation remains an inference, not vendor
confirmation. AAPL did **not** show this same pattern on its 24 tested dividends.

The [Tradier history endpoint](https://docs.tradier.com/reference/brokerage-api-markets-get-history)
has no documented split/dividend adjustment parameter. Its
[field documentation](https://docs.tradier.com/docs/historical) does not resolve
these discrepancies. We did not add an unsupported query switch.

## Full-window and cadence checks

The required daily window is August 10, 2020–September 4, 2026: **1,526 exchange
sessions per ticker**, with no missing sessions in either Tradier or Yahoo.
The required weekly window has 317 completed bars. Additional older weekly
history was examined because the adapter consumes it, rather than limiting the
audit to the six-year fit cutoff.

The comparison tool records every original value, missing/extra date, and
discrepancy. Its 0.0051 price-unit threshold is only a diagnostic comparison
threshold for half-cent rounding plus Yahoo float32 representation. It is not a
new forecast admission tolerance and is never applied to prices.

| Comparison | SPX | SPY | AAPL | AMD |
|---|---:|---:|---:|---:|
| Tradier daily vs Yahoo raw daily: discrepant rows | 14 | 29 | 4 | 6 |
| Tradier weekly vs aggregation of Yahoo daily, full older coverage | 9 | 7 | 2 | 2 |
| Older persisted rows actually prepended by `prepare()` | 218 | 217 | 9 | 217 |
| Older persisted vs current Tradier: discrepant rows | 0 | 1 | 9 | 0 |

All nine original defect periods have weekly Tradier bars consistent with the
corresponding Yahoo daily aggregation at this comparison precision. The missing
SPX day's reported alternative high/low would not change that week's extrema.
Agreement at weekly cadence does not certify the defective daily closes.
Tradier's required weekly bars also agree numerically with its own daily
aggregation; this self-consistency is not independent validation, and pandas'
aggregation can conceal missing daily fields.

Further discrepancies affect consumed inputs even though OHLC inequalities pass:

- AAPL 2023-09-11: Tradier O/H/L/C = 179.45 / 179.55 / 179.45 / 179.55,
  volume 9,681. Yahoo and Massive are approximately
  180.07 / 180.30 / 177.34 / 179.36. Tradier's weekly open and high also differ.
  September 12 close is 176.21 versus 176.30. Individual requests reproduce both.
- AMD 2023-09-11: Tradier O/H/L/C are all 104.80, volume 1,179. Yahoo and
  Massive show a materially wider day, approximately
  107.32 / 107.51 / 103.00 / 105.32. The weekly open differs as well.
- SPX 2026-08-31: Tradier weekly open is 7692.21; Yahoo's is about 7697.52.
  Both are internally valid. FRED's close-only series cannot adjudicate this
  opening value. Six older SPX weeks have additional OHLC discrepancies.
- Yahoo is not an automatically correct replacement: its SPX 2021-08-11 close
  is about 4442.41, while Tradier and FRED both return 4447.70. A narrow Yahoo
  request reproduces this. Six Yahoo closes differ from FRED beyond comparison
  precision; Tradier differs from FRED on 2020-11-03 (3369.02 vs 3369.16).
- Yahoo's weekly endpoint appends an extra non-Monday quote row for all four
  tickers; the audit retains and flags it. SPY's holiday week 2024-09-02 has an
  additional open mismatch: weekly 558.35 versus first actual daily open 560.47.
  Narrow re-request reproduces the weekly value. No extra date is admitted or
  silently dropped.

## Accessible alternatives and conventions

Yahoo chart requests used `^GSPC`, SPY, AAPL and AMD, New York session dates,
`includePrePost=false`, and the raw `indicators.quote` OHLC. No `adjclose`, repair,
back-adjustment or automatic adjustment was used. Yahoo identifies these as
USD, with New York exchange time; `^GSPC` is an index, not SPY. Its
[history legend](https://ca.finance.yahoo.com/quote/SPY/history/) distinguishes
split-adjusted close from split-and-dividend-adjusted close. The raw series
cross-checks AAPL's August 31, 2020 four-for-one split in both sources; Massive
corporate-action metadata independently confirms the split. Numeric agreement
alone does not resolve differences in qualifying trades or index opening time.

The existing Massive connector successfully returned **1,255 daily rows per
stock**, September 7, 2021–September 4, 2026, with `adjusted=true`. Requests for
SPY December 2020 and `I:SPX` March 2026 explicitly returned `NOT_ENTITLED`.
Thus 271 required stock sessions remain outside the demonstrated access and
SPX index access was not available. This is measured account capability, not an
assumption about the subscription's name. Existing app and GitHub credential
names contain no Massive/Polygon key for an unattended job; the connector token
was neither extracted nor copied.

[Massive documents split adjustment, not dividend adjustment](https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends).
Its aggregates depend on qualifying trades and support
[extended hours](https://massive.com/knowledge-base/article/market-data-outside-of-normal-hours).
SPY 2023-10-30 low is 408.91 at Massive versus 412.22 at Yahoo/Tradier. Massive's
separate open-close endpoint returns the same 408.91 low and distinct
pre-market/after-hours fields; merely changing endpoint does not resolve it.
AMD also differs that day. We did not label an aggregate as regular-session
equivalent without evidence. These connector outputs are archived as
**connector-normalized evidence**, not represented as raw HTTP bodies.

[FRED SP500](https://fred.stlouisfed.org/series/SP500) is a price-index closing
series sourced from S&P Dow Jones Indices. It supplies independent close
evidence across the required period, but no O/H/L. The existing Cboe adapter
supplies VIX-family history, not this missing SPX OHLC capability. Cboe's
[SPX-containing EOD product](https://datashop.cboe.com/main-channel-end-of-day-summary)
is separate; no subscription was purchased or presumed accessible.

## Model inputs, existing history, and UI consistency

`compute_hv_windows` receives daily close only: log returns, 5/10/20-session
rolling volatility, weekly sampling and a one-week lag. The model consumes
`hv_ratio` in M3/M4; M1/M2 do not consume daily close directly. Altering unused
daily O/H/L does not change that computation. The regression fixture proves that
the observed SPY close difference changes HV ratio. This does not justify
accepting an unverified close or weakening observation OHLC validation.

Weekly `(H-L)/O` and its log provide range targets and the lagged HAR features in
all four models. Weekly `(C-O)/O` and its absolute lag enter M3/M4. Side placement
uses the pooled distribution of `(H-O)/(H-L)` over all supplied completed weekly
history. Consequently, apparently valid but incorrect weekly opens matter to
all models. No model was refit to select a preferred data source.

The older SPY week 2020-07-06 and nine older AAPL weeks have lower stored price
levels than fresh split-only prices, consistent with a different adjustment
basis. Uniform scaling within a bar cancels in range/return/side ratios: this
is not evidence of an equally large model error. Measured maximum older
range-ratio differences versus current Tradier were about 4.0e-8 for SPY and
1.37e-6 for AAPL; side-share input differences were about 1.19e-6 and 1.24e-4.
Older SPX/AMD prices match current Tradier but still inherit the cross-vendor
disputes above. No stored history or derived calibration was overwritten.

The interactive `bar_sources._validate_bars` does not enforce full OHLC
inequalities and drops missing values when checking columns. It can consume
these suspect histories. Its yfinance fallback also leaves `auto_adjust` at the
library default, unlike this diagnostic's raw quote comparisons. Existing
legacy history has differing last refresh dates. These operational differences
preclude claiming live UI/study recommendation identity from source availability.

The shared recommendation code still has identical UI/headless results under
identical inputs, across all 4 tickers × 4 models × 4 tiers, as tested by the
existing parity test. **No accepted source policy changed in this update**, so
the stable runtime methodology identity remains the same as commit `4adc362`.
Audit scripts and fixtures are not imported by the runtime. Any future accepted
adapter/policy must be included in the methodology hash and immutable input
provenance. No inputs from this diagnostic are accepted for capture or scoring.

## Reproduction and release state

1. `python scripts/diagnose_forward_history.py --secrets-file .streamlit/secrets.toml --output-dir <new-local-directory>`
   makes bounded read-only history requests and reads legacy tables using a
   read-only database transaction. It never calls `prepare`, fitting, capture,
   migration, or registration. Keep full provider responses local.
2. `python scripts/analyze_forward_history.py --evidence-dir <archive> --output <new-report.json>`
   validates archived hashes, compares both cadences, records every discrepancy,
   and reports `UNAVAILABLE_SOURCE_POLICY_NOT_CERTIFIED`. Its exit code 1 is
   deliberate: a numerical comparison is not a release certificate. Optional
   Massive evidence is supplied from the separately authorized read-only connector.
3. `python -m pytest -q` with the explicit isolated
   `FORWARD_TEST_TEST_DATABASE_URL` repeats source regressions, all UI/headless
   parity cases, two-week aggregation, and real PostgreSQL constraints/triggers.
   Use the validated project `.venv\Scripts\python.exe` on Windows.

The exact local full-suite command completed with **565 passed in 129.09s**, no
failures or skips, on PostgreSQL 17.11 at the explicit loopback-only test URL.
This includes 12 new source-audit regressions, both real PostgreSQL tests and
the cumulative scoreboard proof: two distinct actual refits produce one SPY/M2/
Point group with `close_n=2`, `path_n=2`; removing the actual VIX feature produces
a separate group with one eligible week. Fit parameters, covariance, training
cutoffs and source-input hashes remain distinct. These are isolated test
results, not prospective performance. The GitHub workflow is run on the final
branch commit; its result is reported with the release status.
The subsequent missing-fallback regression and focused source-audit command
completed with **13 passed in 3.20s**. No runtime admission code was changed.

Focused public fixtures preserve the original defect shapes and source hashes.
The full local artifacts include `manifest.json`, sanitized raw responses,
`forward-history-analysis-20260905.json`, supplemental narrow requests,
corporate-action/entitlement evidence, derived-input comparisons, the JUnit
report and refreshed prerequisite report. No DEMO_ONLY records entered production.

The September 5 03:19:48 UTC prerequisite check again reached the intended
read-only Neon target fingerprint `36d4cccd11918975`, database `neondb`, with
**zero `ft_*` tables**. Quotes, expirations, chains, SPXW metadata, Cboe and FRED
capabilities remain available; the history gate still fails. All 16 ticker/model
combinations remain uncertified by this broader source review, even where the
original shape-only test passes. Observation OHLC requirements are unchanged.

The concrete missing capability is a **verified, compatible history policy
covering SPX daily closes and weekly OHLC, plus the full required stock windows
and older consumed weekly context**, accessible to the unattended workflow.
In particular the unresolved SPX weekly opening values cannot be settled with
FRED close data or the current Massive entitlement. Corrected primary data or
verified compatible evidence could resolve this; a purchase is not prescribed.

PR #92 remains a draft review branch. Merge, production migrations,
registration, Cloud reboot/deployment, and activation remain blocked. The
prepared GitHub schedule and separate non-cancelling concurrency group are
unchanged; cron-job.org remains unconfigured for this tester. GitHub punctuality
is still a limitation. The next eligible capture, conditional on completed
release, is Tuesday September 8 at 09:45/09:55/10:05 ET, deadline 10:15 exclusive;
observation is 18:10 ET. Recompute the prospective week if that window passes.
Actual live latency, opening anchors, deadline completion and the first live
capture remain session-dependent and unverified. Both the disabled flag and the
absence of production records distinguish this state from activation while
waiting for a first session. The old stopped PostgreSQL folder was left alone.
