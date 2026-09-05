# Verification record — 2026-09-04

Implementation and activation details are in [weekly_forward_test.md](weekly_forward_test.md).

This is the original implementation-only verification record. The subsequent [pre-activation release review](weekly_forward_test_release_review.md) records the newer 553-test result, live source blockers and disabled release state.

## Tests

Interpreter validation: `.venv\Scripts\python.exe --version` returned **Python 3.11.9**.

The original suite passed **486 tests in 20.92 seconds** before implementation. No pre-existing failures were observed. After the completed scorer, source, persistence, UI and export changes, the exact command below passed **549 tests in 53.24 seconds**, with the two PostgreSQL tests enabled and no skips or failures:

```powershell
# FORWARD_TEST_TEST_DATABASE_URL pointed only to the temporary local test DB.
.venv\Scripts\python.exe -m pytest -q --tb=short
```

The final dashboard-only check also passed **2 tests in 9.51 seconds** after retaining the existing download-button behavior for compatibility with the repository's Streamlit minimum.

The isolated PostgreSQL server was version 17.11, listening only on `127.0.0.1:55439`, using the disposable database `gammalens_forward_test`. Each test created and dropped a private schema. The production database was never migrated or written. No Windows service was installed. The temporary browser, preview server and PostgreSQL cluster were stopped after validation; neither test port remains listening.

Automatic approval review blocked deletion of the temporary PostgreSQL folder, without a specific reason. Its stopped cluster, downloaded binaries and test-only credentials remain at `C:\Users\luisq\AppData\Local\Temp\gammalens-forward-postgres`. These are isolated test credentials, not production secrets. No alternate deletion method was attempted.

Coverage includes 16 production ticker/model combinations with four tiers, shared UI/headless parity, no GEX influence, no target-week outcomes in fits, saved-fit incompatibility, deadline admission using the database clock, idempotency/concurrent capture, immutable SQL triggers, atomic rollback, partial failures, bounded capture retries, delayed outcome reconciliation, source freshness/identity, holiday/DST/early-close calendars, expiry and SPXW root identity, strict breaches/touches/recoveries/boundaries, independent denominators and Excel export. A conflicting duplicate minute and the closing trade in a partially observed minute cannot establish a prospective breach.

## End-to-end fixture evidence

`scripts/demo_forward_test.py` used production capture/fit/strike/scoring/storage code with deterministic source fixtures and isolated SQLite persistence. It produced:

| Check | Observed result |
|---|---|
| Capture | 16 model slots and 64 tier forecasts |
| Duplicate trigger with another version | 0 new forecasts; frozen records unchanged |
| Observations | 16 ticker/session observations for the four-session Labor Day week |
| Close results | 48 / 64 inside closes |
| Breach results | 48 / 64 complete paths breached |
| Recovery results | 32 / 48 breached records recovered inside |
| Close failures | 0 below put; 16 above call |
| Pending/missed/incomplete | 0 in this deliberately complete fixture |
| Audit inputs | 1,435,503 JSON bytes |
| Observations | 141,652 JSON bytes |

These numbers are **synthetic test evidence**, not a prospective performance sample. The data use study `DEMO_ONLY` and cohort `deterministic_fixture` and are never imported into production.

Evidence artifacts are in:

`C:\Users\luisq\.codex\visualizations\2026\09\04\01a06eb2-7a63-7780-a9cb-52c32e128112\forward-test-final`

They include `fixture.sqlite`, `fixture-summary.json`, `fixture-rows.json`, `fixture-results.xlsx` and visual previews. The demo can be reproduced in a new directory using the runbook command. The Excel file has Scoreboard, weekly, Breach details and Read me sheets. All four sheets were rendered and visually reviewed. The preview runtime emitted the PNGs but returned a nonzero exit during teardown; workbook content and rendered images were checked independently.

## Dashboard and export

Streamlit AppTest exercised the production results renderer against persisted fixture results: 64 rows, then 16 AMD rows, then four M2 rows, with the Excel button present. It also exercised `streamlit_app.py?tab=forward`, proving that the route renders before the live quote/credential gate. The unactivated schema case gives an explanatory message without writing tables.

A real local browser additionally loaded the read-only fixture viewer at `127.0.0.1:8508`. Selecting AMD changed the eligible count to 16, close hit rate to 0%, breach rate to 100%, recovery rate to 0%, and call-side failures to 16. Browser errors and server exception logs were empty. The snapshot and image showed the synthetic-data notice and capture/scoring health.

The browser's workbook request returned HTTP **200** with the Excel MIME type. The served file was retrieved and opened independently with openpyxl: four expected sheets and exactly 16 AMD Scoreboard rows. The automation tool reported **“Download was canceled”** when asked to save the click-triggered file, so successful completion of that browser save dialog is not claimed. The HTTP download response, file content and application export tests passed. Live Streamlit Cloud behavior remains an activation check.

## Repository and external state

The branch is `codex/weekly-forward-tester`; HEAD remains `ae7930fd7eb95ff3c39a883441c983aef75be2e5`. Changes are local and unstaged. `git diff --check` found no whitespace errors. Git emitted its existing LF/CRLF conversion notices; no normalization commit was made.

The pre-existing `.gitignore` edit, `AGENTS.md`, candidate/gap/target-form research files and candidate-feature test were preserved. The master spreadsheets were read only and were not imported. Legacy scoring/calibration, GEX policy and research collection remain intact.

**Not activated:** no commit/push, production migration, deployment, external scheduler change, live forecast capture or order submission. Remaining steps are review/merge, explicit migration and study registration, deployment, credential/data checks and enabling the prepared workflow as described in the runbook. Official option settlement is unverified under the shipped data adapter.
