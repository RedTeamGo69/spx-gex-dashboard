from __future__ import annotations

import ast
from datetime import date, datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

import capture_gamma_archive as collector
import phase1.gamma_archive_capture as calculation_module
import phase1.gamma_level_history as history
import phase1.rates as rates


NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]


def ny_dt(year, month, day, hour, minute, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=NY)


def completed_calculation(captured_at):
    return SimpleNamespace(
        captured_at=captured_at,
        target_expirations=("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-28"),
        quote={"prevclose": 650.0},
        spot=651.25,
        spot_info={"spot": 651.25},
        gex_df=pd.DataFrame({"strike": [650.0]}),
        stats={"net_gex": 1_000_000.0, "coverage_ratio": 0.98},
        all_options=[],
        levels={
            "zero_gamma": 649.0,
            "call_wall": 655.0,
            "put_wall": 645.0,
            "net_gex": 1_000_000.0,
            "call_wall_gex": 400_000.0,
            "put_wall_gex": -300_000.0,
            "zero_gamma_is_true_crossing": True,
            "zero_gamma_method": "crossing_fine",
        },
        regime_info={"regime": "Positive Gamma"},
        staleness_info={"freshness_score": 94.0},
        confidence_info={"score": 91.0},
        em_analysis={"expected_move": {"expected_move_pts": 4.25}},
    )


def collector_dependencies(events, calculation=None, save_result=True):
    calculation = calculation or completed_calculation(ny_dt(2026, 8, 24, 10, 3))

    class FakeClient:
        def __init__(self, token):
            events.append(("client", token))

        def get_expirations(self, ticker):
            events.append(("expirations", ticker))
            return list(calculation.target_expirations)

    def load_rate(key, **kwargs):
        events.append(("rate", {"key": key, "kwargs": kwargs}))
        return {"rate": 0.04, "curve": {30: 0.041, 91: 0.042}}

    def calculate(**kwargs):
        events.append(("calculation", kwargs))
        return calculation

    def save(**kwargs):
        events.append(("save", kwargs))
        return save_result

    return FakeClient, load_rate, calculate, save


def run_collector(captured_at, events, calculation=None, save_result=True):
    client, rate, calculate, save = collector_dependencies(
        events, calculation=calculation, save_result=save_result
    )
    return collector.capture_intraday_gamma_archive(
        captured_at=captured_at,
        environ={
            "TRADIER_TOKEN": "tradier-test",
            "DATABASE_URL": "postgresql://test",
            "FRED_API_KEY": "fred-test",
        },
        client_factory=client,
        rate_loader=rate,
        calculation_func=calculate,
        save_func=save,
    )


def test_default_and_only_production_ticker_is_spy():
    events = []
    assert collector.RECOMMENDED_ARCHIVE_TICKER == "SPY"
    assert run_collector(ny_dt(2026, 8, 24, 10, 3), events) == 0
    assert ("expirations", "SPY") in events
    calculation_call = next(value for kind, value in events if kind == "calculation")
    assert calculation_call["ticker"] == "SPY"
    save_call = next(value for kind, value in events if kind == "save")
    assert save_call["ticker"] == "SPY"


def test_intraday_bucket_examples_reach_expected_bucket():
    assert history.gamma_archive_bucket(ny_dt(2026, 8, 24, 10, 3)).bucket_start.hour == 10
    assert history.gamma_archive_bucket(ny_dt(2026, 8, 24, 10, 31)).bucket_start.isoformat() == "10:30:00"
    assert history.gamma_archive_bucket(ny_dt(2026, 8, 24, 15, 31)).bucket_start.isoformat() == "15:30:00"


def test_1531_is_valid_but_1600_does_no_io():
    valid_events = []
    assert run_collector(ny_dt(2026, 8, 24, 15, 31), valid_events) == 0
    assert any(event[0] == "save" for event in valid_events)

    invalid_events = []
    assert run_collector(ny_dt(2026, 8, 24, 16, 0), invalid_events) == 0
    assert invalid_events == []


def test_opening_bucket_is_owned_by_existing_job_and_does_no_io():
    events = []
    assert run_collector(ny_dt(2026, 8, 24, 9, 31), events) == 0
    assert events == []


def test_weekend_holiday_and_early_close_invalid_times_do_no_io():
    invalid_times = [
        ny_dt(2026, 8, 23, 10, 3),   # Sunday
        ny_dt(2026, 9, 7, 10, 3),    # Labor Day
        ny_dt(2026, 11, 27, 13, 31), # Black Friday, after 13:00 close
    ]
    for captured_at in invalid_times:
        events = []
        assert run_collector(captured_at, events) == 0
        assert events == []


def test_duplicate_bucket_is_successful_and_preserves_first_observation():
    events = []
    assert run_collector(
        ny_dt(2026, 8, 24, 10, 3), events, save_result=False
    ) == 0
    assert len([event for event in events if event[0] == "save"]) == 1


def test_values_are_timestamped_in_bucket_at_actual_calculation_start():
    clock_values = iter([
        ny_dt(2026, 8, 24, 10, 28, 55),
        ny_dt(2026, 8, 24, 10, 31, 2),
    ])
    saved = {}

    class FakeClient:
        def __init__(self, token):
            pass

        def get_expirations(self, ticker):
            return ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-28"]

    def calculate(**kwargs):
        return completed_calculation(kwargs["captured_at"])

    def save(**kwargs):
        saved.update(kwargs)
        return True

    result = collector.capture_intraday_gamma_archive(
        environ={"TRADIER_TOKEN": "t", "DATABASE_URL": "db"},
        client_factory=FakeClient,
        rate_loader=lambda *args, **kwargs: {"rate": 0.04, "curve": None},
        calculation_func=calculate,
        save_func=save,
        clock=lambda: next(clock_values),
    )

    assert result == 0
    assert saved["captured_at"] == ny_dt(2026, 8, 24, 10, 31, 2)
    assert history.gamma_archive_bucket(saved["captured_at"]).bucket_start.isoformat() == "10:30:00"


def test_archive_failure_isolated_after_unchanged_calculation():
    events = []
    calculation = completed_calculation(ny_dt(2026, 8, 24, 10, 3))
    client, rate, calculate, _save = collector_dependencies(events, calculation=calculation)

    def failed_save(**kwargs):
        events.append(("failed_save", kwargs))
        raise RuntimeError("Neon unavailable")

    result = collector.capture_intraday_gamma_archive(
        captured_at=calculation.captured_at,
        environ={"TRADIER_TOKEN": "t", "DATABASE_URL": "db"},
        client_factory=client,
        rate_loader=rate,
        calculation_func=calculate,
        save_func=failed_save,
    )

    assert result == 1
    assert next(value for kind, value in events if kind == "calculation")["ticker"] == "SPY"
    assert calculation.levels["zero_gamma"] == 649.0


def test_empty_gamma_result_leaves_bucket_missing_without_write():
    events = []
    calculation = completed_calculation(ny_dt(2026, 8, 24, 10, 3))
    calculation.levels = None
    calculation.regime_info = None

    assert run_collector(calculation.captured_at, events, calculation=calculation) == 1
    assert not any(event[0] == "save" for event in events)


def test_canonical_expiration_universe_is_first_four_nonexpired_dates():
    available = [
        "2026-08-21",
        "2026-08-24",
        "2026-08-25",
        "2026-08-26",
        "2026-08-28",
        "2026-08-31",
    ]
    assert calculation_module.canonical_gamma_expirations(
        available, date(2026, 8, 24)
    ) == ("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-28")


def test_canonical_calculation_reuses_quote_and_four_chains(monkeypatch):
    calls = {"quote": 0, "chains": 0, "functions": []}

    class FakeClient:
        def __init__(self):
            self.chain_cache = {}

        def get_full_quote(self, ticker):
            calls["quote"] += 1
            return {"last": 651.0, "close": 650.5, "prevclose": 649.0}

        def get_chain_cached(self, ticker, expiration):
            key = (ticker, expiration)
            if key not in self.chain_cache:
                calls["chains"] += 1
                self.chain_cache[key] = {"status": "ok", "calls": [], "puts": []}
            return self.chain_cache[key]

        def prefetch_chains(self, ticker, expirations):
            for expiration in expirations:
                self.get_chain_cached(ticker, expiration)

    def fake_spot(**kwargs):
        calls["functions"].append("parity")
        kwargs["get_chain_cached_func"](kwargs["ticker"], kwargs["nearest_exp"])
        return {"spot": 651.25, "source": "implied", "market_open": True}

    def fake_calculate(client, ticker, target_exps, spot, **kwargs):
        calls["functions"].append("calculate_all")
        client.prefetch_chains(ticker, target_exps)
        return (
            pd.DataFrame({"strike": [650.0], "net_gex": [1.0]}),
            {"net_gex": 1.0, "coverage_ratio": 1.0},
            [(650.0, 1.0, 0.2, 1, 0.01, target_exps[0])],
            None,
            None,
        )

    monkeypatch.setattr(calculation_module, "get_reference_spot_details", fake_spot)
    monkeypatch.setattr(calculation_module.gex_engine, "calculate_all", fake_calculate)
    monkeypatch.setattr(
        calculation_module.gex_engine,
        "find_key_levels",
        lambda *args, **kwargs: calls["functions"].append("find_key_levels") or {
            "zero_gamma": 649.0,
            "call_wall": 655.0,
            "put_wall": 645.0,
            "net_gex": 1.0,
        },
    )
    monkeypatch.setattr(
        calculation_module.gex_engine,
        "get_gamma_regime_text",
        lambda *args: calls["functions"].append("regime") or {"regime": "Positive Gamma"},
    )
    monkeypatch.setattr(
        calculation_module,
        "build_staleness_info",
        lambda *args, **kwargs: calls["functions"].append("staleness") or {"freshness_score": 100},
    )
    monkeypatch.setattr(
        calculation_module,
        "build_run_confidence",
        lambda *args, **kwargs: calls["functions"].append("confidence") or {"score": 100},
    )
    monkeypatch.setattr(
        calculation_module,
        "build_expected_move_analysis",
        lambda **kwargs: calls["functions"].append("expected_move") or {
            "expected_move": {"expected_move_pts": 4.0}
        },
    )

    result = calculation_module.calculate_gamma_archive_observation(
        client=FakeClient(),
        ticker="SPY",
        available_expirations=[
            "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-28", "2026-08-31"
        ],
        captured_at=ny_dt(2026, 8, 24, 10, 3),
        risk_free_rate=0.04,
        risk_free_curve={30: 0.041, 91: 0.042},
    )

    assert result.target_expirations == (
        "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-28"
    )
    assert calls["quote"] == 1
    assert calls["chains"] == 4
    assert calls["functions"] == [
        "parity", "calculate_all", "staleness", "confidence",
        "find_key_levels", "regime", "expected_move",
    ]


def test_opening_and_intraday_paths_call_same_canonical_calculation():
    opening_source = (ROOT / "scheduled_snapshot.py").read_text(encoding="utf-8")
    intraday_source = (ROOT / "capture_gamma_archive.py").read_text(encoding="utf-8")

    assert "calculate_gamma_archive_observation(" in opening_source
    assert "calculation_func = calculation_func or calculate_gamma_archive_observation" in intraday_source
    assert "calculation = calculation_func(" in intraday_source
    assert "gex_engine.calculate_all" not in opening_source
    assert "canonical_gamma_expirations" not in opening_source


def test_intraday_module_has_no_forbidden_pipeline_or_trading_reachability():
    source = (ROOT / "capture_gamma_archive.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    assert not any(module.startswith("range_finder") for module in imports)
    assert not any("streamlit" in module for module in imports)
    assert calls.isdisjoint({
        "init_all_tables", "build_features", "fit_model", "forecast_next_week",
        "build_spread_plan", "log_spread_plan", "get_history", "save_gex_to_range_finder",
        "place_order", "submit_order", "execute_trade",
    })


def test_workflow_is_dispatch_only_one_job_and_has_no_matrix():
    workflow = (ROOT / ".github/workflows/gamma_archive.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "matrix:" not in workflow
    assert workflow.count("runs-on:") == 1
    assert "TICKER:" not in workflow
    assert "permissions:\n  contents: read" in workflow


def test_same_day_rate_cache_avoids_all_fred_calls(tmp_path, monkeypatch):
    cache_path = tmp_path / ".rate_cache.json"
    cache_path.write_text(json.dumps({
        "rate": 0.041,
        "source": "fred_dtb3",
        "label": "3M T-bill",
        "as_of": "2026-08-24",
        "curve": {"30": 0.040, "91": 0.041},
        "cached_at": datetime(2026, 8, 24, 14, 1, tzinfo=timezone.utc).isoformat(),
    }))
    monkeypatch.setattr(rates, "_RATE_CACHE_PATH", cache_path)
    monkeypatch.setattr(
        rates, "_fetch_fred", lambda _key: (_ for _ in ()).throw(AssertionError("FRED called"))
    )

    result = rates.fetch_risk_free_rate(
        "fred-key",
        prefer_same_day_cache=True,
        now=ny_dt(2026, 8, 24, 10, 30),
    )
    assert result["rate"] == 0.041
    assert result["curve"] == {"30": 0.040, "91": 0.041}


def test_first_intraday_rate_load_uses_existing_five_fred_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(rates, "_RATE_CACHE_PATH", tmp_path / ".rate_cache.json")
    request_count = {"n": 0}

    def scalar(_key):
        request_count["n"] += 1
        return {"rate": 0.041, "source": "fred_dtb3", "label": "3M", "as_of": "2026-08-24"}

    def curve_point(_key, series):
        request_count["n"] += 1
        return {"DGS1MO": 0.040, "DGS3MO": 0.041, "DGS6MO": 0.042, "DGS1": 0.043}[series]

    monkeypatch.setattr(rates, "_fetch_fred", scalar)
    monkeypatch.setattr(rates, "_fetch_fred_series_latest", curve_point)

    result = rates.fetch_risk_free_rate(
        "fred-key",
        prefer_same_day_cache=True,
        now=ny_dt(2026, 8, 24, 10, 0),
    )
    assert result["rate"] == 0.041
    assert request_count["n"] == 5


def test_archive_sql_remains_one_insert_without_select_or_ddl():
    sql = history._INSERT_SQL.upper()
    assert sql.lstrip().startswith("INSERT")
    assert "ON CONFLICT (TICKER, SESSION_DATE, BUCKET_START) DO NOTHING" in sql
    assert "SELECT" not in sql
    assert "CREATE" not in sql
    assert "ALTER" not in sql
