from __future__ import annotations

from datetime import datetime, timezone
import logging
from zoneinfo import ZoneInfo

import pytest

import phase1.gamma_level_history as history


NY = ZoneInfo("America/New_York")


class FakeStore:
    def __init__(self):
        self.rows = {}
        self.statements = []
        self.connections_opened = 0
        self.connections_closed = 0

    def connect(self):
        self.connections_opened += 1
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, store):
        self.store = store

    def cursor(self):
        return FakeCursor(self.store)

    def close(self):
        self.store.connections_closed += 1


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.rowcount = -1

    def execute(self, sql, params=None):
        self.store.statements.append((sql, params))
        if sql.lstrip().upper().startswith("INSERT"):
            key = (params[3], params[1], params[2])
            if key in self.store.rows:
                self.rowcount = 0
            else:
                self.store.rows[key] = params
                self.rowcount = 1

    def close(self):
        pass


@pytest.fixture(autouse=True)
def clear_process_bucket_cache():
    history._completed_bucket_keys.clear()
    yield
    history._completed_bucket_keys.clear()


def ny_dt(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=NY)


def sample_kwargs(captured_at, ticker="SPX"):
    return {
        "captured_at": captured_at,
        "ticker": ticker,
        "spot": 6600.25,
        "zero_gamma": 6585.0,
        "call_wall": 6650.0,
        "put_wall": 6500.0,
        "gamma_regime": "Positive Gamma",
        "net_gex": 1_250_000.0,
        "call_wall_gex": 500_000.0,
        "put_wall_gex": -350_000.0,
        "zero_gamma_is_true_crossing": True,
        "zero_gamma_method": "crossing_fine",
        "expected_move_pts": 42.5,
        "confidence_score": 91.0,
        "freshness_score": 95.0,
        "coverage_ratio": 0.98,
    }


@pytest.mark.parametrize("minute", [31, 44])
def test_opening_times_map_to_0930_bucket(minute):
    bucket = history.gamma_archive_bucket(ny_dt(2026, 8, 24, 9, minute))
    assert bucket.session_date.isoformat() == "2026-08-24"
    assert bucket.bucket_start.isoformat() == "09:30:00"


@pytest.mark.parametrize("minute", [1, 29])
def test_1001_and_1029_map_to_1000_bucket(minute):
    bucket = history.gamma_archive_bucket(ny_dt(2026, 8, 24, 10, minute))
    assert bucket.bucket_start.isoformat() == "10:00:00"


def test_1031_maps_to_1030_bucket():
    bucket = history.gamma_archive_bucket(ny_dt(2026, 8, 24, 10, 31))
    assert bucket.bucket_start.isoformat() == "10:30:00"


def test_duplicate_same_bucket_is_suppressed_without_second_db_statement():
    store = FakeStore()
    first = history.save_gamma_level_snapshot(
        **sample_kwargs(ny_dt(2026, 8, 24, 10, 1)),
        connection_factory=store.connect,
    )
    second = history.save_gamma_level_snapshot(
        **sample_kwargs(ny_dt(2026, 8, 24, 10, 18)),
        connection_factory=store.connect,
    )

    assert first is True
    assert second is False
    assert len(store.rows) == 1
    assert len(store.statements) == 1
    assert store.connections_opened == store.connections_closed == 1


def test_cross_process_duplicate_executes_one_conflict_insert_but_adds_no_row():
    store = FakeStore()
    kwargs = sample_kwargs(ny_dt(2026, 8, 24, 10, 1))
    assert history.save_gamma_level_snapshot(**kwargs, connection_factory=store.connect)

    # Simulate another worker/process whose local completed-bucket cache is empty.
    history._completed_bucket_keys.clear()
    assert not history.save_gamma_level_snapshot(**kwargs, connection_factory=store.connect)

    assert len(store.rows) == 1
    assert len(store.statements) == 2
    assert all("ON CONFLICT" in sql.upper() for sql, _ in store.statements)


def test_tickers_do_not_collide():
    store = FakeStore()
    captured_at = ny_dt(2026, 8, 24, 10, 1)
    for ticker in ("SPX", "SPY", "XSP"):
        assert history.save_gamma_level_snapshot(
            **sample_kwargs(captured_at, ticker=ticker),
            connection_factory=store.connect,
        )

    assert len(store.rows) == 3
    assert {key[0] for key in store.rows} == {"SPX", "SPY", "XSP"}


def test_trading_dates_do_not_collide():
    store = FakeStore()
    for day in (24, 25):
        assert history.save_gamma_level_snapshot(
            **sample_kwargs(ny_dt(2026, 8, day, 10, 1)),
            connection_factory=store.connect,
        )

    assert len(store.rows) == 2
    assert {key[1].isoformat() for key in store.rows} == {"2026-08-24", "2026-08-25"}


def test_utc_date_boundary_does_not_create_false_new_york_session():
    # Monday in UTC is still Sunday evening in New York. UTC calendar dates
    # must never make this look like a Monday RTH observation.
    captured_at = datetime(2026, 6, 15, 0, 30, tzinfo=timezone.utc)
    assert history.gamma_archive_bucket(captured_at) is None


@pytest.mark.parametrize(
    ("captured_at", "expected_date"),
    [
        (datetime(2026, 3, 9, 13, 31, tzinfo=timezone.utc), "2026-03-09"),  # EDT
        (datetime(2026, 11, 2, 14, 31, tzinfo=timezone.utc), "2026-11-02"),  # EST
    ],
)
def test_dst_maps_utc_open_to_correct_new_york_bucket(captured_at, expected_date):
    bucket = history.gamma_archive_bucket(captured_at)
    assert bucket.session_date.isoformat() == expected_date
    assert bucket.bucket_start.isoformat() == "09:30:00"


def test_weekend_holiday_and_after_hours_are_not_archived():
    assert history.gamma_archive_bucket(ny_dt(2026, 8, 23, 10, 0)) is None  # Sunday
    assert history.gamma_archive_bucket(ny_dt(2026, 9, 7, 10, 0)) is None   # Labor Day
    assert history.gamma_archive_bucket(ny_dt(2026, 8, 24, 16, 0)) is None


def test_archival_failure_is_logged_and_cannot_escape(monkeypatch, caplog):
    def fail_save(**_kwargs):
        raise RuntimeError("Neon unavailable")

    monkeypatch.setattr(history, "save_gamma_level_snapshot", fail_save)
    caplog.set_level(logging.WARNING)

    result = history.archive_computed_gamma_levels(
        captured_at=ny_dt(2026, 8, 24, 10, 1),
        ticker="SPX",
        spot=6600.25,
        levels={"zero_gamma": 6585.0},
        regime_info={"regime": "Positive Gamma"},
        stats={"net_gex": 1_250_000.0},
        confidence_info={"score": 91.0},
        staleness_info={"freshness_score": 95.0},
        em_analysis={},
    )

    assert result is False
    assert "Gamma archive save failed: Neon unavailable" in caplog.text


def test_invalid_observation_timestamp_cannot_escape(caplog):
    caplog.set_level(logging.WARNING)

    result = history.archive_computed_gamma_levels(
        captured_at="not-a-timestamp",
        ticker="SPX",
        spot=6600.25,
        levels={"zero_gamma": 6585.0},
        regime_info={"regime": "Positive Gamma"},
        stats={"net_gex": 1_250_000.0},
        confidence_info={"score": 91.0},
        staleness_info={"freshness_score": 95.0},
        em_analysis={},
    )

    assert result is False
    assert "Gamma archive save failed:" in caplog.text


def test_raw_payload_keys_are_not_persisted():
    store = FakeStore()
    raw_chain = object()
    raw_frame = object()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(history, "_get_connection", store.connect)
        assert history.archive_computed_gamma_levels(
            captured_at=ny_dt(2026, 8, 24, 10, 1),
            ticker="SPX",
            spot=6600.25,
            levels={
                "zero_gamma": 6585.0,
                "call_wall": 6650.0,
                "put_wall": 6500.0,
                "raw_chain": raw_chain,
            },
            regime_info={"regime": "Positive Gamma"},
            stats={"net_gex": 1_250_000.0, "gex_df": raw_frame},
            confidence_info={"score": 91.0},
            staleness_info={"freshness_score": 95.0},
            em_analysis={},
        )

    sql, params = store.statements[0]
    assert "chain" not in sql.lower()
    assert "dataframe" not in sql.lower()
    assert raw_chain not in params
    assert raw_frame not in params
    assert len(params) == 18


def test_schema_init_is_one_ddl_and_save_is_one_insert_without_select():
    schema_store = FakeStore()
    history.init_gamma_level_history_schema(schema_store.connect())
    assert len(schema_store.statements) == 1
    assert schema_store.statements[0][0].lstrip().upper().startswith("CREATE TABLE")

    save_store = FakeStore()
    assert history.save_gamma_level_snapshot(
        **sample_kwargs(ny_dt(2026, 8, 24, 10, 1)),
        connection_factory=save_store.connect,
    )
    assert len(save_store.statements) == 1
    sql = save_store.statements[0][0].lstrip().upper()
    assert sql.startswith("INSERT")
    assert "SELECT" not in sql
    assert "CREATE" not in sql
    assert "ALTER" not in sql
