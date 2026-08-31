"""Postgres schema-initialization advisory lock behavior."""
import pytest

import range_finder.db as db


class _RawCursor:
    def __init__(self, events, *, acquire_error=None, unlock_result=True):
        self.events = events
        self.acquire_error = acquire_error
        self.unlock_result = unlock_result
        self._result = None

    def execute(self, sql, params=None):
        if "pg_advisory_unlock" in sql:
            self.events.append(("release", sql, params))
            self._result = (self.unlock_result,)
            return None
        if "pg_advisory_lock" in sql:
            self.events.append(("acquire", sql, params))
            if self.acquire_error is not None:
                raise self.acquire_error
            self._result = (None,)
            return None
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._result


class _Connection:
    def __init__(self, events, **cursor_kwargs):
        self.events = events
        self.cursor_kwargs = cursor_kwargs

    def cursor(self):
        return db.PGCursor(_RawCursor(self.events, **self.cursor_kwargs))


def test_advisory_lock_casts_float_adapted_key_to_bigint():
    events = []

    db._acquire_init_advisory_lock(_Connection(events))

    kind, sql, params = events[0]
    assert kind == "acquire"
    assert sql == "SELECT pg_advisory_lock(CAST(%s AS bigint))"
    # Preserve the global adapter behavior; the SQL's explicit bigint cast is
    # the narrow correction for this overload-sensitive Postgres function.
    assert params == (float(db._INIT_ADVISORY_LOCK_KEY),)


def test_init_acquires_before_body_and_releases_after(monkeypatch):
    events = []
    conn = _Connection(events)
    monkeypatch.setattr(db, "_init_all_tables_body", lambda _conn: events.append(("body",)))

    db.init_all_tables(conn)

    assert [event[0] for event in events] == ["acquire", "body", "release"]


def test_init_releases_lock_when_schema_body_fails(monkeypatch):
    events = []
    conn = _Connection(events)

    def fail_body(_conn):
        events.append(("body",))
        raise ValueError("DDL failed")

    monkeypatch.setattr(db, "_init_all_tables_body", fail_body)

    with pytest.raises(ValueError, match="DDL failed"):
        db.init_all_tables(conn)

    assert [event[0] for event in events] == ["acquire", "body", "release"]


def test_init_does_not_swallow_lock_acquisition_failure(monkeypatch):
    events = []
    conn = _Connection(events, acquire_error=RuntimeError("no bigint overload"))
    body_called = False

    def body(_conn):
        nonlocal body_called
        body_called = True

    monkeypatch.setattr(db, "_init_all_tables_body", body)

    with pytest.raises(RuntimeError, match="Failed to acquire Postgres schema init lock"):
        db.init_all_tables(conn)

    assert body_called is False
    assert [event[0] for event in events] == ["acquire"]


def test_init_does_not_swallow_unlock_failure(monkeypatch):
    events = []
    conn = _Connection(events, unlock_result=False)
    monkeypatch.setattr(db, "_init_all_tables_body", lambda _conn: events.append(("body",)))

    with pytest.raises(RuntimeError, match="Failed to release Postgres schema init lock"):
        db.init_all_tables(conn)

    assert [event[0] for event in events] == ["acquire", "body", "release"]
