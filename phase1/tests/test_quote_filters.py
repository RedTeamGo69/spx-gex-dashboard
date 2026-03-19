from phase1.quote_filters import (
    has_two_sided_quote,
    is_crossed_or_locked,
    is_crossed,
    is_locked,
    quote_spread,
    quote_mid,
    usable_for_parity,
)


def test_good_quote_is_usable():
    row = {"bid": 10.0, "ask": 10.5}
    assert has_two_sided_quote(row) is True
    assert is_crossed_or_locked(row) is False
    assert quote_spread(row) == 0.5
    assert quote_mid(row) == 10.25
    assert usable_for_parity(row, max_spread=2.0) is True


def test_crossed_quote_is_not_usable():
    row = {"bid": 10.5, "ask": 10.0}
    assert has_two_sided_quote(row) is True
    assert is_crossed(row) is True
    assert is_locked(row) is False
    assert is_crossed_or_locked(row) is True
    assert quote_mid(row) is None
    assert usable_for_parity(row, max_spread=2.0) is False


def test_locked_quote_is_usable():
    """Locked quotes (bid == ask) are tight markets and should be usable."""
    row = {"bid": 10.0, "ask": 10.0}
    assert has_two_sided_quote(row) is True
    assert is_crossed(row) is False
    assert is_locked(row) is True
    assert is_crossed_or_locked(row) is True  # backward compat still True
    assert quote_spread(row) == 0.0
    assert quote_mid(row) == 10.0
    assert usable_for_parity(row, max_spread=2.0) is True


def test_wide_spread_quote_is_not_usable():
    row = {"bid": 10.0, "ask": 13.5}
    assert usable_for_parity(row, max_spread=2.0) is False
