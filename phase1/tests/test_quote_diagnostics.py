from phase1.quote_filters import summarize_quote_quality


def test_summarize_quote_quality_counts_correctly():
    rows = [
        {"bid": 1.0, "ask": 1.2},   # usable
        {"bid": 0.0, "ask": 1.2},   # no two-sided quote
        {"bid": 2.0, "ask": 1.9},   # crossed (bid > ask)
        {"bid": 1.0, "ask": 4.5},   # wide spread
    ]

    summary = summarize_quote_quality(rows, max_spread=2.0)

    assert summary["total"] == 4
    assert summary["usable"] == 1
    assert summary["no_two_sided_quote"] == 1
    assert summary["crossed"] == 1
    assert summary["locked"] == 0
    assert summary["crossed_or_locked"] == 1  # backward compat aggregate
    assert summary["wide_spread"] == 1


def test_summarize_quote_quality_locked_counted_separately():
    rows = [
        {"bid": 5.0, "ask": 5.0},   # locked (bid == ask)
        {"bid": 3.0, "ask": 2.5},   # crossed (bid > ask)
    ]

    summary = summarize_quote_quality(rows, max_spread=2.0)

    assert summary["total"] == 2
    assert summary["locked"] == 1
    assert summary["crossed"] == 1
    assert summary["crossed_or_locked"] == 2
