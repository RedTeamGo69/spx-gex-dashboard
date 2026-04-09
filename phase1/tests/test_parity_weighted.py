from phase1.parity import weighted_median, parity_candidate_weight


def test_weighted_median_basic_case():
    values = [10, 20, 30]
    weights = [1, 5, 1]
    result = weighted_median(values, weights)
    assert result == 20


def test_weighted_median_falls_to_middle_when_equal():
    values = [10, 20, 30]
    weights = [1, 1, 1]
    result = weighted_median(values, weights)
    assert result == 20


def test_parity_candidate_weight_prefers_tighter_spread():
    w1 = parity_candidate_weight(strike=5000, vendor_spot=5000, combined_spread=0.5)
    w2 = parity_candidate_weight(strike=5000, vendor_spot=5000, combined_spread=2.0)
    assert w1 > w2


def test_parity_candidate_weight_prefers_nearer_strike():
    w1 = parity_candidate_weight(strike=5000, vendor_spot=5000, combined_spread=1.0)
    w2 = parity_candidate_weight(strike=5050, vendor_spot=5000, combined_spread=1.0)
    assert w1 > w2


def test_weighted_median_interpolates_at_exact_boundary():
    """When cutoff falls exactly on a cumulative boundary, average the two adjacent values."""
    values = [10, 20, 30, 40]
    weights = [1, 1, 1, 1]
    # cum_w = [1, 2, 3, 4], cutoff = 2.0
    # cutoff == cum_w[1] exactly, so interpolate: (20 + 30) / 2 = 25
    result = weighted_median(values, weights)
    assert result == 25.0


def test_weighted_median_single_value():
    assert weighted_median([42], [1]) == 42.0


def test_weighted_median_empty():
    assert weighted_median([], []) is None
