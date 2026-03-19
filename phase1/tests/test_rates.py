from phase1.rates import parse_fred_rate_response, parse_treasury_rate_response


def test_parse_fred_rate_response_returns_first_valid_observation():
    payload = {
        "observations": [
            {"date": "2026-03-15", "value": "."},
            {"date": "2026-03-14", "value": "4.31"},
        ]
    }

    result = parse_fred_rate_response(payload)

    assert result is not None
    assert abs(result["rate"] - 0.0431) < 1e-12
    assert result["source"] == "fred_dtb3"
    assert result["as_of"] == "2026-03-14"


def test_parse_fred_rate_response_returns_none_if_no_valid_values():
    payload = {
        "observations": [
            {"date": "2026-03-15", "value": "."},
            {"date": "2026-03-14", "value": "."},
        ]
    }

    result = parse_fred_rate_response(payload)
    assert result is None


def test_parse_treasury_rate_response_returns_first_record():
    payload = {
        "data": [
            {"record_date": "2026-02-28", "avg_interest_rate_amt": "4.18"}
        ]
    }

    result = parse_treasury_rate_response(payload)

    assert result is not None
    assert abs(result["rate"] - 0.0418) < 1e-12
    assert result["source"] == "treasury_monthly_average"
    assert result["as_of"] == "2026-02-28"


def test_parse_treasury_rate_response_returns_none_if_no_data():
    payload = {"data": []}
    result = parse_treasury_rate_response(payload)
    assert result is None
