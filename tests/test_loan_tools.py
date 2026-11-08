from loan_tools import ESCALATION_THRESHOLD, check_loan_application_status


def test_known_application_returns_required_fields():
    result = check_loan_application_status("CRD-0001")
    assert result["found"] is True
    assert "status" in result
    assert "loan_amount_inr" in result
    assert 0 <= result["escalation_score"] <= 1


def test_lookup_is_case_insensitive():
    result = check_loan_application_status(" crd-0001 ")
    assert result["found"] is True
    assert result["record_id"] == "CRD-0001"


def test_unknown_application_is_handled():
    result = check_loan_application_status("CRD-9999")
    assert result["found"] is False


def test_threshold_is_in_valid_range():
    assert 0 <= ESCALATION_THRESHOLD <= 1
