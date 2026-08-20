from src.validation.validate_schema import validate_schema

from .helpers import valid_match


def test_schema_validation_counts_fields_across_multiple_participants() -> None:
    report, issues = validate_schema([valid_match(), valid_match("ME1_TEST_2")])
    assert "placement" in report["sections"]["participant"]["observed_fields"]
    assert not any(issue.reason_code == "schema_required_field_absent" for issue in issues)
