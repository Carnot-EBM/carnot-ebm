"""SCENARIO-REPORT-8042-COUNTS: structural counts cannot become copied metrics."""

import pytest

from scripts import adversarial_verify as verifier


def test_large_completed_integer_counts() -> None:
    """REQ-REPORT-8042: completing every eligible readout is a legitimate identity."""
    flags = []
    verifier.check_tautology(dict(eligible_count=27043, completed_count=27043), flags)
    assert not flags


@pytest.mark.parametrize(
    "values",
    [
        dict(eligible_count=27043.0, completed_count=27043.0),
        dict(first_loss=27043.0, second_loss=27043.0),
        dict(first_loss=27043, second_loss=27043),
    ],
)
def test_distinct_measurement_equality_still_critical(values: dict) -> None:
    """SCENARIO-REPORT-8042-COUNTS: names or integral values cannot hide measured duplication."""
    flags = []
    verifier.check_tautology(values, flags)
    assert any(f.kind == "TAUTOLOGY" and f.severity == "critical" for f in flags)
