"""REQ-REPORT-7838 public source boundary and evaluator masking."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import source_projection as projection


def sample() -> dict:
    """Provide a complete two-sentence source with one answer unit."""
    return {
        "family_id": "family-a",
        "source_bytes": b"First fact. Second fact.".hex(),
        "answer_bytes": b"First fact.".hex(),
    }


def test_public_projection_and_metadata_isolation() -> None:
    """SCENARIO-REPORT-7838-ISOLATION: metadata cannot reach scoring."""
    row = sample()
    row.update(role="fit", label=1, confidence=0.1, annotations=[{"start": 0}])
    public = projection.public_row(row)
    assert set(public) == {"family_id", "source_bytes", "answer_bytes"}
    feature = projection.extract_row(public)
    assert feature["feature_dim"] == 132
    assert feature["view_a_windows"] == 2
    assert feature["view_b_windows"] == 3
    assert projection.extract_row(projection.public_row({**row, "label": 0})) == feature
    assert (
        projection.extract_row({**public, "family_id": "other"})["feature_hash"]
        == feature["feature_hash"]
    )
    with pytest.raises(ValueError):
        projection.extract_row({**public, "role": "fit"})
    with pytest.raises(ValueError):
        projection.extract_row({**public, "source_bytes": "broken"})


def test_offsets_and_roster_fail_closed() -> None:
    """SCENARIO-REPORT-7838-ISOLATION: reject drift, duplication, loss."""
    row = sample()
    public = projection.public_row(row)
    feature = projection.extract_row(public)
    projection.validate_features([public], [feature], ["family-a"])
    for changed in ({**feature, "feature_hash": "bad"}, {**feature, "role": "fit"}):
        with pytest.raises(ValueError):
            projection.validate_features([public], [changed], ["family-a"])
    with pytest.raises(ValueError):
        projection.validate_features([public, public], [feature, feature], ["family-a"])
    with pytest.raises(ValueError):
        projection.validate_features([], [], ["family-a"])
    bad = deepcopy(public)
    bad["source_offsets"] = [[1, 2]]
    with pytest.raises(ValueError):
        projection.extract_row(bad)


def test_unknown_mask_and_known_sensitivity() -> None:
    """SCENARIO-REPORT-7838-MASK: unknown sentinels have zero effect."""
    loss, gradient = projection.masked_loss([0.2, -0.4], [1, -1], [1, 0])
    changed, changed_gradient = projection.masked_loss([0.2, -0.4], [1, 999], [1, 0])
    assert (loss, gradient) == (changed, changed_gradient)
    assert gradient[1] == 0
    assert projection.masked_loss([0.2], [0], [1])[0] != projection.masked_loss([0.2], [1], [1])[0]
    assert projection.masked_loss([0.2], [1], [0]) == (0.0, [0.0])
    with pytest.raises(ValueError):
        projection.masked_loss([0.2], [1, 0], [1])


def test_real_child_extract_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7838-TERMINAL: child reads only public rows."""
    public = tmp_path / "public.jsonl"
    output = tmp_path / "features.jsonl"
    public.write_text(json.dumps(projection.public_row(sample())) + "\n")
    command = [
        sys.executable,
        "-m",
        "carnot.verify.source_projection",
        "extract",
        str(public),
        str(output),
    ]
    assert subprocess.run(command, check=False, capture_output=True).returncode == 0
    assert (
        subprocess.run([*command[:3], "replay", str(public), str(output)], check=False).returncode
        == 0
    )
    changed = json.loads(output.read_text())
    changed["feature_hash"] = "tampered"
    output.write_text(json.dumps(changed) + "\n")
    assert (
        subprocess.run([*command[:3], "replay", str(public), str(output)], check=False).returncode
        != 0
    )
