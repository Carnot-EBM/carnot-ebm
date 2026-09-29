"""REQ-REPORT-7852: legacy identity and public-byte boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import source_identity_7852 as identity
from carnot.verify import source_projection


def test_legacy_identity_is_hash_bound() -> None:
    """SCENARIO-REPORT-7852-IDENTITY: only the archived alias resolves."""
    artifact = {
        "experiment_id": identity.LEGACY_SLUG,
        "milestone": identity.LEGACY_MILESTONE,
        "run_date": "20260928",
    }
    assert identity.resolve(identity.LEGACY_PATH, identity.LEGACY_HASH, artifact) == 7810
    assert (
        identity.resolve(
            identity.LEGACY_PATH,
            identity.LEGACY_HASH,
            {**artifact, "task_id": identity.LEGACY_SLUG},
        )
        == 7810
    )
    cases = [
        (identity.LEGACY_PATH.with_name("other.json"), identity.LEGACY_HASH, artifact),
        (identity.LEGACY_PATH, "sha256:" + "0" * 64, artifact),
        (
            identity.LEGACY_PATH,
            identity.LEGACY_HASH,
            {**artifact, "experiment_id": "exp7810-other"},
        ),
        (identity.LEGACY_PATH, identity.LEGACY_HASH, {**artifact, "milestone": "2026.09.680"}),
        (identity.LEGACY_PATH, identity.LEGACY_HASH, {**artifact, "run_date": "20260929"}),
        (identity.LEGACY_PATH, identity.LEGACY_HASH, {**artifact, "task_id": "exp7810-other"}),
    ]
    for path, digest, candidate in cases:
        with pytest.raises(ValueError):
            identity.resolve(path, digest, candidate)


def test_real_cli_public_success_and_gate_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7852-TERMINAL: both child routes stay private."""
    row = {
        "family_id": "opaque-1",
        "source_bytes": b"Alpha. Beta.".hex(),
        "answer_bytes": b"Alpha.".hex(),
    }
    public = tmp_path / "public.jsonl"
    output = tmp_path / "features.jsonl"
    public.write_text(json.dumps(row) + "\n")
    base = [
        sys.executable,
        "-u",
        "scripts/experiments/experiment_7852_v682_source_boundary.py",
        "--fixture-public",
        str(public),
        "--fixture-output",
        str(output),
    ]
    assert subprocess.run(base, capture_output=True, check=False).returncode == 0
    source_projection.replay_file(public, output)
    public.write_text(json.dumps({**row, "label": 1}) + "\n")
    assert subprocess.run(base, capture_output=True, check=False).returncode != 0


def test_projection_mutations_and_unknown_mask() -> None:
    """SCENARIO-REPORT-7852-PROJECTION: metadata cannot change tensors."""
    row = {
        "family_id": "opaque-1",
        "source_bytes": b"Alpha. Beta. Gamma.".hex(),
        "answer_bytes": b"Alpha.".hex(),
    }
    original = source_projection.extract_row(row)
    assert original["feature_dim"] == 132
    assert (original["view_a_windows"], original["view_b_windows"]) == (4, 5)
    for key, value in (
        ("label", 999),
        ("role", "other"),
        ("confidence", -1),
        ("annotations", [{"start": 99}]),
        ("generator_identity", "other"),
    ):
        assert (
            source_projection.extract_row(source_projection.public_row({**row, key: value}))
            == original
        )
    changed = source_projection.extract_row({**row, "family_id": "opaque-2"})
    assert changed["feature_hash"] == original["feature_hash"]
    assert changed["views"] == original["views"]
    loss, gradient = source_projection.masked_loss([0.2, -0.4], [1, -1], [1, 0])
    assert (loss, gradient) == source_projection.masked_loss([0.2, -0.4], [1, 999], [1, 0])
    assert gradient[1] == 0
    assert (
        source_projection.masked_loss([0.2], [1], [1])[0]
        != source_projection.masked_loss([0.2], [0], [1])[0]
    )
