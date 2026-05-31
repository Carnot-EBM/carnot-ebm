"""Tests for experiment_3510 (post-.323 G2 drift check).

Spec: REQ-PUBLISH-039C, SCENARIO-PUBLISH-039C

These tests cover the new code added for Exp 3510:
- v3 experiment identifiers (EXP_ID=3510, schema v3, random_seed=20260531)
- v3 verdict string acknowledges "current" refresh at .323
- g2_met always False (Operator-Only External Publication)
All subprocess calls are stubbed; no shell-out or network use.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.eval.fover_g2_package import PACKAGE_NAME

# A faithful slice of the harness stdout the regression run parses.
GREEN_STDOUT = (
    "condition A (production)        mean AUROC: 0.9131\n"
    "condition B (architecture-only) mean AUROC: 0.8947\n"
    "learning contribution:                      0.0185\n"
    "reproducibility_checksum:                   abc123def456\n"
    "condition A in CI [0.9027, 0.9235]: True\n"
    "learning_contribution in CI [0.0125, 0.0245]: True\n"
    "RESULT: PASS — FoVer headline reproduces within published CI\n"
)

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "experiment_3510_fover_g2_regression_verify_external_ask_refresh_v3.py"
)


def _make_fake_tarball(tmp_path: Path) -> Path:
    pkg = tmp_path / PACKAGE_NAME
    (pkg / "scripts").mkdir(parents=True)
    (pkg / "python").mkdir(parents=True)
    (pkg / "scripts" / "reproduce_fover_headline.py").write_text(
        "print('hi')\n", encoding="utf-8"
    )
    (pkg / "requirements.txt").write_text("numpy==2.0\n", encoding="utf-8")
    (pkg / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    tar_path = tmp_path / "g2-fover-repro.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg, arcname=PACKAGE_NAME)
    return tar_path


def _build_fake_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo layout with the tarball, exp3476 sha, runbook, and data."""
    (tmp_path / "dist").mkdir()
    tar_path = _make_fake_tarball(tmp_path / "dist")
    tar_path.rename(tmp_path / "dist" / "g2-fover-repro.tar.gz")
    tar_path = tmp_path / "dist" / "g2-fover-repro.tar.gz"

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    computed_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (results_dir / "experiment_3476_fover_g2_self_contained_external_package_v1.json").write_text(
        json.dumps({"package_sha256": computed_sha}), encoding="utf-8"
    )

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "reproduction-runbook-fover-headline.md").write_text(
        "# Runbook\n", encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text(
        "print('hi')\n", encoding="utf-8"
    )
    return tmp_path


def _run_v3_main(tmp_path: Path) -> tuple[int, dict]:
    """Load experiment_3510 and run main() with stubbed subprocess + patched REPO_ROOT."""
    import carnot.eval.fover_g2_regression as reg
    import carnot.eval.fover_g2_package as pkg_mod

    out_path = (
        tmp_path
        / "results"
        / "experiment_3510_fover_g2_regression_verify_external_ask_refresh_v3.json"
    )

    spec = importlib.util.spec_from_file_location("exp3510", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)

    _orig_check = reg.check_preconditions
    _orig_read = reg.read_recorded_sha256

    def fake_check(_repo_root):
        return _orig_check(tmp_path)

    def fake_read_sha(_repo_root):
        return _orig_read(tmp_path)

    with (
        patch.object(reg, "check_preconditions", side_effect=fake_check),
        patch.object(reg, "read_recorded_sha256", side_effect=fake_read_sha),
        patch.object(
            reg,
            "run_package_in_isolated_dir",
            return_value={
                "method": "isolated_dir",
                "exit_code": 0,
                "stdout": GREEN_STDOUT,
                "stderr": "",
            },
        ),
        patch.object(
            pkg_mod,
            "maybe_ipfs_add",
            return_value={"ipfs_available": False, "package_cid": None},
        ),
    ):
        spec.loader.exec_module(mod)
        mod.REPO_ROOT = tmp_path
        mod.OUT_PATH = out_path
        exit_code = mod.main()

    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    return exit_code, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_experiment_3510_main_returns_zero(tmp_path: Path):
    """REQ-PUBLISH-039C: v3 main() exits 0 on a green regression run."""
    _build_fake_repo(tmp_path)
    exit_code, _ = _run_v3_main(tmp_path)
    assert exit_code == 0


def test_experiment_3510_artifact_identifiers(tmp_path: Path):
    """REQ-PUBLISH-039C: artifact carries v3-specific experiment id, schema, and artifact name."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert data["experiment"] == 3510
    assert data["artifact"] == "experiment_3510_fover_g2_regression_verify_external_ask_refresh_v3"
    assert data["schema"] == "carnot.fover_g2_regression_verify_external_ask_refresh_v3"


def test_experiment_3510_random_seed_is_run_date_not_exp_number(tmp_path: Path):
    """REQ-PUBLISH-039C: random_seed is 20260531 (run-date seed), NOT the experiment number 3510."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert data["random_seed"] == 20260531, (
        f"random_seed must be 20260531 (run-date), got {data['random_seed']}"
    )


def test_experiment_3510_g2_never_self_marked(tmp_path: Path):
    """SCENARIO-PUBLISH-039C: g2_met must always be False (Operator-Only External Publication)."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert data["g2_met"] is False
    assert data["external_run_pending"] is True


def test_experiment_3510_verdict_terminal_prefix(tmp_path: Path):
    """SCENARIO-PUBLISH-039C: honest_verdict starts with 'complete:' per Verdict Terminal-Prefix."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert data["honest_verdict"].startswith("complete:"), (
        f"verdict must start with 'complete:', got: {data['honest_verdict']!r}"
    )


def test_experiment_3510_verdict_uses_current_framing(tmp_path: Path):
    """SCENARIO-PUBLISH-039C: clean verdict says 'external_ask_current' (v3 framing, not 'ready')."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert "external_ask_current" in data["honest_verdict"], (
        f"v3 verdict must use 'current' framing, got: {data['honest_verdict']!r}"
    )


def test_experiment_3510_required_schema_fields_present(tmp_path: Path):
    """REQ-PUBLISH-039C: all REQUIRED ARTIFACT FIELDS from the task spec are present."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    required = [
        "honest_verdict",
        "inference_substrate",
        "package_reproduced_auroc",
        "package_auroc_within_ci",
        "package_sha256_verified",
        "package_cid",
        "external_ask_workflow_path",
        "operator_checklist_path",
        "g2_met",
        "external_run_pending",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    missing = [f for f in required if f not in data]
    assert not missing, f"Missing required fields: {missing}"


def test_experiment_3510_inference_substrate(tmp_path: Path):
    """REQ-PUBLISH-039C: inference_substrate declares verifier_ensemble_against_cached_candidates."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert data["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_experiment_3510_duration_s_present_and_numeric(tmp_path: Path):
    """REQ-PUBLISH-039C: duration_s is present and non-negative (adversarial_verify.py checks 1s floor on live artifacts)."""
    _build_fake_repo(tmp_path)
    _, data = _run_v3_main(tmp_path)
    assert "duration_s" in data
    assert isinstance(data["duration_s"], float)
    assert data["duration_s"] >= 0.0
