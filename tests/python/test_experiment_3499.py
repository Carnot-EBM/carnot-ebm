"""Tests for experiment_3499 (post-.322 G2 drift check) + parametric append_runbook.

Spec: REQ-PUBLISH-039C, SCENARIO-PUBLISH-039C

These tests cover the new code added for Exp 3499:
- ``append_runbook`` parametric exp_id / run_date / artifact_name
- the v2 experiment script's _write / main (via import) with a stub runner
All subprocess calls are stubbed; no shell-out or network use.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.eval.fover_g2_package import PACKAGE_NAME
from carnot.eval.fover_g2_regression import append_runbook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


GREEN_STDOUT = (
    "condition A (production)        mean AUROC: 0.9131\n"
    "condition B (architecture-only) mean AUROC: 0.8947\n"
    "learning contribution:                      0.0185\n"
    "reproducibility_checksum:                   abc123def456\n"
    "condition A in CI [0.9027, 0.9235]: True\n"
    "learning_contribution in CI [0.0125, 0.0245]: True\n"
    "RESULT: PASS — FoVer headline reproduces within published CI\n"
)

# ---------------------------------------------------------------------------
# Tests for parametric append_runbook (the new code)
# ---------------------------------------------------------------------------


def test_append_runbook_parametric_exp_id(tmp_path: Path):
    """append_runbook with exp_id=exp3499 writes exp3499 in the runbook section."""
    import carnot.eval.fover_g2_regression as reg
    rb = tmp_path / reg.RUNBOOK_REL
    rb.parent.mkdir(parents=True)
    rb.write_text("# existing runbook\n")
    ok = append_runbook(
        tmp_path,
        reproduced_auroc=0.9131,
        auroc_within_ci=True,
        clean_env_method="isolated_dir",
        package_sha256="abc",
        package_sha256_verified=True,
        package_cid="QmX",
        exp_id="exp3499",
        run_date="2026-05-31",
        artifact_name="experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2",
    )
    assert ok is True
    text = rb.read_text()
    assert "# existing runbook" in text  # never deletes
    assert "exp3499" in text
    assert "2026-05-31" in text
    assert "experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2" in text
    assert "0.9131" in text


def test_append_runbook_default_still_produces_exp3488(tmp_path: Path):
    """Backward compat: default exp_id keeps the exp3488 content (existing test pattern)."""
    import carnot.eval.fover_g2_regression as reg
    rb = tmp_path / reg.RUNBOOK_REL
    rb.parent.mkdir(parents=True)
    rb.write_text("# existing runbook\n")
    ok = append_runbook(
        tmp_path,
        reproduced_auroc=0.9131,
        auroc_within_ci=True,
        clean_env_method="isolated_dir",
        package_sha256="abc",
        package_sha256_verified=True,
        package_cid=None,
    )
    assert ok is True
    text = rb.read_text()
    assert "exp3488" in text  # default preserved for backward compat


def test_append_runbook_parametric_skips_when_no_runbook(tmp_path: Path):
    """When runbook is absent, returns False regardless of parametric args."""
    ok = append_runbook(
        tmp_path,
        reproduced_auroc=0.9131,
        auroc_within_ci=True,
        clean_env_method="isolated_dir",
        package_sha256="abc",
        package_sha256_verified=True,
        package_cid=None,
        exp_id="exp3499",
        run_date="2026-05-31",
        artifact_name="experiment_3499_dummy",
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Smoke test for the v2 script's main() — SCENARIO-PUBLISH-039C
# ---------------------------------------------------------------------------


def test_experiment_3499_main_green(tmp_path: Path):
    """SCENARIO-PUBLISH-039C: v2 main() produces the artifact with correct schema fields."""
    import sys
    import importlib.util

    # Fake repo layout: tarball + exp3476 artifact (sha) + runbook + data files
    (tmp_path / "dist").mkdir()
    tar_path = _make_fake_tarball(tmp_path / "dist")
    tar_path.rename(tmp_path / "dist" / "g2-fover-repro.tar.gz")
    tar_path = tmp_path / "dist" / "g2-fover-repro.tar.gz"

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import hashlib
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
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text(
        "print('hi')\n", encoding="utf-8"
    )

    # Fake subprocess proc that returns GREEN_STDOUT
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = GREEN_STDOUT
    fake_proc.stderr = ""

    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2.py"
    )
    spec = importlib.util.spec_from_file_location("exp3499", script_path)
    mod = importlib.util.module_from_spec(spec)

    out_path = results_dir / (
        "experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2.json"
    )

    import carnot.eval.fover_g2_regression as reg
    import carnot.eval.fover_g2_package as pkg_mod

    with (
        patch.object(reg, "run_package_in_isolated_dir",
                     return_value={"method": "isolated_dir", "exit_code": 0,
                                   "stdout": GREEN_STDOUT, "stderr": ""}),
        patch.object(pkg_mod, "maybe_ipfs_add",
                     return_value={"ipfs_available": False, "package_cid": None}),
        patch("builtins.open", wraps=open),
    ):
        # Patch REPO_ROOT inside the module to point at tmp_path
        with patch.dict(sys.modules, {}):
            # Run main via direct call with patched REPO_ROOT
            import carnot.eval.fover_g2_regression as reg2
            _orig_check = reg2.check_preconditions.__wrapped__ if hasattr(
                reg2.check_preconditions, "__wrapped__") else reg2.check_preconditions
            _orig_read = reg2.read_recorded_sha256.__wrapped__ if hasattr(
                reg2.read_recorded_sha256, "__wrapped__") else reg2.read_recorded_sha256

            def fake_check(repo_root):
                # Call the real function but redirect to tmp_path
                return _orig_check(tmp_path)

            def fake_read_sha(repo_root):
                return _orig_read(tmp_path)

            with (
                patch.object(reg2, "check_preconditions", side_effect=fake_check),
                patch.object(reg2, "read_recorded_sha256", side_effect=fake_read_sha),
                patch.object(reg2, "run_package_in_isolated_dir",
                             return_value={"method": "isolated_dir", "exit_code": 0,
                                           "stdout": GREEN_STDOUT, "stderr": ""}),
            ):
                # Import and invoke
                spec.loader.exec_module(mod)

                # Patch the module-level REPO_ROOT and OUT_PATH, then call main
                mod.REPO_ROOT = tmp_path
                mod.OUT_PATH = out_path

                exit_code = mod.main()

    assert exit_code == 0
    assert out_path.exists(), "v2 artifact JSON must be written"
    data = json.loads(out_path.read_text())
    assert data["experiment"] == 3499
    assert data["artifact"] == "experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2"
    assert data["schema"] == "carnot.fover_g2_regression_verify_external_ask_refresh_v2"
    assert data["g2_met"] is False  # SCENARIO-PUBLISH-039C: never self-marks G2 met
    assert data["external_run_pending"] is True
    assert data["honest_verdict"].startswith("complete:")
    assert data["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert "package_reproduced_auroc" in data
    assert "package_auroc_within_ci" in data
    assert "package_sha256_verified" in data
    assert "duration_s" in data
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
