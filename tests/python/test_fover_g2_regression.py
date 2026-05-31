"""Tests for carnot.eval.fover_g2_regression (exp 3488).

Spec: REQ-PUBLISH-039, SCENARIO-PUBLISH-039, SCENARIO-PUBLISH-039B

These tests exercise the clean-room regression verifier + external-ask author.
The subprocess runs (fresh venv, isolated dir) and IPFS are stubbed via injected
``runner`` callables / synthetic repo fixtures, so the suite is fast, hermetic,
and never shells out or touches the network.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

import carnot.eval.fover_g2_regression as reg
from carnot.eval.fover_g2_regression import (
    OPERATOR_CHECKLIST_REL,
    REPRO_WORKFLOW_REL,
    REPRODUCER_INVITE_REL,
    append_runbook,
    build_artifact,
    build_operator_checklist,
    build_repro_workflow_yaml,
    build_reproducer_invite,
    check_preconditions,
    determine_verdict,
    read_recorded_sha256,
    regression_verify,
    run_package_in_fresh_venv,
    run_package_in_isolated_dir,
    verify_sha256,
)
from carnot.eval.fover_g2_package import PACKAGE_NAME, TARBALL_REL

# A faithful slice of the harness stdout the regression run parses.
GREEN_STDOUT = (
    "condition A (production)        mean AUROC: 0.9131\n"
    "condition B (architecture-only) mean AUROC: 0.8947\n"
    "learning contribution:                      0.0185\n"
    "reproducibility_checksum:                   abc123def456\n"
    "\n"
    "condition A in CI [0.9027, 0.9235]: True\n"
    "learning_contribution in CI [0.0125, 0.0245]: True\n"
    "\n"
    "RESULT: PASS — FoVer headline reproduces within published CI\n"
)

DRIFT_STDOUT = (
    "condition A (production)        mean AUROC: 0.8500\n"
    "condition B (architecture-only) mean AUROC: 0.8000\n"
    "learning contribution:                      0.0500\n"
    "RESULT: FAIL — one or more numbers outside published CI\n"
)


# ---------------------------------------------------------------------------
# Fixtures: a minimal on-disk tarball that looks like the real package
# ---------------------------------------------------------------------------


def _make_fake_package_tarball(tmp_path: Path) -> Path:
    """Build a tiny tarball with the package layout the runners expect."""
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


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# check_preconditions
# ---------------------------------------------------------------------------


def test_preconditions_ok_when_tarball_present(tmp_path: Path):
    (tmp_path / "dist").mkdir()
    (tmp_path / TARBALL_REL).write_bytes(b"x")
    pre = check_preconditions(tmp_path)
    assert pre["ok"] is True
    assert pre["package_present"] is True
    assert pre["tarball"].endswith("g2-fover-repro.tar.gz")


def test_preconditions_ok_when_rebuildable(tmp_path: Path):
    # No tarball, but harness + corpus exist -> rebuildable -> ok.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text("x")
    pre = check_preconditions(tmp_path)
    assert pre["ok"] is True
    assert pre["package_present"] is False
    assert pre["package_rebuildable"] is True


def test_preconditions_blocked_when_package_unavailable(tmp_path: Path):
    pre = check_preconditions(tmp_path)
    assert pre["ok"] is False
    assert pre["blocked_reason"] == "blocked_g2_package_unavailable"


def test_preconditions_blocked_when_no_isolated_runner(tmp_path: Path, monkeypatch):
    (tmp_path / "dist").mkdir()
    (tmp_path / TARBALL_REL).write_bytes(b"x")
    monkeypatch.setattr(reg.sys, "executable", "")
    pre = check_preconditions(tmp_path)
    assert pre["ok"] is False
    assert pre["blocked_reason"] == "blocked_fresh_env_unavailable"


# ---------------------------------------------------------------------------
# read_recorded_sha256 / verify_sha256
# ---------------------------------------------------------------------------


def test_read_recorded_sha256_reads_artifact(tmp_path: Path):
    art = tmp_path / reg.EXP3476_ARTIFACT_REL
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"package_sha256": "deadbeef"}))
    assert read_recorded_sha256(tmp_path) == "deadbeef"


def test_read_recorded_sha256_missing_returns_none(tmp_path: Path):
    assert read_recorded_sha256(tmp_path) is None


def test_read_recorded_sha256_malformed_returns_none(tmp_path: Path):
    art = tmp_path / reg.EXP3476_ARTIFACT_REL
    art.parent.mkdir(parents=True)
    art.write_text("{not json")
    assert read_recorded_sha256(tmp_path) is None


def test_verify_sha256_match(tmp_path: Path):
    f = tmp_path / "pkg.tar.gz"
    f.write_bytes(b"hello world")
    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    out = verify_sha256(f, expected)
    assert out["verified"] is True
    assert out["computed"] == expected


def test_verify_sha256_mismatch(tmp_path: Path):
    f = tmp_path / "pkg.tar.gz"
    f.write_bytes(b"hello world")
    out = verify_sha256(f, "not_the_real_hash")
    assert out["verified"] is False


def test_verify_sha256_missing_file(tmp_path: Path):
    out = verify_sha256(tmp_path / "absent.tar.gz", "anything")
    assert out["verified"] is False
    assert out["computed"] is None


def test_verify_sha256_no_recorded(tmp_path: Path):
    f = tmp_path / "pkg.tar.gz"
    f.write_bytes(b"x")
    out = verify_sha256(f, None)
    assert out["verified"] is False


# ---------------------------------------------------------------------------
# isolated-dir + fresh-venv runs (mocked runner)
# ---------------------------------------------------------------------------


def test_run_isolated_dir_parses_green(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)

    def fake_runner(cmd, **kwargs):
        # The harness path is the second token; cwd is the extracted dir.
        assert cmd[1] == reg.HARNESS_REL_IN_PKG
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert "python" in kwargs["env"]["PYTHONPATH"]
        return _FakeProc(0, GREEN_STDOUT, "")

    out = run_package_in_isolated_dir(tar_path, runner=fake_runner)
    assert out["method"] == "isolated_dir"
    assert out["exit_code"] == 0
    assert "0.9131" in out["stdout"]


def test_run_isolated_dir_missing_harness(tmp_path: Path):
    # Tarball whose package dir lacks the harness.
    pkg = tmp_path / PACKAGE_NAME
    pkg.mkdir()
    (pkg / "placeholder").write_text("x")
    tar_path = tmp_path / "g2-fover-repro.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg, arcname=PACKAGE_NAME)
    out = run_package_in_isolated_dir(tar_path, runner=lambda *a, **k: _FakeProc())
    assert out["error"] == "package_missing_harness"


def test_run_fresh_venv_happy_path(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            return _FakeProc(0)
        if "pip" in cmd:
            return _FakeProc(0)
        return _FakeProc(0, GREEN_STDOUT, "")

    out = run_package_in_fresh_venv(tar_path, runner=fake_runner)
    assert out["method"] == "fresh_venv"
    assert out["exit_code"] == 0
    assert "0.9131" in out["stdout"]
    # venv create + 2 pip installs + harness run = 4 calls.
    assert len(calls) == 4


def test_run_fresh_venv_missing_run_sh(tmp_path: Path):
    pkg = tmp_path / PACKAGE_NAME
    pkg.mkdir()
    (pkg / "placeholder").write_text("x")
    tar_path = tmp_path / "g2-fover-repro.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg, arcname=PACKAGE_NAME)
    out = run_package_in_fresh_venv(tar_path, runner=lambda *a, **k: _FakeProc())
    assert out["error"] == "package_missing_run_sh"


def test_run_fresh_venv_venv_create_fails(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)

    def fake_runner(cmd, **kwargs):
        return _FakeProc(1, "", "venv boom")

    out = run_package_in_fresh_venv(tar_path, runner=fake_runner)
    assert out["error"] == "venv_create_failed"


def test_run_fresh_venv_pip_requirements_fails(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)

    def fake_runner(cmd, **kwargs):
        if "venv" in cmd:
            return _FakeProc(0)
        if "pip" in cmd:
            return _FakeProc(1, "", "no network")
        return _FakeProc(0)

    out = run_package_in_fresh_venv(tar_path, runner=fake_runner)
    assert out["error"] == "pip_install_requirements_failed"


def test_run_fresh_venv_pip_package_fails(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)
    pip_calls = []

    def fake_runner(cmd, **kwargs):
        if "venv" in cmd:
            return _FakeProc(0)
        if "pip" in cmd:
            pip_calls.append(cmd)
            # First pip (requirements) ok, second pip (-e .) fails.
            return _FakeProc(0) if len(pip_calls) == 1 else _FakeProc(1, "", "boom")
        return _FakeProc(0)

    out = run_package_in_fresh_venv(tar_path, runner=fake_runner)
    assert out["error"] == "pip_install_package_failed"


# ---------------------------------------------------------------------------
# regression_verify orchestration
# ---------------------------------------------------------------------------


def test_regression_verify_isolated_dir_green(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)
    out = regression_verify(
        tar_path, prefer_fresh_venv=False,
        runner=lambda *a, **k: _FakeProc(0, GREEN_STDOUT, ""),
    )
    assert out["clean_env_method"] == "isolated_dir"
    assert out["condition_a_auroc"] == 0.9131
    assert out["condition_a_in_ci"] is True
    assert out["reproduced"] is True
    assert out["isolated_checksum"] == "abc123def456"


def test_regression_verify_drift(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)
    out = regression_verify(
        tar_path, prefer_fresh_venv=False,
        runner=lambda *a, **k: _FakeProc(1, DRIFT_STDOUT, ""),
    )
    assert out["condition_a_auroc"] == 0.85
    assert out["condition_a_in_ci"] is False
    assert out["reproduced"] is False


def test_regression_verify_prefers_venv_then_falls_back(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)

    def fake_runner(cmd, **kwargs):
        # Fail venv create so fresh_venv errors and we fall back to isolated_dir,
        # which then returns green.
        if "venv" in cmd:
            return _FakeProc(1, "", "no venv")
        return _FakeProc(0, GREEN_STDOUT, "")

    out = regression_verify(tar_path, prefer_fresh_venv=True, runner=fake_runner)
    # fresh_venv attempt recorded with error, isolated_dir is what succeeded.
    methods = [a["method"] for a in out["attempts"]]
    assert "fresh_venv" in methods and "isolated_dir" in methods
    assert out["clean_env_method"] == "isolated_dir"
    assert out["reproduced"] is True


def test_regression_verify_venv_success_no_fallback(tmp_path: Path):
    tar_path = _make_fake_package_tarball(tmp_path)

    def fake_runner(cmd, **kwargs):
        if "venv" in cmd or "pip" in cmd:
            return _FakeProc(0)
        return _FakeProc(0, GREEN_STDOUT, "")

    out = regression_verify(tar_path, prefer_fresh_venv=True, runner=fake_runner)
    assert out["clean_env_method"] == "fresh_venv"
    assert [a["method"] for a in out["attempts"]] == ["fresh_venv"]


# ---------------------------------------------------------------------------
# external-ask string builders
# ---------------------------------------------------------------------------


def test_workflow_yaml_is_dispatch_only_and_runs_harness():
    yml = build_repro_workflow_yaml()
    assert "workflow_dispatch:" in yml
    # Must NOT have a schedule (one-click only, no auto-trigger).
    assert "schedule:" not in yml
    assert "scripts/reproduce_fover_headline.py" in yml
    assert "pip install -e ." in yml
    assert "JAX_PLATFORMS: cpu" in yml


def test_reproducer_invite_includes_command_and_integrity():
    invite = build_reproducer_invite("deadbeefsha", "QmCID123")
    assert "g2-fover-repro" in invite
    assert "deadbeefsha" in invite
    assert "QmCID123" in invite
    assert "not the project operator" in invite


def test_reproducer_invite_without_cid_or_sha():
    invite = build_reproducer_invite(None, None)
    assert "ipfs get" not in invite
    assert "sha256" not in invite


def test_operator_checklist_terminal_step_is_external_action():
    md = build_operator_checklist(
        package_path=TARBALL_REL,
        package_sha256="abc",
        package_sha256_verified=True,
        package_cid="QmX",
        reproduced_auroc=0.9131,
        auroc_within_ci=True,
        clean_env_method="isolated_dir",
        workflow_path=REPRO_WORKFLOW_REL,
        invite_path=REPRODUCER_INVITE_REL,
    )
    assert "TERMINAL STEP" in md
    assert "Run workflow" in md
    assert "[x]" in md  # green preconditions
    assert "0.9131" in md


def test_operator_checklist_marks_unverified_preconditions():
    md = build_operator_checklist(
        package_path=None,
        package_sha256=None,
        package_sha256_verified=False,
        package_cid=None,
        reproduced_auroc=None,
        auroc_within_ci=False,
        clean_env_method=None,
        workflow_path=REPRO_WORKFLOW_REL,
        invite_path=REPRODUCER_INVITE_REL,
    )
    assert "[ ]" in md  # at least one unchecked precondition
    assert "n/a" in md


# ---------------------------------------------------------------------------
# append_runbook
# ---------------------------------------------------------------------------


def test_append_runbook_appends_when_present(tmp_path: Path):
    rb = tmp_path / reg.RUNBOOK_REL
    rb.parent.mkdir(parents=True)
    rb.write_text("# existing runbook\n")
    ok = append_runbook(
        tmp_path, reproduced_auroc=0.9131, auroc_within_ci=True,
        clean_env_method="isolated_dir", package_sha256="abc",
        package_sha256_verified=True, package_cid="QmX",
    )
    assert ok is True
    text = rb.read_text()
    assert "# existing runbook" in text  # never deletes
    assert "exp3488" in text
    assert "0.9131" in text


def test_append_runbook_skips_when_absent(tmp_path: Path):
    ok = append_runbook(
        tmp_path, reproduced_auroc=None, auroc_within_ci=False,
        clean_env_method=None, package_sha256=None,
        package_sha256_verified=False, package_cid=None,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# determine_verdict + build_artifact
# ---------------------------------------------------------------------------


def test_determine_verdict_clean():
    v = determine_verdict(package_available=True, auroc_within_ci=True,
                          sha256_verified=True, external_ask_ready=True)
    assert v.startswith("complete: ")
    assert "regression_clean_external_ask_ready" in v


def test_determine_verdict_drift():
    v = determine_verdict(package_available=True, auroc_within_ci=False,
                          sha256_verified=True, external_ask_ready=True)
    assert "regression_drift_detected_needs_rebuild" in v


def test_determine_verdict_blocked():
    v = determine_verdict(package_available=False, auroc_within_ci=False,
                          sha256_verified=False, external_ask_ready=False)
    assert v == "complete: blocked_g2_package_unavailable"


def test_build_artifact_clean_has_all_required_fields():
    artifact = build_artifact(
        start_time=0.0,
        preconditions={"ok": True, "tarball": "dist/g2-fover-repro.tar.gz"},
        regression={
            "clean_env_method": "isolated_dir",
            "condition_a_auroc": 0.9131,
            "learning_contribution": 0.0185,
            "condition_a_in_ci": True,
            "isolated_checksum": "abc123",
            "attempts": [{"method": "isolated_dir", "error": None}],
        },
        sha_check={"computed": "sha_now", "recorded": "sha_now", "verified": True},
        ipfs_result={"ipfs_available": True, "package_cid": "QmCID"},
        workflow_path=REPRO_WORKFLOW_REL,
        invite_path=REPRODUCER_INVITE_REL,
        checklist_path=OPERATOR_CHECKLIST_REL,
        runbook_appended=True,
        clock=lambda: 5.0,
    )
    # Every REQUIRED ARTIFACT FIELD from the task spec.
    for field in (
        "honest_verdict", "inference_substrate", "package_reproduced_auroc",
        "package_auroc_within_ci", "package_sha256_verified", "package_cid",
        "external_ask_workflow_path", "operator_checklist_path", "g2_met",
        "external_run_pending", "random_seed", "reproducibility_checksum",
        "duration_s",
    ):
        assert field in artifact, f"missing required field {field}"
    assert artifact["honest_verdict"].startswith("complete: ")
    assert artifact["inference_substrate"] == (
        "verifier_ensemble_against_cached_candidates"
    )
    assert artifact["package_reproduced_auroc"] == 0.9131
    assert artifact["package_auroc_within_ci"] is True
    assert artifact["package_sha256_verified"] is True
    assert artifact["package_cid"] == "QmCID"
    assert artifact["g2_met"] is False
    assert artifact["g2_independent_reproducer"] is False
    assert artifact["external_run_pending"] is True
    assert artifact["duration_s"] == 5.0
    assert artifact["reproducibility_checksum"] == "abc123"
    assert artifact["random_seed"] == [42, 137, 271, 314, 1729]


def test_build_artifact_blocked_branch():
    artifact = build_artifact(
        start_time=0.0,
        preconditions={"ok": False, "blocked_reason": "blocked_g2_package_unavailable"},
        regression={},
        sha_check={"computed": None, "recorded": None, "verified": False},
        ipfs_result={"ipfs_available": False, "package_cid": None},
        workflow_path=None,
        invite_path=None,
        checklist_path=None,
        runbook_appended=False,
        clock=lambda: 1.0,
    )
    assert artifact["g2_met"] is False
    assert artifact["package_auroc_within_ci"] is False
    assert artifact["external_ask_workflow_path"] is None
    # Verdict from the blocked precondition path.
    assert "blocked_g2_package_unavailable" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"] == "preconditions_blocked"


def test_build_artifact_drift_branch_falls_back_to_sha_checksum():
    artifact = build_artifact(
        start_time=0.0,
        preconditions={"ok": True, "tarball": TARBALL_REL},
        regression={
            "clean_env_method": "isolated_dir",
            "condition_a_auroc": 0.85,
            "condition_a_in_ci": False,
            "isolated_checksum": None,
            "attempts": [],
        },
        sha_check={"computed": "sha_now", "recorded": "sha_old", "verified": False},
        ipfs_result={"ipfs_available": False, "package_cid": None},
        workflow_path=REPRO_WORKFLOW_REL,
        invite_path=REPRODUCER_INVITE_REL,
        checklist_path=OPERATOR_CHECKLIST_REL,
        runbook_appended=True,
        clock=lambda: 2.0,
    )
    assert "regression_drift_detected_needs_rebuild" in artifact["honest_verdict"]
    # No isolated checksum -> falls back to the computed sha256.
    assert artifact["reproducibility_checksum"] == "sha_now"


def test_constants_point_at_working_tree_paths():
    assert REPRO_WORKFLOW_REL == ".github/workflows/fover-g2-repro.yml"
    assert REPRODUCER_INVITE_REL == "docs/g2-reproducer-invite.md"
    assert OPERATOR_CHECKLIST_REL == "ops/g2-external-ask-operator-checklist.md"
