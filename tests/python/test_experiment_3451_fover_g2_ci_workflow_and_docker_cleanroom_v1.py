"""Tests for Exp 3451 FoVer G2 CI workflow + Docker clean-room.

Spec: REQ-PUBLISH-036
      SCENARIO-PUBLISH-036 — CI workflow + Docker clean-room both ready: the
        artifact reports the workflow path, a Docker reproduction in CI, and a
        complete: verdict that never claims G2 met.
      SCENARIO-PUBLISH-036B — Docker unavailable: the artifact still records the
        CI workflow path, falls back to a fresh-venv clean-room, and reports an
        honest docker_unavailable status.

These tests exercise the pure logic of the experiment module (precondition
gating, CI-band classification, verdict/status mapping, workflow-assertion
detection, Dockerfile/content builders, artifact assembly) using synthetic
harness results and a fake clock — they do NOT spawn the heavy Docker build /
venv install path. The actual run produces the on-disk artifact, validated by
the final test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp3451_g2ci", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = _load_module()


# A harness result that lands inside the published CI (the happy path).
_GOOD_HARNESS = {
    "condition_a_production_auroc_mean": 0.9131,
    "learning_contribution_ci95": {"mean": 0.0185, "low": 0.0125, "high": 0.0245},
    "reproducibility_checksum": "deadbeef",
    "live_model_invoked": False,
    "state_files_copied": 17,
}
# A harness result that errored (e.g. a Docker build/run failure).
_ERROR_HARNESS = {"error": "docker_build_failed", "build_stderr": "boom"}
# A harness result whose AUROC is out of CI (numbers present but wrong).
_OUT_OF_CI_HARNESS = {
    "condition_a_production_auroc_mean": 0.80,
    "learning_contribution": 0.001,
    "reproducibility_checksum": "cafe",
}

_CI_READY = {
    "present": True,
    "invokes_harness": True,
    "harness_asserts": True,
    "asserts_cis": True,
    "path": exp.CI_WORKFLOW_REL,
}


# --- preconditions ---------------------------------------------------------


def test_check_preconditions_passes_in_repo():
    result = exp.check_preconditions(REPO_ROOT)
    assert result["ok"] is True
    assert "corpus" in result


def test_check_preconditions_blocks_on_missing_harness(tmp_path):
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_fover_harness_missing"


def test_check_preconditions_blocks_on_missing_corpus(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("# stub\n")
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_fover_corpus_missing"


# --- docker availability probe ---------------------------------------------


def test_docker_is_available_true_when_info_succeeds(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    class _P:
        returncode = 0

    assert exp.docker_is_available(runner=lambda *a, **k: _P()) is True


def test_docker_is_available_false_when_no_binary(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: None)
    assert exp.docker_is_available(runner=lambda *a, **k: None) is False


def test_docker_is_available_false_when_info_fails(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    class _P:
        returncode = 1

    assert exp.docker_is_available(runner=lambda *a, **k: _P()) is False


def test_docker_is_available_false_on_subprocess_error(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    def _boom(*a, **k):
        raise OSError("no daemon")

    assert exp.docker_is_available(runner=_boom) is False


# --- CI workflow assertion check (primary acceptance gate) -----------------


def test_ci_workflow_status_repo_has_asserting_workflow():
    # The committed workflow exists and (via the harness) asserts both CIs.
    info = exp.ci_workflow_status(REPO_ROOT)
    assert info["present"] is True
    assert info["invokes_harness"] is True
    assert info["harness_asserts"] is True
    assert info["asserts_cis"] is True
    assert info["path"] == exp.CI_WORKFLOW_REL


def test_ci_workflow_status_absent_workflow(tmp_path):
    # No workflow file and no harness -> not asserting.
    info = exp.ci_workflow_status(tmp_path)
    assert info["present"] is False
    assert info["asserts_cis"] is False
    assert info["path"] == ""


def test_ci_workflow_status_workflow_not_invoking_harness(tmp_path):
    # Workflow present but does not run the reproducer -> does not assert CIs.
    wf = tmp_path / ".github" / "workflows" / "reproduce-fover-headline.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text("name: unrelated\njobs: {}\n")
    info = exp.ci_workflow_status(tmp_path)
    assert info["present"] is True
    assert info["invokes_harness"] is False
    assert info["asserts_cis"] is False


# --- CI classification -----------------------------------------------------


def test_classify_ci_both_in_range_reproduces():
    cond_a_in, lc_in, repro = exp.classify_ci(0.9131, 0.0185)
    assert (cond_a_in, lc_in, repro) == (True, True, True)


def test_classify_ci_condition_a_out_of_range():
    _, _, repro = exp.classify_ci(0.80, 0.0185)
    assert repro is False


def test_classify_ci_lc_out_of_range():
    _, lc_in, repro = exp.classify_ci(0.9131, 0.001)
    assert lc_in is False
    assert repro is False


def test_classify_ci_none_values_not_in_ci():
    cond_a_in, lc_in, repro = exp.classify_ci(None, None)
    assert (cond_a_in, lc_in, repro) == (False, False, False)


def test_extract_numbers_from_ci95_dict():
    cond_a, lc = exp._extract_numbers(_GOOD_HARNESS)
    assert cond_a == 0.9131
    assert lc == 0.0185


def test_extract_numbers_falls_back_to_flat_lc():
    cond_a, lc = exp._extract_numbers(
        {"condition_a_production_auroc_mean": 0.91, "learning_contribution": 0.02}
    )
    assert cond_a == 0.91
    assert lc == 0.02


# --- verdict / status mapping ----------------------------------------------


def test_verdict_docker_reproduced():
    verdict, status = exp.determine_verdict_and_status(
        docker_available=True, isolation_mode="docker", reproduced=True, has_error=False
    )
    assert verdict.startswith("complete:")
    assert "ci_and_docker_cleanroom_ready" in verdict
    assert status == "ci_and_docker_ready_external_run_pending"


def test_verdict_fresh_venv_reproduced_docker_unavailable():
    verdict, status = exp.determine_verdict_and_status(
        docker_available=False,
        isolation_mode="fresh_venv",
        reproduced=True,
        has_error=False,
    )
    assert "docker_unavailable" in verdict
    assert status == "ci_ready_docker_unavailable"


def test_verdict_still_failing_on_error():
    verdict, status = exp.determine_verdict_and_status(
        docker_available=True, isolation_mode="docker", reproduced=False, has_error=True
    )
    assert "still_failing_container_error" in verdict
    assert status == "still_failing_container_error"


def test_verdict_still_failing_on_out_of_ci():
    verdict, status = exp.determine_verdict_and_status(
        docker_available=True,
        isolation_mode="docker",
        reproduced=False,
        has_error=False,
    )
    assert "still_failing_auroc_outside_published_ci" in verdict
    assert status == "still_failing_auroc_outside_published_ci"


# --- Dockerfile / context content ------------------------------------------


def test_build_dockerfile_content_uses_clean_base_and_runs_harness():
    df = exp.build_dockerfile_content()
    assert exp.DOCKER_BASE_IMAGE in df
    assert "pip install --no-cache-dir -e ." in df
    assert "reproduce_fover_headline.py" in df
    assert "JAX_PLATFORMS=cpu" in df


def test_parse_last_json_line_picks_trailing_json():
    out = "some build noise\nWARNING: foo\n" + json.dumps({"a": 1})
    assert exp._parse_last_json_line(out) == {"a": 1}


def test_parse_last_json_line_empty():
    assert exp._parse_last_json_line("   ") == {"error": "no_stdout"}


def test_parse_last_json_line_no_json():
    res = exp._parse_last_json_line("not json at all\nstill not")
    assert res["error"] == "no_json_line_found"


def test_build_docker_context_copies_minimal_tree(tmp_path):
    # Build a fake repo with just enough structure for the context builder.
    repo = tmp_path / "repo"
    (repo / "python" / "carnot").mkdir(parents=True)
    (repo / "python" / "carnot" / "__init__.py").write_text("x = 1\n")
    (repo / "python" / "carnot" / "junk.pyc").write_text("nope")
    (repo / "scripts").mkdir()
    (repo / exp.HARNESS_REL).write_text("# harness\n")
    (repo / "data").mkdir()
    (repo / "data" / "fover_corpus.jsonl").write_text('{"x":1}\n')
    (repo / "data" / "fr11_state.jsonl").write_text("{}\n")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "README.md").write_text("readme\n")

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    info = exp.build_docker_context(repo, ctx)

    assert (ctx / "Dockerfile").exists()
    assert (ctx / "pyproject.toml").exists()
    assert (ctx / "python" / "carnot" / "__init__.py").exists()
    # .pyc excluded by ignore_patterns.
    assert not (ctx / "python" / "carnot" / "junk.pyc").exists()
    assert (ctx / "data" / "fover_corpus.jsonl").exists()
    # The fr11 state glob was copied.
    assert (ctx / "data" / "fr11_state.jsonl").exists()
    assert info["state_files_copied"] >= 1


# --- artifact assembly -----------------------------------------------------


def _required_fields():
    return {
        "honest_verdict",
        "inference_substrate",
        "ci_workflow_path",
        "docker_available",
        "g2_docker_cleanroom_reproduced",
        "condition_a_auroc_isolated",
        "learning_contribution_isolated",
        "g2_status",
        "g2_independent_reproducer",
        "reproducibility_checksum",
        "random_seed",
        "duration_s",
    }


def test_build_artifact_docker_happy_path_has_all_fields():
    # SCENARIO-PUBLISH-036: CI + Docker both ready.
    artifact = exp.build_artifact(
        start_time=100.0,
        preconditions={"ok": True},
        docker_available=True,
        isolation_mode="docker",
        isolated_result=_GOOD_HARNESS,
        ci_info=_CI_READY,
        clock=lambda: 142.0,
    )
    assert _required_fields().issubset(artifact.keys())
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["ci_workflow_path"] == exp.CI_WORKFLOW_REL
    assert artifact["docker_available"] is True
    assert artifact["g2_docker_cleanroom_reproduced"] is True
    assert artifact["condition_a_auroc_isolated"] == 0.9131
    assert artifact["learning_contribution_isolated"] == 0.0185
    assert artifact["g2_status"] == "ci_and_docker_ready_external_run_pending"
    # NEVER claim G2 met.
    assert artifact["g2_independent_reproducer"] is False
    assert artifact["duration_s"] == 42.0
    assert artifact["random_seed"] == [42, 137, 271, 314, 1729]
    # Every required field carries a principle.
    assert _required_fields().issubset(set(artifact["field_principles"].keys()))


def test_build_artifact_docker_unavailable_fresh_venv():
    # SCENARIO-PUBLISH-036B: Docker unavailable -> fresh-venv fallback.
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        docker_available=False,
        isolation_mode="fresh_venv",
        isolated_result=_GOOD_HARNESS,
        ci_info=_CI_READY,
        clock=lambda: 10.0,
    )
    assert artifact["docker_available"] is False
    assert artifact["docker_base_image"] is None
    assert artifact["g2_status"] == "ci_ready_docker_unavailable"
    assert artifact["g2_docker_cleanroom_reproduced"] is True
    assert artifact["g2_independent_reproducer"] is False
    assert "docker_unavailable" in artifact["honest_verdict"]


def test_build_artifact_error_path_reports_still_failing():
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        docker_available=True,
        isolation_mode="docker",
        isolated_result=_ERROR_HARNESS,
        ci_info=_CI_READY,
        docker_run_error="docker_build_failed",
        clock=lambda: 5.0,
    )
    assert artifact["g2_docker_cleanroom_reproduced"] is False
    assert artifact["condition_a_auroc_isolated"] is None
    assert artifact["g2_status"].startswith("still_failing_")
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["isolated_harness_error_if_any"] == "docker_build_failed"


def test_build_artifact_out_of_ci_reports_still_failing():
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        docker_available=True,
        isolation_mode="docker",
        isolated_result=_OUT_OF_CI_HARNESS,
        ci_info=_CI_READY,
        clock=lambda: 3.0,
    )
    assert artifact["g2_docker_cleanroom_reproduced"] is False
    assert artifact["g2_status"] == "still_failing_auroc_outside_published_ci"


def test_run_experiment_blocks_when_preconditions_fail(monkeypatch):
    # Force a precondition failure and confirm the blocked artifact shape.
    monkeypatch.setattr(
        exp, "check_preconditions", lambda _root: {"ok": False, "blocked_reason": "blocked_fover_corpus_missing"}
    )
    artifact = exp.run_experiment(clock=lambda: 1.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked_fover_corpus_missing" in artifact["honest_verdict"]
    assert artifact["g2_independent_reproducer"] is False


# --- on-disk deliverable ----------------------------------------------------


def test_emitted_artifact_on_disk_is_valid():
    path = REPO_ROOT / "results" / (
        "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.json"
    )
    assert path.exists(), "experiment must have written its artifact"
    data = json.loads(path.read_text())
    assert data["honest_verdict"].startswith("complete:")
    assert data["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert data["ci_workflow_path"] == ".github/workflows/reproduce-fover-headline.yml"
    # The CI workflow mechanism is the primary deliverable and must assert CIs.
    assert data["ci_workflow_asserts_cis"] is True
    # NEVER claim G2 met from autonomous work.
    assert data["g2_independent_reproducer"] is False
    # The isolated clean-room reproduced both numbers in their published CIs.
    assert data["g2_docker_cleanroom_reproduced"] is True
    assert 0.9027 <= float(data["condition_a_auroc_isolated"]) <= 0.9235
    assert 0.0125 <= float(data["learning_contribution_isolated"]) <= 0.0245


def test_emitted_ci_workflow_file_exists_and_asserts():
    wf = REPO_ROOT / ".github" / "workflows" / "reproduce-fover-headline.yml"
    assert wf.exists()
    text = wf.read_text()
    assert "reproduce_fover_headline.py" in text
    assert "pip install -e ." in text
