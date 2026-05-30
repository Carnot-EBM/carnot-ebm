"""Tests for Exp 3463 FoVer G2 CI dry-run + external-reproducer handoff.

Spec: REQ-PUBLISH-037
      SCENARIO-PUBLISH-037 — CI dry-run green + handoff ready: the artifact
        reports ci_workflow_validated, a green stepwise_docker dry-run, a handoff
        package, and a complete: verdict that never claims G2 met.
      SCENARIO-PUBLISH-037B — no isolated runner: the artifact still validates
        the workflow, writes the handoff, and reports an honest status without
        claiming G2 met.

These tests exercise the pure logic of the experiment module (preconditions,
workflow YAML validation, harness-stdout parsing, CI classification, dry-run
green logic, verdict/status mapping, handoff/runbook text builders, artifact
assembly) using synthetic results and a fake clock — they do NOT spawn the heavy
Docker / venv path. The actual run produces the on-disk artifact + docs,
validated by the final tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp3463_g2dryrun", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = _load_module()


# A dry-run result that lands green (exit 0, both numbers in CI).
_GREEN = {
    "condition_a": 0.9131,
    "learning_contribution": 0.0185,
    "reproducibility_checksum": "deadbeef",
    "exit_code": 0,
    "result_pass_line": True,
    "state_files_copied": 21,
}
# A dry-run that errored (e.g. Docker build/run failure).
_ERROR = {"error": "docker_build_failed", "build_stderr": "boom"}
# A dry-run whose AUROC is out of CI (numbers present but wrong, non-zero exit).
_OUT_OF_CI = {
    "condition_a": 0.80,
    "learning_contribution": 0.001,
    "exit_code": 1,
}

_CI_FACTS_OK = {
    "yaml_parses": True,
    "pins_python": True,
    "installs_editable": True,
    "runs_harness": True,
    "harness_asserts_cis": True,
    "ci_workflow_validated": True,
}


# --- preconditions ---------------------------------------------------------


def test_check_preconditions_passes_in_repo():
    result = exp.check_preconditions(REPO_ROOT)
    assert result["ok"] is True
    assert "corpus" in result


def test_check_preconditions_blocks_on_missing_workflow(tmp_path):
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_ci_workflow_missing"


def test_check_preconditions_blocks_on_missing_harness(tmp_path):
    wf = tmp_path / exp.CI_WORKFLOW_REL
    wf.parent.mkdir(parents=True)
    wf.write_text("name: x\n")
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_fover_harness_or_corpus_missing"


# --- static workflow validation --------------------------------------------


def test_validate_workflow_repo_workflow_is_valid():
    facts = exp.validate_workflow(REPO_ROOT)
    assert facts["yaml_parses"] is True
    assert facts["pins_python"] is True
    assert facts["installs_editable"] is True
    assert facts["runs_harness"] is True
    assert facts["harness_asserts_cis"] is True
    assert facts["ci_workflow_validated"] is True


def test_validate_workflow_absent_workflow(tmp_path):
    facts = exp.validate_workflow(tmp_path)
    assert facts["ci_workflow_validated"] is False
    assert facts["yaml_parses"] is False


def test_validate_workflow_unrelated_workflow_not_valid(tmp_path):
    # A workflow that does not run the harness must not validate.
    wf = tmp_path / exp.CI_WORKFLOW_REL
    wf.parent.mkdir(parents=True)
    wf.write_text(
        "name: unrelated\n"
        "on: [push]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo hi\n"
    )
    facts = exp.validate_workflow(tmp_path)
    assert facts["yaml_parses"] is True
    assert facts["runs_harness"] is False
    assert facts["ci_workflow_validated"] is False


def test_validate_workflow_invalid_yaml(tmp_path):
    wf = tmp_path / exp.CI_WORKFLOW_REL
    wf.parent.mkdir(parents=True)
    # Unbalanced bracket -> YAML parse error.
    wf.write_text("name: [unclosed\n")
    facts = exp.validate_workflow(tmp_path)
    assert facts["yaml_parses"] is False
    assert facts["ci_workflow_validated"] is False
    assert facts["parse_error"] is not None


def test_collect_run_commands_extracts_runs():
    doc = {
        "jobs": {
            "a": {"steps": [{"run": "pip install -e ."}, {"uses": "actions/checkout@v4"}]},
            "b": {"steps": [{"run": "python3 scripts/reproduce_fover_headline.py"}]},
            "bad": "not-a-dict",
        }
    }
    cmds = exp._collect_run_commands(doc)
    assert "pip install -e ." in cmds
    assert "python3 scripts/reproduce_fover_headline.py" in cmds


def test_collect_run_commands_no_jobs():
    assert exp._collect_run_commands({}) == []
    assert exp._collect_run_commands({"jobs": "x"}) == []
    assert exp._collect_run_commands({"jobs": {"a": {"steps": "x"}}}) == []


# --- harness stdout parsing ------------------------------------------------


def test_parse_harness_stdout_extracts_numbers():
    out = (
        "condition A (production)        mean AUROC: 0.9131\n"
        "condition B (architecture-only) mean AUROC: 0.8947\n"
        "learning contribution:                      0.0185\n"
        "reproducibility_checksum:                   abc123\n"
        "\nRESULT: PASS — FoVer headline reproduces within published CI\n"
    )
    parsed = exp.parse_harness_stdout(out)
    assert parsed["condition_a"] == 0.9131
    assert parsed["learning_contribution"] == 0.0185
    assert parsed["reproducibility_checksum"] == "abc123"
    assert parsed["result_pass_line"] is True


def test_parse_harness_stdout_handles_missing():
    parsed = exp.parse_harness_stdout("nothing useful here\n")
    assert parsed["condition_a"] is None
    assert parsed["learning_contribution"] is None
    assert parsed["reproducibility_checksum"] is None
    assert parsed["result_pass_line"] is False


# --- CI classification + green logic ---------------------------------------


def test_classify_ci_both_in_range_reproduces():
    assert exp.classify_ci(0.9131, 0.0185) == (True, True, True)


def test_classify_ci_condition_a_out_of_range():
    _, _, repro = exp.classify_ci(0.80, 0.0185)
    assert repro is False


def test_classify_ci_lc_out_of_range():
    _, lc_in, repro = exp.classify_ci(0.9131, 0.001)
    assert lc_in is False
    assert repro is False


def test_classify_ci_none_not_in_ci():
    assert exp.classify_ci(None, None) == (False, False, False)


def test_dryrun_is_green_true_on_zero_exit_and_in_ci():
    assert exp.dryrun_is_green(0, 0.9131, 0.0185) is True


def test_dryrun_is_green_false_on_nonzero_exit():
    assert exp.dryrun_is_green(1, 0.9131, 0.0185) is False


def test_dryrun_is_green_false_when_exit_zero_but_out_of_ci():
    # Defensive: exit 0 but numbers out of CI must not count as green.
    assert exp.dryrun_is_green(0, 0.80, 0.0185) is False


# --- method selection ------------------------------------------------------


def test_select_dryrun_method_prefers_act():
    assert exp.select_dryrun_method(True, True) == "act"


def test_select_dryrun_method_docker_when_no_act():
    assert exp.select_dryrun_method(False, True) == "stepwise_docker"


def test_select_dryrun_method_venv_fallback():
    assert exp.select_dryrun_method(False, False) == "stepwise_venv"


# --- tool availability probe -----------------------------------------------


def test_tool_available_true_no_info_cmd(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/act")
    assert exp.tool_available("act") is True


def test_tool_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: None)
    assert exp.tool_available("docker", ["docker", "info"]) is False


def test_tool_available_true_when_info_succeeds(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    class _P:
        returncode = 0

    assert exp.tool_available("docker", ["docker", "info"], runner=lambda *a, **k: _P()) is True


def test_tool_available_false_when_info_fails(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    class _P:
        returncode = 1

    assert exp.tool_available("docker", ["docker", "info"], runner=lambda *a, **k: _P()) is False


def test_tool_available_false_on_subprocess_error(monkeypatch):
    monkeypatch.setattr(exp.shutil, "which", lambda _: "/usr/bin/docker")

    def _boom(*a, **k):
        raise OSError("no daemon")

    assert exp.tool_available("docker", ["docker", "info"], runner=_boom) is False


# --- verdict / status mapping ----------------------------------------------


def test_verdict_dryrun_green():
    verdict, status = exp.determine_verdict_and_status(
        ci_validated=True, dryrun_green=True, has_error=False, dryrun_ran=True
    )
    assert verdict.startswith("complete:")
    assert "dryrun_green_handoff_ready" in verdict
    assert status == "ci_dryrun_green_handoff_ready_external_run_pending"


def test_verdict_validated_dryrun_unavailable():
    # No runner produced an exit code (dryrun_ran False) but workflow validated.
    verdict, status = exp.determine_verdict_and_status(
        ci_validated=True, dryrun_green=False, has_error=False, dryrun_ran=False
    )
    assert "validated_handoff_ready_dryrun_unavailable" in verdict
    assert status == "ci_validated_dryrun_unavailable"


def test_verdict_still_failing_on_error():
    verdict, status = exp.determine_verdict_and_status(
        ci_validated=True, dryrun_green=False, has_error=True, dryrun_ran=True
    )
    assert "dryrun_failing_container_error" in verdict
    assert status == "still_failing_container_error"


def test_verdict_still_failing_on_out_of_ci():
    # A dry-run ran (exit code present) but numbers were out of CI.
    verdict, status = exp.determine_verdict_and_status(
        ci_validated=True, dryrun_green=False, has_error=False, dryrun_ran=True
    )
    assert "dryrun_failing_auroc_outside_published_ci" in verdict
    assert status == "still_failing_auroc_outside_published_ci"


# --- handoff doc + runbook text --------------------------------------------


def test_build_handoff_doc_has_one_command_and_assertions():
    doc = exp.build_handoff_doc("c0ffee", _GREEN)
    assert "git clone" in doc
    assert "pip install -e ." in doc
    assert "scripts/reproduce_fover_headline.py" in doc
    assert "0.9027" in doc and "0.9235" in doc
    assert "0.0125" in doc and "0.0245" in doc
    assert "c0ffee" in doc
    assert "0.9131" in doc  # the dry-run condition-A value
    # No emojis in public-ish docs.
    assert "✅" not in doc and "🤖" not in doc


def test_build_handoff_doc_handles_missing_numbers():
    doc = exp.build_handoff_doc("c0ffee", {"condition_a": None, "learning_contribution": None})
    assert "n/a" in doc


def test_write_handoff_doc_writes(tmp_path):
    path = exp.write_handoff_doc(tmp_path, "# hello\n")
    assert path.exists()
    assert path.read_text() == "# hello\n"
    assert path.name == "g2-external-reproducer-handoff.md"


def test_build_runbook_append_green_block():
    block = exp.build_runbook_append("stepwise_docker", _GREEN, green=True)
    assert "DRY-RUN" in block
    assert "GREEN" in block
    assert "stepwise_docker" in block
    assert "0.91310" in block


def test_build_runbook_append_not_green_block():
    block = exp.build_runbook_append("stepwise_venv", _OUT_OF_CI, green=False)
    assert "NOT GREEN" in block


def test_append_to_runbook_appends(tmp_path):
    runbook = tmp_path / exp.RUNBOOK_REL
    runbook.parent.mkdir(parents=True)
    runbook.write_text("original\n")
    assert exp.append_to_runbook(tmp_path, "\nappended\n") is True
    text = runbook.read_text()
    assert text.startswith("original\n")
    assert "appended" in text


def test_append_to_runbook_missing_returns_false(tmp_path):
    assert exp.append_to_runbook(tmp_path, "x") is False


# --- Dockerfile / context content ------------------------------------------


def test_build_dockerfile_content_uses_clean_base_and_assert_cmd():
    df = exp.build_dockerfile_content()
    assert exp.DOCKER_BASE_IMAGE in df
    assert "pip install --no-cache-dir -e ." in df
    assert "reproduce_fover_headline.py" in df
    assert "JAX_PLATFORMS=cpu" in df


def test_build_context_copies_minimal_tree(tmp_path):
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
    info = exp.build_context(repo, ctx)

    assert (ctx / "Dockerfile").exists()
    assert (ctx / "pyproject.toml").exists()
    assert (ctx / "python" / "carnot" / "__init__.py").exists()
    assert not (ctx / "python" / "carnot" / "junk.pyc").exists()
    assert (ctx / "data" / "fover_corpus.jsonl").exists()
    assert (ctx / "data" / "fr11_state.jsonl").exists()
    assert info["state_files_copied"] >= 1


def test_sha256_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    import hashlib

    assert exp._sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


# --- artifact assembly -----------------------------------------------------


def _required_fields():
    return {
        "honest_verdict",
        "inference_substrate",
        "ci_workflow_validated",
        "ci_dryrun_method",
        "g2_ci_dryrun_green",
        "condition_a_auroc_isolated",
        "learning_contribution_isolated",
        "g2_handoff_package_ready",
        "handoff_doc_path",
        "g2_status",
        "g2_independent_reproducer",
        "reproducibility_checksum",
        "random_seed",
        "duration_s",
    }


def test_build_artifact_green_path_has_all_fields():
    # SCENARIO-PUBLISH-037: dry-run green + handoff ready.
    artifact = exp.build_artifact(
        start_time=100.0,
        preconditions={"ok": True},
        ci_facts=_CI_FACTS_OK,
        dryrun_method="stepwise_docker",
        isolated=_GREEN,
        handoff_path="docs/g2-external-reproducer-handoff.md",
        handoff_ready=True,
        runbook_appended=True,
        clock=lambda: 142.0,
    )
    assert _required_fields().issubset(artifact.keys())
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["ci_workflow_validated"] is True
    assert artifact["ci_dryrun_method"] == "stepwise_docker"
    assert artifact["g2_ci_dryrun_green"] is True
    assert artifact["condition_a_auroc_isolated"] == 0.9131
    assert artifact["learning_contribution_isolated"] == 0.0185
    assert artifact["g2_handoff_package_ready"] is True
    assert artifact["g2_status"] == "ci_dryrun_green_handoff_ready_external_run_pending"
    # NEVER claim G2 met.
    assert artifact["g2_independent_reproducer"] is False
    assert artifact["duration_s"] == 42.0
    assert artifact["random_seed"] == [42, 137, 271, 314, 1729]
    assert artifact["docker_base_image"] == exp.DOCKER_BASE_IMAGE
    # Every required field carries a principle.
    assert _required_fields().issubset(set(artifact["field_principles"].keys()))


def test_build_artifact_venv_fallback():
    # SCENARIO-PUBLISH-037B: no container -> venv, did not land green.
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        ci_facts=_CI_FACTS_OK,
        dryrun_method="stepwise_venv",
        isolated=_OUT_OF_CI,
        handoff_path="docs/g2-external-reproducer-handoff.md",
        handoff_ready=True,
        runbook_appended=True,
        clock=lambda: 10.0,
    )
    assert artifact["ci_dryrun_method"] == "stepwise_venv"
    assert artifact["docker_base_image"] is None
    assert artifact["g2_handoff_package_ready"] is True
    assert artifact["g2_independent_reproducer"] is False
    # exit_code 1 with numbers out of CI -> still_failing
    assert artifact["g2_status"].startswith("still_failing_")


def test_build_artifact_error_path_reports_still_failing():
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        ci_facts=_CI_FACTS_OK,
        dryrun_method="stepwise_docker",
        isolated=_ERROR,
        handoff_path="docs/g2-external-reproducer-handoff.md",
        handoff_ready=True,
        runbook_appended=True,
        clock=lambda: 5.0,
    )
    assert artifact["g2_ci_dryrun_green"] is False
    assert artifact["condition_a_auroc_isolated"] is None
    assert artifact["g2_status"].startswith("still_failing_")
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["isolated_harness_error_if_any"] == "docker_build_failed"


def test_run_experiment_blocks_when_preconditions_fail(monkeypatch):
    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda _root: {"ok": False, "blocked_reason": "blocked_ci_workflow_missing"},
    )
    artifact = exp.run_experiment(clock=lambda: 1.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked_ci_workflow_missing" in artifact["honest_verdict"]
    assert artifact["g2_independent_reproducer"] is False


# --- on-disk deliverable ----------------------------------------------------


def test_emitted_artifact_on_disk_is_valid():
    path = REPO_ROOT / "results" / (
        "experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.json"
    )
    assert path.exists(), "experiment must have written its artifact"
    data = json.loads(path.read_text())
    assert data["honest_verdict"].startswith("complete:")
    assert data["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    # The CI workflow mechanism is statically validated.
    assert data["ci_workflow_validated"] is True
    # NEVER claim G2 met from autonomous work.
    assert data["g2_independent_reproducer"] is False
    # The handoff package exists.
    assert data["g2_handoff_package_ready"] is True
    assert data["handoff_doc_path"] == "docs/g2-external-reproducer-handoff.md"


def test_emitted_handoff_doc_exists_and_has_one_command():
    doc = REPO_ROOT / "docs" / "g2-external-reproducer-handoff.md"
    assert doc.exists()
    text = doc.read_text()
    assert "git clone" in text
    assert "pip install -e ." in text
    assert "scripts/reproduce_fover_headline.py" in text
    assert "0.9027" in text and "0.9235" in text
