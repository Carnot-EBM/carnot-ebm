"""Tests for Exp 4855 hostile A1 generation-diagnostic audit.

Spec refs: REQ-ARC-WMTE-4855,
SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT,
SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4855_generation_diagnostic_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _good_a1_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4851,
        "honest_verdict": "complete_generation_wall_never_enumerated_dominant",
        "proposer_blind_to_banked_answer": True,
        "positive_control_game": "tu93",
        "positive_control_covered": True,
        "positive_control_coverage": {
            "game": "tu93",
            "adaptered": True,
            "bucket": "COVERED",
            "reached_l1_win": True,
            "winning_prefix_len": 18,
            "matched_winning_prefix_len": 18,
        },
        "dominant_bucket": "NEVER_ENUMERATED",
        "n_games_measured": 3,
        "per_game_coverage": {
            "aa00": {"bucket": "NEVER_ENUMERATED"},
            "bb00": {"bucket": "NEVER_ENUMERATED"},
            "cc00": {"bucket": "COVERED"},
        },
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
    }


def _good_a1_source() -> str:
    return """
def measure_game_with_stepwise_explorer(game, winning_prefix, action_budget):
    explorer = RecordingStepwiseExplorer()
    row = classify_game_coverage(
        game,
        winning_prefix,
        explorer.proposal_records,
        reached_l1_win=False,
        budget_actions=action_budget,
    )
    return row

def measure_adapter_positive_control(game="tu93", winning_prefix=None):
    adaptered = True
    return {"game": game, "adaptered": adaptered, "bucket": "COVERED"}
"""


def _clean_auxiliary_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4855_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4855: OpenSpec anchors the hostile audit artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4855",
        "SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT",
        "SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4855_good_a1_is_genuinely_diagnostic() -> None:
    """SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT: all four hostile gates pass."""

    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_a1_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    assert audit["honest_verdict"] == "complete_a1_generation_diagnostic_audited"
    assert audit["a1_genuinely_diagnostic"] is True
    assert audit["non_diagnostic_reasons"] == []
    assert audit["proposer_blind_confirmed"] is True
    assert audit["positive_control_confirmed"] is True
    assert audit["buckets_match_claim"] is True
    assert audit["live_path_reachable_confirmed"] is True
    assert audit["solve_provenance_confirmed"] is True
    assert audit["checks"]["bucket_distribution"]["bucket_counts"] == {
        "COVERED": 1,
        "NEVER_ENUMERATED": 2,
    }
    assert audit["checks"]["bucket_distribution"]["computed_dominant_bucket"] == (
        "NEVER_ENUMERATED"
    )


def test_scenario_arc_wmte_4855_non_test_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION: each hostile gate can fail."""

    summary, adversarial, lint = _clean_auxiliary_results()
    injected_source = """
def measure_game_with_stepwise_explorer(game, winning_prefix, action_budget):
    explorer = RecordingStepwiseExplorer(seed_prefix=winning_prefix)
    return classify_game_coverage(game, winning_prefix, explorer.proposal_records)
"""

    control_failed = _good_a1_artifact()
    control_failed["positive_control_covered"] = False
    control_failed["positive_control_coverage"] = {"adaptered": True, "bucket": "NEVER_ENUMERATED"}

    bucket_mismatch = _good_a1_artifact()
    bucket_mismatch["dominant_bucket"] = "COVERED"

    too_few_games = _good_a1_artifact()
    too_few_games["n_games_measured"] = 2
    too_few_games["per_game_coverage"] = {
        "aa00": {"bucket": "NEVER_ENUMERATED"},
        "bb00": {"bucket": "COVERED"},
    }

    dishonest = _good_a1_artifact()
    dishonest["live_path_reachable"] = False
    dishonest["solve_provenance"] = "live_agent_self_discovery"
    bad_lint = {"passed": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "boom"}

    flagged_blind = _good_a1_artifact()
    flagged_blind["proposer_blind_to_banked_answer"] = False

    cases = [
        (_good_a1_artifact(), injected_source, lint, "banked_answer_used_before_classification"),
        (control_failed, _good_a1_source(), lint, "positive_control_not_covered"),
        (bucket_mismatch, _good_a1_source(), lint, "dominant_bucket_mismatch"),
        (too_few_games, _good_a1_source(), lint, "n_games_measured_below_3"),
        (dishonest, _good_a1_source(), bad_lint, "live_path_unreachable"),
        (dishonest, _good_a1_source(), lint, "solve_provenance_not_development_proxy"),
        (flagged_blind, _good_a1_source(), lint, "artifact_proposer_blind_flag_false"),
    ]

    for artifact, source_text, lint_result, reason in cases:
        audit = mod.audit_a1_artifact(
            artifact,
            source_text=source_text,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint_result,
        )
        assert audit["a1_genuinely_diagnostic"] is False
        assert reason in audit["non_diagnostic_reasons"]
        assert audit["honest_verdict"].startswith("complete_a1_generation_diagnostic_non_test_")


def test_req_arc_wmte_4855_build_schema_and_report_write(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4855: artifact and markdown writes are checksum-stable."""

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.SOURCE_SCRIPT_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    script.write_text(_good_a1_source(), encoding="utf-8")
    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_a1_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    artifact = mod.build_artifact(
        source_path=source,
        source_script_path=script,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"ok": True, "source_artifact_present": True},
        duration_s=0.0,
    )
    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["duration_s"] == mod.DURATION_FLOOR_S
    assert artifact["source_artifact_checksum"] == mod.file_checksum(source)
    assert artifact["source_script_checksum"] == mod.file_checksum(script)

    result_path = mod.write_artifact(artifact, root=tmp_path)
    report_path = mod.append_markdown_report(artifact, root=tmp_path)
    mod.append_markdown_report(artifact, root=tmp_path)

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")
    assert loaded == artifact
    assert report_text.count("## Experiment 4855 .447 A1 Generation Diagnostic Audit") == 1
    assert "a1_genuinely_diagnostic" in report_text

    broken = dict(artifact)
    broken.update(
        {
            "a1_genuinely_diagnostic": "yes",
            "proposer_blind_confirmed": "yes",
            "positive_control_confirmed": "yes",
            "buckets_match_claim": "yes",
            "field_principles": {},
            "inference_substrate": "live_llm_inference",
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "a1_genuinely_diagnostic_must_be_bool",
        "proposer_blind_confirmed_must_be_bool",
        "positive_control_confirmed_must_be_bool",
        "buckets_match_claim_must_be_bool",
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError, match="a1_genuinely_diagnostic_must_be_bool"):
        mod.write_artifact(broken, root=tmp_path)


def test_req_arc_wmte_4855_run_checked_in_a1_artifact() -> None:
    """SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT: checked-in Exp 4851 passes audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_generation_diagnostic_audited"
    assert artifact["a1_genuinely_diagnostic"] is True
    assert artifact["proposer_blind_confirmed"] is True
    assert artifact["positive_control_confirmed"] is True
    assert artifact["buckets_match_claim"] is True
    assert artifact["live_path_reachable_confirmed"] is True
    assert artifact["solve_provenance_confirmed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["summarizer_result"]["returncode"] == 0
    assert artifact["adversarial_result"]["flag_count"] == 0
    assert artifact["live_lint_result"]["passed"] is True


def test_scenario_arc_wmte_4855_blocked_preconditions_do_not_fabricate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4855: missing A1 inputs produce blocked audit output."""

    artifact = mod.run(root=tmp_path, write=True)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "blocked_a1_artifact_missing"
    assert artifact["a1_genuinely_diagnostic"] is False
    assert artifact["checks"] == {}
    assert "source_artifact_present" in artifact["preconditions_checked"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4855_defensive_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4855: malformed inputs fail closed without fabricated trust."""

    assert mod._safe_suffix([]) == "audited"
    assert mod._mapping([]) == {}
    assert mod._finite_float(True) is None
    assert mod._finite_float("0.5") is None
    assert mod._finite_float(float("nan")) is None
    assert mod._computed_dominant_bucket({}) is None
    assert mod._bucket_counts({"bad": {"bucket": "MAYBE"}}) == {}
    assert mod._call_name(ast.parse("obj.method()").body[0].value.func) == "method"
    assert mod._call_name(ast.parse("(lambda: None)()").body[0].value.func) == ""
    assert mod._first_parent_call(ast.Name(id="x", ctx=ast.Load())) is None

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(not_object)

    summary, adversarial, lint = _clean_auxiliary_results()
    malformed_source_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="def nope(:",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert "a1_source_not_parseable" in malformed_source_audit["non_diagnostic_reasons"]

    missing_function_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="def unrelated(): pass",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert (
        "measure_game_with_stepwise_explorer_missing"
        in (missing_function_audit["non_diagnostic_reasons"])
    )

    no_classification_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="""
def measure_game_with_stepwise_explorer(game, winning_prefix, action_budget):
    explorer = RecordingStepwiseExplorer()
    return {"records": explorer.proposal_records}
""",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert (
        "winning_prefix_not_used_for_classification"
        in (no_classification_audit["non_diagnostic_reasons"])
    )

    bad_control = _good_a1_artifact()
    bad_control["positive_control_coverage"] = {
        "game": "other",
        "adaptered": False,
        "bucket": "COVERED",
        "reached_l1_win": False,
    }
    bad_control_audit = mod.audit_a1_artifact(
        bad_control,
        source_text=_good_a1_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert "positive_control_not_adaptered" in bad_control_audit["non_diagnostic_reasons"]
    assert "positive_control_game_mismatch" in bad_control_audit["non_diagnostic_reasons"]
    assert "positive_control_did_not_reach_l1_win" in (bad_control_audit["non_diagnostic_reasons"])

    bad_buckets = _good_a1_artifact()
    bad_buckets["n_games_measured"] = "many"
    bad_buckets["per_game_coverage"] = {"aa00": {"bucket": "MAYBE"}}
    bad_bucket_audit = mod.audit_a1_artifact(
        bad_buckets,
        source_text=_good_a1_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    for reason in (
        "invalid_per_game_bucket",
        "n_games_measured_not_integer",
        "n_games_measured_mismatch",
        "dominant_bucket_missing",
    ):
        assert reason in bad_bucket_audit["non_diagnostic_reasons"]

    tool_failed = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_a1_source(),
        summarizer_result={"returncode": 2},
        adversarial_result={"loaded": True, "flag_count": 1},
        live_lint_result=lint,
    )
    assert tool_failed["a1_genuinely_diagnostic"] is True
    assert tool_failed["checks"]["summarizer_and_adversarial_verify"]["passed"] is False

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.SOURCE_SCRIPT_RELATIVE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    script.write_text(_good_a1_source(), encoding="utf-8")
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_a1_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    artifact = mod.build_artifact(
        source_path=source,
        source_script_path=script,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"ok": True},
        duration_s=1.0,
    )
    broken = dict(artifact)
    broken.update(
        {
            "honest_verdict": "bad",
            "a1_genuinely_diagnostic": "yes",
            "proposer_blind_confirmed": "yes",
            "positive_control_confirmed": "yes",
            "buckets_match_claim": "yes",
            "field_principles": {},
            "inference_substrate": "live_llm_inference",
            "checks": [],
            "non_diagnostic_reasons": "none",
            "random_seed": 1,
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "honest_verdict_missing_terminal_prefix",
        "a1_genuinely_diagnostic_must_be_bool",
        "proposer_blind_confirmed_must_be_bool",
        "positive_control_confirmed_must_be_bool",
        "buckets_match_claim_must_be_bool",
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "checks_must_be_dict",
        "non_diagnostic_reasons_must_be_list",
        "random_seed_mismatch",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError, match="honest_verdict_missing_terminal_prefix"):
        mod.write_artifact(broken, root=tmp_path)

    report_artifact = dict(artifact)
    report_artifact["checks"] = dict(artifact["checks"], malformed=[])
    assert "malformed" not in mod.render_markdown_section(report_artifact)

    existing_root = tmp_path / "existing_report"
    existing_report = existing_root / mod.AUDIT_REPORT_RELATIVE_PATH
    existing_report.parent.mkdir(parents=True)
    existing_report.write_text("# Prior Audit\n", encoding="utf-8")
    mod.append_markdown_report(artifact, root=existing_root)
    assert "Experiment 4855 .447 A1 Generation Diagnostic Audit" in existing_report.read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(mod, "check_preconditions", lambda _root: {"ok": True})
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: adversarial)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)
    written = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.25]).__next__)
    assert written["duration_s"] == 0.25
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    blocked_no_write = mod.run(root=tmp_path / "missing_no_write", write=False)
    assert blocked_no_write["honest_verdict"] == "blocked_a1_artifact_missing"
    assert not (tmp_path / "missing_no_write" / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        mod.run(root=tmp_path, write=False, now=iter([2.0, 2.25]).__next__)
