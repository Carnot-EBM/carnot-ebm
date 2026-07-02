"""Tests for Exp 5140 symbolic KAN certificate distillation.

Spec refs: REQ-KAN-5140, SCENARIO-KAN-5140.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5140_symbolic_kan_certificate_distillation_v471 as mod
from scripts import experiment_5140_symbolic_kan_certificate_distillation_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
UPSTREAM_PATH = REPO / mod.EXP5128_RESULT_RELATIVE_PATH


def _load_repo_upstream() -> dict[str, object]:
    return json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))


def _write_upstream(root: Path, payload: dict[str, object] | None = None) -> None:
    upstream_path = root / mod.EXP5128_RESULT_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text(
        json.dumps(payload or _load_repo_upstream(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_req_kan_5140_spec_declares_distillation_contract() -> None:
    """REQ-KAN-5140: OpenSpec anchors symbolic distillation and controls."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("## REQ-KAN-5140")
    end = spec.index("## Implementation Status", start)
    section = spec[start:end]

    assert "SCENARIO-KAN-5140" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.MILESTONE in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "LLM judge" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5140_distills_primitives_and_reconstructs_exactly() -> None:
    """SCENARIO-KAN-5140: primitive rules reconstruct source certificates exactly."""

    upstream = _load_repo_upstream()
    rules = mod.distill_symbolic_rules(upstream)
    cycle = mod.cycle_check_rules(rules)
    primitive_names = {
        primitive["name"]
        for rule in rules
        for primitive in rule["primitives"]
    }

    assert len(rules) == len(upstream["certificates_emitted"])
    assert primitive_names == set(mod.SYMBOLIC_PRIMITIVE_NAMES)
    assert cycle["symbolic_equivalence_rate"] == pytest.approx(1.0)
    assert cycle["cycle_reconstruction_rate"] == pytest.approx(1.0)
    assert cycle["certificate_soundness"] is True
    for rule in rules:
        reconstructed = mod.reconstruct_from_rule(rule)
        assert reconstructed == rule["source_metadata"]
        assert mod.soundness_check_reconstruction(rule, reconstructed) is True
        assert "abstain_when" in rule["cycle_predicates"]


def test_req_kan_5140_controls_detect_false_shuffle_margin_and_holdout() -> None:
    """REQ-KAN-5140: false, label-shuffle, near-margin, and holdout controls run."""

    rules = mod.distill_symbolic_rules(_load_repo_upstream())
    controls = mod.evaluate_controls(rules)

    assert controls["false_property_detected"] is True
    assert controls["label_shuffle_control"]["detected"] is True
    assert controls["near_margin_abstention_rate"] == pytest.approx(1.0)
    assert controls["family_holdout_results"]
    assert all(item["certificate_soundness"] for item in controls["family_holdout_results"])
    assert all(
        item["symbolic_equivalence_rate"] == pytest.approx(1.0)
        for item in controls["family_holdout_results"]
    )

    shuffled = mod.label_shuffle_control(rules)
    assert shuffled["detected"] is True
    assert shuffled["mismatch_count"] > 0


def test_req_kan_5140_artifact_reports_required_fields_and_writes_rules(tmp_path: Path) -> None:
    """REQ-KAN-5140: ready artifact and distilled-rules artifact are schema-valid."""

    _write_upstream(tmp_path)
    artifact = mod.write_outputs(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    rules_payload = json.loads(
        (tmp_path / mod.DISTILLED_RULES_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert payload == artifact
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(0.25)
    assert artifact["exp5128_loaded"] is True
    assert artifact["distilled_rules_path"] == mod.DISTILLED_RULES_RELATIVE_PATH
    assert artifact["symbolic_equivalence_rate"] == pytest.approx(1.0)
    assert artifact["certificate_soundness"] is True
    assert artifact["cycle_reconstruction_rate"] == pytest.approx(1.0)
    assert artifact["false_property_detected"] is True
    assert artifact["near_margin_abstention_rate"] == pytest.approx(1.0)
    assert artifact["symbolic_kan_ready"] is True
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert rules_payload["experiment_id"] == mod.EXPERIMENT_ID
    assert len(rules_payload["distilled_rules"]) == len(artifact["distilled_rule_summaries"])
    assert rules_payload["llm_judge_used"] is False


def test_req_kan_5140_missing_or_dirty_upstream_blocks_honestly(tmp_path: Path) -> None:
    """REQ-KAN-5140: absent or incomplete Exp 5128 evidence fails closed."""

    assert mod._repo_root() == REPO
    assert mod._relative_or_absolute(Path("/definitely/outside/carnot.json"), tmp_path).startswith(
        "/definitely/outside"
    )

    missing = mod.write_outputs(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.1,
    )
    mod.validate_artifact(missing)
    assert missing["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert missing["exp5128_loaded"] is False
    assert missing["distilled_rules_path"] is None
    assert missing["symbolic_kan_ready"] is False
    assert not (tmp_path / mod.DISTILLED_RULES_RELATIVE_PATH).exists()

    incomplete = dict(_load_repo_upstream())
    incomplete["explanation_records"] = []
    _write_upstream(tmp_path, incomplete)
    blocked = mod.write_outputs(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.2,
    )
    assert blocked["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert blocked["exp5128_loaded"] is False
    assert "missing certificate/explanation records" in blocked["upstream_status"]["reason"]

    upstream_path = tmp_path / mod.EXP5128_RESULT_RELATIVE_PATH
    upstream_path.write_text("[]\n", encoding="utf-8")
    non_object = mod.write_outputs(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.21,
    )
    assert "artifact is not a JSON object" in non_object["upstream_status"]["reason"]

    not_ready_payload = dict(_load_repo_upstream())
    not_ready_payload["kan_certificate_breadth_ready"] = False
    _write_upstream(tmp_path, not_ready_payload)
    not_ready_upstream = mod.write_outputs(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.22,
    )
    assert "not ready" in not_ready_upstream["upstream_status"]["reason"]


def test_req_kan_5140_validation_edges_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-5140: validation catches drift and the CLI writes the deliverable."""

    _write_upstream(tmp_path)
    rules = mod.distill_symbolic_rules(_load_repo_upstream())
    tampered_rule = copy.deepcopy(rules[0])
    tampered_rule["source_metadata"]["verdict"] = "counterexample"
    assert mod.soundness_check_reconstruction(
        tampered_rule,
        mod.reconstruct_from_rule(tampered_rule),
    ) is False

    refinement_counterexample = copy.deepcopy(
        next(rule for rule in rules if rule["rule_kind"] == "refinement_budget")
    )
    refinement_counterexample["numeric"]["observed"] = (
        refinement_counterexample["numeric"]["threshold"] + 1.0
    )
    assert mod.reconstruct_from_rule(refinement_counterexample)["verdict"] == "counterexample"

    unknown_verdict = copy.deepcopy(rules[0])
    unknown_verdict["source_metadata"] = mod.reconstruct_from_rule(unknown_verdict)
    unknown_verdict["source_metadata"]["verdict"] = "mystery"
    assert mod.soundness_check_reconstruction(
        unknown_verdict,
        unknown_verdict["source_metadata"],
    ) is False

    bad = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.3,
    )
    bad["honest_verdict"] = "ready_without_prefix"
    with pytest.raises(ValueError, match="bad verdict prefix"):
        mod.validate_artifact(bad)

    monkeypatch.setattr(
        mod,
        "evaluate_controls",
        lambda _rules: {
            "false_property_detected": False,
            "near_margin_abstention_rate": 1.0,
            "label_shuffle_control": {"detected": True, "mismatch_count": 1},
            "family_holdout_results": [],
        },
    )
    not_ready = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.4,
    )
    mod.validate_artifact(not_ready)
    assert not_ready["honest_verdict"] == mod.COMPLETE_NOT_READY_VERDICT
    assert not_ready["symbolic_kan_ready"] is False

    output = tmp_path / "cli-result.json"
    rules_output = tmp_path / "cli-rules.json"
    assert (
        script_mod.main(
            [
                "--date",
                "20260702",
                "--root",
                str(tmp_path),
                "--output",
                str(output),
                "--rules-output",
                str(rules_output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
    assert json.loads(rules_output.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
