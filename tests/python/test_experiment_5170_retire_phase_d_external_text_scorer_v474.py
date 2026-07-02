"""Tests for Exp 5170 Phase D external-text-scorer retirement.

Spec refs: REQ-REPORT-5170, SCENARIO-REPORT-5170-LINT,
SCENARIO-REPORT-5170-NARROW-SCOPE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5170_retire_phase_d_external_text_scorer_v474 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_with_entry(entry: dict | None = None) -> dict:
    return {
        "retired": [],
        "retired_experiments": [],
        "retired_extras": [entry or mod.expected_manifest_entry()],
    }


def _roadmap_yaml(*, include_match: bool = False) -> str:
    tasks = [
        {
            "id": "exp5170-retire-phase-d-external-text-scorer-v474",
            "title": "PHASE 0 retire the scoped Phase D external scorer lineage",
            "prompt": "Record the retirement and prove the manifest rule is narrow.",
        },
        {
            "id": "exp5178-hidden-state-verifier-pilot-v474",
            "title": "TrajSelector-style hidden-state verifier pilot",
            "prompt": (
                "This tests internal representation scoring from the generator's own "
                "hidden states. It is not an external generated-text/logprob scorer."
            ),
        },
    ]
    if include_match:
        tasks.append(
            {
                "id": "exp9999-lora-ebm-rerun",
                "title": "train lora ebm scorer v2 on off-arc reasoning corpus",
                "prompt": "Repeat the external text scorer against self-consistency.",
            }
        )
    return yaml.safe_dump({"milestone": "2026.07.474", "tasks": tasks}, sort_keys=False)


def make_repo(
    tmp_path: Path,
    *,
    manifest_entry: dict | None = None,
    retirement_note: bool = True,
    roadmap_match: bool = False,
) -> Path:
    root = tmp_path
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_manifest_with_entry(manifest_entry), sort_keys=False),
        encoding="utf-8",
    )
    note = (
        mod.SANCTIONED_EXCEPTION_DOC_MARKER
        + "\nexternal generated-text/logprob scorer retired; hidden-state/internal-representation "
        "verifier TrajSelector and VerifySteer remain open."
        if retirement_note
        else "no retirement note"
    )
    (root / "ops" / "verifier_gaps.md").write_text(note + "\n", encoding="utf-8")
    (root / "ops" / "known-issues.md").write_text("known issues\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _roadmap_yaml(include_match=roadmap_match),
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "TrajSelector and VerifySteer are hidden-state/internal-representation exceptions.\n",
        encoding="utf-8",
    )
    payloads = {
        "exp4940": {
            "honest_verdict": "success_distributional_energy_verifier_pivot_executable_spec_ready",
            "moat_proven_claimed": False,
        },
        "distributional_energy_verifier_musr": {
            "honest_verdict": "complete_musr_energy_verifier_no_win",
            "self_consistency_accuracy": 0.56,
            "energy_minus_sc_delta": -0.045,
            "energy_minus_sc_ci95": [-0.105, 0.015],
        },
        "exp5031": {
            "honest_verdict": "complete_lora_ebm_no_win_musr_plus_0p080_ci_incl_0",
            "delta_vs_tuned_sc": 0.08,
            "paired_ci95": [0.0, 0.165],
        },
        "exp5032": {
            "honest_verdict": "complete_uprm_no_win_musr_minus_0p110_mcnemar_or_headroom_gate",
            "delta_vs_tuned_sc": -0.11,
        },
        "exp5033": {
            "honest_verdict": "complete_ebrm_no_win_musr_plus_0p080_ci_incl_0",
            "delta_vs_tuned_sc": 0.08,
        },
        "exp5126": {
            "honest_verdict": "complete_distributional_energy_ranker_evaluated_not_ready_for_audit",
            "distributional_energy_delta": 0.0,
            "flagged_adversarial": True,
        },
        "exp5163": {
            "honest_verdict": {
                "value": "complete_mmlu_pro_fewshot_verifier_vs_cheap_delta_+0.025_CI95_[-0.125,0.175]_CI_includes_0"
            },
            "verifier_vs_cheap_delta": {"value": 0.025},
            "verifier_vs_cheap_delta_ci95": {"value": [-0.125, 0.175]},
            "flagged_adversarial": True,
        },
    }
    for source_id, path in mod.SOURCE_RESULT_PATHS.items():
        _write_json(
            root / path,
            payloads.get(
                source_id,
                {
                    "honest_verdict": f"complete_{source_id}_phase_d_context",
                    "delta_vs_tuned_sc": None,
                },
            ),
        )
    return root


def test_req_report_5170_spec_declares_external_text_retirement_contract() -> None:
    """REQ-REPORT-5170: OpenSpec anchors the scoped retirement contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5170",
        "SCENARIO-REPORT-5170-LINT",
        "SCENARIO-REPORT-5170-NARROW-SCOPE",
        mod.ENTRY_ID,
        "results/experiment_5170_retire_phase_d_external_text_scorer_v474.json",
        "BLOCKED_PATTERN_MATCHED",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_req_report_5170_manifest_entry_doc_note_and_artifact_are_valid(tmp_path: Path) -> None:
    """REQ-REPORT-5170: manifest entry, exception note, and artifact fields are valid."""

    root = make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["exclusion_manifest_entry_added"] is True
    assert artifact["entry_id"] == mod.ENTRY_ID
    assert artifact["false_positive_check_against_exp5178"] is True
    assert artifact["synthetic_match_check_passed"] is True
    assert artifact["sanctioned_exception_documented"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert len(artifact["phase_d_artifacts_enumerated"]) >= 7
    assert "exp5163" in artifact["phase_d_artifacts_enumerated"]
    assert artifact["lineage_stage_summary"]["cleanest_point_estimate"]["source_id"] == "exp5031"
    assert (
        artifact["source_artifact_summary"]["distributional_energy_verifier_musr"][
            "energy_minus_sc_delta"
        ]
        == -0.045
    )


def test_scenario_report_5170_lint_blocks_synthetic_but_not_exp5178(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5170-LINT and NARROW-SCOPE: linter fires only on retired scope."""

    root = make_repo(tmp_path)

    current = mod.check_current_roadmap_false_positive(root)
    synthetic = mod.check_synthetic_match(root)

    assert current["passed"] is True
    assert current["exp5178_task_ids"] == ["exp5178-hidden-state-verifier-pilot-v474"]
    assert current["exp5178_entry_risks"] == []
    assert synthetic["passed"] is True
    assert synthetic["matched_risks"][0]["violation_class"] == "BLOCKED_PATTERN_MATCHED"
    assert synthetic["matched_risks"][0]["severity"] == "HARD"
    assert mod.ENTRY_ID in synthetic["matched_risks"][0]["detail"]


def test_req_report_5170_validation_edges_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5170: validation fails closed and CLI writes the artifact."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path / "valid"),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("phase_d_artifacts_enumerated", ["exp5163"]),
        ("exclusion_manifest_entry_added", False),
        ("entry_id", "wrong"),
        ("false_positive_check_against_exp5178", False),
        ("synthetic_match_check_passed", False),
        ("sanctioned_exception_documented", False),
        ("honest_verdict", "blocked_bad"),
        ("inference_substrate", "metadata_and_source_audit"),
        ("duration_s", 0.0),
        ("tests_run", []),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(valid)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload.pop("tests_run")
    with pytest.raises(ValueError, match="invalid Exp 5170 retirement artifact"):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload["field_principles"]["entry_id"] = "wrong"
    with pytest.raises(ValueError, match="invalid Exp 5170 retirement artifact"):
        mod.validate_artifact(payload)

    assert mod._load_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod._number_value(True) is None
    assert mod._number_value("not-a-number") is None
    no_entry_root = tmp_path / "no_entry"
    (no_entry_root / "ops").mkdir(parents=True)
    (no_entry_root / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump({"retired_extras": []}),
        encoding="utf-8",
    )
    assert mod._find_manifest_entry(no_entry_root) == {}

    with monkeypatch.context() as patch:
        patch.setattr(mod.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError):
            mod._load_linter_module()

    missing = mod.build_artifact(
        root=make_repo(tmp_path / "missing_note", retirement_note=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    assert missing["sanctioned_exception_documented"] is False
    assert missing["honest_verdict"] == mod.INCOMPLETE_VERDICT

    bad_entry = mod.expected_manifest_entry()
    bad_entry["blocked_patterns"] = []
    bad = mod.build_artifact(
        root=make_repo(tmp_path / "bad_entry", manifest_entry=bad_entry),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    assert bad["exclusion_manifest_entry_added"] is False
    assert bad["synthetic_match_check_passed"] is False
    assert bad["honest_verdict"] == mod.INCOMPLETE_VERDICT

    root = make_repo(tmp_path / "cli_repo")
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n",
        encoding="utf-8",
    )
    output = root / "module_cli_result.json"
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260702"]) == 0
    assert output.exists()
