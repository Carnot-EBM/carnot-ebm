"""Tests for Exp 3722 convergence synthesis and operator next-thesis request.

Spec refs: REQ-REPORT-3722, SCENARIO-REPORT-3722-SYNTHESIZED,
SCENARIO-REPORT-3722-CANNOT-SYNTHESIZE.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import convergence_synthesis_operator_next_thesis_3722 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _terminal_inputs() -> dict[str, dict[str, object]]:
    return {
        "exp3712": {
            "honest_verdict": "complete: capstone_v339_paper_ready_true",
            "paper_ready": True,
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "p01_status": "honest-negative",
            "frozen_fover_headline_auroc": 0.9131,
            "frozen_headline_unchanged": True,
            "selection_diagnosis_closed": True,
            "code_native_heldout_verdict": "one_point_zero_was_a_leak",
            "shipped_detector_reconciliation": "narrowed_to_math_only_abstain",
            "facts_generalization_retired": True,
            "trained_judge_ood_retired": True,
            "kv260_terminal_status": "latency_transcript_captured_terminal_candidate",
            "fr11_v13_result": "multi_session_consolidation_transferred_no_collapse",
        },
        "exp3715": {
            "honest_verdict": (
                "complete: refreeze_disambiguation_corrigendum_clean_no_candidate_"
                "beats_frozen_headline_stays_0_9131"
            ),
            "no_candidate_beats_frozen": True,
            "frozen_headline_unchanged_assert": True,
            "adversarial_verify_clean": True,
        },
        "exp3716": {
            "honest_verdict": "complete: paper_v6_narrowing_lint_shipped",
            "g3_now_mechanically_enforced": True,
            "current_paper_lint_clean": True,
            "acceptance_gate": {"passed": True},
        },
        "exp3717": {
            "honest_verdict": "complete: g4_fully_traced_every_headline_number",
            "all_numbers_trace_to_clean_artifacts": True,
            "n_numbers_audited": 7,
            "north_star_unmodified_assert": True,
            "acceptance_gate": {"passed": True},
        },
        "exp3718": {
            "honest_verdict": (
                "complete: energy_is_a_better_selective_prediction_signal_than_"
                "entropy_deployable_abstention_gate"
            ),
            "energy_beats_baseline_abstention": True,
            "energy_aurc": 0.000789,
            "baseline_aurc": 0.075498,
            "adversarial_verify_clean": True,
        },
        "exp3719": {
            "honest_verdict": "complete: headline_discrimination_is_fover_specific",
            "generalizes_beyond_fover": False,
            "fresh_corpus_auroc": 0.798604,
            "acceptance_gate_passed": True,
            "adversarial_verify_clean": True,
        },
        "exp3720": {
            "honest_verdict": (
                "complete: fr11_v14_template_falls_back_gracefully_under_shift_"
                "no_collapse"
            ),
            "template_robust_or_graceful_fallback": True,
            "collapse_detected_deploy_arm": False,
            "template_library_bounded": True,
            "acceptance_gate": {"passed": True},
        },
        "exp3721": {
            "honest_verdict": (
                "complete: kv260_terminal_confirmed_mandate_lift_recommended_"
                "polarfire_gatemate_audited"
            ),
            "kv260_terminal_condition_confirmed": True,
            "kv260_mandate_lift_recommendation": (
                "recommend_operator_lift_per_milestone_kv260_mandate"
            ),
            "kv260_terminal_transcript_present": True,
            "speedup_claim_avoided_assert": True,
        },
    }


def _research_program_text() -> str:
    return (
        "## Product Roadmap\n"
        "### Tier B: Build Next\n"
        "| **Safety/Jailbreak Classifier** | Low = safe, high = unsafe/jailbreak |\n"
        "| **Compliance Checker** | Energy encodes regulatory constraints |\n"
    )


def _build_from_inputs(
    payloads: dict[str, dict[str, object]],
    *,
    north_hash_after: str = "north",
    manifest_hash_after: str = "manifest",
) -> dict[str, object]:
    return mod.build_artifact_from_inputs(
        exp3712=payloads["exp3712"],
        exp3715=payloads["exp3715"],
        exp3716=payloads["exp3716"],
        exp3717=payloads["exp3717"],
        exp3718=payloads["exp3718"],
        exp3719=payloads["exp3719"],
        exp3720=payloads["exp3720"],
        exp3721=payloads["exp3721"],
        north_star_text="Frozen FoVer headline AUROC: 0.9131. paper_ready=true.\n",
        research_program_text=_research_program_text(),
        roadmap_text=(
            "project_energy_selection_thesis_bounded; "
            "project_verifier_domain_bound; operator next-thesis\n"
        ),
        north_star_hash_before="north",
        north_star_hash_after=north_hash_after,
        manifest_hash_before="manifest",
        manifest_hash_after=manifest_hash_after,
        conductor_hash_before="conductor",
        conductor_hash_after="conductor",
        started_s=1.0,
        now_s=2.25,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )


def _seed_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "north-star.md").write_text(
        "Frozen FoVer headline AUROC: 0.9131. paper_ready=true.\n",
        encoding="utf-8",
    )
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired_questions: []\n",
        encoding="utf-8",
    )
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor fixture\n",
        encoding="utf-8",
    )
    (root / "research-program.md").write_text(_research_program_text(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        "project_energy_selection_thesis_bounded\n",
        encoding="utf-8",
    )
    for key, payload in _terminal_inputs().items():
        _write_json(root / mod.UPSTREAM_ARTIFACTS[key], payload)


def test_req_report_3722_spec_anchor_exists() -> None:
    """REQ-REPORT-3722: OpenSpec declares the synthesis contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3722" in spec
    assert "SCENARIO-REPORT-3722-SYNTHESIZED" in spec
    assert "SCENARIO-REPORT-3722-CANNOT-SYNTHESIZE" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


@pytest.mark.parametrize(
    ("case_name", "mutate_inputs", "expected_settled", "expected_verdict"),
    [
        pytest.param(
            "convergence_synthesized_theses_presented",
            lambda payloads: payloads,
            True,
            mod.SYNTHESIZED_VERDICT,
            id="convergence_synthesized_theses_presented",
        ),
        pytest.param(
            "cannot_synthesize",
            lambda payloads: {
                **payloads,
                "exp3712": {
                    **payloads["exp3712"],
                    "paper_ready": False,
                    "selection_diagnosis_closed": False,
                },
            },
            False,
            mod.CANNOT_SYNTHESIZE_VERDICT,
            id="cannot_synthesize",
        ),
    ],
)
def test_scenario_report_3722_parametrized_honest_outcomes(
    case_name: str,
    mutate_inputs,
    expected_settled: bool,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3722: synthesized and blocked outcomes stay distinct."""
    payloads = mutate_inputs(_terminal_inputs())

    artifact = _build_from_inputs(payloads)

    mod.validate_artifact(artifact)
    assert case_name in {
        "convergence_synthesized_theses_presented",
        "cannot_synthesize",
    }
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["all_self_generable_threads_settled"] is expected_settled
    assert type(artifact["all_self_generable_threads_settled"]) is bool
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert len(artifact["converged_state_ledger"]) == len(mod.REQUIRED_THREADS)
    assert {
        row["thread"] for row in artifact["converged_state_ledger"]
    } == set(mod.REQUIRED_THREADS)
    assert all(
        {"thread", "terminal_status", "authoritative_artifact", "settled"} <= set(row)
        for row in artifact["converged_state_ledger"]
    )
    assert all(
        {"thesis", "why_not_regrind", "authority", "operator_action_required"} <= set(row)
        for row in artifact["candidate_next_theses"]
    )
    assert any(
        row["thesis"] == "energy_based_selective_prediction_at_scale"
        for row in artifact["candidate_next_theses"]
    )
    assert any(
        row["thesis"] == "tier_b_safety_jailbreak_classifier"
        for row in artifact["candidate_next_theses"]
    )
    assert "Which thesis" in artifact["operator_decision_request"]
    assert artifact["recommended_default_thesis"]["operator_decision_made"] is False
    assert artifact["paper_ready_status"] is payloads["exp3712"]["paper_ready"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_report_3722_write_artifact_preserves_operator_curated_files(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3722: writing persists clean synthesis without protected edits."""
    _seed_repo(tmp_path)
    before_north = (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8")
    before_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    )
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )

    output = mod.write_artifact(
        tmp_path,
        output_path="results/exp3722.json",
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SYNTHESIZED_VERDICT
    assert artifact["all_self_generable_threads_settled"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["flag_count"] == 0
    assert artifact["north_star_unmodified_assert"] is True
    assert artifact["manifest_unmodified_assert"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == before_north
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    ) == before_manifest
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded

    built = mod.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=1.0,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )
    mod.validate_artifact(built)
    assert built["honest_verdict"] == mod.SYNTHESIZED_VERDICT


def test_req_report_3722_validation_and_helper_edges() -> None:
    """REQ-REPORT-3722: validation rejects schema drift and decisive overclaim."""
    artifact = _build_from_inputs(
        _terminal_inputs(),
        north_hash_after="changed",
        manifest_hash_after="changed",
    )
    mod.validate_artifact(artifact)
    assert artifact["north_star_unmodified_assert"] is False
    assert artifact["manifest_unmodified_assert"] is False
    assert mod.adversarial_report_is_clean({"flags": [{"severity": "warn"}]}) is True
    assert mod.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert mod.adversarial_report_is_clean({"flags": "not-list"}) is False
    assert mod.compact_adversarial_report({"flags": [{"severity": "warn"}, "bad"]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }

    missing = dict(artifact)
    missing.pop("candidate_next_theses")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        mod.validate_artifact(bad_verdict)

    bad_bool = dict(artifact, all_self_generable_threads_settled={"value": True})
    with pytest.raises(ValueError, match="all_self_generable_threads_settled"):
        mod.validate_artifact(bad_bool)

    bad_candidate = dict(artifact, candidate_next_theses=[{"thesis": "x"}])
    with pytest.raises(ValueError, match="why_not_regrind"):
        mod.validate_artifact(bad_candidate)

    bad_default = dict(
        artifact,
        recommended_default_thesis={
            **artifact["recommended_default_thesis"],
            "operator_decision_made": True,
        },
    )
    with pytest.raises(ValueError, match="operator decision"):
        mod.validate_artifact(bad_default)

    bad_marker = dict(artifact, forbidden="GGUF")
    with pytest.raises(ValueError, match="compute-bound markers"):
        mod.validate_artifact(bad_marker)

    bad_checksum = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_report_3722_optional_candidates_and_reader_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3722: optional menu and defensive readers stay bounded."""
    payloads = _terminal_inputs()
    artifact = mod.build_artifact_from_inputs(
        exp3712=payloads["exp3712"],
        exp3715=payloads["exp3715"],
        exp3716=payloads["exp3716"],
        exp3717=payloads["exp3717"],
        exp3718={**payloads["exp3718"], "energy_beats_baseline_abstention": False},
        exp3719=payloads["exp3719"],
        exp3720=payloads["exp3720"],
        exp3721=payloads["exp3721"],
        north_star_text="Frozen FoVer headline AUROC: 0.9131.\n",
        research_program_text=_research_program_text(),
        roadmap_text="energy-as-GENERATOR Energy-Based Transformer operator seed\n",
        north_star_hash_before="north",
        north_star_hash_after="north",
        manifest_hash_before="manifest",
        manifest_hash_after="manifest",
        conductor_hash_before="conductor",
        conductor_hash_after="conductor",
        started_s=0.0,
        now_s=1.0,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )

    mod.validate_artifact(artifact)
    assert artifact["recommended_default_thesis"]["thesis"] == "finalize_submit_and_maintenance"
    assert any(
        row["thesis"] == "human_seeded_energy_as_generator_ebt"
        for row in artifact["candidate_next_theses"]
    )
    assert mod._read_json_object(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert mod._read_json_object(invalid) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json_object(list_json) == {}
    assert mod._read_text(tmp_path / "absent.txt") == ""
    assert len(mod._sha256_path(tmp_path / "absent.txt")) == 64
    assert mod._point({"point": 0.1234567}) == 0.123457
    assert mod._point("not-a-number") is None


def test_req_report_3722_adversarial_loader_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3722: verifier import and return-shape failures are explicit."""

    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(RuntimeError, match="adversarial verifier"):
        mod.run_adversarial_verify_report(Path("artifact.json"))

    class NonObjectReportLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module) -> None:
            module.verify_artifact = lambda path: []

    monkeypatch.setattr(
        mod.importlib.util,
        "spec_from_file_location",
        lambda *args: ModuleSpec("fake_adversarial_verify", NonObjectReportLoader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        mod.run_adversarial_verify_report(Path("artifact.json"))
