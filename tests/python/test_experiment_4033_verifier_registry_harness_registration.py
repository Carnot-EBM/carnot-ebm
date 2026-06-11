"""Tests for Exp 4033 GAP-4 verifier registry harness registration.

Spec refs: REQ-VERIFY-4033, SCENARIO-VERIFY-4033.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import scripts.experiments.exp4033_verifier_registry_harness_registration as exp4033


REPO_ROOT = Path(__file__).parents[2]


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": "gap4_program_induction_stack",
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_commit": "HEAD",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "weights_hf": None,
                "weights_cid": None,
                "weights_sha256": None,
                "training_data_ref": "results/arc3_gap4_arc2_chain_ensemble.json",
                "label_source": "gold-posthoc",
                "eval": {
                    "arc2_gold": 19,
                    "arc2_n": 31,
                    "arc1_gold": 28,
                    "arc1_n": 31,
                },
                "agreement_role": "confidence_label_only",
                "agreement_precision_selector": False,
                "lineage": {
                    "from": "arc_grid_combined_verifier (v2)",
                    "change": (
                        "tiered policy: T1 snap tau<=0.005 > "
                        "T2 promote-first-FRESH-demo-perfect > T3 vote; "
                        "agreement = confidence label only"
                    ),
                },
                "status": "candidate",
                "notes": "GAP-4 registered module",
            }
        ]
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        "# Verifier Gaps\n\n### GAP-4: same-shape rule-application consistency\n"
        "- status: open\n",
        encoding="utf-8",
    )
    for name in (
        "arc3_gap4_arc2_chain_ensemble.json",
        "arc3_gap4_induced_programs.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _offarc_raw(
    *,
    arm_a: float,
    arm_b: float,
    ci: list[float],
    oracle: float = 1.0,
) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: fixture",
        "corpus": "mbpp_sanitized",
        "n_tasks": 30,
        "k_candidates_per_task": 8,
        "armA_vote_passrate": arm_a,
        "armA_vote_pass2": arm_a,
        "armAplusplus_aces_passrate": arm_a,
        "armAplusplus_aces_pass2": arm_a,
        "armB_demofit_passrate": arm_b,
        "armB_demofit_pass2": arm_b,
        "delta_pp": round((arm_b - arm_a) * 100.0, 4),
        "bootstrap_ci95": ci,
        "oracle_passrate": oracle,
        "model_specs": {"generator_model": "fixture"},
        "random_seed": 4032,
        "reproducibility_checksum": "fixture",
        "truncation_rate": 0.0,
        "preconditions_checked": [],
        "missing_verifier_gaps": [],
        "inference_substrate": "live_llm_inference",
    }


def test_req_4033_spec_declared() -> None:
    # REQ-VERIFY-4033: OpenSpec declares the registration harness before implementation.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4033",
        "SCENARIO-VERIFY-4033",
        "exp4033_verifier_registry_harness_registration.py",
        "offline_reeval_bitexact",
        "off_arc_outcome_recorded",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec


def test_replay_gap4_arc_numbers_from_cached_artifacts_bitexact() -> None:
    # SCENARIO-VERIFY-4033: cached ARC artifacts reproduce ARC-2 19/31 and ARC-1 28/31.
    replay = exp4033.replay_gap4_arc_numbers(REPO_ROOT)
    assert replay["offline_reeval_bitexact"] is True
    assert replay["arc2"] == {
        "gold": 19,
        "n": 31,
        "pass_at_1": pytest.approx(0.6129),
    }
    assert replay["arc1"] == {
        "covered": 28,
        "n": 31,
        "demo_perfect_coverage": pytest.approx(0.9032),
    }


def test_classify_off_arc_outcome_pending_when_final_raw_is_absent(tmp_path: Path) -> None:
    # REQ-VERIFY-4033: absent or incomplete Exp 4032 evidence records off_arc_pending.
    (tmp_path / "results").mkdir()
    outcome = exp4033.classify_off_arc_outcome(tmp_path)
    assert outcome["off_arc_outcome_recorded"] == "off_arc_pending"
    assert outcome["status"] == "off_arc_pending"


def test_classify_off_arc_outcome_routes_transfer_and_gap(tmp_path: Path) -> None:
    # REQ-VERIFY-4033: completed Exp 4032 transfer goes to registry; no-transfer goes to gaps.
    results = tmp_path / "results"
    results.mkdir()
    raw_path = results / "experiment_4032_offarc_exec_verifier_transfer_raw.json"

    raw_path.write_text(
        json.dumps(_offarc_raw(arm_a=0.2, arm_b=0.4, ci=[5.0, 35.0])), encoding="utf-8"
    )
    assert exp4033.classify_off_arc_outcome(tmp_path)["off_arc_outcome_recorded"] == (
        "code_entry_added"
    )

    raw_path.write_text(
        json.dumps(_offarc_raw(arm_a=0.4, arm_b=0.3, ci=[-20.0, 0.0])), encoding="utf-8"
    )
    assert exp4033.classify_off_arc_outcome(tmp_path)["off_arc_outcome_recorded"] == "gap_logged"


def test_ensure_ledgers_record_pending_off_arc_on_gap4_entry() -> None:
    # REQ-VERIFY-4033: pending off-ARC evidence lands in the registry map, not only an artifact.
    registry = _minimal_registry()
    gaps_text = "# Verifier Gaps\n"
    outcome = {
        "off_arc_outcome_recorded": "off_arc_pending",
        "status": "off_arc_pending",
        "reason": "missing_completed_exp4032_raw_artifact",
    }
    new_registry, new_gaps, changed = exp4033.ensure_ledgers_record_outcome(
        registry, gaps_text, outcome
    )
    entry = exp4033.find_gap4_entry(new_registry)
    assert changed is True
    assert entry["off_arc_transfer"]["outcome"] == "off_arc_pending"
    assert new_gaps == gaps_text


def test_registry_helpers_cover_missing_and_correction_branches(tmp_path: Path) -> None:
    # REQ-VERIFY-4033: missing or stale registry entries are made usable by the harness.
    assert exp4033._load_registry(tmp_path / "missing.yaml") == {"verifiers": []}
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not-a-map\n", encoding="utf-8")
    assert exp4033._load_registry(invalid) == {"verifiers": []}

    with pytest.raises(ValueError, match="not registered"):
        exp4033.find_gap4_entry({"verifiers": []})
    assert exp4033.gap4_registration_is_valid({"verifiers": []}) is False

    empty: dict[str, Any] = {"verifiers": []}
    assert exp4033.ensure_gap4_registered(empty) is True
    assert exp4033.find_gap4_entry(empty)["selection_policy"] == exp4033._selection_policy()

    stale = _minimal_registry()
    gap4 = exp4033.find_gap4_entry(stale)
    gap4["code_path"] = "wrong.py"
    assert exp4033.ensure_gap4_registered(stale) is True
    assert exp4033.find_gap4_entry(stale)["code_path"] == (
        "python/carnot/agentic/gap4_program_induction_stack.py"
    )


def test_ensure_ledgers_adds_code_entry_or_gap() -> None:
    # REQ-VERIFY-4033: transfer adds code-domain coverage; no-transfer appends a gap.
    registry = _minimal_registry()
    gaps_text = "# Verifier Gaps\n"
    transfer = {
        "off_arc_outcome_recorded": "code_entry_added",
        "status": "transfer",
        "artifact_path": "results/experiment_4032_offarc_exec_verifier_transfer_raw.json",
        "delta_pp": 12.5,
        "bootstrap_ci95": [2.0, 20.0],
        "n_tasks": 30,
    }
    new_registry, new_gaps, changed = exp4033.ensure_ledgers_record_outcome(
        registry, gaps_text, transfer
    )
    assert changed is True
    assert new_gaps == gaps_text
    assert any(v["verifier_id"] == exp4033.CODE_TRANSFER_VERIFIER_ID for v in new_registry["verifiers"])

    no_transfer = {
        "off_arc_outcome_recorded": "gap_logged",
        "status": "no_transfer",
        "artifact_path": "results/experiment_4032_offarc_exec_verifier_transfer_raw.json",
        "delta_pp": -5.0,
        "bootstrap_ci95": [-20.0, 5.0],
        "n_tasks": 30,
    }
    _, gap_gaps, gap_changed = exp4033.ensure_ledgers_record_outcome(
        _minimal_registry(), gaps_text, no_transfer
    )
    assert gap_changed is True
    assert "GAP-CODE-EXEC-DEMOFIT" in gap_gaps
    assert "missing discriminator" in gap_gaps


def test_ensure_ledgers_updates_existing_code_entry_and_rejects_unknown_outcome() -> None:
    # REQ-VERIFY-4033: code-domain entries are replaced in place and bad outcomes reject.
    registry = _minimal_registry()
    registry["verifiers"].append({"verifier_id": exp4033.CODE_TRANSFER_VERIFIER_ID, "stale": True})
    transfer = {
        "off_arc_outcome_recorded": "code_entry_added",
        "status": "transfer",
        "artifact_path": "results/experiment_4032_offarc_exec_verifier_transfer_raw.json",
        "delta_pp": 12.5,
        "bootstrap_ci95": [2.0, 20.0],
        "n_tasks": 30,
    }
    new_registry, _, changed = exp4033.ensure_ledgers_record_outcome(registry, "# gaps\n", transfer)
    assert changed is True
    code_entry = exp4033._find_verifier(new_registry, exp4033.CODE_TRANSFER_VERIFIER_ID)
    assert code_entry is not None
    assert code_entry["domain"] == "code"
    assert "stale" not in code_entry
    assert exp4033._find_verifier(new_registry, "missing") is None

    with pytest.raises(ValueError, match="unknown off_arc_outcome_recorded"):
        exp4033.ensure_ledgers_record_outcome(
            _minimal_registry(),
            "# gaps\n",
            {"off_arc_outcome_recorded": "bad", "status": "bad"},
        )


def test_classify_off_arc_pending_error_branches(tmp_path: Path) -> None:
    # REQ-VERIFY-4033: unreadable, incomplete, and blocked Exp 4032 artifacts stay pending.
    results = tmp_path / "results"
    results.mkdir()
    raw_path = results / "experiment_4032_offarc_exec_verifier_transfer_raw.json"

    raw_path.write_text("{not json", encoding="utf-8")
    unreadable = exp4033.classify_off_arc_outcome(tmp_path)
    assert unreadable["off_arc_outcome_recorded"] == "off_arc_pending"
    assert "unreadable_exp4032_artifact" in unreadable["reason"]

    raw_path.write_text(json.dumps({"honest_verdict": "complete: too_small"}), encoding="utf-8")
    incomplete = exp4033.classify_off_arc_outcome(tmp_path)
    assert incomplete["reason"] == "incomplete_exp4032_raw_artifact"

    blocked = _offarc_raw(arm_a=0.0, arm_b=0.0, ci=[0.0, 0.0], oracle=0.0)
    blocked["honest_verdict"] = "blocked_local_gguf_not_cached"
    raw_path.write_text(json.dumps(blocked), encoding="utf-8")
    pending = exp4033.classify_off_arc_outcome(tmp_path)
    assert pending["reason"] == "blocked_local_gguf_not_cached"


def test_raw_offarc_complete_rejects_incomplete_shapes() -> None:
    # REQ-VERIFY-4033: only completed full Exp 4032 raw artifacts can drive ledger outcomes.
    complete = _offarc_raw(arm_a=0.1, arm_b=0.1, ci=[0.0, 0.0])
    assert exp4033._raw_offarc_complete(complete) is True
    missing = dict(complete)
    missing.pop("delta_pp")
    assert exp4033._raw_offarc_complete(missing) is False
    bad_substrate = dict(complete, inference_substrate="aggregation_from_upstream_artifacts")
    assert exp4033._raw_offarc_complete(bad_substrate) is False
    too_few_tasks = dict(complete, n_tasks=29)
    assert exp4033._raw_offarc_complete(too_few_tasks) is False
    too_small_k = dict(complete, k_candidates_per_task=7)
    assert exp4033._raw_offarc_complete(too_small_k) is False


def test_validate_artifact_rejects_schema_poison() -> None:
    artifact = exp4033.build_artifact(
        registry_updated=True,
        offline_reeval={
            "offline_reeval_bitexact": True,
            "arc2": {"gold": 19, "n": 31, "pass_at_1": 0.6129},
            "arc1": {"covered": 28, "n": 31, "demo_perfect_coverage": 0.9032},
        },
        off_arc_outcome={
            "off_arc_outcome_recorded": "off_arc_pending",
            "status": "off_arc_pending",
            "reason": "missing_completed_exp4032_raw_artifact",
        },
        duration_s=0.1,
    )
    exp4033.validate_artifact(artifact)
    artifact["registry_updated"] = "true"
    with pytest.raises(ValueError, match="registry_updated must be a bare bool"):
        exp4033.validate_artifact(artifact)


def test_validate_artifact_rejects_other_required_schema_errors() -> None:
    # REQ-VERIFY-4033: terminal artifacts reject missing fields and invalid enum/substrate values.
    artifact = exp4033.build_artifact(
        registry_updated=True,
        offline_reeval={
            "offline_reeval_bitexact": True,
            "arc2": {"gold": 19, "n": 31, "pass_at_1": 0.6129},
            "arc1": {"covered": 28, "n": 31, "demo_perfect_coverage": 0.9032},
        },
        off_arc_outcome={
            "off_arc_outcome_recorded": "off_arc_pending",
            "status": "off_arc_pending",
            "reason": "missing_completed_exp4032_raw_artifact",
        },
        duration_s=0.1,
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required artifact field"):
        exp4033.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="done")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4033.validate_artifact(bad_verdict)

    bad_outcome = dict(artifact, off_arc_outcome_recorded="unknown")
    with pytest.raises(ValueError, match="unknown value"):
        exp4033.validate_artifact(bad_outcome)

    bad_substrate = dict(artifact, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="aggregation_from_upstream_artifacts"):
        exp4033.validate_artifact(bad_substrate)


def test_run_registration_writes_required_artifact_and_updates_registry(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4033: end-to-end cached registration writes the deliverable JSON.
    _write_minimal_repo(tmp_path)
    artifact = exp4033.run_registration(repo_root=tmp_path)
    out_path = tmp_path / "results" / "experiment_4033_verifier_registry_harness_registration.json"
    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    gap4 = exp4033.find_gap4_entry(registry)

    assert out_path.exists()
    assert artifact["honest_verdict"] == (
        "complete: gap4_stack_registered_offline_reeval_bitexact_offarc_off_arc_pending"
    )
    assert artifact["registry_updated"] is True
    assert artifact["offline_reeval_bitexact"] is True
    assert artifact["off_arc_outcome_recorded"] == "off_arc_pending"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert gap4["off_arc_transfer"]["outcome"] == "off_arc_pending"
