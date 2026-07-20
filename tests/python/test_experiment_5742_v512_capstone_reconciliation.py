"""Tests for the Exp5742 V512 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5742, SCENARIO-CAPSTONE-5742,
SCENARIO-CAPSTONE-5742-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5742-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5742_v512_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5731_TRANSITION_PATH: {
            "schema": "carnot.experiment_5731.transition_v512.v1",
            "experiment_id": "exp5731-transition-v512",
            "honest_verdict": "complete: v512 transition archived terminal .511 evidence",
            "current_task_range": "exp5731-exp5742",
            "arc_registry_delta": 0,
            "preserved_scopes": [{"scope": "finite_choice_proposal_channel"}],
        },
        mod.EXP5732_SOURCE_PATH: {
            "schema": "carnot.experiment_5732.v512_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5732-v512-source-delta-ingestion",
            "honest_verdict": "complete: accepted 3 non-duplicate actionable V512 source deltas",
            "flagged_adversarial": True,
            "benchmark_compute_claimed": False,
        },
        mod.EXP5733_PROPOSAL_PATH: {
            "schema": "carnot.experiment_5733.sota_finite_choice_proposal_channel.v1",
            "experiment_id": "experiment_5733_sota_finite_choice_proposal_channel",
            "honest_verdict": "complete: sealed_finite_choice_proposal_channel_qualified",
            "proposal_channel_ready_score": 1.0,
            "qualified_flagship_model_count": 2,
            "cuda_offload_authenticated_score": 1.0,
            "receipt_failure_count": 0,
            "validator_disagreement_count": 0,
            "qualified_model_ids": ["qwen", "gemma"],
            "freeform_generation_used": False,
            "grammar_runtime_used": False,
            "external_scorer_used": False,
            "retired_runtime_used": False,
        },
        mod.EXP5734_STREAM_PATH: {
            "schema": "carnot.experiment_5734.sota_exact_proposal_stream.v1",
            "experiment_id": "experiment_5734_sota_exact_proposal_stream",
            "honest_verdict": "complete: sealed_chronological_sota_exact_proposal_stream_ready",
            "sota_proposal_stream_ready_score": 1.0,
            "missing_row_count": 0,
            "non_finite_score_count": 0,
            "label_collision_count": 0,
            "validator_disagreement_count": 0,
            "stream_root_commitment": "sha256:" + "a" * 64,
            "prospective_prefix_hash": "sha256:" + "b" * 64,
            "sealed_suffix_hash": "sha256:" + "c" * 64,
            "model_weight_mutation": False,
        },
        mod.EXP5735_ZERO_GATE_PATH: {
            "schema": "carnot.experiment_5735.zero_gate_kan_continuous_self_learning.v1",
            "experiment_id": "experiment_5735_zero_gate_kan_continuous_self_learning",
            "honest_verdict": "complete: zero_gated_residual_spline_kan_csl_ready",
            "zero_gate_csl_ready_score": 1.0,
            "function_preserving_insertion_score": 1.0,
            "suffix_improvement": 0.12,
            "prefix_retention_delta": 0.0,
            "unsafe_update_count": 0,
            "model_weight_mutation": False,
            "production_default_enabled": False,
        },
        mod.EXP5736_LIFECYCLE_PATH: {
            "schema": "carnot.experiment_5736.csl_lifecycle_conflict_rollback.v1",
            "experiment_id": "experiment_5736_csl_lifecycle_conflict_rollback",
            "honest_verdict": "complete: csl_lifecycle_conflict_rollback_ready",
            "csl_lifecycle_ready_score": 1.0,
            "unsafe_propagation_count": 0,
            "rollback_state_hash_matches": True,
            "ledger_replay_equivalence": {"passed": True},
            "model_weight_mutation": False,
            "production_default_enabled": False,
        },
        mod.EXP5737_INGRESS_PATH: {
            "schema": "carnot.experiment_5737.sota_stream_csl_shadow_ingress.v1",
            "experiment_id": "experiment_5737_sota_stream_csl_shadow_ingress",
            "honest_verdict": "complete: sota_stream_csl_shadow_ingress_ready",
            "sota_csl_ingress_ready_score": 1.0,
            "unsafe_update_count": 0,
            "rollback_state_hash_matches": True,
            "model_weight_mutation": False,
            "production_default_enabled": False,
        },
        mod.EXP5738_BATCH_PATH: {
            "schema": "carnot.experiment_5738.one_axis_rust_batched_backend.v1",
            "experiment_id": "exp5738-one-axis-rust-batched-backend",
            "honest_verdict": "complete: one-axis sample_batch backend is ready",
            "batch_backend_ready_score": 1.0,
            "energy_trace_mismatch_count": 0,
            "proposal_mismatch_count": 0,
            "exchange_mismatch_count": 0,
            "checkpoint_mismatch_count": 0,
            "restart_mismatch_count": 0,
            "result_order_mismatch_count": 0,
            "timing_claimed": False,
            "software_speedup_claimed": False,
            "hardware_speedup_claimed": False,
            "fpga_or_tsu_used": False,
        },
        mod.EXP5739_10X_PATH: {
            "schema": "carnot.experiment_5739.one_axis_batched_10x_crossover.v1",
            "experiment_id": "exp5739-one-axis-batched-10x-crossover",
            "honest_verdict": "complete: terminal null; matched batched Rust/Python CPU evidence did not prove the strict consecutive larger-size 10x lower-bound rule",
            "rust_batched_10x_ready_score": 0.0,
            "quality_matched_pair_count": 728,
            "qualified_10x_sizes": [],
            "qualified_10x_thread_regime": None,
            "timing_claimed": True,
            "software_speedup_claimed": False,
            "gpu_speedup_claimed": False,
            "hardware_speedup_claimed": False,
            "fpga_or_tsu_used": False,
        },
        mod.EXP5740_ARC_CAUSAL_PATH: {
            "honest_verdict": "complete: game_blind_primitive_causal_audit_positive_count_7_no_policy_or_registry_credit",
            "positive_causal_primitive_count": 7,
            "source_leak_count": 1,
            "game_identity_leak_count": 2,
            "policy_modified": False,
            "registry_modified": False,
            "solve_provenance": "development_proxy",
            "verifier_is_oracle": True,
            "counterfactual_receipt_coverage": {
                "candidate_count": 7,
                "meets_minimum_n": True,
                "paired_replay_count": 20759,
            },
        },
        mod.EXP5741_ARC_LIVE_PATH: {
            "schema": "blocked_gate_check_v1",
            "experiment": 5741,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "3 of 4 gate(s) failed",
            "gates_evaluated": [
                {"artifact_field": "positive_causal_primitive_count", "passed": True},
                {"artifact_field": "source_leak_count", "passed": False},
                {"artifact_field": "game_identity_leak_count", "passed": False},
            ],
            "blocked_at_layer": "conductor_pre_gate",
        },
    }


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    ids = list(mod.EXPECTED_TASK_IDS) + [mod.EXP5742_TASK_ID]
    for task_id in ids:
        row: JsonDict = {"id": task_id, "title": task_id}
        if task_id == mod.EXP5739_TASK_ID:
            row["prior_failures"] = [
                {
                    "experiment_id": "exp5724-one-axis-rust-python-matched-crossover",
                    "verdict": "complete: terminal null; no consecutive larger-size matched-quality Rust/Python CPU crossover proven; timing claimed without GPU, FPGA, TSU, or hardware claim",
                    "retire_if_same_verdict": True,
                }
            ]
        if task_id == mod.EXP5741_TASK_ID:
            row["gated_on"] = [
                {
                    "upstream": mod.EXP5740_TASK_ID,
                    "artifact_field": "source_leak_count",
                    "op": "==",
                    "value": 0,
                }
            ]
        tasks.append(row)
    return {
        "milestone": "2026.07.512",
        "milestone_doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "tasks": tasks,
    }


def _make_root(root: Path, *, omit: Path | None = None, malformed: Path | None = None) -> None:
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-07-20 01:48 UTC | Transition terminal .511 evidence | OK | tests |",
                "| 2026-07-20 02:19 UTC | Ingest post-V512 source deltas | FLAGGED | DURATION_TOO_SHORT |",
                "| 2026-07-20 02:44 UTC | Qualify a finite-choice proposal channel | OK | tests |",
                "| 2026-07-20 03:09 UTC | Gated on Exp5733 readiness: build a sealed exact-a | OK | tests |",
                "| 2026-07-20 03:32 UTC | Run non-cascading zero-gated KAN continuous self-l | OK | tests |",
                "| 2026-07-20 03:53 UTC | Gated on Exp5735 safety: exercise typed CSL lifecy | OK | tests |",
                "| 2026-07-20 04:10 UTC | Gated on Exp5734 and Exp5736: admit the sealed SOT | OK | tests |",
                "| 2026-07-20 04:35 UTC | Profile the large-size Rust reversal and add a par | OK | tests |",
                "| 2026-07-20 04:55 UTC | Gated on Exp5738 parity: measure a matched batched | OK | tests |",
                "| 2026-07-20 05:12 UTC | Audit game-blind ARC action-effect primitives with | OK | tests |",
                "| 2026-07-20 05:18 UTC | Gated on Exp5740 causal utility: harden one generi | GATE_BLOCK | 3 failed |",
            ]
        )
        + "\n",
    )
    roadmap_path = root / mod.ROADMAP_RELATIVE_PATH
    roadmap_path.parent.mkdir(parents=True, exist_ok=True)
    roadmap_path.write_text(yaml.safe_dump(_roadmap_payload(), sort_keys=False), encoding="utf-8")
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "Task range: `exp5731`-`exp5742`\n")
    _write_text(root, Path("AGENTS.md"), "AGENTS\n")
    _write_text(root, Path("CODEX.md"), "CODEX\n")
    _write_text(root, Path("CLAUDE.md"), "CLAUDE\n")
    _write_text(root, Path("research-program.md"), "program\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.E2E_PLAN_RELATIVE_PATH, "E2E\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor\n")
    _write_text(root, Path("ops/status.md"), "status\n")
    _write_text(root, Path("ops/changelog.md"), "changelog\n")
    _write_text(root, Path("_bmad/traceability.md"), "traceability\n")
    manifest_path = root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump({"retired_extras": []}, sort_keys=False), encoding="utf-8"
    )
    registry_path = root / mod.ARC_REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump({"reproducible_total_levels": 183, "reproducible_total_games": 25}),
        encoding="utf-8",
    )
    spec_path = root / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_capstone_5742_spec_declares_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5742: OpenSpec declares the V512 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5742") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "artifact_reconciliation_only" in section
    assert "arc_registry_delta=0" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5742_live_repo_reconciles_independent_v512_branches() -> None:
    """SCENARIO-CAPSTONE-5742: live repo evidence is reconciled without over-credit."""

    artifact = mod.run_capstone(
        root=REPO,
        validation_results=[{"command": "focused", "exit_code": 0, "status": "passed"}],
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["task_statuses"][mod.EXP5732_TASK_ID] == "flagged"
    assert artifact["task_statuses"][mod.EXP5741_TASK_ID] == "gate_skipped"
    assert mod.EXP5732_SOURCE_PATH.as_posix() in artifact["flagged_artifacts"]
    assert mod.EXP5741_TASK_ID in artifact["gate_skip_receipts"]
    assert (
        artifact["preconditions_checked"]["source_files"]["research-roadmap-next.yaml"]["exists"]
        is False
    )

    assert artifact["proposal_channel_ready"] is True
    assert artifact["sota_proposal_stream_ready"] is True
    assert artifact["zero_gate_csl_ready"] is True
    assert artifact["csl_lifecycle_ready"] is True
    assert artifact["sota_csl_ingress_ready"] is True
    assert artifact["batch_backend_ready"] is True
    assert artifact["rust_batched_10x_ready"] is False
    assert artifact["arc_causal_primitive_ready"] is False
    assert artifact["arc_generic_primitive_live_ready"] is False
    assert artifact["continuous_self_learning_credited"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False

    assert artifact["arc_registry_count_before"] == 183
    assert artifact["arc_registry_count_after"] == 183
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
    assert (
        artifact["solve_provenance_summary"][mod.EXP5740_TASK_ID]["solve_provenance"]
        == "development_proxy"
    )
    assert (
        artifact["solve_provenance_summary"][mod.EXP5740_TASK_ID]["development_proxy_positive"]
        is True
    )
    assert artifact["solve_provenance_summary"][mod.EXP5741_TASK_ID]["status"] == "gate_skipped"

    assert artifact["retirements_required"] == []
    assert artifact["retirements_applied"] == []
    preserved = {row["scope"] for row in artifact["preserved_scopes"]}
    assert {
        "finite_choice_proposal_channel",
        "exact_sota_proposal_stream",
        "zero_gated_kan_csl",
        "typed_csl_lifecycle",
        "sota_csl_shadow_ingress",
        "batched_samplerbackend_contract",
        "batched_10x_timing_null_evidence",
        "arc_causal_primitive_development_proxy",
        "arc_live_attempts",
    } <= preserved
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["timing_claimed"] is True
    assert artifact["software_speedup_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["spec_files_updated"] == [mod.SPEC_RELATIVE_PATH.as_posix()]
    assert all(row["updated"] is False for row in artifact["ops_files_updated"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_capstone_5742_missing_and_malformed_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5742-MISSING-MALFORMED: bad inputs block dependent claims."""

    _make_root(tmp_path, omit=mod.EXP5733_PROPOSAL_PATH, malformed=mod.EXP5734_STREAM_PATH)
    artifact = mod.run_capstone(root=tmp_path)

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5733_PROPOSAL_PATH.as_posix() in artifact["missing_artifacts"]
    assert mod.EXP5734_STREAM_PATH.as_posix() in artifact["malformed_artifacts"]
    assert artifact["proposal_channel_ready"] is False
    assert artifact["sota_proposal_stream_ready"] is False
    assert artifact["sota_csl_ingress_ready"] is True
    assert artifact["continuous_self_learning_credited"] is True
    assert artifact["batch_backend_ready"] is True
    assert artifact["arc_registry_delta"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5742_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5742-FIELD-PRINCIPLES: validation rejects laundering."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(root=tmp_path)
    assert mod.validate_artifact(artifact) == []

    bad_cases = [
        {"field_principles": {"honest_verdict": "x"}},
        {"proposal_channel_ready": False},
        {"sota_proposal_stream_ready": False},
        {"zero_gate_csl_ready": False},
        {"csl_lifecycle_ready": False},
        {"batch_backend_ready": False},
        {"rust_batched_10x_ready": True},
        {"arc_causal_primitive_ready": True},
        {"arc_generic_primitive_live_ready": True},
        {"continuous_self_learning_credited": False},
        {"model_weight_mutation": True},
        {"production_default_enabled": True},
        {"arc_registry_delta": 1},
        {"arc_registry_count_after": artifact["arc_registry_count_before"] + 1},
        {"arc_solve_credited": True},
        {"closed_scopes_reopened": True},
        {"timing_claimed": False},
        {"software_speedup_claimed": True},
        {"hardware_speedup_claimed": True},
        {"inference_substrate": "live_gpu"},
        {"reproducibility_checksum": "bad"},
        {"honest_verdict": "maybe"},
    ]
    for patch in bad_cases:
        assert mod.validate_artifact({**artifact, **patch})

    reopened = json.loads(json.dumps(artifact))
    reopened["preconditions_checked"]["dependency_retired_id_check"]["retired_references"] = [
        {
            "task_id": mod.EXP5742_TASK_ID,
            "upstream": "exp5724-one-axis-rust-python-matched-crossover",
        }
    ]
    assert "closed_scopes_reopened" in " ".join(mod.validate_artifact(reopened))


def test_scenario_capstone_5742_writer_cli_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-5742-FIELD-PRINCIPLES: writer and helpers are stable."""

    _make_root(tmp_path)
    validation = [
        {"command": "focused", "exit_code": 0, "status": "passed"},
        {"command": "audit", "exit_code": 1, "status": "pre_existing_debt"},
    ]
    artifact = mod.write_capstone(root=tmp_path, validation_results=validation)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["e2e_exit_codes"] == {"focused": 0, "audit": 1}

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    assert mod._load_validation_results(validation_path) == validation
    bad_validation = tmp_path / "bad-validation.json"
    bad_validation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation)

    assert mod._extract_outcome("x | OK | y") == "OK"
    assert mod._extract_outcome("x | NOTE | y") == "LOGGED"
    assert mod._fallback_conductor_outcome("complete") == "OK"
    assert mod._fallback_conductor_outcome("unknown-state") == "UNKNOWN"
    assert mod._status_from_meta({}, {"exists": False}) == "missing"
    assert mod._status_from_meta({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._status_from_meta({"flagged_adversarial": True}, {"exists": True, "loadable": True})
        == "flagged"
    )
    assert (
        mod._status_from_meta(
            {"schema": "blocked_gate_check_v1"}, {"exists": True, "loadable": True}
        )
        == "gate_skipped"
    )
    assert (
        mod._status_from_meta({"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True})
        == "blocked"
    )
    assert (
        mod._status_from_meta({"honest_verdict": "complete: x"}, {"exists": True, "loadable": True})
        == "complete"
    )
    assert (
        mod._status_from_meta({"honest_verdict": "other"}, {"exists": True, "loadable": True})
        == "unknown"
    )
    assert mod._registry_count({"reproducible_total_levels": "183"}) == 183
    assert mod._registry_count({"reproducible_total_levels": True}) is None
    assert mod._registry_count({"reproducible_total_levels": "unknown"}) is None
    assert mod._score_ready({}, "missing_score") is False
    retired_check = mod._dependency_retired_id_check(
        {
            "tasks": [
                {"id": "exp-a", "requires": ["exp5724-one-axis-rust-python-matched-crossover"]},
                {"id": "exp-b", "gated_on": [{"upstream": "missing-upstream"}]},
            ]
        }
    )
    assert retired_check["valid"] is False
    assert {row["field"] for row in retired_check["retired_references"]} == {
        "requires",
        "gated_on",
    }
    assert mod._scope_matches(
        "exp9999-sota-finite-choice-proposal-channel",
        "old-sota-finite-choice-proposal-channel",
    )
    required, applied = mod._retirement_rows(
        {
            "tasks": [
                {
                    "id": "exp9999-sota-finite-choice-proposal-channel",
                    "prior_failures": [
                        None,
                        {"retire_if_same_verdict": False},
                        {
                            "experiment_id": "old-sota-finite-choice-proposal-channel",
                            "verdict": "blocked: same",
                            "retire_if_same_verdict": True,
                        },
                    ],
                }
            ]
        },
        {"exp9999-sota-finite-choice-proposal-channel": "blocked: same"},
    )
    assert required[0]["scope_match"] is True
    assert applied[0]["decision"] == "retired_same_verdict_matching_scope"

    assert mod.main(["--root", str(tmp_path), "--output", "custom/capstone.json"]) == 0
    assert (tmp_path / "custom/capstone.json").exists()

    monkeypatch.setattr(mod, "run_capstone", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="invalid Exp5742 capstone artifact"):
        mod.write_capstone(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
    monkeypatch.setattr(mod, "write_capstone", lambda **_kwargs: artifact)
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
