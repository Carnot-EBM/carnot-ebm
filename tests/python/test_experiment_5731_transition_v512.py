"""Tests for the Exp5731 V512 transition receipt.

Spec refs: REQ-CAPSTONE-5731, SCENARIO-CAPSTONE-5731,
SCENARIO-CAPSTONE-5731-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5731_transition_v512 as mod


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


def _model_spec(repo_id: str, hash_suffix: str) -> JsonDict:
    return {
        "model_repo_id": repo_id,
        "hf_id": repo_id,
        "local_model_present": True,
        "resolved_model_path": f"/models/{repo_id.replace('/', '--')}.gguf",
        "model_hash": f"sha256:{hash_suffix}",
        "gguf_filename": f"{repo_id.split('/')[-1]}.gguf",
        "quantization": "Q4_K_M",
    }


def _payloads() -> dict[Path, JsonDict]:
    specs = [
        _model_spec("unsloth/Qwen3.6-35B-A3B-GGUF", "a" * 64),
        _model_spec("unsloth/gemma-4-31B-it-GGUF", "b" * 64),
        _model_spec("unsloth/gemma-4-26B-A4B-it-GGUF", "c" * 64),
    ]
    receipts = {
        row["model_repo_id"]: {
            "local_model_present": True,
            "resolved_model_path": row["resolved_model_path"],
            "model_hash": row["model_hash"],
        }
        for row in specs
    }
    return {
        mod.EXP5717_TRANSITION_PATH: {
            "schema": "carnot.experiment_5717.transition_v511.v1",
            "experiment_id": "exp5717-transition-v511",
            "honest_verdict": "blocked: v511 transition preserved terminal .510 evidence",
            "current_task_range": "exp5717-exp5728",
            "preserved_scopes": [{"scope": "future_clean_prospective_streams"}],
            "retired_scopes": [
                {"scope": "fr11_prospective_shadow_stream_exp5709_same_verdict"}
            ],
        },
        mod.EXP5718_SOURCE_PATH: {
            "schema": "carnot.experiment_5718.v511_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5718-v511-source-delta-ingestion",
            "honest_verdict": "complete: accepted 1 non-duplicate actionable V511 source delta",
            "flagged_adversarial": True,
            "critical_flags": [{"code": "DURATION_TOO_SHORT"}],
        },
        mod.EXP5719_ANSWER_PATH: {
            "schema": "carnot.experiment_5719.sota_answer_channel_forensics.v1",
            "experiment_id": "experiment_5719_sota_answer_channel_forensics",
            "honest_verdict": "blocked: no_qualified_protocol",
            "MODEL_SPECS": specs,
            "resolved_model_receipts": receipts,
            "model_hashes": {row["model_repo_id"]: row["model_hash"] for row in specs},
            "gguf_filenames": {row["model_repo_id"]: row["gguf_filename"] for row in specs},
            "quantizations": {row["model_repo_id"]: row["quantization"] for row in specs},
            "qualified_protocol": {},
            "qualified_model_ids": [],
            "qualified_model_count": 0,
            "answer_channel_ready_score": 0.0,
            "positive_control_parse_rate": 0.0,
            "cuda_offload_authenticated": {
                "unsloth/Qwen3.6-35B-A3B-GGUF": True,
                "unsloth/gemma-4-31B-it-GGUF": True,
                "unsloth/gemma-4-26B-A4B-it-GGUF": True,
            },
            "cuda_offload_authenticated_score": 0.0,
            "parse_failure_count": 82,
            "truncation_count": 41,
            "missing_answer_count": 82,
            "repetition_failure_count": 10,
            "validator_disagreement_count": 0,
            "native_json_grammar_used": False,
            "external_scorer_used": False,
            "retired_runtime_used": False,
        },
        mod.EXP5720_STREAM_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5720,
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "4 of 4 gate(s) failed",
            "gates_evaluated": [{"upstream": "exp5719", "passed": False}],
        },
        mod.EXP5722_RECOVERY_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5722,
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "3 of 3 gate(s) failed",
            "gates_evaluated": [{"upstream": "exp5721", "passed": False}],
        },
        mod.EXP5723_RUST_BACKEND_PATH: {
            "schema": "carnot.experiment_5723.one_axis_rust_samplerbackend_integration.v1",
            "experiment_id": "exp5723-one-axis-rust-samplerbackend-integration",
            "honest_verdict": "complete: one-axis Rust/PyO3 kernel is exposed",
            "one_axis_samplerbackend_ready_score": 1.0,
            "exact_fallback_equivalence_score": 1.0,
            "fallback_equivalence_pass": True,
            "two_axis_code_added": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5724_CROSSOVER_PATH: {
            "schema": "carnot.experiment_5724.one_axis_rust_python_matched_crossover.v1",
            "experiment_id": "exp5724-one-axis-rust-python-matched-crossover",
            "honest_verdict": "complete: terminal null; no crossover proven",
            "quality_matched_pair_count": 178,
            "qualified_crossover_n": None,
            "rust_crossover_ready_score": 0.0,
            "software_speedup_claimed": False,
            "timing_claimed": True,
            "hardware_speedup_claimed": False,
            "gpu_speedup_claimed": False,
            "fpga_or_tsu_used": False,
        },
        mod.EXP5725_ARC_QUAL_PATH: {
            "schema": "carnot.exp5725.arc_epistemic_ledger_live_qualification.v1",
            "experiment": 5725,
            "honest_verdict": "complete: arc_epistemic_ledger_live_reachable_safe_no_solve_claim",
            "arc_epistemic_ledger_ready_score": 1.0,
            "live_path_reachable_score": 1.0,
            "solve_provenance": "development_proxy",
            "new_levels_claimed": 0,
            "registry_updated": False,
            "unsafe_commit_count": 0,
        },
        mod.EXP5726_ARC_AB_PATH: {
            "schema": "carnot.exp5726.arc_epistemic_ledger_live_ab.v1",
            "experiment": 5726,
            "honest_verdict": "complete: epistemic_ledger_live_ab_null_no_promotion",
            "arc_epistemic_live_ab_ready_score": 0.0,
            "successful_pair_count": 6,
            "unsafe_commit_count": 0,
            "new_levels_claimed": 0,
            "registry_updated": False,
            "solve_provenance": "development_proxy",
        },
        mod.EXP5727_ARC_GAP_PATH: {
            "schema": "carnot.exp5727.arc_generalization_live_oracle_gap.v1",
            "experiment": "experiment_5727_arc_generalization_live_oracle_gap_v511",
            "honest_verdict": "complete: arc_generalization_live_oracle_gap_4_of_183_levels_gap_179",
            "games_measured": 25,
            "live_levels_total": 4,
            "oracle_levels_total": 183,
            "gap_total": 179,
            "any_new_level_found": False,
            "new_level_evidence": [],
        },
        mod.EXP5728_CAPSTONE_PATH: {
            "schema": "carnot.experiment_5728.v511_capstone_reconciliation.v1",
            "experiment_id": "exp5728-v511-capstone-reconciliation",
            "honest_verdict": "blocked: v511 reconciled; answer_channel_ready=false; stream_ready=false",
            "continuous_self_learning_credited": False,
            "rust_samplerbackend_status": {"promoted": True},
            "rust_python_crossover_status": {"terminal_null": True},
            "arc_live_attempt_status": {
                "games_measured": 25,
                "live_levels_total": 4,
                "oracle_levels_total": 183,
                "gap_total": 179,
            },
            "arc_registry_delta": 0,
        },
        Path("results/experiment_5721_actions_to_progress_retrieval_ab.json"): {
            "schema": "unrelated.same_range.v1",
            "experiment_id": "experiment_5721_actions_to_progress_retrieval_ab",
            "honest_verdict": "complete_retrieval_unrelated_to_v511_fr11_lifecycle",
        },
    }


def _roadmap_payload() -> JsonDict:
    ids = [f"exp{idx}-fixture-task" for idx in range(5731, 5743)]
    ids[0] = "exp5731-transition-v512"
    ids[1] = "exp5732-v512-source-delta-ingestion"
    ids[2] = "exp5733-sota-finite-choice-proposal-channel"
    ids[3] = "exp5734-sota-exact-proposal-stream"
    ids[4] = "exp5735-zero-gate-kan-continuous-self-learning"
    ids[5] = "exp5736-csl-lifecycle-conflict-rollback"
    tasks: list[JsonDict] = []
    for task_id in ids:
        row: JsonDict = {"id": task_id, "title": task_id}
        if task_id == "exp5734-sota-exact-proposal-stream":
            row["gated_on"] = [
                {
                    "upstream": "exp5733-sota-finite-choice-proposal-channel",
                    "artifact_field": "proposal_channel_ready_score",
                    "op": ">=",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5736-csl-lifecycle-conflict-rollback":
            row["gated_on"] = [
                {
                    "upstream": "exp5735-zero-gate-kan-continuous-self-learning",
                    "artifact_field": "zero_gate_ready_score",
                    "op": ">=",
                    "value": 1.0,
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
    # Exp5721's lifecycle artifact is intentionally not written.
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-07-19 19:18 UTC | Transition terminal .510 evidence | OK | tests |",
                "| 2026-07-19 19:42 UTC | Ingest post-V511 2025-2026 source deltas | FLAGGED | DURATION_TOO_SHORT |",
                "| 2026-07-19 20:07 UTC | Diagnose the .510 GGUF answer-channel failure | OK | tests |",
                "| 2026-07-19 20:13 UTC | Gated on Exp5719 channel readiness: build a sealed | GATE_BLOCK | 4 failed |",
                "| 2026-07-19 20:19 UTC | Gated on Exp5720 exact stream: prospective MemOps | GATE_BLOCK | preemptive |",
                "| 2026-07-19 20:19 UTC | Gated on Exp5721 lifecycle readiness: isolated Com | GATE_BLOCK | 3 failed |",
                "| 2026-07-19 20:49 UTC | Gated on Exp5717 Rust quality: expose one-axis | OK | tests |",
                "| 2026-07-19 21:10 UTC | Gated on Exp5723 production backend: matched Rust/ | OK | tests |",
                "| 2026-07-19 21:32 UTC | Qualify an agent-owned ARC epistemic ledger | OK | tests |",
                "| 2026-07-19 21:50 UTC | Gated on Exp5725 ledger readiness: matched known-l | OK | tests |",
                "| 2026-07-19 23:08 UTC | First ARC-AGI-3 Generalization-Testing Floor task: | OK | tests |",
                "| 2026-07-19 23:55 UTC | Reconcile .511 attested FR-11, Rust backend, ARC e | OK | tests |",
                "| 2026-07-20 00:53 UTC | Milestone 2026.07.512 activated | OK | 12 tasks queued |",
            ]
        )
        + "\n",
    )
    roadmap_path = root / mod.ROADMAP_RELATIVE_PATH
    roadmap_path.parent.mkdir(parents=True, exist_ok=True)
    roadmap_path.write_text(yaml.safe_dump(_roadmap_payload(), sort_keys=False), encoding="utf-8")
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "Milestone: `2026.07.512`\nexp5731-exp5742\n")
    _write_text(root, Path("AGENTS.md"), "AGENTS\n")
    _write_text(root, Path("CODEX.md"), "CODEX\n")
    _write_text(root, Path("CLAUDE.md"), "CLAUDE\n")
    _write_text(root, Path("research-program.md"), "program\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.STATUS_RELATIVE_PATH, "status\n")
    _write_text(root, mod.CHANGELOG_RELATIVE_PATH, "changelog\n")
    _write_text(root, mod.TRACEABILITY_RELATIVE_PATH, "traceability\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor\n")
    manifest = {
        "retired_extras": [
            {"scope_key": "fr11_prospective_shadow_stream_exp5709_same_verdict"},
            {"scope_key": "free_form_gguf_answer_envelope_repair_exp5719_same_protocol"},
            {"scope_key": "two_axis_beta_lambda_tempering_extension_exp5645"},
            {"scope_key": "arc_epistemic_ledger_live_ab_exp5726_same_verdict"},
        ]
    }
    manifest_path = root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    registry_path = root / mod.ARC_REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump({"reproducible_total_levels": 183, "reproducible_total_games": 25}),
        encoding="utf-8",
    )
    spec_path = root / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_capstone_5731_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5731: OpenSpec declares the V512 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5731") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "all three mandated GGUFs resolved" in section
    assert "current_task_range=exp5731-exp5742" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5731_live_repo_archives_v511_and_allocates_v512() -> None:
    """SCENARIO-CAPSTONE-5731: live repo transition preserves terminal .511 evidence."""

    artifact = mod.run_transition(
        root=REPO,
        validation_results=[{"command": "focused", "exit_code": 0, "status": "passed"}],
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["source_capstone_hash"] == mod.path_sha256(REPO / mod.EXP5728_CAPSTONE_PATH)
    assert mod.EXP5721_LIFECYCLE_PATH.as_posix() in artifact["missing_artifacts"]
    assert artifact["v511_task_verdicts"][mod.EXP5719_TASK_ID]["status"] == "blocked"
    assert artifact["v511_task_verdicts"][mod.EXP5720_TASK_ID]["status"] == "gate_skipped"
    assert artifact["v511_task_verdicts"][mod.EXP5721_TASK_ID]["status"] == "missing"
    assert artifact["v511_task_verdicts"][mod.EXP5723_TASK_ID]["status"] == "complete"
    assert artifact["v511_conductor_outcomes"][mod.EXP5718_TASK_ID]["outcome"] == "FLAGGED"
    assert artifact["v511_conductor_outcomes"][mod.EXP5721_TASK_ID]["outcome"] == "GATE_BLOCK"
    assert artifact["v511_conductor_outcomes"][mod.EXP5728_TASK_ID]["outcome"] == "OK"

    root = artifact["answer_channel_root_evidence"]
    assert root["all_three_required_ggufs_resolved"] is True
    assert root["qualified_protocol"] == {}
    assert root["positive_control_parse_rate"] == 0.0
    assert root["truncation_count"] == 41
    assert root["missing_answer_count"] == 82
    assert root["repetition_failure_count"] == 10
    assert root["cuda_offload_authenticated_score"] == 0.0

    assert artifact["answer_channel_ready"] is False
    assert artifact["stream_ready"] is False
    assert artifact["continuous_self_learning_credited"] is False
    assert artifact["rust_samplerbackend_ready"] is True
    assert artifact["rust_python_crossover_null"] is True
    assert artifact["rust_transition_boundaries"]["quality_matched_pair_count"] == 178
    assert artifact["rust_transition_boundaries"]["qualified_crossover_n"] is None
    assert artifact["rust_transition_boundaries"]["timing_boundary"] == "cpu_software_only"
    assert artifact["arc_live_levels"] == 4
    assert artifact["arc_oracle_levels"] == 183
    assert artifact["arc_gap"] == 179
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_transition_boundaries"]["games_measured"] == 25
    assert artifact["arc_transition_boundaries"]["ledger_retried_or_promoted"] is False
    assert artifact["current_task_range"] == "exp5731-exp5742"
    assert artifact["dependency_chain_retired_id_check"]["valid"] is True
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["inference_substrate"] == "artifact_reconciliation_only"
    assert any(
        row["path"] == "results/experiment_5721_actions_to_progress_retrieval_ab.json"
        for row in artifact["all_range_artifact_scan"]
    )


def test_scenario_capstone_5731_fixture_missing_and_malformed_inputs_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5731-MISSING-MALFORMED: bad inputs fail closed."""

    _make_root(tmp_path, omit=mod.EXP5719_ANSWER_PATH, malformed=mod.EXP5724_CROSSOVER_PATH)
    artifact = mod.run_transition(root=tmp_path)

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5719_ANSWER_PATH.as_posix() in artifact["missing_artifacts"]
    assert mod.EXP5724_CROSSOVER_PATH.as_posix() in artifact["malformed_artifacts"]
    assert mod.EXP5721_LIFECYCLE_PATH.as_posix() in artifact["missing_artifacts"]
    assert artifact["answer_channel_ready"] is False
    assert artifact["stream_ready"] is False
    assert artifact["continuous_self_learning_credited"] is False
    assert artifact["rust_python_crossover_null"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5731_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES: validation rejects laundering."""

    _make_root(tmp_path)
    artifact = mod.run_transition(root=tmp_path)
    assert mod.validate_artifact(artifact) == []

    bad_cases = [
        {"field_principles": {"honest_verdict": "x"}},
        {"answer_channel_ready": True},
        {"stream_ready": True},
        {"continuous_self_learning_credited": True},
        {"rust_samplerbackend_ready": False},
        {"rust_python_crossover_null": False},
        {"arc_live_levels": 5},
        {"arc_oracle_levels": 182},
        {"arc_gap": 178},
        {"arc_registry_delta": 1},
        {"current_task_range": "exp5731-exp5741"},
        {"timing_claimed": True},
        {"hardware_speedup_claimed": True},
        {"inference_substrate": "live_gpu"},
        {"reproducibility_checksum": "bad"},
        {"honest_verdict": "maybe"},
    ]
    for patch in bad_cases:
        assert mod.validate_artifact({**artifact, **patch})

    bad_dependency = json.loads(json.dumps(artifact))
    bad_dependency["dependency_map"]["exp5734-sota-exact-proposal-stream"]["depends_on"] = [
        "exp5719-sota-answer-channel-forensics"
    ]
    assert "retired" in " ".join(mod.validate_artifact(bad_dependency))

    bad_gate = json.loads(json.dumps(artifact))
    bad_gate["gate_map"]["exp5734-sota-exact-proposal-stream"][0][
        "upstream"
    ] = "exp5726-arc-epistemic-ledger-live-ab"
    assert "retired" in " ".join(mod.validate_artifact(bad_gate))


def test_scenario_capstone_5731_writer_cli_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-5731-FIELD-PRINCIPLES: writer and helpers are stable."""

    _make_root(tmp_path)
    validation = [
        {"command": "focused", "exit_code": 0, "status": "passed"},
        {"command": "audit", "exit_code": 1, "status": "pre_existing_debt"},
    ]
    artifact = mod.write_transition(root=tmp_path, validation_results=validation)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["test_exit_codes"] == {"focused": 0, "audit": 1}

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
    assert mod._status_from_meta({"schema": "blocked_gate_check_v1"}, {"exists": True, "loadable": True}) == "gate_skipped"
    assert mod._status_from_meta({"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True}) == "blocked"
    assert mod._status_from_meta({"honest_verdict": "complete: x"}, {"exists": True, "loadable": True}) == "complete"
    assert mod._status_from_meta({"honest_verdict": "other"}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._resolved_model_ids(
        {"MODEL_SPECS": [{"model_repo_id": "model-a", "local_model_present": True}]}
    ) == {"model-a"}
    assert mod._resolved_model_ids({}) == set()
    bad_root = json.loads(json.dumps(artifact))
    bad_root["answer_channel_root_evidence"]["truncation_count"] = 40
    assert "answer_channel_root_evidence.truncation_count" in " ".join(
        mod.validate_artifact(bad_root)
    )
    assert "dependency_map and gate_map" in " ".join(
        mod.validate_artifact({**artifact, "dependency_map": []})
    )

    assert mod.main(["--root", str(tmp_path), "--output", "custom/transition.json"]) == 0
    assert (tmp_path / "custom/transition.json").exists()

    monkeypatch.setattr(mod, "run_transition", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="invalid Exp5731 transition artifact"):
        mod.write_transition(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
    monkeypatch.setattr(mod, "write_transition", lambda **_kwargs: artifact)
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
