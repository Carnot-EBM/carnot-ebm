"""Tests for Exp 4754 submitted `.437` agent config confirmation.

Spec refs: REQ-ARC-WMTE-4754, SCENARIO-ARC-WMTE-4754.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _scored_mtp_on() -> bool:
    """Read the scored-MTP answer from the constant the module's own gate reads, so the fixture and
    the gate cannot drift apart. See the comment at the `mtp` key below for why this is derived
    here and pinned as a literal elsewhere."""
    from carnot.agentic.arc_executable_world_model import ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT

    return ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"


_SCORED_MTP_ON = _scored_mtp_on()


def _live_generator_pins() -> tuple[str, str]:
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_MODEL_ID,
        ARC_LIVE_GENERATOR_REPO_SUBSTR,
    )

    return ARC_LIVE_GENERATOR_MODEL_ID, ARC_LIVE_GENERATOR_REPO_SUBSTR


_LIVE_MODEL_ID, _LIVE_REPO_SUBSTR = _live_generator_pins()


def _submitted_config() -> dict[str, Any]:
    return {
        "policy": "E3AgentPolicy",
        "cascade": True,
        "target_levels": 3,
        "live_submit_package_path": "results/experiment_4643_submission_package_operator_resubmit.json",
        "frozen_generator": {
            # DERIVED for the same reason as `mtp` below. These were literals naming
            # `gemma-4-31B-it`, and they went stale when the generator moved to Qwen3.8-27B: the
            # gate compares the config against the live constants, so a fixture pinned to the
            # previous model made a correct gate report `..._confirmation_failed_gate`. That
            # failure was live at HEAD before this change and had nothing to do with MTP; it was
            # simply never noticed, because nothing runs this file on the migration's path.
            "model_id": _LIVE_MODEL_ID,
            "repo_substr": _LIVE_REPO_SUBSTR,
            "model_filename": f"{_LIVE_REPO_SUBSTR}-Q4_K_M.gguf",
            # DERIVED, NOT LITERAL (2026-08-18). This fixture stands in for the frozen config so
            # the module's GATE LOGIC can be exercised; it is not where the scored MTP contract
            # lives. That contract is pinned as a literal in
            # `tests/python/test_arc_submitted_agent_parity.py`, deliberately, so it goes red when
            # someone flips the knob. Hardcoding `True` here as well meant one constant was pinned
            # by two tests that disagreed, and this one failed for a reason that had nothing to do
            # with what it checks -- the gate compares the config against the constant, so a
            # fixture frozen to the old value makes a correct gate look broken.
            "mtp": _SCORED_MTP_ON,
            "spec_type": "draft-mtp" if _SCORED_MTP_ON else None,
            "kv_quant": "q8_0",
            "no_think_prefix": "",
            "llama_server_kind": "cuda-12.8-binary",
            "binary_not_wheel": True,
            "wheel_fallback_allowed": False,
            "port_strategy": "free_non_8919",
            "props_verify_endpoint": "/props",
            "forbidden_models": ["gemma-8919"],
            "forbidden_gpu_targets": ["3090"],
            "required_shared_libraries": ["libllama-common"],
        },
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "pinned_generator_gguf_cached": True,
        "pinned_generator_gguf_paths": ["/models/gemma-4-31B-it-Q4_K_M.gguf"],
        "offline_arcade_ok": True,
        "make_carnot_agent_import_ok": True,
        "spec_has_req_4754": True,
        "submission_entrypoint_present": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4754_spec_declares_submitted_config_contract() -> None:
    """REQ-ARC-WMTE-4754: OpenSpec declares the submitted config confirmation artifact."""

    from carnot import experiment_4754_submitted_agent_config as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4754" in spec
    assert "SCENARIO-ARC-WMTE-4754" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4754_audits_env_gates_only_after_upstream_success() -> None:
    """REQ-ARC-WMTE-4754: `.437` env gates turn on only for validated A1/A2 artifacts."""

    from carnot import experiment_4754_submitted_agent_config as mod

    source = (
        "if os.environ.get('CARNOT_ARC_STRUCTURED_ENGINE') == '1':\n"
        "    from carnot.agentic import arc_structured_world_model\n"
        "structural_alignment_goal_candidate()\n"
        "os.environ.get('CARNOT_ARC_TRUST_METRIC', 'exact')\n"
    )
    unchanged = mod.audit_env_gate_state(
        a1_artifact=None,
        a2_artifact=None,
        live_source_text=source,
    )

    assert unchanged["levers_integrated"] == []
    assert unchanged["env_gate_state"]["CARNOT_ARC_STRUCTURED_ENGINE"] == "0"
    assert unchanged["env_gate_state"]["CARNOT_ARC_TRUST_METRIC"] == "exact"
    assert unchanged["a1"]["reason"] == "a1_artifact_missing_or_not_mapping"
    assert unchanged["a2"]["reason"] == "a2_artifact_missing_or_not_mapping"
    assert unchanged["live_path_hooks"]["structured_engine_hook_present"] is True
    assert unchanged["live_path_hooks"]["fixed_detector_hook_present"] is True

    winning_a1 = {
        "honest_verdict": "success_structured_engine_accuracy_wide_win",
        "verifier_is_oracle": False,
        "structured_engine_non_degenerate": True,
        "structured_heldout_accuracy": 0.56,
        "freeform_heldout_accuracy": 0.12,
    }
    winning_a2 = {
        "honest_verdict": "success_fixed_detector_structural_alignment_goal_provider",
        "verifier_is_oracle": False,
        "structural_alignment_detector_fixed": True,
    }
    validated = mod.audit_env_gate_state(
        a1_artifact=winning_a1,
        a2_artifact=winning_a2,
        live_source_text=source,
    )

    assert validated["levers_integrated"] == [
        "A1_structured_engine",
        "A2_fixed_structural_alignment_detector",
    ]
    assert validated["env_gate_state"]["CARNOT_ARC_STRUCTURED_ENGINE"] == "1"
    assert validated["env_gate_state"]["CARNOT_ARC_TRUST_METRIC"] == "cell_recall"
    assert validated["a1"]["reason"] == "validated_and_live_hook_present"
    assert validated["a2"]["reason"] == "validated_and_live_hook_present"

    missing_hook = mod.audit_env_gate_state(
        a1_artifact=winning_a1,
        a2_artifact=winning_a2,
        live_source_text="no hooks here",
    )
    assert missing_hook["env_gate_state"]["CARNOT_ARC_STRUCTURED_ENGINE"] == "0"
    assert missing_hook["env_gate_state"]["CARNOT_ARC_TRUST_METRIC"] == "exact"
    assert missing_hook["a1"]["reason"] == "structured_engine_live_hook_missing"
    assert missing_hook["a2"]["reason"] == "fixed_detector_live_hook_missing"

    assert mod._as_float(None, 7.0) == 7.0
    assert mod._as_float("bad", 3.0) == 3.0
    assert mod._structured_engine_validated({**winning_a1, "verifier_is_oracle": True}) == (
        False,
        "a1_verifier_oracle_not_false",
    )
    assert mod._structured_engine_validated(
        {**winning_a1, "structured_engine_non_degenerate": False}
    ) == (False, "a1_structured_engine_degenerate_or_unproven")
    assert mod._structured_engine_validated(
        {**winning_a1, "structured_heldout_accuracy": 0.1, "freeform_heldout_accuracy": 0.2}
    ) == (False, "a1_no_accuracy_or_l2_validation")
    assert mod._fixed_detector_validated({**winning_a2, "verifier_is_oracle": True}) == (
        False,
        "a2_verifier_oracle_not_false",
    )
    assert mod._fixed_detector_validated(
        {**winning_a2, "structural_alignment_detector_fixed": False}
    ) == (False, "a2_fixed_detector_unproven")


def test_scenario_arc_wmte_4754_builds_complete_artifact_and_schema_errors() -> None:
    """SCENARIO-ARC-WMTE-4754: green smoke/lint/package writes an operator-ready artifact."""

    from carnot import experiment_4754_submitted_agent_config as mod

    audit = mod.audit_env_gate_state(
        a1_artifact={},
        a2_artifact={},
        live_source_text=(
            "CARNOT_ARC_STRUCTURED_ENGINE arc_structured_world_model "
            "structural_alignment_goal_candidate CARNOT_ARC_TRUST_METRIC"
        ),
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        gate_audit=audit,
        agent_smoke={
            "constructed": True,
            "smoke_step_ran": True,
            "policy": "E3AgentPolicy",
            "move": ["RESET", None],
        },
        frozen_generator_intact={
            "intact": True,
            "submitted_config_declares_qwen35_mtp": True,
            "cached_gguf_matches": True,
        },
        orphan_lint={"passed": True, "returncode": 0, "stdout": "OK"},
        submission_package={"operator_package_present": True, "path": "results/pkg.json"},
        submitted_agent_config=_submitted_config(),
        duration_s=0.2,
    )

    assert artifact["honest_verdict"] == (
        "complete_437_levers_unvalidated_config_unchanged_entrypoint_green"
    )
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["agent_constructs_and_smoke_runs"]["smoke_step_ran"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["duration_s"] == 1.0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    validated_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        gate_audit={
            **audit,
            "levers_integrated": ["A1_structured_engine"],
            "env_gate_state": {
                "CARNOT_ARC_STRUCTURED_ENGINE": "1",
                "CARNOT_ARC_TRUST_METRIC": "exact",
            },
        },
        agent_smoke=artifact["agent_constructs_and_smoke_runs"],
        frozen_generator_intact=artifact["frozen_generator_intact"],
        orphan_lint=artifact["arc_orphan_solver_lint"],
        submission_package=artifact["submission_package"],
        submitted_agent_config=_submitted_config(),
        duration_s=1.0,
    )
    assert validated_artifact["honest_verdict"] == (
        "success_437_validated_levers_integrated_entrypoint_green"
    )

    bad = dict(artifact)
    bad.pop("schema")
    bad.update(
        {
            "honest_verdict": "not-terminal",
            "inference_substrate": "cached",
            "agent_constructs_and_smoke_runs": {"constructed": True},
            "submission_package_ready": "yes",
            "field_principles": {},
            "submitted_to_leaderboard": True,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(bad)
    assert "missing required field schema" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "agent_constructs_and_smoke_runs" in errors
    assert "submission_package_ready_bool" in errors
    assert "field_principles" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4754_run_writes_artifact_and_blocks(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4754: run writes green or blocked terminal artifacts."""

    from carnot import experiment_4754_submitted_agent_config as mod

    assert mod._load_optional_json(tmp_path / "missing.json") is None

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text("REQ-ARC-WMTE-4754\n", encoding="utf-8")
    (tmp_path / "scripts" / "kaggle" / "submission_kernel").mkdir(parents=True)
    (tmp_path / "scripts" / "kaggle" / "submission_kernel" / "main.py").write_text(
        "gemma-4-31B-it CARNOT_ARC_GGUF_PATH\n", encoding="utf-8"
    )
    (tmp_path / "python" / "carnot" / "agentic").mkdir(parents=True)
    (tmp_path / "python" / "carnot" / "agentic" / "arc_competition_agent.py").write_text(
        "CARNOT_ARC_STRUCTURED_ENGINE arc_structured_world_model "
        "structural_alignment_goal_candidate CARNOT_ARC_TRUST_METRIC\n",
        encoding="utf-8",
    )
    package_path = tmp_path / _submitted_config()["live_submit_package_path"]
    package_path.parent.mkdir(parents=True)
    package_path.write_text("{}", encoding="utf-8")
    _write_json(
        tmp_path / mod.A1_RELATIVE_PATH,
        {
            "honest_verdict": "success_structured_engine_accuracy_wide_win",
            "verifier_is_oracle": False,
            "structured_engine_non_degenerate": True,
            "structured_heldout_accuracy": 0.56,
            "freeform_heldout_accuracy": 0.12,
        },
    )
    _write_json(
        tmp_path / mod.A2_RELATIVE_PATH,
        {
            "honest_verdict": "success_fixed_detector_structural_alignment_goal_provider",
            "verifier_is_oracle": False,
            "structural_alignment_detector_fixed": True,
        },
    )

    artifact = mod.run(
        tmp_path,
        gguf_finder=lambda: ["/models/gemma-4-31B-it-Q4_K_M.gguf"],
        offline_arcade_checker=lambda: {"offline_arcade_ok": True},
        agent_import_checker=lambda: {"make_carnot_agent_import_ok": True},
        agent_smoke_runner=lambda _gate: {
            "constructed": True,
            "smoke_step_ran": True,
            "policy": "E3AgentPolicy",
            "move": ["RESET", None],
            "env_gates_applied": _gate,
        },
        orphan_lint_runner=lambda _root: {"passed": True, "returncode": 0, "stdout": "OK"},
        submitted_agent_config_loader=lambda: _submitted_config(),
        now=iter([10.0, 10.1]).__next__,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["env_gate_state"]["CARNOT_ARC_STRUCTURED_ENGINE"] == "1"
    assert artifact["env_gate_state"]["CARNOT_ARC_TRUST_METRIC"] == "cell_recall"
    assert artifact["honest_verdict"] == "success_437_validated_levers_integrated_entrypoint_green"
    assert artifact["submission_package_ready"] is True

    blocked = mod.run(
        tmp_path,
        gguf_finder=lambda: [],
        offline_arcade_checker=lambda: {"offline_arcade_ok": True},
        agent_import_checker=lambda: {"make_carnot_agent_import_ok": True},
        agent_smoke_runner=lambda _gate: {"constructed": True, "smoke_step_ran": True},
        orphan_lint_runner=lambda _root: {"passed": True, "returncode": 0},
        submitted_agent_config_loader=lambda: _submitted_config(),
        now=iter([20.0, 20.1]).__next__,
    )

    assert blocked["honest_verdict"] == "blocked_pinned_generator_gguf_cached"
    assert blocked["preconditions_checked"]["blocked_resource"] == "pinned_generator_gguf_cached"
    assert blocked["agent_constructs_and_smoke_runs"]["smoke_step_ran"] is False
    assert blocked["submission_package_ready"] is False

    blocked_with_loader_error = mod.run(
        tmp_path,
        gguf_finder=lambda: [],
        offline_arcade_checker=lambda: {"offline_arcade_ok": True},
        agent_import_checker=lambda: {"make_carnot_agent_import_ok": True},
        submitted_agent_config_loader=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        now=iter([30.0, 30.1]).__next__,
    )
    assert blocked_with_loader_error["submitted_agent_config"] == {}
