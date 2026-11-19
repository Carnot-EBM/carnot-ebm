"""Claim-provenance duration regressions for Exp6974.

Spec refs: REQ-CONDUCTOR-6974 and SCENARIO-CONDUCTOR-6974-*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


ROOT = Path(__file__).resolve().parents[2]
EXP6967 = ROOT / "results" / "experiment_6967_certified_error_headroom_fixture.json"


def _duration_flags(payload: dict[str, Any]) -> list[av.Flag]:
    """Run only the duration rule so unrelated artifact guards cannot mask this contract."""

    flags: list[av.Flag] = []
    av.check_duration_vs_claim(payload, flags)
    return flags


def _critical_kinds(payload: dict[str, Any]) -> set[str]:
    return {flag.kind for flag in _duration_flags(payload) if flag.severity == "critical"}


def _deterministic_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": 6974,
        "honest_verdict": "complete_circular_claim_provenance_fixture",
        "inference_substrate": "deterministic_z3_and_bounded_enumeration_reducer",
        "duration_s": 1.864951312,
        "random_seed": 6974,
        "reproducibility_checksum": "sha256:" + "7" * 64,
        "source_rows": [
            {
                "attempt_key": "unsloth/Qwen3.6-35B-A3B-GGUF|0-3-0|direct_affine",
                "model_label": "unsloth/gemma-4-31B-it-GGUF",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_req_conductor_6974_spec_declares_claim_provenance_contract() -> None:
    """REQ-CONDUCTOR-6974 precedes implementation and names every required scenario."""

    text = (ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONDUCTOR-6974") :]
    assert all(
        f"SCENARIO-CONDUCTOR-6974-{name}" in section
        for name in ("DETERMINISTIC", "LIVE", "NESTED", "AMBIGUOUS", "DERIVATIVE", "ARTIFACT")
    )


def test_scenario_conductor_6974_deterministic_rows_do_not_claim_invocation() -> None:
    """SCENARIO-CONDUCTOR-6974-DETERMINISTIC ignores model IDs used as row keys."""

    payload = _deterministic_payload(
        source_hash="sha256:unsloth/Qwen3.6-35B-A3B-GGUF",
        source_path="results/GGUF/source.json",
        diagnostics={"detail": "upstream llama.cpp used torch.cuda"},
        model_specs={"labels_only": ["unsloth/gemma-4-26B-A4B-it-GGUF"]},
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_NON_LIVE
    assert classification["reason"] == "deterministic_substrate_claim"
    assert "DURATION_TOO_SHORT" not in _critical_kinds(payload)


def test_scenario_conductor_6974_real_exp6967_is_not_a_live_duration_claim() -> None:
    """SCENARIO-CONDUCTOR-6974-DERIVATIVE rechecks the frozen incident bytes."""

    payload = json.loads(EXP6967.read_text(encoding="utf-8"))
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_NON_LIVE
    assert "DURATION_TOO_SHORT" not in _critical_kinds(payload)


def test_scenario_conductor_6974_genuine_live_claim_keeps_live_floor() -> None:
    """SCENARIO-CONDUCTOR-6974-LIVE preserves the 60-second live floor."""

    payload = _deterministic_payload(
        inference_substrate="live_llm_inference",
        duration_s=5.0,
        model_invoked=True,
        model_specs={"name": "unsloth/Qwen3.6-35B-A3B-GGUF"},
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_LIVE
    assert "DURATION_TOO_SHORT" in _critical_kinds(payload)


def test_scenario_conductor_6974_nested_input_invocation_cannot_hide() -> None:
    """SCENARIO-CONDUCTOR-6974-NESTED scans typed invocation fields recursively."""

    payload = _deterministic_payload(
        input_data={"batch": [{"invocation": {"model_invoked": True}}]},
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_CONTRADICTORY
    assert any(row["path"].endswith("model_invoked") for row in classification["live_evidence"])
    assert "DURATION_TOO_SHORT" in _critical_kinds(payload)


def test_req_conductor_6974_typed_upstream_invocation_stays_external() -> None:
    """REQ-CONDUCTOR-6974 distinguishes cited invocation from this task's invocation."""

    payload = _deterministic_payload(
        upstream_receipt_rows=[
            {
                "invocation_provenance": {
                    "scope": "upstream",
                    "model_invoked": True,
                    "live_duration_s": 72.0,
                }
            }
        ]
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_NON_LIVE
    assert len(classification["external_evidence"]) == 2
    assert "DURATION_TOO_SHORT" not in _critical_kinds(payload)


def test_req_conductor_6974_path_scopes_and_diagnostics_do_not_leak() -> None:
    """REQ-CONDUCTOR-6974 attributes typed source fields and ignores diagnostic copies."""

    payload = _deterministic_payload(
        source_rows=[{"model_invoked": True}],
        diagnostics={"model_invoked": True},
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_NON_LIVE
    assert [row["path"] for row in classification["external_evidence"]] == [
        "source_rows.0.model_invoked"
    ]
    assert classification["live_evidence"] == []


def test_scenario_conductor_6974_ambiguous_model_context_fails_closed() -> None:
    """SCENARIO-CONDUCTOR-6974-AMBIGUOUS keeps the conservative live floor."""

    payload = _deterministic_payload()
    payload.pop("inference_substrate")
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_AMBIGUOUS
    assert classification["reason"] == "compute_markers_without_claim_provenance"
    assert "DURATION_TOO_SHORT" in _critical_kinds(payload)


def test_req_conductor_6974_model_labels_alone_are_not_invocation() -> None:
    """REQ-CONDUCTOR-6974 does not turn a model inventory into execution evidence."""

    payload = _deterministic_payload(
        model_specs={"name": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        target_model="unsloth/gemma-4-31B-it-GGUF",
    )

    assert av._claims_live_model(payload) is False
    assert "DURATION_TOO_SHORT" not in _critical_kinds(payload)


def test_req_conductor_6974_live_duration_gpu_and_methodology_are_evidence() -> None:
    """REQ-CONDUCTOR-6974 recognizes three structured current-task evidence families."""

    fixtures = (
        _deterministic_payload(live_duration_s=4.0),
        _deterministic_payload(
            gpu_receipts={
                "scope": "current_task",
                "model_loaded": True,
                "offloaded_layer_count": 42,
            }
        ),
        _deterministic_payload(
            methodology_note="The current task invoked llama.cpp for live model inference."
        ),
    )
    for payload in fixtures:
        classification = av._classify_current_task_inference_claim(payload)
        assert classification["state"] == av.CLAIM_STATE_CONTRADICTORY
        assert "DURATION_TOO_SHORT" in _critical_kinds(payload)


def test_req_conductor_6974_typed_counts_identity_and_methodology_boundaries() -> None:
    """REQ-CONDUCTOR-6974 covers count claims, task identity, and methodology negation."""

    assert av._methodology_claims_live_inference({"structured": True}) is False
    assert av._methodology_claims_live_inference("This task did not invoke the model.") is False

    identity = {
        "honest_verdict": "complete_live_eval_finished",
        "duration_s": 5.0,
    }
    identity_claim = av._classify_current_task_inference_claim(identity)
    assert identity_claim["state"] == av.CLAIM_STATE_LIVE
    assert identity_claim["live_evidence"][0]["signal"] == "current_task_claim"

    count_claim = {
        "inference_call_count": 2,
        "duration_s": 5.0,
    }
    assert av._classify_current_task_inference_claim(count_claim)["state"] == av.CLAIM_STATE_LIVE

    conflicting = {
        "model_invoked": True,
        "llm_invoked": False,
        "duration_s": 5.0,
    }
    conflict = av._classify_current_task_inference_claim(conflicting)
    assert conflict["state"] == av.CLAIM_STATE_CONTRADICTORY
    assert conflict["reason"] == "conflicting_typed_invocation_evidence"

    nested = {
        "input_data": {"batch": [{"model_invoked": True}]},
        "duration_s": 5.0,
    }
    nested_claim = av._classify_current_task_inference_claim(nested)
    assert nested_claim["state"] == av.CLAIM_STATE_AMBIGUOUS
    assert nested_claim["reason"] == "nested_invocation_without_current_task_provenance"
    assert av.duration_floor_for_artifact(nested)["min_duration_s"] == 60.0


def test_req_conductor_6974_specialized_live_floors_are_preserved() -> None:
    """REQ-CONDUCTOR-6974 changes provenance classification, not calibrated live floors."""

    for substrate, expected in (
        ("live_llm_embedding_extraction", 2.0),
        ("live_llm_inference_local_gguf_sota", 10.0),
        ("local_native_llama_cpp_gguf_backend_bisect", 5.0),
    ):
        floor = av.duration_floor_for_artifact(
            {
                "inference_substrate": substrate,
                "model_invoked": True,
                "duration_s": 1.0,
            }
        )
        assert floor is not None
        assert floor["min_duration_s"] == expected


def test_req_conductor_6974_nonfinite_and_nonlive_methodology_paths() -> None:
    """REQ-CONDUCTOR-6974 reports contradictions before missing-duration return paths."""

    contradiction = _deterministic_payload(model_invoked=True, duration_s=None)
    assert "INFERENCE_PROVENANCE_CONTRADICTION" in _critical_kinds(contradiction)
    assert "DURATION_TOO_SHORT" not in _critical_kinds(contradiction)

    for payload in (
        _deterministic_payload(),
        {"inference_substrate": "unknown_cpu_work", "duration_s": 1.0},
        {
            "inference_substrate": "live_llm_inference",
            "honest_verdict": "blocked_model_missing",
        },
    ):
        flags: list[av.Flag] = []
        av.check_methodology_present(payload, flags)
        assert flags == []


def test_req_conductor_6974_live_claim_with_no_invocation_is_contradictory() -> None:
    """REQ-CONDUCTOR-6974 fails closed when claim and structured invocation disagree."""

    payload = _deterministic_payload(
        inference_substrate="live_llm_inference",
        duration_s=120.0,
        model_invoked=False,
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_CONTRADICTORY
    assert "INFERENCE_PROVENANCE_CONTRADICTION" in _critical_kinds(payload)


def test_req_conductor_6974_blocked_pre_gate_remains_exempt() -> None:
    """REQ-CONDUCTOR-6974 does not call an honest precondition stop a live run."""

    payload = _deterministic_payload(
        honest_verdict="blocked_model_not_cached",
        inference_substrate="live_llm_inference",
        duration_s=0.01,
        model_invoked=False,
    )
    classification = av._classify_current_task_inference_claim(payload)

    assert classification["state"] == av.CLAIM_STATE_BLOCKED
    assert _critical_kinds(payload) == set()
