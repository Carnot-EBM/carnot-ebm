"""Tests for Exp5571 reset-free local SOTA continual harness.

Spec refs: REQ-LEARN-5571,
SCENARIO-LEARN-5571-PRECONDITIONS,
SCENARIO-LEARN-5571-SESSIONS,
SCENARIO-LEARN-5571-RESET-FREE,
SCENARIO-LEARN-5571-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_5571_reset_free_sota_continual_harness as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5571_reset_free_sota_continual_harness.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5571_reset_free_sota_continual_harness.py "
    "-m pytest tests/python/test_experiment_5571_reset_free_sota_continual_harness.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5571_reset_free_sota_continual_harness.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    paths = {}
    for key in ("qwen", "gemma31", "gemma26"):
        path = tmp_path / f"{key}.gguf"
        path.write_bytes(key.encode("ascii"))
        paths[key] = path
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": exp.QWEN_ID,
            "gpu": 0,
            "model_path": str(paths["qwen"]),
            "headline_model": True,
            "optional_replication_model": False,
            "local_model_present": True,
            "legacy_model": False,
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": exp.GEMMA31_ID,
            "gpu": 1,
            "model_path": str(paths["gemma31"]),
            "headline_model": False,
            "optional_replication_model": True,
            "local_model_present": True,
            "legacy_model": False,
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": exp.GEMMA26_ID,
            "gpu": 1,
            "model_path": str(paths["gemma26"]),
            "headline_model": False,
            "optional_replication_model": True,
            "local_model_present": True,
            "legacy_model": False,
        },
    ]


def _complete_gate(model_specs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cached_sota_pair_called": True,
        "cache_gate_passed": True,
        "blocked_reason": "",
        "cached_pair_hf_ids": [exp.QWEN_ID, exp.GEMMA26_ID],
        "selected_headline_model_ids": [exp.QWEN_ID],
        "optional_replication_model_ids": [exp.GEMMA31_ID, exp.GEMMA26_ID],
        "declared_model_ids": [str(row["hf_id"]) for row in model_specs],
        "legacy_cpu_model_substituted": False,
        "upstream_policy_gate_passed": True,
        "upstream_energy_gate_passed": True,
        "corpus_gate_passed": True,
        "offload_gate_passed": True,
        "live_invocation_gate_passed": True,
    }


def _complete_device() -> dict[str, object]:
    return {
        "torch_cuda_available": True,
        "torch_device_count": 2,
        "devices": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090"},
        ],
        "llama_cpp_supports_gpu_offload": True,
        "gpu_offload_authenticated": True,
    }


def _live_receipt() -> dict[str, object]:
    return {
        "invoked": True,
        "model_hf_id": exp.QWEN_ID,
        "model_path": "/tmp/qwen.gguf",
        "duration_s": 64.0,
        "prompt_tokens": 7200,
        "completion_tokens": 1,
        "tokens_generated": 1,
        "gpu_offload_authenticated": True,
    }


def _complete_artifact(tmp_path: Path) -> dict[str, object]:
    model_specs = _fake_model_specs(tmp_path)
    return exp.build_artifact(
        root=REPO,
        model_specs=model_specs,
        gate_receipt=_complete_gate(model_specs),
        device_receipt=_complete_device(),
        live_invocation_receipt=_live_receipt(),
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )


def test_req_learn_5571_spec_declares_reset_free_contract() -> None:
    """REQ-LEARN-5571: OpenSpec anchors preconditions, arms, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5571") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5571",
        "SCENARIO-LEARN-5571-PRECONDITIONS",
        "SCENARIO-LEARN-5571-SESSIONS",
        "SCENARIO-LEARN-5571-RESET-FREE",
        "SCENARIO-LEARN-5571-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.QWEN_ID,
        exp.GEMMA31_ID,
        exp.GEMMA26_ID,
        "blocked_missing_sota_cache",
        "blocked_no_cuda_offload",
        "paired bootstrap",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    assert "SHALL NOT substitute a legacy headline model" in normalized
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert exp.FIELD_PRINCIPLES[field]


def test_scenario_learn_5571_sessions_preserve_paired_instance_ids() -> None:
    """SCENARIO-LEARN-5571-SESSIONS: ordered sessions keep paired IDs."""

    sessions = exp.build_sessions(exp.load_exact_rows(REPO))
    result = exp.run_continual_harness(sessions)

    assert len(sessions) >= 3
    assert all(session["instance_count"] >= 30 for session in sessions)
    assert result["n_independent_instances_per_session"] == 30
    assert set(result["paired_instance_ids_by_arm"]) == set(exp.ARMS)
    first = result["paired_instance_ids_by_arm"][exp.RESET_FREE_ARM]
    assert all(ids == first for ids in result["paired_instance_ids_by_arm"].values())
    assert any(session["family_kind"] == "exact_fsm" for session in sessions)
    assert any(session["family_kind"] == "exact_asp" for session in sessions)


def test_scenario_learn_5571_reset_free_gate_beats_reset_each(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5571-RESET-FREE: candidate gate uses paired evidence."""

    artifact = _complete_artifact(tmp_path)

    assert artifact["continual_harness_candidate"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["gpu_offload_authenticated"] is True
    assert artifact["model_weights_mutated"] is False
    assert artifact["energy_weights_mutated"] is True
    assert artifact["exact_feedback_only"] is True
    assert artifact["harness_state_persisted"][exp.RESET_FREE_ARM] == [
        "exp5569_memory_policy",
        "exp5570_energy_calibrator",
    ]
    assert artifact["new_family_accuracy_by_arm"][exp.RESET_FREE_ARM]["overall"] > artifact[
        "new_family_accuracy_by_arm"
    ][exp.RESET_EACH_ARM]["overall"]
    ci = artifact["confidence_intervals"]["reset_free_vs_reset_each_new_family_accuracy"]
    assert ci["lower"] > 0.0
    assert ci["paired_unit"] == "instance_id"
    assert ci["n_independent_units"] == 120
    assert artifact["prior_family_regression"] <= 0.02
    assert artifact["false_accept_delta"] <= 0.0
    assert artifact["rollback_count"] == 1
    assert artifact["rollback_success"] is True
    assert artifact["cost_receipt"]["by_arm"][exp.RESET_FREE_ARM]["memory_bytes"] > 0
    assert artifact["inference_duration_s"] >= 60.0
    exp.validate_artifact(artifact)


def test_scenario_learn_5571_preconditions_block_without_legacy(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5571-PRECONDITIONS: missing cache/offload blocks cleanly."""

    legacy = tmp_path / "legacy.gguf"
    legacy.write_bytes(b"legacy")
    legacy_pair = [
        {
            "name": "Qwen3.5-0.8B",
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "gpu": 0,
            "model_path": str(legacy),
        },
        {
            "name": "Gemma4-E4B-it",
            "hf_id": "google/gemma-4-E4B-it",
            "gpu": 1,
            "model_path": str(legacy),
        },
    ]
    specs, gate = exp.resolve_model_specs(
        pair_resolver=lambda: legacy_pair,
        gguf_resolver=lambda _model_id: None,
    )
    assert gate["cached_sota_pair_called"] is True
    assert gate["blocked_reason"] == "blocked_missing_sota_cache"
    assert gate["legacy_cpu_model_substituted"] is False
    assert {row["hf_id"] for row in specs} == {exp.QWEN_ID, exp.GEMMA31_ID, exp.GEMMA26_ID}

    artifact = exp.build_artifact(
        root=REPO,
        model_specs=specs,
        gate_receipt=gate,
        device_receipt=_complete_device(),
        live_invocation_receipt=None,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )
    assert artifact["live_model_invoked"] is False
    assert artifact["continual_harness_candidate"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_sota_cache")
    assert artifact["legacy_smoke_models_used"] == []
    exp.validate_artifact(artifact)

    model_specs = _fake_model_specs(tmp_path)
    offload_blocked = exp.build_artifact(
        root=REPO,
        model_specs=model_specs,
        gate_receipt=_complete_gate(model_specs) | {"offload_gate_passed": False},
        device_receipt={"gpu_offload_authenticated": False, "devices": []},
        live_invocation_receipt=None,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )
    assert offload_blocked["honest_verdict"].startswith("blocked_no_cuda_offload")
    assert offload_blocked["gpu_offload_authenticated"] is False
    exp.validate_artifact(offload_blocked)


def test_req_learn_5571_validation_fails_closed_on_gate_mismatches(tmp_path: Path) -> None:
    """REQ-LEARN-5571-5: candidate gate rejects unsafe or inconsistent claims."""

    artifact = _complete_artifact(tmp_path)
    assert exp.validate_artifact(artifact) is True

    bad_cases = [
        ("continuous_self_learning_target", False, "continuous_self_learning_target"),
        ("MODEL_SPECS", [], "MODEL_SPECS"),
        ("live_model_invoked", False, "live_model_invoked"),
        ("gpu_offload_authenticated", False, "gpu_offload_authenticated"),
        ("sessions", [], "sessions"),
        ("n_independent_instances_per_session", 29, "n_independent_instances_per_session"),
        ("arms", list(exp.ARMS[:-1]), "arms"),
        ("model_weights_mutated", True, "model_weights_mutated"),
        ("harness_state_persisted", {}, "harness_state_persisted"),
        ("energy_weights_mutated", False, "energy_weights_mutated"),
        ("exact_feedback_only", False, "exact_feedback_only"),
        ("new_family_accuracy_by_arm", {}, "new_family_accuracy_by_arm"),
        ("backward_retention_by_session", {}, "backward_retention_by_session"),
        ("adaptation_slope", {}, "adaptation_slope"),
        ("false_accept_delta", 0.1, "false_accept_delta"),
        ("rollback_count", 0, "rollback_count"),
        ("rollback_success", False, "rollback_success"),
        ("cost_receipt", {}, "cost_receipt"),
        ("confidence_intervals", {}, "confidence_intervals"),
        ("inference_duration_s", 0.0, "inference_duration_s"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
        ("field_principles", {}, "field_principles"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    non_candidate = deepcopy(artifact)
    non_candidate["confidence_intervals"]["reset_free_vs_reset_each_new_family_accuracy"][
        "lower"
    ] = 0.0
    non_candidate["continual_harness_candidate"] = False
    non_candidate["honest_verdict"] = exp.honest_verdict(non_candidate)
    non_candidate["reproducibility_checksum"] = exp.payload_checksum(non_candidate)
    assert exp.validate_artifact(non_candidate) is True
    assert "not_candidate" in non_candidate["honest_verdict"]

    invalid_claim = deepcopy(non_candidate)
    invalid_claim["continual_harness_candidate"] = True
    invalid_claim["honest_verdict"] = "complete: invalid"
    invalid_claim["reproducibility_checksum"] = exp.payload_checksum(invalid_claim)
    with pytest.raises(ValueError, match="continual_harness_candidate"):
        exp.validate_artifact(invalid_claim)

    missing = deepcopy(artifact)
    missing.pop("continual_harness_candidate")
    missing["reproducibility_checksum"] = exp.payload_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    assert exp.confidence_interval([0.25]) == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
        "n": 1,
    }
    with pytest.raises(ValueError, match="forced"):
        exp._require(False, "forced")


def test_req_learn_5571_defensive_helpers_and_blocker_order(tmp_path: Path) -> None:
    """REQ-LEARN-5571: defensive helpers keep blocked receipts explicit."""

    assert exp.load_exact_rows(tmp_path / "missing-root") == []
    unready = tmp_path / exp.CORPUS_RELATIVE_PATH
    unready.parent.mkdir(parents=True, exist_ok=True)
    unready.write_text('{"corpus_ready": false, "corpus_rows": []}', encoding="utf-8")
    assert exp.load_exact_rows(tmp_path) == []
    assert exp.build_sessions(
        [{"family": "defaults_exceptions", "row_id": "one", "accepted_by_exact_validator": True}]
    ) == []

    assert exp.success_rate([]) == 0.0
    assert exp.false_accept_rate([{"true_label": "valid", "false_accept": False}]) == 0.0
    assert exp.paired_bootstrap_delta([], [], iterations=3) == {
        "mean": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "n_independent_units": 0,
        "n_bootstrap": 3,
        "paired_unit": "instance_id",
    }
    assert exp.confidence_interval([0.1, 0.3])["n"] == 2

    failed_specs, failed_gate = exp.resolve_model_specs(
        pair_resolver=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        gguf_resolver=lambda _model_id: None,
    )
    assert {row["hf_id"] for row in failed_specs} == {
        exp.QWEN_ID,
        exp.GEMMA31_ID,
        exp.GEMMA26_ID,
    }
    assert "RuntimeError" in failed_gate["resolver_error"]

    assert exp.blocked_reason(
        policy_gate=False,
        energy_gate=True,
        cache_gate=True,
        offload_gate=True,
        corpus_gate=True,
        live_gate=True,
    ) == "blocked_upstream_memory_policy_gate"
    assert exp.blocked_reason(
        policy_gate=True,
        energy_gate=False,
        cache_gate=True,
        offload_gate=True,
        corpus_gate=True,
        live_gate=True,
    ) == "blocked_upstream_energy_gate"
    assert exp.blocked_reason(
        policy_gate=True,
        energy_gate=True,
        cache_gate=True,
        offload_gate=True,
        corpus_gate=False,
        live_gate=True,
    ) == "blocked_corpus_unready"
    assert exp.blocked_reason(
        policy_gate=True,
        energy_gate=True,
        cache_gate=True,
        offload_gate=True,
        corpus_gate=True,
        live_gate=False,
    ) == "blocked_live_model_invocation"

    artifact = _complete_artifact(tmp_path)
    bad_legacy = deepcopy(artifact)
    bad_legacy["legacy_smoke_models_used"] = ["Qwen/Qwen3.5-0.8B"]
    bad_legacy["reproducibility_checksum"] = exp.payload_checksum(bad_legacy)
    with pytest.raises(ValueError, match="legacy_smoke_models_used"):
        exp.validate_artifact(bad_legacy)

    bad_research = deepcopy(artifact)
    bad_research["research_conductor_modified"] = True
    bad_research["reproducibility_checksum"] = exp.payload_checksum(bad_research)
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_research)

    assert exp.model_specs_have_headline_qwen("bad") is False
    assert exp.model_specs_have_headline_qwen([object()]) is False
    assert exp.resolve_path(tmp_path, "x.json") == tmp_path / "x.json"
    assert exp.resolve_path(tmp_path, tmp_path / "abs.json") == tmp_path / "abs.json"
    assert exp._load_json(tmp_path / "absent.json") == {}
    assert exp._slope([1.0]) == 0.0
