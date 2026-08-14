"""Tests for Exp6422 held-family policy safety audit.

Spec refs: REQ-ARC-ARM-6422,
SCENARIO-ARC-ARM-6422-HASH-AND-MISSING-INPUTS,
SCENARIO-ARC-ARM-6422-HELD-REPLAY,
SCENARIO-ARC-ARM-6422-RECOMPUTE-AND-ATTACKS,
SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6422_arc_held_family_policy_safety_audit as exp6422


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6422.ARC_SPEC_RELATIVE_PATH


def _artifact(tmp_path: Path) -> dict[str, Any]:
    commands = (
        exp6422.RUN_COMMAND,
        exp6422.FOCUSED_TEST_COMMAND,
        exp6422.COVERAGE_RUN_COMMAND,
        exp6422.COVERAGE_REPORT_COMMAND,
    )
    return exp6422.run(
        date="20260814",
        repo_root=REPO,
        result_path=tmp_path / exp6422.RESULT_RELATIVE_PATH.name,
        duration_s=2.0,
        tests_run=commands,
        test_exit_codes={command: 0 for command in commands},
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6422.payload_checksum(payload)
    return payload


def test_req_arc_arm_6422_spec_declares_held_policy_audit_contract() -> None:
    """REQ-ARC-ARM-6422: OpenSpec names the audit fields and held replay."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6422") :]
    for marker in (
        "SCENARIO-ARC-ARM-6422-HASH-AND-MISSING-INPUTS",
        "SCENARIO-ARC-ARM-6422-HELD-REPLAY",
        "SCENARIO-ARC-ARM-6422-RECOMPUTE-AND-ATTACKS",
        "SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY",
        exp6422.RESULT_RELATIVE_PATH.as_posix(),
        exp6422.MANDATED_GEMMA_MODEL_ID,
        "AutoTokenizer",
    ):
        assert marker in section
    for field in exp6422.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_arm_6422_hashes_inputs_and_preserves_missing_sidecars() -> None:
    """SCENARIO-ARC-ARM-6422-HASH-AND-MISSING-INPUTS: inputs are explicit."""

    expected = exp6422.expected_and_available_exp6421_inputs(REPO)
    hashes = exp6422.upstream_hashes(REPO)
    missing = exp6422.missing_input_findings(expected, hashes)
    held = exp6422.held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks(
        REPO
    )

    assert expected["exp6421_artifact"]["available"] is True
    assert expected["exp6421_sidecar_files"]["available"] is False
    assert any(row["input"] == "exp6421_sidecar_files" for row in missing)
    assert hashes["artifacts"]["exp6421"]["exists"] is True
    assert hashes["sources"]["canonical_live_agent"]["exists"] is True
    assert hashes["checkers"]["adversarial_verify"]["exists"] is True
    assert hashes["determination_records"]["determination_preservation_lint"]["exists"] is True
    assert held["exists"] is True
    assert held["sealed_before_exp6421_outcomes"] is True
    assert held["duplicate_window_count"] == 0
    assert held["already_credited_target_count"] == 0
    assert held["disjoint_from_exp6400"] is True
    assert exp6422._mtime_iso(REPO / "not-a-real-exp6422-input") is None

    missing_tokenizers = copy.deepcopy(hashes)
    missing_tokenizers["model_receipts"]["embedded_tokenizer_receipts"] = {}
    token_missing = exp6422.missing_input_findings(expected, missing_tokenizers)
    assert any(row["input"] == "embedded_gguf_tokenizer_receipts" for row in token_missing)


def test_scenario_arc_arm_6422_held_replay_recomputes_policy_influence() -> None:
    """SCENARIO-ARC-ARM-6422-HELD-REPLAY: held rows match except route."""

    exp6421_artifact = exp6422.load_exp6421_artifact(REPO)
    held_windows = exp6422.load_held_windows(REPO)
    replay = exp6422.run_matched_held_policy_replay(
        models=exp6421_artifact["MODEL_SPECS"],
        held_windows=held_windows,
    )
    recomputed = (
        replay[
            "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
        ]
    )
    delta = recomputed["delta"]
    matched = replay["matched_held_off_and_opt_in_work_receipts"]["matched_receipt"]

    assert replay["matched_held_off_and_opt_in_work_receipts"]["held_window_count"] == 8
    assert matched["matched_contract_passed"] is True
    assert replay["matched_held_off_and_opt_in_work_receipts"]["row_count"] == 48
    assert delta["route_firing_delta"] == 24
    assert delta["changed_legal_executed_action_count"] == 24
    assert delta["legal_action_rate_delta"] == 0.0
    assert delta["exact_observation_consistency_delta"] == 0.0
    assert delta["harmful_regression_delta"] == 0
    assert delta["solve_or_level_credit_delta"] == 0
    assert all(
        row["executed_action"] in row["legal_actions"]
        for row in replay["matched_held_off_and_opt_in_work_receipts"]["rows"]
    )


def test_scenario_arc_arm_6422_attack_matrix_fails_closed() -> None:
    """SCENARIO-ARC-ARM-6422-RECOMPUTE-AND-ATTACKS: attacks fail closed."""

    exp6421_artifact = exp6422.load_exp6421_artifact(REPO)
    replay = exp6422.run_matched_held_policy_replay(
        models=exp6421_artifact["MODEL_SPECS"],
        held_windows=exp6422.load_held_windows(REPO),
    )
    rows = replay["matched_held_off_and_opt_in_work_receipts"]["rows"]
    model_ids = [str(row["hf_id"]) for row in exp6421_artifact["MODEL_SPECS"]]
    attacks = exp6422.attack_matrix(rows=rows, model_ids=model_ids)

    assert {row["attack"] for row in attacks} == set(exp6422.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks)
    assert exp6422._expect_value_error("accepted", lambda: None)["fail_closed"] is False

    model_swap = copy.deepcopy(rows)
    model_swap[0]["model_id"] = "unsloth/bad-model-GGUF"
    with pytest.raises(ValueError, match="model substitution"):
        exp6422.validate_audit_rows(model_swap, model_ids)

    exhaustive = copy.deepcopy(rows)
    exhaustive[0]["exhaustive_search_count"] = 1
    with pytest.raises(ValueError, match="exhaustive_search_count"):
        exp6422.validate_audit_rows(exhaustive, model_ids)

    retuned = copy.deepcopy(rows)
    retuned[0]["hidden_retuning_count"] = 1
    with pytest.raises(ValueError, match="hidden_retuning_count"):
        exp6422.validate_audit_rows(retuned, model_ids)

    outer_loop = copy.deepcopy(rows)
    outer_loop[0]["outer_loop_re_used"] = True
    with pytest.raises(ValueError, match="outer_loop_re_used"):
        exp6422.validate_audit_rows(outer_loop, model_ids)

    not_candidate = copy.deepcopy(rows)
    not_candidate[1]["candidate_actions"] = [4]
    not_candidate[1]["executed_action"] = 5
    with pytest.raises(ValueError, match="action substitution"):
        exp6422.validate_audit_rows(not_candidate, model_ids)

    missing_arm = copy.deepcopy(rows)
    missing_arm[1]["arm"] = "unexpected_arm"
    with pytest.raises(ValueError, match="missing matched arm"):
        exp6422.validate_audit_rows(missing_arm, model_ids)

    mismatched = copy.deepcopy(rows)
    mismatched[1]["prompt_hash"] = "sha256:changed"
    with pytest.raises(ValueError, match="matched arm mismatch"):
        exp6422.validate_audit_rows(mismatched, model_ids)

    no_change = copy.deepcopy(rows)
    no_change[1]["executed_action"] = no_change[0]["executed_action"]
    with pytest.raises(ValueError, match="expected opt-in action change"):
        exp6422.validate_audit_rows(no_change, model_ids)


def test_scenario_arc_arm_6422_artifact_schema_and_no_solve_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY: artifact is complete."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6422.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6422.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp6422.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["models_used"][0] == exp6422.CANONICAL_GENERATOR_MODEL_ID
    assert exp6422.MANDATED_GEMMA_MODEL_ID in artifact["models_used"]
    assert artifact["cached_sota_pair_receipts"]["mandated_gemma_resolved_through_cached_sota_pair"] is True
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["shipped_default_preserved"] is True
    assert artifact["source_access_count"] == 0
    assert artifact["exhaustive_search_count"] == 0
    assert artifact["per_game_adapter_count"] == 0
    assert artifact["hidden_retuning_count"] == 0
    assert artifact["outer_loop_re_used"] is False
    assert artifact["level_solve_claimed"] is False
    assert artifact["solve_registry_modified"] is False
    assert artifact["public_arc_claim_eligibility"] is False
    assert artifact["arc_held_policy_safety_audit_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == exp6422.payload_checksum(artifact)
    exp6422.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_access_count", 1, "source_access_count"),
        ("exhaustive_search_count", 1, "exhaustive_search_count"),
        ("per_game_adapter_count", 1, "per_game_adapter_count"),
        ("hidden_retuning_count", 1, "hidden_retuning_count"),
        ("outer_loop_re_used", True, "outer_loop_re_used"),
        ("level_solve_claimed", True, "level_solve_claimed"),
        ("solve_registry_modified", True, "solve_registry_modified"),
        ("public_arc_claim_eligibility", True, "public_arc_claim_eligibility"),
        ("arc_held_policy_safety_audit_ready_score", 0.0, "ready_score"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_arm_6422_validation_rejects_forbidden_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY: drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6422.validate_artifact(bad)


def test_req_arc_arm_6422_validation_rejects_nested_drift(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6422: nested audit drift fails validation."""

    artifact = _artifact(tmp_path)

    checksum = copy.deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6422.validate_artifact(checksum)

    missing = copy.deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6422.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a["held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks"].__setitem__(
                "sealed_before_exp6421_outcomes", False
            ),
            "held_manifest",
        ),
        (
            lambda a: a["solve_registry_precheck_path_hash_and_results"].__setitem__(
                "all_held_games_prechecked", False
            ),
            "solve_registry_precheck",
        ),
        (
            lambda a: a["matched_held_off_and_opt_in_work_receipts"]["matched_receipt"].__setitem__(
                "matched_contract_passed", False
            ),
            "matched_held",
        ),
        (
            lambda a: a[
                "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
            ]["delta"].__setitem__("changed_legal_executed_action_count", 0),
            "recomputed",
        ),
        (
            lambda a: a[
                "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
            ]["delta"].__setitem__("harmful_regression_delta", 1),
            "recomputed",
        ),
        (
            lambda a: a["attack_matrix"][0].__setitem__("fail_closed", False),
            "attack_matrix",
        ),
        (
            lambda a: a["authenticated_model_and_live_policy_receipts"].__setitem__(
                "all_receipts_authentic", False
            ),
            "authenticated_model",
        ),
        (
            lambda a: a["embedded_gguf_tokenizer_receipts"][exp6422.MANDATED_GEMMA_MODEL_ID].__setitem__(
                "ok", False
            ),
            "embedded_gguf_tokenizer_receipts",
        ),
        (
            lambda a: a["protected_files_unchanged"]["scripts/research_conductor.py"].__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
        (lambda a: a.__setitem__("shipped_default_preserved", False), "shipped_default_preserved"),
        (lambda a: a.__setitem__("models_used", []), "models_used"),
        (
            lambda a: a.__setitem__("models_used", [exp6422.CANONICAL_GENERATOR_MODEL_ID]),
            "models_used",
        ),
        (
            lambda a: a["cached_sota_pair_receipts"].__setitem__(
                "mandated_gemma_resolved_through_cached_sota_pair", False
            ),
            "cached_sota_pair_receipts",
        ),
        (lambda a: a.__setitem__("autotokenizer_usage_count", 1), "autotokenizer_usage_count"),
        (
            lambda a: a[
                "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
            ]["delta"].__setitem__("route_firing_delta", 0),
            "recomputed",
        ),
        (
            lambda a: a[
                "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
            ]["delta"].__setitem__("legal_action_rate_delta", 1.0),
            "recomputed",
        ),
        (
            lambda a: a[
                "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
            ]["delta"].__setitem__("exact_observation_consistency_delta", 1.0),
            "recomputed",
        ),
        (
            lambda a: a["field_principles"].pop("attack_matrix"),
            "field_principles",
        ),
        (
            lambda a: a["field_principles"].pop(f"attack_matrix.{exp6422.ATTACK_IDS[0]}"),
            "field_principles",
        ),
        (lambda a: a.__setitem__("honest_verdict", "blocked"), "honest_verdict"),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6422.validate_artifact(bad)


def test_req_arc_arm_6422_run_can_return_without_writing(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6422: non-writing run still builds a valid artifact."""

    output = tmp_path / "not-written.json"
    artifact = exp6422.run(
        date="20260814",
        repo_root=REPO,
        result_path=output,
        write=False,
    )

    assert output.exists() is False
    assert artifact["duration_s"] >= 0.01
    exp6422.validate_artifact(artifact)


def test_req_arc_arm_6422_build_artifact_uses_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6422: build_artifact validates runner output."""

    artifact = _artifact(tmp_path)

    def fake_run(**kwargs):
        assert kwargs["date"] == "20260814"
        assert kwargs["write"] is True
        return artifact

    monkeypatch.setattr(exp6422, "run", fake_run)

    built = exp6422.build_artifact(
        tmp_path,
        date="20260814",
        output_path=tmp_path / "out.json",
    )
    assert built["arc_held_policy_safety_audit_ready_score"] == 1.0
