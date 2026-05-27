"""Tests for Exp 3185 multi-turn repair ladder v5.

Spec refs: REQ-VERIFY-3185, SCENARIO-VERIFY-3185.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import multi_turn_repair_ladder_v5 as mod


REQUIRED_FIELDS = {
    "multi_turn_repair_ladder_v5_ready",
    "gated_skip",
    "gate_state",
    "repair_attempt_count",
    "models_used",
    "repair_targets",
    "transcript_receipts",
    "exact_check_results",
    "repair_success_delta",
    "remaining_blockers",
    "headline_claim_allowed",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake headline repair claims\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text(
        "cached_sota_pair() before live repair\n", encoding="utf-8"
    )
    spec_path = root / mod.SPEC_REL_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        "REQ-VERIFY-3185\nSCENARIO-VERIFY-3185\n"
        "results/experiment_3185_multi_turn_repair_ladder_v5.json\n",
        encoding="utf-8",
    )


def _gate_payload(
    *,
    state: str = "blocked_receipt_precondition",
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    unblocked = state == "unblocked_for_bounded_repair_ladder"
    return {
        "repair_gate_decision_v4_ready": True,
        "repair_gate_state": state,
        "blocker_reasons": blockers
        if blockers is not None
        else [
            "exp3179.clean_rerun_allowed is not true",
            "exp3181.flagged_adversarial is not false",
            "exp3183.repair_call_ready is not true",
        ],
        "allowed_repair_attempt_budget": {
            "enabled": unblocked,
            "max_total_repair_attempts": 4 if unblocked else 0,
            "max_attempts_per_row": 2 if unblocked else 0,
            "max_distinct_rows": 2 if unblocked else 0,
            "requires_mandated_local_sota": True,
            "requires_exact_authority_acceptance": True,
        },
        "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
        "honest_verdict": "complete: fixture" if unblocked else f"{state}: fixture",
    }


def _certificate_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "row-a",
            "record_scope": "expanded_exact_row",
            "canonical_answer": "INVALID",
            "exact_label": "INVALID",
            "checker_result": "reject",
            "checker_authority": "python_ast_runtime_execution",
            "known_false_accept_or_regression": True,
            "counterexample_family": "known_false_accept:arithmetic_code_assertions",
            "exact_authority_complete": True,
            "pilot_certificate": {
                "violated_constraint": "computed_value == claimed_value",
                "minimal_failing_assignment": {"computed_value": 43, "claimed_value": 47},
            },
        },
        {
            "row_id": "row-b",
            "record_scope": "expanded_exact_row",
            "canonical_answer": "UNSAT",
            "exact_label": "UNSAT",
            "checker_result": "reject",
            "checker_authority": "z3_solver",
            "known_false_accept_or_regression": True,
            "counterexample_family": "known_false_accept:smt_constraints",
            "exact_authority_complete": True,
            "pilot_certificate": {
                "violated_constraint": "x > 3 and x < 2",
                "minimal_failing_assignment": {"x": 3},
            },
        },
        {
            "row_id": "row-clean-anchor",
            "record_scope": "expanded_exact_row",
            "canonical_answer": "VALID",
            "exact_label": "VALID",
            "checker_result": "accept",
            "checker_authority": "exact_authority_replay",
            "known_false_accept_or_regression": False,
            "counterexample_family": "exact_row:arithmetic_code_assertions",
            "exact_authority_complete": True,
            "pilot_certificate": {},
        },
    ]


def _write_standard_sources(
    root: Path,
    *,
    gate_state: str = "blocked_receipt_precondition",
    gate_blockers: list[str] | None = None,
    repair_call_ready: bool = False,
) -> None:
    _write_text_sources(root)
    _write_json(root, mod.EXP3184_REL_PATH, _gate_payload(state=gate_state, blockers=gate_blockers))
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "repair_call_ready": repair_call_ready,
            "blocker_reasons": []
            if repair_call_ready
            else ["flagged_adversarial_evidence_present"],
            "certificate_records": _certificate_rows(),
            "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "clean_rerun_allowed": gate_state == "unblocked_for_bounded_repair_ladder",
            "proof_receipts": [{"receipt_id": "receipt-a", "transcript_hash": "abc"}],
            "inference_substrate": {"live_model_calls": 2},
        },
    )
    _write_json(
        root,
        mod.EXP3181_REL_PATH,
        {
            "clean_live_sota_verifier_rerun_v10_ready": True,
            "gated_skip": gate_state != "unblocked_for_bounded_repair_ladder",
            "models_used": [],
            "proof_receipts_used": [{"receipt_id": "receipt-a"}],
            "inference_substrate": {"live_model_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3169_REL_PATH,
        {
            "repair_ladder_materializer_v4_ready": True,
            "gated_skip": True,
            "repair_attempt_count": 0,
            "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
        },
    )


def test_req_verify_3185_spec_anchor_exists() -> None:
    """REQ-VERIFY-3185: OpenSpec declares the v5 repair ladder artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3185" in spec
    assert "SCENARIO-VERIFY-3185" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "multi_turn_repair_ladder_v5_ready" in spec


def test_scenario_verify_3185_blocked_gate_writes_full_no_call_skip(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3185: blocked Exp 3184 gates forbid repair calls."""

    blockers = [
        "exp3179.clean_rerun_allowed is not true",
        "exp3181.flagged_adversarial is not false",
        "exp3183.repair_call_ready is not true",
    ]
    _write_standard_sources(
        tmp_path, gate_state="blocked_receipt_precondition", gate_blockers=blockers
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("repair runner must not be called while Exp 3184 is blocked")

    artifact = mod.build_artifact(
        tmp_path,
        repair_runner=fail_if_called,
        started_s=4.0,
        now_s=6.5,
        tests_run=["focused-3185"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["multi_turn_repair_ladder_v5_ready"] is True
    assert artifact["gated_skip"] is True
    assert artifact["gate_state"] == "blocked_receipt_precondition"
    assert artifact["repair_attempt_count"] == 0
    assert artifact["models_used"] == []
    assert artifact["repair_targets"] == []
    assert artifact["transcript_receipts"] == []
    assert artifact["exact_check_results"] == []
    assert artifact["repair_success_delta"] == pytest.approx(0.0)
    assert artifact["headline_claim_allowed"] is False
    assert all(blocker in artifact["remaining_blockers"] for blocker in blockers)
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["inference_substrate"]["repair_calls"] == 0
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused-3185"]
    assert artifact["honest_verdict"].startswith("blocked_repair_gate_precondition:")

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=1.0,
        now_s=1.25,
        tests_run=["write"],
    )
    assert output == tmp_path / "results/out.json"
    assert json.loads(output.read_text(encoding="utf-8"))["gated_skip"] is True


def test_scenario_verify_3185_unblocked_stubbed_ladder_scores_exact_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3185: two-turn repair uses exact canonical checks only."""

    _write_standard_sources(
        tmp_path,
        gate_state="unblocked_for_bounded_repair_ladder",
        gate_blockers=[],
        repair_call_ready=True,
    )
    cache_checks: list[int] = []

    def cached_pair() -> list[dict[str, Any]]:
        cache_checks.append(1)
        return [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma.gguf",
                "gpu": 0,
            }
        ]

    monkeypatch.setattr(mod, "cached_sota_pair", cached_pair)

    def repair_runner(
        target: Mapping[str, Any],
        turn: int,
        feedback: Mapping[str, Any],
        model_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert model_spec["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
        if target["row_id"] == "row-a" and turn == 1:
            return {"candidate_answer": "VALID", "raw_response": "first try incorrectly says valid"}
        if target["row_id"] == "row-a" and turn == 2:
            assert feedback["expected_canonical"] == "INVALID"
            assert "computed_value == claimed_value" in feedback["violated_constraint"]
            return {"candidate_answer": "INVALID", "raw_response": "corrected to invalid"}
        return {"candidate_answer": target["canonical_answer"], "raw_response": "already exact"}

    artifact = mod.build_artifact(
        tmp_path,
        repair_runner=repair_runner,
        started_s=1.0,
        now_s=3.0,
        tests_run=["focused-live"],
    )

    assert artifact["gated_skip"] is False
    assert artifact["gate_state"] == "unblocked_for_bounded_repair_ladder"
    assert [target["row_id"] for target in artifact["repair_targets"]] == ["row-a", "row-b"]
    assert artifact["repair_attempt_count"] == 3
    assert len(cache_checks) == artifact["repair_attempt_count"]
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert len(artifact["transcript_receipts"]) == 3
    assert len({row["transcript_hash"] for row in artifact["transcript_receipts"]}) == 3
    assert all(len(row["transcript_hash"]) == 64 for row in artifact["transcript_receipts"])
    assert [row["exact_match"] for row in artifact["exact_check_results"]] == [False, True, True]
    assert artifact["repair_success_delta"] == pytest.approx(1.0)
    assert artifact["remaining_blockers"] == []
    assert artifact["headline_claim_allowed"] is True
    assert artifact["inference_substrate"]["live_model_calls"] == 3
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3185_unblocked_without_cache_or_runner_still_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3185: unblocked gates still need local SOTA cache and runner."""

    _write_standard_sources(
        tmp_path,
        gate_state="unblocked_for_bounded_repair_ladder",
        gate_blockers=[],
        repair_call_ready=True,
    )
    monkeypatch.setattr(mod, "cached_sota_pair", lambda: None)

    artifact = mod.build_artifact(tmp_path, repair_runner=None)

    assert artifact["gated_skip"] is True
    assert artifact["repair_attempt_count"] == 0
    assert artifact["repair_targets"][0]["row_id"] == "row-a"
    assert "live repair runner is not configured" in artifact["remaining_blockers"]
    assert (
        "no mandated local SOTA GGUF cache resolved before repair call"
        in artifact["remaining_blockers"]
    )
    assert artifact["headline_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked_repair_runtime:")


def test_req_verify_3185_unblocked_preflight_and_runner_failures_are_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3185: runtime failures remain exact-scored or blocked."""

    _write_standard_sources(
        tmp_path,
        gate_state="unblocked_for_bounded_repair_ladder",
        gate_blockers=[],
        repair_call_ready=True,
    )
    monkeypatch.setattr(mod, "cached_sota_pair", lambda: None)

    artifact = mod.build_artifact(
        tmp_path,
        repair_runner=lambda *args: {"candidate_answer": "INVALID"},
    )

    assert artifact["gated_skip"] is True
    assert artifact["repair_attempt_count"] == 0
    assert (
        "no mandated local SOTA GGUF cache resolved before repair call"
        in artifact["remaining_blockers"]
    )

    monkeypatch.setattr(
        mod,
        "cached_sota_pair",
        lambda: [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma.gguf",
            }
        ],
    )

    def raising_runner(*args: Any) -> dict[str, Any]:
        raise RuntimeError("unit boom")

    failed = mod.build_artifact(tmp_path, repair_runner=raising_runner)

    assert failed["gated_skip"] is False
    assert failed["repair_attempt_count"] == 4
    assert all(row["exact_match"] is False for row in failed["exact_check_results"])
    assert failed["repair_success_delta"] == pytest.approx(0.0)
    assert failed["headline_claim_allowed"] is False

    limited_receipts: list[dict[str, Any]] = []
    limited_checks: list[dict[str, Any]] = []
    limited_models: list[str] = []
    limited_blockers: list[str] = []
    mod.run_ladder(
        root_path=tmp_path,
        repair_targets=[{"row_id": "a"}, {"row_id": "b"}],
        budget={"max_total_repair_attempts": 1, "max_attempts_per_row": 2},
        repair_runner=lambda *args: {"candidate_answer": "wrong"},
        preflight_checker=lambda *args: {
            "ok": True,
            "model_spec": {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"},
            "receipt_evidence": [{"receipt_id": "r"}],
        },
        transcript_receipts=limited_receipts,
        exact_check_results=limited_checks,
        models_used=limited_models,
        remaining_blockers=limited_blockers,
    )
    assert len(limited_checks) == 1

    empty_receipts = tmp_path / "empty-preflight"
    empty_receipts.mkdir()
    preflight = mod.repair_preflight_from_local_cache(empty_receipts, {"row_id": "a"}, 1)
    assert (
        "exp3179.local_sota_receipt_smoke_v3_ready is not true before repair call"
        in preflight["blockers"]
    )
    assert "exp3179.clean_rerun_allowed is not true before repair call" in preflight["blockers"]
    assert "no local SOTA proof receipts available before repair call" in preflight["blockers"]


def test_req_verify_3185_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3185: helpers fail closed and validation rejects unsafe shapes."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.mapping_rows([{"a": 1}, [], {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert mod.mapping_rows({"not": "rows"}) == []
    assert mod.canonicalize_answer(" accept ") == "VALID"
    assert mod.canonicalize_answer("false") == "INVALID"
    assert mod.canonicalize_answer("forty two") == "FORTY TWO"
    assert mod.rate(1, 0) == pytest.approx(0.0)
    assert mod.rate(3, 2) == pytest.approx(1.0)
    assert mod.duration(5.0, 3.0) == pytest.approx(0.0)
    assert mod.sha256_file(tmp_path / "missing") is None
    assert mod.first_text(None, " ", "") == ""
    assert mod.runtime_preconditions(
        tmp_path,
        {"counterexample_certificate_expansion_v3_ready": False, "repair_call_ready": False},
        [],
        lambda *args: {"candidate_answer": "INVALID"},
    ) == [
        "exp3183.counterexample_certificate_expansion_v3_ready is not true",
        "exp3183.repair_call_ready is not true",
        "no certificate-backed repair targets selected",
    ]

    targets = mod.select_repair_targets(
        [
            {"row_id": "", "known_false_accept_or_regression": True},
            {
                "row_id": "clean",
                "known_false_accept_or_regression": False,
                "exact_authority_complete": True,
                "exact_label": "VALID",
            },
            {
                "row_id": "fragment",
                "counterexample_family": "fragment_code:parser_repair",
                "exact_label": "INVALID",
                "checker_result": "reject",
                "exact_authority_complete": True,
            },
        ],
        max_targets=2,
    )
    assert [target["row_id"] for target in targets] == ["fragment"]

    target = {
        "row_id": "row-a",
        "canonical_answer": "INVALID",
        "exact_label": "INVALID",
        "checker_authority": "python_ast_runtime_execution",
    }
    check = mod.exact_semantic_check(target, {"candidate_answer": "reject"}, "hash-a", 1)
    assert check["accepted_by_exact_authority"] is True
    feedback = mod.counterexample_feedback(target, check)
    assert feedback["row_id"] == "row-a"
    assert feedback["candidate_canonical"] == "INVALID"

    artifact = {
        "multi_turn_repair_ladder_v5_ready": True,
        "gated_skip": True,
        "gate_state": "blocked_receipt_precondition",
        "repair_attempt_count": 0,
        "models_used": [],
        "repair_targets": [],
        "transcript_receipts": [],
        "exact_check_results": [],
        "repair_success_delta": 0.0,
        "remaining_blockers": ["blocked"],
        "headline_claim_allowed": False,
        "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
        "honest_verdict": "blocked_repair_gate_precondition: blocked",
    }
    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="finite rate"):
        mod.validate_artifact(artifact | {"repair_success_delta": float("nan")})
    with pytest.raises(ValueError, match="finite rate"):
        mod.validate_artifact(artifact | {"repair_success_delta": 2.0})
    with pytest.raises(ValueError, match="gated skip"):
        mod.validate_artifact(artifact | {"repair_attempt_count": 1})
    with pytest.raises(ValueError, match="unblocked gate"):
        mod.validate_artifact(artifact | {"gated_skip": False, "repair_attempt_count": 0})
    with pytest.raises(ValueError, match="transcript receipts"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [],
                "exact_check_results": [{"accepted_by_exact_authority": True}],
                "honest_verdict": "complete: missing receipt",
            }
        )
    with pytest.raises(ValueError, match="mandated SOTA models"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": [],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [{"transcript_hash": "a" * 64}],
                "exact_check_results": [{"accepted_by_exact_authority": True}],
                "honest_verdict": "complete: missing model",
            }
        )
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [{"transcript_hash": "short"}],
                "exact_check_results": [{"accepted_by_exact_authority": True}],
                "honest_verdict": "complete: bad hash",
            }
        )
    with pytest.raises(ValueError, match="mandated SOTA"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": ["legacy/small"],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [{"transcript_hash": "a" * 64}],
                "exact_check_results": [{"accepted_by_exact_authority": True}],
                "honest_verdict": "complete: invalid legacy",
            }
        )
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [{"transcript_hash": "a" * 64}],
                "exact_check_results": [{"accepted_by_exact_authority": True}],
                "honest_verdict": "blocked_after_live_attempt",
            }
        )
    with pytest.raises(ValueError, match="headline claim"):
        mod.validate_artifact(
            artifact
            | {
                "gated_skip": False,
                "gate_state": "unblocked_for_bounded_repair_ladder",
                "repair_attempt_count": 1,
                "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "repair_targets": [{"row_id": "row-a"}],
                "transcript_receipts": [{"transcript_hash": "a" * 64}],
                "exact_check_results": [{"accepted_by_exact_authority": False}],
                "headline_claim_allowed": True,
                "honest_verdict": "complete: unsupported headline",
            }
        )
