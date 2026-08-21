"""Tests for Exp6491 local GGUF atomic factor proposal stream.

Spec refs: REQ-VERIFY-6491, SCENARIO-VERIFY-6491-GATES,
SCENARIO-VERIFY-6491-RAW-BYTES, SCENARIO-VERIFY-6491-HELD-ISOLATION,
SCENARIO-VERIFY-6491-COMPILER-AUTHORITY,
SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS, SCENARIO-VERIFY-6491-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6491_sota_factor_proposal_stream as mod
from carnot import task_runtime_receipts as receipts
import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"


class StubRuntime:
    """Small runtime that returns one raw response per event without a model."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = dict(spec)
        self.load_receipt = {
            "model_hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "model_file_sha256": receipts.sha256_file(spec["model_path"]),
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "actual_backend": "llama_cpp.Llama",
            "runtime_version": "stub-llama-cpp-python",
            "embedded_tokenizer": True,
            "tokenizer_source": "gguf_embedded",
            "external_tokenizer_used": False,
            "n_gpu_layers": mod.N_GPU_LAYERS,
            "load_status": "loaded",
            "load_failure": None,
            "load_wall_time_s": 0.01,
            "backend": "stub",
            "gpu": spec.get("gpu"),
        }

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        event: dict[str, Any],
        max_tokens: int,
        seed: int,
    ) -> dict[str, Any]:
        assert request_id
        assert max_tokens == mod.MAX_TOKENS
        assert seed >= mod.RANDOM_SEED
        assert "final_exact" not in prompt
        if self.spec["hf_id"] == QWEN_ID:
            text = json.dumps(
                {
                    "factor_id": "qwen_depth_floor",
                    "kind": "branch_depth_at_least",
                    "scope": ["event"],
                    "weight": 1,
                    "threshold": 1,
                },
                sort_keys=True,
            )
        else:
            threshold = max(1, int(event["visible_context"]["candidate_count_under_partial"]) - 1)
            text = json.dumps(
                {
                    "factor_id": "gemma_candidate_floor",
                    "kind": "candidate_count_at_least",
                    "scope": ["event"],
                    "weight": 1,
                    "threshold": threshold,
                },
                sort_keys=True,
            )
        return {
            "output_text": text,
            "finish_reason": "stop",
            "timed_out": False,
            "duration_s": 0.02,
            "prompt_token_count": 64,
            "completion_token_count": 24,
            "backend_details": {"stub_runtime": True},
        }

    def close(self) -> None:
        self.closed = True


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        QWEN_ID: tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        GEMMA26_ID: tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        GEMMA31_ID: tmp_path / "gemma-4-31B-it-Q4_K_M.gguf",
    }
    for model_id, path in paths.items():
        path.write_bytes(b"GGUF test weight bytes for " + model_id.encode("utf-8"))
    return paths


def _artifact(tmp_path: Path) -> dict[str, Any]:
    model_paths = _fake_model_paths(tmp_path / "models")

    def cache_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        path = model_paths.get(hf_id)
        return str(path) if path else None

    def pair_resolver(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        assert model_indices is None
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": QWEN_ID,
                "gpu": gpu_indices[0],
                "model_path": str(model_paths[QWEN_ID]),
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": GEMMA31_ID,
                "gpu": gpu_indices[1],
                "model_path": str(model_paths[GEMMA31_ID]),
            },
        ]

    return mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_dir=tmp_path / "raw",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        cache_resolver=cache_resolver,
        pair_resolver=pair_resolver,
        runtime_factory=StubRuntime,
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_verify_6491_spec_declares_local_gguf_factor_stream() -> None:
    """REQ-VERIFY-6491: OpenSpec owns the proposal stream contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6491") : text.index("REQ-VERIFY-6486")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6491-GATES",
        "SCENARIO-VERIFY-6491-RAW-BYTES",
        "SCENARIO-VERIFY-6491-HELD-ISOLATION",
        "SCENARIO-VERIFY-6491-COMPILER-AUTHORITY",
        "SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS",
        "SCENARIO-VERIFY-6491-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6491_gates_models_and_raw_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6491-GATES/RAW-BYTES: gates and bytes are bound."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    receipts_by_id = {row["model_hf_id"]: row for row in artifact["model_load_receipts"]}
    spec_by_id = {row["hf_id"]: row for row in artifact["model_specs"]}

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_local_proposal_stream"
    assert artifact["honest_verdict"].startswith("complete_local_proposal_stream:")
    assert artifact["factor_proposal_stream_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    gates = {row["artifact_id"]: row for row in artifact["upstream_gate_receipts"]}
    assert gates["exp6488"]["field"] == "v560_lineage_lock_ready_score"
    assert gates["exp6488"]["expected"] == 1.0
    assert gates["exp6488"]["observed"] == 1.0
    assert gates["exp6489"]["field"] == "trajectory_contract_ready_score"
    assert gates["exp6489"]["expected"] == 1.0
    assert gates["exp6489"]["observed"] == 1.0
    assert artifact["prior_lineage_retirement_receipt"]["artifact_id"] == "exp6463"
    assert artifact["prior_lineage_retirement_receipt"]["observed_ready_score"] == 0.0
    assert "fixed-policy corpus v2 finished" in artifact["prior_lineage_retirement_receipt"]["prior_honest_verdict"]

    assert set(spec_by_id) == {QWEN_ID, GEMMA26_ID, GEMMA31_ID}
    assert spec_by_id[QWEN_ID]["selected_for_inference"] is True
    assert spec_by_id[GEMMA31_ID]["selected_for_inference"] is True
    assert spec_by_id[GEMMA26_ID]["selected_for_inference"] is False
    assert spec_by_id[GEMMA26_ID]["resource_disposition"] == "not_loaded_resource_budget_two_family_precommit"
    assert spec_by_id[QWEN_ID]["quantization"] == "Q4_K_M"
    assert spec_by_id[GEMMA31_ID]["model_family"] == "gemma"

    assert set(receipts_by_id) == {QWEN_ID, GEMMA31_ID}
    for receipt in receipts_by_id.values():
        assert receipt["load_status"] == "loaded"
        assert receipt["embedded_tokenizer"] is True
        assert receipt["tokenizer_source"] == "gguf_embedded"
        assert receipt["external_tokenizer_used"] is False
        assert receipt["model_file_sha256"].startswith("sha256:")
        assert receipt["n_gpu_layers"] == mod.N_GPU_LAYERS

    raw = artifact["raw_request_response_receipts"]
    assert raw["all_raw_bytes_written_before_parse"] is True
    assert raw["request_response_pair_count"] == len(artifact["proposal_rows"])
    for row in raw["rows"]:
        path = Path(row["path"])
        assert path.is_file()
        assert receipts.sha256_file(path) == row["sha256"]
        assert row["written_before_parse"] is True


def test_scenario_verify_6491_held_isolation_non_authority_and_compile_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6491-HELD-ISOLATION/COMPILER-AUTHORITY/ROWS."""

    artifact = _artifact(tmp_path)
    events = artifact["frozen_event_manifest"]["events"]
    proposal_rows = artifact["proposal_rows"]
    compile_rows = artifact["exact_compile_rows"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert all(row["split"] == "development" for row in events)
    assert artifact["held_isolation_receipts"]["selected_event_splits"] == ["development"]
    assert artifact["held_isolation_receipts"]["held_rows_in_request_context_count"] == 0
    assert artifact["held_isolation_receipts"]["final_outcome_fields_in_request_context_count"] == 0
    assert artifact["prompt_commitment"]["prompts_written_before_model_access"] is True
    assert artifact["prompt_commitment"]["grammar_backend"] == "none"
    assert artifact["prompt_commitment"]["retry_to_valid_loop_used"] is False
    assert artifact["prompt_commitment"]["rank_and_select_loop_used"] is False

    assert len(proposal_rows) == len(events) * 2
    assert len(compile_rows) == len(proposal_rows)
    assert all(row["attempt_index"] == 0 for row in proposal_rows)
    assert all(row["retry_count_after_response"] == 0 for row in proposal_rows)
    assert all(row["rank_and_select_candidates"] == 1 for row in proposal_rows)
    assert all(row["model_output_is_oracle"] is False for row in proposal_rows)
    assert all(row["answer_field_present"] is False for row in proposal_rows)
    assert all(row["label_field_present"] is False for row in proposal_rows)
    assert all(row["release_authority_claimed"] is False for row in proposal_rows)

    assert all(row["model_output_is_oracle"] is False for row in compile_rows)
    assert all(row["exact_compiler_is_oracle_for_disposition"] is True for row in compile_rows)
    assert all(row["compile_outcome"] in mod.COMPILE_OUTCOMES for row in compile_rows)
    assert {row["compile_outcome"] for row in compile_rows} == {"accept"}
    assert aggregate == mod.recompute_aggregates_from_rows(artifact["per_unit_rows"])
    assert aggregate["factor_proposal_stream_ready_score_from_rows"] == 1.0
    assert aggregate["completed_model_family_count"] == 2
    assert artifact["non_authority_receipts"]["model_output_label_count"] == 0
    assert artifact["non_authority_receipts"]["model_output_answer_count"] == 0
    assert artifact["non_authority_receipts"]["model_output_release_authority_count"] == 0


def test_scenario_verify_6491_boundary_attacks_and_validation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS: attack mutations fail."""

    artifact = _artifact(tmp_path)
    attacks = {row["attack_id"]: row for row in artifact["boundary_attack_matrix"]["rows"]}

    assert set(attacks) == set(mod.BOUNDARY_ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert all(row["readiness_promoted"] is False for row in attacks.values())
    assert attacks["wrong_tokenizer_path"]["blocked_by"] == "embedded_gguf_tokenizer_receipt"

    missing_raw = deepcopy(artifact)
    raw_path = Path(missing_raw["raw_request_response_receipts"]["rows"][0]["path"])
    raw_path.unlink()
    assert "raw byte receipt missing" in mod.validate_artifact(missing_raw)

    wrong_tokenizer = deepcopy(artifact)
    wrong_tokenizer["model_load_receipts"][0]["embedded_tokenizer"] = False
    wrong_tokenizer["model_load_receipts"][0]["tokenizer_source"] = "external_hf_tokenizer"
    _with_checksum(wrong_tokenizer)
    assert "model_load_receipts must use embedded GGUF tokenizers" in mod.validate_artifact(
        wrong_tokenizer
    )

    identity_omission = deepcopy(artifact)
    del identity_omission["proposal_rows"][0]["model_hf_id"]
    _with_checksum(identity_omission)
    assert "proposal row missing model identity" in mod.validate_artifact(identity_omission)

    retry = deepcopy(artifact)
    retry["proposal_rows"][0]["retry_count_after_response"] = 1
    _with_checksum(retry)
    assert "proposal rows must be one-shot with zero retries" in mod.validate_artifact(retry)

    posthoc = deepcopy(artifact)
    extra = deepcopy(posthoc["proposal_rows"][0])
    extra["event_id"] = "posthoc-event"
    posthoc["proposal_rows"].append(extra)
    _with_checksum(posthoc)
    assert "proposal rows must match frozen event/model cross product" in mod.validate_artifact(
        posthoc
    )

    answer = deepcopy(artifact)
    answer["proposal_rows"][0]["answer_field_present"] = True
    _with_checksum(answer)
    assert "model output must not contain answer, label, verifier, or release fields" in mod.validate_artifact(
        answer
    )

    attack_open = deepcopy(artifact)
    attack_open["boundary_attack_matrix"]["rows"][0]["fail_closed"] = False
    _with_checksum(attack_open)
    assert "boundary attacks must fail closed" in mod.validate_artifact(attack_open)


def test_scenario_verify_6491_parser_and_compiler_reject_bad_proposals(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6491-COMPILER-AUTHORITY: parser and compiler are strict."""

    artifact = _artifact(tmp_path)
    event = artifact["frozen_event_manifest"]["events"][0]

    proposal, parse_receipt = mod.parse_atomic_factor_response(
        json.dumps({"answer": "42", "factor_id": "bad"})
    )
    assert proposal is None
    assert parse_receipt["boundary_violation"] is True
    assert parse_receipt["forbidden_keys"] == ["answer"]

    proposal, parse_receipt = mod.parse_atomic_factor_response("not json")
    assert proposal is None
    assert parse_receipt["parse_status"] == "json_decode_error"

    valid = {
        "factor_id": "depth_floor",
        "kind": "branch_depth_at_least",
        "scope": ["event"],
        "weight": 1,
        "threshold": 1,
    }
    accepted = mod.compile_atomic_factor(valid, event, seen_semantic_hashes=set())
    assert accepted["compile_outcome"] == "accept"

    seen = {accepted["semantic_hash"]}
    duplicate = mod.compile_atomic_factor(valid, event, seen_semantic_hashes=seen)
    assert duplicate["compile_outcome"] == "duplicate"

    rejected = mod.compile_atomic_factor(
        {**valid, "threshold": event["visible_context"]["branch_depth"] + 1},
        event,
        seen_semantic_hashes=set(),
    )
    assert rejected["compile_outcome"] == "reject"
    assert rejected["reason"] == "semantic_predicate_false_on_visible_event"

    no_proposal = mod.compile_no_proposal(event, reason="empty_response")
    assert no_proposal["compile_outcome"] == "no_proposal"
    timeout = mod.compile_timeout(event, reason="generation_timeout")
    assert timeout["compile_outcome"] == "timeout"


def test_scenario_verify_6491_blocked_and_reject_branches_are_deterministic(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6491-GATES/COMPILER-AUTHORITY: blocked paths are explicit."""

    artifact = _artifact(tmp_path / "base")
    event = artifact["frozen_event_manifest"]["events"][0]

    assert mod._read_json(tmp_path / "missing.json") is None
    assert mod._load_existing(tmp_path / "missing.json") is None
    assert mod.quant_from_filename("model-no-quant.gguf") == "unknown"
    assert mod._model_family("other/model-GGUF") == "other"

    model_paths = _fake_model_paths(tmp_path / "fallback")

    def cache_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        return str(model_paths[hf_id])

    def gemma_only_pair(**_: Any) -> list[dict[str, Any]]:
        return [
            {"hf_id": GEMMA26_ID, "name": "Gemma26", "gpu": 0, "model_path": str(model_paths[GEMMA26_ID])},
            {"hf_id": GEMMA31_ID, "name": "Gemma31", "gpu": 1, "model_path": str(model_paths[GEMMA31_ID])},
        ]

    fallback_specs = mod.resolve_model_specs(
        cache_resolver=cache_resolver,
        pair_resolver=gemma_only_pair,
    )
    selected = {row["hf_id"]: row for row in fallback_specs if row["selected_for_inference"]}
    assert set(selected) == {QWEN_ID, GEMMA26_ID}
    assert selected[QWEN_ID]["resource_disposition"] == "selected_from_cached_mandated_family_fallback"

    real_rows = json.loads((REPO / mod.EXP6489_RELATIVE_PATH).read_text())["raw_trajectory_rows"]
    fallback_source = [
        {**row, "checkpoint_id": "first"}
        for row in real_rows
        if row["split"] == "development" and row["backend"] == "z3"
    ][:2]
    exp6489 = tmp_path / "fallback_exp6489.json"
    exp6489.write_text(json.dumps({"raw_trajectory_rows": fallback_source}), encoding="utf-8")
    fallback_events = mod.select_development_events(REPO, exp6489_path=exp6489, max_events=2)
    assert len(fallback_events) == 2

    blocked = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        raw_dir=tmp_path / "blocked_raw",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        cache_resolver=lambda *_: None,
        pair_resolver=lambda **_: None,
        runtime_factory=StubRuntime,
    )
    assert blocked["status"] == "blocked_local_proposal_stream"
    assert blocked["factor_proposal_stream_ready_score"] == 0.0
    assert mod.validate_artifact(blocked) == []

    class BranchRuntime(StubRuntime):
        def generate(
            self,
            prompt: str,
            *,
            request_id: str,
            event: dict[str, Any],
            max_tokens: int,
            seed: int,
        ) -> dict[str, Any]:
            base = super().generate(
                prompt,
                request_id=request_id,
                event=event,
                max_tokens=max_tokens,
                seed=seed,
            )
            if request_id.startswith("exp6491-00-00"):
                return {**base, "timed_out": True}
            if request_id.startswith("exp6491-00-01") and self.spec["hf_id"] == QWEN_ID:
                return {**base, "output_text": ""}
            if request_id.startswith("exp6491-01-00") and self.spec["hf_id"] == GEMMA31_ID:
                return {**base, "output_text": "not json"}
            return {**base, "output_text": json.dumps({"answer": "42"})}

    branch_collection = mod.collect_model_proposals(
        models=[row for row in artifact["model_specs"] if row["selected_for_inference"]],
        events=artifact["frozen_event_manifest"]["events"],
        prompt_commitment=artifact["prompt_commitment"],
        raw_dir=tmp_path / "branch_raw",
        write=True,
        runtime_factory=BranchRuntime,
    )
    outcomes = {row["compile_outcome"] for row in branch_collection["exact_compile_rows"]}
    assert {"timeout", "no_proposal", "reject"} <= outcomes

    proposals = [
        {},
        {"factor_id": "bad", "kind": "branch_depth_at_least", "scope": ["event"], "weight": 1, "threshold": 999},
        {"factor_id": "bad_kind", "kind": "bad", "scope": ["event"], "weight": 1},
        {"factor_id": "bad_scope", "kind": "branch_depth_at_least", "scope": "event", "weight": 1, "threshold": 1},
        {"factor_id": "bad_weight", "kind": "branch_depth_at_least", "scope": ["event"], "weight": 0, "threshold": 1},
        {"factor_id": "bad_depth", "kind": "branch_depth_at_least", "scope": ["x"], "weight": 1, "threshold": 1},
        {"factor_id": "bad_count", "kind": "candidate_count_at_least", "scope": ["event"], "weight": 1},
        {"factor_id": "bad_residual", "kind": "residual_weight_at_most", "scope": ["x"], "weight": 1, "threshold": 0},
        {"factor_id": "bad_partial", "kind": "partial_assignment_eq", "scope": ["x"], "weight": 1, "variable": "x"},
        {"factor_id": "missing_partial", "kind": "partial_assignment_eq", "scope": ["x"], "weight": 1, "variable": "x", "value": 0},
        {"factor_id": "residual_ok", "kind": "residual_weight_at_most", "scope": ["event"], "weight": 1, "threshold": 999},
    ]
    outcomes = [
        mod.compile_atomic_factor(proposal, event, seen_semantic_hashes=set())["compile_outcome"]
        for proposal in proposals
    ]
    assert outcomes.count("reject") == 10
    assert outcomes[-1] == "accept"
    variable, value = next(iter(event["visible_context"]["partial_assignment"].items()))
    partial_accept = mod.compile_atomic_factor(
        {
            "factor_id": "partial_ok",
            "kind": "partial_assignment_eq",
            "scope": [variable],
            "weight": 1,
            "variable": variable,
            "value": value,
        },
        event,
        seen_semantic_hashes=set(),
    )
    assert partial_accept["compile_outcome"] == "accept"
    assert mod._int_field(True) is None
    assert mod._int_field("not-int") is None
    assert mod.parse_atomic_factor_response("[]")[1]["parse_status"] == "non_object"
    assert mod.parse_atomic_factor_response("")[1]["parse_status"] == "empty_response"
    assert mod._request_context_forbidden_count([{"kind": "request", "path": str(tmp_path / "gone.json")}]) == 0

    class FailingRuntime:
        def __init__(self, spec: dict[str, Any]) -> None:
            raise RuntimeError(f"load failed for {spec['hf_id']}")

    failed_collection = mod.collect_model_proposals(
        models=[artifact["model_specs"][0]],
        events=artifact["frozen_event_manifest"]["events"],
        prompt_commitment=artifact["prompt_commitment"],
        raw_dir=tmp_path / "failed_raw",
        write=True,
        runtime_factory=FailingRuntime,
    )
    assert failed_collection["model_load_receipts"][0]["load_status"] == "load_failed"
    assert failed_collection["proposal_rows"] == []


def test_scenario_verify_6491_validation_reports_every_guard(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6491-BOUNDARY-ATTACKS/ROWS: validation guards stay live."""

    clean = _artifact(tmp_path / "clean")
    missing = deepcopy(clean)
    del missing["status"]
    assert mod.validate_artifact(missing) == ["missing required fields: status"]

    original_git_output = mod._git_output
    try:
        mod._git_output = lambda *_: " M scripts/research_conductor.py"
        protected = mod._protected_files_unchanged(REPO)
    finally:
        mod._git_output = original_git_output
    assert protected["active_roadmap_and_conductor_unchanged"] is False

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    for field, message, value in (
        ("field_principles", "field_principles must cover exactly required fields", {}),
        ("field_provenance", "field_provenance must cover exactly required fields", {}),
        ("inference_substrate", "inference_substrate mismatch", "bad"),
        ("verifier_is_oracle", "verifier_is_oracle must be false for model proposals", True),
    ):
        mutated = deepcopy(clean)
        mutated[field] = value
        _with_checksum(mutated)
        assert message in mod.validate_artifact(mutated)

    bad_specs = deepcopy(clean)
    bad_specs["model_specs"] = bad_specs["model_specs"][:-1]
    _with_checksum(bad_specs)
    assert "model_specs must contain all mandated SOTA GGUF ids" in mod.validate_artifact(bad_specs)

    bad_raw = _artifact(tmp_path / "bad_raw")
    Path(bad_raw["raw_request_response_receipts"]["rows"][0]["path"]).write_text("changed", encoding="utf-8")
    assert "raw byte receipt hash mismatch" in mod.validate_artifact(bad_raw)

    for mutate, message in (
        (
            lambda a: a["exact_compile_rows"].append(deepcopy(a["exact_compile_rows"][0])),
            "exact_compile_rows must match proposal_rows",
        ),
        (
            lambda a: a["exact_compile_rows"][0].update({"compile_outcome": "bad"}),
            "compile outcomes must be enumerated",
        ),
        (
            lambda a: a["exact_compile_rows"][0].update({"model_output_is_oracle": True}),
            "compiler rows must keep model non-authority and exact compiler authority",
        ),
        (
            lambda a: a["prompt_commitment"].update({"grammar_backend": "gbnf"}),
            "prompt commitment must disable grammar, retry, and rank-select loops",
        ),
        (
            lambda a: a["held_isolation_receipts"].update({"held_rows_selected_count": 1}),
            "held isolation must keep held and final outcome fields out of requests",
        ),
        (
            lambda a: a["aggregate_row_recomputation"].update({"row_count": -1}),
            "aggregate_row_recomputation mismatch",
        ),
        (
            lambda a: a.update({"factor_proposal_stream_ready_score": 0.0}),
            "factor_proposal_stream_ready_score mismatch",
        ),
        (
            lambda a: a["protected_files_unchanged"].update({"active_roadmap_and_conductor_unchanged": False}),
            "protected files changed",
        ),
    ):
        mutated = deepcopy(clean)
        mutate(mutated)
        _with_checksum(mutated)
        assert message in mod.validate_artifact(mutated)

    one_family = deepcopy(clean)
    one_family["per_unit_rows"] = [
        row
        for row in one_family["per_unit_rows"]
        if row.get("model_hf_id") != GEMMA31_ID
    ]
    one_family["aggregate_row_recomputation"] = mod.recompute_aggregates_from_rows(
        one_family["per_unit_rows"]
    )
    one_family["factor_proposal_stream_ready_score"] = 1.0
    _with_checksum(one_family)
    assert "ready score requires two completed model families" in mod.validate_artifact(one_family)


def test_scenario_verify_6491_adversarial_verify_uses_small_n_gguf_floor() -> None:
    """SCENARIO-VERIFY-6491-GATES: Exp6491's substrate is a short local-GGUF stream."""

    artifact = {
        "honest_verdict": "complete_local_proposal_stream: fixture",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "model_specs": [
            {"hf_id": QWEN_ID, "model_file": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"},
            {"hf_id": GEMMA26_ID, "model_file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        ],
        "duration_s": 54.463968,
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "sha256:fixture",
    }

    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    assert floor is not None
    assert floor["reason"] == "local_sota_gguf_small_n"
    assert float(floor["min_duration_s"]) == 10.0

    flags: list[Any] = []
    adversarial_verify.check_duration_vs_claim(artifact, flags)
    assert [flag.kind for flag in flags if flag.kind == "DURATION_TOO_SHORT"] == []
