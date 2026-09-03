"""Tests for exact-guided plain relation generation.

Spec refs: REQ-INFERENCE-6920 and SCENARIO-INFERENCE-6920-*.
"""

from __future__ import annotations

import base64
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6919_exact_prefix_viability_fixture as fixture_mod
from carnot import experiment_6920_sota_exact_guided_relation_generation as mod


ROOT = Path(__file__).resolve().parents[2]


def _models() -> list[dict[str, object]]:
    """Return exact cached-model identities without touching the live cache."""

    return [
        {
            "hf_id": hf_id,
            "model_path": f"/cache/{family}.gguf",
            "sha256": mod.EXPECTED_MODEL_HASHES[hf_id],
            "model_size_bytes": 1,
            "gpu": 0,
        }
        for hf_id, family in mod.MODEL_FAMILIES.items()
    ]


def _tokenizers() -> list[dict[str, object]]:
    """Return native tokenizer receipts that satisfy the frozen identities."""

    return [
        {
            "hf_id": hf_id,
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": True,
            "used_hf_autotokenizer": False,
            "canonical_tokenizer_payload_sha256": mod.EXPECTED_TOKENIZER_HASHES[hf_id],
            "model_sha256": mod.EXPECTED_MODEL_HASHES[hf_id],
        }
        for hf_id in mod.MODEL_SPECS
    ]


def _preconditions(**overrides: object) -> dict[str, object]:
    """Build one passing synthetic preflight for focused gate tests."""

    values: dict[str, object] = {
        "upstream": {
            "prefix_viability_canary_ready_score": 1,
            "source_artifact_hashes": {
                "module": {"sha256": mod.EXPECTED_ENGINE_HASHES["prefix_engine_module"]},
                "exp6274_compiler": {"sha256": mod.EXPECTED_ENGINE_HASHES["asp_energy_compiler"]},
            },
        },
        "upstream_sha256": mod.EXPECTED_EXP6919_SHA256,
        "models": _models(),
        "tokenizer_receipts": _tokenizers(),
        "gpu_inventory": [
            {
                "index": 0,
                "gpu_uuid": "GPU-test",
                "free_vram_mb": 24576,
                "total_vram_mb": 24576,
            }
        ],
        "lease_probe_rows": [{"gpu_uuid": "GPU-test", "owned": True, "released": True}],
        "outside_arc_job_rows": [],
        "cuda_offload_supported": True,
    }
    values.update(overrides)
    return mod.evaluate_preconditions(**values)


def _runtime_row() -> dict[str, object]:
    """Return one authenticated terminal candidate runtime receipt."""

    return {
        "timed_out": False,
        "truncated": False,
        "server_crashed": False,
        "stop_reason": "stop",
        "parser_attempted": True,
        "runtime_receipt": {
            "authentic": True,
            "process_identity_match": True,
            "server_pid": 4000,
            "server_start_time_ticks": 9000,
            "offload_layers": 65,
            "owned_cuda_residency": True,
        },
    }


def _final_row(
    *,
    model: str,
    arm: str,
    family: str,
    valid: bool,
    parse_success: bool = True,
) -> dict[str, object]:
    """Return one detailed final row for aggregate replay tests."""

    return {
        "model_spec": model,
        "arm": arm,
        "family": family,
        "exact_final_valid": valid,
        "parse_success": parse_success,
        "abstained": False,
        "sampled_tokens": 8,
        "wall_time_s": 0.25,
        "vram_mb": 1024,
        "energy_proxy": 0 if valid else 1,
    }


def _held_sources() -> list[dict[str, object]]:
    """Return the frozen source matrix with its deterministic seed schedule."""

    upstream = json.loads(
        (ROOT / "results/experiment_6919_exact_prefix_viability_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    rows = mod.select_held_source_tasks(upstream)
    for index, row in enumerate(rows):
        row["generation_seed"] = mod.SEEDS[index % len(mod.SEEDS)]
        row["source_order"] = index
    return rows


def _candidate_row(
    *,
    model: str,
    source: dict[str, object],
    arm: str,
    candidate_index: int,
) -> dict[str, object]:
    """Build one preserved, authenticated candidate for artifact replay."""

    cell_id = f"{model}::{source['generation_seed']}::{source['source_task_id']}"
    expected_lines = 2 if arm != "guided_frontier" else 1
    raw = b"alpha relates red\nbeta relates blue" if expected_lines == 2 else b"alpha relates red"
    request_payload = {
        "messages": [{"role": "user", "content": "Write plain relation lines."}],
        "max_tokens": (
            mod.DIRECT_TOKEN_LIMIT
            if arm == "direct_generation"
            else mod.MATCHED_CANDIDATE_TOKEN_LIMIT
        ),
    }
    request_bytes = mod.canonical_json(request_payload)
    row: dict[str, object] = {
        "candidate_id": f"{cell_id}::{arm}::{candidate_index}",
        "cell_id": cell_id,
        "source_task_id": source["source_task_id"],
        "source_id": source["source_id"],
        "family": source["family"],
        "model_spec": model,
        "generation_seed": source["generation_seed"],
        "sample_seed": int(source["generation_seed"]) + candidate_index,
        "arm": arm,
        "candidate_index": candidate_index,
        "frontier_step": 0,
        "request_payload": request_payload,
        "raw_request_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_request_sha256": mod.sha256_bytes(request_bytes),
        "raw_request_byte_count": len(request_bytes),
        "raw_output_b64": base64.b64encode(raw).decode("ascii"),
        "raw_output_sha256": mod.sha256_bytes(raw),
        "raw_output_byte_count": len(raw),
        "raw_output_text": raw.decode("ascii"),
        "generated_tokens": 4,
        "sampled_token_limit": (
            mod.DIRECT_TOKEN_LIMIT
            if arm == "direct_generation"
            else mod.MATCHED_CANDIDATE_TOKEN_LIMIT
        ),
        "token_logprobs": [-0.1],
        "likelihood": -0.1 - candidate_index,
        "mean_token_logprob": -0.1 - candidate_index,
        "stop_reason": "stop",
        "timed_out": False,
        "truncated": False,
        "server_crashed": False,
        "request_error": None,
        "parser_attempted": True,
        "wall_time_s": 0.1,
        "runtime_receipt": deepcopy(_runtime_row()["runtime_receipt"]),
        "expected_line_count": expected_lines,
        "parse_success": True,
        "parsed_lines": raw.decode("ascii").splitlines(),
        "parse_error": None,
        "repair_applied": False,
        "preselection_exact_engine_calls": int(arm == "guided_frontier"),
        "prefix_energy": 0 if arm == "guided_frontier" else None,
        "in_loop_engine": mod.IN_LOOP_ENGINE if arm == "guided_frontier" else None,
        "selection_method": {
            "direct_generation": "single_full_program_draw",
            "unguided_best_of_k": "model_likelihood_only",
            "guided_frontier": "feasible_then_model_likelihood",
        }[arm],
        "selected": candidate_index == 0,
    }
    if arm == "guided_frontier":
        row.update(
            {
                "prior_partial_program": [],
                "parsed_line": raw.decode("ascii"),
                "in_loop_decision_reason": "bounded_completion",
                "rejection_reason": (None if candidate_index == 0 else "lower_model_likelihood"),
                "candidate_partial_program": [raw.decode("ascii")],
            }
        )
    return row


def _successful_acquisition(
    sources: list[dict[str, object]],
) -> dict[str, object]:
    """Build complete synthetic evidence for the artifact composition boundary."""

    candidates: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for model_index, model in enumerate(mod.MODEL_SPECS):
        for source in sources:
            cell_id = f"{model}::{source['generation_seed']}::{source['source_task_id']}"
            for arm in mod.ARMS:
                count = 1 if arm == "direct_generation" else mod.MATCHED_CANDIDATE_BUDGET
                candidates.extend(
                    _candidate_row(
                        model=model,
                        source=source,
                        arm=arm,
                        candidate_index=index,
                    )
                    for index in range(count)
                )
                valid = arm == "direct_generation" or model_index == 2 or arm == "guided_frontier"
                outcomes.append(
                    {
                        "outcome_id": f"{cell_id}::{arm}",
                        "cell_id": cell_id,
                        "source_task_id": source["source_task_id"],
                        "source_id": source["source_id"],
                        "family": source["family"],
                        "model_spec": model,
                        "generation_seed": source["generation_seed"],
                        "arm": arm,
                        "selected_candidate_id": f"{cell_id}::{arm}::0",
                        "selected_program": ["alpha relates red", "beta relates blue"],
                        "parse_success": True,
                        "exact_final_valid": valid,
                        "answer_set_count": int(valid),
                        "answer_set_effect": "admitted_one" if valid else "eliminated_all",
                        "required_prefix_preserved": True,
                        "abstained": False,
                        "exhausted_search_abstention": False,
                        "false_admission": False,
                        "branch_rejection_count": (count - 1 if arm == "guided_frontier" else 0),
                        "sampled_tokens": 4 * count,
                        "wall_time_s": 0.1 * count,
                        "vram_mb": 2048,
                        "energy_proxy": count if arm == "guided_frontier" else 0,
                        "in_loop_engine": mod.IN_LOOP_ENGINE,
                        "final_engine": mod.FINAL_ENGINE,
                        "solver_version": "clingo test",
                        "final_reason": "valid_completion" if valid else "family_constraint",
                    }
                )
    llama_rows = [
        {
            "hf_id": model,
            "model_sha256": mod.EXPECTED_MODEL_HASHES[model],
            "tokenizer_sha256": mod.EXPECTED_TOKENIZER_HASHES[model],
            "offload_layers": 65,
            "owned_cuda_residency": True,
        }
        for model in mod.MODEL_SPECS
    ]
    lifecycle_rows = [
        {
            "hf_id": model,
            "process_exit_confirmed": True,
            "process_reaped": True,
            "port_release_confirmed": True,
            "lease_released": True,
            "leak_free": True,
            "unrelated_process_signal_count": 0,
        }
        for model in mod.MODEL_SPECS
    ]
    matched = [row for row in candidates if row["arm"] != "direct_generation"]
    return {
        "candidate_rows": candidates,
        "final_exact_outcome_rows": outcomes,
        "frontier_size_rows": [],
        "request_count": len(candidates),
        "matched_request_count": len(matched),
        "llama_cpp_receipts": llama_rows,
        "gpu_lease_rows": [{"owned": True, "released": True}] * len(mod.MODEL_SPECS),
        "server_lifecycle_rows": lifecycle_rows,
    }


def test_req_inference_6920_spec_owns_fields_and_scenarios() -> None:
    """REQ-INFERENCE-6920 declares every artifact field and failure scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-INFERENCE-6920") :]
    scenarios = (
        "PRECONDITIONS",
        "BUDGET",
        "DIRECT-ISOLATION",
        "PREFIX-ADMISSION",
        "FEASIBLE-BRANCH",
        "TIE",
        "PARSER",
        "RUNTIME",
        "ENGINE-SEPARATION",
        "AGGREGATES",
        "UTILITY",
    )

    assert all(f"SCENARIO-INFERENCE-6920-{name}" in section for name in scenarios)
    assert all(f"`{field}`" in section for field in mod.REQUIRED_ARTIFACT_FIELDS)


def test_req_inference_6920_resolves_pair_then_dense_extension() -> None:
    """REQ-INFERENCE-6920 resolves every GGUF through repository cache helpers."""

    calls: list[tuple[str, object]] = []

    def pair_provider(**kwargs: object) -> list[dict[str, object]]:
        calls.append(("pair", kwargs))
        return [
            {"hf_id": mod.MODEL_SPECS[0], "model_path": "/cache/qwen.gguf", "gpu": 0},
            {"hf_id": mod.MODEL_SPECS[2], "model_path": "/cache/moe.gguf", "gpu": 0},
        ]

    def dense_resolver(hf_id: str, quant: str) -> str:
        calls.append(("dense", (hf_id, quant)))
        return "/cache/dense.gguf"

    rows = mod.resolve_three_models(
        pair_provider=pair_provider,
        dense_resolver=dense_resolver,
    )

    assert [row["hf_id"] for row in rows] == list(mod.MODEL_SPECS)
    assert rows[1]["model_path"] == "/cache/dense.gguf"
    assert calls[0][0] == "pair"
    assert calls[1] == ("dense", (mod.MODEL_SPECS[1], "Q4_K_M"))


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("wrong_model", "exact_model_files"),
        ("tokenizer_substitution", "native_tokenizer_receipts"),
        ("zero_offload", "cuda_offload_supported"),
    ],
)
def test_scenario_inference_6920_preconditions_fail_closed(
    mutation: str,
    failed_check: str,
) -> None:
    """SCENARIO-INFERENCE-6920-PRECONDITIONS blocks identity and offload drift."""

    models = _models()
    tokenizers = _tokenizers()
    offload = True
    if mutation == "wrong_model":
        models[0]["sha256"] = "sha256:wrong"
    elif mutation == "tokenizer_substitution":
        tokenizers[0]["canonical_tokenizer_payload_sha256"] = "sha256:substitute"
    else:
        offload = False

    report = _preconditions(
        models=models,
        tokenizer_receipts=tokenizers,
        cuda_offload_supported=offload,
    )

    assert report["all_passed"] is False
    assert failed_check in report["gate_check_summary"]["failed_checks"]
    failed = next(row for row in report["checks"] if row["check"] == failed_check)
    assert {"expected", "observed", "passed"} <= set(failed)


def test_req_inference_6920_freezes_thirty_balanced_held_tasks() -> None:
    """REQ-INFERENCE-6920 selects 30 held tasks across all five families."""

    upstream = json.loads(
        (ROOT / "results/experiment_6919_exact_prefix_viability_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    rows = mod.select_held_source_tasks(upstream)
    counts = Counter(str(row["family"]) for row in rows)

    assert len(rows) == 30
    assert len({row["source_task_id"] for row in rows}) == 30
    assert set(counts) == set(fixture_mod.FAMILIES)
    assert set(counts.values()) == {6}
    assert all(row["split"] == "held" for row in rows)


def test_scenario_inference_6920_budget_detects_unequal_and_hidden_samples() -> None:
    """SCENARIO-INFERENCE-6920-BUDGET catches drift and unrecorded extra draws."""

    budgets = mod.build_arm_budget_rows(["cell"])
    candidates = [
        {"cell_id": "cell", "arm": "unguided_best_of_k", "candidate_index": index}
        for index in range(mod.MATCHED_CANDIDATE_BUDGET)
    ] + [
        {"cell_id": "cell", "arm": "guided_frontier", "candidate_index": index}
        for index in range(mod.MATCHED_CANDIDATE_BUDGET)
    ]
    assert mod.candidate_budget_errors(candidates, budgets, request_count=len(candidates)) == []

    unequal = deepcopy(budgets)
    unequal[-1]["sampled_token_limit"] += 1
    assert "unequal_sampled_token_budget:cell" in mod.candidate_budget_errors(
        candidates, unequal, request_count=len(candidates)
    )
    assert "hidden_extra_samples" in mod.candidate_budget_errors(
        candidates, budgets, request_count=len(candidates) + 1
    )


def test_scenario_inference_6920_direct_arm_has_no_verifier_leakage() -> None:
    """SCENARIO-INFERENCE-6920-DIRECT-ISOLATION rejects preselection oracle use."""

    clean = {
        "arm": "direct_generation",
        "preselection_exact_engine_calls": 0,
        "selection_method": "single_full_program_draw",
        "prefix_energy": None,
        "request_payload": {"messages": [{"content": "Write two plain relation lines."}]},
    }
    leaked = {**clean, "preselection_exact_engine_calls": 1, "prefix_energy": 0}

    assert mod.direct_arm_leakage_errors([clean]) == []
    assert mod.direct_arm_leakage_errors([leaked]) == ["direct_arm_verifier_leakage:0"]


def test_scenarios_inference_6920_prefix_admission_and_feasible_branch() -> None:
    """SCENARIO-INFERENCE-6920-PREFIX-ADMISSION and -FEASIBLE-BRANCH fail closed."""

    fixture = fixture_mod.build_relation_fixture("graph_coloring", 15)
    valid = fixture.action_line(fixture.subjects[0], fixture.values[0])
    blocked = fixture.action_line(fixture.subjects[0], fixture.forbidden_value)
    rows, selected = mod.evaluate_guided_candidates(
        fixture=fixture,
        prior_prefix=(),
        candidates=[
            {"candidate_index": 0, "raw_text": blocked, "likelihood": -0.1},
            {"candidate_index": 1, "raw_text": valid, "likelihood": -0.2},
            {"candidate_index": 2, "raw_text": "not relation", "likelihood": -0.01},
        ],
    )

    assert rows[0]["prefix_energy"] > 0
    assert rows[0]["selected"] is False
    assert rows[1]["prefix_energy"] == 0
    assert rows[1]["selected"] is True
    assert rows[2]["parse_success"] is False
    assert selected == rows[1]

    invalid_admission = deepcopy(rows)
    invalid_admission[0]["selected"] = True
    invalid_admission[1]["selected"] = False
    assert "invalid_prefix_admission" in mod.guided_frontier_errors(invalid_admission)
    feasible_rejection = deepcopy(rows)
    feasible_rejection[1]["rejection_reason"] = "exact_prefix_impossible"
    assert "feasible_branch_rejection" in mod.guided_frontier_errors(feasible_rejection)


def test_scenario_inference_6920_tie_rule_does_not_drift() -> None:
    """SCENARIO-INFERENCE-6920-TIE uses likelihood, index, then line text."""

    rows = [
        {
            "candidate_index": 1,
            "parsed_line": "z has_color red",
            "likelihood": -0.5,
            "prefix_energy": 0,
        },
        {
            "candidate_index": 0,
            "parsed_line": "a has_color blue",
            "likelihood": -0.5,
            "prefix_energy": 0,
        },
    ]

    assert mod.select_feasible_candidate(rows)["candidate_index"] == 0
    assert mod.select_feasible_candidate(list(reversed(rows)))["candidate_index"] == 0
    drift = deepcopy(rows)
    drift[0]["selected"] = True
    drift[1]["selected"] = False
    assert "tie_drift" in mod.guided_frontier_errors(drift)


def test_scenario_inference_6920_parser_never_masks_raw_failure() -> None:
    """SCENARIO-INFERENCE-6920-PARSER preserves malformed bytes without repair."""

    parsed = mod.parse_plain_candidate(b"not a relation program", expected_line_count=1)

    assert parsed["raw_output_bytes"] == b"not a relation program"
    assert parsed["parse_success"] is False
    assert parsed["parsed_lines"] == []
    masked = {**parsed, "parse_success": True, "parsed_lines": ["fixture line value"]}
    assert mod.parser_masking_errors([masked]) == ["parser_masking:0"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("timed_out", True, "timeout"),
        ("truncated", True, "truncation"),
        ("server_crashed", True, "server_crash"),
        ("process_identity_match", False, "stale_pid"),
        ("offload_layers", 0, "zero_offload"),
    ],
)
def test_scenario_inference_6920_runtime_failures_are_not_authenticated(
    field: str,
    value: object,
    expected: str,
) -> None:
    """SCENARIO-INFERENCE-6920-RUNTIME retains transport and process failures."""

    row = _runtime_row()
    if field in row:
        row[field] = value
    else:
        row["runtime_receipt"][field] = value

    errors = mod.runtime_authentication_errors(row)

    assert expected in errors
    assert mod.runtime_row_authenticated(row) is False


def test_scenario_inference_6920_final_engine_cannot_reuse_prefix_engine() -> None:
    """SCENARIO-INFERENCE-6920-ENGINE-SEPARATION rejects final-engine reuse."""

    assert (
        mod.engine_separation_errors(
            [
                {
                    "in_loop_engine": mod.IN_LOOP_ENGINE,
                    "final_engine": mod.FINAL_ENGINE,
                }
            ]
        )
        == []
    )
    assert mod.engine_separation_errors(
        [
            {
                "in_loop_engine": mod.IN_LOOP_ENGINE,
                "final_engine": mod.IN_LOOP_ENGINE,
            }
        ]
    ) == ["final_engine_reuse:0"]


def test_scenario_inference_6920_aggregate_disagreement_blocks() -> None:
    """SCENARIO-INFERENCE-6920-AGGREGATES replays summaries from detailed rows."""

    rows = [
        _final_row(model=model, arm=arm, family="graph_coloring", valid=arm != "direct_generation")
        for model in mod.MODEL_SPECS
        for arm in mod.ARMS
    ]
    aggregates = mod.aggregate_outcome_rows(rows)

    assert mod.aggregate_disagreement_errors(rows, aggregates) == []
    bad = deepcopy(aggregates)
    bad["per_model_arm_rows"][0]["exact_final_validity_rate"] = 0.123
    assert "per_model_arm_rows" in mod.aggregate_disagreement_errors(rows, bad)


def test_scenario_inference_6920_utility_gate_and_circular_verdict() -> None:
    """SCENARIO-INFERENCE-6920-UTILITY applies all gain and cost conditions."""

    per_model = []
    for index, model in enumerate(mod.MODEL_SPECS):
        unguided = 0.5
        guided = 0.6 if index < 2 else 0.49
        per_model.extend(
            [
                {
                    "model_spec": model,
                    "arm": "unguided_best_of_k",
                    "exact_final_validity_rate": unguided,
                    "parse_failure_rate": 0.1,
                },
                {
                    "model_spec": model,
                    "arm": "guided_frontier",
                    "exact_final_validity_rate": guided,
                    "parse_failure_rate": 0.1,
                },
            ]
        )
    score, deltas = mod.compute_utility_score(per_model, pareto_complete=True)

    assert score == 1
    assert sum(row["guided_beats_unguided"] for row in deltas) == 2
    assert min(row["validity_delta"] for row in deltas) >= -0.02
    assert mod.verdict_class(run_complete=1, utility_score=score) == "circular_positive"
    assert mod.verdict_class(run_complete=1, utility_score=0) == "null"


def test_req_inference_6920_blocked_artifact_is_complete_and_written(tmp_path: Path) -> None:
    """REQ-INFERENCE-6920 writes a complete blocker before any failed live work."""

    output = tmp_path / "result.json"
    artifact = mod.run(
        date="20260903",
        root=tmp_path,
        result_path=output,
        precondition_collector=lambda _root: _preconditions(models=[]),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["guided_generation_run_complete_score"] == 0
    assert artifact["exact_guidance_utility_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == ("complete_blocked_sota_exact_guided_relation_generation")


def test_req_inference_6920_artifact_validation_rejects_positive_and_gate_drift(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6920 rejects positive oracle claims and inconsistent scores."""

    artifact = mod.build_blocked_artifact(
        date="20260903",
        duration_s=0.1,
        preconditions=_preconditions(models=[]),
        models=[],
        tokenizer_receipts=[],
        root=tmp_path,
    )
    positive = deepcopy(artifact)
    positive["verdict_class"] = "positive"
    gate_drift = deepcopy(artifact)
    gate_drift["guided_generation_run_complete_score"] = 1

    with pytest.raises(ValueError, match="oracle_verdict_cannot_be_positive"):
        mod.validate_artifact(positive)
    with pytest.raises(ValueError, match="run_complete_gate_disagreement"):
        mod.validate_artifact(gate_drift)


def test_req_inference_6920_main_prints_terminal_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-INFERENCE-6920 exposes the dated command-line entry point."""

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date: {
            "honest_verdict": "complete_test",
            "guided_generation_run_complete_score": 1,
            "exact_guidance_utility_score": 0,
        },
    )

    assert mod.main(["--date", "20260903"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "exact_guidance_utility_score": 0,
        "guided_generation_run_complete_score": 1,
        "honest_verdict": "complete_test",
        "result_path": str(mod.RESULT_RELATIVE_PATH),
    }


def test_req_inference_6920_prompt_and_request_are_plain_unconstrained_text() -> None:
    """REQ-INFERENCE-6920 prompts and requests expose no structured decode surface."""

    source = _held_sources()[0]
    program_prompt = mod.build_program_prompt(source)
    frontier_prompt = mod.build_frontier_prompt(source, [])
    request_payload = mod._candidate_request(program_prompt, token_limit=32, seed=7)
    source_without_pairs = {**source, "invalid_pairs": [], "initial_prefix": []}

    assert "STARTING LINES" in program_prompt
    assert "CURRENT PROGRAM\n(empty)" in frontier_prompt
    assert "Disallowed object pairs are: none" in mod.build_program_prompt(source_without_pairs)
    assert request_payload["messages"][0]["content"] == program_prompt
    assert request_payload["max_tokens"] == 32
    assert request_payload["seed"] == 7
    assert request_payload["logprobs"] is True
    assert not ({"grammar", "json_schema", "response_format"} & set(request_payload))


def test_req_inference_6920_source_and_cache_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFERENCE-6920 rejects a missing matrix and preserves cache misses."""

    with pytest.raises(mod.ExactGuidedGenerationError, match="prefix_case_rows_missing"):
        mod.select_held_source_tasks({})
    with pytest.raises(mod.ExactGuidedGenerationError, match="held_family_floor"):
        mod.select_held_source_tasks({"prefix_case_rows": []})

    upstream = json.loads(
        (ROOT / "results/experiment_6919_exact_prefix_viability_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(mod, "SOURCE_TASK_COUNT", 31)
    with pytest.raises(mod.ExactGuidedGenerationError, match="held_source_count:30"):
        mod.select_held_source_tasks(upstream)

    rows = mod.resolve_three_models(
        pair_provider=lambda **_kwargs: None,
        dense_resolver=lambda _hf_id, _quant: None,
    )
    assert [row.get("model_path") for row in rows] == [None, None, None]


def test_scenario_inference_6920_exhaustion_and_lower_likelihood_are_recorded() -> None:
    """SCENARIO-INFERENCE-6920-PREFIX-ADMISSION records terminal branch reasons."""

    fixture = fixture_mod.build_relation_fixture("graph_coloring", 15)
    first = fixture.action_line(fixture.subjects[0], fixture.values[0])
    second = fixture.action_line(fixture.subjects[0], fixture.values[1])
    rows, selected = mod.evaluate_guided_candidates(
        fixture=fixture,
        prior_prefix=(),
        candidates=[
            {"candidate_index": 0, "raw_text": first, "likelihood": -0.1},
            {"candidate_index": 1, "raw_text": second, "likelihood": -0.2},
        ],
    )
    exhausted, no_selection = mod.evaluate_guided_candidates(
        fixture=fixture,
        prior_prefix=(),
        candidates=[{"candidate_index": 0, "raw_text": b"bad", "likelihood": -0.1}],
    )

    assert selected == rows[0]
    assert rows[1]["rejection_reason"] == "lower_model_likelihood"
    assert no_selection is None
    assert exhausted[0]["rejection_reason"] == "parse_failure"
    assert mod.select_feasible_candidate([]) is None


def test_scenario_inference_6920_budget_audits_every_failure_mode() -> None:
    """SCENARIO-INFERENCE-6920-BUDGET names missing, short, and oversized evidence."""

    budgets = mod.build_arm_budget_rows(["cell"])
    candidates = [
        {
            "cell_id": "cell",
            "arm": arm,
            "candidate_index": index,
            "sampled_token_limit": mod.MATCHED_CANDIDATE_TOKEN_LIMIT,
        }
        for arm in ("unguided_best_of_k", "guided_frontier")
        for index in range(mod.MATCHED_CANDIDATE_BUDGET)
    ]
    assert "missing_request_receipts" in mod.candidate_budget_errors(
        candidates, budgets, request_count=len(candidates) - 1
    )

    short = candidates[:-1]
    errors = mod.candidate_budget_errors(short, budgets, request_count=len(short))
    assert "candidate_count:cell:guided_frontier" in errors

    oversized = deepcopy(candidates)
    oversized[-1]["sampled_token_limit"] += 1
    errors = mod.candidate_budget_errors(oversized, budgets, request_count=len(oversized))
    assert "sampled_token_limit:cell:guided_frontier" in errors

    unequal = deepcopy(budgets)
    unequal[-1]["candidate_budget"] -= 1
    errors = mod.candidate_budget_errors(candidates, unequal, request_count=len(candidates))
    assert "unequal_candidate_budget:cell" in errors


def test_scenario_inference_6920_runtime_and_parser_success_paths() -> None:
    """SCENARIO-INFERENCE-6920-RUNTIME authenticates only fully parsed live rows."""

    row = _runtime_row()
    assert mod.runtime_authentication_errors(row) == []
    assert mod.runtime_row_authenticated(row) is True

    unauthenticated = deepcopy(row)
    unauthenticated["runtime_receipt"]["authentic"] = False
    unauthenticated["parser_attempted"] = False
    assert {"unauthenticated_runtime", "parser_bypass"} <= set(
        mod.runtime_authentication_errors(unauthenticated)
    )

    raw = b"left relates red"
    clean_parse = {
        "raw_output_bytes": raw,
        "expected_line_count": 1,
        "parse_success": True,
        "parsed_lines": ["left relates red"],
        "repair_applied": False,
    }
    assert mod.parser_masking_errors([clean_parse]) == []
    assert mod.direct_arm_leakage_errors([{"arm": "guided_frontier"}]) == []


def test_req_inference_6920_builds_complete_circular_artifact() -> None:
    """REQ-INFERENCE-6920 composes complete authenticated rows into a circular result."""

    sources = _held_sources()
    acquisition = _successful_acquisition(sources)
    artifact = mod.build_artifact(
        date="20260903",
        duration_s=61.0,
        root=ROOT,
        preconditions=_preconditions(),
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        source_rows=sources,
        acquisition=acquisition,
    )

    assert artifact["guided_generation_run_complete_score"] == 1
    assert artifact["exact_guidance_utility_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"] == "complete_circular_positive_exact_guidance_utility"
    assert len(artifact["candidate_rows"]) == 810
    assert len(artifact["final_exact_outcome_rows"]) == 270
    assert all("model_spec" not in row for row in artifact["per_family_arm_rows"])
    assert all("branch_rejection_count" in row for row in artifact["final_exact_outcome_rows"])
    assert not artifact["abstention_rows"]
    mod.validate_artifact(artifact)

    partial = mod.build_artifact(
        date="20260903",
        duration_s=0.0,
        root=ROOT,
        preconditions=_preconditions(),
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        source_rows=sources,
        acquisition=acquisition,
    )
    assert partial["honest_verdict"] == ("complete_partial_sota_exact_guided_relation_generation")

    null_acquisition = deepcopy(acquisition)
    for row in null_acquisition["final_exact_outcome_rows"]:
        row["exact_final_valid"] = True
    null_artifact = mod.build_artifact(
        date="20260903",
        duration_s=61.0,
        root=ROOT,
        preconditions=_preconditions(),
        models=_models(),
        tokenizer_receipts=_tokenizers(),
        source_rows=sources,
        acquisition=null_acquisition,
    )
    assert null_artifact["honest_verdict"] == ("complete_null_exact_guidance_utility_not_shown")


def test_req_inference_6920_validation_rejects_each_contract_drift(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6920 validates schema, mechanisms, oracle, and checksum fields."""

    artifact = mod.build_blocked_artifact(
        date="20260903",
        duration_s=0.1,
        preconditions=_preconditions(models=[]),
        models=[],
        tokenizer_receipts=[],
        root=tmp_path,
    )

    variants: list[tuple[dict[str, object], str]] = []
    missing = deepcopy(artifact)
    missing.pop("rows")
    variants.append((missing, "missing_required_fields"))
    principle = deepcopy(artifact)
    principle["field_principles"].pop("rows")
    variants.append((principle, "missing_field_principles"))
    invalid_class = deepcopy(artifact)
    invalid_class["verdict_class"] = "maybe"
    variants.append((invalid_class, "invalid_verdict_class"))
    invalid_score = deepcopy(artifact)
    invalid_score["exact_guidance_utility_score"] = 2
    variants.append((invalid_score, "invalid_gate_score"))
    utility_without_run = deepcopy(artifact)
    utility_without_run["exact_guidance_utility_score"] = 1
    variants.append((utility_without_run, "utility_without_complete_run"))
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "cpu"
    variants.append((substrate, "inference_substrate"))
    retired = deepcopy(artifact)
    retired["repair_prompt_count"] = 1
    variants.append((retired, "retired_mechanism_activation"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = False
    variants.append((oracle, "oracle_declaration"))
    terminal = deepcopy(artifact)
    terminal["honest_verdict"] = "blocked"
    variants.append((terminal, "honest_verdict_not_terminal"))
    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:wrong"
    variants.append((checksum, "reproducibility_checksum"))

    for variant, message in variants:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(variant)


def test_req_inference_6920_successful_run_writes_composed_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6920 runs the admitted acquisition path exactly once."""

    upstream = json.loads(
        (ROOT / "results/experiment_6919_exact_prefix_viability_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    preconditions = _preconditions()
    preconditions.update(
        {
            "upstream": upstream,
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
            "selected_gpu": {"index": 0, "gpu_uuid": "GPU-test"},
        }
    )
    calls: list[dict[str, object]] = []
    composed = {
        "honest_verdict": "complete_test",
        "guided_generation_run_complete_score": 1,
        "exact_guidance_utility_score": 0,
    }

    def runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"candidate_rows": []}

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: composed)
    output = tmp_path / "result.json"
    result = mod.run(
        date="20260903",
        root=tmp_path,
        result_path=output,
        precondition_collector=lambda _root: preconditions,
        acquisition_runner=runner,
    )

    assert result == composed
    assert json.loads(output.read_text(encoding="utf-8")) == composed
    assert len(calls) == 1
    sources = calls[0]["sources"]
    assert len(sources) == 30
    assert {row["generation_seed"] for row in sources} == set(mod.SEEDS)


def test_req_inference_6920_successful_preflight_requires_upstream(
    tmp_path: Path,
) -> None:
    """REQ-INFERENCE-6920 never starts acquisition without the held source artifact."""

    preconditions = _preconditions()
    preconditions.update(
        {
            "models": _models(),
            "tokenizer_receipts": _tokenizers(),
            "selected_gpu": {"index": 0, "gpu_uuid": "GPU-test"},
        }
    )
    with pytest.raises(mod.ExactGuidedGenerationError, match="preflight_upstream_missing"):
        mod.run(
            date="20260903",
            root=tmp_path,
            result_path=tmp_path / "result.json",
            precondition_collector=lambda _root: preconditions,
        )


def test_scenario_inference_6920_partial_and_pareto_fail_closed() -> None:
    """SCENARIO-INFERENCE-6920-UTILITY needs complete telemetry and live rows."""

    score, _rows = mod.compute_utility_score([], pareto_complete=False)
    assert score == 0
    assert mod.verdict_class(run_complete=0, utility_score=0) == "partial"
