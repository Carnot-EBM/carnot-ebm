"""Tests for Exp6289 flagship exact-state refinement benchmark.

Spec refs: REQ-CONSTRAINT-6289,
SCENARIO-CONSTRAINT-6289-SEALED-SOTA,
SCENARIO-CONSTRAINT-6289-MATCHED-BUDGETS,
SCENARIO-CONSTRAINT-6289-ORACLE-VALUE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6289_flagship_exact_state_refinement_benchmark as exp6289


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


class FakeBackend:
    """Deterministic backend that emits ordinary text answers."""

    def __init__(
        self,
        *,
        zero_token_qwen: bool = False,
        nonterminal_model: str | None = None,
    ) -> None:
        self.zero_token_qwen = zero_token_qwen
        self.nonterminal_model = nonterminal_model

    def generate_model(self, model_spec: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for job in jobs:
            task = job["task"]
            answer_sets = task["exact_answer_sets"]
            if answer_sets and job["sample_index"] == 0:
                text = exp6289.format_visible_answer(task, answer_sets[0])
            elif answer_sets:
                text = exp6289.format_visible_answer(task, answer_sets[-1])
            else:
                text = "IMPOSSIBLE"
            tokens = len(text.split())
            if self.zero_token_qwen and model_spec["hf_id"] == exp6289.MANDATED_MODEL_IDS[0]:
                text = ""
                tokens = 0
            rows.append(
                {
                    "task_id": task["task_id"],
                    "sample_index": job["sample_index"],
                    "seed": job["seed"],
                    "raw_output": text,
                    "generated_token_count": tokens,
                    "prompt_token_count": len(job["prompt_text"].split()),
                    "latency_s": 0.01,
                    "finish_reason": "stop",
                    "timeout": False,
                }
            )
        disposition = (
            "running"
            if model_spec["hf_id"] == self.nonterminal_model
            else "complete: fake backend finished"
        )
        return {
            "rows": rows,
            "receipt": {
                "terminal_disposition": disposition,
                "gpu_offload": {"requested": True, "observed": True},
                "peak_vram_mb": 2048,
                "duration_s": 0.25,
            },
        }


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs = []
    for index, template in enumerate(exp6289.MODEL_SPECS):
        path = tmp_path / f"model_{index}-Q4_K_M.gguf"
        path.write_bytes(f"fake-model-{index}".encode("ascii"))
        specs.append({**template, "model_path": str(path), "gpu": index % 2})
    return specs


def test_req_constraint_6289_spec_declares_artifact_contract() -> None:
    """REQ-CONSTRAINT-6289: OpenSpec anchors the 6289 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6289") :]

    for marker in (
        "SCENARIO-CONSTRAINT-6289-SEALED-SOTA",
        "SCENARIO-CONSTRAINT-6289-MATCHED-BUDGETS",
        "SCENARIO-CONSTRAINT-6289-ORACLE-VALUE",
        exp6289.RESULT_RELATIVE_PATH.as_posix(),
        "`source_model_weight_mutation_count`",
        "`verifier_is_oracle`",
        "Cold exact solve success SHALL NOT count as model value",
    ):
        assert marker in section
    for field in exp6289.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_constraint_6289_model_specs_are_mandated_only() -> None:
    """SCENARIO-CONSTRAINT-6289-SEALED-SOTA: headline rows use mandated GGUFs."""

    ids = [spec["hf_id"] for spec in exp6289.MODEL_SPECS]

    assert ids == list(exp6289.MANDATED_MODEL_IDS)
    assert "Qwen/Qwen3.5-0.8B" not in ids
    assert "google/gemma-4-E4B-it" not in ids
    assert len(set(ids)) == 3


def test_scenario_constraint_6289_sealed_prompts_hide_sidecar_and_atom_ids() -> None:
    """SCENARIO-CONSTRAINT-6289-SEALED-SOTA: prompts expose only ordinary text."""

    bundle = exp6289.build_sealed_task_bundle(date="20260810", tasks_per_model=5)
    receipt = exp6289.formal_nonexposure_receipt(bundle)

    assert receipt["all_clear"] is True
    assert set(bundle["tasks_by_model"]) == set(exp6289.MANDATED_MODEL_IDS)
    for tasks in bundle["tasks_by_model"].values():
        for task in tasks:
            prompt = task["prompt_text"]
            assert ":-" not in prompt
            assert "{" not in prompt
            assert "}" not in prompt
            assert "answer set" not in prompt.lower()
            for atom in task["allowed_labels"]:
                if "_" in atom:
                    assert atom not in prompt
            for answer in task["exact_answer_sets"]:
                visible = exp6289.format_visible_answer(task, answer)
                assert visible not in prompt

    leaked = deepcopy(bundle)
    task = next(
        item
        for item in leaked["tasks_by_model"][exp6289.MANDATED_MODEL_IDS[0]]
        if item["exact_answer_sets"] and any("_" in atom for atom in item["allowed_labels"])
    )
    task["prompt_text"] += "\n" + task["program_text"] + "\n" + exp6289.format_visible_answer(
        task, task["exact_answer_sets"][0] if task["exact_answer_sets"] else []
    )
    task["prompt_text"] += "\nrule_id total_energy violation answer set"
    task["prompt_text"] += "\n" + next(atom for atom in task["allowed_labels"] if "_" in atom)
    leaked_receipt = exp6289.formal_nonexposure_receipt(leaked)
    assert leaked_receipt["all_clear"] is False
    assert leaked_receipt["asp_syntax_exposure_count"] >= 1
    assert leaked_receipt["exact_answer_exposure_count"] >= 1
    assert leaked_receipt["formal_atom_id_exposure_count"] >= 1
    assert leaked_receipt["verifier_receipt_exposure_count"] >= 1


def test_req_constraint_6289_artifact_schema_and_matched_arms(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6289: artifact carries required fields and matched arms."""

    result_path = tmp_path / exp6289.RESULT_RELATIVE_PATH.name
    artifact = exp6289.run(
        date="20260810",
        result_path=result_path,
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        tasks_per_model=5,
        duration_s=3.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp6289.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6289.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(exp6289.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["MODEL_SPECS"] == exp6289.MODEL_SPECS
    assert artifact["models_used"] == list(exp6289.MANDATED_MODEL_IDS)
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exact_solver_oracle_receipt"]["cold_exact_solver_counts_as_model_value"] is False
    assert artifact["warm_start_value_ready_score"] == 1.0
    assert artifact["harmful_regressions"]["exact_validity_harm_count"] == 0
    assert all(
        exp6289.is_terminal_disposition(value)
        for value in artifact["terminal_model_dispositions"].values()
    )
    assert (
        artifact["arm_definitions_and_fixed_compute_budget"]["repeated_generation"][
            "generation_samples"
        ]
        == exp6289.REPEATED_GENERATION_BUDGET
    )
    assert artifact["partial_evidence_exact_completion_results"]["warm_minus_cold_work_delta"] > 0
    assert artifact["reproducibility_checksum"] == exp6289.reproducibility_checksum(artifact)
    exp6289.validate_artifact(artifact)


def test_scenario_constraint_6289_rejects_model_substitution(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6289-SEALED-SOTA: legacy substitution fails closed."""

    specs = _fake_model_specs(tmp_path)
    specs[1]["hf_id"] = "Qwen/Qwen3.5-0.8B"

    normalized = exp6289.normalize_model_records(specs, preflight_tokenizers=False)

    assert "mandated_model_id_order" in normalized["blocked_reasons"]
    assert "legacy_model_substitution:Qwen/Qwen3.5-0.8B" in normalized["blocked_reasons"]
    assert normalized["all_resolved"] is False


def test_scenario_constraint_6289_qwen_zero_token_rows_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6289-ORACLE-VALUE: zero-token Qwen evidence is rejected."""

    artifact = exp6289.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(zero_token_qwen=True),
        model_specs=_fake_model_specs(tmp_path),
        tasks_per_model=5,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    control = artifact["qwen_zero_token_control"]
    assert control["qwen_zero_token_rows"] >= 1
    assert control["accepted_as_evidence_count"] == 0
    assert control["terminal_control"].startswith("complete:")
    assert any(
        row["terminal_disposition"].startswith("blocked:")
        for row in artifact["prompt_seed_token_timeout_and_terminal_disposition_by_row"]
        if row["model_hf_id"] == exp6289.MANDATED_MODEL_IDS[0]
    )


def test_scenario_constraint_6289_nonterminal_rows_become_terminal_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6289-ORACLE-VALUE: nonterminal model rows close."""

    nonterminal = exp6289.MANDATED_MODEL_IDS[1]
    artifact = exp6289.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(nonterminal_model=nonterminal),
        model_specs=_fake_model_specs(tmp_path),
        tasks_per_model=5,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    assert artifact["terminal_model_dispositions"][nonterminal].startswith("blocked:")
    assert exp6289.is_terminal_disposition(artifact["terminal_model_dispositions"][nonterminal])
    assert artifact["status"] == "blocked"
    exp6289.validate_artifact(artifact)


def test_req_constraint_6289_resolves_all_three_via_cached_sota_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6289: cache resolution reuses cached_sota_pair."""

    paths = {}
    for spec in exp6289.MODEL_SPECS:
        path = tmp_path / (exp6289.model_slug(spec["hf_id"]) + "-Q4_K_M.gguf")
        path.write_bytes(spec["hf_id"].encode("ascii"))
        paths[spec["hf_id"]] = str(path)
    calls: list[tuple[int, ...] | None] = []

    def fake_cached_sota_pair(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(model_indices)
        selected = (
            [exp6289.MODEL_SPECS[index] for index in model_indices]
            if model_indices is not None
            else exp6289.MODEL_SPECS[:2]
        )
        return [
            {
                "name": row["name"],
                "hf_id": row["hf_id"],
                "gpu": gpu_indices[min(i, 1)],
                "model_path": paths[row["hf_id"]],
            }
            for i, row in enumerate(selected)
        ]

    monkeypatch.setattr(exp6289, "cached_sota_pair", fake_cached_sota_pair)
    resolved = exp6289.resolve_mandated_model_specs(preflight_tokenizers=False)

    assert resolved["all_resolved"] is True
    assert set(calls) >= {(0, 2), (1, 0)}
    assert [row["hf_id"] for row in resolved["records"]] == list(exp6289.MANDATED_MODEL_IDS)
    assert all("cached_sota_pair" in row["cache_policy"] for row in resolved["records"])


def test_scenario_constraint_6289_cuda_fallback_blocks_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CONSTRAINT-6289-ORACLE-VALUE: missing CUDA offload blocks."""

    monkeypatch.setattr(
        exp6289,
        "llama_cpp_python_offload_receipt",
        lambda: {"available": True, "supports_gpu_offload": False},
    )

    result = exp6289.LiveLlamaCppBackend().generate_model(
        {"hf_id": exp6289.MANDATED_MODEL_IDS[0], "model_path": "/missing/model.gguf"},
        [],
    )

    assert result["rows"] == []
    assert result["receipt"]["terminal_disposition"].startswith("blocked:")
    assert result["receipt"]["gpu_offload"]["observed"] is False


def test_scenario_constraint_6289_oracle_value_laundering_is_rejected(
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6289-ORACLE-VALUE: cold exact value cannot open readiness."""

    artifact = exp6289.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        tasks_per_model=5,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    laundered = deepcopy(artifact)
    laundered["partial_evidence_exact_completion_results"]["warm_minus_cold_work_delta"] = 0
    laundered["partial_evidence_continuous_refinement_results"][
        "evidence_warm_minus_blank_success_delta"
    ] = 0
    laundered["warm_start_value_ready_score"] = 1.0
    laundered["reproducibility_checksum"] = exp6289.reproducibility_checksum(laundered)
    with pytest.raises(ValueError, match="oracle_value_laundering"):
        exp6289.validate_artifact(laundered)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = exp6289.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp6289.validate_artifact(bad_oracle)


def test_req_constraint_6289_defensive_branches_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6289: defensive branches fail closed deterministically."""

    bundle = exp6289.build_sealed_task_bundle(date="20260810", tasks_per_model=5)
    task = next(
        item
        for item in bundle["tasks_by_model"][exp6289.MANDATED_MODEL_IDS[0]]
        if item["exact_answer_sets"]
    )
    wrong = next(
        [label]
        for label in task["allowed_labels"]
        if [label] not in task["exact_answer_sets"]
    )
    wrong_score = exp6289.score_text_response(
        task,
        exp6289.format_visible_answer(task, wrong),
        generated_token_count=4,
        row_id="wrong",
    )
    assert wrong_score["parse_success"] is True
    assert wrong_score["semantic_valid"] is False
    assert wrong_score["residual_rule_violations"]

    empty_score = exp6289.score_text_response(
        task,
        "ANSWER: EMPTY",
        generated_token_count=2,
        row_id="empty",
    )
    assert empty_score["parser"] == "visible_empty"

    rejected = exp6289.score_text_response(
        task,
        "No usable assignment text.",
        generated_token_count=4,
        row_id="reject",
    )
    assert rejected["parse_success"] is False
    assert rejected["terminal_disposition"] == "complete: parser_rejected_but_row_terminal"

    missing_specs = _fake_model_specs(tmp_path)
    Path(missing_specs[0]["model_path"]).unlink()
    blocked = exp6289.run(
        date="20260810",
        result_path=tmp_path / "blocked.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=missing_specs,
        tasks_per_model=5,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["raw_output_paths_and_hashes"][exp6289.MANDATED_MODEL_IDS[0]]["row_count"] == 0

    monkeypatch.setattr(
        exp6289,
        "gguf_tokenizer_loadable",
        lambda _path: (False, "forced tokenizer failure"),
    )
    tokenizer_blocked = exp6289.normalize_model_records(
        _fake_model_specs(tmp_path),
        preflight_tokenizers=True,
    )
    assert any(reason.startswith("tokenizer_not_loadable:") for reason in tokenizer_blocked["blocked_reasons"])

    receipt = exp6289._terminalized_receipt("m", {"terminal_disposition": "complete: ok"})
    assert receipt["gpu_offload"]["observed"] is False

    one = {
        "arm": "one_shot",
        "model_hf_id": "m",
        "task_id": "t",
        "fixture_id": "f",
        "fixture_family": "family",
        "parse_success": True,
        "exact_valid": False,
        "evidence_supported": False,
        "state_evaluations": 1,
        "verifier_work": 1,
        "model_generated_tokens": 1,
        "model_prompt_tokens": 1,
        "wall_time_s": 0.0,
        "selected_labels": [],
    }
    warm = {**one, "arm": "partial_evidence_exact_completion", "evidence_supported": True}
    repeated = {**one, "arm": "repeated_generation"}
    route = exp6289._compute_balanced_route_record("m", task, one, repeated, warm)
    assert "exact_complete" in route["route_actions"]

    sparse = {
        "records": {
            arm: ([one] if arm == "one_shot" else [])
            for arm in exp6289.ARMS
        }
    }
    deltas = exp6289.paired_deltas(sparse)
    assert deltas[exp6289.MANDATED_MODEL_IDS[0]] == {}
    assert exp6289._paired_interval([])["sample_size"] == 0
    assert exp6289._paired_interval([1, 0])["sample_size"] == 2
    assert "failing verification" in exp6289.honest_verdict("complete_no_value", 0.0, False)

    artifact = exp6289.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        tasks_per_model=5,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6289.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    bad_cases = [
        ("status", None, "missing required"),
        ("field_principles", {}, "field_principles"),
        ("field_provenance", {}, "field_provenance"),
        ("MODEL_SPECS", [], "model_specs"),
        ("models_used", [], "models_used"),
        ("inference_substrate", "wrong", "inference_substrate"),
        ("source_model_weight_mutation_count", True, "source_model_weight_mutation_count"),
        ("terminal_model_dispositions", {}, "terminal_model_dispositions"),
    ]
    for field, value, match in bad_cases:
        bad = deepcopy(artifact)
        if value is None:
            bad.pop(field)
        else:
            bad[field] = value
        bad["reproducibility_checksum"] = exp6289.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            exp6289.validate_artifact(bad)

    nonterminal = deepcopy(artifact)
    nonterminal["terminal_model_dispositions"][exp6289.MANDATED_MODEL_IDS[0]] = "running"
    nonterminal["reproducibility_checksum"] = exp6289.reproducibility_checksum(nonterminal)
    with pytest.raises(ValueError, match="nonterminal_model_disposition"):
        exp6289.validate_artifact(nonterminal)

    cold_value = deepcopy(artifact)
    cold_value["exact_solver_oracle_receipt"]["cold_exact_solver_counts_as_model_value"] = True
    cold_value["reproducibility_checksum"] = exp6289.reproducibility_checksum(cold_value)
    with pytest.raises(ValueError, match="oracle_value_laundering"):
        exp6289.validate_artifact(cold_value)

    bad_prefix = deepcopy(artifact)
    bad_prefix["honest_verdict"] = "running"
    bad_prefix["reproducibility_checksum"] = exp6289.reproducibility_checksum(bad_prefix)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp6289.validate_artifact(bad_prefix)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "wrong"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6289.validate_artifact(bad_checksum)
