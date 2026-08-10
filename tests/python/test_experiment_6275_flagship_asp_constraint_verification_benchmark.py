"""Tests for Exp6275 sealed flagship ASP benchmark.

Spec refs: REQ-CONSTRAINT-6275,
SCENARIO-CONSTRAINT-6275-SEALED-PROMPTS,
SCENARIO-CONSTRAINT-6275-SEPARATE-OUTCOMES,
SCENARIO-CONSTRAINT-6275-EXACT-REPAIR,
SCENARIO-CONSTRAINT-6275-BLOCKED-CELL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6275_flagship_asp_constraint_verification_benchmark as exp6275


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


class FakeBackend:
    """Small deterministic backend that returns parseable wrong then exact rows."""

    def generate_model(self, model_spec: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for job in jobs:
            task = job["task"]
            if job["sample_index"] == 0:
                label = task["allowed_labels"][0] if task["allowed_labels"] else "NONE"
                text = f"ANSWER: {label}"
            elif task["exact_answer_sets"]:
                text = "ANSWER: " + ", ".join(task["exact_answer_sets"][0])
            else:
                text = "IMPOSSIBLE"
            rows.append(
                {
                    "task_id": task["task_id"],
                    "sample_index": job["sample_index"],
                    "seed": job["seed"],
                    "raw_output": text,
                    "generated_token_count": len(text.split()),
                    "prompt_token_count": len(job["prompt_text"].split()),
                    "latency_s": 0.01,
                    "finish_reason": "stop",
                    "timeout": False,
                }
            )
        return {
            "rows": rows,
            "receipt": {
                "terminal_disposition": "complete",
                "gpu_offload": {"requested": True, "observed": True, "peak_vram_mb": 1024},
                "peak_vram_mb": 1024,
            },
        }


class BlockingBackend:
    """Backend that stops one model cell to test honest blocked receipts."""

    def generate_model(self, model_spec: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
        if model_spec["hf_id"] == "unsloth/gemma-4-31B-it-GGUF":
            return {
                "rows": [],
                "receipt": {
                    "terminal_disposition": "blocked: no_gpu_offload_receipt",
                    "gpu_offload": {"requested": True, "observed": False, "peak_vram_mb": 0},
                    "peak_vram_mb": 0,
                    "failed_cell": "no_gpu_offload_receipt",
                },
            }
        return FakeBackend().generate_model(model_spec, jobs)


class TimeoutBackend(FakeBackend):
    """Backend that marks raw rows as timed out while still returning receipts."""

    def generate_model(self, model_spec: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
        result = super().generate_model(model_spec, jobs)
        for row in result["rows"]:
            row["timeout"] = True
        return result


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs = []
    for index, template in enumerate(exp6275.MODEL_SPECS):
        path = tmp_path / f"model_{index}.gguf"
        path.write_bytes(f"fake-model-{index}".encode("ascii"))
        specs.append({**template, "model_path": str(path), "gpu": index % 2})
    return specs


def test_req_6275_spec_declares_sealed_benchmark_contract() -> None:
    """REQ-CONSTRAINT-6275: OpenSpec declares the Exp6275 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6275") :]

    for marker in (
        "SCENARIO-CONSTRAINT-6275-SEALED-PROMPTS",
        "SCENARIO-CONSTRAINT-6275-SEPARATE-OUTCOMES",
        "SCENARIO-CONSTRAINT-6275-EXACT-REPAIR",
        "SCENARIO-CONSTRAINT-6275-BLOCKED-CELL",
        "results/experiment_6275_flagship_asp_constraint_verification_benchmark.json",
        'inference_substrate="live_llm_inference"',
        "`verifier_is_oracle=true`",
        "oracle-distinct verifier moat",
    ):
        assert marker in section


def test_scenario_6275_sealed_prompts_hide_formal_sidecars_and_answers() -> None:
    """SCENARIO-CONSTRAINT-6275-SEALED-PROMPTS: prompts hide formal data."""

    benchmark = exp6275.build_sealed_benchmark(date="20260810", tasks_per_model=30)
    receipt = exp6275.formal_nonexposure_receipt(benchmark)

    assert set(benchmark["tasks_by_model"]) == set(exp6275.MANDATED_MODEL_IDS)
    assert all(len(tasks) == 30 for tasks in benchmark["tasks_by_model"].values())
    assert receipt["formal_sidecar_exposure_count"] == 0
    assert receipt["exact_answer_exposure_count"] == 0
    assert receipt["asp_syntax_exposure_count"] == 0
    assert receipt["all_clear"] is True

    for tasks in benchmark["tasks_by_model"].values():
        for task in tasks:
            prompt = task["prompt_text"]
            assert ":-" not in prompt
            assert "{" not in prompt
            assert "}" not in prompt
            for answer in task["exact_answer_sets"]:
                if len(answer) > 1:
                    assert "ANSWER: " + ", ".join(answer) not in prompt


def test_scenario_6275_format_and_semantic_outcomes_are_separate() -> None:
    """SCENARIO-CONSTRAINT-6275-SEPARATE-OUTCOMES: parse and semantics split."""

    benchmark = exp6275.build_sealed_benchmark(date="20260810", tasks_per_model=30)
    task = next(
        task
        for task in benchmark["tasks_by_model"][exp6275.MANDATED_MODEL_IDS[0]]
        if task["exact_answer_sets"]
    )

    parsed_wrong = exp6275.score_output(task, f"ANSWER: {task['allowed_labels'][0]}")
    assert parsed_wrong["parse_success"] is True
    assert parsed_wrong["semantic_valid"] is False
    assert parsed_wrong["exact_certificate_present"] is True
    assert parsed_wrong["residual_rule_violations"]

    loose = exp6275.score_output(
        task,
        f"I would choose {task['exact_answer_sets'][0][0]} because it fits.",
        allow_format_repair=True,
    )
    strict = exp6275.score_output(task, loose["raw_output"], allow_format_repair=False)
    assert strict["parse_success"] is False
    assert loose["parse_success"] is True


def test_scenario_6275_energy_repair_uses_exact_certificates() -> None:
    """SCENARIO-CONSTRAINT-6275-EXACT-REPAIR: repair accepts only oracle-valid states."""

    benchmark = exp6275.build_sealed_benchmark(date="20260810", tasks_per_model=30)
    satisfiable = next(
        task
        for task in benchmark["tasks_by_model"][exp6275.MANDATED_MODEL_IDS[0]]
        if task["exact_answer_sets"]
    )
    unsat = next(
        task
        for task in benchmark["tasks_by_model"][exp6275.MANDATED_MODEL_IDS[0]]
        if not task["exact_answer_sets"]
    )

    repaired = exp6275.energy_guided_repair(satisfiable, [satisfiable["allowed_labels"][0]])
    assert repaired["semantic_valid"] is True
    assert repaired["exact_certificate_present"] is True
    assert repaired["repaired_labels"] in satisfiable["exact_answer_sets"]
    assert repaired["residual_rule_violations"] == []

    failed = exp6275.energy_guided_repair(unsat, [unsat["allowed_labels"][0]])
    assert failed["semantic_valid"] is False
    assert failed["exact_certificate_present"] is True
    assert failed["repaired_labels"] == [unsat["allowed_labels"][0]]
    assert failed["residual_rule_violations"]


def test_req_6275_artifact_schema_metrics_and_sidecars(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6275: artifact schema, metrics, sidecars, and zeros hold."""

    result_path = tmp_path / exp6275.RESULT_RELATIVE_PATH.name
    artifact = exp6275.run(
        date="20260810",
        result_path=result_path,
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        duration_s=2.5,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp6275.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6275.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["weight_mutation_count"] == 0
    assert type(artifact["weight_mutation_count"]) is int
    assert artifact["external_text_scorer_call_count"] == 0
    assert type(artifact["external_text_scorer_call_count"]) is int
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["flagship_asp_event_corpus_ready_score"] == 1.0
    assert artifact["formal_sidecar_nonexposure_receipt"]["all_clear"] is True
    assert artifact["protected_files_unchanged"]["scripts/research_conductor.py"]["unchanged"] is True

    for hf_id, family_counts in artifact["task_count_by_model_and_family"].items():
        assert sum(family_counts.values()) >= 30, hf_id
    for metric_name in (
        "parse_success_by_model_family_arm",
        "semantic_validity_by_model_family_arm",
        "exact_certificate_coverage_by_model_family_arm",
    ):
        assert set(artifact[metric_name]) == set(exp6275.MANDATED_MODEL_IDS)
    assert artifact["semantic_repair_margin_by_model_family"]
    assert artifact["format_repair_margin_by_model_family"]
    assert artifact["reproducibility_checksum"] == exp6275.reproducibility_checksum(artifact)


def test_scenario_6275_blocked_model_cell_stops_honestly(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6275-BLOCKED-CELL: blocked cells are not fabricated."""

    artifact = exp6275.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=BlockingBackend(),
        model_specs=_fake_model_specs(tmp_path),
        duration_s=3.0,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
    )

    blocked_model = "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["terminal_disposition_by_model"][blocked_model].startswith("blocked:")
    assert artifact["failed_or_timeout_cells"]
    assert artifact["raw_output_paths_and_hashes"][blocked_model]["row_count"] == 0
    assert artifact["honest_verdict"].startswith("complete_partial:")
    exp6275.validate_artifact(artifact)


def test_req_6275_validation_fails_on_moat_or_missing_principle(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6275: validation rejects missing principles and moat claims."""

    artifact = exp6275.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    missing = json.loads(json.dumps(artifact))
    missing["field_principles"].pop("status")
    missing["reproducibility_checksum"] = exp6275.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="field_principles"):
        exp6275.validate_artifact(missing)

    moat = json.loads(json.dumps(artifact))
    moat["honest_verdict"] = "complete_ready: oracle-distinct moat"
    moat["reproducibility_checksum"] = exp6275.reproducibility_checksum(moat)
    with pytest.raises(ValueError, match="moat"):
        exp6275.validate_artifact(moat)

    plain_moat = json.loads(json.dumps(artifact))
    plain_moat["honest_verdict"] = "complete_ready: moat"
    plain_moat["reproducibility_checksum"] = exp6275.reproducibility_checksum(plain_moat)
    with pytest.raises(ValueError, match="moat"):
        exp6275.validate_artifact(plain_moat)


def test_req_6275_defensive_branches_are_fail_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6275: defensive branches reject leakage and bad schema."""

    benchmark = exp6275.build_sealed_benchmark(date="20260810", tasks_per_model=30)
    hf_id = exp6275.MANDATED_MODEL_IDS[0]
    task = next(
        item
        for item in benchmark["tasks_by_model"][hf_id]
        if item["exact_answer_sets"] and len(item["exact_answer_sets"][0]) > 1
    )

    leaked = json.loads(json.dumps(benchmark))
    leaked_task = next(
        item
        for item in leaked["tasks_by_model"][hf_id]
        if item["task_id"] == task["task_id"]
    )
    leaked_task["prompt_text"] += (
        "\n" + leaked_task["program_text"]
        + "\nANSWER: " + ", ".join(leaked_task["exact_answer_sets"][0])
        + "\n:- rule_id total_energy violation"
    )
    receipt = exp6275.formal_nonexposure_receipt(leaked)
    assert receipt["formal_sidecar_exposure_count"] >= 1
    assert receipt["exact_answer_exposure_count"] >= 1
    assert receipt["asp_syntax_exposure_count"] >= 1
    assert receipt["verifier_receipt_exposure_count"] >= 1

    assert exp6275.parse_assignment(task, "ANSWER: NONE").parse_success is True
    assert exp6275.parse_assignment(task, "ANSWER: not_a_label").parse_success is False
    duplicate = f"ANSWER: {task['allowed_labels'][0]}, {task['allowed_labels'][0]}"
    assert exp6275.parse_assignment(task, duplicate).error == "duplicate_label"
    repaired_fail = exp6275.score_output(task, "no usable labels here", allow_format_repair=True)
    assert repaired_fail["parse_success"] is False
    abstain_arm = exp6275._arm_results(task, [{"raw_output": "IMPOSSIBLE"}])
    assert abstain_arm["energy_guided_repair"]["repair_distance"] is None
    assert exp6275._natural_term({"kind": "stable_support", "payload": {}}).startswith("The selected")

    fixtures = exp6275.exp6274.build_fixture_manifest()
    with pytest.raises(ValueError, match="tasks_per_model"):
        exp6275._balanced_fixture_selection(fixtures[:1], 2)
    assert exp6275._paired_interval([1])["ci95"] == [1.0, 1.0]
    assert exp6275._extract_quantization(Path("thing-Q5_K_M.gguf")) == "Q5_K_M"
    assert exp6275._extract_revision(Path("/tmp/snapshots/abc/model.gguf")) == "abc"
    assert exp6275._blocked_model_receipt({"hf_id": "x"}, "missing")["peak_vram_mb"] == 0
    assert exp6275._honest_verdict("blocked", {}, True).startswith("blocked:")
    assert "test_exit_codes" in exp6275._honest_verdict("complete_partial", {"m": "complete"}, False)

    wrong_specs = _fake_model_specs(tmp_path)
    wrong_specs[0]["hf_id"] = "wrong/model"
    normalized = exp6275._normalize_model_records(wrong_specs)
    assert "mandated_model_id_order" in normalized["blocked_reasons"]

    missing_specs = _fake_model_specs(tmp_path)
    Path(missing_specs[0]["model_path"]).unlink()
    blocked_artifact = exp6275.run(
        date="20260810",
        result_path=tmp_path / "blocked.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=missing_specs,
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    assert blocked_artifact["status"] == "blocked"
    assert blocked_artifact["failed_or_timeout_cells"]

    timeout_artifact = exp6275.run(
        date="20260810",
        result_path=tmp_path / "timeout.json",
        artifact_dir=tmp_path,
        backend=TimeoutBackend(),
        model_specs=_fake_model_specs(tmp_path),
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    assert any(cell["terminal_disposition"] == "timeout" for cell in timeout_artifact["failed_or_timeout_cells"])

    partial = exp6275.aggregate_metrics(
        benchmark,
        {model: [] for model in exp6275.MANDATED_MODEL_IDS},
        {
            "by_model": {
                hf_id: [
                    {
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "arm": "one_shot",
                        "parse_success": True,
                        "semantic_valid": False,
                        "exact_certificate_present": True,
                        "residual_rule_violation_count": 1,
                        "abstention": False,
                    }
                ]
            }
        },
    )
    assert partial["paired_intervals"][hf_id][task["family"]]["semantic"]["sample_size"] == 0

    artifact = exp6275.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        artifact_dir=tmp_path,
        backend=FakeBackend(),
        model_specs=_fake_model_specs(tmp_path),
        duration_s=1.0,
        test_exit_codes={command: 0 for command in exp6275.DEFAULT_TEST_COMMANDS},
        write=False,
    )
    bad_cases = [
        ("status", None, "missing required"),
        ("field_provenance", {}, "field_provenance"),
        ("inference_substrate", "wrong", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("weight_mutation_count", True, "weight_mutation_count"),
        ("external_text_scorer_call_count", True, "external_text_scorer_call_count"),
        ("task_count_by_model_and_family", {hf_id: {"x": 1}}, "task_count"),
    ]
    for field, value, match in bad_cases:
        bad = json.loads(json.dumps(artifact))
        if value is None:
            bad.pop(field)
        else:
            bad[field] = value
        bad["reproducibility_checksum"] = exp6275.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            exp6275.validate_artifact(bad)

    bad_checksum = json.loads(json.dumps(artifact))
    bad_checksum["reproducibility_checksum"] = "wrong"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6275.validate_artifact(bad_checksum)
