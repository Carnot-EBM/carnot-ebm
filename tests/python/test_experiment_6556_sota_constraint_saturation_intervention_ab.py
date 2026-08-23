"""Tests for Exp6556 SOTA constraint saturation intervention A/B.

Spec refs: REQ-BENCH-6556, SCENARIO-BENCH-6556-GATE,
SCENARIO-BENCH-6556-MATCHED-ARMS, SCENARIO-BENCH-6556-CHECKS,
SCENARIO-BENCH-6556-INTERVENTIONS, SCENARIO-BENCH-6556-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6556_sota_constraint_saturation_intervention_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6556", "exit_code": 0}]


class FakeBackend:
    """Small backend that exercises routing without loading local GGUF weights."""

    def __init__(self, *, mode: str = "invalid") -> None:
        self.mode = mode
        self.closed = False

    def load_model(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "loader": "llama_cpp.Llama",
            "load_ok": True,
            "smoke_ok": True,
            "embedded_tokenizer_ok": True,
            "process_id": 4242,
            "gpu": spec["gpu"],
            "load_s": 0.01,
            "peak_vram_mb": 2048,
            "error": "",
        }

    def tokenize(self, _spec: dict[str, Any], text: str) -> int:
        return max(1, len(text.split()))

    def infer(
        self,
        *,
        spec: dict[str, Any],
        prompt: str,
        max_tokens: int,
        timeout_s: float,
        unit_key: str,
    ) -> dict[str, Any]:
        del prompt, timeout_s, unit_key
        output = (
            'FINAL_JSON: {"not": "the expected assignment"}'
            if self.mode == "invalid"
            else 'FINAL_JSON: {"synthetic": "complete"}'
        )
        return {
            "terminal_status": "terminal",
            "timeout": False,
            "parse_failure": False,
            "output_text": output,
            "prompt_tokens": 11 + int(spec["gpu"]),
            "output_tokens": min(max_tokens, 7),
            "wall_time_s": 0.002,
            "first_token_time_s": 0.001,
            "error": "",
        }

    def close(self) -> None:
        self.closed = True


def _runtime() -> dict[str, Any]:
    return {
        "gpu": {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": 20000,
                    "driver_version": "610.43.03",
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "vram_total_mb": 24576,
                    "vram_free_mb": 20000,
                    "driver_version": "610.43.03",
                },
            ],
        },
        "llama_cpp": {
            "available": True,
            "llama_cli_exists": True,
            "llama_cli_sha256": "sha256:" + "1" * 64,
            "gpu_offload_supported": True,
            "cuda_backend_available": True,
        },
        "z3_version": "4.16.0",
        "disk": {"checkpoint_free_bytes": 10_000_000},
    }


def _model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    specs = []
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"gguf-{hf_id}".encode())
        specs.append(
            {
                "name": mod.MODEL_NAMES_BY_HF_ID[hf_id],
                "hf_id": hf_id,
                "role": mod.MODEL_ROLES_BY_HF_ID[hf_id],
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": str(path),
            }
        )
    return specs


def _artifact(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "experiment_6556.json",
        fixture_path=REPO / mod.FIXTURE_RELATIVE_PATH,
        checkpoint_path=tmp_path / "experiment_6556.checkpoint.json",
        write=True,
        duration_s=123.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=_model_specs(tmp_path),
        inference_backend=kwargs.pop("backend", FakeBackend()),
        runtime_state_override=_runtime(),
        cached_pair_override=[
            {"hf_id": mod.MANDATED_HF_IDS[0], "gpu": 0, "model_path": "qwen.gguf"},
            {"hf_id": mod.MANDATED_HF_IDS[2], "gpu": 1, "model_path": "gemma31.gguf"},
        ],
        **kwargs,
    )


def test_req_bench_6556_spec_declares_sota_intervention_contract() -> None:
    """REQ-BENCH-6556: OpenSpec owns the Exp6556 comparison contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6556") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6556-GATE",
        "SCENARIO-BENCH-6556-MATCHED-ARMS",
        "SCENARIO-BENCH-6556-CHECKS",
        "SCENARIO-BENCH-6556-INTERVENTIONS",
        "SCENARIO-BENCH-6556-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "cached_sota_pair(gpu_indices=(0, 1))",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenarios_bench_6556_complete_comparison_closes_positive(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6556-MATCHED-ARMS/CHECKS/INTERVENTIONS/TERMINAL: rows close."""

    backend = FakeBackend()
    artifact = _artifact(tmp_path, backend=backend)
    written = json.loads((tmp_path / "experiment_6556.json").read_text(encoding="utf-8"))

    assert backend.closed is True
    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_sota_constraint_saturation_intervention_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["constraint_saturation_intervention_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert artifact["sample_size_and_power_contract"]["lineage_count_per_model"] == {
        hf_id: 36 for hf_id in mod.MANDATED_HF_IDS
    }
    assert artifact["sample_size_and_power_contract"]["ready_floor_met"] is True
    assert artifact["frozen_arm_and_budget_contract"]["arms"] == list(mod.ARM_IDS)
    assert artifact["frozen_arm_and_budget_contract"]["longer_flat_is_required_control"] is True

    expected_rows = len(mod.MANDATED_HF_IDS) * 36 * len(mod.ARM_IDS)
    assert len(artifact["per_unit_rows"]) == expected_rows
    assert len(artifact["charged_cost_rows"]) == expected_rows
    assert len(artifact["per_clause_and_joint_result_rows"]) == expected_rows
    assert all(row["request_sha256"] for row in artifact["per_unit_rows"])
    assert all(row["response_sha256"] for row in artifact["per_unit_rows"])
    assert all(
        row["exact_final_joint_check"] for row in artifact["per_clause_and_joint_result_rows"]
    )

    route_rows = artifact["route_decomposition_and_fallback_rows"]
    assert any(row["arm_id"] == "bounded_decomposition" for row in route_rows)
    assert any(row["fallback_used"] for row in route_rows)
    assert all(
        row["clauses_preserved"] for row in route_rows if row["arm_id"] == "bounded_decomposition"
    )

    harm = artifact["harmful_intervention_ledger"]
    assert harm["regression_count_vs_flat"] == 0
    assert harm["regression_count_vs_longer_flat"] == 0
    assert harm["recovery_count_vs_longer_flat"] > 0
    assert artifact["constraint_load_phase_curve"]["phase_curve_established"] is True
    assert artifact["aggregate_row_recomputation"]["benefit_beyond_longer_flat"] is True
    assert artifact["gate_check_summary"]["failed_checks"] == []


def test_scenario_bench_6556_failed_gate_blocks_before_rows(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6556-GATE: failed fixture readiness writes a blocked artifact."""

    gate_payload = deepcopy(mod.load_json(REPO / mod.UPSTREAM_FIXTURE_RELATIVE_PATH))
    gate_payload["constraint_saturation_fixture_ready_score"] = 0.0
    blocked = _artifact(tmp_path, upstream_fixture_payload=gate_payload)

    assert blocked["status"] == "blocked_sota_constraint_saturation_intervention_ab"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["constraint_saturation_intervention_ready_score"] == 0.0
    assert blocked["per_unit_rows"] == []
    assert "upstream_fixture_ready" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6556_validation_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6556-TERMINAL: invalid receipts and unsafe claims fail closed."""

    artifact = _artifact(tmp_path)
    cases: list[tuple[dict[str, Any], str]] = []

    missing = deepcopy(artifact)
    del missing["status"]
    cases.append((missing, "missing required fields"))

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "ready"
    cases.append((bad_verdict, "honest_verdict lacks terminal prefix"))

    bad_class = deepcopy(artifact)
    bad_class["verdict_class"] = "circular_positive"
    cases.append((bad_class, "verdict_class outside Exp6556 enum"))

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "other"
    cases.append((bad_substrate, "inference_substrate mismatch"))

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    cases.append((bad_oracle, "verifier_is_oracle must be false"))

    bad_ready = deepcopy(artifact)
    bad_ready["aggregate_row_recomputation"]["invalid_release_delta"] = 1
    cases.append((bad_ready, "positive score requires no safety regression"))

    bad_rows = deepcopy(artifact)
    bad_rows["per_unit_rows"] = bad_rows["per_unit_rows"][:-1]
    cases.append((bad_rows, "matched row count mismatch"))

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    cases.append((bad_provenance, "field_provenance must cover required fields"))

    bad_score = deepcopy(artifact)
    bad_score["constraint_saturation_intervention_ready_score"] = 0.0
    cases.append((bad_score, "ready score mismatch"))

    bad_positive_class = deepcopy(artifact)
    bad_positive_class["verdict_class"] = "partial"
    cases.append((bad_positive_class, "positive score requires positive verdict_class"))

    bad_benefit = deepcopy(artifact)
    bad_benefit["aggregate_row_recomputation"]["benefit_beyond_longer_flat"] = False
    cases.append((bad_benefit, "positive score requires benefit beyond longer-flat"))

    for payload, expected in cases:
        payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        assert any(expected in error for error in mod.validate_artifact(payload)), expected

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(checksum)


def test_scenario_bench_6556_helper_edges_are_deterministic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SCENARIO-BENCH-6556-CHECKS: parsing, selection, and blocked preconditions are stable."""

    fixture_rows = mod.load_jsonl(REPO / mod.FIXTURE_RELATIVE_PATH)
    selected = mod.freeze_held_cells(fixture_rows)
    assert len({row["lineage_id"] for row in selected}) == 36
    assert {row["domain"] for row in selected} == {"logic_grid", "scheduling", "seating"}

    exact = {"A": {"slot": "B"}}
    parsed = mod.parse_final_json('thinking\nFINAL_JSON: {"A": {"slot": "B"}}')
    assert parsed == exact
    assert mod.parse_final_json("no json here") is None
    assert mod.parse_final_json("FINAL_JSON: {") is None
    assert mod.assignment_exact_match(parsed, exact) is True
    assert mod.assignment_exact_match({"A": "B"}, exact) is False
    assert mod.load_json(tmp_path / "missing.json") == {}
    assert mod.load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._source_key(REPO, Path("/tmp/exp6556-outside.json")) == "/tmp/exp6556-outside.json"

    specs = _model_specs(tmp_path)
    specs[0]["model_path"] = str(tmp_path / "missing.gguf")
    missing_receipts, missing_rows = mod.model_cache_and_load_receipts(
        backend=FakeBackend(),
        model_specs=mod.normalize_model_specs(specs),
        may_load=True,
    )
    assert missing_receipts["all_mandated_models_loaded"] is False
    assert missing_rows[0]["error"] == "model_path_missing"

    no_load_receipts, no_load_rows = mod.model_cache_and_load_receipts(
        backend=FakeBackend(),
        model_specs=mod.normalize_model_specs(_model_specs(tmp_path)),
        may_load=False,
    )
    assert no_load_receipts["all_mandated_models_loaded"] is False
    assert no_load_rows[0]["error"] == "not_loaded_before_failed_gate"

    bad_runtime = _runtime()
    bad_runtime["gpu"]["devices"] = []
    bad_runtime["llama_cpp"]["available"] = False
    monkeypatch.setattr(mod.os, "access", lambda *_args: False)
    blocked = mod.preconditions_checked(
        repo_root=REPO,
        result_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        model_specs=mod.normalize_model_specs(specs),
        runtime_state=bad_runtime,
        live_runtime_required=True,
        cached_pair=None,
        run_date="20260823",
    )
    assert "all_mandated_model_paths_resolved" in blocked["failed_preconditions"]
    assert "cached_sota_pair_gpu_0_1" in blocked["failed_preconditions"]
    assert "dual_rtx_3090_gpu_contract" in blocked["failed_preconditions"]
    assert "llama_cpp_cuda_contract" in blocked["failed_preconditions"]
    assert "checkpoint_writable" in blocked["failed_preconditions"]

    checkpoint = tmp_path / "checkpoint.json"
    assert mod.load_checkpoint(checkpoint, "h1")["rows_by_key"] == {}
    mod.atomic_write_json(
        checkpoint,
        {"schema": mod.CHECKPOINT_SCHEMA, "challenge_hash": "h0", "rows_by_key": {"a": {}}},
    )
    assert mod.load_checkpoint(checkpoint, "h1")["rows_by_key"] == {}
    mod.atomic_write_json(
        checkpoint,
        {
            "schema": mod.CHECKPOINT_SCHEMA,
            "challenge_hash": "h1",
            "rows_by_key": {"a": {"ok": True}},
        },
    )
    assert mod.load_checkpoint(checkpoint, "h1")["rows_by_key"]["a"]["ok"] is True

    tiny_checkpoint = tmp_path / "tiny-run-checkpoint.json"
    tiny_specs = mod.normalize_model_specs(_model_specs(tmp_path))[:1]
    first_rows, first_receipt = mod.run_per_unit_rows(
        backend=FakeBackend(),
        model_specs=tiny_specs,
        cells=selected[:1],
        checkpoint_path=tiny_checkpoint,
    )
    second_rows, second_receipt = mod.run_per_unit_rows(
        backend=FakeBackend(),
        model_specs=tiny_specs,
        cells=selected[:1],
        checkpoint_path=tiny_checkpoint,
    )
    assert len(first_rows) == len(mod.ARM_IDS)
    assert first_receipt["reused_row_count"] == 0
    assert second_receipt["reused_row_count"] == len(mod.ARM_IDS)
    assert all(row["checkpoint_reused"] for row in second_rows)

    assert mod.harmful_intervention_ledger([{"arm_id": "flat"}])["rows"] == []

    base_aggregate_args = {
        "gate": {"gate_passed": True},
        "preconditions": {"failed_preconditions": []},
        "load_receipts": {"all_mandated_models_loaded": True},
        "sample_contract": {"ready_floor_met": True},
        "rows": [],
        "route_rows": [],
        "harm": {"regression_count_vs_flat": 0, "regression_count_vs_longer_flat": 0},
        "phase_curve": {"phase_curve_established": False},
        "protected": {"all_unchanged": True},
    }
    disqualified = mod.aggregate_row_recomputation(
        **{
            **base_aggregate_args,
            "rows": [{"arm_id": "flat"}],
            "protected": {"all_unchanged": False},
        }
    )
    assert disqualified["verdict_class_from_rows"] == "disqualified"
    assert mod._status_and_verdict(disqualified)[2] == "disqualified"

    partial_rows = [
        {
            "arm_id": arm,
            "exact_final_validity": arm == "combined_bounded_route",
            "charged_cost": 100.0 if arm in {"flat", "longer_flat"} else 1.0,
        }
        for _index in range(108)
        for arm in mod.ARM_IDS
    ]
    partial = mod.aggregate_row_recomputation(
        **{
            **base_aggregate_args,
            "rows": partial_rows,
            "route_rows": [{"supported_route": True}] * len(partial_rows),
            "phase_curve": {"phase_curve_established": False},
        }
    )
    assert partial["verdict_class_from_rows"] == "partial"
    assert mod._status_and_verdict(partial)[2] == "partial"

    null_rows = [{**row, "exact_final_validity": True, "charged_cost": 1.0} for row in partial_rows]
    null = mod.aggregate_row_recomputation(
        **{
            **base_aggregate_args,
            "rows": null_rows,
            "route_rows": [{"supported_route": True}] * len(null_rows),
            "phase_curve": {"phase_curve_established": True},
        }
    )
    assert null["verdict_class_from_rows"] is None
    assert mod._status_and_verdict(null)[2] is None
