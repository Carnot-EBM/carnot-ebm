"""Tests for Exp6455 verifier-bounded factor-weight CSL.

Spec refs: REQ-LEARN-6455, SCENARIO-LEARN-6455-SPEC,
SCENARIO-LEARN-6455-MODELS, SCENARIO-LEARN-6455-CHRONOLOGY,
SCENARIO-LEARN-6455-VERIFIER-SIGN, SCENARIO-LEARN-6455-ROWS,
SCENARIO-LEARN-6455-ATTACKS, SCENARIO-LEARN-6455-READY.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6455_prospective_verifier_bounded_factor_weight_csl as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6455 fixture GGUF bytes\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str) -> tuple[bool, str]:
    return True, f"embedded tokenizer fixture for {Path(path).name}"


def _host_ok(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture RTX 3090 GPUs"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture tokenizers"},
        {"resource": "exact_local_policy_checkers", "available": True, "detail": "fixture checkers"},
        {"resource": "monotonic_clock", "available": True, "detail": "fixture monotonic clock"},
        {"resource": "atomic_event_storage", "available": True, "detail": "fixture atomic storage"},
        {
            "resource": "fresh_paths",
            "available": not result_path.exists() and not (data_dir / "raw_outputs").exists(),
            "detail": "fresh fixture paths",
        },
        {"resource": "disk_space", "available": True, "detail": "fixture disk"},
    ]


def _host_blocked(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _host_ok(result_path=result_path, data_dir=data_dir, model_specs=model_specs)
    rows[0] = {"resource": "rtx_3090_gpu_count", "available": False, "detail": "only one GPU"}
    return rows


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6455-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        duration_s=12.0,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=write,
    )


def test_req_learn_6455_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6455: OpenSpec owns the Exp6455 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6455") : text.index("REQ-LEARN-6444")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-LEARN-6455-SPEC",
        "SCENARIO-LEARN-6455-MODELS",
        "SCENARIO-LEARN-6455-CHRONOLOGY",
        "SCENARIO-LEARN-6455-VERIFIER-SIGN",
        "SCENARIO-LEARN-6455-ROWS",
        "SCENARIO-LEARN-6455-ATTACKS",
        "SCENARIO-LEARN-6455-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "verifier-bounded arm SHALL derive update sign only from the exact checker result",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        key = f"verifier_bounded_csl_ready_score:{condition}"
        assert key in mod.FIELD_PRINCIPLES
        assert " ".join(condition.split("_")) or normalized


def test_scenario_learn_6455_models_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6455-MODELS: model rows come from cached GGUF helpers."""

    calls: list[dict[str, Any]] = []
    paths = _model_paths(tmp_path)
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["autotokenizer_usage_count"] == 0
    assert resolved["all_resolved"] is True
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])


def test_scenario_learn_6455_chronology_and_verifier_sign(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6455-CHRONOLOGY and VERIFIER-SIGN: exact sign controls writes."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]

    assert artifact["per_unit_rows"]["row_count"] == (
        len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL * len(mod.ARMS)
    )
    qwen = mod.MANDATED_MODEL_IDS[0]
    verifier_zero = next(
        row
        for row in rows
        if row["model"] == qwen
        and row["arm"] == mod.VERIFIER_BOUNDED_ARM
        and row["chronological_index"] == 0
    )
    verifier_one = next(
        row
        for row in rows
        if row["model"] == qwen
        and row["arm"] == mod.VERIFIER_BOUNDED_ARM
        and row["chronological_index"] == 1
    )
    teacher_zero = next(
        row
        for row in rows
        if row["model"] == qwen
        and row["arm"] == mod.SELF_TEACHER_ARM
        and row["chronological_index"] == 0
    )
    frozen_two = next(
        row
        for row in rows
        if row["model"] == qwen
        and row["arm"] == mod.FROZEN_ARM
        and row["chronological_index"] == 2
    )

    assert verifier_zero["selected_candidate"]["candidate_id"] == "candidate_0"
    assert verifier_zero["exact_result"]["exact_success"] is False
    assert verifier_zero["teacher_signal"]["signed_direction"] == 1
    assert verifier_zero["exact_sign"] == -1
    assert verifier_zero["applied_update_sign"] == -1
    assert verifier_zero["magnitude"] >= 0.0
    assert verifier_zero["post_update_weights"]["route_first"] < 0.0
    assert verifier_zero["selection_used_post_update_state"] is False

    assert verifier_one["selected_candidate"]["candidate_id"] == "candidate_1"
    assert verifier_one["exact_result"]["exact_success"] is True
    assert verifier_one["head_before"] == verifier_zero["head_after"]
    assert teacher_zero["applied_update_sign"] == teacher_zero["teacher_signal"]["signed_direction"]
    assert teacher_zero["applied_update_sign"] != teacher_zero["exact_sign"]
    assert frozen_two["head_before"] == frozen_two["head_after"]


def test_scenario_learn_6455_rows_recompute_and_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6455-ROWS and READY: readiness comes from row recomputation."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["chronology_and_future_only_checks"]["same_unit_update_use_count"] == 0
    assert artifact["chronology_and_future_only_checks"]["future_label_leakage_count"] == 0
    assert artifact["raw_output_uniqueness_and_reuse_count"]["reuse_count"] == 0
    assert artifact["raw_output_uniqueness_and_reuse_count"]["missing_raw_hash_count"] == 0
    assert artifact["future_exact_yield_delta"]["verifier_bounded_minus_frozen"] > 0.0
    assert artifact["future_exact_yield_delta"]["verifier_bounded_minus_teacher"] > 0.0
    assert artifact["protected_retention"]["regression_count"] == 0
    assert artifact["contamination_false_accepts_and_abstentions"]["false_accept_count"] == 0
    assert artifact["weight_growth_and_update_sparsity"]["bounded"] is True
    assert artifact["checker_calls_tokens_and_timing"]["checker_call_count"] == len(rows)
    assert artifact["effects_and_uncertainty_over_distinct_future_units"]["distinct_future_unit_count"] == (
        len(mod.MANDATED_MODEL_IDS) * (mod.UNITS_PER_MODEL - 1)
    )
    assert artifact["verifier_bounded_csl_ready_score"] == 1.0
    assert artifact["status"] == "success_ready"
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {"deterministic_exact_checker", "row_arithmetic"}
    assert oracle["false_for"]["self_teacher"] is False
    assert oracle["false_for"]["factor_energy_ranker"] is False


def test_scenario_learn_6455_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6455-ATTACKS: unsafe mutations do not validate."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["readiness_promoted_attack_count"] == 0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        ("ready_delta", lambda data: data["future_exact_yield_delta"].__setitem__("verifier_bounded_minus_frozen", 0.0)),
        ("raw_reuse", lambda data: data["raw_output_uniqueness_and_reuse_count"].__setitem__("reuse_count", 1)),
        ("chronology", lambda data: data["chronology_and_future_only_checks"].__setitem__("same_unit_update_use_count", 1)),
        ("attack_matrix", lambda data: data["attack_matrix"].__setitem__("all_critical_fail_closed", False)),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"checksum", "required_fields"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6455_blocked_preconditions_write_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6455: blocked preconditions still write a terminal artifact."""

    paths = _model_paths(tmp_path / "models")
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "blocked-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_blocked,
        duration_s=0.01,
        test_exit_codes={},
        write=True,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked_preconditions"
    assert artifact["blocked_reason"] == "rtx_3090_gpu_count"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["gate_check_summary"]["failed_check_count"] == 1
    assert artifact["per_unit_rows"]["row_count"] == 0
    assert artifact["verifier_bounded_csl_ready_score"] == 0.0
    assert mod.validate_artifact(artifact) is True

    unresolved = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "unresolved.json",
        data_dir=tmp_path / "unresolved-data",
        cached_pair_func=lambda **_: [],
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        duration_s=0.01,
        test_exit_codes={},
        write=False,
    )
    assert unresolved["status"] == "blocked_preconditions"
    assert "model_not_resolved" in unresolved["blocked_reason"]


def test_req_learn_6455_helper_edges_cover_unwritten_and_findings(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6455: helper edge paths stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing") is None
    snapshot_path = tmp_path / "snapshots" / "rev123" / "model-Q4_K_M.gguf"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"fixture")
    assert mod._revision_from_path(snapshot_path) == "rev123"
    assert mod._quantization_from_path(snapshot_path) == "Q4_K_M"

    artifact = _artifact(tmp_path / "write-false", write=False)
    assert artifact["event_store_and_initial_head_hashes"]["present"] is False
    assert artifact["device_and_runner_receipts"]["raw_pool_receipts"][0]["present"] is False

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["matches_reported"] = False
    bad["raw_output_uniqueness_and_reuse_count"]["reuse_count"] = 1
    bad["chronology_and_future_only_checks"]["same_unit_update_use_count"] = 1
    bad["attack_matrix"]["all_critical_fail_closed"] = False
    findings = mod._critical_findings(bad)
    assert {row["kind"] for row in findings} == {
        "aggregate_row_mismatch",
        "raw_output_reuse",
        "same_unit_update_use",
        "attack_open",
    }
