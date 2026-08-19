"""Tests for Exp6468 unique-event verifier-bounded CSL.

Spec refs: REQ-LEARN-6468, SCENARIO-LEARN-6468-SPEC,
SCENARIO-LEARN-6468-MODELS, SCENARIO-LEARN-6468-SEALED-SPLIT,
SCENARIO-LEARN-6468-UNIQUE-EVENTS, SCENARIO-LEARN-6468-EXACT-VETO,
SCENARIO-LEARN-6468-UPDATE-RULE, SCENARIO-LEARN-6468-AGGREGATES,
SCENARIO-LEARN-6468-ATTACKS, SCENARIO-LEARN-6468-READY.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6468_unique_event_verifier_bounded_csl as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / f"{mod.model_slug(model_id)}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6468 fixture GGUF bytes\n").encode("utf-8"))
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
    sealed_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture RTX 3090 GPUs"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture tokenizers"},
        {"resource": "llama_cpp_cuda_offload", "available": True, "detail": "fixture cuda"},
        {"resource": "new_raw_paths", "available": not (data_dir / "raw_outputs").exists(), "detail": "fresh"},
        {"resource": "result_path_fresh", "available": not result_path.exists(), "detail": "fresh"},
        {"resource": "empty_event_ids", "available": True, "detail": "empty fixture registry"},
        {"resource": "sealed_chronological_split", "available": sealed_manifest["sealed"] is True, "detail": "sealed"},
        {"resource": "exact_checker_authority", "available": True, "detail": "fixture checker"},
        {"resource": "monotonic_clock", "available": True, "detail": "fixture clock"},
        {"resource": "disk_space", "available": True, "detail": "fixture disk"},
    ]


def _host_blocked(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
    sealed_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _host_ok(
        result_path=result_path,
        data_dir=data_dir,
        model_specs=model_specs,
        sealed_manifest=sealed_manifest,
    )
    rows[0] = {"resource": "rtx_3090_gpu_count", "available": False, "detail": "only one GPU"}
    return rows


def _generator(event: dict[str, Any], prompt: str, spec: dict[str, Any]) -> dict[str, Any]:
    assert event["event_id"] in prompt
    assert spec["hf_id"] in prompt
    ordinal = int(event["event_sequence"])
    return {
        "completion_text": f"confidence {55 + ordinal % 40} event {event['event_id']}",
        "duration_s": 0.001,
        "runner_receipt": {
            "backend": "fixture_live_generation",
            "cpu_fallback": False,
            "model_hf_id": spec["hf_id"],
        },
    }


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6468-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        generation_func=_generator,
        duration_s=75.0,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=write,
    )


def test_req_learn_6468_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6468: OpenSpec owns the Exp6468 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6468") : text.index("REQ-LEARN-6444")]

    for marker in (
        "SCENARIO-LEARN-6468-SPEC",
        "SCENARIO-LEARN-6468-MODELS",
        "SCENARIO-LEARN-6468-SEALED-SPLIT",
        "SCENARIO-LEARN-6468-UNIQUE-EVENTS",
        "SCENARIO-LEARN-6468-EXACT-VETO",
        "SCENARIO-LEARN-6468-UPDATE-RULE",
        "SCENARIO-LEARN-6468-AGGREGATES",
        "SCENARIO-LEARN-6468-ATTACKS",
        "SCENARIO-LEARN-6468-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "A failed checker-authority receipt SHALL leave the persistent head unchanged",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        key = f"unique_event_csl_ready_score:{condition}"
        assert key in mod.FIELD_PRINCIPLES


def test_scenario_learn_6468_models_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6468-MODELS: model rows come from cached GGUF helpers."""

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


def test_scenario_learn_6468_sealed_split_and_unique_events(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6468-SEALED-SPLIT and UNIQUE-EVENTS: one row has one raw."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]
    events = artifact["event_rows"]["rows"]
    expected = len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL * len(mod.ARMS)

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["sealed_chronological_manifest"]["sealed"] is True
    assert artifact["sealed_chronological_manifest"]["interval_counts_by_model"] == {
        model_id: {"development": 6, "prospective_update": 10, "future_held": 8}
        for model_id in mod.MANDATED_MODEL_IDS
    }
    assert artifact["exposure_ledger"]["written_before_inference"] is True
    assert artifact["exposure_ledger"]["future_held_outcome_exposure_count"] == 0
    assert len(rows) == expected
    assert len(events) == expected
    assert artifact["event_identity_manifest"]["event_count"] == expected
    assert artifact["event_identity_manifest"]["empty_event_id_count"] == 0
    assert artifact["event_identity_manifest"]["duplicate_event_id_count"] == 0
    assert artifact["raw_output_manifest"]["raw_output_count"] == expected
    assert artifact["raw_output_manifest"]["validated_before_parse_count"] == expected
    assert artifact["one_event_one_raw_hash_check"]["passed"] is True
    assert artifact["one_event_one_raw_hash_check"]["duplicate_raw_hash_count"] == 0
    assert {row["event_id"] for row in rows} == {row["event_id"] for row in events}
    assert len({row["raw_output_sha256"] for row in rows}) == expected

    first_raw = Path(artifact["raw_output_manifest"]["rows"][0]["path"])
    assert first_raw.exists()
    assert mod.sha256_file(first_raw) == artifact["raw_output_manifest"]["rows"][0]["raw_output_sha256"]


def test_scenario_learn_6468_exact_veto_and_update_rule(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6468-EXACT-VETO and UPDATE-RULE: exact authority gates writes."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]
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
        and row["arm"] == mod.SELF_SIGNED_ARM
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
    assert verifier_zero["checker_result"]["checker_authority_passed"] is True
    assert verifier_zero["checker_result"]["exact_success"] is False
    assert verifier_zero["model_confidence"]["signed_direction"] == 1
    assert verifier_zero["exact_sign"] == -1
    assert verifier_zero["applied_update_sign"] == -1
    assert verifier_zero["write_decision"]["checker_ran_before_write"] is True
    assert verifier_zero["write_decision"]["admitted"] is True
    assert verifier_zero["post_state"]["weights"]["route_first"] < 0.0
    assert verifier_zero["selection_used_post_update_state"] is False

    assert verifier_one["selected_candidate"]["candidate_id"] == "candidate_1"
    assert verifier_one["checker_result"]["exact_success"] is True
    assert verifier_one["pre_state"]["head"] == verifier_zero["post_state"]["head"]
    assert teacher_zero["applied_update_sign"] == teacher_zero["model_confidence"]["signed_direction"]
    assert teacher_zero["applied_update_sign"] != teacher_zero["exact_sign"]
    assert frozen_two["pre_state"]["head"] == frozen_two["post_state"]["head"]

    veto = mod.admit_update(
        arm=mod.VERIFIER_BOUNDED_ARM,
        pre_head="head-before",
        post_head_if_written="head-after",
        checker_result={"checker_authority_passed": False, "exact_success": True},
        magnitude=0.2,
    )
    assert veto["admitted"] is False
    assert veto["post_head"] == "head-before"
    assert veto["rollback_pointer"] == "head-before"
    assert veto["veto_reason"] == "checker_authority_failed"


def test_scenario_learn_6468_aggregates_ready_and_validation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6468-AGGREGATES and READY: readiness comes from rows."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["effect_by_arm_and_interval"]["future_held"][mod.VERIFIER_BOUNDED_ARM]["exact_yield"] == 1.0
    assert (
        artifact["effect_by_arm_and_interval"]["future_held"][mod.VERIFIER_BOUNDED_ARM]["exact_yield"]
        > artifact["effect_by_arm_and_interval"]["future_held"][mod.FROZEN_ARM]["exact_yield"]
    )
    assert (
        artifact["effect_by_arm_and_interval"]["future_held"][mod.VERIFIER_BOUNDED_ARM]["exact_yield"]
        > artifact["effect_by_arm_and_interval"]["future_held"][mod.SELF_SIGNED_ARM]["exact_yield"]
    )
    assert artifact["protected_case_retention"]["regression_count"] == 0
    assert artifact["write_and_rollback_counts"]["exact_veto_failed_write_count"] == 0
    assert artifact["cpu_fallback_count"] == 0
    assert artifact["model_file_and_embedded_tokenizer_hashes"]["base_ggufs_frozen"] is True
    assert artifact["exact_veto_before_write_receipts"]["all_admitted_writes_checked_first"] is True
    assert artifact["unique_event_csl_ready_score"] == 1.0
    assert artifact["status"] == "success_ready"
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {"deterministic_exact_checker", "chronology_checks", "row_arithmetic"}
    assert oracle["false_for"]["self_signed_arm"] is False
    assert oracle["false_for"]["model_confidence"] is False
    assert artifact["raw_output_manifest"]["rows"][0]["parse_receipt"]["confidence"] >= 0.0
    assert artifact["event_rows"]["row_hash"] == mod.sha256_json(artifact["event_rows"]["rows"])
    assert artifact["per_unit_rows"]["row_hash"] == mod.sha256_json(rows)


def test_scenario_learn_6468_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6468-ATTACKS: unsafe event mutations do not validate."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["readiness_promoted_attack_count"] == 0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        (
            "one_event_one_raw_hash_check",
            lambda data: data["one_event_one_raw_hash_check"].__setitem__("duplicate_raw_hash_count", 1),
        ),
        (
            "event_identity",
            lambda data: data["event_identity_manifest"].__setitem__("duplicate_event_id_count", 1),
        ),
        (
            "exact_veto",
            lambda data: data["exact_veto_before_write_receipts"].__setitem__(
                "all_admitted_writes_checked_first",
                False,
            ),
        ),
        (
            "protected_retention",
            lambda data: data["protected_case_retention"].__setitem__("regression_count", 1),
        ),
        (
            "aggregate",
            lambda data: data["aggregate_row_recomputation"].__setitem__("matches_reported", False),
        ),
        ("attack_matrix", lambda data: data["attack_matrix"].__setitem__("all_critical_fail_closed", False)),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"checksum", "required_fields"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6468_blocked_preconditions_write_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6468: blocked preconditions still write a terminal artifact."""

    paths = _model_paths(tmp_path / "models")
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "blocked-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_blocked,
        generation_func=_generator,
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
    assert artifact["event_rows"]["row_count"] == 0
    assert artifact["unique_event_csl_ready_score"] == 0.0
    assert mod.validate_artifact(artifact) is True

    unresolved = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "unresolved.json",
        data_dir=tmp_path / "unresolved-data",
        cached_pair_func=lambda **_: [],
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        generation_func=_generator,
        duration_s=0.01,
        test_exit_codes={},
        write=False,
    )
    assert unresolved["status"] == "blocked_preconditions"
    assert "model_not_resolved" in unresolved["blocked_reason"]


def test_req_learn_6468_helper_edges_cover_unwritten_and_findings(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6468: helper edge paths stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing") is None
    with pytest.raises(ValueError, match="chronological_index_out_of_range"):
        mod._interval_for_index(99)
    snapshot_path = tmp_path / "snapshots" / "rev123" / "model-Q4_K_M.gguf"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"fixture")
    assert mod._revision_from_path(snapshot_path) == "rev123"
    assert mod._quantization_from_path(snapshot_path) == "Q4_K_M"

    class CloseableGenerator:
        def __init__(self) -> None:
            self.closed = False

        def __call__(self, event: dict[str, Any], prompt: str, spec: dict[str, Any]) -> dict[str, Any]:
            return _generator(event, prompt, spec)

        def close(self) -> None:
            self.closed = True

    closeable = CloseableGenerator()
    paths = _model_paths(tmp_path / "closeable-models")
    mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "closeable.json",
        data_dir=tmp_path / "closeable-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        generation_func=closeable,
        duration_s=75.0,
        test_exit_codes={},
        write=False,
    )
    assert closeable.closed is True

    artifact = _artifact(tmp_path / "write-false", write=False)
    assert artifact["raw_output_manifest"]["rows"][0]["present"] is False
    assert artifact["event_rows"]["rows"][0]["raw_output_path"].endswith(".json")

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["matches_reported"] = False
    bad["one_event_one_raw_hash_check"]["duplicate_raw_hash_count"] = 1
    bad["event_identity_manifest"]["duplicate_event_id_count"] = 1
    bad["exact_veto_before_write_receipts"]["all_admitted_writes_checked_first"] = False
    bad["attack_matrix"]["all_critical_fail_closed"] = False
    findings = mod._critical_findings(bad)
    assert {row["kind"] for row in findings} == {
        "aggregate_row_mismatch",
        "raw_output_reuse",
        "duplicate_event_id",
        "exact_veto_bypass",
        "attack_open",
    }
