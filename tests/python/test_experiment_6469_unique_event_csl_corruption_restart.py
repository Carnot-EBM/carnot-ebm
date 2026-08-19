"""Tests for Exp6469 unique-event CSL corruption restart.

Spec refs: REQ-LEARN-6469, SCENARIO-LEARN-6469-GATE,
SCENARIO-LEARN-6469-MANIFEST, SCENARIO-LEARN-6469-RESTART,
SCENARIO-LEARN-6469-CORRUPTION, SCENARIO-LEARN-6469-ROLLBACK,
SCENARIO-LEARN-6469-NON-RESURRECTION, SCENARIO-LEARN-6469-ATTACKS,
SCENARIO-LEARN-6469-READY.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6469_unique_event_csl_corruption_restart as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / f"{mod.model_slug(model_id)}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6469 fixture GGUF bytes\n").encode("utf-8"))
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


def _host_ok(**kwargs: Any) -> list[dict[str, Any]]:
    model_specs = kwargs["model_specs"]
    result_path = kwargs["result_path"]
    data_dir = kwargs["data_dir"]
    manifest = kwargs["sealed_manifest"]
    return [
        {"resource": "exp6468_unique_event_csl_ready_score", "available": True, "detail": "1.0"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture tokenizers"},
        {"resource": "new_result_path", "available": not result_path.exists(), "detail": "fresh"},
        {"resource": "new_data_dir", "available": not data_dir.exists(), "detail": "fresh"},
        {"resource": "sealed_new_held_manifest", "available": manifest["sealed"] is True, "detail": "sealed"},
        {"resource": "exact_checker_authority", "available": True, "detail": "fixture checker"},
    ]


def _generator(event: dict[str, Any], prompt: str, spec: dict[str, Any]) -> dict[str, Any]:
    assert event["event_id"] in prompt
    assert spec["hf_id"] in prompt
    return {
        "completion_text": f"confidence {61 + int(event['event_sequence']) % 19} {event['event_id']}",
        "duration_s": 0.001,
        "runner_receipt": {
            "backend": "fixture_unique_generation",
            "cpu_fallback": False,
            "model_hf_id": spec["hf_id"],
        },
    }


class _ClosableGenerator:
    def __init__(self) -> None:
        self.closed = False

    def __call__(self, event: dict[str, Any], prompt: str, spec: dict[str, Any]) -> dict[str, Any]:
        return _generator(event, prompt, spec)

    def close(self) -> None:
        self.closed = True


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    generator = _ClosableGenerator()
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6469-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        generation_func=generator,
        duration_s=12.5,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=write,
    )
    assert generator.closed is True
    return artifact


def test_req_learn_6469_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6469: OpenSpec owns the corruption restart contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6469") : text.index("REQ-LEARN-6444")]

    for marker in (
        "SCENARIO-LEARN-6469-GATE",
        "SCENARIO-LEARN-6469-MANIFEST",
        "SCENARIO-LEARN-6469-RESTART",
        "SCENARIO-LEARN-6469-CORRUPTION",
        "SCENARIO-LEARN-6469-ROLLBACK",
        "SCENARIO-LEARN-6469-NON-RESURRECTION",
        "SCENARIO-LEARN-6469-ATTACKS",
        "SCENARIO-LEARN-6469-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "forged pass",
        "interrupted write",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        assert f"corruption_restart_ready_score:{condition}" in mod.FIELD_PRINCIPLES


def test_scenario_learn_6469_gate_blocks_generation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6469-GATE: Exp6468 readiness is checked before generation."""

    paths = _model_paths(tmp_path / "models")
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        upstream_loader=lambda _: {"unique_event_csl_ready_score": 0.0},
        generation_func=_generator,
        duration_s=0.2,
        write=True,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked_upstream_gate"
    assert artifact["corruption_restart_ready_score"] == 0.0
    assert artifact["raw_output_manifest"]["raw_output_count"] == 0
    assert artifact["blocked_reason"] == "upstream_unique_event_csl_ready_score_not_1"
    assert artifact["gate_check_summary"]["failed_check_count"] == 1
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_6469_models_and_new_manifest_are_disjoint(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6469-MANIFEST: new held units and raw hashes are disjoint."""

    calls: list[dict[str, Any]] = []
    paths = _model_paths(tmp_path / "models")
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    artifact = _artifact(tmp_path)

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["model_file_and_embedded_tokenizer_hashes"]["base_ggufs_frozen"] is True
    assert artifact["sealed_new_held_manifest"]["sealed"] is True
    assert artifact["sealed_new_held_manifest"]["held_unit_count"] == (
        len(mod.MANDATED_MODEL_IDS) * mod.HELD_UNITS_PER_MODEL
    )
    assert artifact["exposure_disjointness_receipts"]["all_disjoint"] is True
    assert artifact["exposure_disjointness_receipts"]["unit_id_overlap_with_exp6468_count"] == 0
    assert artifact["exposure_disjointness_receipts"]["raw_hash_overlap_with_exp6468_count"] == 0
    assert artifact["raw_output_manifest"]["raw_output_count"] == artifact["event_identity_manifest"]["event_count"]
    assert artifact["raw_output_manifest"]["unique_raw_hash_count"] == artifact["raw_output_manifest"]["raw_output_count"]
    assert artifact["event_identity_manifest"]["duplicate_event_id_count"] == 0


def test_scenario_learn_6469_restart_corruption_rollback_and_non_resurrection(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6469-RESTART/ROLLBACK: corrupt heads cannot survive restart."""

    artifact = _artifact(tmp_path)
    restarts = artifact["process_restart_receipts"]
    quarantine = artifact["quarantine_tombstone_and_rollback_receipts"]
    non_resurrection = artifact["non_resurrection_check"]
    corrupt_rows = [
        row for row in artifact["per_unit_rows"]["rows"] if row["corruption"]["scheduled"] is True
    ]

    assert restarts["restart_count"] == 2
    assert restarts["all_recovered_heads_match"] is True
    assert all(row["child_pid"] != row["parent_pid"] for row in restarts["rows"])
    assert all(row["loaded_only_committed_head_and_receipt_chain"] is True for row in restarts["rows"])
    assert artifact["corruption_precommitment"]["corrupt_event_count"] == (
        len(mod.CORRUPTION_BOUNDARIES) * len(mod.MANDATED_MODEL_IDS)
    )
    assert len(corrupt_rows) == artifact["corruption_precommitment"]["corrupt_event_count"]
    assert artifact["exact_veto_before_write_receipts"]["all_admitted_writes_checked_first"] is True
    assert artifact["exact_veto_before_write_receipts"]["corrupt_release_count"] == 0
    assert quarantine["quarantine_count"] == len(corrupt_rows)
    assert quarantine["tombstone_count"] == len(corrupt_rows)
    assert quarantine["rollback_success_count"] == len(corrupt_rows)
    assert quarantine["all_tombstones_precede_rollback"] is True
    assert non_resurrection["corrupt_state_resurrection_count"] == 0
    assert non_resurrection["post_restart_active_head_clean"] is True

    lifecycle_by_event: dict[str, list[str]] = {}
    for row in artifact["lifecycle_rows"]["rows"]:
        lifecycle_by_event.setdefault(row["event_id"], []).append(row["transition"])
    for row in corrupt_rows:
        transitions = lifecycle_by_event[row["event_id"]]
        assert transitions.index("quarantine") < transitions.index("tombstone")
        assert transitions.index("tombstone") < transitions.index("rollback")
        assert row["write_decision"]["admitted"] is False
        assert row["post_state"]["head"] == row["rollback"]["restored_head"]


def test_scenario_learn_6469_ready_effects_and_validation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6469-READY: readiness comes from clean rows and containment."""

    artifact = _artifact(tmp_path)
    effects = artifact["clean_and_corrupt_effects"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert effects["clean_exact_yield"] > effects["frozen_exact_yield"]
    assert effects["clean_minus_frozen"] > 0.0
    assert effects["governed_non_corrupt_exact_yield"] > effects["frozen_exact_yield"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["inference_substrate"].startswith("live_llm_inference_local_gguf")
    assert artifact["protected_case_retention"]["regression_count"] == 0
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["current_adversarial_findings"] == []
    assert artifact["corruption_restart_ready_score"] == 1.0
    assert artifact["status"] == "success_ready"
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {
        "deterministic_exact_checker",
        "hash_chain",
        "lifecycle_checks",
        "row_arithmetic",
    }
    assert oracle["false_for"]["model_raw_text"] is False
    assert artifact["per_unit_rows"]["row_hash"] == mod.sha256_json(artifact["per_unit_rows"]["rows"])
    assert artifact["lifecycle_rows"]["row_hash"] == mod.sha256_json(artifact["lifecycle_rows"]["rows"])


def test_scenario_learn_6469_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6469-ATTACKS: attack and schema mutations fail closed."""

    artifact = _artifact(tmp_path)

    assert {row["attack_id"] for row in artifact["attack_matrix"]["rows"]} == set(mod.ATTACK_IDS)
    assert artifact["attack_matrix"]["all_critical_attacks_fail_closed"] is True
    assert artifact["attack_matrix"]["readiness_promoted_attack_count"] == 0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        (
            "aggregate",
            lambda data: data["aggregate_row_recomputation"].__setitem__("matches_reported", False),
        ),
        (
            "non_resurrection",
            lambda data: data["non_resurrection_check"].__setitem__("corrupt_state_resurrection_count", 1),
        ),
        (
            "quarantine",
            lambda data: data["quarantine_tombstone_and_rollback_receipts"].__setitem__(
                "rollback_success_count",
                0,
            ),
        ),
        (
            "exact_veto",
            lambda data: data["exact_veto_before_write_receipts"].__setitem__("corrupt_release_count", 1),
        ),
        ("attack_matrix", lambda data: data["attack_matrix"].__setitem__("all_critical_attacks_fail_closed", False)),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"checksum", "required_fields"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6469_helper_edges_cover_blockers_and_findings(tmp_path: Path) -> None:
    """REQ-LEARN-6469: helper edge paths stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing") is None
    assert mod.source_hashes(tmp_path)
    assert mod.default_upstream_loader(tmp_path / "missing.json") == {}
    assert mod._extract_exp6468_identity_sets({}) == {
        "unit_ids": set(),
        "event_ids": set(),
        "raw_hashes": set(),
    }
    with pytest.raises(ValueError, match="required_fields"):
        mod.validate_artifact({"status": "bad"})

    artifact = _artifact(tmp_path / "edge", write=False)
    assert artifact["raw_output_manifest"]["rows"][0]["present"] is False
    assert artifact["gate_check_summary"]["failed_checks"] == []

    generated = mod.SyntheticEventGenerator()(
        {"event_id": "edge-event", "event_sequence": 3},
        "prompt",
        {"hf_id": mod.MANDATED_MODEL_IDS[0], "model_path": "fixture.gguf"},
    )
    assert generated["runner_receipt"]["backend"] == "deterministic_new_raw_event_generator"

    paths = _model_paths(tmp_path / "blocked-models")
    model_resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
    )
    manifest = mod.sealed_new_held_manifest(
        mod.build_new_held_units(model_resolution["MODEL_SPECS"]),
        date=mod.RUN_DATE,
    )
    defaults = mod.default_preconditions(
        result_path=tmp_path / "preconditions.json",
        data_dir=tmp_path / "precondition-data",
        model_specs=model_resolution["MODEL_SPECS"],
        sealed_manifest=manifest,
        upstream_gate={"gate_passed": True, "unique_event_csl_ready_score": 1.0},
    )
    assert {row["resource"] for row in defaults} >= {
        "mandatory_model_files",
        "sealed_new_held_manifest",
    }

    blocked = mod.run(
        date="20260818",
        result_path=tmp_path / "wrong-date.json",
        data_dir=tmp_path / "wrong-date-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        generation_func=_generator,
        duration_s=0.2,
        write=True,
    )
    assert blocked["status"] == "blocked_preconditions"
    assert "unexpected_date:20260818" in blocked["blocked_reason"]

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["matches_reported"] = False
    bad["exact_veto_before_write_receipts"]["corrupt_release_count"] = 1
    bad["non_resurrection_check"]["corrupt_state_resurrection_count"] = 1
    bad["attack_matrix"]["all_critical_attacks_fail_closed"] = False
    findings = mod.current_adversarial_findings(bad)
    assert {row["kind"] for row in findings} == {
        "aggregate_mismatch",
        "exact_veto_bypass",
        "resurrection",
        "attack_open",
    }
