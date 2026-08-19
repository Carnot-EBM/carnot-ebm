"""Tests for Exp6463 SOTA fixed-policy candidate corpus v2.

Spec refs: REQ-INFRA-6463, SCENARIO-INFRA-6463-1,
SCENARIO-INFRA-6463-2, SCENARIO-INFRA-6463-3,
SCENARIO-INFRA-6463-4.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6463_sota_fixed_policy_candidate_corpus_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6463 fixture weights\n").encode("utf-8"))
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
    return True, f"fixture embedded tokenizer for {Path(path).name}"


def _host_ok(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture GPUs"},
        {"resource": "free_vram", "available": True, "detail": "fixture VRAM OK"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture OK"},
        {"resource": "llama_cpp_cuda_runner", "available": True, "detail": "fixture CUDA runner"},
        {"resource": "exact_simulator_imports", "available": True, "detail": "fixture exact checker"},
        {"resource": "disk_space", "available": True, "detail": "fixture OK"},
        {"resource": "monotonic_clock", "available": True, "detail": "fixture OK"},
        {
            "resource": "output_paths_fresh_or_resumable",
            "available": not result_path.exists(),
            "detail": "fixture paths OK",
            "path": str(data_dir),
        },
    ]


def _gate_ok(_path: Path) -> dict[str, Any]:
    return {
        "passed": True,
        "status": "success",
        "score": 1.0,
        "gate_check_summary": "fixture canary ready",
        "path": "fixture-exp6462.json",
    }


def _gate_blocked(_path: Path) -> dict[str, Any]:
    return {
        "passed": False,
        "status": "complete_with_findings",
        "score": 0.0,
        "gate_check_summary": "fixture canary not ready",
        "path": "fixture-exp6462.json",
    }


def _event_id(
    *,
    unit_id: str,
    model_hf_id: str,
    candidate_id: str,
    seed: int,
) -> str:
    return f"evt-{unit_id}-{mod.model_slug(model_hf_id)}-{candidate_id}-{seed}"


def _fixture_generation(
    *,
    model: dict[str, Any],
    problem: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    event_id: str,
    decoding_settings: dict[str, Any],
) -> dict[str, Any]:
    raw_text = mod.canonical_json(
        {
            "model_hf_id": model["hf_id"],
            "problem_id": problem["problem_id"],
            "candidate_seed": candidate["candidate_seed"],
            "actions": candidate["actions"],
        }
    )
    return {
        "raw_text": raw_text,
        "runtime_receipt": {
            "pid": 9000 + int(candidate["candidate_index"]),
            "parent_pid": 8000,
            "device_uuid": f"GPU-fixture-{model['gpu']}",
            "gpu_index": model["gpu"],
            "cuda_offload": True,
            "cpu_fallback": False,
            "completion_tokens": 64,
            "first_token_observed": True,
        },
        "timing": {
            "started_monotonic_ns": 1000 + int(candidate["candidate_seed"]),
            "ended_monotonic_ns": 2000 + int(candidate["candidate_seed"]),
            "duration_s": 1.0,
        },
        "prompt_seen_sha256": mod.sha256_text(prompt),
        "decoding_settings_seen": dict(decoding_settings),
        "event_id_seen": event_id,
    }


def _test_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date="20260819",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6463-data",
        canary_gate_func=_gate_ok,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=_fixture_generation,
        event_id_func=_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=125.0,
        write=write,
    )


def test_req_infra_6463_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6463: OpenSpec owns the corpus-v2 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6463") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6463-1",
        "SCENARIO-INFRA-6463-2",
        "SCENARIO-INFRA-6463-3",
        "SCENARIO-INFRA-6463-4",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "development, allocation-held, selection-held, and audit-held",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        assert f"sota_corpus_ready_score:{condition}" in mod.FIELD_PRINCIPLES


def test_scenario_infra_6463_pre_gate_blocks_before_generation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6463-1: Exp6462 canary failure blocks early."""

    calls = {"cache": 0, "host": 0, "generation": 0}

    def cache_not_allowed(**_kwargs: Any) -> list[dict[str, Any]]:
        calls["cache"] += 1
        raise AssertionError("model cache must not run after a failed pre-gate")

    def host_not_allowed(**_kwargs: Any) -> list[dict[str, Any]]:
        calls["host"] += 1
        raise AssertionError("host preflight must not run after a failed pre-gate")

    def generation_not_allowed(**_kwargs: Any) -> dict[str, Any]:
        calls["generation"] += 1
        raise AssertionError("generation must not run after a failed pre-gate")

    artifact = mod.run(
        date="20260819",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        canary_gate_func=_gate_blocked,
        cached_pair_func=cache_not_allowed,
        tokenizer_func=_tokenizer,
        host_preflight_func=host_not_allowed,
        generation_func=generation_not_allowed,
        event_id_func=_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=True,
    )

    assert artifact["status"] == "blocked_gate_check_failed"
    assert artifact["sota_corpus_ready_score"] == 0.0
    assert artifact["blocked_reason"] == "blocked_gate_check_failed"
    assert artifact["honest_verdict"] == "blocked_gate_check_failed"
    assert "fixture canary not ready" in artifact["gate_check_summary"]
    assert artifact["gate_check_summary"]
    assert artifact["preconditions_checked"][0]["resource"] == "exp6462_raw_persistence_canary"
    assert calls == {"cache": 0, "host": 0, "generation": 0}
    assert (tmp_path / "blocked.json").is_file()


def test_scenario_infra_6463_model_specs_and_four_way_manifest(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6463-1: models resolve locally and partitions seal 12 each."""

    calls: list[dict[str, Any]] = []
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(_model_paths(tmp_path / "models"), calls),
        tokenizer_func=_tokenizer,
    )
    problems = mod.build_policy_problems()
    manifest = mod.sealed_problem_and_partition_manifest(tmp_path / "data", problems, write=False)

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["all_resolved"] is True
    assert resolved["autotokenizer_usage_count"] == 0
    assert manifest["problem_count"] == mod.UNIT_COUNT == 48
    assert manifest["partition_counts"] == {
        "allocation_held": 12,
        "audit_held": 12,
        "development": 12,
        "selection_held": 12,
    }
    assert manifest["sealed_before_inference"] is True
    assert manifest["held_label_visible_before_generation_count"] == 0
    assert manifest["partition_exposed_to_prompt_count"] == 0
    assert len(set(manifest["problem_hashes"].values())) == mod.UNIT_COUNT
    assert manifest["label_manifest_sha256"].startswith("sha256:")
    assert manifest["partition_membership_sha256"].startswith("sha256:")


def test_scenario_infra_6463_gate_checkpoint_and_blocked_branches(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6463-1 and SCENARIO-INFRA-6463-2: edge gates are explicit."""

    gate_file = tmp_path / "exp6462.json"
    gate_file.write_text(
        json.dumps(
            {
                "status": "success",
                "raw_persistence_canary_ready_score": 1.0,
                "gate_check_summary": "file gate ready",
            }
        ),
        encoding="utf-8",
    )
    gate = mod.check_exp6462_gate(gate_file)
    assert gate == {
        "passed": True,
        "status": "success",
        "score": 1.0,
        "gate_check_summary": "file gate ready",
        "path": str(gate_file),
    }
    event_a = mod.default_event_id(
        unit_id="u",
        model_hf_id=mod.MANDATED_MODEL_IDS[0],
        candidate_id="c",
        seed=1,
    )
    event_b = mod.default_event_id(
        unit_id="u",
        model_hf_id=mod.MANDATED_MODEL_IDS[0],
        candidate_id="c",
        seed=1,
    )
    assert event_a.startswith("evt-") and event_b.startswith("evt-")
    assert event_a != event_b
    assert mod.inject_attack_rows([]) == []
    receipt = mod.tests_run_receipt({mod.DEFAULT_TEST_COMMANDS[0]: 1})
    assert receipt[0]["status"] == "failed"
    assert receipt[1]["status"] == "pending_external_run"
    assert mod._utc_now().endswith("Z")
    assert mod._gate_summary({"blocked_reason": "manual_gate"}) == "blocked: manual_gate"
    blocked_refresh: dict[str, Any] = {
        "per_unit_rows": {"rows": []},
        "blocked_reason": "manual_gate",
        "status": "blocked",
    }
    mod.refresh_terminal_fields(blocked_refresh)
    assert blocked_refresh["honest_verdict"] == "blocked_manual_gate"

    paths = _model_paths(tmp_path / "models")

    def incomplete_cached_pair(**_kwargs: Any) -> list[dict[str, Any]]:
        model_id = mod.MANDATED_MODEL_IDS[0]
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": 0,
                "model_path": str(paths[model_id]),
            }
        ]

    def generation_not_allowed(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("blocked model resolution must not generate")

    blocked = mod.run(
        date="20260819",
        result_path=tmp_path / "model-blocked.json",
        data_dir=tmp_path / "model-blocked-data",
        canary_gate_func=_gate_ok,
        cached_pair_func=incomplete_cached_pair,
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=generation_not_allowed,
        event_id_func=_event_id,
        test_exit_codes=None,
        duration_s=0.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert "model_resolution" in blocked["blocked_reason"]
    assert blocked["tests_run"][0]["status"] == "pending_external_run"
    assert (tmp_path / "model-blocked.json").is_file()


def test_scenario_infra_6463_event_rows_checkpoint_and_headroom(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6463-2 and SCENARIO-INFRA-6463-3: rows own labels and resume."""

    artifact = _artifact(tmp_path, write=True)
    normal_rows = [row for row in artifact["per_unit_rows"]["rows"] if row["row_kind"] == "normal"]
    written = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "success"
    assert artifact["sota_corpus_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["blocked_reason"] == ""
    assert len(normal_rows) == mod.UNIT_COUNT * len(mod.MANDATED_MODEL_IDS) * len(mod.CANDIDATES)
    assert artifact["checkpoint_and_resume_receipts"]["checkpoint_after_every_event"] is True
    assert artifact["checkpoint_and_resume_receipts"]["checkpoint_write_count"] == len(normal_rows)
    assert artifact["one_event_one_path_one_hash_check"]["passed"] is True
    assert artifact["event_identity_manifest"]["path_receipts_accepted"] == len(normal_rows)
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["cpu_fallback_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    for row in normal_rows:
        raw_path = Path(row["raw_output_path"])
        assert raw_path.is_file()
        assert raw_path.stat().st_size == row["durable_byte_count"]
        assert mod.sha256_file(raw_path) == row["raw_hash"]
        assert row["raw_persisted_before_parse"] is True
        assert row["parser_retry_count"] == 0
        assert row["parser_repair_applied"] is False
        assert row["path_receipt_validation"]["accepted"] is True
        assert row["exact_success"] is (
            row["legal"] is True and row["protected_ok"] is True and row["goal_ok"] is True
        )

    for partition in ("allocation_held", "selection_held", "audit_held"):
        headroom = artifact["candidate_headroom_by_partition"][partition]
        assert headroom["mixed_exact_outcomes"] is True
        assert headroom["has_headroom"] is True
        assert headroom["candidate_selection_cells_with_headroom"] > 0

    calls = {"generation": 0}

    def generation_not_allowed(**_kwargs: Any) -> dict[str, Any]:
        calls["generation"] += 1
        raise AssertionError("resume must not repeat completed events")

    resumed = mod.run(
        date="20260819",
        result_path=tmp_path / "resumed.json",
        data_dir=tmp_path / "exp6463-data",
        canary_gate_func=_gate_ok,
        cached_pair_func=_cached_pair(_model_paths(tmp_path / "resume-models"), []),
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=generation_not_allowed,
        event_id_func=_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=125.0,
        write=True,
    )
    assert resumed["status"] == "success"
    assert resumed["checkpoint_and_resume_receipts"]["resumed_event_count"] == len(normal_rows)
    assert resumed["checkpoint_and_resume_receipts"]["skipped_generation_count"] == len(normal_rows)
    assert calls["generation"] == 0


def test_scenario_infra_6463_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6463-4: attacks and schema validation close readiness."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]
    normal = [row for row in artifact["per_unit_rows"]["rows"] if row["row_kind"] == "normal"]
    attack_rows = [row for row in artifact["per_unit_rows"]["rows"] if row["row_kind"] == "attack"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert {row["attack_id"] for row in attack_rows} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    parse_bad = deepcopy(normal[0])
    parse_bad["parse_valid"] = False
    parsed_aggregates = mod.recompute_aggregates_from_rows([parse_bad])
    assert parsed_aggregates["parse_failures_by_model"][parse_bad["model_hf_id"]] == 1

    bad = deepcopy(artifact)
    normal = [row for row in bad["per_unit_rows"]["rows"] if row["row_kind"] == "normal"]
    normal[1]["event_id"] = normal[0]["event_id"]
    mod.refresh_terminal_fields(bad)
    assert bad["sota_corpus_ready_score"] == 0.0
    assert "duplicate_event_id" in bad["one_event_one_path_one_hash_check"]["reasons"]
    assert "one event/path/hash check failed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["rows"][0]["cpu_fallback"] = True
    mod.refresh_terminal_fields(bad)
    assert bad["sota_corpus_ready_score"] == 0.0
    assert "cpu_fallback_count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["rows"][0]["parser_repair_applied"] = True
    mod.refresh_terminal_fields(bad)
    assert bad["sota_corpus_ready_score"] == 0.0
    assert "parser repair is forbidden" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["rows"][0]["partition_exposed_to_prompt"] = True
    mod.refresh_terminal_fields(bad)
    assert bad["sota_corpus_ready_score"] == 0.0
    assert "held exposure is forbidden" in mod.validate_artifact(bad)

    validation_cases = [
        (lambda row: row.pop("status"), "missing required field: status"),
        (lambda row: row.update(MODEL_SPECS=[]), "MODEL_SPECS mandated ids mismatch"),
        (lambda row: row.update(models_used=["bad"]), "models_used must be empty or match mandated ids"),
        (lambda row: row.update(autotokenizer_usage_count=1), "autotokenizer_usage_count must be zero"),
        (lambda row: row.update(inference_substrate="wrong"), "inference_substrate mismatch"),
        (lambda row: row.update(verifier_is_oracle=False), "verifier_is_oracle must be true for exact simulator and row arithmetic"),
        (lambda row: row["per_unit_rows"].update(row_count=1), "per_unit_rows row_count mismatch"),
        (lambda row: row["per_unit_rows"].update(normal_row_count=1), "normal row count mismatch"),
        (lambda row: row["per_unit_rows"].update(attack_row_count=1), "attack row count mismatch"),
        (lambda row: row["sealed_problem_and_partition_manifest"].update(problem_count=1), "sealed problem count mismatch"),
        (
            lambda row: row["sealed_problem_and_partition_manifest"].update(partition_counts={}),
            "partition counts must be sealed 12/12/12/12",
        ),
        (lambda row: row["attack_matrix"].update(false_accept_count=1), "ready artifact cannot accept attacks"),
        (
            lambda row: row["attack_matrix"].update(all_critical_fail_closed=False),
            "attack matrix must fail closed",
        ),
        (
            lambda row: row["aggregate_row_recomputation"].update(matches_reported=False),
            "reported aggregates must recompute from rows",
        ),
        (lambda row: row["candidate_headroom_by_partition"]["audit_held"].update(has_headroom=False), "each held partition must have candidate headroom"),
        (
            lambda row: row["per_unit_rows"]["rows"][0].update(partition_membership_changed_after_seal=True),
            "membership reassignment is forbidden",
        ),
        (
            lambda row: row["per_unit_rows"]["rows"][0].update(legal=False),
            "exact labels must recompute",
        ),
        (
            lambda row: row["per_unit_rows"]["rows"][0].update(model_ranking_claim=True),
            "model ranking claim is forbidden",
        ),
        (
            lambda row: row["per_unit_rows"]["rows"][0]["path_receipt_validation"].update(accepted=False),
            "path receipts must validate",
        ),
        (lambda row: row.update(status="blocked_gate_check_failed", gate_check_summary=""), "blocked gate requires gate_check_summary"),
        (lambda row: row.update(field_principles={}), "missing field_principles entry: status"),
        (
            lambda row: row["field_principles"].pop("sota_corpus_ready_score:partitions_sealed"),
            "missing readiness field_principles entry: partitions_sealed",
        ),
        (lambda row: row.update(field_provenance={}), "field_provenance must cover exactly required fields"),
        (lambda row: row.update(honest_verdict="bad prefix"), "honest_verdict lacks required terminal prefix"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum mismatch"),
        (lambda row: row.update(sota_corpus_ready_score=1.0, duration_s=0.0), "sota_corpus_ready_score does not recompute"),
    ]
    for mutate, expected in validation_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {
            "reproducibility_checksum mismatch",
            "sota_corpus_ready_score does not recompute",
        }:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_artifact(bad)


def test_scenario_infra_6463_run_schema_failure_branch(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-INFRA-6463-4: schema errors force a non-ready artifact."""

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced schema"])
    artifact = _artifact(tmp_path)
    assert artifact["status"] == "failed_schema"
    assert artifact["sota_corpus_ready_score"] == 0.0
    assert artifact["current_adversarial_findings"] == [
        {"severity": "critical", "kind": "schema_validation", "detail": "forced schema"}
    ]
    assert artifact["honest_verdict"].startswith("complete_failed_schema:")
