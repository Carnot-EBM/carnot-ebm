"""Focused tests for the immutable Gemma constraint-first stream."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6591_gemma4_31b_constraint_first_stream as mod


def _method() -> dict:
    return mod.load_json(mod.REPO_ROOT / mod.METHOD_RELATIVE_PATH)


def _launch() -> dict:
    return mod.load_json(mod.REPO_ROOT / mod.LAUNCH_RELATIVE_PATH)


def _identity() -> dict:
    row = next(
        item
        for item in _launch()["model_cache_identity_rows"]
        if item["repository_id"] == mod.GEMMA_REPOSITORY_ID
    )
    return {
        "model_specs": deepcopy(mod.MODEL_SPECS),
        "identity": deepcopy(row),
        "cached_sota_pair": [
            {
                "name": "Gemma4-31B-it",
                "hf_id": mod.GEMMA_REPOSITORY_ID,
                "gpu": 0,
                "model_path": row["cache_path"],
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": "/cached/gemma.gguf",
            },
        ],
        "llama_cpp_build": {
            "exists": True,
            "executable": True,
            "cuda_linked": True,
            "version_receipt": {"exit_code": 0, "version_text": "b9606"},
        },
        "embedded_tokenizer_used": True,
        "auto_tokenizer_used": False,
        "download_performed": False,
    }


def _process() -> dict:
    model_path = _identity()["identity"]["cache_path"]
    before_device = {
        "index": 0,
        "uuid": "GPU-0",
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mb": 4,
        "memory_free_mb": 24120,
        "utilization_pct": 0,
    }
    during_device = {**before_device, "memory_used_mb": 21000, "utilization_pct": 70}
    command = ["llama-server", "--model", model_path, "--n-gpu-layers", "all"]
    return {
        "process": {
            "pid": 777,
            "parent_pid": 42,
            "fresh_process": True,
            "owned_child": True,
            "command": command,
            "command_sha256": mod.sha256_json(command),
            "selected_blob_path": model_path,
            "selected_gpu": 0,
            "cuda_visible_devices": "0",
            "offloaded_layers": 65,
            "server_healthy": True,
            "http_status": 200,
            "gpu_samples": [
                {"stage": "before", "device": before_device, "compute_processes": []},
                {
                    "stage": "during",
                    "device": during_device,
                    "compute_processes": [{"pid": 777, "used_memory_mb": 20996}],
                },
                {
                    "stage": "during",
                    "device": during_device,
                    "compute_processes": [{"pid": 777, "used_memory_mb": 20996}],
                },
                {"stage": "after", "device": before_device, "compute_processes": []},
            ],
            "shutdown_requested": True,
            "normal_shutdown": True,
            "exit_code": 0,
            "worker_alive_after_exit": False,
            "resident_model_families": [mod.GEMMA_REPOSITORY_ID],
            "signals_sent_to_unrelated_pids": [],
            "stdout_sha256": mod.sha256_bytes(b""),
            "stderr_sha256": mod.sha256_bytes(b"offloaded 65/65 layers"),
            "evidence_mode": "measured",
        },
        "unload": {
            "worker_pid": 777,
            "worker_absent_from_proc": True,
            "worker_absent_from_nvidia_smi": True,
            "port_closed": True,
            "baseline_memory_used_mb": 4,
            "recovered_memory_used_mb": 4,
            "memory_delta_from_baseline_mb": 0,
            "recovery_tolerance_mb": mod.RECOVERY_TOLERANCE_MB,
            "no_task_worker_remains": True,
            "recovery_bounded": True,
            "recovery_complete": True,
            "signals_sent_to_unrelated_pids": [],
        },
    }


def _protected() -> dict:
    return {
        "all_unchanged": True,
        "rows": [
            {
                "path": path.as_posix(),
                "before_sha256": "sha256:same",
                "after_sha256": "sha256:same",
                "unchanged": True,
            }
            for path in mod.PROTECTED_RELATIVE_PATHS
        ],
    }


def _preconditions() -> dict:
    return {
        "all_required_preconditions_available": True,
        "checks": {
            "upstream_gate": True,
            "frozen_method": True,
            "model_identity": True,
            "llama_cpp_cuda": True,
            "idle_rtx_3090": True,
            "owned_process_absent": True,
            "atomic_output": True,
        },
        "failed_preconditions": [],
        "model_process_started": True,
        "repo_wide_suite_is_launch_gate": False,
        "selected_gpu": 0,
        "seed_schedule": _method()["arm_seed_budget_contract"]["seed_schedule"],
        "budgets": _launch()["execution_budget_contract"][0],
    }


def _stage(stage: str, raw: bytes, *, tokens: int = 7, failure: dict | None = None) -> dict:
    return mod.make_stage_receipt(
        stage=stage,
        prompt_sha256=f"sha256:{stage}",
        request_sha256=f"sha256:request-{stage}",
        raw_bytes=raw,
        prompt_tokens=11,
        completion_tokens=tokens,
        latency_s=0.25,
        stop_reason="stop",
        request_status=200,
        recorded_monotonic_ns=10,
        failure_flags=failure,
    )


def _unit_row(unit: dict, seed: int) -> dict:
    registry_hash = _method()["source_binding_and_exact_authority_contract"][
        "exact_obligation_registry"
    ]["registry_sha256"]
    expected_release = unit["expected_action"] == "release"
    final = b"Supported final response." if expected_release else b"I abstain."
    stage1_text = "\n".join(
        f'Constraint: "{row["quoted_span"]}"' for row in unit["gold_constraints"]
    )
    direct = mod.build_arm_row(
        unit=unit,
        arm_name="direct",
        route="direct",
        seed=seed,
        stage_receipts={"direct": _stage("direct", final)},
        registry_hash=registry_hash,
    )
    always = mod.build_arm_row(
        unit=unit,
        arm_name="always_on_cfr",
        route="cfr",
        seed=seed,
        stage_receipts={
            "stage1": _stage("stage1", stage1_text.encode()),
            "stage2": _stage("stage2", final),
        },
        registry_hash=registry_hash,
    )
    route = "cfr" if unit["stratum"] == "restrictive_cue" else "direct"
    routed_stages = (
        {"stage1": _stage("stage1", stage1_text.encode()), "stage2": _stage("stage2", final)}
        if route == "cfr"
        else {"direct": _stage("direct", final)}
    )
    routed = mod.build_arm_row(
        unit=unit,
        arm_name="routed_cfr",
        route=route,
        seed=seed,
        stage_receipts=routed_stages,
        registry_hash=registry_hash,
    )
    return mod.finalize_unit_row(unit, [direct, always, routed])


def _ready_report(tmp_path: Path) -> dict:
    method = _method()
    gate = mod.build_gate_receipt(mod.REPO_ROOT)
    frozen = mod.build_frozen_hash_receipt(mod.REPO_ROOT, method)
    rows = []
    checkpoints = []
    seeds = method["arm_seed_budget_contract"]["seed_schedule"]
    for unit, seed_row in zip(method["source_unit_manifest"]["units"], seeds, strict=True):
        rows.append(_unit_row(unit, seed_row["seed"]))
        checkpoints.append(mod.write_unit_checkpoint(tmp_path / "checkpoints", rows))
    return mod.build_report(
        run_date="20260825",
        gate_receipt=gate,
        frozen_receipt=frozen,
        model_identity=_identity(),
        per_unit_rows=rows,
        checkpoint_receipts=checkpoints,
        gpu_receipts=_process(),
        preconditions=_preconditions(),
        protected=_protected(),
        duration_s=61.0,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
    )


def test_spec_declares_exp6591_requirements_and_scenarios() -> None:
    """REQ-REPORT-6591: the stream has explicit spec and scenario anchors."""

    text = (mod.REPO_ROOT / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6591-PRECONDITIONS",
        "REQ-REPORT-6591-FROZEN",
        "REQ-REPORT-6591-MODEL",
        "REQ-REPORT-6591-ROWS",
        "REQ-REPORT-6591-RAW",
        "REQ-REPORT-6591-AUTHORITY",
        "REQ-REPORT-6591-FAILURES",
        "REQ-REPORT-6591-CHECKPOINTS",
        "REQ-REPORT-6591-LIFECYCLE",
        "REQ-REPORT-6591-ATTACKS",
        "REQ-REPORT-6591-REDUCER",
        "REQ-REPORT-6591-ATOMIC",
        "SCENARIO-REPORT-6591-FROZEN",
        "SCENARIO-REPORT-6591-RAW",
        "SCENARIO-REPORT-6591-AUTHORITY",
        "SCENARIO-REPORT-6591-CHECKPOINTS",
        "SCENARIO-REPORT-6591-LIFECYCLE",
        "SCENARIO-REPORT-6591-ATTACKS",
        "SCENARIO-REPORT-6591-ATOMIC",
    ):
        assert anchor in text


def test_dense_family_scope_restores_qwen_contract_state() -> None:
    """REQ-REPORT-6591-MODEL: family reuse cannot alter the Qwen producer."""

    original_repository = mod.shared.QWEN_REPOSITORY_ID
    original_checkpoint_writer = mod.shared.write_unit_checkpoint
    with mod._family_configuration():
        assert mod.shared.QWEN_REPOSITORY_ID == mod.GEMMA_REPOSITORY_ID
        assert mod.shared.write_unit_checkpoint is mod.write_unit_checkpoint
    assert mod.shared.QWEN_REPOSITORY_ID == original_repository
    assert mod.shared.write_unit_checkpoint is original_checkpoint_writer


def test_gate_and_every_frozen_hash_recompute_before_inference() -> None:
    """SCENARIO-REPORT-6591-FROZEN: exact method drift blocks launch."""

    gate = mod.build_gate_receipt(mod.REPO_ROOT)
    assert gate["field"] == "v574_cfr_launch_ready_score"
    assert gate["observed_value"] == gate["expected_value"] == 1.0
    assert gate["passed"] is True
    frozen = mod.build_frozen_hash_receipt(mod.REPO_ROOT, _method())
    assert frozen["all_frozen_hashes_match"] is True
    assert all(frozen["checks"].values())
    assert frozen["method_artifact_sha256"] == mod.sha256_file(
        mod.REPO_ROOT / mod.METHOD_RELATIVE_PATH
    )

    drift = _method()
    drift["prompt_stage_contract"]["prompts"]["stage1"]["text"] += " drift"
    failed = mod.build_frozen_hash_receipt(mod.REPO_ROOT, drift)
    assert failed["checks"]["prompt_hashes"] is False
    assert failed["all_frozen_hashes_match"] is False


def test_plain_text_parser_binds_quotes_and_exact_authority() -> None:
    """SCENARIO-REPORT-6591-AUTHORITY: source bytes and exact checks own release."""

    unit = _method()["source_unit_manifest"]["units"][0]
    gold = unit["gold_constraints"][0]
    raw = f'Constraint: "{gold["quoted_span"]}". Constraint: "invented span".'.encode()
    proposals = mod.parse_stage1_proposals(raw, unit)
    assert proposals[0]["quoted_span"] == gold["quoted_span"]
    assert proposals[0]["parser_used_gold_semantics"] is True
    assert proposals[1]["unsupported"] is True

    arm = mod.build_arm_row(
        unit=unit,
        arm_name="always_on_cfr",
        route="cfr",
        seed=1,
        stage_receipts={"stage1": _stage("stage1", raw), "stage2": _stage("stage2", b"ok")},
        registry_hash="sha256:registry",
    )
    assert arm["exact_results"]["checker"] == mod.EXACT_CHECKER_NAME
    assert arm["exact_results"]["model_is_release_authority"] is False
    assert arm["source_span_bindings"][0]["source_supported"] is True
    assert arm["source_span_bindings"][1]["unsupported"] is True
    assert arm["failure"]["unsupported_constraint"] is True
    assert arm["unsafe_release"] is False


def test_raw_stages_stay_separate_and_all_cfr_work_is_charged() -> None:
    """SCENARIO-REPORT-6591-RAW: Stage 1 cannot overwrite Stage 2 or avoid cost."""

    unit = _method()["source_unit_manifest"]["units"][0]
    row = _unit_row(unit, 65870001)
    assert [arm["arm_name"] for arm in row["arms"]] == list(mod.ARM_ORDER)
    always = row["arms"][1]
    assert always["raw_stages"]["direct"] is None
    assert always["raw_stages"]["stage1"]["row_hash"] != always["raw_stages"]["stage2"]["row_hash"]
    assert (
        always["raw_stages"]["stage1"]["prompt_sha256"]
        != always["raw_stages"]["stage2"]["prompt_sha256"]
    )
    assert always["stage1_passed_verbatim_to_stage2"] is True
    assert always["tokens"]["stage1_charged"] is True
    assert always["tokens"]["total"] == always["tokens"]["stage1"] + always["tokens"]["stage2"]
    assert always["charged_cost"] == mod.cost_from_stages(always["raw_stages"])
    flattened = mod.build_raw_stage_receipts([row])
    assert {item["stage"] for item in flattened} == {"direct", "stage1", "stage2"}
    assert all(
        base64.b64decode(item["raw_bytes_b64"], validate=True) is not None for item in flattened
    )


def test_identical_stage_text_keeps_separate_authentic_receipts() -> None:
    """SCENARIO-REPORT-6591-RAW: equal model text is not evidence overwrite."""

    unit = _method()["source_unit_manifest"]["units"][0]
    raw = b"No supported constraint."
    arm = mod.build_arm_row(
        unit=unit,
        arm_name="always_on_cfr",
        route="cfr",
        seed=65870001,
        stage_receipts={"stage1": _stage("stage1", raw), "stage2": _stage("stage2", raw)},
        registry_hash=_method()["source_binding_and_exact_authority_contract"][
            "exact_obligation_registry"
        ]["registry_sha256"],
    )
    assert arm["raw_stages"]["stage1"]["raw_sha256"] == arm["raw_stages"]["stage2"]["raw_sha256"]
    assert arm["raw_stages"]["stage1"]["row_hash"] != arm["raw_stages"]["stage2"]["row_hash"]
    assert mod._arm_authentic(arm, "always_on_cfr") is True


def test_failures_remain_rows_and_answer_leakage_fails_closed() -> None:
    """REQ-REPORT-6591-FAILURES: terminal failures and leakage stay visible."""

    unit = _method()["source_unit_manifest"]["units"][0]
    failure = mod.empty_failure_flags()
    failure["timeout"] = True
    stage1 = _stage("stage1", b"The final answer is release.", failure=failure)
    arm = mod.build_arm_row(
        unit=unit,
        arm_name="always_on_cfr",
        route="cfr",
        seed=1,
        stage_receipts={"stage1": stage1, "stage2": _stage("stage2", b"")},
        registry_hash="sha256:registry",
    )
    assert arm["failure"]["timeout"] is True
    assert arm["failure"]["stage1_answer_leakage"] is True
    assert arm["failure"]["any"] is True
    unit_row = mod.finalize_unit_row(unit, [arm])
    failures = mod.build_failure_rows([unit_row])
    assert failures[0]["unit_id"] == unit["unit_id"]
    assert "timeout" in failures[0]["failure_classes"]
    assert "stage1_answer_leakage" in failures[0]["failure_classes"]


def test_checkpoints_are_atomic_monotonic_prefixes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6591-CHECKPOINTS: a completed prefix survives later failure."""

    method = _method()
    first = _unit_row(method["source_unit_manifest"]["units"][0], 1)
    second = _unit_row(method["source_unit_manifest"]["units"][1], 2)
    receipt1 = mod.write_unit_checkpoint(tmp_path, [first])
    receipt2 = mod.write_unit_checkpoint(tmp_path, [first, second])
    assert receipt1["completed_unit_count"] == 1
    assert receipt2["completed_unit_count"] == 2
    assert receipt2["completed_unit_ids"] == [first["unit_id"], second["unit_id"]]
    assert receipt1["atomic_replace"] is receipt2["atomic_replace"] is True
    assert mod.sha256_file(receipt2["absolute_path"]) == receipt2["checkpoint_sha256"]
    stored = json.loads(Path(receipt2["absolute_path"]).read_text(encoding="utf-8"))
    assert stored["completed_unit_rows"][0] == first


def test_model_identity_and_gpu_lifecycle_require_real_gemma_offload() -> None:
    """SCENARIO-REPORT-6591-LIFECYCLE: model identity, ownership, and unload recheck."""

    assert mod.model_identity_checks(_identity()) == {
        "headline_model": True,
        "cached_sota_pair_pattern": True,
        "content_identity": True,
        "embedded_tokenizer": True,
        "no_auto_tokenizer": True,
        "no_download": True,
        "llama_cpp_cuda": True,
    }
    assert all(mod.process_lifecycle_checks(_process()).values())
    substituted = deepcopy(_identity())
    substituted["model_specs"][0]["repository_id"] = "google/gemma-4-E4B-it"
    assert mod.model_identity_checks(substituted)["headline_model"] is False
    zero_offload = deepcopy(_process())
    zero_offload["process"]["offloaded_layers"] = 0
    assert mod.process_lifecycle_checks(zero_offload)["positive_offload"] is False


def test_ready_report_recomputes_all_rows_receipts_and_attacks(tmp_path: Path) -> None:
    """REQ-REPORT-6591-REDUCER: completeness is binary and independently replayable."""

    report = _ready_report(tmp_path)
    expected = _method()["source_unit_manifest"]["bounded_unit_count"]
    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] is None
    assert report["gemma31_cfr_rows_ready_score"] == 1.0
    assert len(report["per_unit_rows"]) == expected
    assert len(report["checkpoint_receipts"]) == expected
    assert len(report["raw_stage_receipts"]) > expected * 3
    assert len(report["exact_checker_receipts"]) == expected * 3
    assert {row["attack_id"] for row in report["attack_rows"]} == set(mod.REQUIRED_ATTACK_IDS)
    assert all(
        row["passed"] and row["candidate_ready_score"] == 0.0 for row in report["attack_rows"]
    )
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []


def test_every_required_attack_mutation_forces_zero(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6591-ATTACKS: drift, loss, overwrite, and substitution fail closed."""

    report = _ready_report(tmp_path)
    mutations = {
        "prompt_drift": lambda value: value["prompt_source_router_hashes"]["checks"].update(
            prompt_hashes=False
        ),
        "post_outcome_unit_loss": lambda value: value["per_unit_rows"].pop(),
        "stage_overwrite": lambda value: value["per_unit_rows"][0]["arms"][1]["raw_stages"][
            "stage2"
        ].update(
            raw_sha256=value["per_unit_rows"][0]["arms"][1]["raw_stages"]["stage1"]["raw_sha256"]
        ),
        "answer_leakage": lambda value: value["per_unit_rows"][0]["arms"][1]["failure"].update(
            stage1_answer_leakage=True, any=True
        ),
        "uncharged_stage1": lambda value: value["per_unit_rows"][0]["arms"][1]["tokens"].update(
            stage1_charged=False
        ),
        "family_label_substitution": lambda value: value["model_spec_and_identity"][
            "identity"
        ].update(repository_id="unsloth/Qwen3.6-35B-A3B-GGUF"),
        "legacy_model_substitution": lambda value: value["model_spec_and_identity"]["model_specs"][
            0
        ].update(repository_id="google/gemma-4-E4B-it"),
        "aggregate_only_output": lambda value: value.update(per_unit_rows=[]),
        "ready_score_with_missing_rows": lambda value: value["per_unit_rows"].pop(),
    }
    assert set(mutations) == set(mod.REQUIRED_ATTACK_IDS)
    for attack_id, mutate in mutations.items():
        candidate = deepcopy(report)
        mutate(candidate)
        assert mod.stream_reducer(candidate, require_attack_rows=False)["ready_score"] == 0.0, (
            attack_id
        )


def test_blocked_partial_and_validator_states_name_exact_failures(tmp_path: Path) -> None:
    """REQ-REPORT-6591-PRECONDITIONS: blocks name the exact failed value."""

    gate = mod.build_gate_receipt(mod.REPO_ROOT)
    preconditions = _preconditions()
    preconditions["all_required_preconditions_available"] = False
    preconditions["failed_preconditions"] = ["idle_rtx_3090"]
    preconditions["checks"]["idle_rtx_3090"] = False
    preconditions["model_process_started"] = False
    blocked = mod.build_blocked_report(
        run_date="20260825",
        gate_receipt=gate,
        frozen_receipt=mod.build_frozen_hash_receipt(mod.REPO_ROOT, _method()),
        model_identity=_identity(),
        preconditions=preconditions,
        protected=_protected(),
        duration_s=1.0,
        tests_run=[],
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["first_failure"] == {
        "check": "idle_rtx_3090",
        "expected": True,
        "observed": False,
    }
    assert blocked["per_unit_rows"] == []
    assert blocked["gemma31_cfr_rows_ready_score"] == 0.0
    assert mod.validate_report(blocked) == []

    ready = _ready_report(tmp_path / "ready")
    for key, value, error in (
        ("inference_substrate", "wrong", "inference_substrate_mismatch"),
        ("verifier_is_oracle", False, "verifier_is_oracle_mismatch"),
        ("verdict_class", "positive", "verdict_class_invalid"),
        ("field_provenance", {}, "field_provenance_mismatch"),
    ):
        changed = deepcopy(ready)
        changed[key] = value
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert error in mod.validate_report(changed)
    missing = deepcopy(ready)
    missing.pop("per_unit_rows")
    assert mod.validate_report(missing)[0].startswith("missing_required_fields:")


def test_defensive_receipt_and_terminal_state_branches(tmp_path: Path) -> None:
    """REQ-REPORT-6591-ATOMIC: malformed receipts and all terminal states fail closed."""

    assert mod._decode_stage({"raw_bytes_b64": "not base64!"}) is None
    assert mod.parse_stage1_proposals(b"\xff", {}) == []
    assert mod._stage_authentic(None, "direct") is False
    assert mod._arm_authentic({"raw_stages": []}, "direct") is False
    assert mod._unit_authentic({"source_bytes_b64": "!", "task_bytes_b64": "!"}, "x") is False

    ready = _ready_report(tmp_path / "ready")
    kwargs = {
        "run_date": "20260825",
        "gate_receipt": ready["gate_check_summary"]["rows"][0],
        "frozen_receipt": ready["prompt_source_router_hashes"],
        "model_identity": ready["model_spec_and_identity"],
        "per_unit_rows": ready["per_unit_rows"],
        "checkpoint_receipts": ready["checkpoint_receipts"],
        "gpu_receipts": ready["gpu_process_receipts"],
        "preconditions": ready["preconditions_checked"],
        "protected": ready["protected_files_unchanged"],
        "duration_s": 61.0,
        "tests_run": ready["tests_run"],
    }
    disqualified = mod.build_report(**{**kwargs, "protected": {"all_unchanged": False, "rows": []}})
    assert disqualified["verdict_class"] == "disqualified"
    partial = mod.build_report(
        **{
            **kwargs,
            "per_unit_rows": ready["per_unit_rows"][:-1],
            "checkpoint_receipts": ready["checkpoint_receipts"][:-1],
        }
    )
    assert partial["verdict_class"] == "partial"

    failed_gate = deepcopy(kwargs["gate_receipt"])
    failed_gate.update(observed_value=0.0, passed=False)
    gate_block = mod.build_blocked_report(
        run_date="20260825",
        gate_receipt=failed_gate,
        frozen_receipt=kwargs["frozen_receipt"],
        model_identity=kwargs["model_identity"],
        preconditions=kwargs["preconditions"],
        protected=kwargs["protected"],
        duration_s=1.0,
        tests_run=kwargs["tests_run"],
    )
    assert gate_block["gate_check_summary"]["first_failure"]["check"] == (
        "v574_cfr_launch_ready_score"
    )
    frozen_drift = deepcopy(kwargs["frozen_receipt"])
    frozen_drift["checks"]["prompt_hashes"] = False
    frozen_drift["all_frozen_hashes_match"] = False
    drift_block = mod.build_blocked_report(
        run_date="20260825",
        gate_receipt=kwargs["gate_receipt"],
        frozen_receipt=frozen_drift,
        model_identity=kwargs["model_identity"],
        preconditions=kwargs["preconditions"],
        protected=kwargs["protected"],
        duration_s=1.0,
        tests_run=kwargs["tests_run"],
    )
    assert drift_block["gate_check_summary"]["first_failure"]["check"] == "prompt_hashes"

    wrong_model = deepcopy(ready)
    wrong_model["model_spec_and_identity"]["model_specs"] = []
    wrong_model["reproducibility_checksum"] = mod.artifact_checksum(wrong_model)
    assert "model_specs_mismatch" in mod.validate_report(wrong_model)
    false_block = deepcopy(ready)
    false_block["verdict_class"] = "blocked"
    false_block["gate_check_summary"]["first_failure"] = None
    false_block["reproducibility_checksum"] = mod.artifact_checksum(false_block)
    errors = mod.validate_report(false_block)
    assert "blocked_report_started_rows" in errors
    assert "blocked_report_missing_gate_value" in errors
    bad_checksum = deepcopy(ready)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_report(bad_checksum)


def test_atomic_write_checksum_and_cli_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6591-ATOMIC: terminal bytes validate before replacement."""

    report = _ready_report(tmp_path / "rows")
    output = tmp_path / "artifact.json"
    receipt = mod.atomic_write_report(output, report)
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert mod.load_json(output) == report
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    bad = deepcopy(report)
    bad["gemma31_cfr_rows_ready_score"] = 0.0
    bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
    with pytest.raises(ValueError, match="ready_score_mismatch"):
        mod.atomic_write_report(output, bad)
    output.write_text("{}", encoding="utf-8")
    assert mod.main(["--validate", "--output", str(output)]) == 1

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "run_experiment", lambda root, date: report)
    assert mod.main(["--date", "20260825"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out
