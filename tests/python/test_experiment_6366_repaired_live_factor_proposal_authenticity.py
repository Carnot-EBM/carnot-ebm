"""Tests for Exp6366 repaired live factor proposal authenticity.

Spec refs: REQ-LEARN-6366, REQ-LEARN-6366-1, REQ-LEARN-6366-2,
REQ-LEARN-6366-3, REQ-LEARN-6366-4, REQ-LEARN-6366-5,
REQ-LEARN-6366-6, REQ-LEARN-6366-7, REQ-LEARN-6366-8,
REQ-LEARN-6366-9, SCENARIO-LEARN-6366-GATE,
SCENARIO-LEARN-6366-MANIFEST, SCENARIO-LEARN-6366-RAW,
SCENARIO-LEARN-6366-SOURCE-BINDING, SCENARIO-LEARN-6366-ISOLATION,
SCENARIO-LEARN-6366-ORACLE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6366_repaired_live_factor_proposal_authenticity as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _write_gate(tmp_path: Path, *, score: float = 1.0) -> Path:
    path = tmp_path / mod.EXP6365_RELATIVE_PATH.name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete" if score == 1.0 else "complete_null",
        "gguf_runtime_observability_ready_score": score,
        "honest_verdict": "complete: fixture gate" if score == 1.0 else "complete_null: fixture",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\n").encode("utf-8"))
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


def _tokenizer(path: str, prompt: str) -> dict[str, Any]:
    return {
        "method": mod.TOKENIZER_METHOD,
        "loadable": path.endswith(".gguf"),
        "prompt_tokens": max(1, len(prompt.split())),
        "tokenizer_detail": "fixture embedded tokenizer",
        "autotokenizer_used": False,
    }


def _gpu_samples(model_id: str) -> dict[str, list[dict[str, Any]]]:
    memory_by_phase = {
        "before_load": 4,
        "after_load": 1200,
        "during_generation": 1280,
        "after_unload": 8,
        "after_cleanup": 4,
    }
    return {
        phase: [
            {
                "model_hf_id": model_id,
                "phase": phase,
                "gpu_index": 0,
                "timestamp_utc": "2026-08-13T00:00:00Z",
                "memory_used_mb": memory,
                "memory_free_mb": 24576 - memory,
                "utilization_pct": 1,
                "process_identity": {"pid": 123, "cmdline": "fixture child"},
            }
        ]
        for phase, memory in memory_by_phase.items()
    }


def _proposal(model_id: str, event: dict[str, Any]) -> dict[str, Any]:
    variable = event["allowed_variables"][0]
    obligation = event["source_obligations"][0]
    edit_span = event["edit_source_spans"][variable]
    return {
        "schema": mod.FACTOR_EDIT_SCHEMA,
        "proposal_id": f"{mod.model_slug(model_id)}:{event['event_id']}:0",
        "event_id": event["event_id"],
        "model_hf_id": model_id,
        "model_family": mod.model_family_for_id(model_id),
        "arm": mod.LIVE_ARM,
        "candidate_index": 0,
        "changed_factor": event["changed_factor"],
        "edits": {variable: 0.5},
        "selection_score": 0.5,
        "obligations": [
            {
                "obligation_id": obligation["obligation_id"],
                "source_start": obligation["span"]["start"],
                "source_end": obligation["span"]["end"],
                "source_sha256": obligation["span"]["sha256"],
                "source_text": obligation["text"],
            }
        ],
        "edit_source_spans": {
            variable: {
                "source_start": edit_span["start"],
                "source_end": edit_span["end"],
                "source_sha256": edit_span["sha256"],
            }
        },
    }


def _fake_generation(
    *,
    spec: dict[str, Any],
    event: dict[str, Any],
    raw_path: Path,
    stderr_path: Path,
    prompt_payload: dict[str, Any],
    prompt_text: str,
    seed: int,
    sampling: dict[str, Any],
    timeout_s: float,
    prompt_token_count: int,
    source_hash: str,
    output_dir: Path,
) -> dict[str, Any]:
    del prompt_payload, timeout_s, output_dir
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(json.dumps(_proposal(spec["hf_id"], event), sort_keys=True).encode("utf-8"))
    stderr_path.write_text("CARNOT_USAGE:{\"prompt_tokens\": 7, \"completion_tokens\": 5, \"total_tokens\": 12}\n", encoding="utf-8")
    return {
        "model_hf_id": spec["hf_id"],
        "event_id": event["event_id"],
        "stdout_path": str(raw_path),
        "stdout_sha256": mod.sha256_file(raw_path),
        "stdout_byte_count": raw_path.stat().st_size,
        "stdout_excerpt": raw_path.read_text(encoding="utf-8")[:200],
        "stderr_path": str(stderr_path),
        "stderr_sha256": mod.sha256_file(stderr_path),
        "stderr_byte_count": stderr_path.stat().st_size,
        "stderr_excerpt": stderr_path.read_text(encoding="utf-8"),
        "raw_output_path": str(raw_path),
        "raw_output_sha256": mod.sha256_file(raw_path),
        "raw_output_bytes": raw_path.stat().st_size,
        "returncode": 0,
        "signal": None,
        "timed_out": False,
        "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        "usage_receipt_valid": True,
        "live_autoregressive_generation_invoked": True,
        "authenticated_gpu_offload": True,
        "gpu_samples_by_phase": _gpu_samples(spec["hf_id"]),
        "phase_timings": {
            phase: {"started_ns": index, "ended_ns": index + 1, "duration_s": 0.1}
            for index, phase in enumerate(mod.REQUIRED_TIMING_PHASES)
        },
        "prompt_context": {
            "model_hf_id": spec["hf_id"],
            "prompt_tokens": prompt_token_count,
            "requested_output_tokens": sampling["max_tokens"],
            "n_ctx": sampling["n_ctx"],
            "capacity_margin": sampling["n_ctx"] - prompt_token_count - sampling["max_tokens"],
            "fits": True,
        },
        "source_hash": source_hash,
        "source_hash_ok": True,
        "stdout_nonempty": True,
        "contract_ok": True,
        "pid": 12345,
        "process_identity": {"pid": 12345, "exists": True, "cmdline": "fixture child"},
        "dispatcher": "fixture_exp6365_child_contract",
        "argv_sanitized": ["python", "-c", "fixture"],
        "argv_sha256": mod.sha256_json({"seed": seed, "model": spec["hf_id"]}),
        "command_hash": mod.sha256_json({"cmd": "fixture", "seed": seed}),
        "prompt_sha256": mod.sha256_text(prompt_text),
        "environment_allowlist_hash": mod.sha256_json({"CUDA_VISIBLE_DEVICES": str(spec["gpu"])}),
        "sampling": sampling,
    }


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True, gate_score: float = 1.0) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        exp6365_path=_write_gate(tmp_path, score=gate_score),
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
        host_checks_func=mod.deterministic_host_receipts,
        generation_func=_fake_generation,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _read_json(receipt: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def test_req_learn_6366_spec_declares_fields_scenarios_and_principles() -> None:
    """REQ-LEARN-6366-1: OpenSpec owns the artifact fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6366") :]
    for token in (
        "SCENARIO-LEARN-6366-GATE",
        "SCENARIO-LEARN-6366-MANIFEST",
        "SCENARIO-LEARN-6366-RAW",
        "SCENARIO-LEARN-6366-SOURCE-BINDING",
        "SCENARIO-LEARN-6366-ISOLATION",
        "SCENARIO-LEARN-6366-ORACLE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_6366_model_specs_use_cached_pair_and_exp6365_gate(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6366-2: the repaired Exp6365 gate controls model calls."""

    paths = _model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )
    gate = mod.exp6365_gate_receipt(_write_gate(tmp_path))

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert resolution["all_resolved"] is True
    assert [row["hf_id"] for row in resolution["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in resolution["MODEL_SPECS"])
    assert gate["gate_passed"] is True
    assert gate["gguf_runtime_observability_ready_score"] == 1.0
    assert mod.AUTOTOKENIZER_USAGE_COUNT == 0

    blocked = mod.exp6365_gate_receipt(_write_gate(tmp_path / "blocked", score=0.0))
    assert blocked["gate_passed"] is False


def test_scenario_learn_6366_manifest_is_balanced_source_bound_and_unlabelled(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6366-MANIFEST: source spans are sealed before calls."""

    artifact = _artifact(tmp_path)
    manifest = _read_json(artifact["sealed_event_manifest_path_hash_license_and_balance"])
    balance = artifact["sealed_event_manifest_path_hash_license_and_balance"]["balance"]
    exposed = json.dumps(manifest["events_exposed_to_proposer"], sort_keys=True)

    assert manifest["event_count"] == 12
    assert balance["balanced"] is True
    assert balance["family_count"] == 3
    assert all(count == 4 for count in balance["events_by_family"].values())
    assert all(sorted(surfaces) == ["symbolic", "symbolic", "verbal", "verbal"] for surfaces in balance["surfaces_by_family"].values())
    assert "protected_outcome" not in exposed
    assert "exact_label" not in exposed
    for event in mod.generated_events():
        source = event["source_text"]
        for obligation in event["source_obligations"]:
            span = obligation["span"]
            assert source[span["start"] : span["end"]] == obligation["text"]
            assert mod.sha256_text(obligation["text"]) == span["sha256"]


def test_scenario_learn_6366_raw_before_parse_source_binding_and_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6366-RAW: authenticated raw outputs freeze first."""

    artifact = _artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["repaired_live_factor_proposal_authenticity_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert all(artifact["live_autoregressive_generation_invoked_by_model"].values())
    assert artifact["raw_output_before_parse_paths_hashes_and_counts"]["all_raw_outputs_frozen_before_parse"] is True
    assert artifact["source_span_alignment_and_decomposition_conflict_counts"]["context_memory_substitution_count"] == 0
    assert artifact["source_span_alignment_and_decomposition_conflict_counts"]["unsupported_obligation_count"] == 0
    assert artifact["exact_checker_paths_hashes_versions_calls_costs_and_errors"]["all_calls_after_raw_freeze"] is True
    assert artifact["exact_pass_fail_counts_by_model"]["total_exact_calls"] == len(mod.MANDATED_MODEL_IDS)
    for model_id in mod.MANDATED_MODEL_IDS:
        assert artifact["parse_valid_invalid_timeout_and_abstain_counts_by_model"]["by_model"][model_id]["valid"] == 1
        assert artifact["raw_output_before_parse_paths_hashes_and_counts"]["by_model"][model_id]["byte_count"] > 0


def test_req_learn_6366_parser_rejects_empty_child_failure_and_substitution(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6366-6: parser failures never become proposals."""

    events = mod.generated_events()
    event = events[0]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    empty = raw_dir / "empty.raw"
    empty.write_bytes(b"")
    timeout = raw_dir / "timeout.raw"
    timeout.write_text(json.dumps(_proposal(mod.MANDATED_MODEL_IDS[1], event)), encoding="utf-8")
    substituted = _proposal(mod.MANDATED_MODEL_IDS[2], event)
    substituted["obligations"][0]["source_text"] = "model prior text"
    substituted_path = raw_dir / "substituted.raw"
    substituted_path.write_text(json.dumps(substituted), encoding="utf-8")

    parsed = mod.parse_raw_outputs(
        {
            "receipts": {
                mod.MANDATED_MODEL_IDS[0]: {
                    "raw_output_path": str(empty),
                    "raw_output_sha256": mod.sha256_file(empty),
                    "raw_output_bytes": 0,
                    "returncode": 0,
                    "timed_out": False,
                },
                mod.MANDATED_MODEL_IDS[1]: {
                    "raw_output_path": str(timeout),
                    "raw_output_sha256": mod.sha256_file(timeout),
                    "raw_output_bytes": timeout.stat().st_size,
                    "returncode": 0,
                    "timed_out": True,
                },
                mod.MANDATED_MODEL_IDS[2]: {
                    "raw_output_path": str(substituted_path),
                    "raw_output_sha256": mod.sha256_file(substituted_path),
                    "raw_output_bytes": substituted_path.stat().st_size,
                    "returncode": 0,
                    "timed_out": False,
                },
            }
        },
        events,
    )

    counts = parsed["counts"]["by_model"]
    conflicts = parsed["source_span_alignment_and_decomposition_conflict_counts"]
    assert counts[mod.MANDATED_MODEL_IDS[0]]["invalid"] == 1
    assert counts[mod.MANDATED_MODEL_IDS[1]]["timeouts"] == 1
    assert counts[mod.MANDATED_MODEL_IDS[2]]["invalid"] == 1
    assert conflicts["context_memory_substitution_count"] == 1
    assert parsed["parsed_proposals"] == []

    abstain = raw_dir / "abstain.raw"
    abstain.write_text(json.dumps({"event_id": event["event_id"], "abstain": True}), encoding="utf-8")
    abstained = mod.parse_raw_outputs(
        {
            "receipts": {
                mod.MANDATED_MODEL_IDS[0]: {
                    "raw_output_path": str(abstain),
                    "raw_output_sha256": mod.sha256_file(abstain),
                    "raw_output_bytes": abstain.stat().st_size,
                    "returncode": 0,
                    "timed_out": False,
                }
            }
        },
        events,
    )
    assert abstained["counts"]["by_model"][mod.MANDATED_MODEL_IDS[0]]["abstain"] == 1
    assert abstained["parsed_proposals"] == []


def test_scenario_learn_6366_isolation_and_oracle_boundary(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6366-ISOLATION: exact checkers alone are oracles."""

    artifact = _artifact(tmp_path)
    isolation = artifact["same_step_read_write_isolation_results"]
    checker = artifact["exact_checker_paths_hashes_versions_calls_costs_and_errors"]

    assert isolation["same_step_write_count"] == 0
    assert isolation["released_snapshot_unchanged"] is True
    assert isolation["event_manifest_unchanged"] is True
    assert isolation["exact_checker_unchanged"] is True
    assert all(row["fail_closed"] is True for row in isolation["mutation_tests"])
    assert artifact["verifier_is_oracle"] is True
    assert checker["protected_exact_task_checkers_are_oracle"] is True
    assert checker["model_proposals_are_oracles"] is False
    assert checker["parsing_is_oracle"] is False
    assert checker["learned_scores_are_oracles"] is False

    bad = deepcopy(artifact)
    bad["source_span_alignment_and_decomposition_conflict_counts"]["context_memory_substitution_count"] = 1
    mod.refresh_terminal_fields(bad)
    assert bad["repaired_live_factor_proposal_authenticity_ready_score"] == 0.0


def test_req_learn_6366_artifact_schema_blocked_path_and_checksum(tmp_path: Path) -> None:
    """REQ-LEARN-6366-8: readiness is conjunctive and checksummed."""

    artifact = _artifact(tmp_path)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blocked = _artifact(tmp_path / "blocked", gate_score=0.0)
    assert blocked["repaired_live_factor_proposal_authenticity_ready_score"] == 0.0
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["models_used"] == []
    assert blocked["live_autoregressive_generation_invoked_by_model"] == {}

    bad = deepcopy(artifact)
    bad["autotokenizer_usage_count"] = 1
    mod.refresh_terminal_fields(bad)
    try:
        mod.validate_artifact(bad)
    except ValueError as exc:
        assert "autotokenizer_usage_count_not_zero" in str(exc)
    else:
        raise AssertionError("validate_artifact accepted AutoTokenizer usage")


def test_req_learn_6366_defensive_edges_are_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6366-6: helper edge cases stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.write_payload_or_hash(tmp_path / "dry.json", {"x": 1}, write=False) == mod.sha256_json(
        {"x": 1}
    )
    assert mod.revision_from_path(Path("/cache/snapshots/rev/model.gguf")) == "rev"
    assert mod.quantization_from_path(Path("model-without-quant.gguf")) == "unknown"
    assert mod.parse_gpu_query("0, Test GPU, 24576, 128, 24448\n") == [
        {"index": 0, "name": "Test GPU", "total_mb": 24576, "used_mb": 128, "free_mb": 24448}
    ]
    assert mod.extract_json_payload("prefix {\"a\": 1} suffix") == {"a": 1}
    assert mod.extract_json_payload("prefix {bad} suffix") is None

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda path, prompt: {  # noqa: ARG005
            "method": mod.TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": "missing",
        },
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_default_missing" in missing["blocked_reasons"]
    assert "cached_sota_pair_dense_missing" in missing["blocked_reasons"]
    assert any(reason.startswith("missing_cached_sota_pair_row:") for reason in missing["blocked_reasons"])
    assert any(reason.startswith("missing_gguf_file:") for reason in missing["blocked_reasons"])
    assert any(reason.startswith("embedded_tokenizer_unavailable:") for reason in missing["blocked_reasons"])

    bad_preconditions = mod.preconditions_checked(
        date="20260813",
        exp6365_gate={"gate_passed": False},
        model_resolution=missing,
        host={
            "cuda_devices": {"available": False, "count": 0},
            "vram": {},
            "disk": {"available_gb": 1.0},
            "llama_cpp": {"gpu_offload_receipt": False},
        },
        event_receipt={"present": False, "sha256": None},
        snapshot_receipt={"present": False, "sha256": None},
        snapshot_read_only={"read_only": False},
        schema_receipt={"present": False, "sha256": None},
        context_receipts={mod.MANDATED_MODEL_IDS[0]: {"fits": False}},
        protected_before={"missing": None},
        data_dir=tmp_path / "no-sidecars",
    )
    for reason in (
        "exp6365_gate_not_ready",
        "two_cuda_gpus_unavailable",
        "llama_cpp_gpu_offload_unavailable",
        "disk_space_below_10gb",
        "insufficient_free_vram",
        "prompt_context_overflow",
        "event_manifest_missing",
        "released_snapshot_not_read_only",
        "bounded_schema_missing",
        "protected_hash_missing",
    ):
        assert reason in bad_preconditions["blocked_reasons"]

    event = mod.generated_events()[0]
    no_obligation = _proposal(mod.MANDATED_MODEL_IDS[0], event)
    no_obligation["obligations"] = "bad"
    assert mod.source_span_alignment(no_obligation, event)["unsupported_obligation_count"] == 1

    unsupported = _proposal(mod.MANDATED_MODEL_IDS[0], event)
    unsupported["obligations"][0]["obligation_id"] = "unknown"
    unsupported_alignment = mod.source_span_alignment(unsupported, event)
    assert unsupported_alignment["unsupported_obligation_count"] == 1

    bad_edit_span = _proposal(mod.MANDATED_MODEL_IDS[0], event)
    variable = event["allowed_variables"][0]
    bad_edit_span["edit_source_spans"][variable]["source_sha256"] = "sha256:bad"
    assert mod.source_span_alignment(bad_edit_span, event)["invalid_edit_span_count"] == 1

    exact_fail = _proposal(mod.MANDATED_MODEL_IDS[0], event)
    exact_fail["edits"][variable] = -0.5
    exact_error = _proposal(mod.MANDATED_MODEL_IDS[1], event)
    exact_error["edits"][variable] = "not-a-number"
    checker, counts = mod.exact_checker_receipts_and_counts([exact_fail, exact_error], [event])
    assert checker["checker_error_count"] == 1
    assert counts["by_model"][mod.MANDATED_MODEL_IDS[0]]["exact_fail"] == 1

    harm = mod.harm_summary(
        model_resolution={
            "MODEL_SPECS": [
                {"hf_id": "missing", "exists": False, "tokenizer_loadable": False}
            ]
        },
        generation={"receipts": {mod.MANDATED_MODEL_IDS[0]: {"contract_ok": False}}},
        raw_before_parse={"all_raw_outputs_frozen_before_parse": False},
        parse_counts={
            "by_model": {
                mod.MANDATED_MODEL_IDS[0]: {"invalid": 1, "timeouts": 1, "abstain": 1}
            }
        },
        conflicts={"context_memory_substitution_count": 1, "unsupported_obligation_count": 1},
    )
    assert harm["missing_model_cells"] == ["missing"]
    assert "raw_before_parse" in harm["flagged_cells"]
    assert "invalid_parse:" + mod.MANDATED_MODEL_IDS[0] in harm["flagged_cells"]
    assert "timeout:" + mod.MANDATED_MODEL_IDS[0] in harm["flagged_cells"]
    assert "abstain:" + mod.MANDATED_MODEL_IDS[0] in harm["flagged_cells"]
    assert "source_substitution" in harm["flagged_cells"]
    assert "unsupported_obligation" in harm["flagged_cells"]
    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}
