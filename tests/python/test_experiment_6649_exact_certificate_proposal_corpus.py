"""Tests for the frozen exact-certificate proposal corpus.

Spec refs: REQ-REPORT-6649, SCENARIO-REPORT-6649-IMMUTABLE-SELECTION,
SCENARIO-REPORT-6649-RAW-RETENTION,
SCENARIO-REPORT-6649-COMPLETENESS-INDEPENDENT-OF-SUCCESS,
SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC, REQ-CONSTRAINT-6649,
SCENARIO-CONSTRAINT-6649-FIRST-FAILURE,
SCENARIO-CONSTRAINT-6649-PARSE-FAILURE, REQ-INFER-SOTA-6649,
SCENARIO-INFER-SOTA-6649-COMPARABLE-ROWS, and
SCENARIO-INFER-SOTA-6649-NO-LEGACY-OR-REPAIR.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from carnot import experiment_6649_exact_certificate_proposal_corpus as exp


REPO = Path(__file__).resolve().parents[2]


def _models() -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.DEFINED_MODEL_SPECS):
        rows.append(
            {
                **spec,
                "model_path": f"/cache/{spec['family_id']}.gguf",
                "model_sha256": f"sha256:{index + 1:064x}",
                "resolved": True,
            }
        )
    return rows


def _receipt(model: dict, index: int = 0) -> dict:
    return {
        "session_id": f"session-{model['family_id']}",
        "pid": 3000 + index,
        "pid_start_ticks": 9000 + index,
        "executable": "/cache/llama-server",
        "argv_sha256": f"sha256:{index + 11:064x}",
        "device_index": model["device_index"],
        "device_uuid": f"GPU-test-{model['device_index']}",
        "model_sha256": model["model_sha256"],
        "cuda_offload": True,
        "offloaded_layers": 40,
        "accelerator_observed": True,
        "owned_process": True,
        "authentic": True,
    }


def _generation(raw: bytes) -> dict:
    return {
        "raw_output": raw,
        "raw_api_response_sha256": exp.sha256_bytes(b"api:" + raw),
        "prompt_tokens": 100,
        "generated_tokens": max(1, len(raw.splitlines())),
        "latency_s": 0.25,
        "started_monotonic_ns": 100,
        "finished_monotonic_ns": 250_000_100,
        "http_status": 200,
        "finish_reason": "stop",
        "failure_kind": None,
    }


def _row(task: dict, model: dict, *, raw: bytes | None = None, index: int = 0) -> dict:
    output = task["exact_target"].encode() if raw is None else raw
    return exp.build_candidate_row(task, model, _generation(output), _receipt(model, index))


def _all_rows(manifest: dict, models: list[dict], *, valid: bool = True) -> list[dict]:
    rows = []
    for model_index, model in enumerate(models):
        for task_index, task in enumerate(manifest["tasks"]):
            raw = None
            if not valid:
                lines = task["exact_target"].splitlines()
                raw = "\n".join([*lines[:2], lines[1]]).encode()
            rows.append(_row(task, model, raw=raw, index=model_index * 100 + task_index))
    return rows


def _upstream(observed: object = True) -> dict:
    return {
        "path": exp.UPSTREAM_PATH.as_posix(),
        "sha256": "sha256:" + "a" * 64,
        "field": "all_mandated_models_admitted",
        "expected_value": True,
        "observed_value": observed,
        "passed": observed is True,
    }


def _protected() -> dict:
    rows = [
        {"path": path.as_posix(), "before_sha256": "x", "after_sha256": "x", "unchanged": True}
        for path in exp.PROTECTED_PATHS
    ]
    return {"rows": rows, "all_unchanged": True}


def _preconditions() -> dict:
    return {
        "all_required_preconditions_available": True,
        "failed_preconditions": [],
        "checks": {
            "upstream_gate": True,
            "model_resolution": True,
            "model_hashes": True,
            "hardware": True,
            "tools": True,
            "task_manifest": True,
            "compiler_and_checker": True,
            "protected_hashes": True,
        },
    }


def _artifact(valid: bool = True) -> dict:
    manifest = exp.build_frozen_task_manifest()
    models = _models()
    return exp.build_artifact(
        date="20260826",
        upstream_gate_receipt=_upstream(),
        model_specs=models,
        manifest=manifest,
        rows=_all_rows(manifest, models, valid=valid),
        preconditions=_preconditions(),
        protected_files=_protected(),
        tests_run=exp.DEFAULT_TESTS_RUN,
        duration_s=120.0,
    )


def test_req_6649_spec_anchors_and_model_policy() -> None:
    """REQ-REPORT-6649 and REQ-INFER-SOTA-6649 freeze the public contract."""

    anchors = {
        exp.REPORT_SPEC_PATH: (
            "REQ-REPORT-6649",
            "SCENARIO-REPORT-6649-IMMUTABLE-SELECTION",
            "SCENARIO-REPORT-6649-RAW-RETENTION",
            "SCENARIO-REPORT-6649-COMPLETENESS-INDEPENDENT-OF-SUCCESS",
            "SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC",
        ),
        exp.CONSTRAINT_SPEC_PATH: (
            "REQ-CONSTRAINT-6649",
            "SCENARIO-CONSTRAINT-6649-FIRST-FAILURE",
            "SCENARIO-CONSTRAINT-6649-PARSE-FAILURE",
        ),
        exp.INFERENCE_SPEC_PATH: (
            "REQ-INFER-SOTA-6649",
            "SCENARIO-INFER-SOTA-6649-COMPARABLE-ROWS",
            "SCENARIO-INFER-SOTA-6649-NO-LEGACY-OR-REPAIR",
        ),
    }
    for path, expected in anchors.items():
        text = path.read_text(encoding="utf-8")
        assert all(anchor in text for anchor in expected)
    assert exp.DEFINED_MODEL_SPECS == [
        {
            "family_id": "qwen36_flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe",
            "quantization": "Q4_K_M",
            "device_index": 0,
            "resolution_method": "cached_sota_pair",
            "headline_eligible": True,
        },
        {
            "family_id": "gemma4_26b_middle_moe",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "role": "middle_moe",
            "quantization": "Q4_K_M",
            "device_index": 1,
            "resolution_method": "cached_sota_pair",
            "headline_eligible": True,
        },
    ]
    assert exp.INFERENCE_SUBSTRATE == "local_llama_cpp_cuda_direct_exact_certificate_generation"
    assert exp.VERIFIER_IS_ORACLE is False


def test_scenario_infer_6649_resolves_exact_cached_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-INFER-SOTA-6649-COMPARABLE-ROWS uses the required resolver call."""

    calls: list[dict] = []

    def pair(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [
            {"hf_id": exp.DEFINED_MODEL_SPECS[0]["hf_id"], "gpu": 0, "model_path": "/q.gguf"},
            {"hf_id": exp.DEFINED_MODEL_SPECS[1]["hf_id"], "gpu": 1, "model_path": "/g.gguf"},
        ]

    monkeypatch.setattr(exp, "cached_sota_pair", pair)
    monkeypatch.setattr(exp, "sha256_file", lambda path: f"sha256:{Path(path).name:0>64}")
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) in {"/q.gguf", "/g.gguf"})
    rows = exp.resolve_model_specs()
    assert calls == [{"gpu_indices": (0, 1), "model_indices": (0, 1)}]
    assert [row["model_path"] for row in rows] == ["/q.gguf", "/g.gguf"]
    assert all(row["resolved"] for row in rows)


def test_scenario_report_6649_manifest_is_frozen_and_balanced() -> None:
    """SCENARIO-REPORT-6649-IMMUTABLE-SELECTION fixes 24 feasible tasks before output."""

    first = exp.build_frozen_task_manifest()
    second = exp.build_frozen_task_manifest()
    assert first == second
    assert first["task_count"] == 24
    assert first["split_counts"] == {"calibration": 12, "held": 12}
    assert len(first["ordered_task_ids"]) == len(set(first["ordered_task_ids"])) == 24
    assert all(task["exact_target"] and task["known_feasible"] for task in first["tasks"])
    assert first["manifest_sha256"] == exp.manifest_checksum(first)
    assert first["parser_version"] == exp.PARSER_VERSION
    assert first["decode_parameters"] == exp.DECODE_PARAMETERS


def test_scenario_constraint_6649_localizes_first_exact_failure() -> None:
    """SCENARIO-CONSTRAINT-6649-FIRST-FAILURE keeps the exact valid prefix."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    lines = task["exact_target"].splitlines()
    candidate = "\n".join([*lines[:2], lines[1]])
    parsed = exp.parse_proposal(task, candidate.encode())
    localized = exp.localize_exact_outcome(task, parsed["parsed_plan"])
    assert parsed["parse_succeeded"] is True
    assert localized["exact_final_validity"] is False
    assert localized["first_failing_step"] == 2
    assert localized["valid_prefix_length"] == 2
    assert localized["per_step_exact_outcomes"][2]["accepted"] is False
    assert localized["per_step_exact_outcomes"][2]["reason"] == "precondition_violation"
    assert localized["valid_prefix"] == "\n".join(lines[:2])


def test_scenario_constraint_6649_valid_and_goal_incomplete_paths() -> None:
    """REQ-CONSTRAINT-6649 records valid plans and terminal goal failures exactly."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    valid = exp.localize_exact_outcome(task, task["exact_target"])
    incomplete = exp.localize_exact_outcome(task, task["exact_target"].splitlines()[0])
    assert valid["exact_final_validity"] is True
    assert valid["first_failing_step"] is None
    assert valid["valid_prefix_length"] == task["target_step_count"]
    assert incomplete["exact_final_validity"] is False
    assert incomplete["first_failing_step"] == 1
    assert incomplete["valid_prefix_length"] == 1
    assert incomplete["exact_final_result"]["reason"] == "unmet_goal"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"", "empty_output"),
        (b"\xff", "invalid_utf8"),
        (b"```\nOPEN(x)\n```", "code_fence_not_allowed"),
        (b"Here is the plan", "noncanonical_line"),
    ],
)
def test_scenario_constraint_6649_parse_failures_are_explicit(raw: bytes, reason: str) -> None:
    """SCENARIO-CONSTRAINT-6649-PARSE-FAILURE never invents an invalid plan zero."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    parsed = exp.parse_proposal(task, raw)
    assert parsed == {
        "parser_version": exp.PARSER_VERSION,
        "parse_succeeded": False,
        "parsed_plan": None,
        "parsed_step_count": None,
        "parse_failure": reason,
    }


def test_scenario_report_6649_raw_is_sealed_before_parse() -> None:
    """SCENARIO-REPORT-6649-RAW-RETENTION keeps raw and parsed evidence together."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    model = _models()[0]
    raw = task["exact_target"].encode()
    row = _row(task, model, raw=raw)
    assert row["raw_output"] == raw.decode()
    assert row["raw_output_sha256"] == exp.sha256_bytes(raw)
    assert row["raw_recorded_before_parse"] is True
    assert row["parsed_plan"] == task["exact_target"]
    assert row["exact_final_validity"] is True
    assert row["attempt_count"] == 1
    assert row["regeneration_attempted"] is False
    assert row["row_sha256"] == exp.candidate_row_checksum(row)


def test_scenario_report_6649_parse_failure_row_has_no_zero_coercion() -> None:
    """SCENARIO-REPORT-6649-RAW-RETENTION preserves a missing parsed denominator."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    row = _row(task, _models()[0], raw=b"not a plan")
    assert row["parse_succeeded"] is False
    assert row["parsed_plan"] is None
    assert row["exact_final_validity"] is None
    assert row["first_failing_step"] is None
    assert row["valid_prefix_length"] is None
    assert exp.parse_failure_rows([row])[0]["parse_failure"] == "noncanonical_line"
    assert exp.regeneration_headroom_rows([row]) == []


def test_scenario_report_6649_completeness_is_independent_of_success() -> None:
    """SCENARIO-REPORT-6649-COMPLETENESS-INDEPENDENT-OF-SUCCESS owns row completeness."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    rows = _all_rows(manifest, models, valid=False)
    reduction = exp.recompute_aggregates(rows, manifest, models)
    assert reduction["candidate_corpus_complete"] is True
    assert reduction["pooled_metrics"]["direct_exact_success_count"] == 0
    assert reduction["pooled_metrics"]["direct_exact_success_rate"] == 0.0
    assert reduction["regeneration_headroom_count"] == len(rows)
    assert all(row["valid_prefix_length"] == 2 for row in rows)


def test_req_report_6649_metrics_use_rows_and_wilson_uncertainty() -> None:
    """REQ-REPORT-6649 reports model and pooled direct rates with uncertainty."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    rows = _all_rows(manifest, models)
    rows[0] = _row(manifest["tasks"][0], models[0], raw=b"not a plan")
    metrics = exp.model_level_metrics(rows, models)
    pooled = exp.pooled_metrics(rows)
    assert metrics[0]["expected_row_count"] == 24
    assert metrics[0]["parse_failure_count"] == 1
    assert metrics[0]["direct_exact_success_count"] == 23
    assert len(metrics[0]["direct_exact_success_interval_95"]) == 2
    assert pooled["row_count"] == 48
    assert pooled["direct_exact_success_count"] == 47
    assert exp.wilson_interval(0, 0) == [0.0, 0.0]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[:-1],
        lambda rows: [rows[0], *rows],
        lambda rows: [{**rows[0], "raw_output_sha256": "sha256:bad"}, *rows[1:]],
        lambda rows: [{**rows[0], "model_hf_id": "Qwen/Qwen3.5-0.8B"}, *rows[1:]],
        lambda rows: [
            {
                **rows[0],
                "process_and_accelerator_receipt": {
                    **rows[0]["process_and_accelerator_receipt"],
                    "cuda_offload": False,
                },
            },
            *rows[1:],
        ],
    ],
)
def test_req_report_6649_row_attacks_fail_closed(mutator) -> None:
    """SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC rejects missing or changed rows."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    rows = _all_rows(manifest, models)
    reduction = exp.recompute_aggregates(mutator(rows), manifest, models)
    assert reduction["candidate_corpus_complete"] is False
    assert reduction["failed_checks"]


def test_req_report_6649_complete_artifact_and_provenance() -> None:
    """REQ-REPORT-6649 emits every required field from authentic complete rows."""

    artifact = _artifact(valid=False)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "null"
    assert artifact["candidate_corpus_complete"] is True
    assert artifact["regeneration_headroom_count"] == 48
    assert artifact["candidate_rows"] == artifact["per_unit_rows"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []


def test_req_report_6649_positive_direct_evidence_has_no_repair_claim() -> None:
    """REQ-REPORT-6649 classifies direct success without claiming repair."""

    artifact = _artifact(valid=True)
    assert artifact["verdict_class"] == "positive"
    assert (
        artifact["aggregate_row_recomputation"]["pooled_metrics"]["direct_exact_success_rate"]
        == 1.0
    )
    assert "repair" in artifact["honest_verdict"]
    assert "no repair was attempted" in artifact["honest_verdict"]


def test_scenario_report_6649_blocked_artifact_names_exact_gate() -> None:
    """SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC retains the failed observed value."""

    artifact = exp.build_blocked_artifact(
        date="20260826",
        failed_condition="upstream_gate",
        expected=True,
        observed=None,
        upstream_gate_receipt=_upstream(None),
        model_specs=_models(),
        manifest=exp.build_frozen_task_manifest(),
        preconditions={"all_required_preconditions_available": False},
        protected_files=_protected(),
        tests_run=exp.DEFAULT_TESTS_RUN,
        duration_s=1.0,
    )
    assert artifact["status"] == "blocked_upstream_gate"
    assert artifact["honest_verdict"].startswith("blocked_upstream_gate")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["observed"] is None
    assert artifact["candidate_rows"] == []
    assert artifact["candidate_corpus_complete"] is False
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_6649_validator_detects_mutations() -> None:
    """SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC rejects aggregate and checksum drift."""

    artifact = _artifact()
    changed = deepcopy(artifact)
    changed["candidate_corpus_complete"] = False
    assert "aggregate_recomputation_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["inference_substrate"] = "cpu"
    assert "inference_substrate_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    assert "verifier_is_oracle_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "winner"
    assert "verdict_class_invalid" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_scenario_report_6649_atomic_writer_and_protected_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC writes one durable JSON document."""

    artifact = _artifact()
    target = tmp_path / "result.json"
    exp.write_artifact_atomic(target, artifact)
    assert json.loads(target.read_text()) == artifact
    assert not list(tmp_path.glob(".exp6649-*"))
    before = exp.protected_hashes(REPO)
    receipt = exp.protected_files_receipt(REPO, before)
    assert receipt["all_unchanged"] is True
    assert len(receipt["rows"]) == 2
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.write_artifact_atomic(target, {**artifact, "duration_s": 1.0})


def test_req_6649_fail_closed_manifest_and_parser_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-6649 rejects infeasible selection and parser identity disagreement."""

    generated = exp.exact_corpus.generate_plan_tasks()
    changed = deepcopy(generated)
    selected = next(row for row in changed if row["task_id"] == exp.FROZEN_TASK_IDS[0])
    selected["known_feasible"] = False
    monkeypatch.setattr(exp.exact_corpus, "generate_plan_tasks", lambda: changed)
    with pytest.raises(ValueError, match="not exactly feasible"):
        exp.build_frozen_task_manifest()
    monkeypatch.setattr(exp.exact_corpus, "generate_plan_tasks", lambda: generated)
    task = exp.build_frozen_task_manifest()["tasks"][0]
    fallback = {key: value for key, value in task.items() if key != "task_payload"}
    assert exp.parse_proposal(fallback, b" \n")["parse_failure"] == "empty_output"
    assert exp.parse_proposal(fallback, task["exact_target"].encode())["parse_succeeded"] is True
    monkeypatch.setattr(exp.exact_corpus, "_executor_parse_plan", lambda *_: ([], "bad"))
    with pytest.raises(ValueError, match="parsed plan failed exact parser"):
        exp.localize_exact_outcome(task, task["exact_target"])


@pytest.mark.parametrize(
    "field",
    [
        "decode_parameters",
        "raw_recorded_before_parse",
        "attempt_contract",
        "no_intervention_contract",
    ],
)
def test_req_6649_recheck_rejects_contract_mutations(field: str) -> None:
    """REQ-REPORT-6649 independently checks every no-intervention row contract."""

    task = exp.build_frozen_task_manifest()["tasks"][0]
    model = _models()[0]
    row = _row(task, model)
    if field == "decode_parameters":
        row["decode_parameters"] = {}
    elif field == "raw_recorded_before_parse":
        row["raw_recorded_before_parse"] = False
    elif field == "attempt_contract":
        row["attempt_count"] = 2
    else:
        row["repair_attempted"] = True
    row["row_sha256"] = exp.candidate_row_checksum(row)
    assert field in exp._recheck_row(row, task, model)


def test_req_6649_recheck_and_reducer_fail_closed_edges() -> None:
    """REQ-REPORT-6649 rejects corrupt bytes, model identity, and manifest drift."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    row = _row(manifest["tasks"][0], models[0])
    row["raw_output_bytes_b64"] = "!"
    assert exp._recheck_row(row, manifest["tasks"][0], models[0]) == ["raw_output_bytes_b64"]
    rows = _all_rows(manifest, models)
    bad_models = deepcopy(models)
    bad_models[0]["resolved"] = False
    assert "model_specs" in exp.recompute_aggregates(rows, manifest, bad_models)["failed_checks"]
    bad_manifest = deepcopy(manifest)
    bad_manifest["manifest_sha256"] = "sha256:bad"
    assert (
        "frozen_task_manifest"
        in exp.recompute_aggregates(rows, bad_manifest, models)["failed_checks"]
    )


@pytest.mark.parametrize("failed", ["upstream", "preconditions", "protected", "rows"])
def test_req_6649_complete_builder_blocks_integrity_failures(failed: str) -> None:
    """SCENARIO-REPORT-6649-BLOCKED-AND-ATOMIC names each blocking evidence class."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    rows = _all_rows(manifest, models)
    upstream = _upstream()
    preconditions = _preconditions()
    protected = _protected()
    if failed == "upstream":
        upstream = _upstream(False)
    elif failed == "preconditions":
        preconditions["all_required_preconditions_available"] = False
    elif failed == "protected":
        protected["all_unchanged"] = False
    else:
        rows = rows[:-1]
    artifact = exp.build_artifact(
        date="20260826",
        upstream_gate_receipt=upstream,
        model_specs=models,
        manifest=manifest,
        rows=rows,
        preconditions=preconditions,
        protected_files=protected,
        tests_run=exp.DEFAULT_TESTS_RUN,
        duration_s=1.0,
    )
    assert artifact["status"] == "blocked_candidate_corpus_integrity"
    assert artifact["verdict_class"] == "blocked"


def _rehash(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


def test_req_6649_validator_covers_schema_and_blocked_edges() -> None:
    """REQ-REPORT-6649 validator rejects every closed-schema edge."""

    artifact = _artifact()
    missing = deepcopy(artifact)
    missing.pop("status")
    assert exp.validate_artifact(missing)[0].startswith("missing_required_fields:")
    changed = deepcopy(artifact)
    changed["field_provenance"] = {}
    assert "field_provenance_mismatch" in exp.validate_artifact(_rehash(changed))
    changed = deepcopy(artifact)
    changed["per_unit_rows"] = []
    assert "candidate_per_unit_rows_mismatch" in exp.validate_artifact(_rehash(changed))
    changed = deepcopy(artifact)
    changed["verdict_class"] = "blocked"
    assert "complete_verdict_class_invalid" in exp.validate_artifact(_rehash(changed))
    blocked = exp.build_blocked_artifact(
        date="20260826",
        failed_condition="identity",
        expected=True,
        observed=False,
        upstream_gate_receipt=_upstream(),
        model_specs=_models(),
        manifest=exp.build_frozen_task_manifest(),
        preconditions={"all_required_preconditions_available": False},
        protected_files=_protected(),
        tests_run=exp.DEFAULT_TESTS_RUN,
        duration_s=1.0,
    )
    mutations = {
        "blocked_status_prefix_missing": {"status": "failed"},
        "blocked_verdict_prefix_missing": {"honest_verdict": "failed"},
        "blocked_verdict_class_mismatch": {"verdict_class": "null"},
        "blocked_corpus_complete": {"candidate_corpus_complete": True},
        "blocked_headroom_nonzero": {"regeneration_headroom_count": 1},
    }
    for expected_error, mutation in mutations.items():
        changed = _rehash({**deepcopy(blocked), **mutation})
        assert expected_error in exp.validate_artifact(changed)


def test_req_6649_upstream_command_and_system_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6649 retains explicit upstream and local system receipts."""

    upstream_path = tmp_path / exp.UPSTREAM_PATH
    upstream_path.parent.mkdir(parents=True)
    upstream_path.write_text("not json", encoding="utf-8")
    receipt = exp.build_upstream_gate_receipt(tmp_path)
    assert receipt["passed"] is False
    assert receipt["observed_value"] is None
    success = exp._command_receipt(
        [str(Path(os.sys.executable)), "-c", "print('ok')"], tmp_path, 10.0
    )
    assert success["exit_code"] == 0
    assert success["summary"] == "ok"
    failure = exp._command_receipt(["/not/a/command"], tmp_path, 0.1)
    assert failure["exit_code"] == 127
    monkeypatch.setattr(
        exp,
        "_command_receipt",
        lambda command, root, timeout: {"command": " ".join(command), "exit_code": 0},
    )
    assert len(exp.run_verification_commands(tmp_path)) == 8
    resources = exp._host_resources(tmp_path)
    assert resources["cpu_count"]
    assert resources["disk_free_bytes"] > 0
    original_read_text = Path.read_text

    def fail_meminfo(path: Path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("unavailable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_meminfo)
    assert exp._host_resources(tmp_path)["ram_bytes"] is None


def test_req_6649_preconditions_and_live_request_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFER-SOTA-6649 checks hardware identities and direct request receipts."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    monkeypatch.setattr(
        exp.canaries,
        "llama_cpp_receipt",
        lambda: {"exists": True, "executable": True, "cuda_linked": True},
    )
    monkeypatch.setattr(
        exp.canaries,
        "gpu_inventory",
        lambda: [{"memory_total_mb": 24576}, {"memory_total_mb": 24576}],
    )
    monkeypatch.setattr(exp, "sha256_file", lambda _path: "sha256:" + "a" * 64)
    monkeypatch.setattr(exp, "_host_resources", lambda _root: {"host": "test"})
    preconditions = exp.collect_preconditions(
        REPO,
        _upstream(),
        models,
        manifest,
        {path.as_posix(): "sha256:" + "b" * 64 for path in exp.PROTECTED_PATHS},
    )
    assert preconditions["all_required_preconditions_available"] is True
    assert exp._free_port() > 0
    command = exp._server_command(models[0], 12345)
    assert command[command.index("--port") + 1] == "12345"
    assert exp._pid_start_ticks(os.getpid()) > 0
    assert exp._pid_start_ticks(999_999_999) == 0
    assert exp._compact_gpu({"index": 0, "uuid": "GPU-x"})["device_uuid"] == "GPU-x"

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return b"{}"

    monkeypatch.setattr(exp.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert exp._http_bytes("http://127.0.0.1/health", None, 1.0) == (200, b"{}")
    task = manifest["tasks"][0]
    response = {
        "choices": [{"message": {"content": task["exact_target"]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    monkeypatch.setattr(
        exp, "_http_bytes", lambda *_args, **_kwargs: (200, json.dumps(response).encode())
    )
    generated = exp._generation_request(12345, task)
    assert generated["generated_tokens"] == 6
    assert generated["raw_output"] == task["exact_target"].encode()
    monkeypatch.setattr(
        exp,
        "_http_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    failed = exp._generation_request(12345, task)
    assert failed["http_status"] == 124
    assert failed["finish_reason"] == "request_failure"


def test_req_6649_run_orchestration_blocked_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6649 owns both blocked and complete terminal orchestration."""

    manifest = exp.build_frozen_task_manifest()
    models = _models()
    writes: list[dict] = []
    monkeypatch.setattr(exp, "protected_hashes", lambda _root: {"p": "sha256:x"})
    monkeypatch.setattr(exp, "build_upstream_gate_receipt", lambda _root: _upstream())
    monkeypatch.setattr(exp, "build_frozen_task_manifest", lambda: manifest)
    monkeypatch.setattr(exp, "resolve_model_specs", lambda: models)
    monkeypatch.setattr(exp, "run_verification_commands", lambda _root: exp.DEFAULT_TESTS_RUN)
    monkeypatch.setattr(exp, "protected_files_receipt", lambda *_args: _protected())
    monkeypatch.setattr(exp, "write_artifact_atomic", lambda _path, payload: writes.append(payload))
    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda *_args: {"all_required_preconditions_available": False, "checks": {"tools": False}},
    )
    blocked = exp.run("20260826", tmp_path)
    assert blocked["status"] == "blocked_tools"
    assert exp._first_failed_precondition(
        {"checks": {}, "all_required_preconditions_available": False}
    ) == (
        "preconditions",
        False,
    )
    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args: _preconditions())

    def session(_root: Path, model: dict, tasks: list[dict]):
        return ([_generation(task["exact_target"].encode()) for task in tasks], _receipt(model))

    monkeypatch.setattr(exp, "_run_model_session", session)
    complete = exp.run("20260826", tmp_path)
    assert complete["status"] == "complete"
    assert len(complete["candidate_rows"]) == 48
    assert writes[-1] == complete


def test_req_6649_cli_validation_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6649 CLI validates artifacts and prints one compact run summary."""

    artifact = _artifact()
    target = tmp_path / "artifact.json"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--validate", str(target)]) == 0
    assert "valid" in capsys.readouterr().out
    target.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", str(target)]) == 1
    assert "missing_required_fields" in capsys.readouterr().out
    monkeypatch.setattr(
        exp,
        "run",
        lambda date: {
            "status": f"complete-{date}",
            "candidate_corpus_complete": True,
            "regeneration_headroom_count": 2,
        },
    )
    assert exp.main(["--date", "20260826"]) == 0
    assert "complete-20260826" in capsys.readouterr().out
