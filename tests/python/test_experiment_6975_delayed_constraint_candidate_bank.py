"""Tests for the delayed-constraint three-schedule candidate bank.

Spec refs: REQ-INF-6975, SCENARIO-INF-6975-GATES,
SCENARIO-INF-6975-ROSTER, SCENARIO-INF-6975-SCHEDULES,
SCENARIO-INF-6975-RAW-FIRST, SCENARIO-INF-6975-ENERGY,
SCENARIO-INF-6975-ATTEMPTS, and SCENARIO-INF-6975-NO-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6975_delayed_constraint_candidate_bank as exp


REPO = Path(__file__).resolve().parents[2]


def _formulation(name: str) -> dict:
    return {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [
            {
                "name": name,
                "kind": "integer",
                "universe": [-1, 0, 1, 2],
                "domain": {"lower": "0", "upper": "1"},
            }
        ],
        "constraints": [{"terms": {name: "1"}, "op": "<=", "rhs": "1"}],
        "objective": {
            "direction": "min",
            "expression": {"kind": "linear", "terms": {name: "1"}, "constant": "0"},
        },
    }


def _fixture() -> dict:
    rows = []
    for split in ("calibration", "heldout", "chronological"):
        for family_index, family in enumerate(exp.FORMULATION_FAMILIES):
            for item_index in range(3):
                pair_id = f"{split[:1]}-{family_index}-{item_index}"
                public = {
                    "record_id": f"{split}:{pair_id}",
                    "split": split,
                    "pair_id": pair_id,
                    "formulation_family": family,
                    "source_formulation": _formulation(f"x{family_index}"),
                    "target_formulation": _formulation(f"y{family_index}"),
                }
                public["prompt_record_hash"] = exp.sha256_json(public)
                rows.append(public)
    return {
        "error_fixture_ready_score": 1,
        "prompt_visible_rows": rows,
        "sealed_label_hashes": [{"pair_id": "secret", "label_hash": "sha256:hidden"}],
        "exact_witness_rows": [{"pair_id": "secret", "witness": {"x": 1}}],
    }


def _specs(tmp_path: Path) -> list[dict]:
    rows = []
    for index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]))
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": "Q4_K_M",
                "resolution_method": "test",
            }
        )
    return rows


def _selected() -> list[dict]:
    return exp.select_balanced_pairs(_fixture())


def _plan(tmp_path: Path) -> dict:
    return exp.freeze_plan(_selected(), model_specs=_specs(tmp_path))


def _phase(
    phase_id: str = "certificate",
    raw_text: str = "{}",
    *,
    attempt_offset: int = 0,
) -> dict:
    raw_bytes = raw_text.encode("utf-8")
    token_ids = list(range(10, 10 + len(raw_bytes)))
    energy_rows = []
    span_rows = []
    for step_index, (token_id, byte_value) in enumerate(zip(token_ids, raw_bytes, strict=True)):
        selected_logit = float(byte_value) / 100.0
        logsumexp = selected_logit + 1.0
        energy_rows.append(
            {
                "phase_id": phase_id,
                "phase_step_index": step_index,
                "attempt_step_index": attempt_offset + step_index,
                "emitted_token_id": token_id,
                "selected_token_logit": selected_logit,
                "full_vocabulary_logsumexp": logsumexp,
                "selected_token_logprob": -1.0,
                "entropy": 0.5,
                "top_probability": 0.6,
                "full_vocabulary_size": 32,
                "full_vocabulary_logits_sha256": "sha256:" + f"{token_id:064x}"[-64:],
            }
        )
        span_rows.append(
            {
                "phase_id": phase_id,
                "phase_step_index": step_index,
                "attempt_step_index": attempt_offset + step_index,
                "emitted_token_id": token_id,
                "phase_byte_start": step_index,
                "phase_byte_end": step_index + 1,
                "attempt_byte_start": attempt_offset + step_index,
                "attempt_byte_end": attempt_offset + step_index + 1,
            }
        )
    return {
        "phase_id": phase_id,
        "raw_text": raw_text,
        "raw_utf8_hex": raw_bytes.hex(),
        "raw_sha256": exp.sha256_bytes(raw_bytes),
        "finish_reason": "eos",
        "token_ids": token_ids,
        "energy_rows": energy_rows,
        "token_span_rows": span_rows,
        "prompt_token_count": 7,
        "live_duration_s": 0.2,
    }


def _raw_result(attempt: dict, raw_text: str = "{}") -> dict:
    phase_id = "direct_certificate" if attempt["schedule_id"] == "direct" else "certificate_tail"
    phase = _phase(phase_id, raw_text)
    return {
        "attempt_key": attempt["attempt_key"],
        "call_status": "complete",
        "failure_reason": None,
        "phase_outputs": [phase],
        "candidate_phase_id": phase_id,
        "candidate_raw_text": raw_text,
        "candidate_raw_sha256": exp.sha256_text(raw_text),
        "finish_reason": "eos",
        "truncated": False,
        "exception_type": None,
        "exception_message": None,
        "live_duration_s": 0.2,
    }


def _complete_rows(plan: dict) -> list[dict]:
    checkpoint = exp.new_checkpoint(
        plan,
        model_file_hashes={model_id: "sha256:" + "a" * 64 for model_id in exp.REQUIRED_MODEL_IDS},
    )
    checkpoint = exp.persist_raw_pair_block(
        checkpoint,
        plan["attempts"],
        [_raw_result(attempt) for attempt in plan["attempts"]],
    )
    checkpoint = exp.finalize_pair_block(
        checkpoint,
        [attempt["attempt_key"] for attempt in plan["attempts"]],
    )
    return checkpoint["attempt_rows"]


def _gpu_rows() -> list[dict]:
    return [
        {
            "hf_id": model_id,
            "process_pid": 100 + index,
            "gpu_indices": [0, 1],
            "gpu_uuids": ["GPU-0", "GPU-1"],
            "used_cuda": True,
        }
        for index, model_id in enumerate(exp.REQUIRED_MODEL_IDS)
    ]


def _teardown_rows() -> list[dict]:
    return [
        {
            "hf_id": model_id,
            "process_exit_code": 0,
            "process_reaped": True,
            "model_closed": True,
        }
        for model_id in exp.REQUIRED_MODEL_IDS
    ]


def _isolation_rows() -> list[dict]:
    return [
        {"check": check, "passed": True}
        for check in (
            "pair_ids_disjoint",
            "held_future_excluded",
            "prompt_authority_terms_absent",
            "other_schedule_outputs_absent",
        )
    ]


def _complete_artifact(tmp_path: Path) -> dict:
    plan = _plan(tmp_path)
    return exp.build_artifact(
        run_date="20260904",
        duration_s=120.0,
        plan=plan,
        attempt_rows=_complete_rows(plan),
        model_specs=_specs(tmp_path),
        model_file_hashes={model_id: "sha256:" + "a" * 64 for model_id in exp.REQUIRED_MODEL_IDS},
        gpu_runtime_rows=_gpu_rows(),
        checkpoint_rows=[{"stage": "family_teardown"}],
        teardown_rows=_teardown_rows(),
        split_isolation_rows=_isolation_rows(),
        preconditions_checked={"all_passed": True, "checks": []},
        source_artifact_hashes={"exp6967": exp.EXPECTED_EXP6967_SHA256},
    )


def test_req_inf_6975_spec_anchors_the_contract() -> None:
    """REQ-INF-6975 exists before the module implementation."""

    text = (REPO / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INF-6975") :]
    for anchor in (
        "SCENARIO-INF-6975-GATES",
        "SCENARIO-INF-6975-ROSTER",
        "SCENARIO-INF-6975-SCHEDULES",
        "SCENARIO-INF-6975-RAW-FIRST",
        "SCENARIO-INF-6975-ENERGY",
        "SCENARIO-INF-6975-ATTEMPTS",
        "SCENARIO-INF-6975-NO-POLICY",
        "expected_attempt_count",
        "candidate_bank_complete_score",
    ):
        assert anchor in section


def test_req_inf_6975_model_specs_use_cached_pair_and_exact_ids(tmp_path: Path) -> None:
    """REQ-INF-6975 excludes legacy models from the headline roster."""

    expected = _specs(tmp_path)
    calls: list[dict] = []

    def pair(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [{**expected[0], "gpu": 0}, {**expected[2], "gpu": 1}]

    rows = exp.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda model_id, _quant: (
            expected[1]["model_path"] if model_id == exp.REQUIRED_MODEL_IDS[1] else None
        ),
    )
    assert calls == [{"gpu_indices": (0, 1)}]
    assert [row["hf_id"] for row in rows] == list(exp.REQUIRED_MODEL_IDS)
    assert exp.model_spec_errors(rows) == []

    changed = deepcopy(rows)
    changed[0]["hf_id"] = "legacy/smoke-GGUF"
    changed[1]["model_path"] = str(tmp_path / "mmproj.gguf")
    changed[2]["gpu_indices"] = [0]
    changed[2]["headline_eligible"] = False
    errors = exp.model_spec_errors(changed)
    assert "model_ids_mismatch" in errors
    assert any(error.startswith("model_path_not_primary_gguf:") for error in errors)
    assert any(error.startswith("dual_gpu_indices_missing:") for error in errors)
    assert any(error.startswith("headline_eligibility_missing:") for error in errors)


def test_scenario_inf_6975_roster_is_balanced_deterministic_and_public() -> None:
    """SCENARIO-INF-6975-ROSTER selects two public rows per family and split."""

    selected = _selected()
    assert selected == exp.select_balanced_pairs(_fixture())
    assert len(selected) == 12
    assert {row["split"] for row in selected} == {"calibration", "heldout"}
    assert len({row["pair_id"] for row in selected}) == 12
    for split in exp.SELECTED_SPLITS:
        for family in exp.FORMULATION_FAMILIES:
            assert (
                sum(
                    row["split"] == split and row["formulation_family"] == family
                    for row in selected
                )
                == 2
            )
    serialized = exp.canonical_json(selected)
    assert "witness" not in serialized
    assert "label" not in serialized
    assert "chronological" not in serialized


def test_scenario_inf_6975_roster_fails_when_a_cell_is_too_small() -> None:
    """SCENARIO-INF-6975-ROSTER does not fill a missing family from another split."""

    fixture = _fixture()
    fixture["prompt_visible_rows"] = [
        row
        for row in fixture["prompt_visible_rows"]
        if not (
            row["split"] == "heldout"
            and row["formulation_family"] == exp.FORMULATION_FAMILIES[0]
            and row["pair_id"].endswith("-2")
        )
    ]
    fixture["prompt_visible_rows"] = [
        row
        for row in fixture["prompt_visible_rows"]
        if not (
            row["split"] == "heldout"
            and row["formulation_family"] == exp.FORMULATION_FAMILIES[0]
            and row["pair_id"].endswith("-1")
        )
    ]
    with pytest.raises(exp.CandidateBankError, match="eligible_pair_cell_too_small"):
        exp.select_balanced_pairs(fixture)


def test_scenario_inf_6975_schedules_and_attempt_roster_are_hash_bound(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-SCHEDULES freezes three distinct mechanisms and 108 keys."""

    plan = _plan(tmp_path)
    assert len(plan["selected_pair_rows"]) == 12
    assert len(plan["attempts"]) == exp.EXPECTED_ATTEMPT_COUNT == 108
    assert len({row["attempt_key"] for row in plan["attempts"]}) == 108
    assert plan["schedule_hashes"] == {
        row["schedule_id"]: exp.sha256_text(row["schedule_text"]) for row in exp.SCHEDULE_ROWS
    }
    assert plan["split_hash"] == exp.compute_split_hash(plan["selected_pair_rows"])
    assert plan["decoding_settings"] == {
        "temperature": 0.35,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
        "completion_token_cap": 128,
        "planning_token_cap": 48,
        "certificate_token_cap": 80,
    }
    assert {row["schedule_id"] for row in plan["attempts"]} == {
        "direct",
        "trigger_switched",
        "draft_conditioned",
    }
    assert all(exp.audit_prompt(row["prompt"]) == [] for row in plan["attempts"])


def test_scenario_inf_6975_generation_prompts_reject_authority_and_cross_arm_leaks() -> None:
    """REQ-INF-6975 keeps labels, solvers, held-future rows, and other outputs hidden."""

    for forbidden in (
        "expected_label",
        "exact witness",
        "solver outcome",
        "chronological",
        "another schedule's result",
        "candidate answer id",
    ):
        errors = exp.audit_prompt(f"public input then {forbidden}")
        assert errors


def test_scenario_inf_6975_raw_capture_precedes_parser_and_never_repairs(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-RAW-FIRST preserves malformed bytes before parsing."""

    plan = _plan(tmp_path)
    checkpoint = exp.new_checkpoint(plan, model_file_hashes={"m": "sha256:x"})
    attempt = plan["attempts"][0]
    raw = _raw_result(attempt, "```json\n{bad}\n```")
    durable = exp.persist_raw_pair_block(checkpoint, [attempt], [raw])
    row = durable["attempt_rows"][0]
    assert row["raw_durable"] is True
    assert row["terminal"] is False
    assert "parser_diagnostic" not in row
    assert row["candidate_raw_text"] == "```json\n{bad}\n```"

    terminal = exp.finalize_pair_block(durable, [attempt["attempt_key"]])
    parsed = terminal["attempt_rows"][0]
    assert parsed["terminal"] is True
    assert parsed["parser_diagnostic"]["json_valid"] is False
    assert parsed["parser_diagnostic"]["syntax_reason"] == "malformed_json"
    assert parsed["candidate_raw_text"] == "```json\n{bad}\n```"
    assert parsed["raw_durable_sequence"] < parsed["parse_sequence"]


def test_scenario_inf_6975_checkpoint_rejects_plan_and_content_drift(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-RAW-FIRST binds recovery to the full frozen plan."""

    plan = _plan(tmp_path)
    checkpoint = exp.new_checkpoint(plan, model_file_hashes={"m": "sha256:x"})
    path = tmp_path / "checkpoint.json"
    exp.write_checkpoint(path, checkpoint)
    assert exp.load_checkpoint(path, plan=plan, model_file_hashes={"m": "sha256:x"}) == checkpoint

    changed = deepcopy(plan)
    changed["split_hash"] = "sha256:" + "0" * 64
    with pytest.raises(exp.CandidateBankError, match="checkpoint_plan_drift"):
        exp.load_checkpoint(path, plan=changed, model_file_hashes={"m": "sha256:x"})
    with pytest.raises(exp.CandidateBankError, match="checkpoint_model_hash_drift"):
        exp.load_checkpoint(path, plan=plan, model_file_hashes={"m": "sha256:y"})

    document = json.loads(path.read_text(encoding="utf-8"))
    document["checkpoint_sha256"] = "sha256:" + "f" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.CandidateBankError, match="checkpoint_checksum_mismatch"):
        exp.load_checkpoint(path, plan=plan, model_file_hashes={"m": "sha256:x"})


def test_scenario_inf_6975_energy_statistics_capture_full_distribution_scalars() -> None:
    """SCENARIO-INF-6975-ENERGY stores sufficient scalars, not the full vector."""

    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    row = exp.energy_statistics(logits, emitted_token_id=1)
    expected_lse = math.log(sum(math.exp(value) for value in logits))
    probabilities = np.exp(logits - expected_lse)
    expected_entropy = -float(np.sum(probabilities * np.log(probabilities)))
    assert row["emitted_token_id"] == 1
    assert row["selected_token_logit"] == 1.0
    assert row["full_vocabulary_logsumexp"] == pytest.approx(expected_lse)
    assert row["selected_token_logprob"] == pytest.approx(1.0 - expected_lse)
    assert row["entropy"] == pytest.approx(expected_entropy)
    assert row["top_probability"] == pytest.approx(float(max(probabilities)))
    assert row["full_vocabulary_size"] == 3
    assert row["full_vocabulary_logits_sha256"].startswith("sha256:")
    assert "logits" not in row
    with pytest.raises(exp.CandidateBankError, match="emitted_token_id_out_of_range"):
        exp.energy_statistics(logits, emitted_token_id=9)
    with pytest.raises(exp.CandidateBankError, match="no_finite_logits"):
        exp.energy_statistics(np.asarray([np.nan, -np.inf]), emitted_token_id=0)


def test_scenario_inf_6975_parser_reports_syntax_only() -> None:
    """SCENARIO-INF-6975-NO-POLICY never turns JSON shape into semantic success."""

    valid = exp.parse_syntax(
        json.dumps(
            {
                "schema_version": "carnot.constraint_ir.mapping.v1",
                "variable_map": [],
                "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
            }
        )
    )
    assert valid == {
        "json_valid": True,
        "object_valid": True,
        "constraintir_shape_valid": True,
        "syntax_reason": None,
    }
    assert exp.parse_syntax("")["syntax_reason"] == "empty_output"
    assert exp.parse_syntax("[]")["syntax_reason"] == "json_object_required"
    assert exp.parse_syntax("{}")["syntax_reason"] == "constraintir_keys"
    assert not any(
        term in exp.canonical_json(valid)
        for term in ("semantic", "certified", "correct", "solver", "selected_schedule")
    )


def test_scenario_inf_6975_attempt_gate_requires_exact_rows_energy_and_cuda(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-ATTEMPTS rejects duplicates, gaps, drift, and CPU rows."""

    plan = _plan(tmp_path)
    rows = _complete_rows(plan)
    assert (
        exp.completion_errors(
            plan=plan,
            attempt_rows=rows,
            gpu_runtime_rows=_gpu_rows(),
            schedule_rows=exp.SCHEDULE_ROWS,
            split_isolation_rows=_isolation_rows(),
            teardown_rows=_teardown_rows(),
        )
        == []
    )

    duplicate = rows + [deepcopy(rows[0])]
    assert "attempt_count_mismatch" in exp.completion_errors(
        plan=plan,
        attempt_rows=duplicate,
        gpu_runtime_rows=_gpu_rows(),
        schedule_rows=exp.SCHEDULE_ROWS,
        split_isolation_rows=_isolation_rows(),
        teardown_rows=_teardown_rows(),
    )
    broken_energy = deepcopy(rows)
    broken_energy[0]["energy_trace_rows"][0]["entropy"] = None
    assert any(
        error.startswith("energy_trace_insufficient:")
        for error in exp.completion_errors(
            plan=plan,
            attempt_rows=broken_energy,
            gpu_runtime_rows=_gpu_rows(),
            schedule_rows=exp.SCHEDULE_ROWS,
            split_isolation_rows=_isolation_rows(),
            teardown_rows=_teardown_rows(),
        )
    )
    cpu = _gpu_rows()
    cpu[0]["used_cuda"] = False
    assert "cuda_model_roster_mismatch" in exp.completion_errors(
        plan=plan,
        attempt_rows=rows,
        gpu_runtime_rows=cpu,
        schedule_rows=exp.SCHEDULE_ROWS,
        split_isolation_rows=_isolation_rows(),
        teardown_rows=_teardown_rows(),
    )


def test_scenario_inf_6975_attempt_gate_accepts_empty_exception_rows(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-ATTEMPTS keeps terminal failures without inventing tokens."""

    plan = _plan(tmp_path)
    rows = _complete_rows(plan)
    row = rows[0]
    row.update(
        {
            "call_status": "exception",
            "failure_reason": "RuntimeError: boom",
            "phase_outputs": [],
            "candidate_phase_id": None,
            "candidate_raw_text": "",
            "candidate_raw_sha256": exp.sha256_text(""),
            "energy_trace_rows": [],
            "token_span_rows": [],
            "parser_diagnostic": exp.parse_syntax(""),
        }
    )
    assert not any(
        error.startswith("energy_trace_insufficient:")
        for error in exp.completion_errors(
            plan=plan,
            attempt_rows=rows,
            gpu_runtime_rows=_gpu_rows(),
            schedule_rows=exp.SCHEDULE_ROWS,
            split_isolation_rows=_isolation_rows(),
            teardown_rows=_teardown_rows(),
        )
    )


def test_req_inf_6975_artifact_has_required_projections_and_bare_gate(tmp_path: Path) -> None:
    """REQ-INF-6975 emits each required field and derives the bare completion score."""

    plan = _plan(tmp_path)
    rows = _complete_rows(plan)
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=120.0,
        plan=plan,
        attempt_rows=rows,
        model_specs=_specs(tmp_path),
        model_file_hashes={model_id: "sha256:" + "a" * 64 for model_id in exp.REQUIRED_MODEL_IDS},
        gpu_runtime_rows=_gpu_rows(),
        checkpoint_rows=[{"stage": "family_teardown"}],
        teardown_rows=_teardown_rows(),
        split_isolation_rows=_isolation_rows(),
        preconditions_checked={"all_passed": True, "checks": []},
        source_artifact_hashes={"exp6967": exp.EXPECTED_EXP6967_SHA256},
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert type(artifact["candidate_bank_complete_score"]) is int
    assert artifact["candidate_bank_complete_score"] == 1
    assert artifact["expected_attempt_count"] == artifact["observed_attempt_count"] == 108
    assert artifact["models_used"] == list(exp.REQUIRED_MODEL_IDS)
    assert len(artifact["raw_output_rows"]) == 108
    assert len(artifact["parser_diagnostic_rows"]) == 108
    assert artifact["energy_trace_rows"]
    assert artifact["token_span_rows"]
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["schedule_hashes"]["direct"] = "sha256:" + "0" * 64
    assert "schedule_hashes_mismatch" in exp.validate_artifact(changed)


def test_scenario_inf_6975_gates_emit_full_blocked_schema_without_acquisition(
    tmp_path: Path,
) -> None:
    """SCENARIO-INF-6975-GATES starts no live work after one failed bare gate."""

    checks = [exp.gate_check("lease_aware_runtime_ready_score", 1, 0)]
    artifact = exp.blocked_artifact(
        run_date="20260904",
        duration_s=0.1,
        checks=checks,
        model_specs=_specs(tmp_path),
        source_artifact_hashes={"exp6973": "sha256:x"},
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["candidate_bank_complete_score"] == 0
    assert artifact["observed_attempt_count"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_delayed_constraint_candidate_bank"
    assert artifact["gate_check_summary"] == {
        "failed_check": "lease_aware_runtime_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "checks": checks,
        "passed": False,
    }
    assert exp.validate_artifact(artifact) == []

    calls: list[str] = []
    output = tmp_path / "blocked.json"
    result = exp.run(
        run_date="20260904",
        result_path=output,
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=_specs(tmp_path),
        preflight_fn=lambda _specs, _path: {
            "all_passed": False,
            "checks": checks,
            "source_artifact_hashes": {"exp6973": "sha256:x"},
            "model_file_hashes": {},
            "selected_pair_rows": [],
            "split_isolation_rows": [],
            "gpu_topology": {"devices": []},
        },
        acquisition_fn=lambda **_kwargs: calls.append("called"),
    )
    assert calls == []
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_req_inf_6975_preconditions_check_every_required_resource(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-GATES checks upstreams, pairs, models, CUDA, and storage."""

    specs = _specs(tmp_path)
    upstream = {
        "exp6973": {"lease_aware_runtime_ready_score": 1, "model_file_hashes": {}},
        "exp6974": {"fixture_admissibility_ready_score": 1},
        "exp6967": _fixture(),
    }
    paths = {
        "exp6973": tmp_path / "6973.json",
        "exp6974": tmp_path / "6974.json",
        "exp6967": tmp_path / "6967.json",
    }
    for key, path in paths.items():
        path.write_text(json.dumps(upstream[key]), encoding="utf-8")

    preflight = exp.collect_preconditions(
        model_specs=specs,
        checkpoint_path=tmp_path / "checkpoint.json",
        upstream_paths=paths,
        json_reader=lambda path: upstream[
            next(key for key, value in paths.items() if value == path)
        ],
        file_hasher=lambda path: (
            exp.EXPECTED_EXP6967_SHA256
            if path == paths["exp6967"]
            else "sha256:" + path.name.encode().hex().ljust(64, "0")[:64]
        ),
        gpu_probe=lambda: {
            "query_ok": True,
            "devices": [
                {"index": 0, "uuid": "GPU-0", "name": "NVIDIA RTX 3090"},
                {"index": 1, "uuid": "GPU-1", "name": "NVIDIA RTX 3090"},
            ],
        },
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        writable_probe=lambda _path: True,
        model_hasher=lambda path: "sha256:" + path.name.encode().hex().ljust(64, "0")[:64],
    )
    assert preflight["all_passed"] is True
    assert len(preflight["selected_pair_rows"]) == 12
    assert {row["check"] for row in preflight["checks"]} == {
        "lease_aware_runtime_ready_score",
        "fixture_admissibility_ready_score",
        "exact_exp6967_source_hash",
        "eligible_calibration_pair_count_at_least",
        "eligible_heldout_pair_count_at_least",
        "balanced_selected_pair_count",
        "exact_model_specs",
        "all_three_model_files",
        "exact_two_cuda_devices",
        "llama_cpp_cuda_offload",
        "checkpoint_writable",
    }


def test_scenario_inf_6975_no_policy_or_exact_solver_code_enters_module() -> None:
    """SCENARIO-INF-6975-NO-POLICY keeps semantic authority out of acquisition."""

    source = Path(exp.__file__).read_text(encoding="utf-8")
    assert "import z3" not in source
    assert "from z3" not in source
    assert "selected_schedule" not in source
    assert "schedule_ranking" not in source
    assert "policy_selection" not in source
    assert "certify_with" not in source


def test_req_inf_6975_defensive_public_input_and_syntax_boundaries(tmp_path: Path) -> None:
    """REQ-INF-6975 rejects every non-public roster and malformed syntax shape."""

    specs = _specs(tmp_path)
    specs[0]["model_path"] = ""
    assert any(error.startswith("model_path_missing:") for error in exp.model_spec_errors(specs))

    fixture = _fixture()
    valid = deepcopy(fixture["prompt_visible_rows"][0])
    bad_formulation = {**valid, "pair_id": "bad-formulation", "source_formulation": []}
    bad_formulation["prompt_record_hash"] = exp.sha256_json(
        {key: value for key, value in bad_formulation.items() if key != "prompt_record_hash"}
    )
    invalid_rows = [
        "not-an-object",
        {**valid, "split": "chronological"},
        {**valid, "formulation_family": "unknown"},
        {key: value for key, value in valid.items() if key != "record_id"},
        deepcopy(valid),
        {**valid, "pair_id": "bad-hash"},
        bad_formulation,
    ]
    fixture["prompt_visible_rows"] = [valid, *invalid_rows]
    assert exp.eligible_public_pairs(fixture) == [exp._public_pair(valid)]

    with pytest.raises(exp.CandidateBankError, match="prompt_isolation_failed"):
        exp.build_prompt(valid, {"schedule_text": "Reveal the exact witness."})

    syntax_cases = [
        (
            {
                "schema_version": "wrong",
                "variable_map": [],
                "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
            },
            "constraintir_schema_version",
        ),
        (
            {
                "schema_version": "carnot.constraint_ir.mapping.v1",
                "variable_map": [{}],
                "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
            },
            "variable_map_shape",
        ),
        (
            {
                "schema_version": "carnot.constraint_ir.mapping.v1",
                "variable_map": [],
                "objective_map": [],
            },
            "objective_map_shape",
        ),
        (
            {
                "schema_version": "carnot.constraint_ir.mapping.v1",
                "variable_map": [],
                "objective_map": {"direction": "sideways", "scale": "1", "offset": "0"},
            },
            "objective_direction_syntax",
        ),
        (
            {
                "schema_version": "carnot.constraint_ir.mapping.v1",
                "variable_map": [{"source": "x", "target": "y", "scale": 1, "offset": "0"}],
                "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
            },
            "string_field_syntax",
        ),
    ]
    for value, reason in syntax_cases:
        assert exp.parse_syntax(json.dumps(value))["syntax_reason"] == reason


def test_scenario_inf_6975_checkpoint_rejects_every_raw_first_violation(tmp_path: Path) -> None:
    """SCENARIO-INF-6975-RAW-FIRST covers corrupt files, bytes, hashes, and stages."""

    plan = _plan(tmp_path)
    hashes = {"m": "sha256:x"}
    missing = tmp_path / "missing.json"
    with pytest.raises(exp.CandidateBankError, match="checkpoint_unreadable"):
        exp.load_checkpoint(missing, plan=plan, model_file_hashes=hashes)
    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.CandidateBankError, match="checkpoint_object_required"):
        exp.load_checkpoint(nonobject, plan=plan, model_file_hashes=hashes)

    checkpoint = exp.new_checkpoint(plan, model_file_hashes=hashes)
    attempt = plan["attempts"][0]
    with pytest.raises(exp.CandidateBankError, match="pair_block_result_count_mismatch"):
        exp.persist_raw_pair_block(checkpoint, [attempt], [])

    mutations = [
        (
            "raw_phase_hex_invalid",
            lambda row: row["phase_outputs"][0].__setitem__("raw_utf8_hex", "x"),
        ),
        (
            "raw_phase_text_mismatch",
            lambda row: row["phase_outputs"][0].__setitem__("raw_text", "x"),
        ),
        (
            "raw_phase_hash_mismatch",
            lambda row: row["phase_outputs"][0].__setitem__("raw_sha256", "sha256:bad"),
        ),
        (
            "candidate_raw_hash_mismatch",
            lambda row: row.__setitem__("candidate_raw_sha256", "sha256:bad"),
        ),
    ]
    for reason, mutate in mutations:
        result = _raw_result(attempt)
        mutate(result)
        with pytest.raises(exp.CandidateBankError, match=reason):
            exp.persist_raw_pair_block(checkpoint, [attempt], [result])

    wrong_key = _raw_result(attempt)
    wrong_key["attempt_key"] = "wrong"
    with pytest.raises(exp.CandidateBankError, match="raw_attempt_key_mismatch"):
        exp.persist_raw_pair_block(checkpoint, [attempt], [wrong_key])

    two_attempts = plan["attempts"][:2]
    durable = exp.persist_raw_pair_block(
        checkpoint,
        two_attempts,
        [_raw_result(attempt_row) for attempt_row in two_attempts],
    )
    with pytest.raises(exp.CandidateBankError, match="duplicate_raw_attempt"):
        exp.persist_raw_pair_block(durable, [attempt], [_raw_result(attempt)])
    terminal_one = exp.finalize_pair_block(durable, [two_attempts[0]["attempt_key"]])
    with pytest.raises(exp.CandidateBankError, match="raw_first_state_invalid"):
        exp.finalize_pair_block(terminal_one, [two_attempts[0]["attempt_key"]])
    with pytest.raises(exp.CandidateBankError, match="pair_block_attempt_missing"):
        exp.finalize_pair_block(durable, ["not-present"])


def test_scenario_inf_6975_energy_and_completion_reject_all_insufficient_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-INF-6975-ENERGY checks every scalar, token, hash, and span invariant."""

    plan = _plan(tmp_path)
    baseline = _complete_rows(plan)[0]
    mutations = [
        lambda row: row.__setitem__("phase_outputs", {}),
        lambda row: row["token_span_rows"].pop(),
        lambda row: row["energy_trace_rows"][0].__setitem__("attempt_step_index", 4),
        lambda row: row["token_span_rows"][0].__setitem__("attempt_step_index", 4),
        lambda row: row["energy_trace_rows"][0].__setitem__("emitted_token_id", 999),
        lambda row: row["energy_trace_rows"][0].__setitem__("entropy", True),
        lambda row: row["energy_trace_rows"][0].__setitem__("top_probability", 2.0),
        lambda row: row["energy_trace_rows"][0].__setitem__("full_vocabulary_size", 0),
        lambda row: row["energy_trace_rows"][0].__setitem__("full_vocabulary_logits_sha256", "bad"),
        lambda row: row["energy_trace_rows"][0].__setitem__("logits", []),
        lambda row: row["token_span_rows"][0].__setitem__("phase_byte_start", 0.5),
        lambda row: row["token_span_rows"][0].update({"phase_byte_start": 2, "phase_byte_end": 1}),
        lambda row: row["token_span_rows"][0].update(
            {"attempt_byte_start": 2, "attempt_byte_end": 1}
        ),
        lambda row: row["phase_outputs"][0].__setitem__("raw_utf8_hex", "x"),
        lambda row: row["phase_outputs"][0].__setitem__("raw_sha256", "bad"),
        lambda row: row["phase_outputs"][0]["token_span_rows"][-1].__setitem__(
            "phase_byte_end", 99
        ),
    ]
    for mutate in mutations:
        changed = deepcopy(baseline)
        mutate(changed)
        assert exp._energy_trace_sufficient(changed) is False

    rows = _complete_rows(plan)
    completion_mutations = [
        lambda changed: changed[0].__setitem__("terminal", False),
        lambda changed: changed[0].__setitem__("raw_durable", False),
        lambda changed: changed[0].__setitem__(
            "parse_sequence", changed[0]["raw_durable_sequence"]
        ),
        lambda changed: changed[0].__setitem__("candidate_raw_sha256", "bad"),
    ]
    expected = [
        "nonterminal_attempt",
        "raw_not_durable:",
        "parse_preceded_raw:",
        "candidate_raw_hash_mismatch:",
    ]
    for mutate, marker in zip(completion_mutations, expected, strict=True):
        changed = deepcopy(rows)
        mutate(changed)
        assert any(
            marker in error
            for error in exp.completion_errors(
                plan=plan,
                attempt_rows=changed,
                gpu_runtime_rows=_gpu_rows(),
                schedule_rows=exp.SCHEDULE_ROWS,
                split_isolation_rows=_isolation_rows(),
                teardown_rows=_teardown_rows(),
            )
        )
    changed_schedules = deepcopy(exp.SCHEDULE_ROWS)
    changed_schedules[0]["schedule_text"] = "changed"
    assert "schedule_rows_mismatch" in exp.completion_errors(
        plan=plan,
        attempt_rows=rows,
        gpu_runtime_rows=_gpu_rows(),
        schedule_rows=changed_schedules,
        split_isolation_rows=_isolation_rows(),
        teardown_rows=_teardown_rows(),
    )


def test_req_inf_6975_validator_recomputes_every_required_projection(tmp_path: Path) -> None:
    """REQ-INF-6975 validates identity, projections, outcomes, and its stable checksum."""

    artifact = _complete_artifact(tmp_path)
    cases = [
        ("missing_field:rows", lambda row: row.pop("rows")),
        ("field_principles_mismatch", lambda row: row["field_principles"].pop("rows")),
        ("identity_mismatch", lambda row: row.__setitem__("schema", "wrong")),
        ("run_date_mismatch", lambda row: row.__setitem__("run_date", "20260905")),
        ("inference_substrate_mismatch", lambda row: row.__setitem__("inference_substrate", "cpu")),
        (
            "expected_attempt_count_mismatch",
            lambda row: row.__setitem__("expected_attempt_count", 1),
        ),
        (
            "candidate_bank_complete_score_not_bare_int",
            lambda row: row.__setitem__("candidate_bank_complete_score", True),
        ),
        (
            "observed_attempt_count_mismatch",
            lambda row: row.__setitem__("observed_attempt_count", 1),
        ),
        ("rows_projection_mismatch", lambda row: row.__setitem__("rows", [])),
        ("raw_output_rows_projection_mismatch", lambda row: row.__setitem__("raw_output_rows", [])),
        (
            "parser_diagnostic_rows_projection_mismatch",
            lambda row: row.__setitem__("parser_diagnostic_rows", []),
        ),
        (
            "energy_trace_rows_projection_mismatch",
            lambda row: row.__setitem__("energy_trace_rows", []),
        ),
        ("token_span_rows_projection_mismatch", lambda row: row.__setitem__("token_span_rows", [])),
        ("split_hash_mismatch", lambda row: row.__setitem__("split_hash", "bad")),
        (
            "candidate_bank_complete_score_mismatch",
            lambda row: row.__setitem__("candidate_bank_complete_score", 0),
        ),
        ("models_used_mismatch", lambda row: row.__setitem__("models_used", [])),
        ("positive_verdict_mismatch", lambda row: row.__setitem__("honest_verdict", "wrong")),
        ("verifier_is_oracle_mismatch", lambda row: row.__setitem__("verifier_is_oracle", True)),
        ("verdict_class_invalid", lambda row: row.__setitem__("verdict_class", "unknown")),
        (
            "reproducibility_checksum_mismatch",
            lambda row: row.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]
    for marker, mutate in cases:
        changed = deepcopy(artifact)
        mutate(changed)
        assert marker in exp.validate_artifact(changed)

    partial = deepcopy(artifact)
    partial["per_attempt_rows"][0]["terminal"] = False
    partial = exp.build_artifact(
        run_date="20260904",
        duration_s=1.0,
        plan=_plan(tmp_path),
        attempt_rows=partial["per_attempt_rows"],
        model_specs=_specs(tmp_path),
        model_file_hashes=partial["model_file_hashes"],
        gpu_runtime_rows=_gpu_rows(),
        checkpoint_rows=[],
        teardown_rows=_teardown_rows(),
        split_isolation_rows=_isolation_rows(),
        preconditions_checked={"all_passed": True, "checks": []},
        source_artifact_hashes={},
    )
    partial["verdict_class"] = "positive"
    partial["reproducibility_checksum"] = exp.reproducibility_checksum(partial)
    assert "partial_verdict_mismatch" in exp.validate_artifact(partial)

    blocked = exp.blocked_artifact(
        run_date="20260904",
        duration_s=0.0,
        checks=[exp.gate_check("x", 1, 0)],
        model_specs=_specs(tmp_path),
        source_artifact_hashes={},
    )
    blocked_cases = [
        ("blocked_score_mismatch", lambda row: row.__setitem__("candidate_bank_complete_score", 1)),
        ("blocked_verdict_mismatch", lambda row: row.__setitem__("verdict_class", "partial")),
        ("blocked_gate_summary_incomplete", lambda row: row.__setitem__("gate_check_summary", {})),
    ]
    for marker, mutate in blocked_cases:
        changed = deepcopy(blocked)
        mutate(changed)
        assert marker in exp.validate_artifact(changed)


def test_req_inf_6975_preflight_and_run_cover_terminal_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INF-6975-GATES fails closed and the authorized path writes once."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(exp.CandidateBankError, match="required_json_unreadable"):
        exp._read_json(malformed)
    nonobject = tmp_path / "list.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.CandidateBankError, match="required_json_object_expected"):
        exp._read_json(nonobject)
    valid = tmp_path / "object.json"
    valid.write_text('{"ok": true}', encoding="utf-8")
    assert exp._read_json(valid) == {"ok": True}

    specs = _specs(tmp_path)
    paths = {name: tmp_path / f"{name}.json" for name in ("exp6973", "exp6974", "exp6967")}

    def broken_reader(path: Path) -> dict:
        if path == paths["exp6973"]:
            raise exp.CandidateBankError("broken")
        if path == paths["exp6974"]:
            return {"fixture_admissibility_ready_score": 1}
        return {"prompt_visible_rows": []}

    failed = exp.collect_preconditions(
        model_specs=specs,
        checkpoint_path=tmp_path / "checkpoint.json",
        upstream_paths=paths,
        json_reader=broken_reader,
        file_hasher=lambda _path: "sha256:bad",
        gpu_probe=lambda: {"query_ok": False, "devices": []},
        llama_probe=lambda: {"importable": False, "gpu_offload": False},
        writable_probe=lambda _path: False,
    )
    assert failed["all_passed"] is False
    assert any(row["check"] == "balanced_selected_pair_count" for row in failed["checks"])

    fixture = _fixture()
    documents = {
        exp.EXP6973_PATH: {
            "lease_aware_runtime_ready_score": 1,
            "model_file_hashes": {model_id: "sha256:model" for model_id in exp.REQUIRED_MODEL_IDS},
        },
        exp.EXP6974_PATH: {"fixture_admissibility_ready_score": 1},
        exp.EXP6967_PATH: fixture,
    }
    monkeypatch.setattr(exp, "_source_hashes", lambda: {"source": "sha256:test"})
    default_paths = exp.collect_preconditions(
        model_specs=specs,
        checkpoint_path=tmp_path / "checkpoint.json",
        json_reader=lambda path: documents[path],
        file_hasher=lambda path: (
            exp.EXPECTED_EXP6967_SHA256 if path == exp.EXP6967_PATH else "sha256:test"
        ),
        gpu_probe=lambda: {
            "query_ok": True,
            "devices": [{"uuid": "GPU-0"}, {"uuid": "GPU-1"}],
        },
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        writable_probe=lambda _path: True,
    )
    assert default_paths["all_passed"] is True
    assert default_paths["source_artifact_hashes"] == {"source": "sha256:test"}
    assert default_paths["model_file_hashes"] == documents[exp.EXP6973_PATH]["model_file_hashes"]

    assert exp._tail_prompt({"prompt": "p"}, "DRAFT", "d").endswith(
        "Now emit only the ConstraintIR certificate JSON object."
    )
    failure = exp._failure_result({"attempt_key": "a"}, TimeoutError("late"))
    assert failure["call_status"] == "timeout"

    def passed_preflight(_specs: list[dict], _path: Path) -> dict:
        return {
            "all_passed": True,
            "checks": [],
            "source_artifact_hashes": {"source": "sha256:test"},
            "model_file_hashes": {model_id: "sha256:model" for model_id in exp.REQUIRED_MODEL_IDS},
            "selected_pair_rows": _selected(),
            "split_isolation_rows": _isolation_rows(),
        }

    def acquire(**kwargs: object) -> dict:
        plan = kwargs["plan"]
        assert isinstance(plan, dict)
        return {
            "attempt_rows": _complete_rows(plan),
            "gpu_runtime_rows": _gpu_rows(),
            "checkpoint_rows": [],
            "teardown_rows": _teardown_rows(),
        }

    result_path = tmp_path / "complete.json"
    result = exp.run(
        result_path=result_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=specs,
        preflight_fn=passed_preflight,
        acquisition_fn=acquire,
    )
    assert result["candidate_bank_complete_score"] == 1
    assert json.loads(result_path.read_text(encoding="utf-8")) == result

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(exp.CandidateBankError, match="blocked_artifact_validation_failed"):
        exp.run(
            result_path=tmp_path / "never-blocked.json",
            checkpoint_path=tmp_path / "checkpoint.json",
            model_specs=specs,
            preflight_fn=lambda _specs, _path: {
                "all_passed": False,
                "checks": [exp.gate_check("x", 1, 0)],
                "source_artifact_hashes": {},
            },
        )
    with pytest.raises(exp.CandidateBankError, match="artifact_validation_failed"):
        exp.run(
            result_path=tmp_path / "never-complete.json",
            checkpoint_path=tmp_path / "checkpoint.json",
            model_specs=specs,
            preflight_fn=passed_preflight,
            acquisition_fn=acquire,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validate)

    monkeypatch.setattr(exp, "freeze_plan", lambda *_args, **_kwargs: {"attempts": []})
    with pytest.raises(exp.CandidateBankError, match="attempt_budget_mismatch"):
        exp.run(
            result_path=tmp_path / "never-budget.json",
            checkpoint_path=tmp_path / "checkpoint.json",
            model_specs=specs,
            preflight_fn=passed_preflight,
            acquisition_fn=acquire,
        )
