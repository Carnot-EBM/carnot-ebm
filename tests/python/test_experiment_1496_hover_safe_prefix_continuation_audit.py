"""Tests for Exp 1496 HoVer safe-prefix continuation audit.

Spec: REQ-VERIFY-1496, SCENARIO-VERIFY-1496.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import cctu_trigger_certificate_export as certificates
from carnot.eval import hover_safe_prefix_continuation_audit as exp


def test_req_verify_1496_selects_trigger_boundary_as_last_safe_prefix() -> None:
    """REQ-VERIFY-1496: interrupted trigger rows keep only the safe trigger boundary."""

    case = certificates.cctu.build_benchmark_cases()[0]
    bad_output = _trigger_output(case, final_answer="not 45")
    source_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": bad_output,
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )
    monitor_event = {
        "event_id": "interwhen-1495-test",
        "case_id": case.case_id,
        "lane": certificates.TRIGGER_LANE,
        "token_offset": 128,
        "polling_interval_tokens": 64,
        "interruption_triggered": True,
        "error_detected": True,
    }

    selection = exp.select_last_safe_prefix(source_row, [monitor_event])

    assert selection["case_id"] == case.case_id
    assert selection["selected_event_id"] == "interwhen-1495-test"
    assert selection["last_safe_token_offset"] == 64
    assert selection["safe_prefix"].endswith(certificates.TRIGGER_TOKEN + "\n")
    assert "not 45" not in selection["safe_prefix"]
    assert selection["selection_reason"] == "monitor_trigger_boundary_before_unsafe_certificate"


def test_req_verify_1496_selection_edge_cases_are_deterministic() -> None:
    """REQ-VERIFY-1496: fallback prefix boundaries do not depend on model judgement."""

    case_id = "cctu-1486-arith-001"
    event = {"event_id": "e", "token_offset": 10, "polling_interval_tokens": 4}
    no_reasoning = exp.select_last_safe_prefix(
        {
            "case_id": case_id,
            "model_output": f"safe prose\n{certificates.TRIGGER_TOKEN}\n{{\"bad\": true}}",
        },
        [event],
    )
    trigger_only = exp.select_last_safe_prefix(
        {"case_id": case_id, "trigger_token_present": True, "model_output": ""},
        [event],
    )
    before_json = exp.select_last_safe_prefix(
        {"case_id": case_id, "model_output": "safe prose {\"bad\": true}"},
        [event],
    )
    no_json = exp.select_last_safe_prefix(
        {"case_id": case_id, "model_output": "safe prose only"},
        [event],
    )

    assert no_reasoning["safe_prefix"] == f"safe prose\n{certificates.TRIGGER_TOKEN}\n"
    assert trigger_only["safe_prefix"] == f"{certificates.TRIGGER_TOKEN}\n"
    assert before_json["safe_prefix"] == "safe prose "
    assert no_json["safe_prefix"] == "safe prose only"
    assert before_json["selection_reason"] == "monitor_error_before_json_boundary"


def test_req_verify_1496_missing_monitor_manifest_blocks_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1496: missing Exp 1495 monitor events write a terminal blocker."""

    output_path = tmp_path / "experiment_1496.json"
    continuation_manifest = tmp_path / "continuations.jsonl"

    artifact = exp.run_experiment(
        output_path=output_path,
        manifest_path=continuation_manifest,
        monitor_artifact_path=tmp_path / "missing_monitor.json",
        monitor_event_manifest_path=tmp_path / "missing_events.jsonl",
        certificate_manifest_path=tmp_path / "missing_certificates.jsonl",
        validator_manifest_path=tmp_path / "missing_validators.jsonl",
        collect_continuations_fn=lambda _spec, _plans: _raise("collector should not run"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["safe_prefix_continuation_ready"] is False
    assert artifact["cases_attempted"] == 0
    assert "missing_monitor_artifact" in artifact["blockers"]
    assert "missing_monitor_event_manifest" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert continuation_manifest.read_text(encoding="utf-8") == ""


def test_req_verify_1496_not_ready_monitor_no_plans_and_no_model_specs_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1496: gates, empty selections, and no mandated specs fail closed."""

    monitor_artifact = tmp_path / "monitor.json"
    monitor_events = tmp_path / "events.jsonl"
    certificate_manifest = tmp_path / "certificates.jsonl"
    validator_manifest = tmp_path / "validators.jsonl"
    case = certificates.cctu.build_benchmark_cases()[0]
    valid_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": certificates.certificate_text_for_case(
                case,
                lane=certificates.TRIGGER_LANE,
                reasoning_text="45 is the result.",
            ),
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )

    _write_json(monitor_artifact, {"status": "complete", "monitor_intervention_ready": False})
    _write_jsonl(monitor_events, [])
    _write_jsonl(certificate_manifest, [valid_row])
    _write_jsonl(validator_manifest, [_compiled_validator_row(case)])

    blockers = exp.gated_input_blockers(
        monitor_artifact_path=monitor_artifact,
        monitor_event_manifest_path=monitor_events,
        certificate_manifest_path=certificate_manifest,
        validator_manifest_path=validator_manifest,
    )
    assert blockers == ["monitor_gate_not_ready"]

    _write_json(monitor_artifact, {"status": "complete", "monitor_intervention_ready": True})
    no_plan_artifact = exp.run_experiment(
        output_path=tmp_path / "no_plan.json",
        manifest_path=tmp_path / "no_plan.jsonl",
        monitor_artifact_path=monitor_artifact,
        monitor_event_manifest_path=monitor_events,
        certificate_manifest_path=certificate_manifest,
        validator_manifest_path=validator_manifest,
        model_specs=[],
    )
    assert no_plan_artifact["status"] == "blocked"
    assert no_plan_artifact["blockers"] == ["no_interrupting_cctu_trigger_cases_selected"]

    bad_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": _trigger_output(case, final_answer="not 45"),
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )
    _write_jsonl(certificate_manifest, [bad_row])
    _write_jsonl(validator_manifest, [])
    _write_jsonl(
        monitor_events,
        [
            {
                "event_id": "interwhen-1495-test",
                "case_id": case.case_id,
                "lane": certificates.TRIGGER_LANE,
                "token_offset": 128,
                "polling_interval_tokens": 64,
                "interruption_triggered": True,
                "error_detected": True,
            }
        ],
    )
    no_spec_artifact = exp.run_experiment(
        output_path=tmp_path / "no_spec.json",
        manifest_path=tmp_path / "no_spec.jsonl",
        monitor_artifact_path=monitor_artifact,
        monitor_event_manifest_path=monitor_events,
        certificate_manifest_path=certificate_manifest,
        validator_manifest_path=validator_manifest,
        model_specs=[],
    )
    assert no_spec_artifact["status"] == "blocked"
    assert "no_mandated_sota_gguf_model_available" in no_spec_artifact["blockers"]
    assert "legacy_headline_fallback_disallowed" in no_spec_artifact["blockers"]
    assert no_spec_artifact["baseline_validator_pass_rate"] == 0.0


def test_scenario_verify_1496_runner_writes_matched_rates_and_manifest(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1496: safe-prefix, full-regeneration, and baseline rates are matched."""

    case = certificates.cctu.build_benchmark_cases()[0]
    bad_output = _trigger_output(case, final_answer="not 45")
    certificate_manifest = tmp_path / "certificates.jsonl"
    validator_manifest = tmp_path / "validators.jsonl"
    monitor_artifact = tmp_path / "monitor.json"
    monitor_events = tmp_path / "events.jsonl"
    output_path = tmp_path / "experiment_1496.json"
    continuation_manifest = tmp_path / "continuations.jsonl"

    source_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": bad_output,
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )
    _write_jsonl(certificate_manifest, [source_row])
    _write_jsonl(validator_manifest, [_compiled_validator_row(case)])
    _write_json(
        monitor_artifact,
        {
            "status": "complete",
            "monitor_intervention_ready": True,
            "monitor_event_manifest_path": str(monitor_events),
        },
    )
    _write_jsonl(
        monitor_events,
        [
            {
                "event_id": "interwhen-1495-test",
                "case_id": case.case_id,
                "lane": certificates.TRIGGER_LANE,
                "token_offset": 128,
                "polling_interval_tokens": 64,
                "interruption_triggered": True,
                "error_detected": True,
            }
        ],
    )

    def fake_collect(_spec: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for plan in plans:
            rows.append(
                {
                    "case_id": plan["case_id"],
                    "mode": exp.SAFE_PREFIX_MODE,
                    "generated_text": json.dumps(
                        certificates.certificate_for_case(plan["case"]),
                        sort_keys=True,
                    ),
                    "generation_source": "live_sota_llamacpp",
                    "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                    "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                    "elapsed_seconds": 0.2,
                    "blocker": None,
                }
            )
            rows.append(
                {
                    "case_id": plan["case_id"],
                    "mode": exp.FULL_REGENERATION_MODE,
                    "output_text": bad_output,
                    "generation_source": "live_sota_llamacpp",
                    "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                    "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                    "elapsed_seconds": 0.2,
                    "blocker": None,
                }
            )
        return {
            "summary": {
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        output_path=output_path,
        manifest_path=continuation_manifest,
        monitor_artifact_path=monitor_artifact,
        monitor_event_manifest_path=monitor_events,
        certificate_manifest_path=certificate_manifest,
        validator_manifest_path=validator_manifest,
        model_specs=[
            {
                **exp.MANDATED_MODEL_SPECS[0],
                "model_path": "/tmp/fake-qwen.gguf",
            }
        ],
        collect_continuations_fn=fake_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0, "gpus": []},
        tests_run=["focused pytest"],
    )
    rows = [
        json.loads(line)
        for line in continuation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert artifact["status"] == "complete"
    assert artifact["safe_prefix_continuation_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cases_attempted"] == 1
    assert artifact["continuations_completed"] == 1
    assert artifact["baseline_validator_pass_rate"] == 0.0
    assert artifact["safe_prefix_validator_pass_rate"] == 1.0
    assert artifact["full_regeneration_validator_pass_rate"] == 0.0
    assert artifact["verifier_false_accept_rate"] == 0.0
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["mode"] for row in rows} == {
        exp.NO_CONTINUATION_MODE,
        exp.SAFE_PREFIX_MODE,
        exp.FULL_REGENERATION_MODE,
    }
    assert all(row["compiled_validator_available"] is True for row in rows)
    assert [row["final_validator_passed"] for row in rows] == [False, True, False]


def test_req_verify_1496_model_load_failure_blocks_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1496: failed mandated GGUF loading blocks headline continuation."""

    case = certificates.cctu.build_benchmark_cases()[0]
    certificate_manifest = tmp_path / "certificates.jsonl"
    validator_manifest = tmp_path / "validators.jsonl"
    monitor_artifact = tmp_path / "monitor.json"
    monitor_events = tmp_path / "events.jsonl"

    source_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": _trigger_output(case, final_answer="not 45"),
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )
    _write_jsonl(certificate_manifest, [source_row])
    _write_jsonl(validator_manifest, [_compiled_validator_row(case)])
    _write_json(monitor_artifact, {"status": "complete", "monitor_intervention_ready": True})
    _write_jsonl(
        monitor_events,
        [
            {
                "event_id": "interwhen-1495-test",
                "case_id": case.case_id,
                "lane": certificates.TRIGGER_LANE,
                "token_offset": 128,
                "polling_interval_tokens": 64,
                "interruption_triggered": True,
                "error_detected": True,
            }
        ],
    )

    artifact = exp.run_experiment(
        output_path=tmp_path / "experiment_1496.json",
        manifest_path=tmp_path / "continuations.jsonl",
        monitor_artifact_path=monitor_artifact,
        monitor_event_manifest_path=monitor_events,
        certificate_manifest_path=certificate_manifest,
        validator_manifest_path=validator_manifest,
        model_specs=[
            {
                **exp.MANDATED_MODEL_SPECS[0],
                "model_path": "/tmp/fake-qwen.gguf",
            }
        ],
        collect_continuations_fn=lambda _spec, _plans: {
            "summary": {
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
                "model_used": False,
                "blocker": "llama_cpp_import_failed",
            },
            "rows": [],
        },
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["safe_prefix_continuation_ready"] is False
    assert artifact["models_used"] == []
    assert "llama_cpp_import_failed" in artifact["blockers"]
    assert "legacy_headline_fallback_disallowed" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1496_live_collector_is_injectable_without_real_gguf() -> None:
    """REQ-VERIFY-1496: live collection reports cache, import, load, and generation states."""

    case = certificates.cctu.build_benchmark_cases()[0]
    source_row = certificates.build_manifest_row(
        case,
        {
            "case_id": case.case_id,
            "lane": certificates.TRIGGER_LANE,
            "prompt": certificates.build_trigger_prompt(case),
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "generation_source": "live_sota_llamacpp",
            "output_text": _trigger_output(case, final_answer="not 45"),
            "elapsed_seconds": 0.1,
            "blocker": None,
        },
    )
    plan = {
        "case": case,
        "case_id": case.case_id,
        "family": case.family,
        "source_row": source_row,
        "original_prompt": certificates.build_trigger_prompt(case),
        **exp.select_last_safe_prefix(
            source_row,
            [{"event_id": "e", "token_offset": 128, "polling_interval_tokens": 64}],
        ),
    }
    spec = {**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake-qwen.gguf"}

    missing = exp.collect_live_continuations(
        {"hf_id": "missing/model", "name": "missing"},
        [plan],
        resolver=lambda _hf_id: None,
    )
    import_failed = exp.collect_live_continuations(
        spec,
        [plan],
        llama_importer=lambda: (False, None, "llama_cpp missing"),
        env_preparer=lambda: {},
    )

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    load_failed = exp.collect_live_continuations(
        spec,
        [plan],
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )

    class FakeLlama:
        def __init__(self, **_kwargs: Any) -> None:
            self.calls = 0
            self.closed = False

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("generation failed")
            return {
                "choices": [
                    {
                        "text": json.dumps(
                            certificates.certificate_for_case(case),
                            sort_keys=True,
                        )
                    }
                ]
            }

        def close(self) -> None:
            self.closed = True

    ok = exp.collect_live_continuations(
        spec,
        [plan],
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )

    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert "load failed" in load_failed["summary"]["blocker"]
    assert ok["summary"]["model_used"] is True
    assert [row["mode"] for row in ok["rows"]] == [
        exp.SAFE_PREFIX_MODE,
        exp.FULL_REGENERATION_MODE,
    ]
    assert ok["rows"][0]["output_text"].startswith(plan["safe_prefix"])
    assert ok["rows"][0]["blocker"] is None
    assert ok["rows"][1]["blocker"] == "RuntimeError: generation failed"


def test_req_verify_1496_metric_and_verdict_edges_are_explicit(
    monkeypatch: Any,
) -> None:
    """REQ-VERIFY-1496: metric helpers preserve blocked and false-accept distinctions."""

    monkeypatch.setattr(
        exp.certificates,
        "resolve_model_specs",
        lambda: [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
    )
    assert exp.resolve_model_specs() == [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}]
    assert exp._false_accept_rate(
        [{"cctu_verifier_result": {"base_valid": True}, "verifier_false_accept": False}]
    ) == 0.0
    assert exp._honest_verdict(
        ready=True,
        baseline_rate=1.0,
        safe_rate=1.0,
        false_accept_rate=0.0,
    ).startswith("complete: safe-prefix continuation measured without")
    assert exp._honest_verdict(
        ready=True,
        baseline_rate=1.0,
        safe_rate=0.0,
        false_accept_rate=0.5,
    ).startswith("complete: safe-prefix continuation measured but")


def _trigger_output(cert_case: certificates.cctu.BenchmarkCase, *, final_answer: str) -> str:
    payload = certificates.certificate_for_case(cert_case)
    payload["final_answer"] = final_answer
    payload["verifier"] = {"accept": True}
    return (
        "Reasoning: execute the tool and then certify the result.\n"
        f"{certificates.TRIGGER_TOKEN}\n"
        f"{json.dumps(payload, sort_keys=True)}"
    )


def _compiled_validator_row(cert_case: certificates.cctu.BenchmarkCase) -> dict[str, Any]:
    return {
        "prompt_id": cert_case.case_id,
        "validator_compiled": True,
        "manual_review_required": False,
        "compiled_validator": {
            "case_id": cert_case.case_id,
            "expected_final_answer": cert_case.expected_final_answer,
            "family": cert_case.family,
            "kind": "cctu_tool_transcript",
            "tool_arguments": cert_case.tool_arguments,
            "tool_name": cert_case.tool_name,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raise(message: str) -> None:
    raise AssertionError(message)
