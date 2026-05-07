"""Tests for Exp 1493 trigger-token CCTU certificate export.

Spec: REQ-VERIFY-1493, SCENARIO-VERIFY-1493.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu
from carnot.eval import cctu_trigger_certificate_export as exp


def test_req_verify_1493_parser_splits_only_on_trigger_token() -> None:
    """REQ-VERIFY-1493: parser preserves reasoning and fails closed on trigger errors."""

    case = cctu.build_benchmark_cases()[0]
    output = exp.certificate_text_for_case(
        case,
        lane=exp.TRIGGER_LANE,
        reasoning_text="I can solve the arithmetic locally before exporting.",
    )

    parsed = exp.parse_certificate_output(output, lane=exp.TRIGGER_LANE)

    assert parsed["parsed"] is True
    assert parsed["trigger_token_present"] is True
    assert parsed["free_form_reasoning_text"] == (
        "I can solve the arithmetic locally before exporting."
    )
    assert parsed["certificate_json"]["case_id"] == case.case_id
    assert parsed["parse_error"] is None

    missing = exp.parse_certificate_output(
        json.dumps(exp.certificate_for_case(case)),
        lane=exp.TRIGGER_LANE,
    )
    duplicated = exp.parse_certificate_output(
        f"first {exp.TRIGGER_TOKEN} {{}} {exp.TRIGGER_TOKEN} {{}}",
        lane=exp.TRIGGER_LANE,
    )

    assert missing["parsed"] is False
    assert missing["parse_error"] == "missing_trigger_token"
    assert duplicated["parsed"] is False
    assert duplicated["parse_error"] == "duplicate_trigger_token"


def test_req_verify_1493_always_constrained_parser_accepts_json_without_trigger() -> None:
    """REQ-VERIFY-1493: always-constrained baseline parses direct certificate JSON."""

    case = cctu.build_benchmark_cases()[1]
    certificate = exp.certificate_for_case(case)
    parsed = exp.parse_certificate_output(
        f"```json\n{json.dumps(certificate, sort_keys=True)}\n```",
        lane=exp.ALWAYS_CONSTRAINED_LANE,
    )

    assert parsed["parsed"] is True
    assert parsed["trigger_token_present"] is False
    assert parsed["free_form_reasoning_text"] == ""
    assert parsed["certificate_json"] == certificate


def test_req_verify_1493_validator_reuses_cctu_checks_and_case_id_guard() -> None:
    """REQ-VERIFY-1493: deterministic validation catches schema and CCTU violations."""

    case = cctu.build_benchmark_cases()[2]
    valid = exp.validate_certificate(case, exp.certificate_for_case(case))
    wrong_answer = exp.certificate_for_case(case)
    wrong_answer["final_answer"] = "wrong"
    wrong_answer["verifier"] = {"accept": True}
    bad_answer = exp.validate_certificate(case, wrong_answer)
    wrong_case_id = exp.certificate_for_case(case)
    wrong_case_id["case_id"] = "not-this-case"
    bad_case_id = exp.validate_certificate(case, wrong_case_id)
    missing = exp.validate_certificate(case, None)

    assert valid["validator_result"]["case_id_valid"] is True
    assert valid["verifier_result"]["accepted"] is True
    assert bad_answer["validator_result"]["final_answer_valid"] is False
    assert bad_answer["verifier_result"]["accepted"] is False
    assert bad_answer["verifier_result"]["caught_invalid"] is True
    assert bad_case_id["validator_result"]["case_id_valid"] is False
    assert bad_case_id["verifier_result"]["base_valid"] is False
    assert missing["validator_result"]["parse_error"] == "missing_certificate_json"


def test_req_verify_1493_manifest_rows_keep_reasoning_parser_and_false_accept() -> None:
    """REQ-VERIFY-1493: each row records reasoning, parser result, validation, and false accept."""

    case = cctu.build_benchmark_cases()[3]
    generation_row = {
        "case_id": case.case_id,
        "lane": exp.TRIGGER_LANE,
        "prompt": exp.build_trigger_prompt(case),
        "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
        "model_name": exp.MANDATED_MODEL_SPECS[0]["name"],
        "generation_source": "live_sota_llamacpp",
        "output_text": exp.certificate_text_for_case(
            case,
            lane=exp.TRIGGER_LANE,
            reasoning_text="The weighted total is computed from the supplied table.",
        ),
        "elapsed_seconds": 0.01,
        "blocker": None,
    }

    row = exp.build_manifest_row(case, generation_row)

    assert row["lane"] == exp.TRIGGER_LANE
    assert row["trigger_token_present"] is True
    assert row["free_form_reasoning_text"].startswith("The weighted total")
    assert row["certificate_json"]["tool_call"]["name"] == case.tool_name
    assert row["parser_result"]["parsed"] is True
    assert row["validator_result"]["tool_result_consistent"] is True
    assert row["deterministic_validation_passed"] is True
    assert row["false_accept_status"] is False


def test_req_verify_1493_aggregate_metrics_are_lane_specific() -> None:
    """REQ-VERIFY-1493: trigger and always-constrained rates are computed separately."""

    cases = cctu.build_benchmark_cases()[:2]
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            exp.build_manifest_row(
                case,
                {
                    "case_id": case.case_id,
                    "lane": exp.TRIGGER_LANE,
                    "model_hf_id": "m",
                    "model_name": "model",
                    "generation_source": "live_sota_llamacpp",
                    "output_text": exp.certificate_text_for_case(
                        case,
                        lane=exp.TRIGGER_LANE,
                        reasoning_text="reason",
                    ),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                },
            )
        )
        rows.append(
            exp.build_manifest_row(
                case,
                {
                    "case_id": case.case_id,
                    "lane": exp.ALWAYS_CONSTRAINED_LANE,
                    "model_hf_id": "m",
                    "model_name": "model",
                    "generation_source": "live_sota_llamacpp",
                    "output_text": (
                        json.dumps(exp.certificate_for_case(case))
                        if case is cases[0]
                        else "not json"
                    ),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                },
            )
        )

    metrics = exp.aggregate_manifest_metrics(rows)

    assert metrics["certificate_parse_rate"] == pytest.approx(1.0)
    assert metrics["certificate_validation_rate"] == pytest.approx(1.0)
    assert metrics["always_constrained_parse_rate"] == pytest.approx(0.5)
    assert metrics["always_constrained_validation_rate"] == pytest.approx(0.5)
    assert metrics["verifier_false_accept_rate"] == pytest.approx(0.0)


def test_scenario_verify_1493_runner_writes_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1493: runner writes paired CCTU rows and required artifact fields."""

    def fake_collect(spec: dict[str, Any], cases: list[cctu.BenchmarkCase]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            rows.append(
                {
                    "case_id": case.case_id,
                    "lane": exp.TRIGGER_LANE,
                    "prompt": exp.build_trigger_prompt(case),
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec.get("name"),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": exp.certificate_text_for_case(
                        case,
                        lane=exp.TRIGGER_LANE,
                        reasoning_text=f"Free solve for {case.case_id}.",
                    ),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "lane": exp.ALWAYS_CONSTRAINED_LANE,
                    "prompt": exp.build_always_constrained_prompt(case),
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec.get("name"),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": json.dumps(exp.certificate_for_case(case), sort_keys=True),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    output_path = tmp_path / "experiment_1493.json"
    manifest_path = tmp_path / "manifest_1493.jsonl"

    artifact = exp.run_experiment(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_model_outputs_fn=fake_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["trigger_certificate_ready"] is True
    assert artifact["cctu_cases_attempted"] == 20
    assert artifact["cctu_cases_completed"] == 20
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["certificate_validation_rate"] == pytest.approx(1.0)
    assert artifact["always_constrained_parse_rate"] == pytest.approx(1.0)
    assert artifact["always_constrained_validation_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["certificate_manifest_path"] == str(manifest_path)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == 40
    assert {row["lane"] for row in manifest_rows} == {
        exp.TRIGGER_LANE,
        exp.ALWAYS_CONSTRAINED_LANE,
    }


def test_req_verify_1493_runner_blocks_without_sota_specs(tmp_path: Path) -> None:
    """REQ-VERIFY-1493: no SOTA model means terminal blocker, not legacy fallback."""

    output_path = tmp_path / "blocked.json"
    manifest_path = tmp_path / "blocked.jsonl"

    artifact = exp.run_experiment(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260507",
        model_specs=[],
        collect_model_outputs_fn=lambda _spec, _cases: pytest.fail("should not collect"),
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["trigger_certificate_ready"] is False
    assert artifact["cctu_cases_attempted"] == 20
    assert artifact["cctu_cases_completed"] == 0
    assert artifact["blockers"] == ["no_mandated_sota_gguf_model_available"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert manifest_path.read_text(encoding="utf-8") == ""


def test_req_verify_1493_runner_blocks_when_collector_produces_no_live_rows(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1493: model collection without live rows records a terminal blocker."""

    artifact = exp.run_experiment(
        output_path=tmp_path / "no_rows.json",
        manifest_path=tmp_path / "no_rows.jsonl",
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_model_outputs_fn=lambda spec, _cases: {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "no_usable_generations",
            },
            "rows": [],
        },
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["blockers"] == [
        "no_usable_generations",
        "live_sota_generation_unavailable",
    ]


def test_req_verify_1493_model_resolution_prefers_cached_pair_then_singletons() -> None:
    """REQ-VERIFY-1493: model specs come from cached_sota_pair or mandated GGUF hits."""

    pair = [
        {"name": "A", "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 3, "model_path": "a"}
    ]
    assert exp.resolve_model_specs(cached_pair_fn=lambda **_kwargs: pair) == pair

    calls: list[str] = []

    def fake_resolver(hf_id: str) -> str | None:
        calls.append(hf_id)
        return "/tmp/qwen.gguf" if hf_id == exp.MANDATED_MODEL_SPECS[0]["hf_id"] else None

    singleton = exp.resolve_model_specs(
        cached_pair_fn=lambda **_kwargs: None,
        resolver_fn=fake_resolver,
    )

    assert singleton == [
        {
            "name": exp.MANDATED_MODEL_SPECS[0]["name"],
            "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "role": exp.MANDATED_MODEL_SPECS[0]["role"],
            "gpu": exp.MANDATED_MODEL_SPECS[0]["gpu"],
            "model_path": "/tmp/qwen.gguf",
        }
    ]
    assert calls == [spec["hf_id"] for spec in exp.MANDATED_MODEL_SPECS]


def test_req_verify_1493_live_collector_is_injectable_without_gguf() -> None:
    """REQ-VERIFY-1493: live collector supports unit-test injection around llama.cpp."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            case_id = prompt.split("Case: ", 1)[1].split("\n", 1)[0]
            case = next(c for c in cctu.build_benchmark_cases() if c.case_id == case_id)
            lane = (
                exp.TRIGGER_LANE
                if exp.TRIGGER_TOKEN in prompt
                else exp.ALWAYS_CONSTRAINED_LANE
            )
            return {
                "choices": [
                    {
                        "text": exp.certificate_text_for_case(
                            case,
                            lane=lane,
                            reasoning_text="reason",
                        )
                    }
                ]
            }

        def close(self) -> None:
            self.closed = True

    cases = cctu.build_benchmark_cases()[:1]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 0}
    ok = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: "/tmp/fake.gguf",
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    missing = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: None,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_model_outputs(
        {**spec, "model_path": "/tmp/fake.gguf"},
        cases,
        llama_importer=lambda: (False, None, "llama missing"),
        env_preparer=lambda: {},
    )

    assert ok["summary"]["model_used"] is True
    assert len(ok["rows"]) == 2
    assert {row["lane"] for row in ok["rows"]} == {
        exp.TRIGGER_LANE,
        exp.ALWAYS_CONSTRAINED_LANE,
    }
    assert FakeLlama.prompts == [
        exp.build_trigger_prompt(cases[0]),
        exp.build_always_constrained_prompt(cases[0]),
    ]
    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama missing"


def test_req_verify_1493_live_collector_reports_load_and_generation_errors() -> None:
    """REQ-VERIFY-1493: live collection records model-load and generation blockers."""

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    cases = cctu.build_benchmark_cases()[:1]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/tmp/fake.gguf"}
    load_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    generate_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, GenerateFails, None),
        env_preparer=lambda: {},
    )

    assert load_failed["summary"]["model_used"] is False
    assert "load failed" in load_failed["summary"]["blocker"]
    assert generate_failed["summary"]["model_used"] is False
    assert len(generate_failed["rows"]) == 2
    assert generate_failed["rows"][0]["blocker"] == "RuntimeError: generation failed"


def test_req_verify_1493_gpu_probe_handles_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1493: GPU probe records available GPUs or the blocker string."""

    class Result:
        returncode = 0
        stdout = "GPU A, 24576 MiB\n\nGPU B, 24576 MiB\n"
        stderr = ""

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Result())
    available = exp.probe_gpu()

    assert available["nvidia_smi_available"] is True
    assert available["gpu_count"] == 2
    assert available["gpus"][0]["name"] == "GPU A"

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(exp.subprocess, "run", boom)
    blocked = exp.probe_gpu()

    assert blocked["nvidia_smi_available"] is False
    assert blocked["gpu_count"] == 0
    assert "no nvidia-smi" in blocked["error"]


def test_req_verify_1493_completed_case_count_ignores_blocked_lanes() -> None:
    """REQ-VERIFY-1493: completed cases require both lanes to generate without blockers."""

    assert (
        exp._completed_case_count(
            [
                {"case_id": "c1", "lane": exp.TRIGGER_LANE, "blocker": None},
                {"case_id": "c1", "lane": exp.ALWAYS_CONSTRAINED_LANE, "blocker": None},
                {"case_id": "c2", "lane": exp.TRIGGER_LANE, "blocker": None},
                {
                    "case_id": "c2",
                    "lane": exp.ALWAYS_CONSTRAINED_LANE,
                    "blocker": "empty_generation",
                },
            ]
        )
        == 1
    )


def test_req_verify_1493_main_prints_key_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1493: CLI reports the terminal readiness and rate fields."""

    def fake_run() -> dict[str, Any]:
        return {
            "trigger_certificate_ready": True,
            "certificate_parse_rate": 1.0,
            "certificate_validation_rate": 0.75,
            "honest_verdict": "complete: test",
        }

    monkeypatch.setattr(exp, "run_experiment", fake_run)

    assert exp.main([]) == 0
    printed = capsys.readouterr().out
    assert "ready=True" in printed
    assert "parse=1.0" in printed
    assert "validation=0.75" in printed
