"""Tests for live three-family span-first relation acquisition.

Spec refs: REQ-REPORT-6929 and SCENARIO-REPORT-6929-*.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import types

import pytest

from carnot import experiment_6929_three_family_span_acquisition as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


def _inference_result(raw_output: str) -> dict[str, object]:
    return {
        "raw_output": raw_output,
        "timed_out": False,
        "runner_failure": None,
        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
        "generation_duration_s": 0.25,
        "runner_receipt": {"runner": "fixture", "returncode": 0},
    }


def _all_terminal_rows(raw_output: str = "not-json") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in mod.MODEL_SPECS:
        for case in mod.HELDOUT_CASES:
            rows.append(mod.build_attempt_row(model, case, _inference_result(raw_output)))
    return rows


def _passing_preflight(tmp_path: Path) -> dict[str, object]:
    checks = [mod.check_row("fixture", True, True)]
    return {
        "checks": checks,
        "passed": True,
        "model_files": [
            {
                **dict(spec),
                "model_path": str(tmp_path / f"model-{index}.gguf"),
                "model_sha256": f"sha256:model-{index}",
                "cache_state": "hit",
            }
            for index, spec in enumerate(mod.MODEL_SPECS)
        ],
        "gpus": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-one"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-two"},
        ],
        "source_artifact_hashes": {"fixture": "sha256:fixture"},
    }


def test_req_report_6929_spec_freezes_models_fields_and_cells() -> None:
    """REQ-REPORT-6929: the spec fixes models, schema, and ninety attempts."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6929") :]
    assert [row["hf_id"] for row in mod.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert len(mod.FAMILIES) == 5
    assert len(mod.HELDOUT_CASES) == 30
    assert mod.EXAMPLES_PER_CELL == 6
    assert mod.EXPECTED_ATTEMPTS == 90
    assert mod.heldout_split_hash() == mod.EXPECTED_HELDOUT_SPLIT_HASH
    assert len({row["prompt_id"] for row in mod.HELDOUT_CASES}) == 30
    assert all(
        sum(row["family"] == family for row in mod.HELDOUT_CASES) == 6 for family in mod.FAMILIES
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_report_6929_utf8_offsets_use_source_bytes() -> None:
    """SCENARIO-REPORT-6929-UTF8: multibyte prefixes do not shift byte receipts."""

    source = "Préface Ω: Task α comes before Task β."
    raw = json.dumps(
        {"span_a": "Task α", "span_b": "Task β", "relation": "comes before"},
        ensure_ascii=False,
    )
    parsed = mod.parse_model_output(source, raw)
    encoded = source.encode("utf-8")
    assert parsed["parse_state"] == "accepted"
    assert parsed["span_a_offsets"] == {
        "start_utf8": encoded.index("Task α".encode()),
        "end_utf8": encoded.index("Task α".encode()) + len("Task α".encode()),
    }
    assert parsed["span_b_offsets"] == {
        "start_utf8": encoded.index("Task β".encode()),
        "end_utf8": encoded.index("Task β".encode()) + len("Task β".encode()),
    }
    assert (
        encoded[
            parsed["span_a_offsets"]["start_utf8"] : parsed["span_a_offsets"]["end_utf8"]
        ].decode()
        == "Task α"
    )


def test_scenario_report_6929_duplicate_spans_retain_all_candidates() -> None:
    """SCENARIO-REPORT-6929-AMBIGUOUS: duplicate text is never resolved by guess."""

    source = "Node α is near Node α before Node β."
    parsed = mod.parse_model_output(
        source,
        json.dumps({"span_a": "Node α", "span_b": "Node β", "relation": "precedes"}),
    )
    assert parsed["parse_state"] == "span_a_ambiguous"
    assert len(parsed["span_a_candidates"]) == 2
    assert parsed["span_a_offsets"] is None
    assert parsed["alias_normalized"] is False
    assert parsed["directed_relation"] is None


def test_scenario_report_6929_reversed_direction_retains_offsets() -> None:
    """SCENARIO-REPORT-6929-DIRECTION: reversed source order fails closed."""

    parsed = mod.parse_model_output(
        "Task B follows Task A.",
        json.dumps({"span_a": "Task A", "span_b": "Task B", "relation": "precedes"}),
    )
    assert parsed["parse_state"] == "span_direction_reversed"
    assert parsed["span_a_offsets"]["start_utf8"] > parsed["span_b_offsets"]["start_utf8"]
    assert parsed["direction_valid"] is False
    assert parsed["alias_normalized"] is False


def test_scenario_report_6929_aliases_normalize_only_after_grounding() -> None:
    """SCENARIO-REPORT-6929-ALIASES: accepted aliases follow both span receipts."""

    accepted = mod.parse_model_output(
        "Task A comes before Task B.",
        json.dumps({"span_a": "Task A", "span_b": "Task B", "relation": "comes before"}),
    )
    assert accepted["parse_state"] == "accepted"
    assert accepted["raw_relation"] == "comes before"
    assert accepted["alias_normalized"] is True
    assert accepted["directed_relation"] == {
        "family": "precedence",
        "predicate": "precedes",
        "polarity": "positive",
        "source": "A",
        "target": "B",
    }

    missing = mod.parse_model_output(
        "Only Task A appears.",
        json.dumps({"span_a": "Task A", "span_b": "Task B", "relation": "comes before"}),
    )
    assert missing["parse_state"] == "span_b_absent"
    assert missing["alias_normalized"] is False
    assert missing["directed_relation"] is None


def test_scenario_report_6929_invalid_json_and_missing_fields_are_terminal() -> None:
    """SCENARIO-REPORT-6929-MALFORMED: malformed outputs keep exact failure causes."""

    invalid = mod.parse_model_output("Task A precedes Task B.", "{broken")
    missing = mod.parse_model_output(
        "Task A precedes Task B.", json.dumps({"span_a": "Task A", "relation": "precedes"})
    )
    absent = mod.parse_model_output(
        "Task A precedes Task B.",
        json.dumps({"span_a": "Task A", "span_b": "Task C", "relation": "precedes"}),
    )
    unsupported = mod.parse_model_output(
        "Task A precedes Task B.",
        json.dumps({"span_a": "Task A", "span_b": "Task B", "relation": "teleports"}),
    )
    assert invalid["parse_state"] == "invalid_json"
    assert missing["parse_state"] == "missing_required_field"
    assert absent["parse_state"] == "span_b_absent"
    assert unsupported["parse_state"] == "unsupported_relation_alias"
    assert all(row["terminal"] is True for row in (invalid, missing, absent, unsupported))


def test_scenario_report_6929_hidden_labels_are_isolated_from_all_prompts() -> None:
    """SCENARIO-REPORT-6929-ISOLATION: hidden metadata never enters model context."""

    for case in mod.HELDOUT_CASES:
        prompt = mod.build_prompt(case)
        row = mod.hidden_label_isolation(case, prompt, [])
        assert row["passed"] is True
        assert row["forbidden_exposures"] == []
        assert "offset" not in prompt.casefold()
        assert "alias" not in prompt.casefold()
        assert "positive" not in prompt.casefold()
        assert "negative" not in prompt.casefold()
        assert "unknown" not in prompt.casefold()
        assert case["source_text"] in prompt

    leaked = mod.hidden_label_isolation(
        mod.HELDOUT_CASES[0], mod.build_prompt(mod.HELDOUT_CASES[0]), ["Expected label: positive"]
    )
    assert leaked["passed"] is False
    assert "hidden_label:positive" in leaked["forbidden_exposures"]


def test_scenario_report_6929_attempt_rows_preserve_failures_and_timeouts() -> None:
    """SCENARIO-REPORT-6929-MALFORMED: every runner outcome becomes a terminal row."""

    model = mod.MODEL_SPECS[0]
    case = mod.HELDOUT_CASES[0]
    malformed = mod.build_attempt_row(model, case, _inference_result("not-json"))
    timed_out = mod.build_attempt_row(
        model,
        case,
        {
            **_inference_result(""),
            "timed_out": True,
            "runner_failure": "model_timeout",
        },
    )
    assert malformed["parse_state"] == "invalid_json"
    assert malformed["raw_output"] == "not-json"
    assert malformed["failure_reason"] == "invalid_json"
    assert timed_out["parse_state"] == "timeout"
    assert timed_out["failure_reason"] == "model_timeout"
    assert timed_out["terminal"] is True
    assert timed_out["token_budget"]["max_tokens"] == mod.DECODING_SETTINGS["max_tokens"]


def test_scenario_report_6929_readiness_counts_terminal_attempts_not_accuracy() -> None:
    """SCENARIO-REPORT-6929-READINESS: malformed rows can form a complete raw bank."""

    rows = _all_terminal_rows()
    report = mod.readiness_report(rows)
    assert report["passed"] is True
    assert report["expected_attempt_count"] == 90
    assert report["observed_attempt_count"] == 90
    assert len(report["model_family_cell_rows"]) == 15
    assert all(row["attempt_count"] == 6 for row in report["model_family_cell_rows"])
    assert all(row["terminal_count"] == 6 for row in report["model_family_cell_rows"])

    assert mod.readiness_report(rows[:-1])["passed"] is False
    duplicated = rows + [deepcopy(rows[0])]
    assert mod.readiness_report(duplicated)["passed"] is False
    nonterminal = deepcopy(rows)
    nonterminal[0]["terminal"] = False
    assert mod.readiness_report(nonterminal)["passed"] is False


def test_req_report_6929_blocked_artifact_names_failed_gate() -> None:
    """REQ-REPORT-6929: precondition failure writes the complete blocked schema."""

    failed = mod.check_row("dual_cuda", 2, 1)
    artifact = mod.blocked_artifact(
        date="20260903",
        duration_s=0.1,
        checks=[failed],
        source_hashes={"fixture": "missing"},
        model_files=[],
    )
    assert artifact["honest_verdict"] == "blocked_three_family_span_acquisition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["span_acquisition_bank_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "dual_cuda"
    assert artifact["gate_check_summary"]["expected"] == 2
    assert artifact["gate_check_summary"]["observed"] == 1
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()


def test_req_report_6929_complete_artifact_projects_every_required_row() -> None:
    """REQ-REPORT-6929: a complete raw bank derives all required evidence tables."""

    rows = _all_terminal_rows()
    artifact = mod.build_artifact(
        date="20260903",
        duration_s=61.0,
        checks=[mod.check_row("all", True, True)],
        source_hashes={"fixture": "sha256:fixture"},
        model_files=[dict(row) for row in mod.MODEL_SPECS],
        gpus=[{"uuid": "GPU-one"}, {"uuid": "GPU-two"}],
        rows=rows,
        process_rows=[{"model_id": "fixture", "status": "complete"}],
    )
    assert artifact["span_acquisition_bank_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_span_acquisition_bank_ready"
    assert len(artifact["raw_output_rows"]) == 90
    assert len(artifact["parse_failure_rows"]) == 90
    assert len(artifact["hidden_label_isolation_rows"]) == 90
    assert len(artifact["source_text_rows"]) == 30
    assert len(artifact["per_game_results"]) == 30
    assert mod.validate_artifact(artifact) == []


def test_req_report_6929_worker_uses_chat_json_and_keeps_runner_errors() -> None:
    """REQ-REPORT-6929: the live worker preserves successful and failed calls."""

    calls: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            if len(calls) == 3:
                raise RuntimeError("fixture failure")
            return {
                "choices": [{"message": {"content": '{\\"span_a\\": \\"x\\"}'}}],
                "usage": {"completion_tokens": 5},
            }

        def close(self) -> None:
            calls.append({"closed": True})

    outputs = mod.worker_acquire(
        model_file={**dict(mod.MODEL_SPECS[0]), "model_path": "/cache/model.gguf"},
        cases=mod.HELDOUT_CASES[:2],
        llama_factory=FakeLlama,
        pid=4321,
    )
    assert len(outputs) == 2
    assert outputs[0]["raw_output"].startswith("{")
    assert outputs[0]["runner_receipt"]["worker_pid"] == 4321
    assert outputs[1]["runner_failure"].startswith("RuntimeError")
    assert calls[0]["tensor_split"] == [0.5, 0.5]
    assert calls[-1] == {"closed": True}


def test_req_report_6929_execute_models_fills_missing_worker_rows(tmp_path: Path) -> None:
    """REQ-REPORT-6929: a timed-out worker still yields every scheduled tuple."""

    model_files = _passing_preflight(tmp_path)["model_files"]

    def executor(_model: dict[str, object], _index: int) -> dict[str, object]:
        return {
            "status": "timeout",
            "timed_out": True,
            "returncode": None,
            "stderr": "deadline",
            "duration_s": 2.0,
            "command": ["fixture-worker"],
            "outputs": [],
        }

    rows, process_rows = mod.execute_models(model_files, executor=executor)
    assert len(rows) == 90
    assert len(process_rows) == 3
    assert all(row["parse_state"] == "timeout" for row in rows)
    assert all(row["terminal"] is True for row in rows)
    assert mod.readiness_report(rows)["passed"] is True


def test_req_report_6929_run_writes_blocked_and_complete_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-6929: orchestration writes one validated terminal artifact."""

    blocked_path = tmp_path / "blocked.json"
    failed_preflight = {
        "checks": [mod.check_row("model", "available", "missing")],
        "passed": False,
        "model_files": [],
        "gpus": [],
        "source_artifact_hashes": {"fixture": "missing"},
    }
    blocked = mod.run(
        date="20260903",
        output_path=blocked_path,
        preflight_fn=lambda **_kwargs: failed_preflight,
    )
    assert json.loads(blocked_path.read_text())["honest_verdict"] == blocked["honest_verdict"]

    complete_path = tmp_path / "complete.json"
    passing = _passing_preflight(tmp_path)
    rows = _all_terminal_rows()
    complete = mod.run(
        date="20260903",
        output_path=complete_path,
        preflight_fn=lambda **_kwargs: passing,
        execute_fn=lambda _files: (rows, [{"model_id": "fixture", "status": "complete"}]),
    )
    assert complete["span_acquisition_bank_ready_score"] == 1
    assert (
        json.loads(complete_path.read_text())["reproducibility_checksum"]
        == complete["reproducibility_checksum"]
    )
    assert mod.validate_artifact(complete) == []


def test_req_report_6929_preflight_and_cli_error_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6929: source, resolver, CUDA, and CLI failures stay explicit."""

    output = tmp_path / "result.json"
    preflight = mod.check_preconditions(
        repo_root=tmp_path,
        output_path=output,
        resolver=lambda _hf_id, _quant: None,
        gpu_probe=lambda: [],
        llama_probe=lambda: {"importable": False, "supports_gpu_offload": False},
    )
    assert preflight["passed"] is False
    failed_names = {row["check"] for row in preflight["checks"] if not row["passed"]}
    assert "exp6926_fixture_contract" in failed_names
    assert "dual_cuda_devices" in failed_names
    assert "llama_cpp_cuda_offload" in failed_names
    assert any(name.startswith("model_cache:") for name in failed_names)

    monkeypatch.setattr(
        mod,
        "run",
        lambda **_kwargs: {
            "honest_verdict": "blocked_three_family_span_acquisition",
            "span_acquisition_bank_ready_score": 0,
        },
    )
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0


def test_req_report_6929_chat_content_and_hash_helpers_handle_edge_shapes() -> None:
    """REQ-REPORT-6929: bounded helpers reject unusable response and artifact shapes."""

    assert mod.chat_content({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert mod.chat_content({"choices": []}) == ""
    assert mod.chat_content(types.SimpleNamespace()) == ""
    assert mod.span_candidates(b"aaaa", "aa") == [
        {"start_utf8": 0, "end_utf8": 2},
        {"start_utf8": 1, "end_utf8": 3},
        {"start_utf8": 2, "end_utf8": 4},
    ]
    assert mod.span_candidates(b"abc", "") == []
    artifact = mod.blocked_artifact(
        date="20260903",
        duration_s=0.0,
        checks=[mod.check_row("x", True, False)],
        source_hashes={},
        model_files=[],
    )
    artifact["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in mod.validate_artifact(artifact)


def test_scenario_report_6929_parser_rejects_remaining_grounding_edges() -> None:
    """SCENARIO-REPORT-6929-MALFORMED: all strict parser exits retain evidence."""

    assert mod.parse_model_output("A then B", "[]")["parse_state"] == "json_not_object"
    assert (
        mod.parse_model_output(
            "A then B", json.dumps({"span_a": "C", "span_b": "B", "relation": "precedes"})
        )["parse_state"]
        == "span_a_absent"
    )
    duplicate_b = mod.parse_model_output(
        "A then B and B",
        json.dumps({"span_a": "A", "span_b": "B", "relation": "precedes"}),
    )
    assert duplicate_b["parse_state"] == "span_b_ambiguous"
    overlap = mod.parse_model_output(
        "Task AB",
        json.dumps({"span_a": "Task AB", "span_b": "AB", "relation": "precedes"}),
    )
    assert overlap["parse_state"] == "spans_overlap"
    assert overlap["direction_valid"] is False

    runner_failed = mod.build_attempt_row(
        mod.MODEL_SPECS[0],
        mod.HELDOUT_CASES[0],
        {**_inference_result(""), "runner_failure": "load_failed"},
    )
    assert runner_failed["parse_state"] == "runner_failure"
    assert runner_failed["failure_reason"] == "load_failed"


def test_scenario_report_6929_isolation_detects_private_markers() -> None:
    """SCENARIO-REPORT-6929-ISOLATION: checker metadata in a reprompt is detected."""

    case = mod.HELDOUT_CASES[0]
    row = mod.hidden_label_isolation(
        case,
        mod.build_prompt(case),
        ["offset alias expected answer checker output prior failure"],
    )
    assert row["passed"] is False
    assert row["reprompt_count"] == 1
    assert len(row["forbidden_exposures"]) == 5


def test_req_report_6929_chat_content_rejects_bad_choice_and_message_shapes() -> None:
    """REQ-REPORT-6929: chat extraction does not invent absent response text."""

    assert mod.chat_content({"choices": ["bad"]}) == ""
    assert mod.chat_content({"choices": [{"message": "bad"}]}) == ""
    assert mod.chat_content({"choices": [{"message": {"content": None}}]}) == ""


def test_req_report_6929_worker_default_factory_and_pid(monkeypatch) -> None:
    """REQ-REPORT-6929: the default worker binds the installed llama.cpp factory."""

    calls: list[dict[str, object]] = []

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def close(self) -> None:
            calls.append({"closed": True})

    monkeypatch.setitem(
        __import__("sys").modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama)
    )
    outputs = mod.worker_acquire(
        model_file={**dict(mod.MODEL_SPECS[0]), "model_path": "/cache/model.gguf"},
        cases=[],
    )
    assert outputs == []
    assert calls[0]["model_path"] == "/cache/model.gguf"
    assert calls[-1] == {"closed": True}


def test_req_report_6929_execute_worker_preserves_transport_outcomes(monkeypatch) -> None:
    """REQ-REPORT-6929: child success, timeout, and launch failure are all terminal."""

    model = {**dict(mod.MODEL_SPECS[0]), "model_path": "/cache/model.gguf"}
    valid = json.dumps({"prompt_id": "p", "raw_output": "{}"})
    completed = types.SimpleNamespace(
        stdout=f"{valid}\nnot-json\n[]\n",
        stderr="warning",
        returncode=0,
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: completed)
    success = mod.execute_worker(model, 0)
    assert success["status"] == "complete"
    assert len(success["outputs"]) == 1
    assert success["malformed_worker_lines"] == ["not-json", "[]"]
    assert success["command"] == mod._worker_command(model, 0)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("worker", 1, output=b'{"prompt_id":"p"}\n', stderr=b"late")

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    timed_out = mod.execute_worker(model, 0)
    assert timed_out["status"] == "timeout"
    assert timed_out["timed_out"] is True
    assert timed_out["stderr"] == "late"

    def missing(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(mod.subprocess, "run", missing)
    failed = mod.execute_worker(model, 0)
    assert failed["status"] == "failed"
    assert failed["stderr"].startswith("OSError")


def test_req_report_6929_execute_models_handles_unique_and_duplicate_outputs(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6929: worker transport cannot duplicate an attempted tuple."""

    model_files = _passing_preflight(tmp_path)["model_files"]
    first = mod.HELDOUT_CASES[0]
    second = mod.HELDOUT_CASES[1]

    def executor(_model: dict[str, object], _index: int) -> dict[str, object]:
        unique = {"prompt_id": first["prompt_id"], **_inference_result("{}")}
        duplicate = {"prompt_id": second["prompt_id"], **_inference_result("{}")}
        return {
            "status": "complete",
            "timed_out": False,
            "returncode": 0,
            "stderr": "",
            "duration_s": 1.0,
            "command": ["fixture-worker"],
            "outputs": [unique, duplicate, duplicate],
        }

    rows, _process_rows = mod.execute_models(model_files, executor=executor)
    first_rows = [row for row in rows if row["prompt_id"] == first["prompt_id"]]
    second_rows = [row for row in rows if row["prompt_id"] == second["prompt_id"]]
    assert all(row["parse_state"] == "missing_required_field" for row in first_rows)
    assert all(row["failure_reason"] == "duplicate_worker_outputs" for row in second_rows)


def test_req_report_6929_gpu_and_llama_probes_cover_host_failures(monkeypatch) -> None:
    """REQ-REPORT-6929: malformed host probes cannot satisfy CUDA preflight."""

    good = types.SimpleNamespace(
        returncode=0,
        stdout=(
            "0, NVIDIA GeForce RTX 3090, GPU-one, 24576\n"
            "bad,line\n"
            "x, NVIDIA GeForce RTX 3090, GPU-two, nope\n"
        ),
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: good)
    assert mod.query_gpus() == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-one",
            "memory_total_mb": 24576,
        }
    ]

    monkeypatch.setattr(
        mod.subprocess, "run", lambda *_args, **_kwargs: types.SimpleNamespace(returncode=1)
    )
    assert mod.query_gpus() == []

    def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(mod.subprocess, "run", unavailable)
    assert mod.query_gpus() == []
    assert mod.llama_cpp_status()["importable"] is True

    real_import = builtins.__import__

    def import_without_llama(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("fixture")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_llama)
    assert mod.llama_cpp_status()["supports_gpu_offload"] is False


def test_req_report_6929_real_fixture_contract_and_successful_preflight(tmp_path: Path) -> None:
    """REQ-REPORT-6929: the pinned Exp6926 sources and frozen live inputs pass together."""

    fixture, hashes = mod._fixture_contract(REPO)
    assert fixture["artifact_sha256"] == mod.EXP6926_SHA256
    assert fixture["source_hashes_valid"] is True
    assert len(hashes) == 6

    model_paths: dict[str, str] = {}
    for index, spec in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"gguf")
        model_paths[spec["hf_id"]] = str(path)
    preflight = mod.check_preconditions(
        repo_root=REPO,
        output_path=tmp_path / "result.json",
        resolver=lambda hf_id, _quant: model_paths[hf_id],
        gpu_probe=lambda: [
            {"index": 1, "uuid": "GPU-two"},
            {"index": 0, "uuid": "GPU-one"},
        ],
        llama_probe=lambda: {"importable": True, "supports_gpu_offload": True},
    )
    assert preflight["passed"] is True
    assert [row["uuid"] for row in preflight["gpus"]] == ["GPU-one", "GPU-two"]
    assert all(row["cache_state"] == "hit" for row in preflight["model_files"])


def test_req_report_6929_fixture_and_resolver_malformed_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6929: invalid receipts and resolver errors stay failed preconditions."""

    fixture_path = tmp_path / mod.EXP6926_PATH
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text(
        json.dumps(
            {
                "schema": mod.EXP6926_SCHEMA,
                "span_relation_fixture_ready_score": 1,
                "source_artifact_hashes": {
                    "bad": "not-a-receipt",
                    "mismatch": {
                        "path": "missing-source",
                        "expected_sha256": "sha256:x",
                        "observed_sha256": "sha256:x",
                    },
                },
            }
        )
    )
    observed, _hashes = mod._fixture_contract(tmp_path)
    assert observed["source_hashes_valid"] is False

    def broken_resolver(_hf_id: str, _quant: str) -> str:
        raise RuntimeError("resolver failed")

    preflight = mod.check_preconditions(
        repo_root=tmp_path,
        output_path=tmp_path / "result.json",
        resolver=broken_resolver,
        gpu_probe=lambda: [{"index": 0, "uuid": "same"}, {"index": 1, "uuid": "same"}],
        llama_probe=lambda: {"importable": True, "supports_gpu_offload": True},
    )
    assert preflight["passed"] is False
    assert all(row["resolver_error"].startswith("RuntimeError") for row in preflight["model_files"])


def test_req_report_6929_validator_rejects_all_contract_drift() -> None:
    """REQ-REPORT-6929: each schema, verdict, and readiness mutation is rejected."""

    blocked = mod.blocked_artifact(
        date="20260903",
        duration_s=0.0,
        checks=[mod.check_row("x", True, False)],
        source_hashes={},
        model_files=[],
    )
    mutations = {
        "field_principles": None,
        "inference_substrate": "wrong",
        "verifier_is_oracle": True,
        "verdict_class": "wrong",
        "honest_verdict": "wrong",
    }
    for field, value in mutations.items():
        artifact = deepcopy(blocked)
        artifact[field] = value
        artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
        assert field in mod.validate_artifact(artifact)

    for field, value, expected_error in (
        ("honest_verdict", "blocked_wrong", "blocked_honest_verdict"),
        ("span_acquisition_bank_ready_score", 1, "blocked_ready_score"),
        ("gate_check_summary", {"passed": True}, "blocked_gate_summary"),
    ):
        artifact = deepcopy(blocked)
        artifact[field] = value
        artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
        assert expected_error in mod.validate_artifact(artifact)

    complete = mod.build_artifact(
        date="20260903",
        duration_s=61.0,
        checks=[mod.check_row("all", True, True)],
        source_hashes={},
        model_files=mod.MODEL_SPECS,
        gpus=[],
        rows=_all_terminal_rows(),
        process_rows=[],
    )
    for field, value, expected_error in (
        ("span_acquisition_bank_ready_score", 0, "ready_score_drift"),
        ("model_family_cell_rows", [], "cell_rows_drift"),
        ("gate_check_summary", {}, "gate_summary_drift"),
    ):
        artifact = deepcopy(complete)
        artifact[field] = value
        artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
        assert expected_error in mod.validate_artifact(artifact)


def test_req_report_6929_run_validation_worker_and_cli_guards(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """REQ-REPORT-6929: orchestration and internal CLI guards cannot fail silently."""

    failed_preflight = {
        "checks": [mod.check_row("x", True, False)],
        "passed": False,
        "model_files": [],
        "gpus": [],
        "source_artifact_hashes": {},
    }
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["fixture_error"])
    with pytest.raises(RuntimeError, match="artifact_validation_failed"):
        mod.run(
            date="20260903",
            output_path=tmp_path / "bad.json",
            preflight_fn=lambda **_kwargs: failed_preflight,
        )

    monkeypatch.setattr(
        mod,
        "worker_acquire",
        lambda **_kwargs: [{"prompt_id": "p", "raw_output": "{}"}],
    )
    assert mod.worker_main(0, "/cache/model.gguf") == 0
    assert '"prompt_id":"p"' in capsys.readouterr().out

    monkeypatch.setattr(mod, "worker_main", lambda _index, _path: 7)
    assert mod.main(["--worker", "--model-index", "0", "--model-path", "/cache/m.gguf"]) == 7
    with pytest.raises(SystemExit):
        mod.main(["--worker"])
    with pytest.raises(SystemExit):
        mod.main([])
