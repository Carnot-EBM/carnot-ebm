"""REQ-REPORT-7787 and REQ-VERIFY-7787 cover the prospective producer."""

import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7787_v677_qwen_event_confidence as exp


def family(name="a", unsupported=True):
    """Use private source bytes and a natural-style annotation in each fixture."""
    return {
        "family_id": name,
        "source": "The source says blue.\nSecond sentence stays intact.",
        "answer": "The answer says green.",
        "annotation_types": ["unsupported"] if unsupported else [],
    }


def test_protocol_freezes_same_bytes_grammar_seed_and_budget():
    """SCENARIO-VERIFY-7787-TRANSPORT changes only the requested event."""
    p = exp.make_protocol([family()])
    assert p["seed"] == 67701
    assert p["family_ids"] == ["a"]
    assert p["input_hashes"]["a"]["source_sha256"] != p["input_hashes"]["a"]["answer_sha256"]
    left = exp.make_request(p, family(), "generic")
    right = exp.make_request(p, family(), "event")
    assert left["messages"][1] == right["messages"][1]
    assert left["response_format"] == right["response_format"]
    assert all(req["seed"] == 67701 and req["max_tokens"] == 256 for req in (left, right))
    assert left["messages"][0] != right["messages"][0]


@pytest.mark.parametrize(
    "text,finish,valid,risk",
    [
        ('{"probability":0.8}', "stop", True, 0.2),
        ('{"probability":0.8}', "length", False, 0.5),
        ('{"probability":1.1}', "stop", False, 0.5),
        ('{"probability":true}', "stop", False, 0.5),
        ('{"other":0.8}', "stop", False, 0.5),
        ("broken", "stop", False, 0.5),
        ("", "missing", False, 0.5),
    ],
)
def test_parser_keeps_invalid_denominator(text, finish, valid, risk):
    """SCENARIO-VERIFY-7787-TRANSPORT treats syntax apart from truth."""
    actual = exp.parse_probability(text, finish, "generic")
    assert actual["valid"] is valid
    assert actual["unsupported_risk"] == pytest.approx(risk)
    assert actual["forced_escalation"] is not valid


def test_preflight_rejects_missing_producer_and_receipt_substitution(tmp_path):
    """SCENARIO-REPORT-7787-CUSTODY records the actual absent producer."""
    panel = tmp_path / exp.PANEL
    panel.parent.mkdir(parents=True)
    panel.write_text(json.dumps([family()]))
    rows, checks, _ = exp.preflight(tmp_path)
    assert rows == []
    failures = [c for c in checks if not c["passed"]]
    assert any(c["upstream_id"] == "exp7745" and c["field"] == "exists" for c in failures)
    assert all("expected" in c and "observed" in c and "artifact_sha256" in c for c in failures)


def test_reducer_counts_pairs_and_penalizes_confident_error():
    """SCENARIO-REPORT-7787-TERMINAL counts families and checks false accepts."""
    rows = []
    for item in (family("a", True), family("b", False)):
        for arm, probability in (("generic", 0.9), ("event", 0.9)):
            parsed = exp.parse_probability(json.dumps({"probability": probability}), "stop", arm)
            rows.append(
                exp.score_row(item, arm, parsed, {"prompt_tokens": 10, "completion_tokens": 3}, 0.2)
            )
    reduced = exp.reduce_rows(rows, seed=67701)
    assert reduced["independent_n"] == 2
    assert reduced["parse_coverage_by_arm"] == {"generic": 1.0, "event": 1.0}
    assert reduced["semantic_comparison_rows"]["generic"]["false_accepts"] == 1
    assert reduced["semantic_comparison_rows"]["event"]["false_accepts"] == 0
    assert reduced["benefit_passed"] is False
    with pytest.raises(ValueError, match="duplicate"):
        exp.reduce_rows(rows + rows[:1], seed=67701)


def test_cold_reader_rejects_changed_response(tmp_path):
    """SCENARIO-VERIFY-7787-REPLAY opens exact request and reply bytes."""
    item = family()
    p = exp.make_protocol([item])
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = exp.capture_family(
        item,
        p,
        lambda _request: {
            "choices": [{"message": {"content": '{"probability":0.8}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "model": exp.MODEL_ID,
        },
        raw,
        0.0,
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows, "protocol": p, "panel": [item]}))
    assert exp.cold_reduce(candidate)["independent_n"] == 1
    Path(rows[0]["raw_response_path"]).write_text("changed")
    with pytest.raises(ValueError, match="response_hash"):
        exp.cold_reduce(candidate)


def test_authentic_private_producer_and_blocked_fields(tmp_path):
    """SCENARIO-REPORT-7787-CUSTODY verifies a complete private producer."""
    from carnot.reporting.current_work_receipt import sha256_file

    panel = [family(str(i), bool(i % 2)) for i in range(24)]
    panel_path = tmp_path / exp.PANEL
    panel_path.parent.mkdir(parents=True)
    panel_path.write_text(json.dumps(panel))
    producer = tmp_path / exp.PRODUCER
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text(
        json.dumps(
            {
                "milestone": "2026.09.674",
                "verdict_class": "null",
                "qwen_localization_complete_score": 1,
                "flagged_adversarial": False,
                "paired_family_results": {"paired_families": 24},
                "source_artifact_hashes": {
                    "pre_gate_receipts": {str(exp.PANEL): sha256_file(panel_path)}
                },
            }
        )
    )
    rows, checks, hashes = exp.preflight(tmp_path)
    assert len(rows) == 24 and all(check["passed"] for check in checks)
    assert hashes["producer"]["eligible"] is True
    value = json.loads(producer.read_text())
    value["verdict_class"] = "disqualified"
    producer.write_text(json.dumps(value))
    rows, checks, hashes = exp.preflight(tmp_path)
    assert rows == [] and not hashes["producer"]["eligible"]
    assert any(c["field"] == "verdict_class" and not c["passed"] for c in checks)


def test_error_capture_and_reader_detects_each_layer(tmp_path):
    """SCENARIO-VERIFY-7787-REPLAY retains transport failure and detects edits."""
    from carnot.reporting.current_work_receipt import sha256_file

    item = family()
    protocol = exp.make_protocol([item])
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = exp.capture_family(
        item, protocol, lambda _request: (_ for _ in ()).throw(TimeoutError("late")), raw, 0.0
    )
    assert all(row["metrics"]["unsupported_risk"] == 0.5 for row in rows)
    candidate = tmp_path / "candidate.json"
    value = {"rows": rows, "panel": [item], "protocol": protocol, "reduced": exp.reduce_rows(rows)}
    candidate.write_text(json.dumps(value))
    assert exp.cold_reduce(candidate)["independent_n"] == 1
    value["rows"][0]["raw_request_sha256"] = "sha256:bad"
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="request_hash"):
        exp.cold_reduce(candidate)
    value["rows"][0]["raw_request_sha256"] = sha256_file(Path(rows[0]["raw_request_path"]))
    value["rows"][0]["response_text"] = "changed"
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="row_changed"):
        exp.cold_reduce(candidate)
    value["rows"][0]["response_text"] = ""
    value["reduced"] = {}
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reduction_changed"):
        exp.cold_reduce(candidate)
    value["reduced"] = exp.reduce_rows(rows)
    value["rows"].append({"family_id": "b", "arm": "generic", "disposition": "unstarted"})
    candidate.write_text(json.dumps(value))
    assert exp.cold_reduce(candidate)["independent_n"] == 1
    request_path = Path(rows[0]["raw_request_path"])
    request = json.loads(request_path.read_text())
    request["temperature"] = 0.1
    request_path.write_text(json.dumps(request))
    value["rows"][0]["raw_request_sha256"] = sha256_file(request_path)
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="request_changed"):
        exp.cold_reduce(candidate)


def test_reducer_unstarted_and_empty():
    """SCENARIO-REPORT-7787-TERMINAL keeps an empty pilot unmeasured."""
    empty = exp.reduce_rows([])
    assert empty["independent_n"] == 0
    assert empty["parse_coverage_by_arm"] == {"generic": None, "event": None}
    assert (
        exp.reduce_rows([{"family_id": "a", "arm": "generic", "disposition": "unstarted"}])[
            "independent_n"
        ]
        == 0
    )


def test_owned_capture_with_private_server(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7787-CAPTURE checks owned load, canaries and raw pairs."""
    import hashlib
    import importlib
    import time as real_time
    from types import SimpleNamespace
    import urllib.request

    arc = importlib.import_module("carnot.agentic.arc_executable_world_model")
    sentinel = importlib.import_module("carnot.experiment_7431_v651_arc_live_sentinel")
    bounded = importlib.import_module("carnot.experiment_7581_v662_arc_bounded_canary")
    pilot = importlib.import_module("carnot.experiment_7604_v664_evidence_pilot")

    class FakeProposer:
        def __init__(self, **kwargs):
            self._proc = SimpleNamespace(pid=123)
            self._stderr_log_path = None
            self.kwargs = kwargs
            self.stopped = False

        def _ensure_server(self):
            return True

        def server_props(self):
            return {"chat_template": "template"}

        def _url(self):
            return "http://127.0.0.1:12345"

        def stop(self):
            self.stopped = True

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": exp.MODEL_ID}]}).encode()

    monkeypatch.setattr(arc, "LocalGGUFProposer", FakeProposer)
    monkeypatch.setattr(sentinel, "_free_port", lambda: 12345)
    monkeypatch.setattr(bounded, "_call_with_heartbeats", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(bounded, "_observed_offload_layers", lambda _path: 66)
    monkeypatch.setattr(bounded, "_owned_vram_mb", lambda _pid: 18000)
    monkeypatch.setattr(
        pilot, "_offload_receipt", lambda *_args: {"actual_offload": True, "loaded_layers": 66}
    )
    monkeypatch.setattr(pilot, "_runtime_build_receipt", lambda: {"backend": "fake"})

    def reply(_url, payload, _timeout):
        return json.dumps(
            {
                "model": exp.MODEL_ID,
                "choices": [
                    {"message": {"content": '{"probability":0.8}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            }
        ).encode()

    monkeypatch.setattr(pilot, "_post_json", reply)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    model.write_text("fake")
    context = {
        "selected": {"index": 0, "uuid": "fake"},
        "model_path": model,
        "model_sha256": "sha256:fake",
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
    }
    panel = [family()]
    protocol = exp.make_protocol(panel)
    rows, runtime = exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    assert len(rows) == 2
    assert runtime["canary_calls"] == 2 and runtime["panel_calls"] == 2
    assert runtime["gpu_offload_receipt"]["actual_offload"]
    assert exp.reduce_rows(rows)["independent_n"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    with monkeypatch.context() as stage:
        stage.setattr(FakeProposer, "_ensure_server", lambda self: False)
        with pytest.raises(RuntimeError, match="owned_qwen_load_failed"):
            exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    with monkeypatch.context() as stage:
        stage.setattr(FakeProposer, "server_props", lambda self: {"chat_template": "other"})
        with pytest.raises(RuntimeError, match="chat_template_mismatch"):
            exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    with monkeypatch.context() as stage:
        stage.setattr(pilot, "_offload_receipt", lambda *_args: {"actual_offload": False})
        with pytest.raises(RuntimeError, match="qwen_gpu_offload_not_authenticated"):
            exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    with monkeypatch.context() as stage:
        stage.setattr(FakeResponse, "read", lambda self: b'{"data":[{"id":"wrong"}]}')
        with pytest.raises(RuntimeError, match="wrong_model_server"):
            exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    with monkeypatch.context() as stage:
        stage.setattr(
            pilot,
            "_post_json",
            lambda *_args: json.dumps(
                {
                    "model": exp.MODEL_ID,
                    "choices": [{"message": {"content": "broken"}, "finish_reason": "stop"}],
                }
            ).encode(),
        )
        with pytest.raises(RuntimeError, match="schema_canary_failed"):
            exp.owned_capture(tmp_path, panel, protocol, context, 0.0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"
    for phase_name, jump, expected in (("before", 100, "capture_deadline"), ("after", 1801, None)):
        state = {"armed": False, "calls": 0}

        def clock():
            if state["armed"]:
                state["calls"] += 1
                if state["calls"] == 2:
                    return real_time.monotonic() + jump
            return real_time.monotonic()

        def boundary(_start, phase, event, units=0):
            if phase == "canary" and event == phase_name and (phase_name == "before" or units == 2):
                state.update(armed=True, calls=0)

        with monkeypatch.context() as stage:
            stage.setattr(exp, "time", SimpleNamespace(monotonic=clock, time=real_time.time))
            stage.setattr(exp, "progress", boundary)
            if expected:
                with pytest.raises(TimeoutError, match=expected):
                    exp.owned_capture(tmp_path / phase_name, panel, protocol, context, 0.0)
            else:
                no_rows, _runtime = exp.owned_capture(
                    tmp_path / phase_name, panel, protocol, context, 0.0
                )
                assert no_rows == []


def test_private_entrypoint_runs_frozen_scope_and_terminal_readers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7787-TERMINAL exercises the real orchestration with private rows."""
    import importlib
    from carnot.reporting import experiment_7303_validation_scope as validation

    panel = [family(str(i), bool(i % 2)) for i in range(24)]
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    model.write_text("private model fixture")
    source_result = tmp_path / exp.PRODUCER
    source_result.parent.mkdir(parents=True)
    source_result.write_text(
        json.dumps({"current_model_receipts": {"model_sha256": "sha256:fixture"}})
    )
    check = {
        "check": "fixture",
        "upstream_id": "exp7745",
        "artifact_path": str(source_result),
        "artifact_sha256": "sha256:fixture",
        "field": "exists",
        "operator": "==",
        "expected": True,
        "observed": True,
        "passed": True,
    }
    monkeypatch.setattr(
        exp,
        "preflight",
        lambda _root: (
            panel,
            [check],
            {"producer": {"sha256": "sha256:fixture"}, "panel": {"sha256": "sha256:fixture"}},
        ),
    )
    context = {
        "model_path": model,
        "model_sha256": "sha256:fixture",
        "selected": {"index": 0},
        "chat_template_sha256": "fixture",
    }
    monkeypatch.setattr(exp.source, "prepare_panel", lambda *_args: (panel, [], {}, context))
    sota = importlib.import_module("carnot.inference.sota_models")
    monkeypatch.setattr(
        sota, "cached_current_model", lambda: {"hf_id": exp.MODEL_ID, "model_path": str(model)}
    )

    def fake_capture(root, families, protocol, _context, started):
        raw = root / exp.RAW / "private"
        raw.mkdir(parents=True)
        rows = []
        for item in families:

            def reply(_request):
                return {
                    "model": exp.MODEL_ID,
                    "choices": [
                        {"message": {"content": '{"probability":0.8}'}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                }

            rows += exp.capture_family(item, protocol, reply, raw, started)
        return rows, {
            "model_load_completed": 1,
            "canary_calls": 2,
            "panel_calls": 48,
            "model_path": str(model),
            "model_sha256": "sha256:fixture",
            "gpu_offload_receipt": {"actual_offload": True},
        }

    monkeypatch.setattr(exp, "owned_capture", fake_capture)
    names = []

    def fake_commands(_root, commands, **_kwargs):
        names.extend(spec.name for spec in commands)
        return [
            {
                "name": spec.name,
                "command_argv": list(spec.argv),
                "exit_code": 0,
                "passed": True,
                "log_sha256": "sha256:fixture",
            }
            for spec in commands
        ]

    monkeypatch.setattr(validation, "run_commands", fake_commands)
    output = tmp_path / "result.json"
    result = exp.run_experiment(tmp_path, "20260927", output)
    assert output.is_file() and len(result["rows"]) == 48
    assert result["sample_size_budget"]["independent_n"] == 24
    assert "cold_replay" in names and "adversarial_verify" in names
    assert result["historical_full_suite_status"]["qwen_runner_ready_score"] == 0
    assert exp.cold_reduce(tmp_path / exp.RAW / "candidate.json")["independent_n"] == 24
    with monkeypatch.context() as stage:
        stage.setattr(
            exp, "owned_capture", lambda *_args: (_ for _ in ()).throw(RuntimeError("load"))
        )
        failed = exp.run_experiment(tmp_path, "20260927", tmp_path / "failed.json")
    assert failed["verdict_class"] == "disqualified"
    assert len(failed["rows"]) == 48
    assert all(row["disposition"] == "unstarted" for row in failed["rows"])
    source_result.write_text(
        json.dumps({"current_model_receipts": {"model_sha256": "sha256:wrong"}})
    )
    blocked = exp.run_experiment(tmp_path, "20260927", tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert any(not check["passed"] for check in blocked["gate_check_summary"])
    source_result.write_text(
        json.dumps({"current_model_receipts": {"model_sha256": "sha256:fixture"}})
    )
    original_reduce = exp.reduce_rows

    def positive_reduce(rows, seed=exp.SEED):
        value = original_reduce(rows, seed)
        value["benefit_passed"] = True
        return value

    with monkeypatch.context() as stage:
        stage.setattr(exp, "reduce_rows", positive_reduce)
        positive = exp.build_artifact(
            panel,
            result["protocol"],
            result["rows"],
            [check],
            result["source_artifact_hashes"],
            result["current_model_receipts"],
            {"required": [{"passed": True}]},
            0.0,
            [],
            "20260927",
        )
    assert positive["verdict_class"] == "positive"


def test_main_cold_and_dated_path(tmp_path, monkeypatch, capsys):
    """SCENARIO-REPORT-7787-TERMINAL covers both CLI branches."""
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    monkeypatch.setattr(exp, "cold_reduce", lambda _path: {"independent_n": 1})
    assert exp.main(["--cold-replay", str(candidate)]) == 0
    assert '"independent_n": 1' in capsys.readouterr().out
    monkeypatch.setattr(
        exp, "run_experiment", lambda *_args: {"honest_verdict": "complete_null_exposed_pilot"}
    )
    assert exp.main(["--date", "20260927"]) == 0
    assert "complete_null_exposed_pilot" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        exp.main(["--date", "20260926"])


def test_foreign_model_response_is_rejected(tmp_path):
    """SCENARIO-VERIFY-7787-TRANSPORT rejects an unrelated server reply."""
    item = family()
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = exp.capture_family(
        item, exp.make_protocol([item]), lambda _request: {"model": "wrong"}, raw, 0.0
    )
    assert all(row["disposition"] == "rejected" for row in rows)


def test_thin_cli_calls_main(monkeypatch):
    """SCENARIO-REPORT-7787-TERMINAL covers the actual command wrapper."""
    import runpy

    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(
            str(exp.ROOT / "scripts/experiments/experiment_7787_v677_qwen_event_confidence.py"),
            run_name="__main__",
        )
    assert error.value.code == 0
