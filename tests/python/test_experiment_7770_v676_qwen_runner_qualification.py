"""REQ-REPORT-7770 checks the CPU fixture and frozen confidence contract."""

import json
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import pytest

from carnot import experiment_7770_v676_qwen_runner_qualification as exp


@pytest.fixture
def protocol():
    """SCENARIO-REPORT-7770-CUSTODY uses the real frozen task protocol."""
    return json.loads((exp.ROOT / exp.PROTOCOL).read_text())


@pytest.fixture
def family():
    """One private family is enough to check paired request construction."""
    return {
        "family_id": "fixture-a",
        "source": "Source byte 1.\nSource byte 2.",
        "answer": "Claim.",
    }


def test_requests_share_bytes_grammar_and_budget(protocol, family):
    """SCENARIO-REPORT-7770-TRANSPORT fixes only the confidence target."""
    generic = exp.make_request(protocol, family, "generic")
    event = exp.make_request(protocol, family, "event")
    assert generic["messages"][1] == event["messages"][1]
    assert json.loads(generic["messages"][1]["content"]) == {
        "complete_source": family["source"],
        "original_answer": family["answer"],
    }
    for request in (generic, event):
        assert request["response_format"] == protocol["request"]["response_format"]
        assert request["max_tokens"] == 256
        assert request["seed"] == 67670
        assert request["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "body,finish,valid,risk",
    [
        ('{"probability":0.8}', "stop", True, 0.2),
        ('{"probability":0.8,"evidence_sentence_ids":[0,2]}', "stop", True, 0.2),
        ('{"probability":true}', "stop", False, 0.5),
        ('{"probability":1.1}', "stop", False, 0.5),
        ('{"probability":NaN}', "stop", False, 0.5),
        ('{"probability":0.8,"extra":1}', "stop", False, 0.5),
        ('{"probability":0.8}', "length", False, 0.5),
        ("broken", "stop", False, 0.5),
    ],
)
def test_parse_and_escalation(body, finish, valid, risk):
    """SCENARIO-REPORT-7770-TRANSPORT retains malformed output."""
    generic = exp.parse_probability(body, finish, "generic")
    assert generic["valid"] is valid
    assert generic["unsupported_risk"] == pytest.approx(risk)
    assert generic["forced_escalation"] is (not valid)
    event = exp.parse_probability(body, finish, "event")
    assert event["unsupported_risk"] == pytest.approx(0.8 if valid else 0.5)


def test_old_parse_counts_are_recomputed_from_raw_rows():
    """SCENARIO-REPORT-7770-CUSTODY reopens all 72 historical dispositions."""
    counts = exp.old_parse_counts(exp.ROOT / exp.OLD_ROWS)
    assert counts == {
        "canonical": 1,
        "paired": 2,
        "source_withheld": 9,
        "both_complete_source_families": 1,
        "denominator": 24,
    }


def test_checkpoint_resume_is_atomic(tmp_path):
    """SCENARIO-REPORT-7770-TRANSPORT keeps one complete row per key."""
    path = tmp_path / "nested" / "checkpoint.json"
    row = {"family_id": "fixture-a", "arm": "generic", "disposition": "completed"}
    exp.save_checkpoint(path, [row])
    assert exp.load_checkpoint(path) == [row]
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(ValueError):
        exp.save_checkpoint(path, [row, row])
    assert exp.load_checkpoint(path) == [row]


def test_independent_reduction_rejects_duplicate_and_counts_family():
    """SCENARIO-REPORT-7770-TERMINAL counts pairs once and retains invalid rows."""
    rows = [
        {
            "family_id": "a",
            "arm": "generic",
            "metrics": {"valid": True, "unsupported_risk": 0.2, "forced_escalation": False},
        },
        {
            "family_id": "a",
            "arm": "event",
            "metrics": {"valid": False, "unsupported_risk": 0.5, "forced_escalation": True},
        },
    ]
    assert exp.reduce_rows(rows)["effective_independent_n"] == 1
    assert exp.reduce_rows(rows)["forced_escalations"] == 1
    with pytest.raises(ValueError):
        exp.reduce_rows(rows + rows[:1])


class FixtureHandler(BaseHTTPRequestHandler):
    """A private HTTP peer enforces the fields the future server must honor."""

    model_name = "unsloth/Qwen3.8-27B-GGUF"
    bad_reply = False
    delay_s = 0.0
    seen = []

    def log_message(self, *_args):
        """Keep test output focused on failed assertions."""

    def do_GET(self):
        """Expose a model roster so a foreign endpoint cannot be adopted."""
        data = json.dumps({"data": [{"id": self.model_name}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        """Use SSE chunks and refuse requests that could ignore the schema."""
        import time

        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(payload)
        grammar = payload.get("response_format", {})
        schema = grammar.get("json_schema", {}).get("schema", {})
        if (
            grammar.get("type") != "json_schema"
            or schema.get("required") != ["probability"]
            or payload.get("max_tokens", 257) > 256
        ):
            self.send_error(400, "schema required")
            return
        time.sleep(type(self).delay_s)
        if type(self).bad_reply:
            chunks = [b"not-json"]
        else:
            chunks = [b'{"prob', b'ability":0.7}']
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            event = {
                "model": self.model_name,
                "choices": [{"delta": {"content": chunk.decode()}, "finish_reason": None}],
            }
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        finish = {"model": self.model_name, "choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(b"data: " + json.dumps(finish).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def server():
    """SCENARIO-REPORT-7770-TRANSPORT owns and closes a local CPU server."""
    FixtureHandler.model_name = "unsloth/Qwen3.8-27B-GGUF"
    FixtureHandler.bad_reply = False
    FixtureHandler.delay_s = 0.0
    FixtureHandler.seen = []
    http = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}"
    http.shutdown()
    http.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_transport_identity_stream_and_schema_probe(server, protocol, family):
    """SCENARIO-REPORT-7770-TRANSPORT tests real HTTP and strict grammar."""
    request = exp.make_request(protocol, family, "event")
    assert exp.check_identity(server, request["model"])
    reply = exp.chat_stream(server, request, timeout_s=2)
    assert reply == ('{"probability":0.7}', "stop")
    assert len(FixtureHandler.seen) == 1
    request["response_format"] = {"type": "json_object"}
    with pytest.raises(HTTPError):
        exp.chat_stream(server, request, timeout_s=2)
    FixtureHandler.model_name = "wrong-model"
    assert not exp.check_identity(server, protocol["request"]["model"])


def test_transport_malformed_and_timeout(server, protocol, family):
    """SCENARIO-REPORT-7770-TRANSPORT records bad output and bounded timeout."""
    request = exp.make_request(protocol, family, "generic")
    FixtureHandler.bad_reply = True
    text, finish = exp.chat_stream(server, request, timeout_s=2)
    assert exp.parse_probability(text, finish, "generic")["forced_escalation"]
    FixtureHandler.delay_s = 0.2
    with pytest.raises(TimeoutError):
        exp.chat_stream(server, request, timeout_s=0.02)


def test_real_cli_and_cold_reader(tmp_path):
    """SCENARIO-REPORT-7770-TERMINAL runs the actual entrypoint in a child."""
    output = tmp_path / "output.json"
    cli = exp.ROOT / "scripts/experiments/experiment_7770_v676_qwen_runner_qualification.py"
    command = [sys.executable, "-u", str(cli), "--date", "20260927", "--output", str(output)]
    run = subprocess.run(command, cwd=exp.ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    artifact = json.loads(output.read_text())
    assert artifact["qwen_runner_ready_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invocation_counts"] == {"loads": 0, "calls": 0}
    assert len(artifact["rows"]) == 48
    cold = subprocess.run(
        [sys.executable, "-u", str(cli), "--cold-replay", str(output)],
        cwd=exp.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cold.returncode == 0, cold.stdout + cold.stderr


def test_inprocess_main_and_cold_replay(tmp_path, capsys, monkeypatch):
    """SCENARIO-REPORT-7770-TERMINAL measures every owned module statement."""
    output = tmp_path / "inprocess.json"
    assert exp.main(["--date", "20260927", "--output", str(output)]) == 0
    assert exp.main(["--cold-replay", str(output)]) == 0
    import runpy

    cli = exp.ROOT / "scripts/experiments/experiment_7770_v676_qwen_runner_qualification.py"
    monkeypatch.setattr(sys, "argv", [str(cli), "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as ended:
        runpy.run_path(str(cli), run_name="__main__")
    assert ended.value.code == 0
    assert '"effective_independent_n": 24' in capsys.readouterr().out
    assert exp.cold_replay(output)["completed_calls"] == 48


def test_preflight_distinguishes_producer_from_receipt(tmp_path, protocol):
    """SCENARIO-REPORT-7770-CUSTODY never substitutes a pre-gate panel."""
    panel, checks = exp.preflight(tmp_path, protocol)
    assert panel == []
    assert {check["upstream_id"] for check in checks if not check["passed"]} == {
        "exp7745",
        "exp7745_pre_gate",
        "exp7759_raw",
        "exp7759_diagnostic",
    }


def test_cold_replay_detects_mutation(tmp_path):
    """SCENARIO-REPORT-7770-TERMINAL rejects changed raw response bytes."""
    output = tmp_path / "output.json"
    exp.main(["--output", str(output)])
    artifact = json.loads(output.read_text())
    raw = Path(artifact["rows"][0]["raw_response_path"])
    raw.write_text('{"text":"bad","finish":"stop"}')
    with pytest.raises(ValueError, match="response_hash_changed"):
        exp.cold_replay(output)


def test_misc_invalid_inputs(protocol, family, tmp_path):
    """SCENARIO-REPORT-7770-TRANSPORT rejects unplanned arms and empty state."""
    with pytest.raises(ValueError, match="unplanned_arm"):
        exp.make_request(protocol, family, "other")
    with pytest.raises(ValueError, match="unplanned_arm"):
        exp.parse_probability("{}", "stop", "other")
    assert exp.load_checkpoint(tmp_path / "absent.json") == []
    assert exp.sha(exp.ROOT / exp.PROTOCOL).startswith("sha256:")


def test_foreign_stream_model_is_rejected(server, protocol, family):
    """SCENARIO-REPORT-7770-TRANSPORT refuses a foreign reply identity."""
    request = exp.make_request(protocol, family, "generic")
    FixtureHandler.model_name = "foreign-model"
    with pytest.raises(ValueError, match="foreign_response_model"):
        exp.chat_stream(server, request, 2)


def test_cold_reader_rejects_each_layer(tmp_path):
    """SCENARIO-REPORT-7770-TERMINAL checks hashes, scores and summary."""
    output = tmp_path / "output.json"
    exp.main(["--output", str(output)])
    original = json.loads(output.read_text())
    changed = json.loads(output.read_text())
    changed["rows"][0]["raw_request_sha256"] = "sha256:wrong"
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="request_hash_changed"):
        exp.cold_replay(output)
    changed = json.loads(json.dumps(original))
    changed["rows"][0]["metrics"]["unsupported_risk"] = 0.123
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="row_metrics_changed"):
        exp.cold_replay(output)
    changed = json.loads(json.dumps(original))
    changed["reduced"]["effective_independent_n"] = 0
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reduction_changed"):
        exp.cold_replay(output)


def test_resume_and_server_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7770-TRANSPORT resumes or refuses bad server contracts."""
    output = tmp_path / "output.json"
    exp.main(["--output", str(output)])
    assert exp.main(["--output", str(output)]) == 0
    old_identity = exp.check_identity
    monkeypatch.setattr(exp, "check_identity", lambda *_: False)
    with pytest.raises(ValueError, match="fixture_server_identity"):
        exp.run_experiment(exp.ROOT, "20260927", tmp_path / "other.json")
    monkeypatch.setattr(exp, "check_identity", old_identity)
    old_stream = exp.chat_stream
    monkeypatch.setattr(exp, "chat_stream", lambda *_: ('{"probability":0.5}', "stop"))
    with pytest.raises(ValueError, match="grammar_field_ignored"):
        exp.run_experiment(exp.ROOT, "20260927", tmp_path / "other.json")
    monkeypatch.setattr(exp, "chat_stream", old_stream)


def test_nested_basetemp_parent_supports_real_child(tmp_path):
    """SCENARIO-REPORT-7770-TERMINAL starts pytest after nested parent setup."""
    parent = tmp_path / "nested" / "pytest"
    parent.mkdir(parents=True)
    target = tmp_path / "child_test.py"
    target.write_text("def test_child():\n    assert 2 + 2 == 4\n")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={parent / 'run'}",
            str(target),
            "-q",
        ],
        cwd=exp.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_missing_producer_is_terminal_block_with_check_rows(tmp_path):
    """SCENARIO-REPORT-7770-CUSTODY preserves every failed prerequisite."""
    protocol = tmp_path / exp.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((exp.ROOT / exp.PROTOCOL).read_bytes())
    artifact = exp.run_experiment(tmp_path, "20260927", tmp_path / "blocked.json")
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["qwen_runner_ready_score"] == 0
    assert artifact["gate_check_summary"]
    assert len(artifact["rows"]) == len(artifact["preconditions_checked"])


def test_readiness_requires_exact_full_receipts(tmp_path):
    """SCENARIO-REPORT-7770-TERMINAL rejects an incomplete validation roster."""
    import shutil

    inputs = [
        exp.PROTOCOL,
        exp.PANEL,
        exp.OLD_ROWS,
        exp.OLD_RESULT,
        Path("results/experiment_7745_v674_qwen_localization.json"),
    ]
    for relative in inputs:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exp.ROOT / relative, target)
    receipt = tmp_path / exp.RAW / "validation_receipts.json"
    complete = {
        "coverage_percent": 100,
        "flagged_adversarial": False,
        "required_commands": [
            {"name": name, "exit_code": 0, "duration_s": 1.0}
            for name in sorted(exp.REQUIRED_CHECKS)
        ],
    }
    receipt.write_text(json.dumps(complete))
    output = tmp_path / "output.json"
    qualified = exp.run_experiment(tmp_path, "20260927", output)
    assert qualified["qwen_runner_ready_score"] == 1
    assert qualified["verdict_class"] == "circular_positive"
    full_suite = next(
        check for check in complete["required_commands"] if check["name"] == "full_python_suite"
    )
    full_suite["exit_code"] = 2
    receipt.write_text(json.dumps(complete))
    failed_suite = exp.run_experiment(tmp_path, "20260927", output)
    assert failed_suite["qwen_runner_ready_score"] == 0
    full_suite["exit_code"] = 0
    complete["required_commands"].pop()
    receipt.write_text(json.dumps(complete))
    unqualified = exp.run_experiment(tmp_path, "20260927", output)
    assert unqualified["qwen_runner_ready_score"] == 0
