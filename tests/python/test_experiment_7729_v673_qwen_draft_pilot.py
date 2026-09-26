"""REQ-REPORT-7729 and REQ-VERIFY-7729 contract tests."""

import json
from pathlib import Path
import hashlib
import copy
from types import SimpleNamespace

import pytest

from carnot import experiment_7729_v673_qwen_draft_pilot as pilot


def panel_row():
    return {
        "family_id": "family-a",
        "source": "The city is Paris. The river is Seine.",
        "answer": "The city is Paris. The river is Seine.",
        "annotation_types": [],
        "official_split": "test",
        "prior_exposure": True,
    }


def fake_reply(text, output_tokens=7, finish="stop"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 19, "completion_tokens": output_tokens},
    }


def test_scenario_report_7729_requests_keep_context_and_budget():
    row = panel_row()
    requests = [
        pilot.make_request(row, "direct_schema", "answer"),
        pilot.make_request(row, "draft_schema", "draft"),
        pilot.make_request(row, "draft_schema", "answer", "a draft"),
        pilot.make_request(row, "draft_plain", "draft"),
        pilot.make_request(row, "draft_plain", "answer", "a draft"),
    ]
    assert [request["max_tokens"] for request in requests] == [256, 128, 128, 128, 128]
    assert all(request["temperature"] == 0 for request in requests)
    assert all(request["seed"] == pilot.SEED for request in requests)
    assert all(row["source"] in json.dumps(request) for request in requests)
    assert all(row["answer"] in json.dumps(request) for request in requests)
    assert "response_format" in requests[0]
    assert "response_format" in requests[2]
    assert "response_format" not in requests[4]
    assert "a draft" in json.dumps(requests[2])
    assert len(pilot.arm_order(row["family_id"])) == 3
    assert set(pilot.arm_order(row["family_id"])) == set(pilot.ARMS)
    with pytest.raises(ValueError):
        pilot.make_request(row, "bogus", "answer")


def test_scenario_verify_7729_binary_and_quote_limits():
    source = "Paris is in France. Paris is in France."
    unsupported = ["Baseless Claim"]
    for decision in ("contradiction", "insufficient_evidence"):
        result = pilot.score_response(
            source, f'{{"decision":"{decision}","quote":"France"}}', "stop", unsupported
        )
        assert result["binary_accuracy"] is True
        assert result["three_way_accuracy"] is None
        assert result["quote_valid"] is False
        assert result["semantic_verified"] is False
    malformed = pilot.score_response(source, "bad", "length", unsupported)
    assert malformed["syntax_valid"] is False
    assert malformed["binary_accuracy"] is False
    assert malformed["missing_answer"] is True
    assert malformed["truncated"] is True
    supported = pilot.score_response(
        "Paris is in France.",
        "decision: support\nquote: Paris is in France.",
        "stop",
        [],
    )
    assert supported["syntax_valid"] is False
    assert supported["binary_accuracy"] is True
    assert supported["quote_valid"] is True
    assert supported["semantic_verified"] is False


def test_scenario_report_7729_scripted_transport_cold_replay(tmp_path):
    row = panel_row()
    calls = []

    def transport(request):
        calls.append(request)
        if "Write a short private draft" in request["messages"][0]["content"]:
            return fake_reply("A short draft about Paris.")
        if "response_format" in request:
            return fake_reply('{"decision":"support","quote":"The city is Paris."}')
        return fake_reply("decision: support\nquote: The city is Paris.")

    raw = tmp_path / "raw"
    raw.mkdir()
    pilot.custody.atomic_json(raw / "frozen_panel.json", [row])
    arms = pilot.execute_family(row, transport, raw, 0.0)
    assert len(calls) == 5
    assert len(arms) == 3
    assert {arm["arm"] for arm in arms} == set(pilot.ARMS)
    assert all(arm["metrics"]["binary_accuracy"] is True for arm in arms)
    assert all(len(arm["calls"]) == (1 if arm["arm"] == "direct_schema" else 2) for arm in arms)
    assert sum(call["output_tokens"] for arm in arms for call in arm["calls"]) == 35
    artifact = {
        "rows": arms,
        "paired_family_results": pilot.reduce_pairs(arms),
        "verdict_class": "null",
    }
    candidate = tmp_path / "candidate.json"
    pilot.custody.atomic_json(candidate, artifact)
    assert pilot.cold_reduce(candidate, raw)["passed"] is True
    request_path = Path(arms[0]["calls"][0]["request_path"])
    request_path.write_text("{}")
    assert pilot.cold_reduce(candidate, raw)["passed"] is False


def test_scenario_report_7729_pair_denominator_includes_failures():
    rows = [
        {
            "family_id": "a",
            "arm": arm,
            "metrics": {
                "binary_accuracy": arm != "draft_plain",
                "syntax_valid": arm != "draft_plain",
                "quote_valid": True,
                "unknown": False,
                "missing_answer": arm == "draft_plain",
                "truncated": False,
            },
            "input_tokens": 10,
            "output_tokens": 4,
            "latency_s": 0.5,
            "censored": False,
        }
        for arm in pilot.ARMS
    ]
    result = pilot.reduce_pairs(rows)
    assert result["paired_families"] == 1
    assert result["by_arm"]["draft_plain"]["binary_accuracy_numerator"] == 0
    assert result["by_arm"]["draft_plain"]["denominator"] == 1
    assert result["comparisons"]["draft_schema_vs_direct_schema"]["paired_n"] == 1


def test_scenario_report_7729_preflight_binds_prior_chat_template(tmp_path, monkeypatch):
    prior_path = tmp_path / pilot.prior.RESULT
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(
        json.dumps(
            {
                "current_model_receipts": {
                    "model_sha256": "sha256:model",
                    "server_props": {"chat_template": "template"},
                }
            }
        )
    )
    monkeypatch.setattr(
        pilot.prior,
        "preflight",
        lambda root, started: (
            [],
            {"flagged_historical_evidence": {}, "pre_gate_receipts": {}},
            {"model_sha256": "sha256:model"},
        ),
    )
    checks, hashes, context = pilot.preflight(tmp_path, 0.0)
    assert all(item["passed"] for item in checks)
    assert hashes["flagged_historical_evidence"]
    assert context["chat_template_sha256"] == hashlib.sha256(b"template").hexdigest()
    prior_path.unlink()
    checks, _, _ = pilot.preflight(tmp_path, 0.0)
    assert [item["check"] for item in checks if not item["passed"]] == [
        "prior_runtime_receipt",
        "same_model_bytes",
        "qwen_chat_template",
    ]


def test_scenario_report_7729_full_cpu_orchestration_and_blocked(tmp_path, monkeypatch):
    panel = [{**panel_row(), "family_id": f"family-{i:02d}"} for i in range(24)]
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    context = {
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "sha256:model",
        "selected": {"uuid": "gpu"},
    }
    monkeypatch.setattr(pilot, "preflight", lambda root, started: ([], hashes, context))
    monkeypatch.setattr(
        pilot.prior, "freeze_panel", lambda root, started: (panel, {"fresh_roster_count": 0})
    )

    def capture(root, families, actual_context, started):
        def transport(request):
            if "Write a short private draft" in request["messages"][0]["content"]:
                return fake_reply("a draft")
            if "response_format" in request:
                return fake_reply('{"decision":"support","quote":"The city is Paris."}')
            return fake_reply("decision: support\nquote: The city is Paris.")

        raw = root / pilot.RAW / "runs" / "scripted"
        raw.mkdir(parents=True)
        rows = [
            arm
            for family in families
            for arm in pilot.execute_family(family, transport, raw, started)
        ]
        return rows, {
            "model_load_attempted": 1,
            "generation_attempted": 120,
            "model_path": str(context["model_path"]),
            "model_sha256": "sha256:model",
            "device_uuid": "gpu",
        }

    monkeypatch.setattr(pilot, "owned_capture", capture)
    monkeypatch.setattr(
        pilot.validation,
        "build_scoped_commands",
        lambda *args, **kwargs: [
            pilot.validation.CommandSpec("focused_pytest", ("pytest",), "tests")
        ],
    )

    def commands(root, specs, **kwargs):
        if specs[0].name == "independent_cold_replay":
            assert pilot.cold_reduce(Path(specs[0].argv[-1]), root / pilot.RAW)["passed"]
        return [
            {
                "name": spec.name,
                "passed": True,
                "exit_code": 0,
                "command": list(spec.argv),
                "log_sha256": "sha256:fake",
            }
            for spec in specs
        ]

    monkeypatch.setattr(pilot.validation, "run_commands", commands)
    output = tmp_path / "result.json"
    assert pilot.run_experiment(tmp_path, "20260926", output) == 0
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "null"
    assert result["qwen_pilot_complete_score"] == 1
    assert result["invocation_counts"]["generations"] == 120
    assert result["paired_family_results"]["paired_families"] == 24
    assert result["fresh_generalization_eligible"] is False
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    assert pilot.main(["--cold-replay", str(tmp_path / pilot.RAW / "terminal_candidate.json")]) == 0

    blocked = pilot.gate("missing_model", "cache", tmp_path / "absent.gguf", "exists", True, False)
    monkeypatch.setattr(pilot, "preflight", lambda root, started: ([blocked], hashes, {}))
    blocked_output = tmp_path / "blocked.json"
    assert pilot.run_experiment(tmp_path, "20260926", blocked_output) == 0
    blocked_result = json.loads(blocked_output.read_text())
    assert blocked_result["honest_verdict"] == "complete_blocked_missing_model"
    assert blocked_result["model_invoked"] is False
    assert blocked_result["gate_check_summary"][0]["upstream_id"] == "cache"


@pytest.mark.parametrize("failure", [None, "recheck", "load", "offload", "template"])
def test_scenario_report_7729_owned_launch_lifecycle(tmp_path, monkeypatch, failure):
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot import experiment_7604_v664_evidence_pilot as v664
    from carnot import experiment_7581_v662_arc_bounded_canary as heartbeat
    from carnot import experiment_7431_v651_arc_live_sentinel as sentinel
    from carnot.agentic import arc_executable_world_model as arc
    from carnot import gpu_lease_phase_journal as journal

    leases = []

    class FakeLease:
        def __init__(self):
            self.document = {"phase": "preflight"}
            self.events = []
            leases.append(self)

        @classmethod
        def acquire(cls, **kwargs):
            return cls()

        def owner_receipt(self):
            return {"owned": True}

        def transition(self, phase, **kwargs):
            self.document["phase"] = phase
            self.events.append((phase, kwargs))

        def release(self):
            return {"released": True}

    class FakeProposer:
        def __init__(self, **kwargs):
            self._proc = SimpleNamespace(pid=123)
            self._stderr_log_path = None
            self.stopped = False

        def _ensure_server(self):
            return failure != "load"

        def server_props(self):
            return {"chat_template": "changed" if failure == "template" else "template"}

        def _url(self):
            return "http://fake"

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(journal, "GpuLease", FakeLease)
    monkeypatch.setattr(arc, "LocalGGUFProposer", FakeProposer)
    monkeypatch.setattr(sentinel, "_free_port", lambda: 18000)
    monkeypatch.setattr(ownership, "_current_inventory", lambda: {})
    monkeypatch.setattr(
        ownership, "recheck_before_launch", lambda *args: {"passed": failure != "recheck"}
    )
    monkeypatch.setattr(heartbeat, "_call_with_heartbeats", lambda operation, **kwargs: operation())
    monkeypatch.setattr(heartbeat, "_owned_vram_mb", lambda pid: 18000)
    monkeypatch.setattr(
        heartbeat,
        "_observed_offload_layers",
        lambda path: {"loaded_layers": 0 if failure == "offload" else 40},
    )
    monkeypatch.setattr(heartbeat, "process_start_tick", lambda pid: 42)
    monkeypatch.setattr(v664, "_runtime_build_receipt", lambda: {"build": "scripted"})
    monkeypatch.setattr(
        v664,
        "_post_json",
        lambda url, request, timeout: json.dumps(
            fake_reply(
                "draft"
                if "Write a short private draft" in request["messages"][0]["content"]
                else '{"decision":"support","quote":"The city is Paris."}'
            )
        ).encode(),
    )
    context = {
        "selected": {"uuid": "gpu", "index": 0, "memory_used_mb": 0},
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "sha256:model",
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
        "registry": object(),
    }
    if failure is None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior")
        panel = [{**panel_row(), "family_id": f"family-{i:02d}"} for i in range(24)]
        rows, runtime = pilot.owned_capture(tmp_path, panel, context, 0.0)
        assert len(rows) == 72
        assert runtime["generation_attempted"] == 120
        assert runtime["offload_layers"]["actual_offload"] is True
        assert leases[0].document["phase"] == "terminal_complete"
        assert pilot.os.environ["CUDA_VISIBLE_DEVICES"] == "prior"
    else:
        with pytest.raises(RuntimeError):
            pilot.owned_capture(tmp_path, [panel_row()], context, 0.0)
        assert leases[0].document["phase"] == "terminal_blocked"


def test_scenario_report_7729_cold_reader_rejects_tampering(tmp_path):
    row = panel_row()
    raw = tmp_path / "raw"
    raw.mkdir()
    pilot.custody.atomic_json(raw / "frozen_panel.json", [row])

    def transport(request):
        return fake_reply(
            "draft"
            if "Write a short private draft" in request["messages"][0]["content"]
            else '{"decision":"support","quote":"The city is Paris."}'
        )

    rows = pilot.execute_family(row, transport, raw, 0.0)
    baseline = {
        "rows": rows,
        "paired_family_results": pilot.reduce_pairs(rows),
        "verdict_class": "null",
    }
    candidate = tmp_path / "candidate.json"

    def checked(changed, reason):
        pilot.custody.atomic_json(candidate, changed)
        assert pilot.cold_reduce(candidate, raw)["reason"] == reason

    changed = copy.deepcopy(baseline)
    changed["rows"][0]["family_id"] = "alien"
    checked(changed, "family_arm_identity")
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["calls"] = []
    checked(changed, "call_count")
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["metrics"]["binary_accuracy"] = False
    checked(changed, "metric_mismatch")
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["calls"][0]["output_tokens"] = 300
    checked(changed, "token_mismatch")
    call = baseline["rows"][0]["calls"][0]
    request_path = Path(call["request_path"])
    original_request = request_path.read_bytes()
    request_path.write_text("{}")
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["calls"][0]["request_sha256"] = pilot.custody.sha256_file(request_path)
    checked(changed, "request_mismatch")
    request_path.write_bytes(original_request)
    response_path = Path(call["raw_response_path"])
    original_response = response_path.read_bytes()
    response = json.loads(original_response)
    response["choices"][0]["message"]["content"] = "altered"
    response_path.write_text(json.dumps(response))
    changed = copy.deepcopy(baseline)
    changed["rows"][0]["calls"][0]["raw_response_sha256"] = pilot.custody.sha256_file(response_path)
    checked(changed, "response_mismatch")
    response_path.write_bytes(original_response)
    response = json.loads(original_response)
    response["usage"]["completion_tokens"] = 300
    response_path.write_text(json.dumps(response))
    changed = copy.deepcopy(baseline)
    changed_call = changed["rows"][0]["calls"][0]
    changed_call["raw_response_sha256"] = pilot.custody.sha256_file(response_path)
    changed_call["output_tokens"] = 300
    checked(changed, "token_budget")
    response_path.write_bytes(original_response)


@pytest.mark.parametrize(
    "mode", ["panel_error", "capture_error", "validation_error", "reader_error"]
)
def test_scenario_report_7729_failure_custody(tmp_path, monkeypatch, mode):
    panel = [{**panel_row(), "family_id": f"family-{i:02d}"} for i in range(24)]
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    context = {
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "sha256:model",
        "selected": {"uuid": "gpu"},
    }
    check = pilot.gate("missing_model", "cache", tmp_path / "missing", "exists", True, False)
    monkeypatch.setattr(
        pilot,
        "preflight",
        lambda root, started: (([check] if mode == "reader_error" else []), hashes, context),
    )

    def frozen(root, started):
        if mode == "panel_error":
            raise ValueError("bad release")
        return panel, {"fresh_roster_count": 0}

    monkeypatch.setattr(pilot.prior, "freeze_panel", frozen)

    def capture(root, families, context, started):
        if mode == "capture_error":
            checkpoint = root / pilot.RAW / "runs" / f"1-{pilot.os.getpid()}" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            pilot.custody.atomic_json(checkpoint, {"rows": []})
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(pilot, "owned_capture", capture)
    monkeypatch.setattr(
        pilot.validation,
        "build_scoped_commands",
        lambda *args, **kwargs: [
            pilot.validation.CommandSpec("focused_pytest", ("pytest",), "tests")
        ],
    )

    def commands(root, specs, **kwargs):
        return [
            {
                "name": spec.name,
                "passed": (
                    not (mode == "validation_error" and spec.name == "focused_pytest")
                    and not (mode == "reader_error" and spec.name == "adversarial_verify")
                ),
                "exit_code": 0,
                "log_sha256": "sha256:scripted",
            }
            for spec in specs
        ]

    monkeypatch.setattr(pilot.validation, "run_commands", commands)
    output = tmp_path / "result.json"
    exit_code = pilot.run_experiment(tmp_path, "20260926", output)
    result = json.loads(output.read_text())
    assert result["verdict_class"] == (
        "disqualified"
        if mode in {"validation_error", "reader_error"}
        else "blocked"
        if mode == "panel_error"
        else "partial"
    )
    assert exit_code == int(result["verdict_class"] == "disqualified")
    if mode == "reader_error":
        assert result["flagged_adversarial"] is True
    if mode == "capture_error":
        assert result["gate_check_summary"][0]["check"] == "owned_capture"


def test_scenario_report_7729_cli_and_date_guard(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="date_must"):
        pilot.run_experiment(tmp_path, "wrong", tmp_path / "result.json")
    monkeypatch.setattr(pilot, "run_experiment", lambda root, date, output: 7)
    assert pilot.main(["--date", "20260926", "--output", str(tmp_path / "out.json")]) == 7
    with pytest.raises(ValueError, match="draft_required"):
        pilot.make_request(panel_row(), "draft_schema", "answer")
