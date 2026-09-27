"""REQ-REPORT-7745 and REQ-VERIFY-7745 paired transport contracts."""

import json
import hashlib
import os
from pathlib import Path

import pytest

from carnot import experiment_7745_v674_qwen_localization as pilot


@pytest.fixture
def family():
    return {
        "family_id": "family-1",
        "source_id": "source-1",
        "response_id": "response-1",
        "source": "Blue paint is dry. Red paint is wet.",
        "answer": "Blue paint is wet. Red paint is wet.",
        "annotation_types": ["Evident Conflict"],
        "sentence_targets": [1, 0],
        "sentence_offsets": [[0, 19], [19, 37]],
        "official_split": "train",
        "prior_exposure": True,
    }


def response(payload, finish="stop", tokens=25):
    return {
        "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 50, "completion_tokens": tokens},
    }


def test_requests_match_budgets_and_hide_labels(family):
    """SCENARIO-REPORT-7745-PAIRED: same input and decoding budget."""
    direct = pilot.make_request(family, "direct")
    local = pilot.make_request(family, "localized")
    assert direct["max_tokens"] == local["max_tokens"] == 256
    assert direct["temperature"] == local["temperature"] == 0
    assert direct["seed"] == local["seed"] == 67445
    assert direct["response_format"] == local["response_format"]
    assert direct["messages"][1] == local["messages"][1]
    assert "sentence_targets" not in json.dumps(direct)
    assert "annotation_types" not in json.dumps(local)
    with pytest.raises(ValueError, match="unplanned_arm"):
        pilot.make_request(family, "draft")
    assert set(pilot.arm_order("family-1")) == {"direct", "localized"}


def test_exact_spans_and_semantic_separation(family):
    """SCENARIO-VERIFY-7745-OFFSETS: valid addresses are not entailment."""
    content = json.dumps(
        {
            "decision": "unsupported",
            "quote": "Blue paint is dry.",
            "unsupported_spans": [{"text": "Blue paint is wet.", "quote": "Blue paint is dry."}],
        }
    )
    metrics = pilot.score_response(family, "localized", content, "stop")
    assert metrics["syntax_valid"]
    assert metrics["quote_valid"]
    assert metrics["span_address_valid"]
    assert metrics["binary_accuracy"]
    assert metrics["localization_tp"] == 1
    assert metrics["localization_fp"] == 0
    assert metrics["localization_fn"] == 0
    assert metrics["semantic_verified"] is False
    malformed = pilot.score_response(family, "localized", "bad JSON", "length")
    assert malformed["binary_accuracy"] is False
    assert malformed["truncated"] and malformed["censored"]
    assert malformed["localization_fn"] == 1


def test_ambiguous_quote_abstention_and_false_accept(family):
    """SCENARIO-VERIFY-7745-OFFSETS: unresolved evidence never becomes proof."""
    repeated = {**family, "source": "wet wet"}
    abstain = pilot.score_response(
        repeated, "direct", json.dumps({"decision": "abstain", "quote": "wet"}), "stop"
    )
    assert abstain["unknown"] and not abstain["quote_valid"]
    assert not abstain["binary_accuracy"]
    accept = pilot.score_response(
        family,
        "direct",
        json.dumps({"decision": "supported", "quote": "Blue paint is dry."}),
        "stop",
    )
    assert accept["false_accept"] and not accept["false_reject"]


def test_transport_bytes_and_cold_replay(tmp_path, family):
    """SCENARIO-REPORT-7745-PAIRED: actual request path and cold byte replay."""
    seen = []

    def transport(request):
        seen.append(request)
        payload = {"decision": "unsupported", "quote": "Blue paint is dry."}
        if "unsupported_spans" in request["messages"][0]["content"]:
            payload["unsupported_spans"] = [
                {"text": "Blue paint is wet.", "quote": "Blue paint is dry."}
            ]
        return json.dumps(response(payload)).encode()

    rows = pilot.execute_family(family, transport, tmp_path, 0.0)
    assert len(rows) == len(seen) == 2
    assert {row["arm"] for row in rows} == {"direct", "localized"}
    assert all(row["denominator"] == 1 and row["output_tokens"] == 25 for row in rows)
    paired = pilot.reduce_pairs(rows)
    assert paired["paired_families"] == 1
    assert paired["by_arm"]["localized"]["binary_accuracy_numerator"] == 1
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps([family]))
    assert pilot.cold_reduce_rows(rows, panel)["passed"]
    raw = tmp_path / "family-1_direct_response.json"
    raw.write_text("{}")
    assert pilot.cold_reduce_rows(rows, panel)["passed"] is False


def test_parser_invalid_shapes_and_censored_transport(tmp_path, family):
    """SCENARIO-VERIFY-7745-OFFSETS: bad structures and errors stay measured."""
    bad = [
        {"decision": "unsupported", "quote": "Blue paint is dry.", "unsupported_spans": [4]},
        {
            "decision": "unsupported",
            "quote": "Blue paint is dry.",
            "unsupported_spans": [{"text": "missing", "quote": "missing"}],
        },
        {"decision": "unsupported", "quote": "Blue paint is dry.", "unsupported_spans": "wrong"},
    ]
    assert not pilot.score_response(family, "localized", json.dumps(bad[0]), "stop")[
        "span_address_valid"
    ]
    assert not pilot.score_response(family, "localized", json.dumps(bad[1]), "stop")[
        "span_address_valid"
    ]
    assert not pilot.score_response(family, "localized", json.dumps(bad[2]), "stop")["syntax_valid"]
    assert not pilot.score_response(
        family, "direct", json.dumps({"decision": [], "quote": "x"}), "stop"
    )["syntax_valid"]
    assert pilot.score_response(family, "direct", "{}", "timeout")["censored"]
    supported = {**family, "annotation_types": []}
    rejected = pilot.score_response(
        supported,
        "direct",
        json.dumps({"decision": "unsupported", "quote": "Blue paint is dry."}),
        "stop",
    )
    assert rejected["false_reject"]

    def broken(_):
        raise OSError("transport down")

    rows = pilot.execute_family(family, broken, tmp_path, 0.0)
    assert len(rows) == 2 and all(row["censored"] for row in rows)
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps([family]))
    assert pilot.cold_reduce_rows(rows, panel)["passed"]
    assert pilot.reduce_pairs([])["paired_families"] == 0
    assert pilot.cold_reduce_rows(rows + rows[:1], panel)["reason"] == "family_arm_identity"
    row = rows[0]
    original = json.loads(Path(row["calls"][0]["request_path"]).read_text())
    original["seed"] = 1
    Path(row["calls"][0]["request_path"]).write_text(json.dumps(original))
    row["calls"][0]["request_sha256"] = pilot.custody.sha256_file(
        Path(row["calls"][0]["request_path"])
    )
    assert pilot.cold_reduce_rows(rows, panel)["reason"] == "request_mismatch"
    original["seed"] = pilot.SEED
    Path(row["calls"][0]["request_path"]).write_text(json.dumps(original))
    row["calls"][0]["request_sha256"] = pilot.custody.sha256_file(
        Path(row["calls"][0]["request_path"])
    )
    row["metrics"]["binary_accuracy"] = True
    assert pilot.cold_reduce_rows(rows, panel)["reason"] == "metric_or_token_mismatch"
    row["metrics"]["binary_accuracy"] = False
    row["source_sha256"] = "wrong"
    assert pilot.cold_reduce_rows(rows, panel)["reason"] == "row_identity_or_totals"


def test_build_artifact_classes_and_cold_reduction(tmp_path, family):
    """SCENARIO-REPORT-7745-TERMINAL: a null and a block have distinct gates."""
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    failure = pilot.prior.gate("cache", "local_model_cache", tmp_path, "present", True, False)
    blocked = pilot.build_artifact([], [failure], hashes, {}, [], 0.1, "20260927")
    assert blocked["honest_verdict"] == "complete_blocked_cache"
    assert blocked["MODEL_SPECS"] == []
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked))
    assert pilot.cold_reduce(path)["passed"]
    partial = pilot.build_artifact([], [], hashes, {"model_load_attempted": 1}, [], 1, "20260927")
    assert partial["verdict_class"] == "partial"
    assert partial["MODEL_SPECS"] == [pilot.MODEL_ID]
    rows = []
    for index in range(24):
        member = {**family, "family_id": f"f{index}"}
        rows.extend(
            pilot.execute_family(
                member,
                lambda _: response({"decision": "supported", "quote": "Blue paint is dry."}),
                tmp_path,
                0.0,
            )
        )
    done = pilot.build_artifact(
        rows,
        [],
        hashes,
        {"model_load_attempted": 1, "model_load_completed": 1, "generation_attempted": 48},
        [],
        12,
        "20260927",
    )
    assert done["verdict_class"] == "null"
    assert done["paired_family_results"]["paired_families"] == 24
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps([{**family, "family_id": f"f{index}"} for index in range(24)]))
    path.write_text(json.dumps(done))
    panel.rename(tmp_path / "frozen_panel.json")
    assert pilot.cold_reduce(path, tmp_path)["passed"]
    done["paired_family_results"]["paired_families"] = 0
    path.write_text(json.dumps(done))
    assert pilot.cold_reduce(path, tmp_path)["reason"] == "paired_reduction_mismatch"


def test_prepare_panel_authenticates_original_bytes(tmp_path, monkeypatch, family):
    """SCENARIO-REPORT-7745-PAIRED: rebuild source and labels from release."""
    from carnot.inference import sota_models
    from carnot import experiment_7423_v651_annotated_protocol as release
    from carnot import experiment_7740_v674_sentence_label_protocol as labels

    panel = [{**family, "family_id": f"family-{i}"} for i in range(24)]
    previous = tmp_path / "results/raw/experiment_7729_v673_qwen_draft_pilot/frozen_panel.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps(panel))
    history = tmp_path / pilot.prior.RESULT
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("{}")
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    context = {"model_path": Path("/tmp/cached.gguf")}
    monkeypatch.setattr(pilot.prior, "preflight", lambda *_: ([], hashes, context))
    monkeypatch.setattr(pilot.prior.prior, "freeze_panel", lambda *_: (panel, {}))
    monkeypatch.setattr(
        sota_models, "cached_current_model", lambda: {"model_path": "/tmp/cached.gguf"}
    )
    monkeypatch.setattr(release, "authenticate_assets", lambda *_: {})
    monkeypatch.setattr(
        release, "load_release", lambda *_, **__: ([], [{"id": "response-1", "labels": []}])
    )
    monkeypatch.setattr(
        labels,
        "map_targets",
        lambda *_: {"targets": [0, 1], "char_offsets": [[0, 19], [19, 37]], "reason": "mapped"},
    )
    rebuilt, checks, result_hashes, _ = pilot.prepare_panel(tmp_path, 0)
    assert len(rebuilt) == 24 and all(check["passed"] for check in checks)
    assert rebuilt[0]["sentence_targets"] == [0, 1]
    assert result_hashes["valid_producers"][str(previous.relative_to(tmp_path))]
    monkeypatch.setattr(sota_models, "cached_current_model", lambda: None)
    rebuilt, checks, _, _ = pilot.prepare_panel(tmp_path, 0)
    assert rebuilt == [] and checks[-1]["check"] == "cached_current_model"


def test_owned_capture_uses_owned_lease_and_real_transport_shape(tmp_path, monkeypatch, family):
    """SCENARIO-REPORT-7745-TERMINAL: fake server tests lifecycle only."""
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot import experiment_7604_v664_evidence_pilot as v664
    from carnot.agentic import arc_executable_world_model as world
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot import experiment_7581_v662_arc_bounded_canary as canary
    from carnot import gpu_lease_phase_journal as journal

    class FakeLease:
        def __init__(self):
            self.document = {"phase": "preflight"}
            self.phases = []

        def owner_receipt(self):
            return {"owner": "task"}

        def transition(self, phase, **_):
            self.document["phase"] = phase
            self.phases.append(phase)

        def release(self):
            return {"released": True, "phase": self.document["phase"]}

    lease = FakeLease()
    monkeypatch.setattr(journal.GpuLease, "acquire", lambda **_: lease)
    monkeypatch.setattr(ownership, "recheck_before_launch", lambda *_: {"passed": True})
    monkeypatch.setattr(ownership, "_current_inventory", lambda: [])
    monkeypatch.setattr(canary, "_call_with_heartbeats", lambda fn, **_: fn())
    monkeypatch.setattr(canary, "_owned_vram_mb", lambda *_: 18000)
    monkeypatch.setattr(canary, "_observed_offload_layers", lambda *_: 66)
    monkeypatch.setattr(canary, "process_start_tick", lambda *_: 123)
    monkeypatch.setattr(v664, "_offload_receipt", lambda *_: {"actual_offload": True})
    monkeypatch.setattr(v664, "_runtime_build_receipt", lambda: {"build": "fake"})
    monkeypatch.setattr(
        v664,
        "_post_json",
        lambda *_: json.dumps(
            response({"decision": "supported", "quote": "Blue paint is dry."})
        ).encode(),
    )
    monkeypatch.setattr("carnot.experiment_7431_v651_arc_live_sentinel._free_port", lambda: 12345)

    class FakeProposer:
        last = None

        def __init__(self, **_):
            self._proc = type("Proc", (), {"pid": 456})()
            self._stderr_log_path = None
            self.stopped = False
            FakeProposer.last = self

        def _ensure_server(self):
            return True

        def server_props(self):
            return {"chat_template": "template"}

        def _url(self):
            return "http://localhost:12345"

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(world, "LocalGGUFProposer", FakeProposer)
    context = {
        "selected": {"uuid": "GPU-test", "index": 0, "memory_used_mb": 5},
        "registry": object(),
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "hash",
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
    }
    panel = [{**family, "family_id": f"f{i}"} for i in range(24)]
    rows, runtime = pilot.owned_capture(tmp_path, panel, context, 0.0)
    assert len(rows) == 48 and runtime["generation_attempted"] == 48
    assert runtime["lease_release"]["phase"] == "terminal_complete"
    assert FakeProposer.last.stopped
    assert runtime["server_pid"] == 456
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "previous")
    monkeypatch.setattr(ownership, "recheck_before_launch", lambda *_: {"passed": False})
    with pytest.raises(RuntimeError, match="foreign_or_capacity_recheck_failed"):
        pilot.owned_capture(tmp_path, panel[:1], context, 0.0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "previous"
    monkeypatch.setattr(ownership, "recheck_before_launch", lambda *_: {"passed": True})
    monkeypatch.setattr(FakeProposer, "_ensure_server", lambda *_: False)
    with pytest.raises(RuntimeError, match="owned_qwen_load_failed"):
        pilot.owned_capture(tmp_path, panel[:1], context, 0.0)
    monkeypatch.setattr(FakeProposer, "_ensure_server", lambda *_: True)
    monkeypatch.setattr(v664, "_offload_receipt", lambda *_: {"actual_offload": False})
    with pytest.raises(RuntimeError, match="qwen_gpu_offload_not_authenticated"):
        pilot.owned_capture(tmp_path, panel[:1], context, 0.0)
    monkeypatch.setattr(v664, "_offload_receipt", lambda *_: {"actual_offload": True})
    with pytest.raises(RuntimeError, match="chat_template_changed_after_load"):
        pilot.owned_capture(tmp_path, panel[:1], {**context, "chat_template_sha256": "wrong"}, 0.0)


def test_producer_private_e2e_and_blocked_terminal(tmp_path, monkeypatch, family):
    """SCENARIO-REPORT-7745-TERMINAL: cold candidate gates publication."""
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    panel = [{**family, "family_id": f"f{i}"} for i in range(24)]
    context = {
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "hash",
        "selected": {"uuid": "GPU-fake"},
    }
    monkeypatch.setattr(pilot, "prepare_panel", lambda *_: (panel, [], hashes, context))
    monkeypatch.setattr(pilot.validation, "build_scoped_commands", lambda *_, **__: [])

    def run_commands(_, commands, *, log_dir, **__):
        if "affected" in str(log_dir):
            return [{"name": "focused_pytest", "passed": True, "exit_code": 0}]
        candidate = tmp_path / pilot.RAW / "terminal_candidate.json"
        assert pilot.cold_reduce(candidate, tmp_path / pilot.RAW)["passed"]
        return [{"name": command.name, "passed": True, "exit_code": 0} for command in commands]

    monkeypatch.setattr(pilot.validation, "run_commands", run_commands)

    def fake_capture(root, rows, _, started):
        run_dir = root / pilot.RAW / "fake_run"
        run_dir.mkdir(parents=True)
        output = []
        for row in rows:
            output.extend(
                pilot.execute_family(
                    row,
                    lambda _: response({"decision": "supported", "quote": "Blue paint is dry."}),
                    run_dir,
                    started,
                )
            )
        return output, {
            "model_load_attempted": 1,
            "model_load_completed": 1,
            "generation_attempted": 48,
            "model_sha256": "hash",
            "model_path": str(context["model_path"]),
            "device_uuid": "GPU-fake",
        }

    monkeypatch.setattr(pilot, "owned_capture", fake_capture)
    output = tmp_path / "final.json"
    assert pilot.run_experiment(tmp_path, "20260927", output) == 0
    final = json.loads(output.read_text())
    assert final["qwen_localization_complete_score"] == 1
    assert final["verdict_class"] == "null"
    assert len(final["validation_receipts"]["terminal_readers"]) == 3
    assert pilot.cold_reduce(output, tmp_path / pilot.RAW)["passed"]

    failure = pilot.prior.gate(
        "missing_source", "RAGTruth", tmp_path / "absent", "bytes", True, False
    )
    monkeypatch.setattr(pilot, "prepare_panel", lambda *_: ([], [failure], hashes, {}))
    assert pilot.run_experiment(tmp_path, "20260927", output) == 0
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["upstream_id"] == "RAGTruth"
    assert blocked["qwen_localization_complete_score"] == 0


def test_producer_validation_and_reader_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7745-TERMINAL: failed owned checks disqualify."""
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    failure = pilot.prior.gate("missing", "RAGTruth", tmp_path, "bytes", True, False)
    monkeypatch.setattr(pilot, "prepare_panel", lambda *_: ([], [failure], hashes, {}))
    monkeypatch.setattr(pilot.validation, "build_scoped_commands", lambda *_, **__: [])

    def failed_commands(_, commands, *, log_dir, **__):
        if "affected" in str(log_dir):
            return [{"name": "coverage", "passed": False, "exit_code": 1}]
        return [
            {
                "name": command.name,
                "passed": command.name != "adversarial_verify",
                "exit_code": int(command.name == "adversarial_verify"),
            }
            for command in commands
        ]

    monkeypatch.setattr(pilot.validation, "run_commands", failed_commands)
    output = tmp_path / "final.json"
    assert pilot.run_experiment(tmp_path, "20260927", output) == 1
    artifact = json.loads(output.read_text())
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"]
    assert not artifact["acceptance_gate_results"]["readiness"]["passed"]
    with pytest.raises(ValueError, match="date_must_be_20260927"):
        pilot.run_experiment(tmp_path, "20260926", output)


def test_producer_catches_source_and_owned_capture_errors(tmp_path, monkeypatch, family):
    """SCENARIO-REPORT-7745-TERMINAL: real errors retain bounded custody."""
    hashes = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    monkeypatch.setattr(pilot.validation, "build_scoped_commands", lambda *_, **__: [])

    def run_commands(_, commands, *, log_dir, **__):
        if "affected" in str(log_dir):
            return [{"name": "focused_pytest", "passed": True, "exit_code": 0}]
        return [{"name": command.name, "passed": True, "exit_code": 0} for command in commands]

    monkeypatch.setattr(pilot.validation, "run_commands", run_commands)
    monkeypatch.setattr(
        pilot, "prepare_panel", lambda *_: (_ for _ in ()).throw(ValueError("bad source"))
    )
    output = tmp_path / "final.json"
    assert pilot.run_experiment(tmp_path, "20260927", output) == 0
    assert (
        json.loads(output.read_text())["gate_check_summary"][0]["check"] == "source_reconstruction"
    )

    panel = [{**family, "family_id": "f0"}]
    context = {
        "model_path": tmp_path / "model.gguf",
        "model_sha256": "hash",
        "selected": {"uuid": "GPU-fake"},
    }
    monkeypatch.setattr(pilot, "prepare_panel", lambda *_: (panel, [], hashes, context))
    run_dir = tmp_path / pilot.RAW / "runs" / f"1-{os.getpid()}"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.json").write_text(json.dumps({"rows": []}))
    monkeypatch.setattr(
        pilot, "owned_capture", lambda *_: (_ for _ in ()).throw(RuntimeError("load failed"))
    )
    assert pilot.run_experiment(tmp_path, "20260927", output) == 0
    incomplete = json.loads(output.read_text())
    assert incomplete["verdict_class"] == "partial"
    assert incomplete["gate_check_summary"][0]["check"] == "owned_capture"
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    assert pilot.phase_span("test", 0.0, 0.0, 1, checkpoint, "20260927")["checkpoint_sha256"]


def test_main_read_only_routes(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7745-TERMINAL: CLI cold replay is read only."""
    monkeypatch.setattr(pilot, "cold_reduce", lambda *_: {"passed": True})
    assert pilot.main(["--cold-replay", str(tmp_path / "candidate.json")]) == 0
    monkeypatch.setattr(pilot, "cold_reduce", lambda *_: {"passed": False})
    assert pilot.main(["--cold-replay", str(tmp_path / "candidate.json")]) == 1
    monkeypatch.setattr(pilot, "run_experiment", lambda *args: 7)
    assert pilot.main(["--date", "20260927"]) == 7
