"""REQ-REPORT-7759 and SCENARIO-REPORT-7759-* behavior checks."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7759_v675_qwen_evidence_views as exp


@pytest.fixture
def family():
    """Keep one original byte pair and a known human binary label."""
    return {
        "family_id": "family-a",
        "source": "First sentence. Second one! Third item?",
        "answer": "An answer.",
        "annotation_types": ["unsupported"],
        "official_split": "train",
    }


def test_view_requests_preserve_bytes_and_only_change_windows(family):
    """SCENARIO-REPORT-7759-VIEWS keeps complete evidence in both arms."""
    canonical = exp.make_request(family, "canonical")
    paired = exp.make_request(family, "paired")
    withheld = exp.make_request(family, "source_withheld")
    a = json.loads(canonical["messages"][1]["content"])
    b = json.loads(paired["messages"][1]["content"])
    c = json.loads(withheld["messages"][1]["content"])
    assert canonical["messages"][0] == paired["messages"][0] == withheld["messages"][0]
    assert a["complete_source"] == b["complete_source"] == family["source"]
    assert a["original_answer"] == b["original_answer"] == c["original_answer"]
    assert [x["text"] for x in a["source_windows"]] == [
        "First sentence. ",
        "Second one! ",
        "Third item?",
        family["source"],
    ]
    assert [x["text"] for x in b["source_windows"]][-2:] == [
        "First sentence. Second one! ",
        "Second one! Third item?",
    ]
    assert "complete_source" not in c and not c["source_windows"]
    assert family["source"] not in withheld["messages"][1]["content"]
    assert canonical["seed"] == paired["seed"] == withheld["seed"]
    assert canonical["max_tokens"] == 256


def test_probability_parsing_keeps_invalid_in_denominator(family):
    """SCENARIO-REPORT-7759-DENOMINATOR rejects malformed and censored replies."""
    good = '{"decision":"unsupported","probability_unsupported":0.8,"quote":"First sentence."}'
    valid = exp.parse_reply(family, "canonical", good, "stop")
    assert valid["parse_valid"] and valid["brier"] == pytest.approx(0.04)
    assert valid["typed_decision"] == "unsupported"
    for text, finish in (
        ("bad", "stop"),
        (good, "length"),
        ('{"decision":"unsupported","probability_unsupported":true,"quote":""}', "stop"),
    ):
        invalid = exp.parse_reply(family, "canonical", text, finish)
        assert invalid["brier"] is None
        assert invalid["typed_decision"] is None
    sensitivity = exp.parse_reply(family, "source_withheld", good, "stop")
    assert sensitivity["human_binary_unsupported"] is None
    assert sensitivity["brier"] is None


def test_reduce_counts_families_and_ignores_invalid_brier(family):
    """SCENARIO-REPORT-7759-DENOMINATOR uses families as independent units."""
    rows = []
    for arm, probability in (("canonical", 0.8), ("paired", 0.6), ("source_withheld", 0.4)):
        metrics = exp.parse_reply(
            family,
            arm,
            json.dumps(
                {
                    "decision": "unsupported",
                    "probability_unsupported": probability,
                    "quote": "First sentence.",
                }
            ),
            "stop",
        )
        rows.append(
            {
                "family_id": "family-a",
                "arm": arm,
                "metrics": metrics,
                "disposition": "completed",
                "input_tokens": 10,
                "output_tokens": 10,
                "elapsed_s": 1.0,
            }
        )
    result = exp.reduce_rows(rows)
    assert result["effective_independent_n"] == 1
    assert result["paired_probability_disagreement"] == pytest.approx(0.2)
    assert result["by_arm"]["canonical"]["brier"] == pytest.approx(0.04)
    assert result["by_arm"]["source_withheld"]["brier"] is None
    rows[0]["metrics"] = exp.parse_reply(family, "canonical", "garbage", "stop")
    result = exp.reduce_rows(rows)
    assert result["by_arm"]["canonical"]["denominator"] == 1
    assert result["by_arm"]["canonical"]["brier"] is None
    assert result["paired_probability_disagreement"] is None


def test_private_basetemp_parent_exists_in_real_child(tmp_path):
    """REQ-REPORT-7759 establishes nested pytest parents before child launch."""
    parent = tmp_path / "nested" / "pytest"
    exp.prepare_basetemp(parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.is_dir(); (p/'child').mkdir()",
            str(parent),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (parent / "child").is_dir()


def test_prompt_budget_marks_unstarted_without_truncation(family, tmp_path):
    """SCENARIO-REPORT-7759-VIEWS bounds a complete request before dispatch."""
    sent = []
    rows = exp.capture_family(
        family, lambda request: sent.append(request), tmp_path, 0.0, max_prompt_bytes=5
    )
    assert not sent
    assert len(rows) == 3
    assert all(row["disposition"] == "unstarted_prompt_over_budget" for row in rows)
    assert all(row["raw_request_path"] is None for row in rows)


def reply(probability=0.8, finish="stop"):
    """Give the capture path one OpenAI-compatible transport reply."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "decision": "unsupported",
                            "probability_unsupported": probability,
                            "quote": "First sentence.",
                        }
                    )
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 12},
    }


def test_capture_raw_bytes_and_cold_replay(family, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7759-REPLAY rejects changed requests, replies, and metrics."""
    seen = []

    def transport(request):
        seen.append(request)
        return json.dumps(reply()).encode()

    rows = exp.capture_family(family, transport, tmp_path / "raw", 0.0)
    assert len(rows) == len(seen) == 3
    assert all(row["output_tokens"] == 12 for row in rows)
    panel = tmp_path / exp.prior.RAW / "frozen_panel.json"
    panel.parent.mkdir(parents=True)
    panel.write_text(json.dumps([family]))
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    artifact = {
        "verdict_class": "null",
        "rows": rows,
        "paired_family_results": exp.reduce_rows(rows),
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert exp.cold_reduce(path)["passed"]
    assert exp.cold_reduce(path)["calls"] == 3
    assert exp.terminal_commands(tmp_path, path)[-1].name == "verdict_row_consistency_strict"
    duplicate = {**artifact, "rows": rows + rows[:1]}
    path.write_text(json.dumps(duplicate))
    assert exp.cold_reduce(path)["reason"] == "row_identity"
    path.write_text(json.dumps(artifact))
    raw = Path(rows[0]["raw_response_path"])
    original = raw.read_bytes()
    raw.write_bytes(b"{}")
    assert exp.cold_reduce(path)["reason"] == "raw_hash_or_request"
    raw.write_bytes(original)
    rows[0]["metrics"]["parse_valid"] = False
    path.write_text(json.dumps(artifact))
    assert exp.cold_reduce(path)["reason"] == "raw_metric_or_token"
    rows[0]["metrics"]["parse_valid"] = True
    artifact["paired_family_results"]["effective_independent_n"] = 0
    path.write_text(json.dumps(artifact))
    assert exp.cold_reduce(path)["reason"] == "aggregate_or_roster"


def test_capture_transport_failure_and_all_unstarted(family, tmp_path):
    """SCENARIO-REPORT-7759-DENOMINATOR keeps failures and blocked launches."""

    def broken(_):
        raise OSError("transport down")

    rows = exp.capture_family(family, broken, tmp_path / "error", 0.0)
    assert len(rows) == 3 and all(row["censored"] for row in rows)
    assert all(row["disposition"] == "censored_transport_error" for row in rows)
    assert all(Path(row["raw_response_path"]).is_file() for row in rows)
    expired = exp.capture_family(family, broken, tmp_path / "expired", 0.0, deadline=0)
    assert all(row["disposition"] == "unstarted_time_budget" for row in expired)
    exhausted = exp.capture_family(family, broken, tmp_path / "exhausted", 0.0, output_left=0)
    assert all(row["disposition"] == "unstarted_output_budget" for row in exhausted)


def test_artifact_claim_classes_and_gate_operands(family, tmp_path):
    """REQ-REPORT-7759 separates external blocks, nulls, and owned partial work."""
    failure = exp.gate("missing", "exp7745", tmp_path / "absent", "exists", True, False)
    blocked = exp.build_artifact([], [failure], {}, {}, [], 1.0, "20260927")
    assert blocked["honest_verdict"] == "complete_blocked_missing"
    assert blocked["MODEL_SPECS"] == []
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked))
    assert exp.cold_reduce(path)["passed"]
    partial = exp.build_artifact([], [], {}, {"model_load_attempted": 1}, [], 2.0, "20260927")
    assert partial["verdict_class"] == "partial"
    assert partial["model_specs"][0]["hf_id"] == exp.MODEL_ID
    rows = []
    for index in range(24):
        member = {**family, "family_id": f"family-{index}"}
        rows.extend(exp.capture_family(member, lambda _: reply(), tmp_path / "calls", 0.0))
    runtime = {
        "model_load_attempted": 1,
        "model_load_completed": 1,
        "generation_attempted": 72,
        "offload_layers": {"actual_offload": True},
    }
    done = exp.build_artifact(rows, [], {}, runtime, [], 3.0, "20260927")
    assert done["verdict_class"] == "null"
    assert done["sample_size_budget"]["effective_independent_n"] == 24
    assert done["model_invocation_counts"]["calls"] == 72
    assert done["paired_family_results"]["by_arm"]["source_withheld"]["brier"] is None


def test_input_validation_and_span_receipts(family, tmp_path):
    """SCENARIO-REPORT-7759-VIEWS rejects unknown arms and hashes checkpoints."""
    with pytest.raises(ValueError, match="unplanned_arm"):
        exp.make_request(family, "wrong")
    with pytest.raises(ValueError, match="unplanned_arm"):
        exp.indexed_windows(family["source"], "wrong")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    assert exp.phase_span("test", 0.0, 0.0, 1, checkpoint)["checkpoint_sha256"]
    assert exp.phase_span("test", 0.0, 0.0, 1, tmp_path / "missing")["checkpoint_sha256"] is None
    assert exp.gate("hash", "self", checkpoint, "present", True, True)["artifact_sha256"]


def test_preflight_keeps_producer_and_panel_custody_separate(family, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7759-REPLAY distinguishes result and pre-gate panel."""
    panel_path = tmp_path / exp.prior.RAW / "frozen_panel.json"
    result_path = tmp_path / exp.prior.RESULT
    panel_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    panel = [{**family, "family_id": f"f{i}"} for i in range(24)]
    panel_path.write_text(json.dumps(panel))
    prior_record = {
        "milestone": "2026.09.674",
        "qwen_localization_complete_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "paired_family_results": {"paired_families": 24},
        "source_artifact_hashes": {
            "pre_gate_receipts": {
                str(exp.prior.RAW / "frozen_panel.json"): exp.custody.sha256_file(panel_path)
            }
        },
        "current_model_receipts": {"model_sha256": "model-hash"},
    }
    result_path.write_text(json.dumps(prior_record))
    context = {"model_sha256": "model-hash", "selected": {"uuid": "fake"}}
    monkeypatch.setattr(exp.prior, "prepare_panel", lambda *_: (panel, [], {}, context))
    rebuilt, checks, hashes, observed = exp.preflight(tmp_path, 0.0)
    assert len(rebuilt) == 24 and all(check["passed"] for check in checks)
    assert hashes["valid_producers"][str(exp.prior.RESULT)]
    assert observed["model_sha256"] == "model-hash"
    monkeypatch.setattr(exp.prior, "prepare_panel", lambda *_: (panel[:1], [], {}, context))
    rebuilt, checks, _, _ = exp.preflight(tmp_path, 0.0)
    assert not rebuilt and checks[-2]["check"] == "same_exp7745_panel"
    monkeypatch.setattr(exp.prior, "prepare_panel", lambda *_: (panel, [], {}, context))
    prior_record["qwen_localization_complete_score"] = 0
    result_path.write_text(json.dumps(prior_record))
    rebuilt, checks, _, _ = exp.preflight(tmp_path, 0.0)
    assert not rebuilt and any(
        check["check"] == "exp7745_qualified" and not check["passed"] for check in checks
    )
    result_path.unlink()
    rebuilt, checks, hashes, _ = exp.preflight(tmp_path, 0.0)
    assert not rebuilt and hashes["missing_custody"] == [str(result_path)]


def test_owned_cuda_capture_lifecycle_with_fake_server(family, tmp_path, monkeypatch):
    """REQ-REPORT-7759 covers lease ownership and GPU receipts without a model."""
    import hashlib
    import os

    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot import experiment_7604_v664_evidence_pilot as v664
    from carnot.agentic import arc_executable_world_model as world
    from carnot import experiment_7581_v662_arc_bounded_canary as canary
    from carnot import gpu_lease_phase_journal as journal

    class FakeLease:
        def __init__(self):
            self.document = {"phase": "preflight"}

        def owner_receipt(self):
            return {"owner": "task"}

        def transition(self, phase, **_):
            self.document["phase"] = phase

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
    monkeypatch.setattr(v664, "_post_json", lambda *_: json.dumps(reply()).encode())
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
    rows, runtime = exp.owned_capture(tmp_path, panel, context, 0.0)
    assert len(rows) == 72 and runtime["generation_attempted"] == 72
    assert runtime["lease_release"]["phase"] == "terminal_complete"
    assert FakeProposer.last.stopped and runtime["offload_layers"]["actual_offload"]
    assert len(runtime["active_window_memory_mb"]) == 24
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "previous")
    monkeypatch.setattr(ownership, "recheck_before_launch", lambda *_: {"passed": False})
    with pytest.raises(RuntimeError, match="foreign_or_capacity_recheck_failed"):
        exp.owned_capture(tmp_path, panel[:1], context, 0.0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "previous"
    monkeypatch.setattr(ownership, "recheck_before_launch", lambda *_: {"passed": True})
    monkeypatch.setattr(FakeProposer, "_ensure_server", lambda *_: False)
    with pytest.raises(RuntimeError, match="owned_qwen_load_failed"):
        exp.owned_capture(tmp_path, panel[:1], context, 0.0)
    monkeypatch.setattr(FakeProposer, "_ensure_server", lambda *_: True)
    monkeypatch.setattr(v664, "_offload_receipt", lambda *_: {"actual_offload": False})
    with pytest.raises(RuntimeError, match="qwen_gpu_offload_not_authenticated"):
        exp.owned_capture(tmp_path, panel[:1], context, 0.0)
    monkeypatch.setattr(v664, "_offload_receipt", lambda *_: {"actual_offload": True})
    with pytest.raises(RuntimeError, match="chat_template_changed_after_load"):
        exp.owned_capture(tmp_path, panel[:1], {**context, "chat_template_sha256": "wrong"}, 0.0)
