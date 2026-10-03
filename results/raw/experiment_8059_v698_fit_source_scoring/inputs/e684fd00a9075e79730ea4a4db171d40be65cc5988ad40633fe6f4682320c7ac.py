"""REQ-REPORT-8033: private controls cannot supply live scientific evidence."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from carnot.inference import scoring_isolation_8033 as s
from carnot.inference import likelihood_isolation_runtime_8033 as r
from carnot import experiment_8033_v696_scoring_isolation as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def panel():
    return [
        dict(
            family_id=f"fit-{i}",
            role="fit",
            source_bytes="61",
            answer_bytes="62",
            source_normalized_hash=f"source-{i}",
            target_tokens=[1, 2],
            response_token_offsets=[[0, 1], [1, 2]],
            views=dict(full=dict(tokens=[0, 0, 1, 2], response_start=2)),
        )
        for i in range(8)
    ]


class Controller:
    def score(self, view, condition):
        result = s.likelihood([[0.0, math.log(2.0), math.log(3.0)]] * 4, view["tokens"], 2)
        return dict(
            result,
            context_identity="private",
            kv_lifecycle=[],
            actual_shapes=[dict(n_tokens=4, n_past=0)],
            context_closed=True,
        )


def test_schedule_capture_and_reduction(tmp_path):
    """SCENARIO-REPORT-8033-SCORE: repetitions preserve order and independent n."""
    p = panel()
    rows = s.capture(p, Controller(), tmp_path, deadline=float("inf"))
    assert len(rows) == 132
    assert sum(x["purpose"] == "target" for x in rows) == 128
    reduced = s.reduce(p, rows)
    assert reduced["selected_condition"] == "fresh_full"
    assert reduced["scored_tokens"] == 264
    assert all(x["passed"] for x in reduced["condition_rows"])
    assert {x["preceded_by"] for x in rows if x["purpose"] == "target"} == {"self", "different"}
    bad = copy.deepcopy(rows)
    bad[0]["token_rows"][0]["log_probability"] -= 0.1
    with pytest.raises(ValueError, match="aggregate"):
        s.reduce(p, bad)
    bad = copy.deepcopy(rows)
    bad[1]["token_rows"][0]["token_id"] = 0
    with pytest.raises(ValueError, match="alignment"):
        s.reduce(p, bad)
    bad_view = copy.deepcopy(rows)
    bad_view[0]["token_view_hash"] = "stale"
    with pytest.raises(ValueError, match="alignment_view"):
        s.reduce(p, bad_view)
    with pytest.raises(ValueError, match="roster"):
        s.reduce(p, [])
    bad = copy.deepcopy(rows)
    for row in bad:
        if row["repeat"] == 1:
            row["token_rows"][0]["log_probability"] -= 0.01
            row["token_rows"][0]["probability"] = math.exp(row["token_rows"][0]["log_probability"])
            row["numerator"] += 0.01
            row["mean_nll"] = row["numerator"] / 2
    assert s.reduce(p, bad)["selected_condition"] is None
    assert s.oracle()["passed"]


def test_failure_nonstarts_and_owned_oracle(tmp_path):
    """SCENARIO-REPORT-8033-LIFECYCLE: failures retain all frozen slots."""

    class Broken(Controller):
        def score(self, view, condition):
            raise TimeoutError("private_forward_timeout")

    failed = s.capture(panel(), Broken(), tmp_path / "fail", deadline=float("inf"))
    assert failed[0]["status"] == "failed"
    assert all(x["status"] == "censored" for x in failed[1:])
    assert not s.reduce(panel(), failed)["complete"]
    assert all(
        x["status"] == "censored"
        for x in s.capture(panel(), Controller(), tmp_path / "cap", deadline=0)
    )
    with patch.object(s.base, "target_likelihood", return_value={"target_logprobs": [0.0, 0.0]}):
        with pytest.raises(ValueError, match="oracle"):
            s.oracle()
    with pytest.raises(ValueError):
        s.likelihood([[0, 1, 2]], [0, 1, 2], 2)


def test_length_roster_is_label_free():
    """SCENARIO-REPORT-8033-SCORE: rank selection never reads likelihood or labels."""
    rows = panel() + panel()
    for i, row in enumerate(rows):
        row["family_id"] = str(i)
        row["source_normalized_hash"] = str(i)
        row["views"]["full"]["tokens"] = [0] * (i + 4)
    selected = s.select(rows)
    assert len(selected) == 8
    assert selected[0]["family_id"] == "0" and selected[-1]["family_id"] == "15"
    with pytest.raises(ValueError, match="eight"):
        s.select(rows[:7])


class Context:
    def __init__(self, **kwargs):
        self.ctx = 123
        self.memory = 1
        self.position = -1
        self.closed = False

    def kv_cache_seq_rm(self, *args):
        self.position = args[1] - 1
        return True

    def decode(self, batch):
        self.position = batch.n_past + batch.n_tokens - 1

    def kv_cache_clear(self):
        self.position = -1

    def close(self):
        self.closed = True


class NativeModel:
    def __init__(self):
        self._ctx = Context()
        self._model = object()
        self.context_params = SimpleNamespace(n_batch=256, n_ubatch=256, flash_attn_type=0)
        self.n_batch = 256
        self.n_tokens = 0
        self._batch = SimpleNamespace(n_past=0, n_tokens=0)
        self._batch.batch = self._batch
        self.scores = np.zeros((4, 3))

    def reset(self):
        self.n_tokens = 0

    def eval(self, tokens):
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for start in range(0, len(tokens), self.n_batch):
            batch = tokens[start : start + self.n_batch]
            self._batch.n_past = self.n_tokens
            self._batch.n_tokens = len(batch)
            self._ctx.decode(self._batch)
            self.n_tokens += len(batch)


def test_native_context_shapes_and_stale_buffer(tmp_path):
    """SCENARIO-REPORT-8033-LIFECYCLE: fresh context closes; scalars survive buffer reuse."""
    from llama_cpp import llama_cpp as native
    from llama_cpp import _internals

    model = NativeModel()
    runtime = SimpleNamespace(model=model)
    with (
        patch.object(_internals, "LlamaContext", Context),
        patch.object(native, "llama_memory_seq_pos_min", return_value=-1),
        patch.object(
            native, "llama_memory_seq_pos_max", side_effect=lambda mem, seq: model._ctx.position
        ),
    ):
        controller = s.NativeController(runtime, float("inf"))
        for condition in s.CONDITIONS:
            result = controller.score(panel()[0]["views"]["full"], condition)
            assert result["conditional_logit_positions"] == [1, 2]
            assert result["kv_lifecycle"]
        model.scores[:] = 100
        assert result["target_logprobs"] == pytest.approx([-math.log(3)] * 2)
        assert model._ctx is controller.original
        model.eval = lambda tokens: None
        with pytest.raises(ValueError, match="alignment"):
            controller.score(panel()[0]["views"]["full"], "fresh_full")
        controller.deadline = 0
        with pytest.raises(TimeoutError):
            controller.score(panel()[0]["views"]["full"], "fresh_full")


def test_runtime_adapter_checks_and_cleanup(tmp_path):
    """SCENARIO-REPORT-8033-LIFECYCLE: direct adapter preserves native failed loads."""
    plan = dict(panel=panel(), gguf_sha256="sha256:expected", scoring_config_hash="hash")
    pp, out = tmp_path / "plan.json", tmp_path / "capture.json"
    atomic_json(pp, plan)
    with patch.object(r.runtime, "cached_current_model", return_value=None):
        r.worker(pp, out)
    assert not json.loads(out.read_text())["checks"][0]["passed"]
    with (
        patch.object(r.runtime, "cached_current_model", return_value=dict(model_path=str(pp))),
        patch.object(r, "sha256_file", return_value="bad"),
    ):
        r.worker(pp, out)
    assert json.loads(out.read_text())["checks"][0]["observed"] == "bad"

    def loader(path, output):

        with patch.object(r, "NativeController", return_value=Controller()):
            model = SimpleNamespace(model=SimpleNamespace(model_params=SimpleNamespace(main_gpu=0)))
            with patch.object(
                r.runtime, "run_commands", return_value=[dict(passed=True, output_tail="24000")]
            ):
                r.base.progress("before_benchmark", 0)
                qualified = r.base.qualify(model, tmp_path / "forwards")
                r.base.progress("after_benchmark", 0)
            with patch.object(
                r.runtime, "run_commands", return_value=[dict(passed=True, output_tail="100")]
            ):
                with pytest.raises(ValueError, match="reserve"):
                    r.base.qualify(model, tmp_path / "forwards")
        assert len(qualified["rows"]) == 132
        r.runtime.bounded(lambda: None, 1)
        with patch.object(r.time, "monotonic", return_value=float("inf")):
            with pytest.raises(TimeoutError):
                r.runtime.bounded(lambda: None, 1)
        with patch.object(r, "verify_panel", return_value=plan["panel"]):
            assert r.base.freeze_panel({}, None) == plan["panel"]
        with patch.object(r, "original_acquire", return_value="lease"):
            assert r.runtime.GpuLease.acquire(task_id="private") == "lease"
        atomic_json(
            output, dict(qualification=qualified, checks=[], cleanup=dict(model_closed=True))
        )

    with (
        patch.object(r.runtime, "cached_current_model", return_value=dict(model_path=str(pp))),
        patch.object(r, "sha256_file", return_value="sha256:expected"),
        patch.object(r.runtime, "worker", side_effect=loader),
    ):
        r.worker(pp, out)
    assert json.loads(out.read_text())["qualification"]["selected_condition"] == "fresh_full"
    with patch.object(r.base, "prepare", side_effect=lambda row, tok: dict(row, views={})):
        with pytest.raises(ValueError, match="panel"):
            r.verify_panel(panel(), None)
    with patch.object(r.base, "prepare", side_effect=lambda row, tok: panel()[0]):
        assert r.verify_panel([panel()[0]], None) == [panel()[0]]


def test_private_cli_routes(tmp_path):
    """SCENARIO-REPORT-8033-SEAL: real CLI rejects dates and missing replay bytes."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if os.environ.get("CARNOT_8033_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["CARNOT_8033_COVERAGE_CONFIG"]]
    cli = str(e.ROOT / e.CLI)
    run = subprocess.run([*prefix, cli, "--date", "bad"], env=env, capture_output=True, timeout=30)
    assert run.returncode == 2
    run = subprocess.run(
        [*prefix, cli, "--cold-replay", str(tmp_path / "missing")],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 1
    pp, out = tmp_path / "p.json", tmp_path / "out.json"
    atomic_json(pp, dict(panel=[], gguf_sha256="wrong"))
    run = subprocess.run(
        [*prefix, cli, "--runtime-child", str(pp), "--output", str(out)],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode == 0 and out.is_file()


def private_plan(root):
    results = root / "results"
    atomic_json(
        results / "experiment_8022_v695_likelihood_protocol.json",
        dict(
            experiment_id=8022,
            likelihood_protocol_ready_score=1,
            token_scoring_ready_score=1,
            flagged_adversarial=False,
            gguf_sha256="private",
            public_panel_manifest=dict(rows=panel()),
        ),
    )
    atomic_json(
        results / "experiment_8023_v695_likelihood_calibration.json",
        dict(
            experiment_id=8023,
            forward_pass_counts=384,
            honest_verdict="complete_disqualified",
            duplicate_drift_rows=[dict(drift=0.0063769592214454884)],
        ),
    )


def test_main_publication_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8033-SEAL: private full CLI flow binds checkpoints and claims."""
    private_plan(tmp_path)
    plan = e.authenticate(tmp_path)
    assert len(plan["panel"]) == 8
    assert e.authenticate(tmp_path / "absent")["panel"] == []
    output = tmp_path / "results" / (e.NAME + ".json")
    totals = dict(percent_covered=100, num_statements=1, covered_lines=1, missing_lines=0)

    def commands(scratch):
        atomic_json(scratch / "coverage.json", dict(totals=totals))
        return [e.CommandSpec("private_owned", (sys.executable, "-c", "pass"), "owned")]

    def run(root, specs, **kwargs):
        if not specs:
            return [dict(name="private_owned", passed=True, scope="owned")]
        if specs[0].name == "owned_scoring_child":
            path = Path(specs[0].argv[-1])
            rows = s.capture(panel(), Controller(), path.parent / "forwards", deadline=float("inf"))
            atomic_json(
                path,
                dict(
                    qualification=dict(rows=rows),
                    model_invocation_counts={
                        **e.ZERO_INVOCATION_COUNTS,
                        "model_loads_completed": 2,
                        "model_loads_attempted": 2,
                    },
                    cleanup=dict(model_closed=True, lease_released=True),
                    offload_evidence=dict(supported=True),
                    duration_s=3,
                ),
            )
        return [dict(passed=True, name=specs[0].name, scope="owned", exit_code=0)]

    monkeypatch.setattr(e, "commands", commands)
    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(e, "terminal_readers", lambda _: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=True))
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["scoring_isolation_ready_score"] == 1
    e.replay(value)
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--output", str(output)]) == 1
    bad = copy.deepcopy(value)
    bad["scored_tokens"] += 1
    with pytest.raises(ValueError, match="cold_reduction"):
        e.replay(bad)
    raw = output.parent / "raw" / output.stem
    first = next((raw / "runtime/forwards").glob("*.json"))
    original = first.read_bytes()
    modified = json.loads(original)
    modified["status"] = "censored"
    atomic_json(first, modified)
    value["checkpoint_references"] = [
        e.reference(Path(x["path"])) for x in value["checkpoint_references"]
    ]
    with pytest.raises(ValueError, match="checkpoint"):
        e.replay(value)
    first.write_bytes(original)
    atomic_json(output, bad)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.relocate(["/tmp/old/x", 2], Path("/tmp/old"), Path("/tmp/new")) == ["/tmp/new/x", 2]


def test_main_failures_and_validation_manifest(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8033-SEAL: failures in cold or published readers exit one."""
    (tmp_path / "config").mkdir()
    specs = e.commands(tmp_path / "config")
    assert any(x.name == "repository_health" for x in specs)
    assert any("--strict" in x.argv for x in specs)
    monkeypatch.setattr(e, "commands", lambda scratch: [])
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "cold/results" / (e.NAME + ".json")),
            ]
        )
        == 1
    )
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=True)])
    monkeypatch.setattr(e, "publish_primary", lambda *a: {})
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=False))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "pub/results" / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_private_missing_contract_cli(tmp_path):
    """SCENARIO-REPORT-8033-SEAL: actual blocked CLI publishes then cold replays."""
    prefix = [sys.executable]
    if os.environ.get("CARNOT_8033_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["CARNOT_8033_COVERAGE_CONFIG"]]
    out = tmp_path / "results" / (e.NAME + ".json")
    argv = [*prefix, str(e.ROOT / e.CLI), "--root", str(tmp_path / "absent"), "--output", str(out)]
    run = subprocess.run(argv, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout.decode() + run.stderr.decode()
    assert json.loads(out.read_text())["verdict_class"] == "blocked"


def test_native_memory_clear_and_loader_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8033-LIFECYCLE: actual loader failure releases a private lease."""
    import llama_cpp
    from llama_cpp import llama_cpp as native, _internals
    from carnot.gpu_lease_phase_journal import GpuLease

    model = NativeModel()
    monkeypatch.setattr(native, "llama_memory_clear", lambda *args: None)
    model.reset = lambda: (setattr(model, "n_tokens", 0), native.llama_memory_clear(1, True))
    monkeypatch.setattr(_internals, "LlamaContext", Context)
    monkeypatch.setattr(native, "llama_memory_seq_pos_min", lambda *args: -1)
    monkeypatch.setattr(native, "llama_memory_seq_pos_max", lambda *args: -1)
    result = s.NativeController(SimpleNamespace(model=model), float("inf")).score(
        panel()[0]["views"]["full"], "reset_full"
    )
    assert any(x["operation"] == "memory_clear" for x in result["kv_lifecycle"])
    real_acquire = GpuLease.acquire
    monkeypatch.setattr(
        r,
        "original_acquire",
        lambda **kwargs: real_acquire(**dict(kwargs, runtime_dir=tmp_path / "leases")),
    )
    pp, out = tmp_path / "plan.json", tmp_path / "capture.json"
    atomic_json(pp, dict(panel=panel(), gguf_sha256="expected", scoring_config_hash="private"))
    monkeypatch.setattr(
        r.runtime, "cached_current_model", lambda: dict(hf_id=r.runtime.MODEL, model_path=str(pp))
    )
    monkeypatch.setattr(r, "sha256_file", lambda _: "expected")
    monkeypatch.setattr(native, "llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(
        r.runtime,
        "run_commands",
        lambda *a, **k: [dict(output_tail="0, private-GPU, 4, 24000", passed=True)],
    )

    class FailedLlama:
        reset = llama_cpp.Llama.reset
        eval = llama_cpp.Llama.eval

        def __new__(cls, **kwargs):
            raise ValueError("private_failed_load")

    monkeypatch.setattr(llama_cpp, "Llama", FailedLlama)
    # The adapter inspects methods before loading; keep the real source inspection here.
    with (
        patch.object(r.inspect, "getsource", return_value="private inspected binding"),
        patch.object(r.inspect, "getfile", return_value=str(pp)),
    ):
        r.worker(pp, out)
    captured = json.loads(out.read_text())
    assert captured["model_invocation_counts"]["model_loads_failed"] == 1
    assert captured["gpu_lease_receipt"]["release"]["released"]


def test_child_exit_and_retained_failure_slots(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8033-LIFECYCLE: child timeouts finish blocked with nonstarts."""
    private_plan(tmp_path)
    plan = e.authenticate(tmp_path)
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    monkeypatch.setattr(e, "commands", lambda scratch: [])
    log = tmp_path / "child.log"
    log.write_text("private child timed out\n")

    def run(root, specs, **kwargs):
        if specs[0].name == "owned_scoring_child":
            return [dict(passed=False, log_path=str(log), exit_code=124)]
        return [dict(passed=True)]

    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(e, "terminal_readers", lambda _: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=True))
    output = tmp_path / "timeout/results" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "private"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["censored_count"] == 128
