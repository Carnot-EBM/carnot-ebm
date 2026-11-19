"""REQ-REPORT-9098-V776 / SCENARIO-REPORT-9098-V776: actual capability or exact refusal."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import embedding_runtime_9098 as d
from carnot.inference import embedding_runtime_9098 as native


class FixtureRuntime:
    """CPU controls test the protocol; their vectors never become model evidence."""

    width = 3
    pad_id = 0

    def tokenize(self, text: str) -> list[int]:
        return [1] * (text.count(" x") if text.startswith(" x") else len(text))

    def hidden(self, tokens: list[list[int]]) -> list[np.ndarray]:
        return [np.tile([1.0, 2.0, 3.0], (len(t), 1)) for t in tokens]

    def likelihood(self, prompt: str, continuation: str) -> dict[str, Any]:
        return dict(
            tokens=[1],
            token_logprobs=[-1.0],
            log_probability=-1.0,
            raw_logits_finite=True,
            raw_logits_width=8,
            raw_logits_unmasked=True,
        )


def test_frozen_fixtures_and_pooling() -> None:
    """REQ-REPORT-9098-V776: caps and masks precede all observed vectors."""
    p = d.protocol()
    assert len(p["fixtures"]) == 12
    assert p["pooling_contract"] == d.methods.protocol()["embedding"]
    runtime = FixtureRuntime()
    runtime.last_hidden_references = [dict(path="private-cpu-control", sha256=None)]
    rows = d.exercise(runtime)
    assert len(rows) == 12 and all(r["embedding_passed"] for r in rows)
    assert [r["retained_token_count"] for r in rows[-4:]] == [256, 768, 256, 768]
    assert [r["truncated"] for r in rows[-4:]] == [False, False, True, True]
    assert np.allclose(d.pool([[3, 4], [99, 99]], [True, False], 2), [0.6, 0.8])
    assert np.array_equal(d.pool([], [], 2, empty=True), [0, 0])
    for vectors, mask, width, reason in [
        ([[1, 2]], [True], 3, "hidden_width"),
        ([[float("nan"), 1]], [True], 2, "nonfinite"),
        ([[0, 0]], [True], 2, "zero_vector"),
        ([], [], 2, "empty"),
    ]:
        with pytest.raises(ValueError, match=reason):
            d.pool(vectors, mask, width)


def test_independent_readiness_and_cold_parity() -> None:
    """SCENARIO-REPORT-9098-V776: a logits result cannot authorize embeddings."""
    rows = d.exercise(FixtureRuntime())
    reduced = d.reduce(rows, deepcopy(rows), runtime_ok=True)
    assert reduced["embeddings_ready_score"] == reduced["logits_ready_score"] == 1
    bad = deepcopy(rows)
    bad[2]["vector"][0] += 0.01
    assert d.reduce(rows, bad, runtime_ok=True)["embeddings_ready_score"] == 0
    bad = deepcopy(rows)
    bad[2]["logits_passed"] = False
    assert d.reduce(bad, bad, runtime_ok=True)["embeddings_ready_score"] == 1
    assert d.reduce(bad, bad, runtime_ok=True)["logits_ready_score"] == 0
    assert d.reduce(rows, rows, runtime_ok=False)["embeddings_ready_score"] == 0


def test_teacher_forcing_full_continuations() -> None:
    """REQ-REPORT-9098-V776: every literal token contributes with fresh context."""

    class Model:
        n_tokens = 0
        scores = np.array([[0.0, 1.0, 2.0, 3.0]] * 8)
        resets = 0
        metadata = {"tokenizer.ggml.padding_token_id": "0"}

        def n_embd(self) -> int:
            return 3

        def n_vocab(self) -> int:
            return 4

        @property
        def _ctx(self) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(kv_cache_clear=lambda: None)

        def tokenize(self, text: bytes, **kwargs: Any) -> list[int]:
            assert kwargs == dict(add_bos=False, special=False)
            return [1, 2] if text == b"Yes" else [3]

        def reset(self) -> None:
            self.n_tokens = 0
            self.resets += 1

        def eval(self, tokens: list[int]) -> None:
            self.n_tokens += len(tokens)

    model = Model()
    adapter = native.FrozenRuntime(model)
    yes, no = adapter.likelihood("prompt", "Yes"), adapter.likelihood("prompt", "No")
    assert model.resets == 2
    assert len(yes["token_logprobs"]) == 2 and len(no["token_logprobs"]) == 1
    assert yes["log_probability"] == sum(yes["token_logprobs"])
    assert yes["raw_logits_unmasked"]


def test_cancellation_and_bad_native_outputs() -> None:
    """SCENARIO-REPORT-9098-V776: deadlines and invalid hidden states fail closed."""
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        d.exercise(FixtureRuntime(), deadline=0)

    class Wrong(FixtureRuntime):
        def hidden(self, tokens: list[list[int]]) -> list[np.ndarray]:
            return [np.ones((len(t), 2)) for t in tokens]

    rows = d.exercise(Wrong())
    assert not any(r["embedding_passed"] for r in rows if not r["empty"])
    assert all(r["logits_passed"] for r in rows)


@pytest.fixture
def native_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """REQ-REPORT-9098-V776: test native boundaries without claiming CPU vectors are Qwen."""
    import llama_cpp
    from types import SimpleNamespace

    lib = tmp_path / "binding"
    lib.mkdir()
    (lib / "__init__.py").write_text("fixture")
    (lib / "libllama.so").write_bytes(b"fixture")
    gguf = tmp_path / "Q4_K_M.gguf"
    gguf.write_bytes(b"fixture")
    log = tmp_path / "native.stderr"
    log.write_text("offloaded 65/65 layers\n")
    gpu = dict(index=0, uuid="fixture-gpu", used_mb=0, free_mb=24000)
    resources = dict(model=dict(model_path=str(gguf), sha256=d.ref(gguf)["sha256"]), gpu=gpu)
    admission = dict(
        admitted=True, content_metadata=dict(quantization="Q4_K_M", tokenizer_metadata={})
    )
    monkeypatch.setattr(native, "preflight", lambda _: deepcopy(resources))
    monkeypatch.setattr(native, "build_gguf_admission_record", lambda *a, **k: deepcopy(admission))
    states = iter([gpu, dict(gpu, used_mb=18000), gpu])
    monkeypatch.setattr(native, "gpu_state", lambda: next(states))
    monkeypatch.setattr(llama_cpp, "__file__", str(lib / "__init__.py"))
    monkeypatch.setattr(llama_cpp, "llama_print_system_info", lambda: b"fixture CUDA")
    monkeypatch.setattr(native.os, "readlink", lambda _: str(log))

    class Lease:
        document = dict(phase="preflight")

        def owner_receipt(self) -> dict:
            return dict(fixture=True)

        def transition(self, phase: str, **kwargs: Any) -> None:
            self.document["phase"] = phase

        def release(self) -> dict:
            return dict(released=True)

        def close(self) -> None:
            pass

    monkeypatch.setattr(native.GpuLease, "acquire", lambda **_: Lease())
    model = SimpleNamespace(metadata={"qwen35.embedding_length": "3"}, close=lambda: None)
    monkeypatch.setattr(llama_cpp, "Llama", lambda **_: model)
    adapter = FixtureRuntime()
    adapter.embedding_calls = 30
    adapter.close = lambda: None
    monkeypatch.setattr(native, "FrozenRuntime", lambda *args: adapter)
    return SimpleNamespace(
        admission=admission,
        resources=resources,
        log=log,
        model=model,
        adapter=adapter,
        llama=llama_cpp,
        gpu=gpu,
    )


@pytest.mark.parametrize(
    "failure", [None, "cache", "admission", "library", "memory", "load", "offload", "width"]
)
def test_native_worker_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_boundary: Any, failure: str | None
) -> None:
    """SCENARIO-REPORT-9098-V776: exact cache, lease, offload and width failures are terminal."""
    b = native_boundary
    if failure == "cache":
        monkeypatch.setattr(
            native, "preflight", lambda _: (_ for _ in ()).throw(RuntimeError("cache_missing"))
        )
    elif failure == "admission":
        b.admission["admitted"] = False
    elif failure == "library":
        (Path(b.llama.__file__).parent / "libllama.so").unlink()
    elif failure == "memory":
        b.gpu["free_mb"] = 0
    elif failure == "load":
        monkeypatch.setattr(
            b.llama, "Llama", lambda **_: (_ for _ in ()).throw(ValueError("load_failed"))
        )
    elif failure == "offload":
        b.log.write_text("CPU only")
    elif failure == "width":
        b.model.metadata["qwen35.embedding_length"] = "5120"
    value = native.worker(tmp_path / "raw")
    assert value["generated_tokens"] == 0 and len(value["rows"]) == 12
    assert (value["blocker"] is None) == (failure is None)
    if failure:
        assert value["gate_check_summary"][-1]["passed"] is False
    else:
        assert value["runtime_ok"] and value["lease_release"]["released"]


def test_bad_fixture_likelihood_and_parity() -> None:
    """REQ-REPORT-9098-V776: exact count, nonfinite likelihood and parity are separate failures."""

    class Bad(FixtureRuntime):
        calls = 0

        def hidden(self, tokens: list[list[int]]) -> list[np.ndarray]:
            self.calls += 1
            return [np.tile([float(self.calls), 2.0, 3.0], (len(t), 1)) for t in tokens]

        def likelihood(self, prompt: str, continuation: str) -> dict[str, Any]:
            return dict(
                tokens=[1],
                token_logprobs=[float("nan")],
                log_probability=float("nan"),
                raw_logits_finite=False,
            )

        def tokenize(self, text: str) -> list[int]:
            return [1] * len(text)

    rows = d.exercise(Bad())
    assert all(not r["logits_passed"] for r in rows)
    assert rows[2]["embedding_error"] == "batch_parity"
    assert rows[2]["vector"]
    assert rows[-1]["embedding_error"] == "exact_token_count"


def test_native_hidden_buffers_and_raw_logit_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-9098-V776: native buffers must be hidden-width and full-vocabulary arrays."""
    import ctypes
    from types import SimpleNamespace
    from llama_cpp import _internals, llama_cpp as cpp

    class Params(ctypes.Structure):
        _fields_ = [
            ("embeddings", ctypes.c_bool),
            ("n_seq_max", ctypes.c_int),
            ("kv_unified", ctypes.c_bool),
            ("pooling_type", ctypes.c_int),
        ]

    class Context:
        missing = False

        def kv_cache_clear(self) -> None:
            pass

        def decode(self, batch: Any) -> None:
            pass

        def get_embeddings_ith(self, index: int) -> list[float]:
            return [] if self.missing else [1.0, 2.0, 3.0]

        def close(self) -> None:
            pass

    context = Context()
    model = SimpleNamespace(
        n_embd=lambda: 3,
        n_vocab=lambda: 4,
        context_params=Params(),
        metadata={"tokenizer.ggml.padding_token_id": "0"},
        _model=SimpleNamespace(model=None),
        _ctx=context,
        scores=np.array([[0.0, 1.0, 2.0, 3.0]] * 8),
        n_tokens=1,
        tokenize=lambda text, **kwargs: [] if text == b"" else [1, 2],
        eval=lambda tokens: None,
        reset=lambda: None,
        _batch=SimpleNamespace(reset=lambda: None, add_sequence=lambda *args: None),
    )
    monkeypatch.setattr(_internals, "LlamaContext", lambda **kwargs: context)
    monkeypatch.setattr(cpp, "llama_model_n_embd_out", lambda _: 3)
    adapter = native.FrozenRuntime(model, tmp_path)
    assert adapter.hidden([[1, 2], [3]])[0].shape == (2, 3)
    assert adapter.last_hidden_references[0]["sha256"]
    assert adapter.hidden([[1]])[0].shape == (1, 3)
    likelihood = adapter.likelihood("p", "Yes")
    assert np.load(likelihood["raw_logits_reference"]["path"]).shape == (2, 4)
    context.missing = True
    with pytest.raises(ValueError, match="hidden_states_unavailable"):
        adapter.hidden([[1]])
    monkeypatch.setattr(cpp, "llama_model_n_embd_out", lambda _: 2)
    with pytest.raises(ValueError, match="classifier_scores"):
        adapter.hidden([[1]])
    with pytest.raises(ValueError, match="empty_likelihood"):
        adapter.likelihood("", "Yes")
    model.scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite_or_wrong_logit_width"):
        adapter.likelihood("p", "Yes")
    adapter.close()
    native.FrozenRuntime(model).close()


def private_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, positive: bool = False
) -> dict:
    """REQ-REPORT-9098-V776: CPU protocol controls remain private and contain no model calls."""
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    def fake(raw: Path, deadline: float) -> dict:
        logs = raw / "logs"
        logs.mkdir(parents=True)
        out, err = logs / "fixture.stdout", logs / "fixture.stderr"
        out.write_text("private CPU protocol control")
        err.write_text("")
        value = dict(
            rows=d.exercise(FixtureRuntime()) if positive else d.censored("cache_missing"),
            blocker=None if positive else "cached_runtime",
            runtime_ok=positive,
            model_loads=0,
            embedding_calls=0,
            runtime_reference={},
            tokenizer_metadata={},
            source_references=[],
            gate_check_summary=[],
            current_work_receipt={},
        )
        atomic_json(raw / "runtime.json", value)
        value["process_receipt"] = dict(
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=d.ref(out)["sha256"],
            stderr_sha256=d.ref(err)["sha256"],
        )
        return value

    monkeypatch.setattr(runner, "run_worker", fake)
    return runner.measure(tmp_path / "work", "20261115", {})


def test_build_replay_and_independent_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9098-V776: aggregate and log tampering cannot acquire readiness."""
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json
    import json

    reference = private_work(tmp_path, monkeypatch, positive=True)
    assert runner.build(reference, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(runner, "receipts_pass", lambda *args: True)
    value = runner.build(reference, {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert value["embeddings_ready_score"] == value["logits_ready_score"] == 1
    assert runner.replay(path)["passed"]
    changed = deepcopy(value)
    changed["embeddings_ready_score"] = 0
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)
    atomic_json(path, value)
    changed = deepcopy(value)
    changed["source_references"] = [dict(path=str(tmp_path / "missing"), sha256="wrong")]
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="operand_hash"):
        runner.replay(path)
    atomic_json(path, value)
    work = json.loads(Path(reference["path"]).read_bytes())
    output = Path(work["warm"]["process_receipt"]["stdout_path"])
    output.write_text("changed")
    with pytest.raises(ValueError, match="runtime_log_hash"):
        runner.replay(path)
    output.write_text("private CPU protocol control")
    original = output.parent.parent / "runtime.json"
    atomic_json(original, {})
    with pytest.raises(ValueError, match="runtime_reduction_drift"):
        runner.replay(path)
    with monkeypatch.context() as m:
        m.setattr(d, "protocol", lambda: {})
        with pytest.raises(ValueError, match="protocol_drift"):
            runner.replay(path)


def test_supervised_native_child_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9098-V776: cancellation retains a started load; missing cache adds no load."""
    import json
    import time
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    start = time.monotonic_ns()
    call = dict(
        passed=False,
        exit_code=-9,
        duration_s=0.01,
        stderr_path=str(tmp_path / "err"),
        started_monotonic_ns=start,
        ended_monotonic_ns=start + 10_000_000,
    )
    monkeypatch.setattr(runner, "child", lambda *a, **k: call)
    raw = tmp_path / "empty"
    value = runner.run_worker(raw, time.monotonic() + 1)
    assert value["model_loads"] == 0 and value["blocker"] == "native_worker"
    raw = tmp_path / "started"
    raw.mkdir()
    atomic_json(
        raw / "events.json",
        dict(
            events=[
                dict(
                    call_id="frozen-qwen",
                    operation="model_load",
                    scope="current",
                    transport="owned_runtime",
                    run_id=raw.name,
                    owner_pid=123,
                    state="attempted",
                    monotonic_ns=start,
                )
            ]
        ),
    )
    value = runner.run_worker(raw, time.monotonic() + 1)
    assert value["model_loads"] == 1
    assert value["current_work_receipt"]["invocation_counts"]["model_loads_cancelled"] == 1
    call["passed"] = True
    assert runner.run_worker(raw, time.monotonic() + 1)["rows"] == value["rows"]
    assert json.loads((raw / "runtime.json").read_bytes())["model_loads"] == 1


def test_cli_publication_and_refusal_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9098-V776: private CLI success, external block and date refusal preserve bytes."""
    import json
    import runpy
    import sys
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    reference = private_work(tmp_path, monkeypatch)
    assert runner.scope_manifest([])["owned_paths"] == d.OWNED
    assert not runner.receipts_pass({}, {})
    work = tmp_path / "reference.json"
    atomic_json(work, reference)
    scope, receipts = tmp_path / "scope.json", tmp_path / "receipts.json"
    atomic_json(scope, {})
    atomic_json(receipts, {})
    output = tmp_path / (d.NAME + ".json")
    args = [
        "--work",
        str(work),
        "--scope",
        str(scope),
        "--receipts",
        str(receipts),
        "--output",
        str(output),
    ]
    assert runner.main(args) == 1 and not output.exists()
    monkeypatch.setattr(runner, "receipts_pass", lambda *a: True)
    monkeypatch.setattr(runner, "validate", lambda p, logs: dict(passed=True))
    assert runner.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 12
    assert runner.main(["--cold-replay", str(output)]) == 0
    value["completed_count"] = 12
    atomic_json(output, value)
    assert runner.main(["--cold-replay", str(output)]) == 1
    assert runner.main(["--date", "wrong"]) == 1
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [d.CLI, "--date", "wrong"])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(d.ROOT / d.CLI), run_name="__main__")
        assert exit_info.value.code == 1
    monkeypatch.setattr(native, "worker", lambda raw: {})
    assert runner.main(["--worker", str(tmp_path / "worker")]) == 0
    missing = tmp_path / "absent"
    monkeypatch.setattr(runner, "measure", lambda *a: reference)
    assert (
        runner.main(
            [
                "--scope",
                str(missing),
                "--receipts",
                str(missing),
                "--work",
                str(missing),
                "--output",
                str(output),
            ]
        )
        == 0
    )


def test_unchanged_scanner_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9098-V776: the real scanner helper retains its original behavior."""
    from carnot.reporting import embedding_runtime_runner_9098 as runner

    monkeypatch.setattr(runner.prior, "validate", lambda *args: dict(passed=True))
    assert runner.validate(tmp_path / "candidate", tmp_path / "logs")["passed"]


def test_partial_capability_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9098-V776: embedding failure keeps independently available logits visible."""
    import json
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    reference = private_work(tmp_path, monkeypatch, positive=True)
    work_path = Path(reference["path"])
    work = json.loads(work_path.read_bytes())
    work["warm"]["rows"][2]["embedding_passed"] = False
    for row in work["warm"]["rows"]:
        row["embedding_passed"] = False
    work["prior_current_runs"] = [work["warm"]]
    atomic_json(work_path, work)
    monkeypatch.setattr(runner, "receipts_pass", lambda *args: True)
    value = runner.build(d.ref(work_path), {})
    assert value["honest_verdict"] == "complete_null_runtime_capability_limit"
    assert value["verdict_class"] == "null"
    assert all(
        r["artifact_field"].endswith("current_work_receipt") for r in value["current_work_receipts"]
    )
    assert value["logits_ready_score"] == 1 and value["embeddings_ready_score"] == 0
    assert value["duration_projection"]["shard_a_s"] is not None


def test_real_foreign_cli_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9098-V776: fresh CLI rejects rehashed tampering without PYTHONPATH."""
    import os
    import subprocess
    from carnot.reporting import embedding_runtime_runner_9098 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    reference = private_work(tmp_path, monkeypatch)
    value = runner.build(reference, {})
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / d.CLI), "--cold-replay", str(path)]
    valid = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert valid.returncode == 0 and '"replay_passed": true' in valid.stdout
    value["completed_count"] = 12
    value["reproducibility_checksum"] = d.canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(path, value)
    altered = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
    )
    assert altered.returncode == 1 and "reduction_drift" in altered.stdout


def test_cold_process_after_parity_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-9098-V776: failed warm parity does not erase the cold qualification control."""
    from carnot.reporting import embedding_runtime_runner_9098 as runner

    calls = []

    def run(raw: Path, deadline: float) -> dict:
        calls.append(raw)
        return dict(runtime_ok=True, rows=[dict(embedding_passed=False)])

    monkeypatch.setattr(runner, "run_worker", run)
    runner.measure(tmp_path, "20261115", {})
    assert len(calls) == 2
