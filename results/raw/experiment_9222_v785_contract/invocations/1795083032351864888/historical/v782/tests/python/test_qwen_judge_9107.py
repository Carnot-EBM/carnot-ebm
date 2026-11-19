"""REQ-REPORT-9107-V776 / SCENARIO-REPORT-9107-V776: predictions need raw evidence."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import qwen_judge_9107 as d
from carnot.reporting.current_work_receipt import atomic_json


def test_protocol_and_quoted_prompt(tmp_path: Path) -> None:
    """REQ-REPORT-9107-V776: budgets and allowlisted text precede outcomes."""
    p = d.protocol()
    assert p["temperatures"] == [0.5, 1, 2, 4]
    assert p["bootstrap_draws"] == 10000 and p["bootstrap_seed"] == 775092
    model = FixtureModel()
    view = dict(question="q" * 257, answer_text="a" * 769, ref_str="SECRET", qid="SECRET")
    text, counts = d.prompt(model, view)
    assert "SECRET" not in text
    assert json.dumps("q" * 256) in text and json.dumps("a" * 768) in text
    assert counts["full_question_tokens"] == 257
    assert counts["full_answer_tokens"] == 769
    assert counts["retained_question_tokens"] == 256
    assert counts["retained_answer_tokens"] == 768


class FixtureModel:
    """A CPU control exposes token accounting without claiming a Qwen invocation."""

    metadata = {
        "tokenizer.chat_template": "{{ messages[0].content }}<assistant>",
        "tokenizer.ggml.padding_token_id": "0",
    }

    def tokenize(self, text: bytes, **kwargs: Any) -> list[int]:
        return list(text)

    def detokenize(self, tokens: list[int], **kwargs: Any) -> bytes:
        return bytes(tokens)

    def token_eos(self) -> int:
        return 0

    def token_bos(self) -> int:
        return 0


def test_raw_score_and_missing_logits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9107-V776: both complete continuations require raw logits."""

    class Runtime:
        model = FixtureModel()

        def likelihood(self, prompt: str, word: str) -> dict[str, Any]:
            path = tmp_path / (word + ".npy")
            np.save(path, np.array([[0.0, 1.0], [1.0, 0.0]]))
            lp = [-float(np.logaddexp(0, 1)), -float(np.logaddexp(0, -1))]
            return dict(
                tokens=[0, 0],
                token_logprobs=lp,
                log_probability=sum(lp),
                raw_logits_reference=d.ref(path),
                raw_logits_finite=True,
                raw_logits_unmasked=True,
                raw_logits_width=2,
            )

    identity = dict(unit_id="u", qid="q", group_id="g", role="CAL")
    row = d.score(Runtime(), identity, dict(question="q", answer_text="a"), tmp_path)
    assert row["p_error"] == pytest.approx(0.5)
    assert row["continuation_token_evaluations"] == 4
    d.verify_likelihood(row["likelihoods"]["Yes"])
    broken = deepcopy(row["likelihoods"]["Yes"])
    broken["token_logprobs"].pop()
    with pytest.raises(ValueError, match="likelihood"):
        d.verify_likelihood(broken)
    Path(row["likelihoods"]["Yes"]["raw_logits_reference"]["path"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        d.verify_likelihood(row["likelihoods"]["Yes"])


def test_calibration_and_group_reduction() -> None:
    """REQ-REPORT-9107-V776: CAL alone selects temperature; siblings stay grouped."""
    rows = [
        dict(
            unit_id=str(i),
            qid=str(i // 3),
            group_id=str(i // 3),
            role="CAL" if i < 6 else "TEST",
            status="completed",
            yes_logprob=-1.0,
            no_logprob=-1.0,
        )
        for i in range(12)
    ]
    labels = {str(i): i % 2 for i in range(12)}
    result = d.reduce_scores(rows, labels, draws=20)
    assert result["temperature"] == 0.5
    assert result["metrics"]["Brier"] == 0.25
    assert result["metrics"]["accepted_error"] is None
    assert result["metrics"]["coverage"] == 0
    assert len(result["qid_rows"]) == 2
    assert result["uncertainty"]["draws"] == 20
    changed = dict(labels, **{str(i): 1 for i in range(6, 12)})
    assert d.reduce_scores(rows, changed, draws=20)["temperature"] == 0.5
    rows[-1]["status"] = "censored"
    with pytest.raises(ValueError, match="incomplete"):
        d.reduce_scores(rows, labels, draws=1)


def test_authentication_missing_zero_and_tamper(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9107-V776: a missing operand differs from an observed zero."""
    result = d.authenticate(tmp_path)
    assert result["blocker"] and result["checks"][0]["observed"] == "missing_file"
    results = tmp_path / "results"
    results.mkdir()
    authority = results / d.AUTHORITY
    atomic_json(authority, dict(source_ready_score=0))
    monkeypatch.setitem(d.PINS, d.AUTHORITY, d.ref(authority)["sha256"])
    result = d.authenticate(tmp_path)
    assert any(c["observed"] == 0 and not c["passed"] for c in result["checks"])
    atomic_json(authority, {})
    monkeypatch.setitem(d.PINS, d.AUTHORITY, d.ref(authority)["sha256"])
    assert any(c["observed"] == "missing_field" for c in d.authenticate(tmp_path)["checks"])
    atomic_json(authority, dict(source_ready_score=1))
    assert any(
        c["check"].endswith("_sha256") and not c["passed"]
        for c in d.authenticate(tmp_path)["checks"]
    )


@pytest.fixture
def panel(tmp_path: Path, monkeypatch: Any) -> Path:
    """REQ-REPORT-9107-V776: private controls carry the exact original group counts."""
    results = tmp_path / "results"
    results.mkdir()
    rows, views, labels = [], [], []
    for i in range(450):
        view = dict(question=f"q{i // 3}", answer_text=f"a{i}")
        row = dict(
            unit_id=f"u{i}",
            qid=f"q{i // 3}",
            group_id=f"g{i // 3}",
            role="CAL" if i < 150 else "TEST",
            feature_input_sha256=d.canonical_hash(view),
        )
        rows.append(row)
        views.append(view)
        labels.append(dict(row, annotator1=i % 2))
    files = {}
    for key in [
        "original_group_partition_reference",
        "static_protocol_reference",
        "evaluator_label_reference",
        "historical_producer_snapshot_reference",
    ]:
        path = tmp_path / (key + ".json")
        atomic_json(path, dict(rows=labels) if key == "evaluator_label_reference" else {})
        files[key] = d.ref(path)
    for key, sl in [
        ("shard_a_roster_reference", slice(0, 225)),
        ("shard_b_roster_reference", slice(225, 450)),
    ]:
        feature_path = tmp_path / (key + "_features.json")
        atomic_json(feature_path, dict(views=views[sl]))
        path = tmp_path / (key + ".json")
        atomic_json(path, dict(rows=rows[sl], feature_reference=d.ref(feature_path)))
        files[key] = d.ref(path)
    atomic_json(results / d.AUTHORITY, dict(files, source_ready_score=1))
    atomic_json(results / d.RUNTIME, dict(logits_ready_score=1))
    for name in d.PINS:
        monkeypatch.setitem(d.PINS, name, d.ref(results / name)["sha256"])
    return tmp_path


def test_complete_panel_and_bound_failures(panel: Path) -> None:
    """SCENARIO-REPORT-9107-V776: reference metadata does not change byte identity."""
    assert d.authenticate(panel)["blocker"] is None
    a = json.loads((panel / "results" / d.AUTHORITY).read_bytes())
    snap = a["historical_producer_snapshot_reference"]
    d.read_path(dict(snap, git_commit="historical-metadata"))
    Path(snap["path"]).write_text("changed")
    assert d.authenticate(panel)["blocker"] == "immutable_inputs"


def test_bad_likelihood_reduction_and_roles(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9107-V776: invalid logits and missing labels cannot pass."""
    path = tmp_path / "raw.npy"
    np.save(path, np.zeros((1, 2)))
    bad = dict(
        tokens=[0],
        token_logprobs=[-9.0],
        log_probability=-9.0,
        raw_logits_width=2,
        raw_logits_finite=True,
        raw_logits_unmasked=True,
        raw_logits_reference=d.ref(path),
    )
    with pytest.raises(ValueError, match="likelihood_reduction"):
        d.verify_likelihood(bad)
    with pytest.raises(ValueError, match="incomplete_labels"):
        d.reduce_scores(
            [dict(unit_id="x", role="CAL", status="completed", yes_logprob=0.0, no_logprob=0.0)],
            {"x": 0},
        )
    m = d.metrics(np.array([0.01, 0.02, 0.98, 0.99]), np.array([0, 1, 0, 1]))
    assert m["accepted_error"] == 0.5 and m["half_coverage_error"] == 0.5


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "preflight",
        "admission",
        "libraries",
        "memory",
        "load",
        "offload",
        "template",
        "deadline",
        "score",
    ],
)
def test_native_worker_boundaries(tmp_path: Path, monkeypatch: Any, failure: str) -> None:
    """REQ-REPORT-9107-V776: no failure authorizes fallback, generation or GPU1."""
    from types import SimpleNamespace
    import llama_cpp
    from carnot.inference import qwen_judge_9107 as native

    atomic_json(
        tmp_path / "job.json",
        dict(rows=[dict(unit_id="u")], views=[dict(question="q", answer_text="a")]),
    )
    states = []
    model = FixtureModel()
    model.metadata = dict(model.metadata)
    if failure == "template":
        model.metadata.pop("tokenizer.chat_template")
    model.close = lambda: states.append("closed")
    binding = tmp_path / "binding.py"
    binding.write_text("binding")
    if failure != "libraries":
        (tmp_path / "lib.so").write_text("native-build")
    monkeypatch.setattr(llama_cpp, "__file__", str(binding))
    monkeypatch.setattr(llama_cpp, "llama_print_system_info", lambda: b"CUDA")

    def load(**kwargs: Any) -> Any:
        assert kwargs["main_gpu"] == 0 and kwargs["n_gpu_layers"] == -1
        if failure == "load":
            raise ValueError("actual_load_failure")
        return model

    monkeypatch.setattr(llama_cpp, "Llama", load)

    def preflight(raw: Path) -> dict[str, Any]:
        if failure == "preflight":
            raise RuntimeError("cache_missing")
        return dict(
            model=dict(model_path=str(binding), sha256=d.ref(binding)["sha256"]),
            gpu=dict(uuid="GPU0", used_mb=0),
        )

    monkeypatch.setattr(native, "preflight", preflight)
    monkeypatch.setattr(
        native,
        "build_gguf_admission_record",
        lambda *a, **kw: dict(
            admitted=failure != "admission", content_metadata=dict(quantization="Q4_K_M")
        ),
    )
    state_calls = []

    def gpu() -> dict[str, Any]:
        state_calls.append(1)
        return dict(
            index=0,
            uuid="GPU0",
            free_mb=0 if failure == "memory" else 24000,
            used_mb=0 if len(state_calls) == 1 else 18000,
        )

    monkeypatch.setattr(native, "gpu_state", gpu)

    class Lease:
        document = {"phase": "preflight"}

        @classmethod
        def acquire(cls, **kwargs: Any) -> Any:
            assert kwargs["device_uuid"] == "GPU0"
            return cls()

        def owner_receipt(self) -> dict[str, Any]:
            return dict(device="GPU0")

        def transition(self, phase: str, **kwargs: Any) -> None:
            self.document = dict(phase=phase)

        def heartbeat(self) -> None:
            states.append("heartbeat")

        def release(self) -> dict[str, Any]:
            return dict(released=True)

        def close(self) -> None:
            states.append("lease_closed")

    monkeypatch.setattr(native, "GpuLease", Lease)
    log = tmp_path / "stderr"
    log.write_text("offloaded 65/65 layers" if failure != "offload" else "")
    monkeypatch.setattr(native.os, "readlink", lambda _: str(log))
    monkeypatch.setattr(native, "JudgeRuntime", lambda *a: SimpleNamespace(model=model))

    def score(*args: Any) -> dict[str, Any]:
        if failure == "score":
            raise ValueError("missing_raw_logits")
        return dict(unit_id="u")

    monkeypatch.setattr(d, "score", score)
    if failure == "deadline":
        times = iter([0, 1, 2, 4000 * 10**9, 4000 * 10**9 + 1])
        monkeypatch.setattr(native.time, "monotonic_ns", lambda: next(times))
    value = native.worker(tmp_path)
    assert bool(value["blocker"]) == (failure != "none")
    assert value["generated_tokens"] == 0
    assert value["model_loads"] == int(
        failure in {"none", "load", "offload", "template", "deadline", "score"}
    )
    assert value["receipt"]["invocation_counts"]["generation_calls_attempted"] == 0
    assert value["receipt"]["inference_substrate_class"] == (
        "model_load_no_generation" if value["model_loads"] else "blocked_no_run"
    )


def test_private_cli_replay_and_rehashed_tamper(panel: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9107-V776: actual cold CLI works outside checkout without PYTHONPATH."""
    from carnot.reporting import qwen_judge_runner_9107 as r
    from carnot.reporting.v709_execution import child

    raw = panel / "owned"
    raw.mkdir()
    scope = r.plan(raw)
    reference = r.measure(panel / "absent", raw, "20261115", scope)
    validation = [
        dict(name=c["name"], passed=True, normal_exit=True, timed_out=False)
        for c in scope["commands"]
    ]
    coverage = dict(statement_percent=100, excluded_statements=0)
    value = r.build(reference, validation, coverage)
    assert value["verdict_class"] == "blocked" and value["generated_tokens"] == 0
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["requested_inference_substrate_class"] == "model_load_no_generation"
    assert value["current_work_receipts"][0]["inference_substrate_class"] == "blocked_no_run"
    path = raw / (d.NAME + ".json")
    atomic_json(path, value)
    assert r.replay(path)["passed"]
    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        str(raw),
        str(d.ROOT / ".venv/bin/python"),
        str(d.ROOT / d.CLI),
        "--replay",
        str(path),
    ]
    assert child("cold_success", argv, raw / "cli", heartbeat=20)["passed"]
    value["completed_count"] = 450
    value["reproducibility_checksum"] = d.canonical_hash(value)
    atomic_json(path, value)
    assert child("rehashed_tamper", argv, raw / "cli", expected=1, heartbeat=20)["passed"]
    assert r.main(["--replay", str(path)]) == 1
    atomic_json(path, r.build(reference, validation, coverage))
    assert r.main(["--replay", str(path)]) == 0
    assert r.validate(path, raw / "artifact_checks")["passed"]
    atomic_json(raw / "candidate.json", json.loads(path.read_bytes()))
    monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=True))
    assert r.validate(path, raw)["passed"]


def test_measure_complete_scores_and_native_failure(panel: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9107-V776: failed child evidence stays distinct from complete raw scores."""
    from carnot.reporting import qwen_judge_runner_9107 as r
    from carnot.reporting.current_work_receipt import build_current_work_receipt

    raw = panel / "measurement"
    raw.mkdir()
    scope = r.plan(raw)

    def call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        native = raw / "native"
        np.save(native / "logits.npy", np.zeros((1, 2)))
        likelihood = dict(
            tokens=[0],
            token_logprobs=[-float(np.log(2))],
            log_probability=-float(np.log(2)),
            raw_logits_width=2,
            raw_logits_unmasked=True,
            raw_logits_finite=True,
            raw_logits_reference=d.ref(native / "logits.npy"),
        )
        job = json.loads((native / "job.json").read_bytes())
        rows = [
            dict(
                row,
                likelihoods=dict(Yes=likelihood, No=likelihood),
                yes_logprob=-float(np.log(2)),
                no_logprob=-float(np.log(2)),
                p_error=0.5,
                status="completed",
                outer_duration_s=0.1,
                started_monotonic_ns=0,
                ended_monotonic_ns=100000000,
                prompt_token_evaluations=2,
                continuation_token_evaluations=2,
            )
            for row in job["rows"]
        ]
        receipt = build_current_work_receipt(
            run_id="fixture",
            owner_pid=1,
            events=[],
            inference_substrate="cpu_control",
            inference_substrate_details={},
            inference_substrate_class="no_model_load",
            execution_venue="host",
            started_monotonic_ns=0,
            ended_monotonic_ns=1,
        )
        atomic_json(
            native / "runtime.json",
            dict(
                rows=rows,
                checks=[],
                blocker=None,
                model_loads=0,
                runtime_reference={},
                receipt=receipt,
            ),
        )
        return dict(passed=True)

    monkeypatch.setattr(r, "child", call)
    reference = r.measure(panel, raw, "20261115", scope)
    checks = [
        dict(name=c["name"], passed=True, normal_exit=True, timed_out=False)
        for c in scope["commands"]
    ]
    coverage = dict(statement_percent=100, excluded_statements=0)
    value = r.build(reference, checks, coverage)
    assert value["temperature"] == 0.5 and value["completed_count"] == 450
    assert value["metrics"]["Brier"] == 0.25
    work_path = Path(reference["path"])
    work = json.loads(work_path.read_bytes())
    work["runtime"]["rows"][0]["yes_logprob"] = -100
    atomic_json(work_path, work)
    with pytest.raises(ValueError, match="prediction_drift"):
        r.build(d.ref(work_path), checks, coverage)
    sealed_path = Path(work["judge_prediction_reference"]["path"])
    atomic_json(sealed_path, dict(rows=work["runtime"]["rows"]))
    work["judge_prediction_reference"] = d.ref(sealed_path)
    atomic_json(work_path, work)
    with pytest.raises(ValueError, match="raw_score_drift"):
        r.build(d.ref(work_path), checks, coverage)
    failed = panel / "failed"
    failed.mkdir()

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        native = failed / "native"
        atomic_json(
            native / "events.json",
            dict(
                events=[
                    dict(
                        call_id="load",
                        operation="model_load",
                        state="attempted",
                        scope="current",
                        transport="owned_runtime",
                        run_id="native",
                        owner_pid=1,
                        monotonic_ns=0,
                    )
                ]
            ),
        )
        return dict(passed=False, exit_code=-9, stderr_path=str(failed / "stderr"))

    monkeypatch.setattr(r, "child", fail)
    blocked = r.measure(panel, failed, "20261115", scope)
    result = r.build(blocked, checks, coverage)
    assert result["censored_count"] == 450 and len(result["rows"]) == 450
    assert result["verdict_class"] == "blocked"
    assert result["inference_substrate_class"] == "model_load_no_generation"
    assert result["current_work_receipts"][0]["inference_substrate_class"] == (
        "model_load_no_generation"
    )
    assert r.build(blocked, [], coverage)["verdict_class"] == "disqualified"


@pytest.mark.parametrize("case", ["publish", "owned_failure", "artifact_failure"])
def test_main_publication_guards(tmp_path: Path, monkeypatch: Any, case: str) -> None:
    """SCENARIO-REPORT-9107-V776: publication follows owned checks and strict artifact checks."""
    from carnot.reporting import qwen_judge_runner_9107 as r
    from carnot.inference import qwen_judge_9107 as native

    monkeypatch.setattr(native, "worker", lambda path: {})
    assert r.main(["--worker", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "wrong"])
    monkeypatch.setattr(r, "plan", lambda raw: dict(commands=[]))
    monkeypatch.setattr(r, "measure", lambda *a: {})
    monkeypatch.setattr(r, "execute", lambda *a: [])
    monkeypatch.setattr(r, "statement_coverage", lambda *a: {})
    monkeypatch.setattr(
        r,
        "build",
        lambda *a: dict(required_checks_passed=case != "owned_failure", verdict_class="blocked"),
    )
    monkeypatch.setattr(r, "validate", lambda *a: dict(passed=case != "artifact_failure"))
    published = []

    def publish(output: Path, value: Any, validator: Any) -> None:
        published.append(validator(output))

    monkeypatch.setattr(r, "publish_primary", publish)
    output = tmp_path / "out.json"
    assert r.main(["--private", str(tmp_path), "--output", str(output)]) == int(
        case == "artifact_failure"
    )
    assert bool(published) == (case == "publish")


def test_special_template_tokenization() -> None:
    """REQ-REPORT-9107-V776: template tokens are recognized without generation."""
    from carnot.inference.qwen_judge_9107 import JudgeRuntime

    runtime = object.__new__(JudgeRuntime)
    runtime.model = FixtureModel()
    assert runtime.tokenize("Yes") == list(b"Yes")


def test_statement_accounting(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9107-V776: missing statements remain visible without exclusions."""
    import coverage
    from carnot.reporting import qwen_judge_runner_9107 as r

    class Data:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def combine(self, **kwargs: Any) -> None:
            pass

        def save(self) -> None:
            (tmp_path / ".coverage").write_text("private-control")

        def analysis2(self, path: str) -> tuple[Any, ...]:
            return path, [1, 2], [], [2], "2"

    monkeypatch.setattr(coverage, "Coverage", Data)
    result = r.statement_coverage(tmp_path)
    assert result["statement_percent"] == 50 and result["excluded_statements"] == 0


@pytest.mark.parametrize("mutation", ["missing_key", "alignment", "feature", "counts"])
def test_panel_schema_controls(panel: Path, monkeypatch: Any, mutation: str) -> None:
    """SCENARIO-REPORT-9107-V776: malformed authority cannot silently reduce a cohort."""
    path = panel / "results" / d.AUTHORITY
    authority = json.loads(path.read_bytes())
    if mutation == "missing_key":
        del authority["evaluator_label_reference"]
    else:
        shard_path = Path(authority["shard_a_roster_reference"]["path"])
        shard = json.loads(shard_path.read_bytes())
        if mutation == "alignment":
            shard["rows"].pop()
        elif mutation == "feature":
            shard["rows"][0]["feature_input_sha256"] = "wrong"
        else:
            shard["rows"][0]["role"] = "TRAIN"
        atomic_json(shard_path, shard)
        authority["shard_a_roster_reference"] = d.ref(shard_path)
    atomic_json(path, authority)
    monkeypatch.setitem(d.PINS, d.AUTHORITY, d.ref(path)["sha256"])
    assert d.authenticate(panel)["blocker"] == "immutable_inputs"
