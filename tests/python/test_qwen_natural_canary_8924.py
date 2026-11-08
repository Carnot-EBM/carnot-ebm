"""REQ-REPORT-8924: dependent annotations never become independent truth."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import qwen_natural_canary_8924 as c
from carnot.reporting.current_work_receipt import atomic_json


def case():
    """SCENARIO-REPORT-8924-CONTRACT: small source bytes expose offset mistakes."""
    return dict(
        unit_id="a",
        cluster_id="doc",
        source_task="QA",
        source="é cat",
        response="dog",
        original=[dict(start=0, end=3, label="EvidentConflict")],
        revised=[],
    )


def output(decision="contradicted", **updates):
    """REQ-REPORT-8924: one completion is shared by both annotation arms."""
    data = dict(decision=decision, response_span=[0, 3], source_span=[2, 5])
    data.update(updates)
    return dict(
        unit_id="a", answer=json.dumps(data), usage=dict(prompt_tokens=10, completion_tokens=20)
    )


def test_scoring_and_valid_null():
    """REQ-REPORT-8924: exact locations do not settle semantic disagreement."""
    result = c.reduce([case()], [output()])
    assert result["completed_count"] == 1
    assert result["criterion_sensitivity"]["document_disagreements"] == 1
    assert [
        r["numerator"] for r in result["paired_criterion_rows"] if r["metric"] == "answer_agreement"
    ] == [1, 0]
    assert result["source_span_rows"][0]["source_text"] == "cat"
    assert result["source_span_rows"][0]["semantic_correctness"] == "unmeasured"
    assert result["independent_generalization_score"] == 0
    assert c.verdict(result, owned=True, live=True, blocker=None) == (
        1,
        "null",
        "complete_null_paired_criterion_canary",
    )
    assert c.verdict(result, owned=False, live=True, blocker=None)[1] == "disqualified"
    assert c.verdict(result, owned=True, live=False, blocker="resource")[1] == "blocked"


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "{}",
        "[]",
        "null",
        '{"decision":"maybe"}',
        '{"decision":"entailed","response_span":[0,99],"source_span":null}',
    ],
)
def test_bad_outputs_stay_completed(answer):
    """REQ-REPORT-8924: bad completed output is a null, not missing evidence."""
    row = output()
    row["answer"] = answer
    result = c.reduce([case()], [row])
    assert result["completed_count"] == 1
    assert result["parse_failures"] == 1
    assert c.verdict(result, owned=True, live=True, blocker=None)[1] == "null"


def test_labels_abstentions_and_censoring():
    """REQ-REPORT-8924: retain refusal, malformed spans and every scheduled slot."""
    item = case()
    for spans, expected in [
        ([], "entailed"),
        ([dict(label="SubtleBaselessInfo")], "baseless"),
        ([dict(label="IncorrectRefusal")], "entailed"),
    ]:
        assert c.criterion(spans) == expected
    result = c.reduce([item], [output("abstain", response_span=None, source_span=None)])
    assert result["abstentions"] == 1
    assert c.reduce([item], [])["censored_count"] == 1
    assert c.reduce([item], [dict(unit_id="a", status="failed")])["failed_count"] == 1
    invalid = c.reduce([item], [output(source_span=[0, 99])])
    assert invalid["source_span_rows"][0]["valid"] is False
    assert c.reduce([item], [output("baseless", source_span=None)])["parse_failures"] == 0
    with pytest.raises(ValueError):
        c.reduce([item], [output(), output()])
    with pytest.raises(ValueError):
        c.reduce([item], [dict(output(), unit_id="foreign")])
    with pytest.raises(ValueError):
        c.reduce([item, item], [])


def test_authentication_and_missing_gate(tmp_path):
    """SCENARIO-REPORT-8924-CONTRACT: authenticate original producers and source custody."""
    evidence = c.authenticate(c.ROOT / "results")
    assert len(evidence["cases"]) == 24
    assert len(evidence["manifest"]) == 24
    assert evidence["checks"] == []
    assert c.prompt(evidence["cases"][0]).count("source_span") >= 1
    with pytest.raises(c.ExternalGate) as exc:
        c.authenticate(tmp_path)
    assert exc.value.row["observed"] == "missing_file"
    path = tmp_path / c.UPSTREAMS[0][0]
    path.write_text("{}")
    with pytest.raises(c.ExternalGate) as exc:
        c.authenticate(tmp_path)
    assert exc.value.row["observed"] == "rejected_producer"


def test_reference_mutation_and_require(tmp_path):
    """REQ-REPORT-8924: absent fields, zero and changed bytes stay distinct."""
    path = tmp_path / "x.json"
    path.write_text("{}")
    ref = c.reference(path)
    c.verify_reference(ref)
    path.write_text("[]")
    with pytest.raises(c.ExternalGate):
        c.verify_reference(ref)
    for observed in ["missing_field", 0]:
        with pytest.raises(c.ExternalGate) as exc:
            c.require("ready", 1, observed, path, "upstream")
        assert exc.value.row["observed"] == observed
    c.require("ready", 1, 1, path, "upstream")


def test_prompt_has_no_annotation_leak():
    """REQ-REPORT-8924: model input excludes both criteria and their labels."""
    item = case()
    before = c.prompt(item)
    changed = deepcopy(item)
    changed["original"] = []
    changed["revised"] = item["original"]
    assert before == c.prompt(changed)
    assert "contradicted" in before and "128" not in before


def fake_runtime(monkeypatch, tmp_path, *, fail=None):
    """SCENARIO-REPORT-8924-CONTRACT: scripted transport never supplies live results."""
    from types import SimpleNamespace
    from carnot.inference import qwen_natural_runtime_8924 as r

    model = tmp_path / "model.gguf"
    model.write_text("fixture")
    backend = tmp_path / "server"
    backend.write_text("fixture")
    resources = dict(
        model=dict(model_path=str(model), sha256=c.sha256_file(model)),
        gpu=dict(index=0, uuid="gpu0", used_mb=0, free_mb=24000),
        llama_cpp=dict(sha256="binding"),
    )
    loaded = dict(
        authenticated=True,
        gguf_sha256=resources["model"]["sha256"],
        offload_layers=[66, 66],
        props=dict(build_info="build", chat_template="template"),
    )
    pilot = dict(
        model_receipt=dict(loaded, command=[str(backend)]), preconditions_checked=resources
    )

    class Lease:
        document = dict(phase="admitted")

        @classmethod
        def acquire(cls, **kwargs):
            if fail == "lease":
                raise RuntimeError("busy")
            assert kwargs["task_id"] == c.TASK
            return cls()

        def owner_receipt(self):
            return dict(lease_id="fixture")

        def transition(self, phase, **kwargs):
            self.document["phase"] = phase

        def heartbeat(self):
            pass

        def release(self):
            return dict(released=True)

    class Runtime:
        def __init__(self, *args):
            pass

        def load(self):
            if fail == "load":
                raise RuntimeError("load_failed")
            return loaded

        def generate(self, payload):
            assert payload["max_tokens"] == 128 and payload["n"] == 1
            assert payload["chat_template_kwargs"] == dict(enable_thinking=False)
            if fail == "generation":
                raise TimeoutError("resource_timeout")
            return dict(
                choices=[dict(message=dict(content=output()["answer"]))],
                usage=dict(prompt_tokens=20, completion_tokens=20),
            )

        def close(self):
            return dict(leak_free=fail != "cleanup")

    monkeypatch.setattr(r, "preflight", lambda raw: resources)
    monkeypatch.setattr(r, "gpu_state", lambda: resources["gpu"])
    monkeypatch.setattr(r, "GpuLease", Lease)
    monkeypatch.setattr(r, "QwenRuntime", Runtime)
    monkeypatch.setattr(
        r.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="build", stderr="", returncode=0),
    )
    monkeypatch.setattr(r, "bounded", lambda fn, *a, **k: fn())
    monkeypatch.setattr(r.time, "monotonic", lambda: 100.0)
    return r, dict(cases=[case()], pilot=pilot), resources, loaded


@pytest.mark.parametrize("failure", [None, "lease", "load", "generation", "cleanup"])
def test_owned_runtime_paths(monkeypatch, tmp_path, failure):
    """REQ-REPORT-8924: fresh lease, no retries, precise failed call counts and cleanup."""
    r, evidence, resources, loaded = fake_runtime(monkeypatch, tmp_path, fail=failure)
    result = r.run(evidence, tmp_path / "raw")
    assert result["counts"]["generation_calls_attempted"] <= 1
    if failure is None:
        assert len(result["outputs"]) == 1
        assert result["counts"]["generation_calls_completed"] == 1
        assert result["blocker"] == "generation_floor_not_met"
    elif failure == "cleanup":
        assert result["owned_failure"]
    else:
        assert result["blocker"]
    if failure == "generation":
        assert result["counts"]["generation_calls_failed"] == 1
    if failure == "load":
        assert result["counts"]["model_loads_failed"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["model", "binding", "backend", "gpu", "template", "offload", "tokens", "budget", "success"],
)
def test_runtime_drift_and_bounds(monkeypatch, tmp_path, mutation):
    """REQ-REPORT-8924: changed runtime blocks; no fallback or device one is permitted."""
    r, evidence, resources, loaded = fake_runtime(monkeypatch, tmp_path)
    if mutation == "model":
        evidence["pilot"]["model_receipt"] = dict(loaded, gguf_sha256="changed")
    if mutation == "binding":
        evidence["pilot"]["preconditions_checked"] = dict(
            resources, llama_cpp=dict(sha256="changed")
        )
    if mutation == "backend":
        evidence["pilot"]["model_receipt"]["props"] = dict(
            build_info="changed", chat_template="template"
        )
    if mutation == "gpu":
        resources["gpu"]["used_mb"] = 2000
    if mutation == "template":
        loaded["props"] = dict(build_info="build", chat_template="changed")
    if mutation == "offload":
        loaded["offload_layers"] = [0, 66]
    if mutation == "tokens":
        monkeypatch.setattr(
            r.QwenRuntime,
            "generate",
            lambda *a: dict(
                choices=[dict(message=dict(content="{}"))], usage=dict(completion_tokens=129)
            ),
        )
    if mutation in {"budget", "success"}:
        ticks = iter(range(0, 100000, 2000 if mutation == "budget" else 20))
        monkeypatch.setattr(r.time, "monotonic", lambda: next(ticks))
    result = r.run(evidence, tmp_path / "raw")
    if mutation == "success":
        assert result["live"] and result["blocker"] is None
    else:
        assert result["blocker"]


def test_build_replay_and_cold_cli(tmp_path):
    """SCENARIO-REPORT-8924-CONTRACT: replay rejects rehashed aggregates and outputs."""
    import os
    import subprocess
    import sys
    from carnot.reporting import qwen_natural_runner_8924 as r

    raw = tmp_path / "raw"
    raw.mkdir()
    measurement = dict(
        cases=[case()],
        runtime=dict(
            outputs=[output()],
            counts=dict(r.ZERO_INVOCATION_COUNTS),
            ledger=[],
            model_receipt={},
            live=False,
            blocker="fixture_no_run",
            checks=[],
            owned_failure=False,
        ),
    )
    operand = raw / "measurement.json"
    atomic_json(operand, measurement)
    value = r.build(measurement, [c.reference(operand)], [], [], {}, 0.1)
    target = tmp_path / (c.NAME + ".json")
    atomic_json(target, value)
    assert r.replay(target)["passed"]
    cli = c.ROOT / r.CLI
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [sys.executable, "-u", str(cli), "--cold-replay", str(target)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value["completed_count"] = 99
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(target, value)
    assert r.main(["--cold-replay", str(target)]) == 1
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(target)
    value["reproducibility_checksum"] = "bad"
    atomic_json(target, value)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(target)
    with pytest.raises(SystemExit):
        r.main(["--date", "bad"])
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_runner_controls(monkeypatch, tmp_path):
    """REQ-REPORT-8924: owned validation fails closed; external absence publishes blocked."""
    from carnot.reporting import qwen_natural_runner_8924 as r

    private = tmp_path / "private"
    private.mkdir()
    plan = r.validation_plan(private)
    assert any(x.name == "e2e021" for x in plan)
    assert "exclude_lines =\n" in (private / "coverage.ini").read_text()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "validate_owned", lambda raw: ([], {}))
    monkeypatch.setattr(r, "run_commands", lambda *a, **k: [])
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda target, value, validator: (atomic_json(target, value), dict(path=str(target)))[1],
    )
    assert r.main(["--date", "20261108"]) == 0
    target = tmp_path / "results" / (c.NAME + ".json")
    value = json.loads(target.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["gate_check_summary"][0]["observed"] == "missing_file"


@pytest.mark.parametrize("has_coverage", [False, True])
def test_validation_receipts(monkeypatch, tmp_path, has_coverage):
    """REQ-REPORT-8924: raw validation receipts bind code and measured exclusions."""
    from carnot.reporting import qwen_natural_runner_8924 as r

    private = tmp_path / "private"
    private.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(r.tempfile, "mkdtemp", lambda **kwargs: str(private))

    def commands(root, plan, **kwargs):
        if has_coverage:
            atomic_json(
                private / "coverage.json", dict(totals=dict(percent_covered=100, excluded_lines=0))
            )
        return [dict(name="owned", passed=True)]

    monkeypatch.setattr(r, "run_commands", commands)
    receipts, coverage = r.validate_owned(raw)
    assert bool(coverage) == has_coverage
    assert receipts[0]["executed_source_hashes"]


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_main_owned_path_and_terminal_failure(monkeypatch, tmp_path, terminal_failure):
    """SCENARIO-REPORT-8924-CONTRACT: a failed owned terminal gate disqualifies."""
    from carnot.reporting import qwen_natural_runner_8924 as r

    monkeypatch.setattr(r, "ROOT", tmp_path)
    evidence = dict(cases=[case()], manifest=[], upstream_references=[])
    monkeypatch.setattr(c, "authenticate", lambda results: evidence)
    receipts = [dict(passed=True, name="owned")]
    monkeypatch.setattr(
        r, "validate_owned", lambda raw: (receipts, dict(percent_covered=100, excluded_lines=0))
    )
    runtime = dict(
        outputs=[],
        counts=dict(r.ZERO_INVOCATION_COUNTS),
        ledger=[],
        model_receipt={},
        live=False,
        checks=[dict(check="resource", observed="busy")],
        blocker="busy",
        owned_failure=False,
    )
    monkeypatch.setattr(r, "run", lambda evidence, private: runtime)
    monkeypatch.setattr(
        r,
        "run_commands",
        lambda *a, **k: [
            dict(
                passed=not terminal_failure,
                name="strict_rows",
                exit_code=1 if terminal_failure else 0,
            )
        ],
    )
    calls = []

    def publish(target, value, validator):
        atomic_json(target, value)
        report = validator(target)
        calls.append(report)
        if not report["passed"]:
            raise ValueError("candidate_rejected")
        return dict(path=str(target))

    monkeypatch.setattr(r, "publish_primary", publish)
    assert r.main(["--date", "20261108"]) == 0
    value = json.loads((tmp_path / "results" / (c.NAME + ".json")).read_text())
    assert value["verdict_class"] == ("disqualified" if terminal_failure else "blocked")
    assert len(calls) == (2 if terminal_failure else 1)


def test_replay_rehashed_output_mutations(tmp_path):
    """REQ-REPORT-8924: rehashing a primary cannot change output, counts or verdict."""
    from carnot.reporting import qwen_natural_runner_8924 as r

    payload = output()
    raw_response = dict(
        choices=[dict(message=dict(content=payload["answer"]))], usage=payload["usage"]
    )
    payload["raw_response"] = raw_response
    runtime = dict(
        outputs=[payload],
        counts=dict(r.ZERO_INVOCATION_COUNTS),
        ledger=[dict(call_id="a", response_sha256=c.canonical_hash(raw_response))],
        model_receipt={},
        live=True,
        blocker=None,
        checks=[],
        owned_failure=False,
    )
    measurement = dict(cases=[case()], runtime=runtime)
    operand = tmp_path / "measurement.json"
    atomic_json(operand, measurement)
    value = r.build(measurement, [c.reference(operand)], [], [], {}, 1)
    target = tmp_path / (c.NAME + ".json")
    atomic_json(target, value)
    assert r.replay(target)["passed"]
    for key, replacement, expected in [
        ("model_invocation_counts", {}, "runtime_binding"),
        ("verdict_class", "positive", "qualification"),
    ]:
        changed = deepcopy(value)
        changed[key] = replacement
        changed["reproducibility_checksum"] = r.checksum(changed)
        atomic_json(target, changed)
        with pytest.raises(ValueError, match=expected):
            r.replay(target)
    measurement["runtime"]["outputs"][0]["raw_response"]["choices"][0]["message"]["content"] = (
        "changed"
    )
    atomic_json(operand, measurement)
    value["raw_shard_hashes"] = [c.reference(operand)]
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(target, value)
    with pytest.raises(ValueError, match="output_binding"):
        r.replay(target)


def test_cluster_diagnostics():
    """REQ-REPORT-8924: source clusters retain quality, coverage and disagreement counts."""
    result = c.reduce([case()], [output()])
    row = result["source_cluster_diagnostics"][0]
    assert row["cluster_id"] == "doc" and row["completed_count"] == 1
    assert row["parse_failures"] == row["abstentions"] == 0
    assert row["criterion_disagreements"] == 1
    assert result["case_rows"][0]["criterion_disagreement"]


@pytest.mark.parametrize("failure", ["close", "release", "transition"])
def test_cleanup_errors_are_owned(monkeypatch, tmp_path, failure):
    """REQ-REPORT-8924: cleanup failures cannot escape a terminal owned result."""
    r, evidence, _, _ = fake_runtime(monkeypatch, tmp_path)

    def error(*args, **kwargs):
        raise RuntimeError("cleanup_failed")

    if failure == "transition":
        previous = r.GpuLease.transition

        def transition(self, phase, **kwargs):
            if phase == "validating":
                raise RuntimeError("journal_failed")
            return previous(self, phase, **kwargs)

        monkeypatch.setattr(r.GpuLease, "transition", transition)
    else:
        monkeypatch.setattr(
            r.QwenRuntime if failure == "close" else r.GpuLease,
            "close" if failure == "close" else "release",
            error,
        )
    result = r.run(evidence, tmp_path / "raw")
    assert result["owned_failure"]
    assert result["checks"]


def test_uncaught_publication_error(monkeypatch, tmp_path):
    """REQ-REPORT-8924: identity conflicts remain visible instead of being rewritten."""
    from carnot.reporting import qwen_natural_runner_8924 as r

    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "validate_owned", lambda raw: ([], {}))
    monkeypatch.setattr(r, "run_commands", lambda *a, **k: [])

    def reject(*args):
        raise ValueError("conflicting_identity")

    monkeypatch.setattr(r, "publish_primary", reject)
    with pytest.raises(ValueError, match="conflicting_identity"):
        r.main(["--date", "20261108"])


def test_live_replay_source_and_ledger_binding(monkeypatch, tmp_path):
    """REQ-REPORT-8924: rehashed source or ledger replacements cannot borrow runtime."""
    from carnot.reporting import qwen_natural_runner_8924 as r
    from carnot.verify.qwen_development_capture_7995 import Ledger

    payload = output()
    response = dict(choices=[dict(message=dict(content=payload["answer"]))], usage=payload["usage"])
    payload["raw_response"] = response
    request = dict(c.SETTINGS, messages=[dict(role="user", content=c.prompt(case()))])
    ledger = Ledger(tmp_path / "invocation_ledger.json")
    ledger.start("model_load", "load", {})
    ledger.finish("load", "completed", {})
    ledger.start("generation", "a", request)
    ledger.finish("a", "completed", response)
    runtime = dict(
        outputs=[payload],
        counts=ledger.counts(),
        ledger=ledger.rows,
        model_receipt=dict(
            authenticated=True,
            offload_layers=[66, 66],
            gguf_sha256="hash",
            preflight_gguf_sha256="hash",
        ),
        live=True,
        generation_duration_s=20,
        blocker=None,
        checks=[],
        owned_failure=False,
    )
    measurement = dict(cases=[case()], manifest=[], runtime=runtime)
    operand = tmp_path / "measurement.json"
    atomic_json(operand, measurement)
    monkeypatch.setattr(c, "authenticate", lambda results: dict(cases=[case()], manifest=[]))
    value = r.build(
        measurement,
        [c.reference(operand), c.reference(ledger.path)],
        [],
        [dict(passed=True)],
        dict(percent_covered=100, excluded_lines=0),
        20,
    )
    target = tmp_path / (c.NAME + ".json")
    atomic_json(target, value)
    assert r.replay(target)["passed"]
    for key, replacement in [("cases", []), ("manifest", [dict(unit_id="extra")])]:
        changed = deepcopy(measurement)
        changed[key] = replacement
        atomic_json(operand, changed)
        updated = deepcopy(value)
        updated["raw_shard_hashes"][0] = c.reference(operand)
        updated["reproducibility_checksum"] = r.checksum(updated)
        atomic_json(target, updated)
        with pytest.raises(ValueError):
            r.replay(target)
    changed = deepcopy(measurement)
    changed["runtime"]["ledger"] = []
    atomic_json(operand, changed)
    value["raw_shard_hashes"][0] = c.reference(operand)
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(target, value)
    with pytest.raises(ValueError, match="ledger"):
        r.replay(target)
    changed = deepcopy(measurement)
    changed["runtime"]["ledger"][1]["request_sha256"] = "changed"
    atomic_json(ledger.path, dict(rows=changed["runtime"]["ledger"]))
    atomic_json(operand, changed)
    value["raw_shard_hashes"] = [c.reference(operand), c.reference(ledger.path)]
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(target, value)
    with pytest.raises(ValueError, match="prompt_binding"):
        r.replay(target)
