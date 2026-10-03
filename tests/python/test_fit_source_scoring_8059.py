"""REQ-REPORT-8059: current token custody, repeatability and terminal routes."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_8059_v698_fit_source_scoring as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def panel():
    return [
        dict(
            family_id=f"f{i}",
            source_cluster_id=f"s{i}",
            role="fit",
            slot=i,
            source_bytes=b"source".hex(),
            answer_bytes=b"a".hex(),
            eligible=True,
            exclusion_reason=None,
            target_tokens=[1],
            response_token_offsets=[[0, 1]],
            views={k: dict(tokens=[0, 1], response_start=1) for k in ["full", "no_source"]},
        )
        for i in range(8)
    ]


class Controller:
    def __init__(self, drift=0, error=False):
        self.calls = 0
        self.drift, self.error = drift, error

    def score(self, view, condition):
        assert condition == "fresh_full"
        self.calls += 1
        if self.error:
            raise RuntimeError("uncertain forward")
        lp = -1 - (self.drift if self.calls > 16 else 0)
        return dict(
            target_logprobs=[lp],
            conditional_logit_positions=[0],
            normalization_max_error=0,
            normalization_dtype="float64",
            context_identity=f"fresh-{self.calls}",
            context_closed=True,
        )


def test_capture_repeatability_and_roster(tmp_path):
    """SCENARIO-REPORT-8059-CAPTURE: duplicates do not add sources."""
    p = panel()
    rows = m.capture(p, Controller(), tmp_path, deadline=math.inf)
    v = m.reduce(p, rows)
    assert v["scored_tokens"] == 32 and v["forward_pass_counts"] == 32
    assert len(v["feature_rows"]) == 8 and v["pilot_passed"]
    assert all(r["drift"] == 0 for r in v["duplicate_drift_rows"])
    assert all(
        r["full_mean_nll"] == 1 and r["no_source_minus_full"] == 0 for r in v["feature_rows"]
    )
    assert len(list(tmp_path.glob("pass-*.json"))) == 32
    for field, value in [("token_id", 99), ("logit_position", 1), ("probability", 0.99)]:
        bad = deepcopy(rows)
        bad[0]["token_rows"][0][field] = value
        with pytest.raises(ValueError):
            m.reduce(p, bad)
    bad = deepcopy(rows)
    bad[0]["numerator"] = 99
    with pytest.raises(ValueError):
        m.reduce(p, bad)
    with pytest.raises(ValueError):
        m.reduce(p, rows[:-1])


@pytest.mark.parametrize("mode", ["drift", "failure", "deadline", "excluded", "tokens"])
def test_capture_stops_without_retry(tmp_path, mode, monkeypatch):
    """REQ-REPORT-8059: exclusions and stopped calls retain denominators."""
    p = panel() + [dict(panel()[0], family_id="later", source_cluster_id="later", slot=8)]
    c = Controller(drift=1e-4 if mode == "drift" else 0, error=mode == "failure")
    if mode == "excluded":
        p[0].update(eligible=False, exclusion_reason="over_context_budget")
    if mode == "tokens":
        monkeypatch.setitem(m.LIMITS, "tokens", 1)
    rows = m.capture(p, c, tmp_path, deadline=0 if mode == "deadline" else math.inf)
    v = m.reduce(p, rows)
    assert len(rows) == 36
    if mode == "drift":
        assert c.calls == 32 and not v["pilot_passed"]
    if mode == "failure":
        assert c.calls == 1 and rows[0]["status"] == "failed"
    if mode == "deadline":
        assert c.calls == 0
    if mode == "excluded":
        assert sum(r["status"] == "excluded" for r in rows) == 4


def test_authenticate_gates_and_manifest(tmp_path):
    """SCENARIO-REPORT-8059-TERMINAL: current structured gates authenticate bytes."""
    v = m.authenticate(m.ROOT, tmp_path / "raw")
    assert not v["failures"] and len(v["originals"]) == 96
    assert v["repository_health"]
    missing = m.authenticate(tmp_path / "absent", tmp_path / "missing")
    assert missing["failures"][0]["field"] == "resource_exists"
    assert m.manifest(tmp_path / "private")


def test_cli_private_success_blocked_mutation(tmp_path):
    """SCENARIO-REPORT-8059-TERMINAL: real CLI works outside checkout."""
    fixture = tmp_path / "panel.json"
    atomic_json(fixture, dict(panel=panel()))
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    config = os.environ.get("CARNOT_8059_COVERAGE_CONFIG")
    base = [str(m.ROOT / ".venv/bin/python")]
    if config:
        base += ["-m", "coverage", "run", "--rcfile=" + config]
    base += [str(m.ROOT / m.CLI)]

    def run(*args):
        return subprocess.run(
            base + list(args), cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )

    output = tmp_path / "results" / (m.NAME + ".json")
    r = run("--fixture-input", str(fixture), "--output", str(output))
    assert r.returncode == 0, r.stdout + r.stderr
    v = json.loads(output.read_text())
    assert v["verdict_class"] == "circular_positive" and v["fit_capture_ready_score"] == 0
    assert "preconditions_checked" in v
    assert v["model_invocation_counts"]["generation"]["attempted"] == 0
    receipt = json.loads(Path(v["current_work_receipt"]["path"]).read_text())
    assert receipt["invocation_counts"]["generation_calls_attempted"] == 0
    assert run("--cold-replay", str(output)).returncode == 0
    changed = deepcopy(v)
    changed["current_work_receipt"]["receipt_payload_sha256"] = "bad"
    atomic_json(output, changed)
    assert run("--cold-replay", str(output)).returncode == 1
    atomic_json(output, v)
    primitive = next((output.parent / "raw" / output.stem / "forwards").glob("pass-*.json"))
    saved = primitive.read_bytes()
    row = json.loads(saved)
    row["token_rows"][0]["token_id"] = 99
    atomic_json(primitive, row)
    assert run("--cold-replay", str(output)).returncode == 1
    primitive.write_bytes(saved)
    v["feature_rows"][0]["full_mean_nll"] = 999
    atomic_json(output, v)
    assert run("--cold-replay", str(output)).returncode == 1
    blocked = tmp_path / "blocked" / "results" / output.name
    assert (
        run(
            "--root", str(tmp_path / "missing"), "--output", str(blocked), "--private-validation"
        ).returncode
        == 0
    )
    b = json.loads(blocked.read_text())
    assert (
        b["verdict_class"] == "blocked"
        and b["model_invocation_counts"]["model_loads_attempted"] == 0
    )
    assert (
        run(
            "--fixture-input",
            str(fixture),
            "--output",
            str(tmp_path / "mutation" / "results" / output.name),
            "--mutate",
        ).returncode
        == 1
    )


class Tokenizer:
    def tokenize(self, b, **kwargs):
        return list(b)

    def detokenize(self, tokens):
        return bytes(tokens)

    def render(self, source, question):
        return b"q:" + source + b"\n"


@pytest.mark.parametrize("mode", ["ok", "long", "context", "boundary", "roundtrip", "overlap"])
def test_freeze_intact_bytes(mode):
    """REQ-REPORT-8059: complete source and answer are never truncated."""
    from carnot.verify.evidence_features_7980 import normalized

    source = b"s" * (5000 if mode == "context" else 1)
    answer = b"a" * (385 if mode == "long" else 1)
    r = dict(
        panel()[0],
        source_bytes=source.hex(),
        answer_bytes=answer.hex(),
        source_cluster_id=normalized(source),
    )
    tok = Tokenizer()
    if mode == "roundtrip":
        tok.detokenize = lambda _: b"wrong"
    if mode == "boundary":
        original = tok.tokenize
        tok.tokenize = lambda b, **kw: (
            original(b) + ([99] if b.endswith(b"a") and len(b) > 1 else [])
        )
    if mode == "overlap":
        with pytest.raises(ValueError):
            m.freeze([r, r], tok)
        return
    frozen = m.freeze([r], tok)
    assert frozen[0]["eligible"] == (mode == "ok")
    assert frozen[0]["answer_bytes"] == answer.hex()


def test_worker_reuses_runtime_and_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-8059: owned worker counts loads and uses fresh_full only."""
    plan = dict(originals=panel())
    atomic_json(tmp_path / "plan.json", plan)
    monkeypatch.setattr(m, "freeze", lambda originals, tok: originals)
    monkeypatch.setattr(m, "NativeController", lambda model, deadline: Controller())

    def fake_worker(plan_path, output):
        p = m.base.freeze_panel({}, Tokenizer())
        assert len(p["rows"]) == 8
        result = m.base.qualify(object(), tmp_path / "forwards")
        assert result["passed"]
        atomic_json(output, dict(qualification=result, checks=[]))

    monkeypatch.setattr(m.runtime, "worker", fake_worker)
    m.worker(tmp_path / "plan.json", tmp_path / "runtime.json")
    assert json.loads((tmp_path / "runtime.json").read_text())["qualification"]["pilot_passed"]


def test_eligibility_runs_only_after_seal(tmp_path):
    """SCENARIO-REPORT-8059-REDUCTION: separate target reader binds answer bytes."""
    p = panel()
    for role in ["public", "evaluator"]:
        rows = [
            dict(
                r,
                response_sha256=m.byte_hash(bytes.fromhex(r["answer_bytes"])),
                completely_annotated=True,
                custody_passed=True,
                eligible_y=i % 2,
            )
            for i, r in enumerate(p)
        ]
        atomic_json(tmp_path / (role + ".json"), dict(rows=rows))
    plan = dict(
        target_references={
            "fit": {r: m.reference(tmp_path / (r + ".json")) for r in ["public", "evaluator"]}
        }
    )
    features = m.reduce(p, m.capture(p, Controller(), tmp_path / "calls", deadline=math.inf))[
        "feature_rows"
    ]
    seal = dict(panel=p, feature_rows=features, sha256=canonical_hash(features))
    assert m.eligibility(plan, seal)["fit"]["class_counts"] == {"0": 4, "1": 4}
    bad = deepcopy(seal)
    bad["sha256"] = "bad"
    with pytest.raises(ValueError):
        m.eligibility(plan, bad)
    changed = deepcopy(seal)
    changed["panel"][0]["answer_bytes"] = b"wrong".hex()
    with pytest.raises(ValueError):
        m.eligibility(plan, changed)
    p[0]["answer_bytes"] = b"wrong".hex()
    with pytest.raises(ValueError):
        m.eligibility(plan, seal)


@pytest.mark.parametrize("failed", [False, True, "eligibility"])
def test_main_owned_children_and_validation(tmp_path, monkeypatch, failed):
    """REQ-REPORT-8059: owned failures cannot qualify a model capture."""
    output = tmp_path / "results" / (m.NAME + ".json")
    raw = output.parent / "raw" / output.stem
    monkeypatch.setattr(
        m,
        "authenticate",
        lambda root, dest: dict(
            originals=panel(),
            failures=[],
            references=[],
            target_references={},
            repository_health=[],
        ),
    )
    monkeypatch.setattr(
        m,
        "manifest",
        lambda private: [dict(name="test", expected_exit=0, argv=["true"], deadline_s=1)],
    )

    def check(root, spec, private, durable, **kw):
        log = tmp_path / (spec["name"] + ".log")
        log.write_text("owned stub unit control")
        if spec["name"] == "current_model_capture":
            rows = m.capture(panel(), Controller(), raw / "forwards", deadline=math.inf)
            atomic_json(
                raw / "runtime.json",
                dict(
                    panel=dict(rows=panel()), qualification=dict(m.reduce(panel(), rows), rows=rows)
                ),
            )
        elif spec["name"] == "independent_eligibility":
            atomic_json(raw / "support.json", {})
        else:
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1), missing_lines=[]) for p in m.OWNED
                    }
                ),
            )
        fail = failed is True or (
            failed == "eligibility" and spec["name"] == "independent_eligibility"
        )
        return dict(
            spec,
            passed=not fail,
            exit_code=int(fail),
            log_path=str(log),
            log_sha256=m.sha256_file(log),
        )

    monkeypatch.setattr(m, "run_check", check)
    assert m.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["fit_capture_ready_score"] == 0
    assert value["verdict_class"] == (
        "disqualified" if failed is True else "blocked" if failed else "null"
    )
    assert m.main(["--output", str(output)]) == 1
    if not failed:
        assert m.replay(output)
        plan, measured, validation, work = [
            json.loads((raw / n).read_text())
            for n in ["plan.json", "runtime.json", "validation.json", "work.json"]
        ]
        assert m.build(plan, {}, raw, validation, work)["completed_count"] == 0
        drifting = deepcopy(measured)
        r = drifting["qualification"]["rows"][16]
        t = r["token_rows"][0]
        t.update(log_probability=-1.0001, probability=math.exp(-1.0001))
        r.update(numerator=1.0001, mean_nll=1.0001)
        result = m.build(plan, drifting, raw, validation, work)
        assert result["verdict_class"] == "disqualified"
        gate = next(
            g for g in result["gate_check_summary"] if g["check"] == "duplicate_mean_nll_drift"
        )
        assert gate["upstream"] == m.TASK and gate["op"] == "<=" and gate["expected"] == 1e-6
        failed_runtime = dict(
            measured, checks=[dict(artifact_field="cache", expected=True, observed=False)]
        )
        assert m.build(plan, failed_runtime, raw, validation, work)["verdict_class"] == "blocked"
        row_path = raw / "forwards/pass-000.json"
        row_saved = row_path.read_bytes()
        row = json.loads(row_saved)
        row["token_rows"][0]["token_id"] = 99
        atomic_json(row_path, row)
        changed = deepcopy(value)
        for ref in changed["raw_shard_hashes"]:
            if ref["path"] == str(row_path):
                ref["sha256"] = m.sha256_file(row_path)
        atomic_json(output, changed)
        assert not m.replay(output)
        row_path.write_bytes(row_saved)
        work_path = raw / "work.json"
        work_saved = work_path.read_bytes()
        changed_work = dict(work, support={"fit": "fabricated"})
        atomic_json(work_path, changed_work)
        changed = deepcopy(value)
        for ref in changed["raw_shard_hashes"]:
            if ref["path"] == str(work_path):
                ref["sha256"] = m.sha256_file(work_path)
        atomic_json(output, changed)
        assert not m.replay(output)
        work_path.write_bytes(work_saved)
        atomic_json(output, value)
        changed = deepcopy(value)
        changed["code_config_hashes"][m.MODULE] = "bad"
        atomic_json(output, changed)
        assert not m.replay(output)
        changed = deepcopy(value)
        changed["validation_receipts"][0]["log_sha256"] = "bad"
        atomic_json(output, changed)
        assert not m.replay(output)
        atomic_json(output, value)
        (raw / "runtime.json").write_text("{}")
        assert not m.replay(output)
    assert not m.replay(tmp_path / "absent.json")
    assert m.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1


def test_unsafe_private_route_and_worker_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8059: fixture shortcuts cannot write protected results."""
    with pytest.raises(ValueError, match="private_route"):
        m.main(["--private-validation"])
    monkeypatch.setattr(m, "worker", lambda plan, output: None)
    assert m.main(["--worker", str(tmp_path / "plan")]) == 0
    atomic_json(tmp_path / "plan.json", dict(target_references={}))
    atomic_json(tmp_path / "seal.json", dict(panel=[], feature_rows=[], sha256=canonical_hash([])))
    assert (
        m.main(
            [
                "--eligibility",
                str(tmp_path / "seal.json"),
                "--output",
                str(tmp_path / "support.json"),
            ]
        )
        == 0
    )


def test_worker_budget_and_lease(tmp_path, monkeypatch):
    """REQ-REPORT-8059: loading shares the total budget and lease TTL."""
    monkeypatch.setattr(m.runtime.GpuLease, "acquire", lambda **kw: kw)
    atomic_json(tmp_path / "plan.json", dict(originals=[]))

    def fake(plan, output):
        assert m.runtime.GpuLease.acquire(task_id="unit")["ttl_s"] == m.LIMITS["seconds"] + 120
        assert m.runtime.bounded(lambda: 7, 1) == 7
        atomic_json(output, {})

    monkeypatch.setattr(m.runtime, "worker", fake)
    m.worker(tmp_path / "plan.json", tmp_path / "out.json")
    monkeypatch.setitem(m.LIMITS, "seconds", 0)
    with pytest.raises(TimeoutError):
        m.worker(tmp_path / "plan.json", tmp_path / "out2.json")


def test_mutated_parent_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8059: absent fields and stale terminal hashes fail closed."""
    parents = tmp_path / "repo" / "results"
    parents.mkdir(parents=True)
    for name, field in [
        ("experiment_8057_v698_fixture_consumer_contract", "fixture_consumer_ready_score"),
        ("experiment_8058_v698_sealed_evidence_methods", "source_protocol_ready_score"),
        ("experiment_8045_v697_scorer_workspace", "scorer_fixture_ready_score"),
    ]:
        atomic_json(
            parents / (name + ".json"),
            dict(
                verdict_class="disqualified",
                flagged_adversarial=True,
                scorer_code_hashes=[dict(path=str(m.ROOT / m.MODULE), sha256="bad")],
            ),
        )
    plan = m.authenticate(tmp_path / "repo", tmp_path / "raw")
    assert any(r["field"] == "terminal_hash_clean" for r in plan["failures"])
    assert any(r["field"] == "sha256" for r in plan["failures"])
