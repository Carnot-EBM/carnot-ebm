"""REQ-REPORT-8023: fixed human answers and sealed source roles govern fitting."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot import experiment_8023_v695_likelihood_calibration as e
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import likelihood_calibration_8023 as c
from test_likelihood_protocol_8022 import Tokens, public


def panel(n=2):
    return s.freeze_panel(public(n), Tokens(), dict(fit=n, tune=n))


class Runtime(Tokens):
    """Byte-token fixture exercises scoring arithmetic without science credit."""

    def reset(self):
        pass

    def eval(self, tokens):
        self.n_tokens = len(tokens)
        self.scores = np.zeros((len(tokens), 257))


def test_capture_and_independent_reduction(tmp_path):
    """SCENARIO-REPORT-8023-SCORE: every view retains primitive tokens and offsets."""
    p = panel()
    rows = c.capture(p, Runtime(), tmp_path, deadline=float("inf"))
    reduced = c.reduce(p, rows)
    assert len(rows) == 16 and all(r["status"] == "completed" for r in rows)
    assert reduced["passed"] and len(reduced["source_feature_rows"]) == 4
    assert reduced["scored_tokens"] == 80
    assert all(r["no_source_minus_full"] == 0 for r in reduced["source_feature_rows"])
    bad = copy.deepcopy(rows)
    bad[0]["token_rows"][0]["log_probability"] -= 0.1
    with pytest.raises(ValueError, match="token_aggregate"):
        c.reduce(p, bad)
    for field, value in [("generated_tokens", 1), ("answer_bytes", "00")]:
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            c.reduce(p, bad)
    bad = copy.deepcopy(rows)
    bad[1]["token_rows"][0]["log_probability"] -= 0.1
    bad[1]["numerator"] += 0.1
    bad[1]["mean_nll"] = bad[1]["numerator"] / bad[1]["denominator"]
    assert not c.reduce(p, bad)["passed"]


def test_nonstarts_failures_and_alignment(tmp_path):
    """SCENARIO-REPORT-8023-SCORE: failures never delete or replace intended views."""
    p = panel(1)
    censored = c.capture(p, Runtime(), tmp_path / "censor", deadline=0)
    assert len(censored) == 8 and all(r["status"] == "censored" for r in censored)
    assert not c.reduce(p, censored)["passed"]

    class Broken(Runtime):
        def eval(self, tokens):
            raise TimeoutError("fixture_timeout")

    rows = c.capture(p, Broken(), tmp_path / "failure", deadline=float("inf"))
    assert rows[0]["status"] == "failed" and rows[1]["status"] == "censored"
    assert not c.reduce(p, rows)["passed"]

    class Shifted(Runtime):
        def eval(self, tokens):
            super().eval(tokens)
            self.n_tokens -= 1

    assert (
        c.capture(p, Shifted(), tmp_path / "shift", deadline=float("inf"))[0]["status"] == "failed"
    )
    with pytest.raises(ValueError, match="slot_roster"):
        c.reduce(p, [])


def data(n=48):
    return [
        dict(
            family_id=f"{role}-{i}",
            source_cluster_id=f"{role}-{i}",
            role=role,
            q=(i % 5) / 5,
            y=i % 2,
            full_mean_nll=1 + i / 100,
            no_source_minus_full=(i % 3) / 10,
            removal_minus_full=(i % 7) / 10,
            status="completed",
        )
        for role, count in [("fit", n), ("tune", 24)]
        for i in range(count)
    ]


def test_fit_support_scaling_and_identity(tmp_path):
    """SCENARIO-REPORT-8023-FIT: tune chooses ridge; labels never become features."""
    rows = data()
    result = c.train(rows, tmp_path)
    assert result["ready"] and len(result["heads"]) == 6
    assert result["role_support"]["fit"]["independent"] == 48
    assert result["config"]["ridge_grid"] == [0.001, 0.01, 0.1, 1]
    energy, identity = result["heads"][:2]
    assert np.allclose(c.predict(energy, rows), c.predict(identity, rows), atol=1e-15)
    assert result["config"]["thresholds"] == {"accept_below": 0.1, "reject_above": 0.5}
    assert all(h["converged"] and h["finite"] for h in result["heads"])
    for ref in result["checkpoints"]:
        assert json.loads(Path(ref["path"]).read_text()) in result["heads"]
    missing = data(10)
    assert not c.train(missing, tmp_path / "missing")["ready"]
    rows[0]["y"] = None
    assert not c.train(rows, tmp_path / "unknown")["role_support"]["fit"]["passed"]


def test_missing_inputs_private_cli(tmp_path):
    """SCENARIO-REPORT-8023-PUBLISH: missing external contracts finish blocked."""
    output = tmp_path / "results" / (e.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if os.environ.get("CARNOT_8023_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["CARNOT_8023_COVERAGE_CONFIG"]]
    run = subprocess.run(
        [
            *prefix,
            str(e.ROOT / e.CLI),
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["likelihood_calibration_ready_score"] == 0
    assert value["verdict_class"] == "blocked" and value["eval_label_access_count"] == 0
    assert value["gate_check_summary"][0]["artifact_field"]
    e.replay(value)


def test_runtime_adapter_gates_and_retokenization(tmp_path, monkeypatch):
    """REQ-REPORT-8023: original GGUF and frozen contexts precede any load."""
    from carnot.inference import likelihood_capture_8023 as runtime

    p = panel(1)
    with pytest.raises(ValueError, match="fit_tune_roster"):
        runtime.verify_panel(p, Tokens())
    bad = copy.deepcopy(p)
    bad["rows"][0]["target_tokens"] = [0]
    with pytest.raises(ValueError, match="frozen_panel_drift"):
        runtime.verify_panel(bad, Tokens())
    path, output = tmp_path / "plan.json", tmp_path / "capture.json"
    atomic_json(path, dict(panel=p, gguf_sha256="expected"))
    monkeypatch.setattr(runtime.runtime, "cached_current_model", lambda: None)
    runtime.worker(path, output)
    assert not json.loads(output.read_text())["checks"][0]["passed"]
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    monkeypatch.setattr(
        runtime.runtime, "cached_current_model", lambda: dict(model_path=str(model))
    )
    runtime.worker(path, output)
    assert json.loads(output.read_text())["checks"][0]["artifact_field"] == "gguf_sha256"


def test_join_only_original_complete_targets(tmp_path):
    """SCENARIO-REPORT-8023-FIT: incomplete annotations remain unknown."""
    p = panel(1)
    refs = {}
    for role in ["fit", "tune"]:
        item = next(r for r in p["rows"] if r["role"] == role)
        pub, ev = tmp_path / (role + "-pub.json"), tmp_path / (role + "-ev.json")
        atomic_json(pub, dict(rows=[dict(item, q=0.4)]))
        atomic_json(
            ev,
            dict(
                rows=[
                    dict(
                        family_id=item["family_id"],
                        eligible_y=1,
                        completely_annotated=True,
                        custody_passed=True,
                        response_sha256=e.byte_hash(bytes.fromhex(item["answer_bytes"])),
                    )
                ]
            ),
        )
        refs[role] = dict(public=e.reference(pub), evaluator=e.reference(ev))
    features = c.reduce(p, c.capture(p, Runtime(), tmp_path / "passes", deadline=float("inf")))
    joined = e.join(dict(panel=p, target_references=refs), features)
    assert [r["y"] for r in joined] == [1, 1]
    v = json.loads(ev.read_text())
    v["rows"][0]["completely_annotated"] = False
    atomic_json(ev, v)
    refs["tune"]["evaluator"] = e.reference(ev)
    assert e.join(dict(panel=p, target_references=refs), features)[1]["y"] is None


def source_fixture(tmp_path, n=1):
    """SCENARIO-REPORT-8023-FIT: private original annotations bind fixture bytes."""
    p = s.freeze_panel(public(64), Tokens(), dict(fit=64, tune=32)) if n == 64 else panel(n)
    refs = {}
    for role in ("fit", "tune"):
        items = [r for r in p["rows"] if r["role"] == role]
        pub, ev = tmp_path / (role + "-public.json"), tmp_path / (role + "-labels.json")
        atomic_json(pub, dict(rows=[dict(item, q=0.4) for item in items]))
        atomic_json(
            ev,
            dict(
                rows=[
                    dict(
                        family_id=item["family_id"],
                        eligible_y=i % 2,
                        completely_annotated=True,
                        custody_passed=True,
                        response_sha256=e.byte_hash(bytes.fromhex(item["answer_bytes"])),
                    )
                    for i, item in enumerate(items)
                ]
            ),
        )
        refs[role] = dict(public=e.reference(pub), evaluator=e.reference(ev))
    return dict(
        panel=p,
        target_references=refs,
        checks=[dict(passed=True)],
        references=[],
        gguf_sha256="fixture",
    )


def test_authenticate_contracts(tmp_path, monkeypatch):
    """REQ-REPORT-8023: contract fields and complete disjoint roster are gates."""
    p = s.freeze_panel(public(64), Tokens(), dict(fit=64, tune=32))
    targets = source_fixture(tmp_path)
    root = tmp_path / "root"
    dest = root / "results"
    dest.mkdir(parents=True)
    a, b = dest / (e.protocol.NAME + ".json"), dest / "experiment_8019_v695_eligible_targets.json"
    value = dict(
        experiment_id=8022,
        flagged_adversarial=False,
        likelihood_protocol_ready_score=1,
        token_scoring_ready_score=1,
        method_map=s.METHODS,
        public_panel_manifest=p,
        gguf_sha256="fixture",
        protocol_fingerprint="fixture",
    )
    labels = dict(
        experiment_id=8019,
        flagged_adversarial=False,
        fit_targets_ready_score=1,
        public_manifests={r: refs["public"] for r, refs in targets["target_references"].items()},
        evaluator_manifests={
            r: refs["evaluator"] for r, refs in targets["target_references"].items()
        },
    )
    atomic_json(a, value)
    atomic_json(b, labels)
    monkeypatch.setattr(e.protocol, "replay", lambda _: None)
    assert not all(r["passed"] for r in e.authenticate(tmp_path / "absent")["checks"])
    assert all(r["passed"] for r in e.authenticate(root)["checks"])
    for change in ["method", "roster", "overlap"]:
        bad = copy.deepcopy(value)
        if change == "method":
            bad["method_map"] = {}
        elif change == "roster":
            bad["public_panel_manifest"]["rows"].pop()
        else:
            bad["public_panel_manifest"]["rows"][1]["source_normalized_hash"] = p["rows"][0][
                "source_normalized_hash"
            ]
        atomic_json(a, bad)
        assert not all(r["passed"] for r in e.authenticate(root)["checks"])
    atomic_json(a, value)
    labels["public_manifests"]["fit"]["sha256"] = "changed"
    atomic_json(b, labels)
    assert not all(r["passed"] for r in e.authenticate(root)["checks"])


@pytest.mark.parametrize("n", [2, 64])
def test_pipeline_all_owned_routes(tmp_path, monkeypatch, n):
    """SCENARIO-REPORT-8023-PUBLISH: owned states survive publication and replay."""
    plan = source_fixture(tmp_path, n)
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    real_terminal = e.terminal

    def commands(root, specs, **kwargs):
        receipts = [
            dict(
                name=x.name,
                scope=x.scope,
                passed=True,
                exit_code=0,
                log_path=str(tmp_path / "log"),
                log_sha256=e.sha256_file(tmp_path / "log"),
            )
            for x in specs
        ]
        if specs[0].name == "current_fit_tune_capture":
            raw = output.parent / "raw" / output.stem
            rows = c.capture(plan["panel"], Runtime(), raw / "forwards", deadline=float("inf"))
            atomic_json(
                raw / "capture.json",
                dict(
                    qualification=dict(rows=rows),
                    duration_s=3,
                    gguf_sha256="fixture",
                    offload_evidence=dict(supported=True),
                    cleanup=dict(model_closed=True, lease_released=True),
                    model_invocation_counts=dict(
                        e.ZERO_INVOCATION_COUNTS, model_loads_attempted=1, model_loads_completed=1
                    ),
                ),
            )
        if any(x.name == "coverage" for x in specs):
            arg = next(x for x in specs if x.name == "coverage").argv
            atomic_json(Path(arg[arg.index("-o") + 1]), dict(totals=dict(percent_covered=100)))
        return receipts

    (tmp_path / "log").write_text("fixture log")
    monkeypatch.setattr(e, "run_commands", commands)
    monkeypatch.setattr(e.protocol, "verify_references", lambda _: None)
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["rows"] and value["likelihood_calibration_ready_score"] == int(n == 64)
    assert e.main(["--cold-replay", str(output)]) == 0
    bad = dict(value, claim_scope="changed")
    with pytest.raises(ValueError, match="cold_reduction_drift"):
        e.replay(bad)
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    original_capture = json.loads((raw / "capture.json").read_text())
    bad_capture = copy.deepcopy(original_capture)
    bad_capture["qualification"]["rows"][0]["mean_nll"] += 0.1
    atomic_json(raw / "capture.json", bad_capture)
    mutated = copy.deepcopy(value)
    mutated["raw_shard_hashes"] = [
        e.reference(Path(r["path"])) for r in mutated["raw_shard_hashes"]
    ]
    with pytest.raises(ValueError, match="per_view_checkpoint_drift"):
        e.replay(mutated)
    atomic_json(raw / "capture.json", original_capture)
    if n == 64:
        headref = next(r for r in value["checkpoint_references"] if "/heads/" in r["path"])
        alternate = tmp_path / "wrong-head.json"
        atomic_json(alternate, {})
        original_checked = e.checked
        with monkeypatch.context() as m:
            m.setattr(e, "checked", lambda r: alternate if r == headref else original_checked(r))
            with pytest.raises(ValueError, match="head_reload"):
                e.replay(value)
    assert e.main(["--output", str(output)]) == 1
    monkeypatch.setattr(e.runtime, "worker", lambda *args: None)
    assert e.main(["--runtime-child", str(tmp_path / "missing")]) == 0
    # A changed target join is rejected even if another reduction is edited to match it.
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    joined = json.loads((raw / "joined.json").read_text())
    joined[0]["q"] = 0.9
    atomic_json(raw / "joined.json", joined)
    value["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    with pytest.raises(ValueError, match="target_join_drift"):
        e.replay(value)
    monkeypatch.setattr(e, "terminal", real_terminal)


def test_training_and_token_negative_states(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-FIT: nonfinite predictors and false identity fail."""
    rows = data()
    rows[0]["full_mean_nll"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite_features"):
        c.train(rows, tmp_path / "nan")
    original = c.predict
    monkeypatch.setattr(
        c, "predict", lambda h, r: original(h, r) + (0.1 if h["arm"] == "sigmoid_identity" else 0)
    )
    with pytest.raises(ValueError, match="sigmoid_identity"):
        c.train(data(), tmp_path / "identity")
    p = panel(1)
    rows = c.capture(p, Runtime(), tmp_path / "tokens", deadline=float("inf"))
    rows[0]["token_rows"][0]["token_id"] = 0
    with pytest.raises(ValueError, match="token_identity"):
        c.reduce(p, rows)


def test_runtime_adapter_bounded_execution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-SCORE: adapter limits calls and lease to current work."""
    from carnot.inference import likelihood_capture_8023 as r
    from unittest.mock import patch

    p = s.freeze_panel(public(64), Tokens(), dict(fit=64, tune=32))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    digest = e.sha256_file(model)
    plan, output = tmp_path / "plan.json", tmp_path / "capture.json"
    atomic_json(plan, dict(panel=p, gguf_sha256=digest))
    monkeypatch.setattr(r.runtime, "cached_current_model", lambda: dict(model_path=str(model)))
    monkeypatch.setattr(r.runtime, "bounded", lambda fn, timeout: fn())
    monkeypatch.setattr(r.GpuLease, "acquire", lambda **kw: kw)

    def worker(*args):
        assert r.runtime.GpuLease.acquire(ttl_s=1)["ttl_s"] == 2520
        assert r.runtime.bounded(lambda: 9, 300) == 9
        assert r.runtime.sha256_file(model) == digest
        assert s.freeze_panel({}, Tokens()) == p
        s.progress("before_benchmark", 0, 0, 8)
        result = s.qualify(Runtime(), tmp_path / "passes")
        s.progress("after_benchmark", 0, 8)
        assert result["passed"] and result["forward_pass_counts"] == 384
        with patch.object(r.time, "monotonic", return_value=1e30):
            with pytest.raises(TimeoutError, match="total_model_work_cap"):
                r.runtime.bounded(lambda: 9, 1)

    monkeypatch.setattr(r.runtime, "worker", worker)
    r.worker(plan, output)


def test_cli_and_terminal_failure_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-PUBLISH: terminal rejection never publishes success."""
    import runpy

    monkeypatch.setattr(e, "main", lambda _: 0)
    monkeypatch.setattr(sys, "argv", [e.CLI])
    # The actual thin script reaches its guarded entry point without PYTHONPATH.
    monkeypatch.setattr(e, "main", lambda: 0)
    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert exit_code.value.code == 0


def test_failed_child_cold_and_published_readers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-PUBLISH: retain failed subprocess assertions."""
    monkeypatch.setattr(e, "authenticate", lambda _: dict(source_fixture(tmp_path), checks=[]))

    def failed(root, specs, **kwargs):
        return [
            dict(
                name=s.name,
                scope=s.scope,
                passed=False,
                exit_code=1,
                log_path="fixture",
                log_sha256="fixture",
            )
            for s in specs
        ]

    monkeypatch.setattr(e, "run_commands", failed)
    monkeypatch.setattr(e, "commands", lambda _: [])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 1
    raw = output.parent / "raw" / output.stem
    assert json.loads((raw / "capture.json").read_text())["checks"][0]["observed"] == 1
    # A later terminal reader failure is a separate owned failure, never input evidence.
    output = tmp_path / "again" / (e.NAME + ".json")
    monkeypatch.setattr(
        e,
        "run_commands",
        lambda root, specs, **kw: [dict(name=s.name, scope=s.scope, passed=True) for s in specs],
    )
    monkeypatch.setattr(
        e, "authenticate", lambda _: dict(source_fixture(tmp_path), checks=[dict(passed=False)])
    )
    monkeypatch.setattr(e, "publish_primary", lambda *args: {})
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=False))
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kw: dict(passed=True))
    assert e.main(["--output", str(output)]) == 1


def test_cold_fit_mutations(tmp_path):
    """SCENARIO-REPORT-8023-PUBLISH: cold readers recompute scaler and selection."""
    rows = data()
    fitted = c.train(rows, tmp_path / "fit")
    c.check_fit(rows, fitted)
    bad = copy.deepcopy(fitted)
    bad["config"] = {}
    with pytest.raises(ValueError, match="fit_contract_drift"):
        c.check_fit(rows, bad)
    bad = copy.deepcopy(fitted)
    trial = json.loads(Path(bad["trials"][0]["path"]).read_text())
    trial["mean"][0] += 1
    path = tmp_path / "trial.json"
    atomic_json(path, trial)
    bad["trials"][0] = e.reference(path)
    with pytest.raises(ValueError, match="fit_state_drift"):
        c.check_fit(rows, bad)
    bad = copy.deepcopy(fitted)
    bad["heads"][0]["parameters"][0] += 0.1
    with pytest.raises(ValueError, match="tune_selection_drift"):
        c.check_fit(rows, bad)
    bad = copy.deepcopy(fitted)
    bad["selected_comparator"] = "changed"
    with pytest.raises(ValueError, match="comparator_drift"):
        c.check_fit(rows, bad)


def test_changed_original_bytes(tmp_path):
    """SCENARIO-REPORT-8023-FIT: annotation bytes cannot silently change labels."""
    plan = source_fixture(tmp_path)
    reduced = c.reduce(
        plan["panel"],
        c.capture(plan["panel"], Runtime(), tmp_path / "passes", deadline=float("inf")),
    )
    ref = plan["target_references"]["fit"]["evaluator"]
    value = json.loads(Path(ref["path"]).read_text())
    value["rows"][0]["response_sha256"] = "changed"
    atomic_json(Path(ref["path"]), value)
    plan["target_references"]["fit"]["evaluator"] = e.reference(Path(ref["path"]))
    with pytest.raises(ValueError, match="original_answer"):
        e.join(plan, reduced)


def test_disqualified_group_still_fits_remaining_roster(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-FIT: failed duplicate groups do not erase other data."""
    plan = source_fixture(tmp_path, 64)
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    monkeypatch.setattr(e, "commands", lambda _: [])
    measured = {}

    def run(root, specs, **kwargs):
        if specs[0].scope == "runtime":
            raw = output.parent / "raw" / output.stem
            rows = c.capture(plan["panel"], Runtime(), raw / "forwards", deadline=float("inf"))
            rows[1]["token_rows"][0]["log_probability"] -= 0.1
            rows[1]["numerator"] += 0.1
            rows[1]["mean_nll"] = rows[1]["numerator"] / rows[1]["denominator"]
            atomic_json(raw / "forwards/pass-001.json", rows[1])
            measured.update(qualification=dict(rows=rows), duration_s=3)
            atomic_json(raw / "capture.json", measured)
        return [dict(name=x.name, scope=x.scope, passed=True, exit_code=0) for x in specs]

    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(e, "publish_primary", lambda out, val, *args: atomic_json(out, val) or {})
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kwargs: dict(passed=True))
    assert e.main(["--output", str(output), "--root", str(tmp_path)]) == 0
    value = json.loads(output.read_text())
    assert len(value["fitted_heads"]) == 6
    assert (
        value["verdict_class"] == "disqualified"
        and value["likelihood_calibration_ready_score"] == 0
    )
    assert value["role_support"]["fit"]["independent"] == 63


def test_reduction_reuses_only_bound_capture(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8023-PUBLISH: reporting correction never adds model calls."""
    plan = source_fixture(tmp_path)
    output = tmp_path / "results" / (e.NAME + ".json")
    raw = output.parent / "raw" / output.stem
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    monkeypatch.setattr(e, "commands", lambda _: [])
    monkeypatch.setattr(e, "publish_primary", lambda out, val, *args: atomic_json(out, val) or {})
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kw: dict(passed=True))
    calls = []

    def run(root, specs, **kw):
        calls.extend(s.name for s in specs)
        if specs[0].scope == "runtime":
            rows = c.capture(plan["panel"], Runtime(), raw / "forwards", deadline=float("inf"))
            atomic_json(raw / "capture.json", dict(qualification=dict(rows=rows), duration_s=3))
        return [dict(name=s.name, scope=s.scope, passed=True, exit_code=0) for s in specs]

    monkeypatch.setattr(e, "run_commands", run)
    args = ["--output", str(output), "--root", str(tmp_path)]
    assert e.main(args) == 0
    saved = json.loads((raw / "validation_commands.json").read_text())
    changed = copy.deepcopy(plan)
    changed["gguf_sha256"] = "changed"
    monkeypatch.setattr(e, "authenticate", lambda _: changed)
    assert e.main(args + ["--reduce-existing"]) == 1
    monkeypatch.setattr(e, "authenticate", lambda _: plan)
    calls.clear()
    assert e.main(args + ["--reduce-existing"]) == 0
    assert calls == ["cold_reduction"]
    assert list((raw / "prior_reporting").glob("*/capture.json"))
    assert json.loads((raw / "validation_commands.json").read_text())["methods"] == saved["methods"]
