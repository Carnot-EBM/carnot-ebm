"""REQ-REPORT-8843: causal streams retain actual learning and untouched retention."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import causal_streams_8843 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_manifest_and_worlds() -> None:
    """SCENARIO-REPORT-8843-CONTRACT: freeze independent streams before labels."""
    frozen = r.manifest()
    assert len(frozen) == 24
    assert [s["seed"] for s in frozen] == [11, 23, 37, 47, 59] * 4 + [11, 23, 37, 47]
    assert len({s["rng_seed"] for s in frozen}) == 24
    assert all(len(s["online"]) == 48 and len(s["retention"]) == 12 for s in frozen)
    rows = r.census(frozen, "unexecuted")
    assert len(rows) == 12096
    assert len({(x["cluster_id"], x["arm"], x["unit_id"]) for x in rows}) == 12096
    worlds = [w for s in frozen for k in ("online", "retention") for w in s[k]]
    assert len({w["structure_id"] for w in worlds}) == 1440
    for recipe in worlds[:60]:
        payload = r.payload(recipe)
        assert len({t for q in payload["query"] for t in (q[0], q[2])}) == 6
        assert r.c.m.decide(payload)["decision"] in r.c.m.LABELS
    assert r.manifest() == frozen


def test_stream_mechanism(tmp_path: Path) -> None:
    """REQ-REPORT-8843: all arms restart, drain, seal, and preserve retention."""
    heads = {(11, o): r.c.head(11, o) for o in ("categorical", "linear")}
    stream = r.manifest()[0]
    work = r.run_stream(tmp_path, stream, heads)
    assert len(work["per_event_results"]) == 288
    assert len(work["retention_rows"]) == 216
    assert len(work["label_release_rows"]) == 288
    assert len(work["gradient_rows"]) == 192
    assert all(
        x["feedback_ordinal"] == x["prediction_ordinal"] + 8 for x in work["label_release_rows"]
    )
    assert all(x["feedback_label"] in x["released_prefix"] for x in work["label_release_rows"])
    assert any(x["feedback_label"] != x["verified_label"] for x in work["label_release_rows"])
    assert all(x["gradients"] and x["parameter_delta"] > 0 for x in work["gradient_rows"])
    assert all(x["unchanged"] for x in work["restart_rows"])
    for ref in work["final_state_refs"]:
        state = r.old.bound(ref)
        assert not state["pending"] and len(state["acks"]) == 48
    assert any(x["template_uses"] for x in work["template_use_rows"])
    assert all(x["learning_state_unchanged"] for x in work["retention_rows"])


def test_release_guards(tmp_path: Path) -> None:
    """REQ-REPORT-8843: active, early, changed, and duplicate releases cannot commit."""
    store = r.c.Store(tmp_path / "store.json", r.c.head(11, "categorical"))
    payload = r.payload(r.manifest()[0]["online"][0])
    store.predict("q1", payload, 0.5)
    before = canonical_hash(store.state)
    with pytest.raises(ValueError, match="feedback_boundary"):
        r.release(store, "q1", payload, 9, "categorical_combined")
    store.close()
    for event, value, ordinal in [
        ("q1", payload, 8),
        ("q1", r.c.m.world("bad", "support"), 9),
        ("q2", payload, 9),
    ]:
        with pytest.raises(ValueError, match="feedback_boundary"):
            r.release(store, event, value, ordinal, "categorical_combined")
        assert canonical_hash(store.state) == before
    row, gradient = r.release(store, "q1", payload, 9, "categorical_combined")
    assert row["template_added"] and gradient["gradients"]
    with pytest.raises(ValueError, match="feedback_boundary"):
        r.release(store, "q1", payload, 10, "categorical_combined")


def test_projection_and_integrity(tmp_path: Path) -> None:
    """REQ-REPORT-8843: pilots project the entire cohort and do not select gains."""
    assert r.projection([1.0, 2.0], 10.0)["passed"]
    assert not r.projection([100.0, 200.0], 10.0)["passed"]
    assert r.projection([1.0, 2.0], 10.0)["registered_streams"] == 24
    recipe = r.manifest()[0]
    heads = {(11, o): r.c.head(11, o) for o in ("categorical", "linear")}
    work = r.run_stream(tmp_path, recipe, heads)
    assert r.integrity(work, [recipe])["passed"]
    bad = deepcopy(work)
    bad["per_event_results"][0]["metric"] = -1
    with pytest.raises(ValueError, match="row_integrity"):
        r.integrity(bad, [recipe])
    bad = deepcopy(work)
    bad["label_release_rows"][0]["feedback_ordinal"] = 0
    with pytest.raises(ValueError, match="release_integrity"):
        r.integrity(bad, [recipe])


def test_preflight_and_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8843: authenticate original authorities and reconstruct every admitted fit."""
    command = (
        "from pathlib import Path; from carnot.reporting import causal_streams_8843 as r; "
        f"e,p=r.preflight(Path({str(tmp_path / 'real')!r})); "
        "assert p['feedback_delay']==8 and len(e['fits'])==45; "
        "assert p['fresh_query_structure'] not in p['exposed_structure_hashes']; "
        "print('original_authorities_authenticated',flush=True)"
    )
    child = subprocess.run(
        [r.sys.executable, "-u", "-c", command], capture_output=True, text=True, timeout=90
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "original_authorities_authenticated" in child.stdout
    evidence = dict(
        development_rows=[dict(split="fit"), dict(split="validation")],
        fits=[
            dict(seed=s, objective=o, lr=0.1, parameters=[[0.0]])
            for s in r.c.d.SEEDS
            for o in ("categorical", "linear")
        ],
    )
    original = r.c.d.fit
    monkeypatch.setattr(
        r.c.d,
        "fit",
        lambda a, b, seed, objective, rate: next(
            x
            for x in evidence["fits"]
            if x["seed"] == seed and x["objective"] == objective and x["lr"] == rate
        ),
    )
    heads, rows = r.initialize(evidence)
    assert len(heads) == len(rows) == 10
    assert all(row["updates"] == 400 for row in rows)
    monkeypatch.setattr(r.c.d, "fit", lambda *a: {})
    with pytest.raises(ValueError, match="original_fit_drift"):
        r.initialize(evidence)
    monkeypatch.setattr(r.c.d, "fit", original)
    with patch.object(r.sys, "prefix", "/tmp/not-the-venv"):
        with pytest.raises(RuntimeError, match="internal_resources"):
            r.preflight(tmp_path / "resource")
    authority = json.loads(r.AUTHORITY.read_bytes())
    authority["memory_execution_ready_score"] = 0
    fixture = tmp_path / "authority.json"
    atomic_json(fixture, authority)
    with patch.object(r, "AUTHORITY_PIN", r.sha256_file(fixture)):
        with pytest.raises(PermissionError, match="observed_zero"):
            r.preflight(tmp_path / "zero", fixture)


def test_execute_pilots_checkpoint_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8843: two separate timing pilots precede a complete registered cohort."""
    frozen = r.manifest()
    monkeypatch.setattr(r, "initialize", lambda e: ({}, []))
    monkeypatch.setattr(r, "run_stream", lambda *a: {k: [] for k in r.COLLECTIONS})
    monkeypatch.setattr(r, "integrity", lambda *a: dict(passed=True))
    work: dict[str, Any] = {k: [] for k in r.COLLECTIONS}
    work.update(stream_manifest=frozen, stream_checkpoint_refs=[])
    r.execute(tmp_path / "complete", {}, work, r.time.monotonic())
    assert len(work["stream_checkpoint_refs"]) == 24
    assert len(work["resource_projection"]["pilot_seconds"]) == 2
    assert (tmp_path / "complete/execution_checkpoint.json").is_file()
    with pytest.raises(TimeoutError, match="registered_execution_deadline"):
        r.execute(tmp_path / "timeout", {}, work, r.time.monotonic() - 4000)
    monkeypatch.setattr(r, "projection", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="projected_budget"):
        r.execute(tmp_path / "projection", {}, work, r.time.monotonic())


def test_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8843: owned coverage and terminal scans use the shared bounded runner."""
    with patch.object(r.validation, "validate", return_value=dict(passed=True)):
        assert r.validate(tmp_path)["passed"]
    with patch.object(r.validation, "terminal", return_value=dict(passed=True)):
        assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_main_verdicts_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8843-CONTRACT: classify external absence separately from owned failures."""
    monkeypatch.setattr(
        r,
        "validate",
        lambda raw: dict(passed=True, receipts=[], coverage=dict(complete=True, percent=100)),
    )
    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=True, receipts=[]))
    output = tmp_path / (r.NAME + ".json")
    base = ["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]
    assert r.main(["--date", "20000101"]) == 1
    for exception, verdict in [
        (FileNotFoundError("missing"), "blocked"),
        (KeyError("field"), "blocked"),
        (PermissionError("rejected"), "blocked"),
        (RuntimeError("internal"), "disqualified"),
        (TimeoutError("unfinished"), "partial"),
    ]:

        def fail(*args: Any) -> Any:
            raise exception

        monkeypatch.setattr(r, "preflight", fail)
        assert r.main(base) == 0
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == verdict and value["stream_rows_ready_score"] == 0
        assert value["censored_count"] == 12096
        if verdict == "blocked":
            assert r.replay(output)["passed"]
    monkeypatch.setattr(r, "preflight", lambda *a: ({}, {}))
    monkeypatch.setattr(r, "execute", lambda *a: None)
    monkeypatch.setattr(r, "validate", lambda raw: dict(passed=False, receipts=[], coverage={}))
    assert r.main(base) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=False, receipts=[]))
    assert r.main(base) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert r.main(["--replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(r, "replay", lambda *a: dict(passed=True))
    assert r.main(["--replay", str(output)]) == 0


def test_private_cold_cli(tmp_path: Path) -> None:
    """REQ-REPORT-8843: the direct entrypoint works outside the checkout without PYTHONPATH."""
    env = dict(r.os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI)]
    for args, expected in [
        (["--date", "20000101"], "date_rejected"),
        (["--replay", str(tmp_path / "missing.json")], "replay_rejected"),
    ]:
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert child.returncode == 1 and expected in child.stdout


def test_replay_recomputes_causal_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8843: repaired outer hashes cannot hide numerical or aggregate drift."""
    stream = r.manifest()[0]
    heads = {(11, o): r.c.head(11, o) for o in ("categorical", "linear")}
    work = r.run_stream(tmp_path / "original", stream, heads)
    shard = tmp_path / "stream.json"
    atomic_json(shard, work)
    work.update(
        stream_manifest=[stream],
        initialization_rows=[],
        resource_projection={},
        stream_checkpoint_refs=[r.old.reference(shard)],
    )
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, work)
    log = tmp_path / "validation.log"
    log.write_text("real fixture validation operand\n")
    value = dict(
        work,
        rows=r.result_rows(work, "partial"),
        verdict_class="partial",
        intended_count=12096,
        completed_count=504,
        censored_count=11592,
        stream_rows_ready_score=0,
        required_checks_passed=False,
        code_config_hashes=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[r.old.reference(evidence)],
        validation_receipts=[dict(log_path=str(log), log_sha256=r.sha256_file(log))],
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(r, "manifest", lambda *a: [stream])
    monkeypatch.setattr(r, "preflight", lambda *a: ({}, {}))
    monkeypatch.setattr(r, "initialize", lambda *a: (heads, []))
    actual = r.run_stream
    monkeypatch.setattr(r, "run_stream", lambda *a: deepcopy(r.old.bound(r.old.reference(shard))))
    assert r.replay(candidate)["replay_passed"]
    for field, replacement, reason in [
        ("rows", [], "reduction_drift"),
        ("completed_count", 0, "count_drift"),
        ("stream_manifest", [], "evidence_drift"),
    ]:
        altered = dict(value, **{field: replacement})
        atomic_json(candidate, altered)
        with pytest.raises(ValueError, match=reason):
            r.replay(candidate)
    atomic_json(candidate, value)

    def changed(*args: Any) -> Any:
        result = deepcopy(r.old.bound(r.old.reference(shard)))
        result["gradient_rows"][0]["loss"] += 1
        return result

    monkeypatch.setattr(r, "run_stream", changed)
    with pytest.raises(ValueError, match="stream_drift"):
        r.replay(candidate)
    monkeypatch.setattr(r, "run_stream", lambda *a: deepcopy(r.old.bound(r.old.reference(shard))))
    monkeypatch.setattr(r, "initialize", lambda *a: (heads, [dict(changed=True)]))
    with pytest.raises(ValueError, match="initialization_drift"):
        r.replay(candidate)
    monkeypatch.setattr(r, "initialize", lambda *a: (heads, []))
    monkeypatch.setattr(r, "run_stream", actual)
    assert r.replay(candidate)["passed"]
