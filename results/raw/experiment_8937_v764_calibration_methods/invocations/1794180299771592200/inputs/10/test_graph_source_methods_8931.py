"""REQ-REPORT-8931; SCENARIO-REPORT-8931-CONTRACT: freeze before collection."""

from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_8931 as s
from carnot.reporting import graph_source_methods_8931 as m


@pytest.fixture(scope="module")
def protocol():
    """Use the actual fixed roster; fixtures never count as generated answers."""
    return s.graph_protocol(pilot.graph_manifest())


def response(answer="AA", logprob=-0.2):
    """A server-shaped fixture keeps token evidence independent of graph truth."""
    return dict(
        choices=[
            dict(
                message=dict(content=answer),
                logprobs=dict(
                    content=[
                        dict(
                            token=answer,
                            logprob=logprob,
                            top_logprobs=[
                                dict(token=answer, logprob=-0.2),
                                dict(token="other", logprob=-1.2),
                            ],
                        )
                    ]
                ),
            )
        ],
        usage=dict(completion_tokens=1),
    )


def test_shortest_paths_against_matrix():
    """REQ-REPORT-8931: walks through triangles and disconnected nodes are not distances."""
    nodes = ["AA", "BB", "CC", "DD", "EE"]
    edges = [["AA", "BB"], ["BB", "CC"], ["AA", "CC"], ["CC", "DD"]]
    assert s.distance_nodes(nodes, edges, "AA", 2) == ["DD"]
    assert s.distance_nodes(nodes, edges, "AA", 3) == []
    assert s.distance_nodes(nodes, edges, "EE", 2) == []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        matrix = np.triu((rng.random((8, 8)) < 0.3).astype(int), 1)
        matrix += matrix.T
        names = ["A" + chr(65 + i) for i in range(8)]
        pairs = [[names[i], names[j]] for i in range(8) for j in range(i + 1, 8) if matrix[i, j]]
        for i, start in enumerate(names):
            reachable = np.eye(8, dtype=int)
            for k in (1, 2, 3):
                walks = np.linalg.matrix_power(matrix, k)
                expected = [names[j] for j in range(8) if walks[i, j] and not reachable[i, j]]
                assert s.distance_nodes(names, pairs, start, k) == expected
                reachable |= walks > 0


def test_fixed_roles_hashes_and_rejections(protocol, monkeypatch):
    """REQ-REPORT-8931: all 370 identities and 420 queries precede answers and splits."""
    assert protocol == s.graph_protocol(pilot.graph_manifest())
    graphs = protocol["graph_manifest"]
    assert (
        len(graphs)
        == len({g["seed"] for g in graphs})
        == len({g["graph_hash"] for g in graphs})
        == 370
    )
    assert not {g["seed"] for g in graphs} & {g["seed"] for g in pilot.graph_manifest()}
    assert len(s.scheduled(protocol, "control")) == 100
    assert Counter(r["role"] for r in protocol["split_manifest"]) == dict(
        train=160, calibration=80, test=80
    )
    for shard in ("A", "B"):
        units = s.scheduled(protocol, shard)
        assert len(units) == 160
        assert Counter(u["role"] for u in units) == dict(train=80, calibration=40, test=40)
    for g in graphs:
        assert 50 <= len(g["nodes"]) <= 60
        assert g["edge_probability"] == 3 / (len(g["nodes"]) - 1)
        assert all(len(n) == 2 and n.isupper() for n in g["nodes"])
        assert g["graph_hash"] == s.graph_hash(g)
        for q in g["questions"]:
            assert q["expected"] == s.distance_nodes(g["nodes"], g["edges"], g["start"], q["k"])
            assert q["expected"]
    original = s.candidate
    calls = []

    def reject_once(seed):
        calls.append(seed)
        return (
            dict(nodes=["AA"], edges=[], start="AA", edge_probability=0)
            if len(calls) == 1
            else original(seed)
        )

    monkeypatch.setattr(s, "candidate", reject_once)
    rejected = s.graph_protocol(pilot.graph_manifest())
    assert rejected["graph_rejections"][0]["reason"] == "empty_requested_distance"
    assert rejected["graph_rejections"][0]["seed"] == calls[0]
    monkeypatch.setattr(s, "candidate", lambda seed: original(123))
    old = [dict(original(123), seed=123)]
    with pytest.raises(ValueError, match="graph_rejection_budget"):
        s.graph_protocol(old)


@pytest.mark.parametrize(
    "answer,category",
    [
        (" AA.\n", "correct"),
        ("BB!", "valid_wrong_distance"),
        ("ZZ", "out_of_graph"),
        ("AA BB", "malformed"),
        ("node AA", "malformed"),
        ("aa", "malformed"),
        ("", "malformed"),
    ],
)
def test_categories(answer, category):
    """REQ-REPORT-8931: strict single-node parsing never selects among guesses."""
    assert s.category(answer, ["AA", "BB"], ["AA"]) == category


def test_signal_view_and_absence():
    """REQ-REPORT-8931: only two finite server-derived scalars reach learned features."""
    raw = response("AA")
    got = s.signals(raw)
    assert got["features"] == dict(mean_token_nll=0.2, first_answer_top2_margin=1.0)
    assert got["token_count"] == 1
    assert set(got["features"]) == set(s.FEATURE_SCHEMA["learned_features"])
    tokens = raw["choices"][0]["logprobs"]["content"]
    tokens.insert(0, dict(token=" ", logprob=-0.4, top_logprobs=[]))
    raw["choices"][0]["message"]["content"] = " AA"
    assert s.signals(raw)["features"]["mean_token_nll"] == pytest.approx(0.3)
    cases = []
    for probability in [float("nan"), float("inf"), 0.1, True, "-1"]:
        cases.append(response(logprob=probability))
    absent = response()
    absent["choices"][0].pop("logprobs")
    cases.append(absent)
    for token_change in [
        [],
        [dict(token="AA", logprob=-0.2, top_logprobs=[])],
        [dict(token="XX", logprob=-0.2, top_logprobs=[])],
    ]:
        altered = response()
        altered["choices"][0]["logprobs"]["content"] = token_change
        cases.append(altered)
    for answer in ["", " "]:
        cases.append(response(answer))
    duplicate = response()
    duplicate["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = "AA"
    cases.append(duplicate)
    for raw in cases:
        with pytest.raises(ValueError):
            s.signals(raw)


def test_raw_before_labels_and_resume(protocol, tmp_path):
    """SCENARIO-REPORT-8931-CONTRACT: first raw bytes survive crashes and exact resume."""
    units = s.scheduled(protocol, "A")[:2]
    model = dict(hf_id=s.SETTINGS["model"], sha256=s.canonical_hash("fixture"))
    calls = []

    def transport(payload, timeout):
        calls.append(payload)
        assert payload["logprobs"] is True and payload["top_logprobs"] == 2
        assert timeout <= 30
        assert (tmp_path / units[len(calls) - 1]["unit_id"] / "intent.json").is_file()
        return response(units[len(calls) - 1]["expected"][0])

    rows = s.collect(units, transport, tmp_path, model)
    assert len(calls) == 2 and all(r["status"] == "completed" and r["value"] == 0 for r in rows)
    assert rows == s.collect(
        units, lambda *_: pytest.fail("completed unit regenerated"), tmp_path, model
    )
    for unit in units:
        folder = tmp_path / unit["unit_id"]
        assert (folder / "response.json").stat().st_mtime_ns <= (
            folder / "label.json"
        ).stat().st_mtime_ns
        assert set(json.loads((folder / "detector.json").read_text())) == set(
            s.FEATURE_SCHEMA["learned_features"]
        )
    for changed in [dict(model, sha256="changed"), dict(model, hf_id="changed")]:
        with pytest.raises(ValueError, match="binding"):
            s.collect(units, transport, tmp_path, changed)


def test_censored_missing_signals_and_interrupted_intents(protocol, tmp_path):
    """REQ-REPORT-8931: runtime failure is censored; absent logprobs cannot become truth features."""
    units = s.scheduled(protocol, "B")[:3]
    model = dict(hf_id=s.SETTINGS["model"], sha256=s.canonical_hash("fixture"))

    def transport(payload, timeout):
        if payload["messages"][0]["content"] == units[0]["prompt"]:
            raise TimeoutError("transport_timeout")
        raw = response(units[1]["start"])
        raw["choices"][0].pop("logprobs")
        return raw

    rows = s.collect(units, transport, tmp_path, model)
    assert rows[0]["status"] == "censored" and rows[0]["value"] is None
    assert rows[1]["status"] == "completed" and rows[1]["value"] == 1
    assert rows[1]["signal_status"] == "blocked" and rows[1]["features"] is None
    assert rows == s.collect(
        units, lambda *_: pytest.fail("terminal failure retried"), tmp_path, model
    )
    folder = tmp_path / units[2]["unit_id"]
    (folder / "response.json").unlink()
    (folder / "label.json").unlink()
    resumed = s.collect(units, lambda *_: pytest.fail("interrupted unit retried"), tmp_path, model)
    assert resumed[2]["missing_reason"] == "interrupted_after_intent"
    empty = s.collect(units, transport, tmp_path / "budget", model, budget_s=0)
    assert all(r["missing_reason"] == "shard_budget" and r["value"] is None for r in empty)
    bad = s.collect(units[:1], lambda *_: dict(choices=[]), tmp_path / "malformed", model)
    assert bad[0]["status"] == "censored"
    unit = deepcopy(units[0])
    unit["expected"] = [unit["start"]]
    with pytest.raises(ValueError, match="label_truth"):
        s.label_response(unit, response("AA"))
    response_path = tmp_path / units[1]["unit_id"] / "response.json"
    raw = json.loads(response_path.read_text())
    raw["binding"] = "changed"
    s.atomic_json(response_path, raw)
    with pytest.raises(ValueError, match="binding"):
        s.collect(units, transport, tmp_path, model)


def test_pair_acceptance_and_uncertainty(protocol):
    """REQ-REPORT-8931: empirical depth inequality is a control, not statistical proof."""
    units = s.scheduled(protocol, "control")
    rows = [
        s.label_response(u, response(u["expected"][0] if u["k"] == 2 else u["start"]))
        for u in units
    ]
    got = s.paired_control(units, rows)
    assert got["accepted"] and got["independent_count"] == 50
    assert got["error_k2"] == 0 and got["error_k3"] == 1
    assert got["paired_interval"]["mean_difference"] == 1
    assert got["inequality_is_statistical_proof"] is False
    equal = [s.label_response(u, response(u["start"])) for u in units]
    assert not s.paired_control(units, equal)["accepted"]
    assert not s.paired_control(units, rows[:-1])["accepted"]
    tampered = deepcopy(rows)
    tampered[0]["value"] = 1
    assert not s.paired_control(units, tampered)["labels_verified"]
    missing = deepcopy(rows)
    missing[0]["features"] = None
    assert not s.paired_control(units, missing)["usable_signals"]
    with pytest.raises(ValueError, match="roster"):
        s.paired_control(units, rows + [rows[0]])


def input_tree(root):
    """Copy actual producer bytes; the private tree never invents source evidence."""
    for name in [*m.INPUT_PINS, m.branch.DESIGN, m.branch.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """A fixture receipt is intentionally insufficient for research readiness."""
    return dict(
        receipts=[dict(name="fixture", passed=passed)],
        coverage=dict(percent_covered=100, excluded_lines=0),
    )


def test_authentication_absence_fields_and_pins(tmp_path, monkeypatch):
    """REQ-REPORT-8931: absence, rejected identity and observed zero stay distinct."""
    root = input_tree(tmp_path / "inputs")
    assert not m.authenticate(root)["failures"]
    missing = m.authenticate(tmp_path / "absent")
    assert missing["failures"] and missing["failures"][0]["observed"] == "missing_file"
    path = root / m.PILOT
    original = json.loads(path.read_text())
    for field, changed, observed in [
        ("pilot_ready_score", None, "missing_field"),
        ("pilot_ready_score", 0, "observed_zero"),
        ("pilot_ready_score", True, "rejected_producer"),
        ("experiment_id", 1, "rejected_producer"),
    ]:
        value = deepcopy(original)
        if changed is None:
            del value[field]
        else:
            value[field] = changed
        s.atomic_json(path, value)
        monkeypatch.setitem(m.INPUT_PINS, m.PILOT, pilot.sha256_file(path))
        assert observed in [r["observed"] for r in m.authenticate(root)["failures"]]
    path.write_text("malformed")
    assert m.authenticate(root)["failures"]
    monkeypatch.setitem(m.INPUT_PINS, m.PILOT, pilot.sha256_file(path))
    assert m.authenticate(root)["failures"]


def test_build_replay_and_terminal_classes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8931-CONTRACT: administrative readiness never claims a model result."""
    work = m.freeze(input_tree(tmp_path / "inputs"), tmp_path / "raw")
    checks = validation()
    value = m.build(work, checks, fixture=True)
    assert value["source_methods_ready_score"] == 0 and value["fixture_only"]
    assert value["rows"] == [] and value["independent_count"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    path = tmp_path / "candidate.json"
    s.atomic_json(path, value)
    assert m.replay(path)["passed"]
    monkeypatch.setattr(m, "qualified", lambda *_: True)
    positive_methods = m.build(work, checks)
    assert (
        positive_methods["verdict_class"] == "null"
        and positive_methods["source_methods_ready_score"] == 1
    )
    monkeypatch.setattr(m, "qualified", lambda *_: False)
    assert m.build(work, checks)["verdict_class"] == "disqualified"
    blocked = deepcopy(work)
    blocked["authentication"]["failures"] = [dict(check="absent", observed="missing_file")]
    assert m.build(blocked, checks, fixture=True)["verdict_class"] == "blocked"
    value["source_methods_ready_score"] = 1
    value["reproducibility_checksum"] = pilot.checksum(value)
    s.atomic_json(path, value)
    assert not m.replay(path)["passed"]
    s.atomic_json(path, m.build(work, checks, fixture=True))
    source = Path(work["sources"][0]["path"])
    source.write_bytes(b"changed original producer")
    assert not m.replay(path)["passed"]
    assert not m.replay(tmp_path / "absent")["passed"]


def test_private_actual_cli_cold_replay(tmp_path):
    """SCENARIO-REPORT-8931-CONTRACT: the real child works outside checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    target = tmp_path / (m.NAME + ".json")

    def invoke(*args):
        return subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    result = invoke("--date", "20261108", "--fixture-e2e", str(target))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(target.read_text())
    assert value["fixture_only"] and value["source_methods_ready_score"] == 0
    assert invoke("--cold-replay", str(target)).returncode == 0
    value["split_manifest"][0]["role"] = "test"
    value["reproducibility_checksum"] = pilot.checksum(value)
    s.atomic_json(target, value)
    assert invoke("--cold-replay", str(target)).returncode == 1
    assert invoke("--date", "bad", "--fixture-e2e", str(target)).returncode != 0


def test_plan_qualification_and_production_main(tmp_path, monkeypatch):
    """REQ-REPORT-8931: owned checks and unchanged terminal validators govern publication."""
    plan = m.validation_plan(tmp_path / "plan")
    assert {
        "owned_unit_child",
        "coverage_combine",
        "coverage_json",
        "strict_mypy",
        "e2e018",
        "e2e021",
    } <= {c.name for c in plan}
    config = (tmp_path / "plan" / "coverage.ini").read_text()
    assert all(name in config for name in m.OWNED) and "exclude_lines =\n" in config
    assert not m.qualified(validation())
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(
        m,
        "freeze",
        lambda *a: dict(
            raw=str(tmp_path / "raw"),
            started_ns=time.monotonic_ns(),
            phase_spans=[dict(duration_s=0)],
        ),
    )
    (tmp_path / "raw").mkdir()
    work = m.freeze()
    value = dict(
        experiment_id=8931,
        task_id=m.TASK,
        honest_verdict="complete_null_methods",
        verdict_class="null",
        completed_count=0,
        censored_count=0,
        source_methods_ready_score=1,
        required_checks_passed=True,
        acceptance_gates=dict(owned_checks=True),
        validation_receipts=[],
        raw_shard_hashes=[],
    )
    monkeypatch.setattr(m, "build", lambda *a, **kw: deepcopy(value))
    monkeypatch.setattr(m, "validation_plan", lambda *a: [])
    monkeypatch.setattr(m, "replay", lambda *a: dict(passed=True))

    def fake_commands(root, commands, **kwargs):
        return [dict(name=c.name, passed=True, exit_code=0) for c in commands]

    monkeypatch.setattr(m, "run_commands", fake_commands)
    coverage = tmp_path / "qualification"
    coverage.mkdir()
    s.atomic_json(
        coverage / "coverage.json", dict(totals=dict(percent_covered=100, excluded_lines=0))
    )
    monkeypatch.setattr(m.tempfile, "mkdtemp", lambda **kw: str(coverage))
    (coverage / ".coverage").write_bytes(b"fixture, never a qualification database")
    monkeypatch.setattr(m, "measured", lambda *a: dict(complete=False))
    callbacks = []

    def publish(output, artifact, validator):
        callbacks.append(validator(tmp_path / "candidate.json"))
        return dict(primary_path=str(output))

    monkeypatch.setattr(m, "publish_primary", publish)
    assert m.main(["--date", "20261108"]) == 0
    assert callbacks[-1]["passed"] and len(callbacks[-1]["receipts"]) == 3
    monkeypatch.setattr(m, "terminal_passed", lambda *_: False)

    def rejected(output, artifact, validator):
        report = validator(tmp_path / "candidate.json")
        if not report["passed"]:
            raise ValueError("candidate_rejected")
        assert (
            artifact["verdict_class"] == "disqualified"
            and artifact["source_methods_ready_score"] == 0
        )
        return dict(primary_path=str(output))

    monkeypatch.setattr(m, "publish_primary", rejected)
    assert m.main(["--date", "20261108"]) == 0
    monkeypatch.setattr(m, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("other")))
    with pytest.raises(ValueError, match="other"):
        m.main(["--date", "20261108"])
    assert work["raw"] == str(tmp_path / "raw")


def test_resume_rejects_corrupt_labels_and_invalid_units(protocol, tmp_path, monkeypatch):
    """REQ-REPORT-8931: resume binds all inputs and refuses to repair a conflicting label."""
    unit = s.scheduled(protocol, "A")[0]
    model = dict(hf_id=s.SETTINGS["model"], sha256=s.canonical_hash("fixture"))
    for roster in [[unit, unit], [dict(unit, unit_id="../escape")]]:
        with pytest.raises(ValueError, match="unit_roster"):
            s.collect(roster, lambda *_: response(), tmp_path, model)
    rows = s.collect([unit], lambda *_: response(unit["expected"][0]), tmp_path, model)
    for changed in [
        dict(unit, role="test" if unit["role"] != "test" else "train"),
        dict(unit, prompt="changed"),
        dict(unit, edges=[]),
    ]:
        with pytest.raises(ValueError, match="binding"):
            s.collect(
                [changed], lambda *_: pytest.fail("changed input regenerated"), tmp_path, model
            )
    with monkeypatch.context() as patch:
        patch.setitem(s.SETTINGS, "max_tokens", 15)
        with pytest.raises(ValueError, match="binding"):
            s.collect([unit], lambda *_: response(), tmp_path, model)
    with monkeypatch.context() as patch:
        patch.setitem(s.FEATURE_SCHEMA, "version", "changed")
        with pytest.raises(ValueError, match="binding"):
            s.collect([unit], lambda *_: response(), tmp_path, model)
    label = tmp_path / unit["unit_id"] / "label.json"
    s.atomic_json(label, dict(rows[0], value=1))
    with pytest.raises(ValueError, match="label_checkpoint_drift"):
        s.collect([unit], lambda *_: pytest.fail("corruption regenerated"), tmp_path, model)
    invalid = deepcopy(unit)
    invalid["expected"] = [unit["start"]]
    with pytest.raises(ValueError, match="label_truth"):
        s.collect([invalid], lambda *_: response(), tmp_path / "truth", model)
    malformed = response()
    malformed["choices"][0]["message"]["content"] = 4
    assert s.collect([unit], lambda *_: malformed, tmp_path / "content", model)[0]["value"] is None
    failed = tmp_path / "failure"
    s.collect(
        [unit], lambda *_: (_ for _ in ()).throw(TimeoutError("fixture_timeout")), failed, model
    )
    failure_path = failed / unit["unit_id"] / "failure.json"
    stored = json.loads(failure_path.read_text())
    stored["row"]["status"] = "completed"
    s.atomic_json(failure_path, stored)
    with pytest.raises(ValueError, match="failure_binding"):
        s.collect([unit], lambda *_: pytest.fail("failure regenerated"), failed, model)


def test_finite_outputs_model_identity_and_pair_denominators(protocol, tmp_path):
    """REQ-REPORT-8931: extreme probabilities, foreign models and invalid denominators fail closed."""
    raw = response()
    tokens = [
        dict(
            token="A",
            logprob=-1.7e308,
            top_logprobs=[dict(token="A", logprob=-0.2), dict(token="B", logprob=-1)],
        )
    ] * 2
    raw["choices"][0]["logprobs"]["content"] = tokens
    try:
        got = s.signals(raw)
    except ValueError:
        pass
    else:
        assert all(math.isfinite(v) for v in got["features"].values())
    unit = s.scheduled(protocol, "A")[0]
    for model in [{}, dict(hf_id="foreign", sha256=s.canonical_hash("fixture"))]:
        with pytest.raises(ValueError, match="model_binding"):
            s.collect([unit], lambda *_: response(), tmp_path / str(len(model)), model)
    units = s.scheduled(protocol, "control")
    rows = [
        s.label_response(u, response(u["expected"][0] if u["k"] == 2 else u["start"]))
        for u in units
    ]
    for field, value in [("denominator", 0), ("numerator", 1), ("cluster_id", "foreign")]:
        altered = deepcopy(rows)
        altered[0][field] = value
        assert not s.paired_control(units, altered)["accepted"]


def test_rehashed_retained_inputs_and_truth_are_rejected(tmp_path):
    """SCENARIO-REPORT-8931-CONTRACT: rehashed operands cannot redefine authenticated inputs or BFS truth."""
    work = m.freeze(input_tree(tmp_path / "inputs"), tmp_path / "raw")
    checks = validation()
    path = tmp_path / "candidate.json"
    original = deepcopy(work)
    retained = Path(work["raw"]) / "inputs" / m.PILOT
    saved = retained.read_bytes()
    retained.write_text("{}")
    work["input_references"] = [m.reference(Path(r["path"])) for r in work["input_references"]]
    s.atomic_json(Path(work["raw"]) / "work.json", work)
    s.atomic_json(path, m.build(work, checks, fixture=True))
    assert not m.replay(path)["passed"]
    retained.write_bytes(saved)
    work = original
    protocol_path = Path(work["protocol_reference"]["path"])
    changed = json.loads(protocol_path.read_text())
    changed["graph_manifest"][0]["questions"][0]["expected"] = [
        changed["graph_manifest"][0]["start"]
    ]
    s.atomic_json(protocol_path, changed)
    work["protocol_reference"] = m.reference(protocol_path)
    s.atomic_json(Path(work["raw"]) / "work.json", work)
    s.atomic_json(path, m.build(work, checks, fixture=True))
    assert not m.replay(path)["passed"]
