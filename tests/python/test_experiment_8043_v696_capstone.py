"""REQ-REPORT-8043: reader controls cannot substitute for scientific evidence."""

import copy
import gzip
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import v696_capstone as cap
from carnot.reporting import v696_capstone_reduction as red
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def fixture(root):
    """SCENARIO-REPORT-8043-CUSTODY: freeze a private complete invocation."""
    snapshots = {}
    for role, name in (("active", "active.yaml"), ("design", "design.md")):
        p = root / name
        p.write_bytes(
            gzip.decompress((cap.ROOT / "tests/fixtures/v696" / (name + ".gz")).read_bytes())
        )
        snapshots[role] = dict(exists=True, snapshot_path=str(p), sha256=sha256_file(p))
    import yaml

    tasks = yaml.safe_load((root / "active.yaml").read_bytes())["tasks"]
    atomic_json(
        root / "results/experiment_8031_v696_contract_methods.json",
        dict(
            authority_snapshots=snapshots,
            canonical_tasks_sha256=cap.lifecycle.tasks_digest(tasks),
            method_freeze={},
            experiment_id=8031,
            task_id=tasks[0]["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            flagged_adversarial=False,
        ),
    )
    return tasks


def test_authority_and_tamper(tmp_path):
    """SCENARIO-REPORT-8043-CUSTODY: compare the full tasks, including prompts."""
    tasks = fixture(tmp_path)
    assert cap.authorities(tmp_path)[0] == tasks
    path = tmp_path / "results/experiment_8031_v696_contract_methods.json"
    v = json.loads(path.read_text())
    v["canonical_tasks_sha256"] = "wrong"
    atomic_json(path, v)
    with pytest.raises(ValueError, match="authority"):
        cap.authorities(tmp_path)


def test_margin_family_and_absence():
    """SCENARIO-REPORT-8043-SCIENCE: test the registered margins, not zero."""
    missing = red.bootstrap([], 0.01)
    assert missing["gain"] is None and missing["raw_p"] == 1
    results = red.family(
        [red.bootstrap([0.009] * 80, 0.01), red.bootstrap([0.03] * 80, 0.02), missing],
        [True, True, False],
    )
    assert results[0]["positive_claim"] is False
    assert results[1]["positive_claim"] is True
    assert results[2]["family_p"] == 1
    null = red.family([red.bootstrap([0.0] * 80, m) for m in (0.01, 0.02, 0.02)], [True] * 3)
    assert not any(r["positive_claim"] for r in null)
    blocked = red.family([red.bootstrap([0.5] * 80, m) for m in (0.01, 0.02, 0.02)], [False] * 3)
    assert all(r["family_p"] == 1 for r in blocked)
    assert red.bootstrap([float("nan")], 0.02, 32)["completed_draws"] == 0


def test_absent_science():
    """SCENARIO-REPORT-8043-SCIENCE: absence gives no feature or head estimate."""
    assert red.independent({}, 8037)["measurement_available"] is False
    assert red.independent({}, 8040)["measurement_available"] is False
    assert red.independent({}, 8039)["measurement_available"] is False


def test_terminal_readiness(tmp_path):
    """SCENARIO-REPORT-8043-TERMINAL: blocked science can have a correct reader."""
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261002", tmp_path / "raw")
    assert len(v["rows"]) == 13 and not v["rows"][-1]["completed"]
    assert v["verdict_class"] == "blocked"
    counts = {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED}
    cap.complete(v, [dict(name="owned", passed=True, classification="required")], counts)
    assert v["capstone_execution_ready_score"] == 1
    assert v["completed_count"] == 13 and v["generalized_learning_benefit_score"] == 0
    bad = copy.deepcopy(v)
    cap.complete(bad, [dict(name="owned", passed=False)], counts)
    assert bad["verdict_class"] == "disqualified" and bad["capstone_execution_ready_score"] == 0
    cap.seal(v, tmp_path / "raw")
    assert cap.cold_replay(v) == []
    v["science_ready"] = True
    assert cap.cold_replay(v)


def test_cli_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8043-TERMINAL: exercise the direct entrypoint and error exit."""
    monkeypatch.setattr(sys, "argv", [cap.CLI, "--cold-replay", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(cap.ROOT / cap.CLI), run_name="__main__")
    assert error.value.code == 1
    with pytest.raises(ValueError, match="date"):
        cap.main(["--date", "20261003"])


def test_independent_learning(tmp_path):
    """SCENARIO-REPORT-8043-SCIENCE: replay real private events, targets and heads."""
    from test_learning_benefit_8039 import bundle

    b = bundle(tmp_path)
    r = red.learning.replay_trajectory(Path(b["trajectory"]))
    pred = tmp_path / "retention_seal.json"
    retained = red.learning.retention(b, r, pred)
    compared = red.learning.compare(r["rows"], retained["retention_rows"])
    b["historical_exposure"] = dict(scope="private circular control")
    p = tmp_path / "bundle.json"
    atomic_json(p, b)
    data = dict(
        r,
        **retained,
        **compared,
        audit_bundle=dict(path=str(p), sha256=sha256_file(p)),
        retention_prediction_seal=dict(path=str(pred), sha256=sha256_file(pred)),
    )
    reduced = red.independent(data, 8039)
    assert reduced["measurement_available"] and reduced["numerical_agreement"]["measured"]
    assert reduced["primary"]["eligible_slots"] > 0
    data["rows"][0]["probability"] = 0.99
    with pytest.raises(ValueError, match="producer_drift"):
        red.independent(data, 8039)


def test_collect_stale_code_and_failed_reduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8043-CUSTODY: reject stale producer code and broken primitives."""
    tasks = fixture(tmp_path)
    p = tmp_path / tasks[0]["deliverable"]
    code = tmp_path / "changed.py"
    code.write_text("print('changed')")
    atomic_json(
        p,
        dict(
            code_config_hashes=[
                dict(path=str(code), sha256="wrong"),
                dict(path="doc.md", sha256="old"),
            ]
        ),
    )
    row = dict(task_id=tasks[0]["id"], path=str(p), gate_check_summary=[])
    monkeypatch.setattr(cap.previous, "collect", lambda *a: ([row], [], [], []))
    monkeypatch.setattr(
        red, "independent", lambda *a: (_ for _ in ()).throw(ValueError("bad seal"))
    )
    rows, refs, failed, audits = cap.collect(tmp_path, tasks)
    assert not rows[0]["eligible"] and len(failed) == 2 and refs
    assert audits[0]["measurement_available"] is False
    atomic_json(p, dict(code_config_hashes={str(code): sha256_file(code)}))
    monkeypatch.setattr(red, "independent", lambda *a: dict(measurement_available=False))
    row["gate_check_summary"] = []
    assert cap.collect(tmp_path, tasks)[2] == []


def test_retirement_spelling_and_new_null(tmp_path):
    """SCENARIO-REPORT-8043-DECISIONS: a changed verdict does not erase failed operands."""
    old = tmp_path / "results/prior.json"
    current = tmp_path / "results/current.json"
    prior = dict(
        honest_verdict="complete_disqualified_likelihood_calibration",
        gate_check_summary=[dict(artifact_field="duplicate_parity", passed=False)],
    )
    now = dict(
        honest_verdict="complete_disqualified_scoring_isolation",
        verdict_class="disqualified",
        gate_check_summary=[dict(artifact_field="duplicate_parity", passed=False)],
    )
    atomic_json(old, prior)
    atomic_json(current, now)
    atomic_json(
        tmp_path / "results/experiment_8030_v695_capstone.json",
        dict(
            task_contract=[
                dict(id="exp8023-likelihood-calibration", deliverable="results/prior.json")
            ]
        ),
    )
    task = dict(
        id="exp8033-scoring-isolation",
        prior_failures=[
            dict(
                experiment_id="exp8023-likelihood-calibration",
                verdict=prior["honest_verdict"],
                addressed_by="Fresh scorer contexts; original parity gate.",
                retire_if_same_verdict=True,
            )
        ],
    )
    row = dict(honest_verdict=now["honest_verdict"], path=str(current), sha256=sha256_file(current))
    result = cap.retirements(tmp_path, [task], [row])[0]
    assert result["prior_authenticated"] and result["retire"] and result["mechanism_repeat"]
    now.update(
        honest_verdict="complete_null_new_mechanism", verdict_class="null", gate_check_summary=[]
    )
    atomic_json(current, now)
    row.update(honest_verdict=now["honest_verdict"], sha256=sha256_file(current))
    assert not cap.retirements(tmp_path, [task], [row])[0]["retire"]


def test_main_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8043-TERMINAL: run final-byte consumers on a private blocked reader."""
    fixture(tmp_path)
    frozen = cap.manifest(tmp_path / "scratch")
    assert any(s["name"] == "full_suite" and s["deadline_s"] <= 90 for s in frozen["commands"])
    assert cap.OWNED[1] in frozen["coverage_includes"]
    original = cap.run_check

    def checked(root, spec, private, durable):
        if spec["name"].startswith(("candidate_", "published_")):
            return original(root, spec, private, durable)
        publication = dict(
            paper_ready=True, unmet_gates=[], gates={f"G{i}": dict(pass_=True) for i in range(1, 5)}
        )
        for row in publication["gates"].values():
            row["pass"] = row.pop("pass_")
        log = tmp_path / (spec["name"] + ".json")
        atomic_json(log, publication)
        atomic_json(
            private / "coverage.json",
            dict(
                files={p: dict(summary=dict(num_statements=1, covered_lines=1)) for p in cap.OWNED}
            ),
        )
        return dict(
            spec, passed=True, log_path=str(log), log_sha256=sha256_file(log), actual_exit=0
        )

    monkeypatch.setattr(cap, "run_check", checked)
    monkeypatch.setattr(
        cap,
        "manifest",
        lambda p: dict(
            commands=[dict(name="publication_gate", argv=[], expected_exit=0, deadline_s=30)]
        ),
    )
    assert cap.main(["--root", str(tmp_path)]) == 0
    out = tmp_path / "results/experiment_8043_v696_capstone.json"
    v = json.loads(out.read_text())
    assert v["capstone_execution_ready_score"] == 1 and v["paper_ready"]
    assert cap.main(["--cold-replay", str(out)]) == 0
    v["canonical_tasks_sha256"] = "wrong"
    assert "authority_drift" in cap.cold_replay(v)
    v["task_dispositions"][0]["numerator"] = 999
    assert "independent_reduction_drift" in cap.cold_replay(v)
    v["code_config_hashes"][0]["sha256"] = "wrong"
    assert cap.cold_replay(v) == ["source_bytes_changed"]
    monkeypatch.setattr(
        cap,
        "run_check",
        lambda *a: dict(name="failed", passed=False, log_path=str(tmp_path / "no_log")),
    )
    with pytest.raises(ValueError, match="owned_validation_failed"):
        cap.main(["--root", str(tmp_path)])


def test_publication_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8043-TERMINAL: failed final bytes and consumer drift reject publication."""
    fixture(tmp_path)
    raw = tmp_path / "raw"
    v = cap.build(tmp_path, "20261002", raw)
    out = tmp_path / "results/experiment_8043_v696_capstone.json"
    monkeypatch.setattr(
        cap, "run_check", lambda *a: dict(passed=False, log_path=str(tmp_path / "absent"))
    )
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        cap.publish(v, out, tmp_path / "private", raw)
    monkeypatch.setattr(
        cap, "run_check", lambda *a: dict(passed=True, log_path=str(tmp_path / "absent"))
    )
    monkeypatch.setattr(cap, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="published_reader_drift"):
        cap.publish(v, out, tmp_path / "private", raw)


@pytest.mark.parametrize("number", [8040, 8042])
def test_deployment_raw_reduction(number):
    """SCENARIO-REPORT-8043-DECISIONS: local costs and fallback remain separate from service benefit."""
    p = next((cap.ROOT / "results").glob(f"experiment_{number}_*.json"))
    data = json.loads(p.read_text())
    reduced = red.independent(data, number)
    assert reduced["measurement_available"] and reduced["rows"]
    assert reduced["complete_service_measured"] is False
    data["rows"] = [] if number == 8042 else data["rows"]
    if number == 8042:
        data["rows"] = [dict(tampered=True)]
        with pytest.raises(ValueError, match="primitive"):
            red.independent(data, number)


def test_frozen_code_drift(tmp_path):
    """SCENARIO-REPORT-8043-CUSTODY: matching saved bytes cannot excuse stale live code."""
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261002", tmp_path / "raw")
    cap.seal(v, tmp_path / "raw")
    p = tmp_path / "different.py"
    p.write_text("changed_configuration")
    v["code_config_hashes"][0]["original_path"] = str(p)
    assert cap.cold_replay(v) == ["code_configuration_changed"]
    v["code_config_hashes"] = []
    freeze = next(r for r in v["checkpoint_references"] if r["role"] == "method_freeze")
    value = json.loads(Path(freeze["path"]).read_text())
    value["code_config_hashes"][cap.OWNED[0]] = "wrong"
    atomic_json(Path(freeze["path"]), value)
    freeze["sha256"] = sha256_file(Path(freeze["path"]))
    assert cap.cold_replay(v) == ["code_configuration_changed"]


@pytest.mark.parametrize("mutation", ["table", "digest"])
def test_visible_authority_contract(tmp_path, mutation):
    """SCENARIO-REPORT-8043-CUSTODY: a matching JSON list cannot excuse changed visible authority."""
    fixture(tmp_path)
    p = tmp_path / "results/experiment_8031_v696_contract_methods.json"
    v = json.loads(p.read_text())
    design = Path(v["authority_snapshots"]["design"]["snapshot_path"])
    text = design.read_text()
    if mutation == "table":
        text = text.replace("| 1 |", "| 99 |", 1)
    else:
        text = text.replace(v["canonical_tasks_sha256"], "f" * 64, 1)
    design.write_text(text)
    v["authority_snapshots"]["design"]["sha256"] = sha256_file(design)
    atomic_json(p, v)
    with pytest.raises(ValueError, match="authority"):
        cap.authorities(tmp_path)
