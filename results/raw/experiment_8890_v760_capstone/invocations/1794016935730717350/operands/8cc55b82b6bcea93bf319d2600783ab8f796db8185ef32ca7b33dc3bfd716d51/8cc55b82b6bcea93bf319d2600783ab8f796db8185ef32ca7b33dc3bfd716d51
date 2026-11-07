"""REQ-REPORT-8815 and REQ-VERIFY-8815: finite diagnosis before another stream."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from carnot.reporting import learning_signal_diagnosis_8815 as d


def test_scenario_report_8815_finite_floor():
    """Conflicting labels need a nonzero floor even with perfect optimization."""
    rows = [
        dict(counts=[5, 5, 0, 1, 0, 1], truth=y, structure="a") for y in ["support", "insufficient"]
    ]
    table, floor = d.collisions(rows, "off")
    assert floor == pytest.approx(0.5)
    assert table[0]["conflicting"] and table[0]["distinct_structures"] == 1
    assert sum(table[0]["label_counts"].values()) == 2


def test_scenario_verify_8815_controls():
    """Independent numeric controls check gradients and the unchanged serving path."""
    controls = d.controls()
    assert controls and all(row["passed"] for row in controls)
    assert {
        "finite_difference",
        "serving_parity",
        "mixed_update_parity",
        "separable",
        "conflicting",
        "shuffled",
    } <= {r["check"] for r in controls}


def test_req_verify_8815_development():
    """Canonical query structures, rather than entity names, define the split."""
    rows, split = d.development(set())
    assert len(split["fit"]) == 40 and len(split["validation"]) == 20
    assert set(split["fit"]).isdisjoint(split["validation"])
    assert len({r["structure"] for r in rows}) == 60
    assert all(len(r["input"]["query"]) == 5 for r in rows)
    assert len(d.structure(rows[0]["input"])) > 10
    renamed = d.corpus.rename(rows[0]["input"])
    assert d.structure(renamed) == d.structure(rows[0]["input"])
    again, excluded = d.development({split["fit"][0]})
    assert split["fit"][0] not in excluded["fit"] + excluded["validation"]
    assert again


def test_req_report_8815_selection():
    """Four paired seeds and controls are required; a mean alone cannot qualify."""
    fits = [
        dict(seed=s, objective=o, lr=lr, validation_brier=v, updates=400)
        for s in d.SEEDS
        for o, v in [("mixed", 0.6), ("categorical", 0.58), ("linear", 0.4)]
        for lr in d.LRS
    ]
    selected = d.selection(fits, True, True)
    assert selected["ready"] == 1 and selected["winning_seeds"] == 5
    assert d.selection(fits, False, True)["ready"] == 0
    assert d.selection(fits, True, False)["ready"] == 0
    fits[3]["validation_brier"] = 0.9
    assert d.selection(fits, True, True)["ready"] in (0, 1)


def test_req_report_8815_hash_boundary(tmp_path):
    """Changing bytes cannot be hidden behind a retained reference."""
    p = tmp_path / "operand.json"
    d.atomic_json(p, {"x": 1})
    ref = d.reference(p)
    assert d.load(ref) == {"x": 1}
    p.write_text("{}")
    with pytest.raises(ValueError, match="operand_hash"):
        d.load(ref)
    with pytest.raises(FileNotFoundError):
        d.load(dict(path=str(tmp_path / "absent"), sha256="sha256:x"))


def test_req_verify_8815_fit_curves():
    """Each objective uses the same ordered examples and records actual updates."""
    rows = [dict(counts=[5, 5, 0, i, 0, 1], truth=d.m.LABELS[i], split="fit") for i in range(3)]
    for objective in d.OBJECTIVES:
        fitted = d.fit(rows, rows, 11, objective, 0.03, steps=3)
        assert fitted["updates"] == 3
        assert len(fitted["curve"]) == 4
        assert all(np.isfinite(c["loss"]) for c in fitted["curve"])
        assert fitted["curve"][-1]["gradient_norm"] > 0


def test_scenario_report_8815_replay(tmp_path, monkeypatch):
    """Cold reduction rejects rehashed aggregates rather than trusting the outer JSON."""
    from carnot.reporting import learning_signal_runner_8815 as r

    evidence = dict(
        diagnostic_rows=[
            dict(counts=[5, 5, 0, 1, 0, 1], truth="support", structure="a", template_mode=mode)
            for mode in ("on", "off")
        ],
        development_rows=[],
        split_manifest=dict(fit=["a"], validation=["b"]),
        fits=[
            dict(
                seed=s,
                objective=o,
                lr=lr,
                updates=400,
                validation_brier=0.6,
                curve=[dict(update=400, validation_brier=0.6)],
            )
            for s in d.SEEDS
            for o in d.OBJECTIVES
            for lr in d.LRS
        ],
        controls=[dict(check="fixture", passed=True)],
        source_artifact_hashes=[],
    )
    path = tmp_path / "evidence.json"
    d.atomic_json(path, evidence)
    value = r.reduce(evidence)
    value.update(raw_shard_hashes=[d.reference(path)], code_config_hashes=[])
    candidate = tmp_path / "candidate.json"
    log = tmp_path / "validation.log"
    log.write_text("owned validation")
    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=d.reference(log)["sha256"])]
    value["code_config_hashes"] = [d.reference(Path(d.__file__))]
    d.atomic_json(candidate, value)
    monkeypatch.setattr(r, "authenticate", lambda: (evidence["diagnostic_rows"], [], set()))
    monkeypatch.setattr(d, "development", lambda excluded: ([], evidence["split_manifest"]))
    monkeypatch.setattr(d, "controls", lambda: evidence["controls"])
    monkeypatch.setattr(
        d,
        "fit",
        lambda train, valid, seed, objective, lr: next(
            f
            for f in evidence["fits"]
            if (f["seed"], f["objective"], f["lr"]) == (seed, objective, lr)
        ),
    )
    assert r.replay(candidate)["passed"]
    assert r.main(["--replay", str(candidate)]) == 0
    value["empirical_feature_floor"]["on"] = 0.8
    d.atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)


def test_req_report_8815_runner(tmp_path, monkeypatch):
    """Owned failure disqualifies; unavailable inputs block with explicit operands."""
    from carnot.reporting import learning_signal_runner_8815 as r

    assert r.main(["--date", "bad"]) == 1
    assert r.main(["--replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(r, "execute", lambda raw: (_ for _ in ()).throw(FileNotFoundError("x")))
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=True, receipts=[], coverage=dict(complete=True))
    )
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=True, receipts=[]))
    output = tmp_path / "experiment_8815_v755_learning_signal_diagnosis.json"
    assert r.main(["--date", "20261104", "--output", str(output)]) == 0
    import json

    blocked = json.loads(output.read_bytes())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["status"] == "missing_file"
    assert blocked["diagnosis_rows_ready_score"] == 0
    monkeypatch.setattr(r, "execute", lambda raw: (_ for _ in ()).throw(KeyError("absent")))
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["gate_check_summary"][0]["status"] == "missing_field"
    monkeypatch.setattr(r, "execute", lambda raw: (_ for _ in ()).throw(ValueError("rejected")))
    assert r.main(["--date", "20261104", "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_req_report_8815_success_and_failed_qualification(tmp_path, monkeypatch):
    """Numeric evidence stays available when an owned or terminal check disqualifies it."""
    from carnot.reporting import learning_signal_runner_8815 as r
    import json

    fitted = [
        d.fit(
            [dict(counts=[5, 5, 0, 1, 0, 1], truth="support")],
            [dict(counts=[5, 5, 0, 1, 0, 1], truth="support")],
            seed,
            objective,
            lr,
            steps=1,
        )
        for seed in d.SEEDS
        for objective in d.OBJECTIVES
        for lr in d.LRS
    ]
    evidence = dict(
        diagnostic_rows=[
            dict(counts=[5, 5, 0, 1, 0, 1], truth="support", structure="old", template_mode=mode)
            for mode in ("on", "off")
        ],
        development_rows=[
            dict(
                unit_id="new",
                structure="v0",
                split="validation",
                counts=[5, 5, 0, 1, 0, 1],
                truth="support",
            )
        ],
        split_manifest=dict(
            fit=[f"f{i}" for i in range(40)], validation=[f"v{i}" for i in range(20)]
        ),
        fits=fitted,
        controls=[dict(check="fixture", passed=True)],
        source_artifact_hashes=[],
    )

    def execute(raw):
        d.atomic_json(raw / "evidence.json", evidence)
        return evidence

    monkeypatch.setattr(r, "execute", execute)
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=True, receipts=[], coverage=dict(complete=True))
    )
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=True, receipts=[]))
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--private-output", str(output), "--raw-dir", str(tmp_path / "private")]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["diagnosis_rows_ready_score"] == 1
    assert value["completed_count"] == 15
    monkeypatch.setattr(r, "validate", lambda raw: dict(passed=False, receipts=[], coverage={}))
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=False, receipts=[]))
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["diagnosis_rows_ready_score"] == 0
    assert not value["required_checks_passed"]


def test_req_report_8815_authenticated_orchestration(tmp_path, monkeypatch):
    """The runner reuses authenticated sealed rows and keeps every finite fit slot."""
    from carnot.reporting import learning_signal_runner_8815 as r

    payload = d.m.world("old", "support")
    public = dict(online=[payload], retention=[payload])
    recipe = dict(unit_id="old", structural_hash="old-full")
    frozen = dict(streams=[dict(worlds=[recipe], retention=[recipe], stream_id="s0")])
    source = dict(input_refs=["public"])
    old = dict(required_checks_passed=True, raw_shard_hashes=[dict(sha256=r.audit.PIN), "frozen"])

    def load(ref):
        if ref == "public":
            return public
        if ref == "frozen":
            return frozen
        if ref == dict(sha256=r.audit.PIN):
            return source
        return old

    monkeypatch.setattr(d, "load", load)
    monkeypatch.setattr(
        r.audit,
        "reduce",
        lambda source, frozen: dict(leakage_checks=dict(checks=dict(authenticated=True))),
    )
    diagnostic, refs, excluded = r.authenticate()
    assert len(diagnostic) == 4 and len(excluded) == 1 and len(refs) == 3
    monkeypatch.setattr(
        d, "development", lambda excluded: ([dict(split="fit"), dict(split="validation")], {})
    )
    monkeypatch.setattr(d, "controls", lambda: [dict(passed=True)])
    monkeypatch.setattr(
        d,
        "fit",
        lambda train, valid, seed, objective, lr: dict(seed=seed, objective=objective, lr=lr),
    )
    evidence = r.execute(tmp_path)
    assert len(evidence["fits"]) == 45 and (tmp_path / "evidence.json").is_file()


def test_req_report_8815_validation_scope(tmp_path, monkeypatch):
    """A health failure is distinct from an owned failure, and terminal argv is cold."""
    from carnot.reporting import learning_signal_runner_8815 as r
    from types import SimpleNamespace

    seen = []

    def commands(plan, logs):
        seen.extend(plan)
        return [dict(name=n, passed=scope == "owned", scope=scope) for n, _, scope, _ in plan]

    monkeypatch.setattr(r, "commands", commands)
    monkeypatch.setattr(
        r,
        "Coverage",
        lambda **kwargs: SimpleNamespace(combine=lambda **kwargs: None, save=lambda: None),
    )
    monkeypatch.setattr(r, "coverage_summary", lambda data, owned: dict(complete=True))
    qualified = r.validate(tmp_path)
    assert qualified["passed"] and len(qualified["receipts"]) == 7
    assert any(p[2] == "repository_health" for p in seen)
    assert r.terminal(tmp_path / "candidate", tmp_path)["passed"]
    cold = next(p[1] for p in seen if p[0] == "cold_replay")
    assert cold[:6] == ["env", "-u", "PYTHONPATH", "-C", "/tmp", str(r.ROOT / ".venv/bin/python")]


def test_req_report_8815_cli_and_forwarded_output(tmp_path):
    """The real entrypoint imports outside the checkout and rejects an invalid date."""
    from carnot.reporting import learning_signal_runner_8815 as r
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-u", str(r.ROOT / r.CLI), "--date", "bad"],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1 and "date_rejected" in result.stdout
    result = r.commands(
        [("forwarded", [sys.executable, "-u", "-c", 'print("child_work")'], "owned", 10)], tmp_path
    )
    assert result[0]["passed"] and "child_work" in Path(result[0]["log_path"]).read_text()
