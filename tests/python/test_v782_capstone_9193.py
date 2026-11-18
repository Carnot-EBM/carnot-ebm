"""REQ-REPORT-9193-V782: terminal accounting cannot invent scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.historical_contract_inputs import original_root
from carnot.reporting import v782_capstone as m
from carnot.reporting import v782_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, validate_current_work_receipt


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Retained producer bytes support accounting without rerunning model work."""
    return m.collect(
        tmp_path_factory.mktemp("capstone9193"),
        "20261117",
        original_root(tmp_path_factory.mktemp("inputs9193"), 782),
    )


def test_fourteen_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9193-V782: failed science never prevents complete accounting."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert all(r["passed"] for r in work["checks"]), [r for r in work["checks"] if not r["passed"]]
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    slots = {r["experiment_id"]: r for r in value["outcome_rows"]}
    assert list(slots) == list(range(9180, 9194))
    assert slots[9180]["verdict_class"] == "null"
    assert slots[9181]["verdict_class"] == slots[9192]["verdict_class"] == "disqualified"
    assert slots[9182]["scheduler_alternative"]["sha256"]
    assert slots[9182]["gate_operands"][0]["observed"] == 0
    assert slots[9183]["gate_operands"][0]["observed"] == "missing_field"
    assert slots[9184]["gate_operands"][1]["observed"] == "missing_file"
    assert all(slots[i]["primary_status"] == "missing_file" for i in range(9182, 9192))
    assert len(value["two_bound_rows"]) == 2 and len(value["seven_bound_rows"]) == 7
    assert len(value["ces_summary"]["ranking_bound_rows"]) == 3
    assert value["ces_summary"]["complete_score"] == 0
    assert not value["ces_summary"]["primary_gate_rescue"]
    for row in (
        value["two_bound_rows"]
        + value["seven_bound_rows"]
        + value["ces_summary"]["ranking_bound_rows"]
    ):
        assert row["absolute_metric"] is None and row["denominator"] == 0
    assert value["collection_partition"]["qid_counts"] == [100] * 3
    assert value["collection_partition"]["answer_count"] == 900
    assert value["validation_obligations"]["union_count"] == 1040
    assert value["validation_obligations"]["current_qualification_credit"] == 0
    assert value["previous_milestone_accounting"] == dict(
        executed=4, skipped=10, scientific_success_inferred=False
    )
    assert not value["arc_summary"]["live_path_observed"]
    assert not value["independent_human_confirmation"]["warranted_now"]
    assert value["MODEL_SPECS"] == []
    assert value["random_seed"] == 782193
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert not value["publication_gates"]["current_science_paper_ready"]
    assert not value["activation_authorized"] and not value["external_publication_authorized"]
    assert all(r["next_reopen_condition"] for r in value["continuation_decisions"])
    retired = {r["task_id"] for r in value["retirement_recommendations"] if r["condition_met"]}
    assert {"exp9181-validation-loop-repair", "exp9182-forward-qualification"} <= retired
    assert all(r["hypothesis_retired"] is False for r in value["retirement_recommendations"])
    assert (
        validate_current_work_receipt(
            json.loads(Path(work["current_work_reference"]["path"]).read_bytes())
        )
        == []
    )


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9193-V782: external absence blocks; owned failure disqualifies."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261117", tmp_path / "absent")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert len(value["outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += " altered"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_strict_operands(tmp_path):
    """REQ-REPORT-9193-V782: zero, false, missing field and missing file differ."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {"exp1-test": entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {"exp1-test": entry})["observed"] == 0
    entry["value"]["ready"] = True
    assert not e.edge(gate, {"exp1-test": entry})["passed"]
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_and_row_controls(work, tmp_path):
    """SCENARIO-REPORT-9193-V782: rehashing cannot bless changed rows or producers."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key in ["outcome_rows", "science_decisions", "current_llm_invocations"]:
        changed = dict(value, **{key: 1})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    for field in ["cost_summary", "donor_changed_label_fraction"]:
        changed = deepcopy(work)
        prior = changed["values"]["prior:9179"]["value"]
        if field == "cost_summary":
            prior[field]["scientific_request_counts"]["completed"] = 1
        else:
            prior[field] += 0.1
        with pytest.raises(ValueError, match="cost_aggregate|donor_aggregate"):
            m.build(changed)


def test_private_real_cli(work, tmp_path):
    """SCENARIO-REPORT-9193-V782: real CLI succeeds, refuses and rejects tampering."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9193] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9193] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261116").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_main_paths(work, tmp_path, monkeypatch):
    """REQ-REPORT-9193-V782: shared orchestration retains refusal and validation gates."""

    def collect(raw, *_):
        return dict(deepcopy(work), raw=str(raw))

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1


def test_shared_deadline(work, tmp_path, monkeypatch):
    """REQ-REPORT-9193-V782: an exhausted shared reserve refuses another child."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    monkeypatch.setenv("CARNOT_V782_CAPSTONE_DEADLINE", "0")
    assert m.main(["--from-work", str(source), "--output", str(tmp_path / (m.NAME + ".json"))]) == 1


def test_retirement_manifest(work):
    """REQ-REPORT-9193-V782: exact failed execution routes enter existing exclusions."""
    value = m.build(work)
    rows = [r for r in value["retirement_recommendations"] if r["condition_met"]]
    assert rows and all(r["manifest_retirement_applied"] for r in rows)
    assert any(
        r["experiment_id"] == "exp9168-forward-qualification"
        for r in value["preserved_v781_retirement"]
    )
