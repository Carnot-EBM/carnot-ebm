"""Fixtures for REQ-LEARN-7760 and its causal, durable, terminal scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from carnot import experiment_7760_v675_online_runner as exp


def test_preflight_authenticates_current_bytes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-TERMINAL: missing producer is a terminal external block."""
    checks, hashes, names = exp.preflight(tmp_path)
    assert names == []
    assert any(not item["passed"] and item["upstream_id"] == "7742" for item in checks)
    assert hashes["missing_inputs"]
    checks, hashes, names = exp.preflight(exp.ROOT)
    assert all(item["passed"] for item in checks)
    assert len(names) == 16
    assert hashes["eligible_producers"]


def test_causal_admission_and_controls(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-CAUSAL: a delayed label adds a useful absent coefficient."""
    names = exp.preflight(exp.ROOT)[2]
    run = exp.run_fixture(tmp_path, names)
    adaptive = run["arms"]["adaptive"]
    assert len(adaptive["rows"]) == 8 * 20
    assert any(row["kind"] == "admission" for row in adaptive["rows"])
    assert any(
        row["kind"] == "update" and row["label_arrival_block"] > row["block"]
        for row in adaptive["rows"]
    )
    assert adaptive["admitted_predicates"]
    assert adaptive["decision_changed_after_admission"]
    assert set(run["arms"]["complete_static"]["initial_active"]) == set(names)
    assert run["arms"]["frozen"]["admitted_predicates"] == []
    assert all(arm["update_calls"] == 8 * 12 for arm in run["arms"].values())
    assert all(arm["parameter_ceiling"] == 16 for arm in run["arms"].values())
    assert exp.cold_reduce(tmp_path / "raw.json")["valid"]
    artifact = exp.build_artifact(
        "20260927", time.monotonic(), [], exp.preflight(exp.ROOT)[1], run, [], {}
    )
    assert artifact["verdict_class"] == "circular_positive"
    assert len(artifact["rows"]) == 640
    assert artifact["acceptance_gate_results"]["decision_benefit"] is None
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert exp.main(["--cold-reduce", str(candidate)]) == 0
    changed_source = json.loads(candidate.read_text())
    changed_source["source_artifact_hashes"]["eligible_producers"][exp.SOURCE] = "sha256:changed"
    candidate.write_text(json.dumps(changed_source))
    with pytest.raises(ValueError, match="candidate_source_hash_invalid"):
        exp.main(["--cold-reduce", str(candidate)])
    candidate.write_text(json.dumps(artifact))
    changed_candidate = json.loads(candidate.read_text())
    changed_candidate["rows"][0]["decision"] = not changed_candidate["rows"][0]["decision"]
    candidate.write_text(json.dumps(changed_candidate))
    with pytest.raises(ValueError, match="candidate_rows_invalid"):
        exp.main(["--cold-reduce", str(candidate)])
    broken = json.loads((tmp_path / "raw.json").read_text())
    broken["arms"]["adaptive"]["rows"][0]["label"] = 9
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="raw_rows_invalid"):
        exp.cold_reduce(corrupt)


def test_guarded_queue_and_credit(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-CAUSAL: no early label, duplicate credit, or silent overflow."""
    names = exp.preflight(exp.ROOT)[2]
    runner = exp.OnlineRunner(tmp_path / "one.json", tmp_path / "bank.json", names, "adaptive")
    with pytest.raises(ValueError, match="premature_label_access"):
        runner.label(0, 0)
    runner.predict_block(0)
    assert len(runner.state["queue"]) == 12
    with pytest.raises(ValueError, match="pending_overflow"):
        runner.predict_block(1)
    assert runner.state["overflow"] == 1
    assert (
        len([row for row in runner.state["rows"] if row.get("status") == "unstarted_overflow"])
        == 20
    )
    runner.release_block(0, 1)
    with pytest.raises(ValueError, match="credit_reused"):
        runner.release_block(0, 1)
    assert runner.state["credits"] == 1
    with pytest.raises(ValueError, match="block_order"):
        runner.predict_block(0)
    with pytest.raises(ValueError, match="unfinished_feedback"):
        runner.finish()


def test_expiration_rows_are_censored(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-CAUSAL: expired handles remain visible in raw rows."""
    names = exp.preflight(exp.ROOT)[2]
    runner = exp.OnlineRunner(
        tmp_path / "expired.json", tmp_path / "expired-bank.json", names, "adaptive"
    )
    runner.predict_block(0)
    runner.expire_block(0, 1)
    assert runner.state["expired"] == 20
    assert not runner.state["queue"]
    assert sum(row["censored"] for row in runner.state["rows"]) == 20
    assert all(row["label"] is None for row in runner.state["rows"] if row["censored"])


def test_empty_static_rejection_and_corrupt_state(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-DURABLE: controls and corrupt restart fail safely."""
    names = exp.preflight(exp.ROOT)[2]
    with pytest.raises(ValueError, match="complete_static_empty"):
        exp.OnlineRunner(
            tmp_path / "empty.json", tmp_path / "empty-bank.json", [], "complete_static"
        )
    runner = exp.OnlineRunner(
        tmp_path / "reject.json", tmp_path / "reject-bank.json", names, "adaptive", admit=False
    )
    runner.predict_block(0)
    runner.release_block(0, 1)
    assert runner.state["active"] == []
    runner.state_path.write_text("{}")
    with pytest.raises(ValueError, match="state_invalid"):
        exp.OnlineRunner(runner.state_path, runner.bank_path, names, "adaptive")


def test_new_process_restart_and_hard_exit(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-DURABLE: resumed child and uninterrupted run match."""
    names = exp.preflight(exp.ROOT)[2]
    control = exp.run_arm(tmp_path / "control", names, "adaptive")
    staged = exp.OnlineRunner(
        tmp_path / "staged.json", tmp_path / "staged-bank.json", names, "adaptive"
    )
    for block in range(4):
        if block:
            staged.release_block(block - 1, block)
        staged.predict_block(block)
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_7760_v675_online_runner",
            "--resume-private",
            str(staged.state_path),
            str(staged.bank_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resumed = json.loads(child.stdout.splitlines()[-1])
    assert len(child.stdout.splitlines()[-1]) < 4000
    assert resumed["decision_hash"] == control["decision_hash"]
    assert resumed["bank_hash"] == control["bank_hash"]
    assert resumed["queue_hash"] == control["queue_hash"]
    assert resumed["credits"] == control["credits"]
    assert resumed["rng_counter"] == control["rng_counter"]
    assert resumed["model_hash"] == control["model_hash"]
    assert exp.hard_exit_commit_check(tmp_path / "crash", names)
    receipt = json.loads((tmp_path / "crash" / "receipt.json").read_text())
    assert len(receipt["children"]) == 2
    assert all(row["exit_code"] == 0 and row["log_sha256"] for row in receipt["children"])


def test_blocked_artifact_and_main(tmp_path: Path) -> None:
    """SCENARIO-LEARN-7760-TERMINAL: absent external bytes complete as blocked."""
    output = tmp_path / "blocked.json"
    assert (
        exp.main(["--root", str(tmp_path), "--output", str(output), "--raw", str(tmp_path / "raw")])
        == 0
    )
    artifact = json.loads(output.read_text())
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["online_runtime_ready_score"] == 0
    assert artifact["gate_check_summary"]
    assert all("artifact_sha256" in item for item in artifact["gate_check_summary"])


def test_validation_orchestration_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-LEARN-7760-TERMINAL: affected failures close readiness; global debt stays separate."""
    reference = {key: "same" for key in ("decision_hash", "bank_hash", "queue_hash", "model_hash")}
    reference.update({"credits": 8, "rng_counter": 160})
    row = {"label": 0, "excluded": False, "censored": False}
    timing = dict.fromkeys(("lookup_s", "update_s", "serialization_s", "acknowledgement_s"), 0.0)
    measured = {
        "arms": {
            arm: {
                "rows": [row],
                "lifecycle": [],
                "timings_s": timing,
                **(reference if arm == "adaptive" else {}),
            }
            for arm in exp.ARMS
        }
    }

    def fake_fixture(folder: Path, names: list[str]) -> dict:
        path = folder / "raw.json"
        path.write_text("{}")
        return {**measured, "raw_path": str(path)}

    def fake_reduce(path: Path) -> dict:
        return {"valid": True, "raw_sha256": exp.sha256_file(path)}

    class FakeRunner:
        def __init__(self, state_path: Path, bank_path: Path, *args: object) -> None:
            self.state_path, self.bank_path = state_path, bank_path

        def predict_block(self, block: int) -> None:
            assert block < 8

        def release_block(self, block: int, arrival: int) -> None:
            assert arrival == block + 1

    mode = {
        "terminal_failure": False,
        "parity_failure": False,
        "hard_exit_failure": False,
        "malformed_restart": False,
    }

    def fake_commands(root: Path, commands: list, **kwargs: object) -> list[dict]:
        receipts = []
        for command in commands:
            passed = command.name != "full_python_suite"
            if command.name == "adversarial_verify" and mode["terminal_failure"]:
                passed = False
            child = dict(reference)
            if mode["parity_failure"]:
                child["model_hash"] = "changed"
            receipts.append(
                {
                    "name": command.name,
                    "passed": passed,
                    "exit_code": 0 if passed else 1,
                    "log_path": str(tmp_path / f"{command.name}.log"),
                    "output_tail": "broken"
                    if command.name == "cold_restart" and mode["malformed_restart"]
                    else json.dumps(child),
                }
            )
        return receipts

    monkeypatch.setattr(exp, "run_fixture", fake_fixture)
    monkeypatch.setattr(exp, "cold_reduce", fake_reduce)
    monkeypatch.setattr(exp, "OnlineRunner", FakeRunner)
    monkeypatch.setattr(exp, "run_commands", fake_commands)
    monkeypatch.setattr(
        exp, "hard_exit_commit_check", lambda folder, names: not mode["hard_exit_failure"]
    )
    raw = tmp_path / "validation_raw"
    clean = exp.run_experiment(exp.ROOT, "20260927", tmp_path / "clean.json", raw=raw)
    assert clean["verdict_class"] == "circular_positive"
    assert clean["online_runtime_ready_score"] == 1
    assert not clean["validation_receipts"]["repository_health"]["full_python_suite_passed"]
    mode["terminal_failure"] = True
    failed = exp.run_experiment(exp.ROOT, "20260927", tmp_path / "failed.json", raw=raw)
    assert failed["verdict_class"] == "disqualified"
    assert failed["flagged_adversarial"]
    assert failed["online_runtime_ready_score"] == 0
    mode["terminal_failure"] = False
    mode["parity_failure"] = True
    no_parity = exp.run_experiment(exp.ROOT, "20260927", tmp_path / "parity.json", raw=raw)
    assert no_parity["verdict_class"] == "disqualified"
    assert any(item["field"] == "cold_restart_parity" for item in no_parity["gate_check_summary"])
    mode["parity_failure"] = False
    mode["hard_exit_failure"] = True
    no_commit = exp.run_experiment(exp.ROOT, "20260927", tmp_path / "commit.json", raw=raw)
    assert no_commit["verdict_class"] == "disqualified"
    assert any(item["field"] == "hard_exit_commit" for item in no_commit["gate_check_summary"])
    mode["hard_exit_failure"] = False
    mode["malformed_restart"] = True
    malformed = exp.run_experiment(exp.ROOT, "20260927", tmp_path / "malformed.json", raw=raw)
    assert malformed["verdict_class"] == "disqualified"
    no_validation = exp.run_experiment(
        exp.ROOT, "20260927", tmp_path / "no_validation.json", raw=raw, validate=False
    )
    assert no_validation["online_runtime_ready_score"] == 0


def test_private_main_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-LEARN-7760-TERMINAL: child modes use private state and explicit paths."""
    names = exp.preflight(exp.ROOT)[2]
    monkeypatch.setattr(exp, "preflight", lambda root: ([], {}, names))
    monkeypatch.setattr(
        exp, "run_fixture", lambda folder, selected: {"raw_path": str(folder / "raw.json")}
    )
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"valid": True})
    assert exp.main(["--private-e2e", "--root", str(tmp_path)]) == 0
    state_path = tmp_path / "private.json"
    bank_path = tmp_path / "private-bank.json"
    state_path.write_text(json.dumps({"names": names, "arm": "adaptive"}))
    calls: list[tuple] = []

    class FakeRunner:
        def __init__(self, state: Path, bank: Path, selected: list[str], arm: str) -> None:
            assert state == state_path and bank == bank_path
            self.state = {"completed_blocks": list(range(4))}

        def release_block(self, block: int, arrival: int) -> None:
            calls.append(("release", block, arrival))

        def predict_block(self, block: int) -> None:
            calls.append(("predict", block))

        def finish(self) -> dict:
            return {"valid": True}

    monkeypatch.setattr(exp, "OnlineRunner", FakeRunner)
    assert exp.main(["--resume-private", str(state_path), str(bank_path)]) == 0
    assert calls[-1] == ("release", 7, 8)
    invoked: list[tuple] = []
    monkeypatch.setattr(
        exp, "run_experiment", lambda *args, **kwargs: invoked.append((args, kwargs))
    )
    assert (
        exp.main(["--root", str(tmp_path), "--output", "out.json", "--raw", "raw", "--no-validate"])
        == 0
    )
    assert invoked[0][1]["validate"] is False


def test_malformed_custody_and_runner_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-LEARN-7760-CAUSAL: malformed inputs and illegal lifecycle calls fail closed."""
    for relative in (exp.SOURCE, exp.MANIFEST, exp.BANK_MODULE, exp.FEATURE_MODULE):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{")
    checks, hashes, names = exp.preflight(tmp_path)
    assert names == []
    assert not all(item["passed"] for item in checks)
    assert hashes["module_hashes"]
    valid = exp.preflight(exp.ROOT)[2]
    with pytest.raises(ValueError, match="model_config_invalid"):
        exp.OnlineRunner(tmp_path / "bad.json", tmp_path / "bad-bank.json", valid, "other")
    runner = exp.OnlineRunner(
        tmp_path / "guard.json", tmp_path / "guard-bank.json", valid, "adaptive"
    )
    with pytest.raises(ValueError, match="premature_label_access"):
        runner.release_block(0, 1)
    with pytest.raises(ValueError, match="premature_label_access"):
        runner.expire_block(0, 1)
    runner.predict_block(0)
    with pytest.raises(ValueError, match="credit_reused"):
        runner.release_block(0, 0)
    with pytest.raises(ValueError, match="credit_reused"):
        runner.expire_block(0, 0)
    runner.state["queue"].pop()
    with pytest.raises(ValueError, match="queue_invalid"):
        runner.release_block(0, 1)
    runner.state["queue"].append("adaptive-0-update-11")
    primitives = exp.grammar()["primitives"]
    monkeypatch.setattr(runner.bank, "propose", lambda: {"pair": primitives[4:6], "weight": 0.2})
    monkeypatch.setattr(runner.bank, "admit", lambda accepted: {"accepted": accepted})
    runner.release_block(0, 1)
    assert runner.state["admitted_predicates"] == []
    with pytest.raises(ValueError, match="credit_reused"):
        runner.expire_block(0, 1)


def test_hard_exit_failure_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-LEARN-7760-DURABLE: a failed owned child cannot pass commit parity."""
    names = exp.preflight(exp.ROOT)[2]
    actual = exp.subprocess.run
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )
    assert not exp.hard_exit_commit_check(tmp_path / "before", names)
    assert (
        json.loads((tmp_path / "before/receipt.json").read_text())["children"][0]["exit_code"] == 1
    )
    calls = 0

    def second_fails(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            return SimpleNamespace(returncode=1, stdout="", stderr="failed")
        return actual(*args, **kwargs)

    monkeypatch.setattr(exp.subprocess, "run", second_fails)
    assert not exp.hard_exit_commit_check(tmp_path / "after", names)


def test_cold_reducer_rejects_each_changed_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-LEARN-7760-TERMINAL: rows, model, queue, budget, and delay are independent gates."""

    class FakeBank:
        def __init__(self, *args: object) -> None:
            self.state_hash = "bankhash"

        def replay_ledger(self) -> dict:
            return {"valid": True}

    monkeypatch.setattr(exp, "Bank", FakeBank)
    arms: dict[str, dict] = {}
    for arm in exp.ARMS:
        rows = []
        for block in range(8):
            for kind, count in (("update", 12), ("admission", 8)):
                for index in range(count):
                    mapped = (
                        index
                        if arm != "shuffled"
                        else (index * 5 + 3) % 12
                        if kind == "update"
                        else (index * 3 + 1) % 8
                    )
                    rows.append(
                        {
                            "event_id": f"{arm}-{block}-{kind}-{index}",
                            "block": block,
                            "kind": kind,
                            "index": index,
                            "decision": False,
                            "probability": 0.4,
                            "label": int(mapped % 4 == 3),
                            "label_arrival_block": block + 1,
                        }
                    )
        state_path = tmp_path / f"{arm}.json"
        state_path.write_text(json.dumps({"rows": rows, "weights": {}, "queue": []}))
        arms[arm] = {
            "rows": rows,
            "state_path": str(state_path),
            "bank_path": str(tmp_path / f"{arm}-bank.json"),
            "bank_hash": "bankhash",
            "decision_hash": exp.canonical_hash(
                [(r["event_id"], r["decision"], r["probability"]) for r in rows]
            ),
            "model_hash": exp.canonical_hash({}),
            "queue_hash": exp.canonical_hash([]),
            "credits": 8,
            "update_calls": 96,
            "rng_counter": 160,
        }
    raw = {"schema": "exp7760-raw-v1", "arms": arms}
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    assert exp.cold_reduce(path)["valid"]
    mutations = (
        ("scope", "raw_scope_invalid"),
        ("length", "raw_rows_invalid"),
        ("bank", "raw_rows_invalid"),
        ("decision", "decision_hash_invalid"),
        ("model", "model_hash_invalid"),
        ("queue", "queue_hash_invalid"),
        ("budget", "budget_invalid"),
        ("delay", "feedback_delay_invalid"),
        ("label", "label_invalid"),
    )
    for kind, error in mutations:
        altered = json.loads(json.dumps(raw))
        result = altered["arms"]["adaptive"]
        if kind == "scope":
            altered["schema"] = "wrong"
        elif kind == "length":
            result["rows"].pop()
        elif kind == "bank":
            result["bank_hash"] = "wrong"
        elif kind == "decision":
            result["decision_hash"] = "wrong"
        elif kind == "model":
            result["model_hash"] = "wrong"
        elif kind == "queue":
            result["queue_hash"] = "wrong"
        elif kind == "budget":
            result["credits"] = 9
        elif kind == "delay":
            result["rows"][0]["label_arrival_block"] = 9
        elif kind == "label":
            result["rows"][0]["label"] = 9
        if kind in {"delay", "label"}:
            Path(result["state_path"]).write_text(
                json.dumps({"rows": result["rows"], "weights": {}, "queue": []})
            )
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match=error):
            exp.cold_reduce(path)
        Path(result["state_path"]).write_text(
            json.dumps({"rows": arms["adaptive"]["rows"], "weights": {}, "queue": []})
        )
