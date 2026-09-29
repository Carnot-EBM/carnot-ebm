"""Direct checks for the V680 independent audit (REQ-REPORT-7835)."""

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_7835_v680_independent_evidence_audit as audit


ROOT = Path(__file__).resolve().parents[2]


def test_current_branch_inventory_and_raw_custody():
    """SCENARIO-REPORT-7835-MISSING and SCENARIO-REPORT-7835-CUSTODY."""
    sources, failures = audit.inspect_sources(ROOT)
    assert [s["upstream_id"] for s in sources] == [
        "Exp7824",
        "Exp7826",
        "Exp7827",
        "Exp7829",
        "Exp7830",
        "Exp7832",
    ]
    assert {s["state"] for s in sources} >= {"disqualified", "conductor_only", "missing"}
    assert failures
    isolated = audit.audit_isolation(ROOT)
    assert isolated["independent_n"] == 640
    assert len(isolated["rows"]) == 640
    assert not isolated["failed_checks"]
    assert {r["role"] for r in isolated["rows"]} >= {"fit", "evaluation"}


def test_private_feature_and_unknown_label_mutations():
    """SCENARIO-REPORT-7835-CUSTODY: no private value enters public features."""
    clean = {
        "family_id": "f",
        "feature_dim": 2,
        "view_a_tensor_rows": [],
        "view_a_tensor_sha256": "x",
        "view_b_tensor_rows": [],
        "view_b_tensor_sha256": "y",
    }
    assert audit.check_public_feature(clean) == []
    for private in ("confidence", "gold_label", "unknown_label_sentinel"):
        changed = {**clean, private: 1}
        assert "private_feature" in audit.check_public_feature(changed)
    assert "public_schema" in audit.check_public_feature({"family_id": "f"})


def test_policy_and_random_abstention_challenges():
    """REQ-REPORT-7835: labels cannot choose actions or random controls."""
    ids = ["a", "b", "c", "d"]
    first = audit.random_matched_ids(ids, 2, 68001)
    assert first == audit.random_matched_ids(list(reversed(ids)), 2, 68001)
    assert len(first) == 2
    rows = [
        {
            "family_id": "a",
            "probability": 0.8,
            "label": 1,
            "action": "escalate",
            "selection_role": "fit",
        }
    ]
    assert audit.check_policy_rows(rows, 0.7) == []
    assert "label_selected_action" in audit.check_policy_rows(
        [{**rows[0], "action": "accept"}], 0.7
    )
    assert "post_evaluation_tuning" in audit.check_policy_rows(
        [{**rows[0], "selection_role": "evaluation"}], 0.7
    )


def test_manifest_dispatch_rejects_undeclared_and_drift(tmp_path):
    """SCENARIO-REPORT-7835-DISPATCH: exact name, argv and class are frozen."""
    manifest = audit.load_manifest(ROOT)
    command = manifest["commands"][0]
    for name, argv, classification in (
        ("unknown", command["argv"], "required"),
        (command["name"], ["true"], "required"),
        (command["name"], command["argv"], "diagnostic"),
    ):
        with pytest.raises(ValueError):
            audit.assert_declared(manifest, name, argv, classification)
    assert audit.assert_declared(manifest, command["name"], command["argv"], "required")


def test_sealed_log_retry_and_mutation(tmp_path):
    """SCENARIO-REPORT-7835-DISPATCH: closed log bytes are immutable."""
    first = audit.seal_log(tmp_path, "test", b"first\n")
    second = audit.seal_log(tmp_path, "test", b"second\n")
    assert first["log_path"] != second["log_path"]
    assert audit.check_log(first) == []
    Path(first["log_path"]).write_bytes(b"changed\n")
    assert audit.check_log(first) == ["validation_log_changed"]


def test_blocked_artifact_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-7835-MISSING: blocked is terminal and source-bound."""
    sources, failures = audit.inspect_sources(ROOT)
    result = audit.build_artifact(ROOT, "20260928", sources, failures, audit.audit_isolation(ROOT))
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["verdict_class"] == "blocked"
    assert result["independent_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["oracle_distinct_gate_eligible"] is False
    assert all(row["rejected"] for row in result["mutation_results"])
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert audit.cold_replay(path) == []
    changed = copy.deepcopy(result)
    changed["rows"].pop()
    path.write_text(json.dumps(changed))
    assert "rows_changed" in audit.cold_replay(path)


def test_malformed_and_qualified_producers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7835-MISSING: reject invalid JSON, accept exact gates."""
    path = tmp_path / "science.json"
    monkeypatch.setattr(audit, "PLAN", ((7827, "science.json"),))
    path.write_text("{")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified" and failures
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7827,
                "milestone": "2026.09.680",
                "run_date": "20260928",
                "flagged_adversarial": False,
                "verdict_class": "null",
                "decision_measurement_ready_score": 1,
            }
        )
    )
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["eligibility"] is True and not failures


def test_isolation_rejects_corrupted_joins_and_bytes(monkeypatch):
    """SCENARIO-REPORT-7835-CUSTODY: raw joins, bytes and roles are checked."""
    original_reader = audit._jsonl

    def changed_rows(path):
        rows = original_reader(path)
        if path.name == "public_features.jsonl":
            rows.pop()
        elif path.name == "public_records.jsonl":
            rows[0]["private_extra"] = 1
            rows[1]["view_a"]["source_bytes"] = "00"
        elif path.name == "label_sidecar.jsonl":
            rows[2]["role"] = "wrong"
            rows[3]["response_sha256"] = "wrong"
        return rows

    monkeypatch.setattr(audit, "_jsonl", changed_rows)
    result = audit.audit_isolation(ROOT)
    assert {
        "family_join",
        "public_allowlist",
        "role_join",
        "source_answer_bytes",
        "label_join",
    } <= set(result["failed_checks"])
    monkeypatch.setattr(audit, "sha256_file", lambda path: "wrong")
    with pytest.raises(ValueError, match="changed feature_path"):
        audit.audit_isolation(ROOT)


def test_cold_replay_rejects_raw_and_sealed_log_changes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7835-DISPATCH: changed inputs and logs fail replay."""
    sources, failures = audit.inspect_sources(ROOT)
    result = audit.build_artifact(ROOT, "20260928", sources, failures, audit.audit_isolation(ROOT))
    log = audit.seal_log(tmp_path, "closed", b"ok")
    result["observed_child_commands"] = [log]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert audit.cold_replay(path) == []
    Path(log["log_path"]).write_bytes(b"bad")
    assert "validation_log_changed" in audit.cold_replay(path)
    monkeypatch.setattr(audit, "audit_isolation", lambda root: (_ for _ in ()).throw(ValueError()))
    assert audit.cold_replay(path) == ["raw_custody_changed"]


def test_invalid_policy_inputs():
    """REQ-REPORT-7835: bad probabilities and matched counts fail closed."""
    with pytest.raises(ValueError):
        audit.random_matched_ids(["a"], 2, 1)
    assert "invalid_probability" in audit.check_policy_rows(
        [{"selection_role": "fit", "probability": None, "action": "accept"}], 0.5
    )


def test_declared_child_exit_timeout_and_unknown(tmp_path):
    """SCENARIO-REPORT-7835-DISPATCH: real children exit or meet owned deadline."""
    import sys

    command = {
        "name": "tiny",
        "argv": [sys.executable, "-c", "print('ok')"],
        "classification": "required",
        "timeout_s": 5,
    }
    manifest = {"commands": [command]}
    receipt = audit.run_declared(ROOT, manifest, "tiny", tmp_path)
    assert receipt["exit_code"] == 0 and audit.check_log(receipt) == []
    with pytest.raises(ValueError):
        audit.run_declared(ROOT, manifest, "unlisted", tmp_path)
    sleepy = {
        **command,
        "argv": [sys.executable, "-c", "import time;time.sleep(1)"],
        "timeout_s": 0.01,
    }
    receipt = audit.run_declared(ROOT, {"commands": [sleepy]}, "tiny", tmp_path)
    assert receipt["timed_out"] is True and receipt["exit_code"] != 0


def test_real_cli_modes_and_validation_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7835-DISPATCH: the real CLI uses frozen command names."""
    import importlib.util
    import sys

    path = ROOT / "scripts/experiments/experiment_7835_v680_independent_evidence_audit.py"
    spec = importlib.util.spec_from_file_location("exp7835_cli_test", path)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path / "output.json")
    monkeypatch.setattr(cli, "CANDIDATE", tmp_path / "candidate.json")
    command = {"name": "tiny", "argv": ["true"], "classification": "required", "timeout_s": 5}
    monkeypatch.setattr(cli.audit, "load_manifest", lambda root: {"commands": [command]})
    log = cli.audit.seal_log(tmp_path, "tiny", b"ok")
    receipt = {
        "name": "tiny",
        "classification": "required",
        "exit_code": 0,
        "timed_out": False,
        **log,
    }
    monkeypatch.setattr(cli.audit, "run_declared", lambda *args: receipt)
    monkeypatch.setattr(sys, "argv", [str(path), "--date", "20260928"])
    assert cli.main() == 0
    assert json.loads(cli.OUTPUT.read_text())["verdict_class"] == "blocked"
    monkeypatch.setattr(sys, "argv", [str(path), "--check-only", str(cli.CANDIDATE)])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", [str(path), "--dispatch", "tiny"])
    assert cli.main() == 0
    monkeypatch.setattr(cli.audit, "run_declared", lambda *args: {**receipt, "exit_code": 1})
    monkeypatch.setattr(sys, "argv", [str(path), "--date", "20260928"])
    assert cli.main() == 0
    assert json.loads(cli.OUTPUT.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(sys, "argv", [str(path), "--date", "20260927"])
    with pytest.raises(SystemExit):
        cli.main()
