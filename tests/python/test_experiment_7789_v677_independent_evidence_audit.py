"""REQ-REPORT-7789: independent V677 custody, rows, and terminal replay."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7789_v677_independent_evidence_audit as audit
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting import experiment_7303_validation_scope as checks


def rows(role: str, families: int, arms: tuple[str, ...]) -> list[dict]:
    """Give each family all three seeds and the same immutable label."""
    return [
        {
            "family_id": f"{role}-{i}",
            "label_join": f"{role}-{i}",
            "role": role,
            "label": i % 2,
            "arm": arm,
            "seed": seed,
            "probability": 0.8 if i % 2 else 0.2,
            "action": "reject" if i % 2 else "accept",
            "head_sha256": "head-0",
            "source_sha256": f"source-{i}",
            "prediction_tick": 1,
            "label_tick": 2,
            "query_open_tick": 1,
            "query_close_tick": 2,
            "scoring_version": "v0",
            "censored": False,
        }
        for i in range(families)
        for arm in arms
        for seed in range(3)
    ]


def test_custody_missing_and_optional_qwen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7789-CUSTODY: missing science stays blocked."""
    sources, failures = audit.inspect_sources(tmp_path)
    assert {f["upstream_id"] for f in failures} == {"Exp7786", "Exp7788"}
    assert all(f["observed"] == "missing" and f["artifact_hash"] is None for f in failures)
    result = audit.build_artifact(tmp_path, "20260927", sources, failures, {})
    assert result["honest_verdict"] == "complete_blocked_required_v677_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["independent_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["sample_size_budget"]["independent_n"] == 0
    assert len(result["rows"]) == 3
    assert result["source_artifact_hashes"][2]["upstream_id"] == "Exp7787"


def test_custody_checks_exact_fields_and_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7789-CUSTODY: a pre-gate receipt cannot qualify."""
    receipt = tmp_path / audit.PRE_GATE[7786]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    producer = tmp_path / audit.PLAN[7786]
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text(json.dumps({"experiment_id": 7786, "verdict_class": "blocked"}))
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["pre_gate_receipt"]["role"] == "explanation_only"
    assert sources[0]["sha256"] == sha256_file(producer)
    assert {f["field"] for f in failures} >= {"milestone", "run_date", "verdict_class"}


def test_row_reducer_and_private_tampers() -> None:
    """SCENARIO-REPORT-7789-ROWS: every paired unit and label is checked."""
    data = rows("evaluation64", 64, audit.DECISION_ARMS)
    clean = audit.reduce_rows(data, "evaluation64", 64, audit.DECISION_ARMS, "head-0")
    assert clean["failed_checks"] == []
    assert clean["independent_n"] == 64
    assert clean["coverage"] == 1
    assert clean["by_arm"][audit.DECISION_ARMS[0]]["brier"] == pytest.approx(0.04)
    assert clean["by_arm"][audit.DECISION_ARMS[0]]["cost"] == 0
    assert clean["comparisons"] and all("holm_p" in c for c in clean["comparisons"])
    for change, expected in (
        (lambda r: r.pop(), "roster"),
        (lambda r: r[0].update(label_join="wrong"), "label_join"),
        (lambda r: r[0].update(seed=1), "roster"),
        (lambda r: r[0].update(head_sha256="forged"), "head_digest"),
        (lambda r: r[0].update(prediction_tick=3), "label_chronology"),
        (lambda r: r[0].update(brier=0.9), "saved_metric"),
    ):
        altered = deepcopy(data)
        change(altered)
        assert (
            expected
            in audit.reduce_rows(altered, "evaluation64", 64, audit.DECISION_ARMS, "head-0")[
                "failed_checks"
            ]
        )
    retention = audit.reduce_rows(
        rows("retention32", 32, audit.LEARNING_ARMS),
        "retention32",
        32,
        audit.LEARNING_ARMS,
        "head-0",
    )
    assert retention["independent_n"] == 32 and retention["failed_checks"] == []


def test_causal_event_checks_and_no_admission() -> None:
    """SCENARIO-REPORT-7789-ROWS: immutable queries and later changes matter."""
    events = [
        {"kind": "prediction", "tick": 1, "query_id": "q", "version": "v0"},
        {"kind": "feedback", "tick": 3, "query_id": "q", "role": "update"},
        {"kind": "proposal", "tick": 4, "feedback_id": "q", "role": "update"},
        {"kind": "admission", "tick": 5, "feedback_id": "q"},
        {"kind": "commit", "tick": 6, "version": "v1"},
        {"kind": "later_decision", "tick": 7, "before_action": "reject", "after_action": "accept"},
        {"kind": "restart", "tick": 8, "exact_parity": True},
    ]
    assert audit.audit_events(events, [{"query_open_tick": 1, "query_close_tick": 2}], False) == []
    assert audit.audit_events([], [], False) == []
    for change, expected in (
        (lambda e: e[1].update(tick=0), "future_feedback"),
        (lambda e: e[4].update(tick=1), "commit_inside_query"),
        (lambda e: e[2].update(role="evaluation64"), "selection_label_leakage"),
        (lambda e: e[5].update(after_action="reject"), "no_later_change"),
        (lambda e: e[6].update(exact_parity=False), "restart"),
    ):
        altered = deepcopy(events)
        change(altered)
        assert expected in audit.audit_events(
            altered, [{"query_open_tick": 1, "query_close_tick": 2}], False
        )
    assert "identical_learned_frozen" in audit.audit_events(events, [], True)


def test_cold_replay_detects_forged_result(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7789-TERMINAL: a fresh read detects a forged headline."""
    sources, failures = audit.inspect_sources(tmp_path)
    value = audit.build_artifact(tmp_path, "20260927", sources, failures, {})
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    assert audit.cold_replay(path) == []
    value["independent_evidence_ready_score"] = 1
    path.write_text(json.dumps(value))
    assert "independent_evidence_ready_score_changed" in audit.cold_replay(path)


def test_authenticated_raw_branch_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7789-ROWS: raw inputs, never headlines, supply scores."""
    monkeypatch.setattr(
        audit,
        "paired",
        lambda values, seed=audit.SEED: {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            "lower95": 0,
            "upper95": 0,
            "p": 1,
        },
    )
    for number in (7786, 7788):
        producer = tmp_path / audit.PLAN[number]
        producer.parent.mkdir(parents=True, exist_ok=True)
        branch_rows = {
            "evaluation64": rows(
                "evaluation64", 64, audit.DECISION_ARMS if number == 7786 else audit.LEARNING_ARMS
            ),
        }
        if number == 7788:
            branch_rows["retention32"] = rows("retention32", 32, audit.LEARNING_ARMS)
        raw = tmp_path / f"raw-{number}.json"
        raw.write_text(
            json.dumps(
                {
                    "rowsets": branch_rows,
                    "head_sha256": "head-0",
                    "learned_head_sha256": "head-1",
                    "frozen_head_sha256": "head-0",
                }
            )
        )
        record = {
            "experiment_id": number,
            "milestone": "2026.09.677",
            "run_date": "20260927",
            "flagged_adversarial": False,
            "honest_verdict": "complete_null_valid",
            "verdict_class": "null",
            "raw_rows_path": raw.name,
            "raw_rows_sha256": sha256_file(raw),
        }
        for name in ("frozen_heads_manifest", "role_manifest", "rejection_records"):
            path = tmp_path / f"{name}-{number}.json"
            content = (
                {"head_sha256": "head-0"}
                if name == "frozen_heads_manifest"
                else {
                    role: sorted({row["family_id"] for row in role_rows})
                    for role, role_rows in branch_rows.items()
                }
                if name == "role_manifest"
                else []
            )
            path.write_text(json.dumps(content))
            record[f"{name}_path"] = path.name
            record[f"{name}_sha256"] = sha256_file(path)
        if number == 7788:
            path = tmp_path / "events-7788.json"
            path.write_text("[]")
            record["event_rows_path"] = path.name
            record["event_rows_sha256"] = sha256_file(path)
        producer.write_text(json.dumps(record))
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == [] and all(s["state"] == "eligible" for s in sources[:2])
    branches, failures = audit.read_branches(tmp_path, sources)
    assert failures == [] and branches[7788]["retention32"]["independent_n"] == 32
    value = audit.build_artifact(tmp_path, "20260927", sources, failures, branches)
    assert value["independent_evidence_ready_score"] == 1
    assert len(value["audit_rows"]) == 3
    output = tmp_path / "output.json"
    assert audit.main(["--fixture-root", str(tmp_path), "--output", str(output)]) == 0
    assert output.is_file()
    assert audit.main(["--cold", str(output)]) == 0
    assert "cold_replay_errors" in capsys.readouterr().out
    manifest = tmp_path / "frozen_heads_manifest-7786.json"
    manifest.write_text(json.dumps({"head_sha256": "changed"}))
    producer = tmp_path / audit.PLAN[7786]
    changed = json.loads(producer.read_text())
    changed["frozen_heads_manifest_sha256"] = sha256_file(manifest)
    producer.write_text(json.dumps(changed))
    current, _ = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "head_manifest" for f in audit.read_branches(tmp_path, current)[1])
    manifest.write_text(json.dumps({"head_sha256": "head-0"}))
    changed["frozen_heads_manifest_sha256"] = sha256_file(manifest)
    roles = tmp_path / "role_manifest-7786.json"
    roles.write_text(json.dumps({"evaluation64": []}))
    changed["role_manifest_sha256"] = sha256_file(roles)
    producer.write_text(json.dumps(changed))
    current, _ = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "role_manifest" for f in audit.read_branches(tmp_path, current)[1])
    roles.write_text(json.dumps({"evaluation64": [f"evaluation64-{i}" for i in range(64)]}))
    changed["role_manifest_sha256"] = sha256_file(roles)
    rejected = tmp_path / "rejection_records-7786.json"
    rejected.write_text("{}")
    changed["rejection_records_sha256"] = sha256_file(rejected)
    producer.write_text(json.dumps(changed))
    current, _ = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "rejection_records" for f in audit.read_branches(tmp_path, current)[1])
    altered = json.loads(output.read_text())
    altered["audit_rows"] = []
    output.write_text(json.dumps(altered))
    assert audit.main(["--cold", str(output)]) == 1


def test_validation_receipts_and_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7789-TERMINAL: owned failures disqualify readiness."""
    mode = {"fail": False, "bad_json": False, "cached": False}

    def fake_commands(
        root: Path, commands: list[checks.CommandSpec], *, log_dir: Path, **_: object
    ) -> list[dict]:
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for index, command in enumerate(commands):
            assert not (mode["cached"] and command.name == "full_python_suite")
            log = log_dir / f"{index:02d}_{command.name}.log"
            log.write_text(
                "invalid"
                if command.name == "adversarial_verify" and mode["bad_json"]
                else '{"flagged_count": 1}'
                if command.name == "adversarial_verify" and mode["fail"]
                else '{"flagged_count": 0}'
                if command.name == "adversarial_verify"
                else "ok"
            )
            passed = not (mode["fail"] and command.name in {"focused_pytest", "cold_replay"})
            receipts.append(
                {
                    "name": command.name,
                    "passed": passed,
                    "exit_code": 0 if passed else 1,
                    "command_argv": list(command.argv),
                    "log_path": str(log.relative_to(root)),
                    "log_sha256": sha256_file(log),
                }
            )
        return receipts

    monkeypatch.setattr(checks, "run_commands", fake_commands)
    output = tmp_path / "output.json"
    good = audit.run_experiment(tmp_path, "20260927", output)
    assert good["verdict_class"] == "blocked"
    assert len(good["validation_receipts"]["required_commands"]) == 8
    assert len(good["validation_receipts"]["terminal_readers"]) == 3
    assert good["flagged_adversarial"] is False
    assert output.is_file()
    full_log = tmp_path / "full-suite-diagnostic.log"
    full_log.write_text("repository suite reported failures")
    cached = (
        tmp_path
        / "results/raw/experiment_7789_v677_independent_evidence_audit/validation/full/receipt.json"
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps(
            {
                "name": "full_python_suite",
                "passed": False,
                "exit_code": 1,
                "command_argv": [
                    str(tmp_path / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                ],
                "log_path": str(full_log.relative_to(tmp_path)),
                "log_sha256": sha256_file(full_log),
            }
        )
    )
    mode["cached"] = True
    mode["fail"] = True
    bad = audit.run_experiment(tmp_path, "20260927", output)
    assert bad["validation_receipts"]["full_python_suite"][0]["passed"] is False
    assert bad["verdict_class"] == "disqualified"
    assert bad["flagged_adversarial"] is True
    assert bad["acceptance_gate_results"]["validity"] is False
    assert bad["independent_evidence_ready_score"] == 0
    mode["bad_json"] = True
    assert audit.run_experiment(tmp_path, "20260927", output)["flagged_adversarial"] is True
    full_log.write_text("changed")
    damaged = audit.run_experiment(tmp_path, "20260927", output)
    assert (
        damaged["validation_receipts"]["full_python_suite"][0]["error"]
        == "invalid_cached_receipt:ValueError"
    )


def test_remaining_custody_and_reducer_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7789-CUSTODY/ROWS: corrupt bytes lose eligibility."""
    producer = tmp_path / audit.PLAN[7786]
    producer.parent.mkdir(parents=True)
    producer.write_text("[]")
    event_source = tmp_path / audit.PLAN[7788]
    event_source.write_text(
        json.dumps(
            {
                "experiment_id": 7788,
                "milestone": "2026.09.677",
                "run_date": "20260927",
                "flagged_adversarial": False,
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "event_rows_path": "missing.json",
                "event_rows_sha256": "forged",
            }
        )
    )
    _, failures = audit.inspect_sources(tmp_path)
    assert {f["field"] for f in failures} >= {"schema", "event_rows_path", "event_rows_sha256"}
    assert any(f["field"] == "raw_rows_path" for f in failures)
    producer.write_text("not-json")
    assert any(f["field"] == "schema" for f in audit.inspect_sources(tmp_path)[1])
    for bad in (
        {"probability": 2, "label": 0, "action": "accept"},
        {"probability": 0.5, "label": 0, "action": "guess"},
    ):
        with pytest.raises(ValueError):
            audit.score(bad)
    assert audit.paired([])["n"] == 0
    one = rows("evaluation64", 1, audit.DECISION_ARMS)
    one[0].update(role="retention32", probability=2, label=1, source_sha256="wrong")
    monkeypatch.setattr(audit, "paired", lambda *_: {"n": 0, "p": 1})
    reduced = audit.reduce_rows(one, "evaluation64", 2, audit.DECISION_ARMS, "head-0")
    assert {"role", "metric", "family_count", "label_join"} <= set(reduced["failed_checks"])
    events = [
        {"kind": "admission", "feedback_id": "a", "tick": 1},
        {"kind": "admission", "feedback_id": "a", "tick": 2},
    ]
    assert "duplicate_admission" in audit.audit_events(events, [], False)
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    changed = json.loads(event_source.read_text())
    changed.update(raw_rows_path=raw.name, raw_rows_sha256="forged")
    event_source.write_text(json.dumps(changed))
    assert any(f["field"] == "raw_rows_sha256" for f in audit.inspect_sources(tmp_path)[1])
    (tmp_path / "heads.json").write_text(json.dumps({"head_sha256": "head-0"}))
    (tmp_path / "roles.json").write_text(json.dumps({"evaluation64": ["evaluation64-0"]}))
    (tmp_path / "rejections.json").write_text("[]")
    source = {
        "upstream_id": "Exp7786",
        "state": "eligible",
        "raw_paths": {
            "raw_rows_path": {"path": raw.name},
            "frozen_heads_manifest_path": {"path": "heads.json"},
            "role_manifest_path": {"path": "roles.json"},
            "rejection_records_path": {"path": "rejections.json"},
        },
    }
    _, failures = audit.read_branches(tmp_path, [source])
    assert failures[0]["field"] == "raw_schema"
    raw.write_text(json.dumps({"head_sha256": "head-0", "rowsets": {"evaluation64": one}}))
    _, failures = audit.read_branches(tmp_path, [source])
    assert any(f["field"].startswith("evaluation64.") for f in failures)


def test_optional_qwen_rows_recomputed_from_raw(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7789-CUSTODY: Qwen rows inform but do not gate CPU."""
    request = tmp_path / "request.json"
    reply = tmp_path / "reply.json"
    request.write_text('{"messages": []}')
    reply.write_text('{"choices": [{"message": {"content": "{}"}}]}')
    qwen_rows = []
    for family in ("q0", "q1"):
        for arm in ("generic", "event"):
            qwen_rows.append(
                {
                    "family_id": family,
                    "arm": arm,
                    "natural_unsupported": family == "q1",
                    "metrics": {
                        "unsupported_risk": 0.8 if family == "q1" else 0.2,
                        "forced_escalation": False,
                        "valid": True,
                    },
                    "brier": 0.04,
                    "decision_cost": 1 if family == "q1" else 0,
                    "escalated": family == "q1",
                    "false_accept": False,
                    "raw_request_path": str(request),
                    "raw_request_sha256": sha256_file(request),
                    "raw_response_path": str(reply),
                    "raw_response_sha256": sha256_file(reply),
                    "censored": False,
                }
            )
    reduced = audit.reduce_qwen(tmp_path, qwen_rows, expected_families=2)
    assert reduced["failed_checks"] == []
    assert reduced["independent_n"] == 2
    assert reduced["by_arm"]["event"]["brier"] == pytest.approx(0.04)
    assert len(reduced["rows"]) == 4
    missing_sources, missing_failures = audit.inspect_sources(tmp_path)
    optional_only = audit.build_artifact(
        tmp_path, "20260927", missing_sources, missing_failures, {7787: {"qwen_optional": reduced}}
    )
    assert optional_only["sample_size_budget"]["independent_n"] == 0
    assert optional_only["sample_size_budget"]["by_role"]["qwen_optional"] == 2
    bad = deepcopy(qwen_rows)
    bad[0]["brier"] = 1
    bad[1]["raw_response_sha256"] = "forged"
    bad.pop()
    assert {"saved_metric", "raw_hash", "roster"} <= set(
        audit.reduce_qwen(tmp_path, bad, expected_families=2)["failed_checks"]
    )
    bad = deepcopy(qwen_rows)
    bad[0]["metrics"]["unsupported_risk"] = 1.2
    bad[0]["decision_cost"] = 9
    bad[1]["natural_unsupported"] = True
    assert {"risk", "saved_metric", "label_join", "family_count"} <= set(
        audit.reduce_qwen(tmp_path, bad, expected_families=3)["failed_checks"]
    )
    qwen_path = tmp_path / audit.PLAN[7787]
    qwen_path.parent.mkdir(parents=True, exist_ok=True)
    qwen_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp7787-qwen-event-confidence",
                "milestone": "2026.09.677",
                "run_date": "20260927",
                "flagged_adversarial": False,
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "rows": qwen_rows,
            }
        )
    )
    sources, failures = audit.inspect_sources(tmp_path)
    assert {f["upstream_id"] for f in failures} == {"Exp7786", "Exp7788"}
    assert sources[2]["state"] == "eligible"
    branches, _ = audit.read_branches(tmp_path, sources)
    assert branches[7787]["qwen_optional"]["independent_n"] == 2
    assert sources[2]["state"] == "disqualified"
    assert "family_count" in sources[2]["optional_failures"]
    sources[2]["state"] = "eligible"
    qwen_path.write_text("invalid")
    assert audit.read_branches(tmp_path, sources)[0] == {}
    assert sources[2]["optional_failures"] == ["raw_schema:JSONDecodeError"]


def test_private_mutations_use_row_reducer() -> None:
    """SCENARIO-REPORT-7789-ROWS: every named corruption is rejected."""
    outcomes = audit.private_tampers()
    assert {item["mutation"] for item in outcomes} == {
        "dropped_family",
        "swapped_label_join",
        "future_feedback",
        "commit_inside_query",
        "forged_aggregate",
        "missing_arm",
        "duplicate_seed",
        "altered_head_digest",
    }
    assert all(item["rejected"] and item["failed_check"] for item in outcomes)
    sources, failures = audit.inspect_sources(Path("/tmp/exp7789-nonexistent"))
    assert (
        audit.build_artifact(Path("/tmp/exp7789-nonexistent"), "20260927", sources, failures, {})[
            "rejected_mutations"
        ]
        == outcomes
    )
