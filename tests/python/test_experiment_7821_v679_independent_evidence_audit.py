"""REQ-REPORT-7821: independent current science and dispatch evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import pytest

from carnot import experiment_7821_v679_independent_evidence_audit as audit
from carnot.reporting.current_work_receipt import sha256_file
from scripts.experiments import experiment_7821_v679_independent_evidence_audit as cli


def _row(family: str = "f0") -> dict:
    return {
        "family_id": family,
        "seed": 67815,
        "arm": "candidate",
        "role": "evaluation64",
        "label": 1,
        "label_origin": "independent_annotation",
        "label_join": family,
        "feature_names": ["public_source_length"],
        "source_sha256": "sha256:" + hashlib.sha256(b"A. B.").hexdigest(),
        "answer_sha256": "sha256:" + hashlib.sha256(b"First. Second.").hexdigest(),
        "prediction_tick": 1,
        "label_tick": 2,
        "probability": 0.8,
        "action": "escalate",
        "brier": 0.04,
        "source_bytes": "A. B.",
        "answer_bytes": "First. Second.",
        "target_sentence_span": [0, 6],
        "target_sentence": "First.",
        "annotation_sentence_span": [0, 6],
        "checkpoint_parameters": [0.1, 0.2],
    }


def test_current_missing_receipt_is_explanation_only(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-CUSTODY: a conductor file is not science."""
    receipt = tmp_path / audit.PRE_GATE[7815]
    receipt.parent.mkdir(parents=True)
    historical = Path("results/experiment_7815_qwen_counter_evidence.json")
    receipt.write_bytes(historical.read_bytes())
    sources, failures = audit.inspect_sources(tmp_path)
    assert [s["state"] for s in sources] == ["missing"] * 3
    assert sources[1]["pre_gate_receipt"]["sha256"] == sha256_file(receipt)
    assert sources[1]["pre_gate_receipt"]["role"] == "explanation_only"
    assert {f["field"] for f in failures} >= {"producer_path", "counter_evidence_ready_score"}
    result = audit.build_artifact(tmp_path, "20260928", sources, failures, {})
    assert result["honest_verdict"] == "complete_blocked_required_v679_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["independent_evidence_ready_score"] == 0
    assert len(result["branch_dispositions"]) == 3
    assert result["aligned_label_count"] == 0


def test_valid_null_and_malformed_source_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-CUSTODY: a valid null differs from malformed bytes."""
    raw = tmp_path / "raw.json"
    row = _row()
    raw.write_text(json.dumps({"rowsets": {"evaluation64": [row]}}))
    producer = tmp_path / audit.PLAN[7813]
    producer.parent.mkdir(parents=True)
    producer.write_text(
        json.dumps(
            {
                "experiment_id": 7813,
                "milestone": "2026.09.679",
                "run_date": "20260928",
                "flagged_adversarial": False,
                "verdict_class": "null",
                "honest_verdict": "complete_null_valid",
                "raw_rows_path": "raw.json",
                "raw_rows_sha256": sha256_file(raw),
            }
        )
    )
    malformed = tmp_path / audit.PLAN[7816]
    malformed.write_text("[]")
    sources, failures = audit.inspect_sources(tmp_path)
    assert [s["state"] for s in sources] == ["eligible", "missing", "disqualified"]
    branches, raw_failures = audit.read_branches(tmp_path, sources)
    assert raw_failures == []
    assert branches[7813]["evaluation64"]["rows"][0]["metrics"]["brier"] == 0.04
    result = audit.build_artifact(tmp_path, "20260928", sources, failures, branches)
    assert any(r.get("family_id") == "f0" for r in result["rows"])
    assert result["aligned_label_count"] == 1
    assert {f["field"] for f in failures} >= {"schema", "producer_path"}


def test_raw_mutations_and_event_alignment() -> None:
    """SCENARIO-REPORT-7821-RAW: private defects fail the primitive reader."""
    row = _row()
    assert audit.reduce_rows([row], 1)["failed_checks"] == []
    changes = (
        ("feature_names", ["private_label"], "feature_leakage"),
        ("feature_names", ["gold_confidence"], "feature_leakage"),
        ("label_origin", "self_label", "self_label"),
        ("probability", 0.1, "saved_metric"),
        ("prediction_tick", 3, "future_feedback"),
        ("annotation_sentence_span", [7, 14], "sentence_label_join"),
    )
    for field, value, expected in changes:
        bad = deepcopy(row)
        bad[field] = value
        assert expected in audit.reduce_rows([bad], 1)["failed_checks"]
    assert "family_roster" in audit.reduce_rows([], 1)["failed_checks"]
    assert "seed_as_sample" in audit.check_interval_units([row], 3)


def test_peak_error_debt_rejects_final_net() -> None:
    """SCENARIO-REPORT-7821-RAW: a later recovery does not erase peak debt."""
    debt = audit.worst_window([1, 1, -1, -1])
    assert debt["peak"] == debt["running_peak"] == 2
    assert debt["final_net"] == 0
    assert debt["window"] == [0, 1]
    assert "peak_debt" in audit.check_debt_claim([1, 1, -1, -1], 0)


def test_qwen_first_sentence_and_actual_token_receipts() -> None:
    """SCENARIO-REPORT-7821-RAW: labels and token counts match the elicited event."""
    row = _row()
    row.update(
        arm="intact",
        label=1,
        unsupported_probability=0.8,
        witness_token_count=4,
        control_token_count=5,
        token_counter="gguf",
        removed_source_label=None,
    )
    assert audit.reduce_rows([row], 1)["failed_checks"] == []
    bad = deepcopy(row)
    bad["token_counter"] = "word_count"
    assert "gguf_token_receipt" in audit.reduce_rows([bad], 1)["failed_checks"]
    bad = deepcopy(row)
    bad["arm"] = "witness_removed"
    bad["removed_source_label"] = 1
    assert "modified_source_label" in audit.reduce_rows([bad], 1)["failed_checks"]


def test_qwen_pairs_and_feedback_peak_are_family_units() -> None:
    """SCENARIO-REPORT-7821-RAW: pairs and error bursts use one family once."""
    pairs = []
    for family in ("a", "b"):
        for arm, probability in (
            ("intact", 0.2),
            ("witness_removed", 0.7),
            ("control_removed", 0.3),
        ):
            row = _row(family)
            row.update(
                arm=arm,
                unsupported_probability=probability,
                witness_token_count=4,
                control_token_count=5,
                token_counter="gguf",
                removed_source_label=None,
                label=None if arm != "intact" else 1,
            )
            row.pop("brier")
            pairs.append(row)
    paired = audit.reduce_qwen_pairs(pairs, 2)
    assert paired["independent_n"] == 2
    assert paired["mean_shift"] == 0.4
    assert paired["aligned_label_count"] == 2
    assert all(row["modified_source_label"] is None for row in paired["rows"])
    bad = deepcopy(pairs)
    bad.pop()
    assert "paired_roster" in audit.reduce_qwen_pairs(bad, 2)["failed_checks"]
    events = [
        {"kind": "prediction", "arm": "adaptive", "family_id": "a", "tick": 1},
        {"kind": "feedback", "arm": "adaptive", "family_id": "a", "tick": 2},
        {"kind": "admission", "arm": "adaptive", "family_id": "a", "tick": 3},
        {"kind": "restart", "tick": 4, "queued": [["adaptive", "a"]]},
        {"kind": "commit", "arm": "adaptive", "family_id": "a", "tick": 5},
    ]
    assert audit.check_events(events) == []
    bad_events = deepcopy(events)
    bad_events[3]["queued"] = []
    assert "restart_queue_mismatch" in audit.check_events(bad_events)
    assert audit.reduce_feedback_debt([1, 1, -1, -1], 0)["failed_checks"] == ["peak_debt"]


def test_manifest_rejects_appended_child_and_retry_seals_new_log(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: exact commands and later logs stay separate."""
    root = Path(__file__).resolve().parents[2]
    manifest = cli.load_manifest(root)
    copied = tmp_path / audit.MANIFEST
    copied.parent.mkdir(parents=True)
    changed = deepcopy(manifest)
    changed["commands"].append(
        {
            "name": "hidden_child",
            "argv": ["true"],
            "classification": "required",
            "private_root": "/tmp/hidden",
        }
    )
    copied.write_text(json.dumps(changed))
    try:
        cli.load_manifest(tmp_path)
    except ValueError as exc:
        assert "frozen_manifest_hash_changed" in str(exc)
    else:
        raise AssertionError("appended child was accepted")
    one = {"attempt_id": "first", "commands": [deepcopy(manifest["commands"][0])]}
    two = deepcopy(one)
    two["attempt_id"] = "retry"
    one["commands"][0]["private_root"] = str(tmp_path / "first-private")
    two["commands"][0]["private_root"] = str(tmp_path / "retry-private")
    for plan in (one, two):
        plan["commands"][0]["argv"][-1] = (
            "--basetemp=" + plan["commands"][0]["private_root"] + "/missing/child"
        )

    def record(item: dict, private: Path) -> tuple[int, bytes]:
        assert Path(item["argv"][-1].split("=", 1)[1]).parent.is_dir()
        return 0, b"closed log bytes"

    first = cli.dispatch(root, one, record, tmp_path / "sealed", 0.0)[0]
    retry = cli.dispatch(root, two, record, tmp_path / "sealed", 0.0)[0]
    assert first["log_path"] != retry["log_path"]
    assert Path(first["log_path"]).read_bytes() == Path(retry["log_path"]).read_bytes()


def test_custody_rejects_bad_fields_and_receipt_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-CUSTODY: each failed field keeps its own operand."""
    receipt = tmp_path / audit.PRE_GATE[7815]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("not json")
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    path = tmp_path / audit.PLAN[7813]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": 99,
                "milestone": "old",
                "run_date": "old",
                "flagged_adversarial": True,
                "verdict_class": "blocked",
                "honest_verdict": "unfinished",
                "raw_rows_path": "raw.json",
                "raw_rows_sha256": "wrong",
            }
        )
    )
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert {f["field"] for f in failures} >= {
        "experiment_id",
        "milestone",
        "run_date",
        "flagged_adversarial",
        "verdict_class",
        "honest_verdict",
        "raw_rows_sha256",
        "pre_gate_schema",
    }
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7813,
                "milestone": "2026.09.679",
                "run_date": "20260928",
                "flagged_adversarial": False,
                "verdict_class": "null",
                "honest_verdict": "complete_null",
                "raw_rows_path": "../unsafe.json",
            }
        )
    )
    _, failures = audit.inspect_sources(tmp_path)
    assert "raw_rows_path" in {f["field"] for f in failures}


def test_row_defects_and_clock_rejections() -> None:
    """SCENARIO-REPORT-7821-RAW: reject every primitive custody defect."""
    row = _row()
    for field, value, expected in (
        ("label_join", "other", "label_join"),
        ("source_sha256", "wrong", "source_answer_bytes"),
        ("checkpoint_parameters", [float("nan")], "checkpoint_parameters"),
        ("target_sentence_span", [1, 6], "target_sentence_span"),
        ("probability", None, "primitive_schema"),
    ):
        bad = deepcopy(row)
        bad[field] = value
        assert expected in audit.reduce_rows([bad], 1)["failed_checks"]
    bad = deepcopy(row)
    bad.update(arm="intact", token_counter="gguf", witness_token_count=4, control_token_count=9)
    assert "control_token_match" in audit.reduce_rows([bad], 1)["failed_checks"]
    events = [
        {"kind": "feedback", "arm": "a", "family_id": "f", "tick": 1},
        {"kind": "admission", "arm": "a", "family_id": "f", "tick": 2},
        {"kind": "admission", "arm": "a", "family_id": "f", "tick": 3},
        {"kind": "commit", "arm": "a", "family_id": "other", "tick": 4},
        {"kind": "shuffle", "arm": "a", "source_arm": "b", "tick": 5},
    ]
    assert {
        "future_feedback",
        "duplicate_or_early_admission",
        "unreleased_commit",
        "cross_arm_shuffle",
    } <= set(audit.check_events(events))


def test_qwen_empty_duplicate_and_holm() -> None:
    """SCENARIO-REPORT-7821-RAW: empty and duplicated families stay visible."""
    assert audit.reduce_qwen_pairs([], 0)["paired_interval95"] is None
    row = _row()
    row.update(
        arm="intact",
        unsupported_probability=0.5,
        witness_token_count=2,
        control_token_count=2,
        token_counter="gguf",
    )
    row.pop("brier")
    assert "paired_roster" in audit.reduce_qwen_pairs([row, row], 1)["failed_checks"]
    assert audit.holm_adjust([0.01, 0.03, 0.2]) == [0.03, 0.06, 0.2]


def test_bad_raw_branch_survives_other_sources(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-CUSTODY: a malformed raw branch stays disqualified."""
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"rowsets": []}))
    source = {
        "upstream_id": "Exp7816",
        "state": "eligible",
        "eligibility": True,
        "raw_paths": {"raw_rows_path": {"path": "raw.json"}},
    }
    branches, failures = audit.read_branches(tmp_path, [source])
    assert branches == {}
    assert source["state"] == "disqualified"
    assert failures[0]["field"] == "raw_reduction"
    raw.write_text(
        json.dumps(
            {
                "rowsets": {"evaluation64": [_row()]},
                "events": [{"kind": "commit", "arm": "x", "family_id": "x", "tick": 1}],
            }
        )
    )
    source["state"] = "eligible"
    _, failures = audit.read_branches(tmp_path, [source])
    assert "unreleased_commit" in failures[0]["observed"]


def test_dispatch_rejects_log_reuse_and_undeclared_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: sealed bytes and names cannot drift."""
    item = {
        "name": "affected_pytest",
        "argv": ["true"],
        "classification": "required",
        "private_root": str(tmp_path / "private"),
    }
    plan = {"attempt_id": "a", "commands": [item]}
    digest = hashlib.sha256(b"good").hexdigest()
    sealed = tmp_path / "sealed/a/affected_pytest" / (digest + ".log")
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"bad")
    with pytest.raises(ValueError, match="sealed_log_changed"):
        cli.dispatch(
            tmp_path,
            plan,
            lambda _item, _private: (0, b"good"),
            tmp_path / "sealed",
            time.monotonic(),
        )
    sealed.unlink()
    sealed.with_suffix(".pending").write_bytes(b"stale")
    with pytest.raises(ValueError, match="pending_log_reused"):
        cli.dispatch(
            tmp_path,
            plan,
            lambda _item, _private: (0, b"good"),
            tmp_path / "sealed",
            time.monotonic(),
        )
    item["name"] = "undeclared"
    with pytest.raises(ValueError, match="undeclared_child"):
        cli.dispatch(
            tmp_path,
            plan,
            lambda _item, _private: (0, b"good"),
            tmp_path / "sealed",
            time.monotonic(),
        )


def test_owned_child_and_cli_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: subprocesses exit and CLI cold reads."""
    private = tmp_path / "child"
    private.mkdir()
    code, output = cli.execute_child(
        {"name": "owned", "argv": [sys.executable, "-c", "print('done')"], "timeout_s": 5}, private
    )
    assert code == 0 and output == b"done\n"
    code, _ = cli.execute_child(
        {
            "name": "owned",
            "argv": [sys.executable, "-c", "import time; time.sleep(1)"],
            "timeout_s": 0.01,
        },
        private,
    )
    assert code == 124
    sources, failures = audit.inspect_sources(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(audit.build_artifact(tmp_path, "20260928", sources, failures, {}))
    )
    assert cli.main(["--cold", str(candidate)]) == 0
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda *_: {"honest_verdict": "complete_blocked_fixture", "verdict_class": "blocked"},
    )
    assert cli.main(["--date", "20260928"]) == 0


def test_required_failure_disqualifies_without_hiding_health(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: owned failures zero readiness."""
    root = Path(__file__).resolve().parents[2]

    def fail(item: dict, private: Path) -> tuple[int, bytes]:
        return (
            1 if item["name"] in ("ruff_check", "repository_health") else 0,
            b'{"flagged_count": 0}\n',
        )

    result = cli.run_experiment(root, "20260928", tmp_path / "failed.json", fail, tmp_path / "logs")
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["repository_health"]["exit_code"] == 1


def test_remaining_raw_and_receipt_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7821-RAW: saved debt and relative bytes replay exactly."""
    receipt = tmp_path / audit.PRE_GATE[7815]
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "gates_evaluated": [
                    {
                        "passed": False,
                        "artifact_path": "raw.json",
                        "artifact_field": "ready",
                        "expected": 1,
                        "actual": 0,
                        "op": "==",
                        "upstream": "fixture",
                    }
                ]
            }
        )
    )
    _, failures = audit.inspect_sources(tmp_path)
    assert any(
        f["field"] == "ready" and f["artifact_path"] == str(tmp_path / "raw.json") for f in failures
    )
    raw = tmp_path / "raw.json"
    row = _row()
    bad = deepcopy(row)
    bad["label_join"] = "wrong"
    raw.write_text(json.dumps({"rowsets": {"evaluation64": [bad]}}))
    source = {
        "upstream_id": "Exp7816",
        "path": audit.PLAN[7816],
        "sha256": None,
        "state": "eligible",
        "eligibility": True,
        "raw_paths": {"raw_rows_path": {"path": "raw.json"}},
    }
    _, failures = audit.read_branches(tmp_path, [source])
    assert "label_join" in failures[0]["observed"]
    source["state"] = "eligible"
    raw.write_text(
        json.dumps(
            {
                "rowsets": {"evaluation64": [row]},
                "feedback_debt": {"adaptive": {"g_t": [1, 1, -1, -1], "peak": 2}},
            }
        )
    )
    branches, failures = audit.read_branches(tmp_path, [source])
    assert failures == [] and branches[7816]["adaptive"]["peak"] == 2
    artifact = audit.build_artifact(tmp_path, "20260928", [source], [], branches)
    assert any(
        r.get("metrics", {}).get("peak_debt") == 2
        for r in artifact["rows"]
        if isinstance(r.get("metrics"), dict)
    )
    source["state"] = "eligible"
    raw.write_text(
        json.dumps(
            {
                "rowsets": {"evaluation64": [row]},
                "feedback_debt": {"adaptive": {"g_t": [1, 1, -1, -1], "peak": 0}},
            }
        )
    )
    _, failures = audit.read_branches(tmp_path, [source])
    assert "peak_debt" in failures[0]["observed"]
    log = tmp_path / "relative.log"
    log.write_bytes(b"ok")
    candidate = tmp_path / "candidate.json"
    sources, failures = audit.inspect_sources(tmp_path)
    candidate.write_text(
        json.dumps(
            {
                **audit.build_artifact(tmp_path, "20260928", sources, failures, {}),
                "observed_child_commands": [
                    {"log_path": "relative.log", "log_sha256": sha256_file(log)}
                ],
            }
        )
    )
    assert audit.cold_replay(candidate) == []


def test_qwen_modified_label_and_cli_manifest_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: modified labels and command drift fail."""
    rows = []
    for arm in ("intact", "witness_removed", "control_removed"):
        row = _row()
        row.update(
            arm=arm,
            unsupported_probability=0.5,
            witness_token_count=2,
            control_token_count=2,
            token_counter="gguf",
        )
        row.pop("brier")
        rows.append(row)
    assert "modified_source_label" in audit.reduce_qwen_pairs(rows, 1)["failed_checks"]
    root = Path(__file__).resolve().parents[2]
    manifest = cli.load_manifest(root)
    copied = tmp_path / audit.MANIFEST
    copied.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "sha256_file", lambda _path: cli.FROZEN_MANIFEST_SHA256)
    for change, expected in (
        (lambda d: d["commands"].append(deepcopy(d["commands"][0])), "undeclared_child"),
        (lambda d: d["commands"][0].update(classification="diagnostic"), "command_class_changed"),
        (
            lambda d: d["commands"][1].update(private_root=d["commands"][0]["private_root"]),
            "command_root_reused",
        ),
    ):
        changed = deepcopy(manifest)
        change(changed)
        copied.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=expected):
            cli.load_manifest(tmp_path)


def test_invalid_terminal_report_and_script_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: malformed terminal output disqualifies."""
    root = Path(__file__).resolve().parents[2]
    result = cli.run_experiment(
        root,
        "20260928",
        tmp_path / "bad.json",
        lambda _item, _private: (0, b"not json"),
        tmp_path / "logs",
    )
    assert result["flagged_adversarial"] is True
    assert result["verdict_class"] == "disqualified"
    candidate = tmp_path / "candidate.json"
    sources, failures = audit.inspect_sources(tmp_path)
    candidate.write_text(
        json.dumps(audit.build_artifact(tmp_path, "20260928", sources, failures, {}))
    )
    monkeypatch.setattr(sys, "argv", ["experiment_7821.py", "--cold", str(candidate)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(
            str(root / "scripts/experiments/experiment_7821_v679_independent_evidence_audit.py"),
            run_name="__main__",
        )
    assert exited.value.code == 0


def test_real_cli_e2e(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7821-DISPATCH: the real CLI dispatches the frozen list."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / audit.MANIFEST).read_bytes())
    seen = []

    def record(item: dict, private: Path) -> tuple[int, bytes]:
        seen.append((item["name"], item["argv"], item["classification"]))
        assert private.parent.is_dir()
        return (1 if item["name"] == "repository_health" else 0, b'{"flagged_count": 0}\n')

    output = tmp_path / "result.json"
    result = cli.run_experiment(root, "20260928", output, record, tmp_path / "logs")
    assert seen == [(x["name"], x["argv"], x["classification"]) for x in manifest["commands"]]
    assert result["repository_health"]["exit_code"] == 1
    assert result["verdict_class"] == "blocked"
    assert output.is_file()
    assert audit.cold_replay(output) == []
    log = root / result["observed_child_commands"][0]["log_path"]
    if not log.is_file():
        log = Path(result["observed_child_commands"][0]["log_path"])
    log.write_bytes(log.read_bytes() + b"x")
    assert "validation_log_changed" in audit.cold_replay(output)
