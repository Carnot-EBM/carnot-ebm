"""REQ-REPORT-7810 and SCENARIO-REPORT-7810-* custody regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot import experiment_7810_v679_source_view_qualification as exp
from carnot.experiment_7727_v673_development_corpus import COUNTS
from carnot.reporting.current_work_receipt import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OLD_RAW = ROOT / "results/raw/experiment_7796_v678_source_view_qualification"
OLD_CANDIDATE = ROOT / "results/experiment_7796_v678_source_view_qualification.json"
MANIFEST = ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"


def test_scenario_report_7810_saved_mismatch_is_preserved() -> None:
    """The repair names the saved one-row candidate and 640 raw rows exactly."""
    report = json.loads((exp.RAW / "exp7796_mismatch_reproduction.json").read_text())
    assert report["saved_candidate_count"] == 1
    assert report["raw_count"] == 640
    assert report["first_discrepancy"]["index"] == 0
    assert report["first_discrepancy"]["candidate"] == 1
    assert (
        report["saved_candidate_ordered_identity_sha256"] != report["raw_ordered_identity_sha256"]
    )


def test_scenario_report_7810_manifest_and_dispatch_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real dispatcher records all children, including appended readers."""
    manifest = exp.load_command_manifest()
    seen: list[tuple[str, list[str], str]] = []

    def recording_child(command: dict, index: int, scope: dict) -> dict:
        seen.append((command["name"], command["argv"], command["classification"]))
        log = tmp_path / f"{index}.log"
        log.write_bytes(command["name"].encode())
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "classification": command["classification"],
            "exit_code": 0,
            "passed": True,
            "log_path": str(log),
            "log_sha256": sha256_file(log),
        }

    monkeypatch.setattr(exp, "execute_child", recording_child)
    observed = exp.dispatch(manifest)
    assert seen == [
        (item["name"], item["argv"], item["classification"]) for item in manifest["commands"]
    ]
    assert len(observed) == len(manifest["commands"])
    assert (Path(manifest["private_root"]) / "basetemp").is_dir()
    assert all("experiment_7782_v677_historical_compatibility.py" not in x[1][2:4] for x in seen)
    assert any(x[0] == "cold_replay" for x in seen)
    assert any(x[0] == "repository_health" and x[2] == "diagnostic" for x in seen)


@pytest.mark.parametrize(
    "name,target",
    [
        ("focused_pytest", "tests/python"),
        ("cold_replay", "experiment_7796_v678_source_view_qualification.py"),
        ("focused_pytest", "-k"),
    ],
)
def test_scenario_report_7810_manifest_rejects_undeclared(name: str, target: str) -> None:
    """Changing a command vector cannot silently narrow or expand a gate."""
    changed = copy.deepcopy(exp.load_command_manifest())
    next(row for row in changed["commands"] if row["name"] == name)["argv"].append(target)
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        exp.validate_command_manifest(changed)


def test_scenario_report_7810_manifest_rejects_appended_child() -> None:
    """The final child is checked as strictly as generated helper commands."""
    changed = copy.deepcopy(exp.load_command_manifest())
    changed["commands"].append(copy.deepcopy(changed["commands"][-1]))
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        exp.validate_command_manifest(changed)


def test_scenario_report_7810_full_roster_and_mutations(tmp_path: Path) -> None:
    """All 640 positions survive, including the 36 abstaining families."""
    original = json.loads(OLD_CANDIDATE.read_text())
    assert original["sample_size_budget"]["independent_n"] == 640
    assert original["sample_size_budget"]["rejected"] == 36
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(original))
    assert exp.check_candidate_roster(MANIFEST, OLD_RAW, path)["families"] == 640
    for mutation in ("deleted", "duplicated", "reordered", "altered"):
        value = copy.deepcopy(original)
        rows = value["rows"]
        if mutation == "deleted":
            rows.pop(0)
        elif mutation == "duplicated":
            rows[1] = copy.deepcopy(rows[0])
        elif mutation == "reordered":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[0]["source_sha256"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="candidate_roster_mismatch"):
            exp.check_candidate_roster(MANIFEST, OLD_RAW, path)
    path.write_text(json.dumps(original))
    child = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/experiments/experiment_7810_v679_source_view_qualification.py",
            "--date",
            "20260928",
            "--cold-replay",
            str(MANIFEST),
            str(OLD_RAW),
            str(path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.splitlines()[-1])["families"] == sum(COUNTS.values())


def test_scenario_report_7810_reader_rejects_raw_and_target_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate checks also reject a changed raw family or target join."""
    inventory = json.loads(MANIFEST.read_text())
    ids = [family for role in COUNTS for family in inventory["roles"][role]["families"]]
    public = [{"family_id": family} for family in ids]
    targets = copy.deepcopy(public)
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"rows": []}')
    monkeypatch.setattr(exp, "_rows", lambda path: public if path.name == "rows.jsonl" else targets)
    public[0]["family_id"] = "changed"
    with pytest.raises(ValueError, match="candidate_roster_mismatch"):
        exp.check_candidate_roster(MANIFEST, tmp_path, candidate)
    public[0]["family_id"] = ids[0]
    targets[0]["family_id"] = "changed"
    with pytest.raises(ValueError, match="candidate_roster_mismatch"):
        exp.check_candidate_roster(MANIFEST, tmp_path, candidate)
    monkeypatch.setattr(exp, "check_candidate_roster", lambda *a: {"families": 640})
    monkeypatch.setattr(exp.views, "replay_corpus", lambda *a: {"families": 639})
    with pytest.raises(ValueError, match="candidate_roster_mismatch"):
        exp.cold_reduce(MANIFEST, tmp_path, candidate)
    monkeypatch.setattr(exp.views, "replay_corpus", lambda *a: {"families": 640})
    assert exp.cold_reduce(MANIFEST, tmp_path, candidate)["families"] == 640


def test_scenario_report_7810_durable_logs_retry_and_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later attempt gets a new path; one changed byte breaks its receipt."""
    command = {
        "name": "probe",
        "argv": [sys.executable, "-c", "print(1)"],
        "classification": "required",
        "timeout_s": 10,
    }

    def fake_run(*args: object, log_dir: Path, **kwargs: object) -> list[dict]:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "00_probe.log"
        path.write_bytes(b"closed log\n")
        return [
            {
                "name": "probe",
                "command_argv": command["argv"],
                "exit_code": 0,
                "passed": True,
                "log_path": str(path),
            }
        ]

    monkeypatch.setattr(exp, "run_commands", fake_run)
    paths = []
    for attempt in ("first", "retry"):
        scope = {
            "private_root": str(tmp_path / attempt / "private"),
            "raw_root": str(tmp_path / attempt / "raw"),
        }
        receipt = exp.execute_child(command, 0, scope)
        paths.append(receipt["log_path"])
        assert Path(receipt["log_path"]).read_bytes() == b"closed log\n"
        exp.validate_log_receipt(receipt)
        with pytest.raises(ValueError, match="validation_log_path_reused"):
            exp.execute_child(command, 0, scope)
    assert paths[0] != paths[1]
    log = Path(paths[1])
    log.write_bytes(b"closed log!\n")
    with pytest.raises(ValueError, match="validation_log_drift"):
        exp.validate_log_receipt(receipt)
    log.unlink()
    with pytest.raises(ValueError, match="validation_log_drift"):
        exp.validate_log_receipt(receipt)


def test_scenario_report_7810_manifest_integrity_and_observed_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a mutated manifest byte or executor reply cannot create a pass."""
    original = exp.load_command_manifest()
    monkeypatch.setattr(exp, "sha256_file", lambda path: "sha256:wrong")
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        exp.load_command_manifest()
    monkeypatch.undo()
    for key in ("duplicate", "required", "diagnostic"):
        changed = copy.deepcopy(original)
        if key == "duplicate":
            changed["commands"].pop()
        elif key == "required":
            changed["commands"][0]["classification"] = "diagnostic"
        else:
            changed["commands"][8]["classification"] = "required"
        monkeypatch.setattr(exp, "load_command_manifest", lambda: changed)
        with pytest.raises(ValueError, match="validation_manifest_drift"):
            exp.validate_command_manifest(changed)
        monkeypatch.undo()
    monkeypatch.setattr(
        exp,
        "execute_child",
        lambda command, index, scope: {
            "name": "wrong",
            "command_argv": command["argv"],
            "classification": command["classification"],
            "log_path": str(tmp_path / "missing"),
            "log_sha256": "sha256:wrong",
        },
    )
    with pytest.raises(ValueError, match="observed_child_command_drift"):
        exp.dispatch(original)


def test_scenario_report_7810_terminal_paths_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocked, failed, and successful paths keep their different readiness."""
    manifest = exp.load_command_manifest()
    scope = {
        **manifest,
        "raw_root": str(tmp_path / "raw"),
        "candidate_path": str(tmp_path / "raw" / "candidate.json"),
    }
    monkeypatch.setattr(exp, "load_command_manifest", lambda: scope)
    monkeypatch.setattr(exp, "validate_command_manifest", lambda value: None)
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(exp, "check_candidate_roster", lambda *args: {"families": 640})
    prepared = {"rows": [{"family_id": "one"}], "targets": []}
    monkeypatch.setattr(exp.views, "prepare_corpus", lambda *args: prepared)

    def fake_base(
        root: Path, checks: list, prep: object, receipts: list, spans: list, **kwargs: object
    ) -> dict:
        blocked = prep is None
        ready = not blocked and len(receipts) == 11 and all(x["passed"] for x in receipts)
        return {
            "rows": [] if blocked else [{"family_id": "one"}],
            "source_artifact_hashes": [],
            "field_principles": {},
            "verdict_class": "blocked"
            if blocked
            else "circular_positive"
            if ready
            else "disqualified",
            "sentence_protocol_ready_score": int(ready),
            "evidence_view_ready_score": int(ready),
            "source_view_manifest_path": "sealed" if ready else None,
        }

    monkeypatch.setattr(exp.prior, "build_candidate", fake_base)
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run_experiment("20260927")
    monkeypatch.setattr(exp.prior, "preconditions", lambda root: [{"passed": False}])
    blocked = exp.run_experiment("20260928")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["observed_child_commands"] == []
    with pytest.raises(ValueError, match="attempt_root_reused"):
        exp.run_experiment("20260928")
    scope["raw_root"] = str(tmp_path / "second")
    scope["candidate_path"] = str(tmp_path / "second" / "candidate.json")
    monkeypatch.setattr(exp.prior, "preconditions", lambda root: [{"passed": True}])
    log = tmp_path / "adversarial.json"
    log.write_text('{"flagged_count": 0}')
    receipts = [
        {
            "name": item["name"],
            "command_argv": item["argv"],
            "classification": item["classification"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": sha256_file(log),
        }
        for item in scope["commands"]
    ]
    monkeypatch.setattr(exp, "dispatch", lambda value: receipts)
    ready = exp.run_experiment("20260928")
    assert ready["sentence_protocol_ready_score"] == 1
    assert ready["evidence_view_ready_score"] == 1
    assert ready["repository_health"]["broad_suite_classification"] == "diagnostic"
    assert len(ready["observed_child_commands"]) == 12
    scope["raw_root"] = str(tmp_path / "third")
    scope["candidate_path"] = str(tmp_path / "third" / "candidate.json")
    log.write_text("invalid json")
    receipts[0]["passed"] = False
    failed = exp.run_experiment("20260928")
    assert failed["sentence_protocol_ready_score"] == 0
    assert failed["evidence_view_ready_score"] == 0
    assert failed["source_view_manifest_path"] is None
    monkeypatch.setattr(exp, "run_experiment", lambda date: {"verdict_class": "disqualified"})
    assert exp.main(["--date", "20260928"]) == 1
    monkeypatch.setattr(exp, "cold_reduce", lambda *args: {"families": 640})
    assert exp.main(["--date", "20260928", "--cold-replay", "a", "b", "c"]) == 0
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260927", "--cold-replay", "a", "b", "c"])
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as wrapped:
        runpy.run_path(
            str(ROOT / "scripts/experiments/experiment_7810_v679_source_view_qualification.py"),
            run_name="__main__",
        )
    assert wrapped.value.code == 0
