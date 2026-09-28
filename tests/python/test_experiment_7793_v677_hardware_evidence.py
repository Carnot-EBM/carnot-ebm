"""REQ-REPORT-7793: prospective hardware accounting from immutable sources."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from carnot.reporting.experiment_7793_v677_hardware_evidence import (
    _git_blob,
    build_candidate,
    cold_reduce,
    main,
)


ROOT = Path(__file__).resolve().parents[2]


def test_authentic_board_accounting(capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7793-CUSTODY: old board scope survives a bad old gate."""
    candidate = build_candidate(ROOT, "20260927")
    progress = capsys.readouterr().err
    assert progress.count("phase=git_blob event=before_subprocess") == 3
    assert progress.count("phase=git_blob event=after_subprocess") == 3
    assert candidate["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert candidate["verdict_class"] == "blocked"
    assert candidate["hardware_continuity_accounted"] is True
    assert candidate["service_measured"] is False
    assert candidate["acceptance_gate_results"]["efficiency"] is None
    assert (
        candidate["source_artifact_hashes"]["results/experiment_7779_v676_hardware_evidence.json"][
            "eligible"
        ]
        is False
    )
    assert candidate["preconditions_checked"]["historical_exp7779_full_suite_exit"] == 2
    boards = {row["board"]: row for row in candidate["board_rows"]}
    assert boards["KV260"]["k_max"] == 5
    assert boards["PolarFire"]["processor_class"] == "linux_cpu"
    assert boards["GateMate"]["blocker"] == "0xffffffff"
    assert boards["GateMate"]["state"].startswith("blocked_")
    assert boards["GateMate"]["excluded"] is True
    assert candidate["sample_size_budget"]["eligible"] == 2
    assert candidate["sample_size_budget"]["effective_independent_n"] == 2
    assert all(row["service_fraction"] is None for row in candidate["board_rows"])
    assert any(check["upstream_id"] == "Exp7792" for check in candidate["gate_check_summary"])
    assert candidate["evidence_unresolved"] == []
    assert candidate["claim_scope"]["hardware_advantage"] == "unmeasured"
    assert (
        candidate["preconditions_checked"]["conductor_pre_gate_receipt"]["is_science_producer"]
        is False
    )
    assert all(
        check["passed"]
        for check in candidate["preconditions_checked"]["custody_checks"]
        if check["field"] == "inventory_raw_evidence_sha256"
    )
    assert "field_principles" in candidate["field_principles"]


def test_cold_replay_on_authentic_sources(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7793-REPLAY: current rows replay from source bytes."""
    candidate = build_candidate(ROOT, "20260927")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    replay = cold_reduce(path, ROOT)
    assert replay["row_count"] == len(candidate["rows"])
    candidate["rows"][0]["state"] = "fabric_speedup"
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="row_mismatch"):
        cold_reduce(path, ROOT)


def test_private_changed_digest_and_missing_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7793-REPLAY: private source changes cannot qualify."""
    candidate = build_candidate(ROOT, "20260927")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    private = tmp_path / "private"
    for rel in candidate["source_artifact_hashes"]:
        source = ROOT / rel
        if source.is_file():
            target = private / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # Recreate only the immutable Git object needed for historical literature.
    blob = "7766c92e07b4cee74ece8d9e0ddfbea14fb69173"
    old_bytes = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", blob])
    subprocess.run(["git", "init", "-q", str(private)], check=True)
    stored = subprocess.run(
        ["git", "-C", str(private), "hash-object", "-w", "--stdin"],
        input=old_bytes,
        capture_output=True,
        check=True,
    )
    assert stored.stdout.decode().strip() == blob
    assert cold_reduce(path, private)["row_count"] == len(candidate["rows"])
    board = private / "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"
    board.write_bytes(board.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(path, private)
    shutil.copy2(ROOT / board.relative_to(private), board)
    board.unlink()
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        cold_reduce(path, private)


def test_private_service_and_unresolved_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7793-CUSTODY: only an eligible producer supplies fractions."""
    private = tmp_path / "private"
    for rel in (
        "results/experiment_7751_v674_hardware_continuity.json",
        "results/raw/experiment_7751_v674_hardware_continuity/rows.json",
        "results/experiment_7779_v676_hardware_evidence.json",
        "results/raw/experiment_7779_v676_hardware_evidence/rows.json",
        "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json",
        "results/experiment_7231_v636_board_continuity.json",
        "results/experiment_6559_gatemate_changed_state_continuity.json",
    ):
        target = private / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    blob = "7766c92e07b4cee74ece8d9e0ddfbea14fb69173"
    old_bytes = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", blob])
    subprocess.run(["git", "init", "-q", str(private)], check=True)
    subprocess.run(
        ["git", "-C", str(private), "hash-object", "-w", "--stdin"],
        input=old_bytes,
        check=True,
        capture_output=True,
    )
    service = private / "results/experiment_7792_v677_service_cost.json"
    service.write_text(
        json.dumps(
            {
                "experiment_id": 7792,
                "service_evidence_ready_score": 1,
                "flagged_adversarial": False,
                "verdict_class": "null",
                "service_fractions": {"KV260": 0.2},
            }
        )
    )
    candidate = build_candidate(private, "20260927")
    assert candidate["service_measured"] is True
    assert candidate["board_rows"][0]["service_fraction"] == 0.2
    assert candidate["honest_verdict"] == "complete_null_hardware_accounting_only"
    service.unlink()
    (private / "results/experiment_7231_v636_board_continuity.json").unlink()
    candidate = build_candidate(private, "20260927")
    assert candidate["board_rows"][1]["state"] == "evidence_unresolved"
    assert candidate["evidence_unresolved"][0]["observed_hash"] is None
    assert candidate["honest_verdict"] == "complete_blocked_historical_evidence"


def test_replay_rejects_summary_mutations_and_bad_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7793-REPLAY: every published summary is byte-bound."""
    original = build_candidate(ROOT, "20260927")
    path = tmp_path / "candidate.json"
    cases = [
        ("source_artifact_hashes", "source_set_mismatch"),
        ("board_rows", "board_row_mismatch"),
        ("evidence_unresolved", "summary_mismatch"),
        ("gate_check_summary", "gate_mismatch"),
    ]
    for field, expected in cases:
        changed = json.loads(json.dumps(original))
        if field == "source_artifact_hashes":
            changed[field]["phantom"] = {"sha256": None, "eligible": None}
        elif field == "board_rows":
            changed[field] = []
        elif field == "evidence_unresolved":
            changed[field] = [{"board": "phantom"}]
        else:
            changed[field] = []
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=expected):
            cold_reduce(path, ROOT)
    assert _git_blob(ROOT, None) is None

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("git", 1)

    monkeypatch.setattr(
        "carnot.reporting.experiment_7793_v677_hardware_evidence.subprocess.run", timeout
    )
    assert _git_blob(ROOT, "bad") is None


def test_publish_guard_rejects_invalid_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7793-TERMINAL: failed owned checks cannot publish."""
    candidate = build_candidate(ROOT, "20260927")
    path = tmp_path / "candidate.json"
    monkeypatch.setenv("CARNOT_7793_CANDIDATE", str(path))
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="validated_candidate_required"):
        main(["--root", str(ROOT), "--date", "20260927"])
    candidate["validation_receipts"]["required_checks_passed"] = True
    candidate["validation_receipts"]["checks"] = [{"exit_code": 1}]
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="failed_validation_cannot_publish"):
        main(["--root", str(ROOT), "--date", "20260927"])


def test_cli_candidate_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7793-TERMINAL: CLI exposes private candidate replay."""
    target = tmp_path / "candidate.json"
    assert main(["--root", str(ROOT), "--date", "20260927", "--prepare", str(target)]) == 0
    assert target.is_file()
    assert main(["--root", str(ROOT), "--cold-replay", str(target)]) == 0


def test_cli_publishes_only_replayable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7793-TERMINAL: validated publish rechecks exact rows."""
    candidate = build_candidate(ROOT, "20260927")
    candidate["validation_receipts"]["required_checks_passed"] = True
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    monkeypatch.setenv("CARNOT_7793_CANDIDATE", str(path))
    published: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        "carnot.reporting.experiment_7793_v677_hardware_evidence.atomic_json",
        lambda output, value: published.append((output, value)),
    )
    assert main(["--root", str(ROOT), "--date", "20260927"]) == 0
    assert published[0][0] == ROOT / "results/experiment_7793_v677_hardware_evidence.json"
    assert published[0][1]["rows"] == candidate["rows"]
