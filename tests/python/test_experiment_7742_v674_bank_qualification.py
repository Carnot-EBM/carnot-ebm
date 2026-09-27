"""REQ-REPORT-7742 and REQ-CL-7742-BANK fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot import experiment_7742_v674_bank_qualification as exp


def test_7742_preflight_closes_missing_and_malformed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7742-CUSTODY: absent protocol bytes block before work."""
    checks, hashes, predicates = exp.preflight(tmp_path)
    assert predicates == []
    assert any(not row["passed"] and row["field"] == "exists" for row in checks)
    assert hashes["missing_inputs"]
    artifact = exp.run_experiment(tmp_path, "20260927", tmp_path / "blocked.json", validate=False)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["gate_check_summary"]
    assert artifact["acquisition_protocol_ready_score"] == 0


@pytest.mark.parametrize("bad", ["{", "[]", '{"advisory_dictionary": [1]}'])
def test_7742_malformed_manifest_is_a_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """SCENARIO-REPORT-7742-CUSTODY: malformed predicate bytes fail closed."""
    monkeypatch.setattr(exp, "MANIFEST", exp.ROOT / "private-manifest.json")
    (tmp_path / "private-manifest.json").write_text(bad)
    checks, _, names = exp.preflight(tmp_path)
    assert names == []
    assert any(not row["passed"] and row["field"] == "advisory_dictionary_count" for row in checks)


def test_7742_static_fits_all_declared_predicates() -> None:
    """SCENARIO-CL-7742-STATIC: every name is fitted on development fixtures."""
    from carnot.reporting.acquisition_qualification import fixture_groups

    names = json.loads(exp.MANIFEST.read_text())["advisory_dictionary"]
    fit = exp.fit_sentence_static(fixture_groups()["development"], names)
    assert len(fit["weights"]) == 16
    assert set(fit["weights"]) == set(names)
    assert fit["nonzero_weight_count"] == 16
    assert fit["fit_role"] == "development_only"
    assert all(
        set(exp.sentence_features(case, names)) == set(names)
        for case in fixture_groups()["development"]
    )
    with pytest.raises(ValueError, match="sixteen_predicates_required"):
        exp.sentence_features(fixture_groups()["development"][0], names[:-1])
    with pytest.raises(ValueError, match="static_fit_scope_invalid"):
        exp.fit_sentence_static(fixture_groups()["development"][:-1], names)


@pytest.fixture(scope="module")
def private_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """Run the production bank once; retain its private raw evidence."""
    folder = tmp_path_factory.mktemp("exp7742")
    artifact = exp.run_experiment(
        exp.ROOT, "20260927", folder / "candidate.json", raw=folder / "raw", validate=False
    )
    return artifact, folder / "candidate.json"


def test_7742_lifecycle_and_cold_replay(private_run: tuple[dict, Path]) -> None:
    """SCENARIO-CL-7742-LIFECYCLE: durable lifecycle and exact identifiers."""
    artifact, path = private_run
    assert artifact["verdict_class"] == "circular_positive"
    assert exp.cold_reduce(path)["valid"] is True
    assert all(row["passed"] for row in artifact["lifecycle_rows"])
    assert artifact["bank_protocol_path"]["pending_capacity"] == 12
    assert artifact["bank_protocol_path"]["proposal_limit"] == 8
    assert artifact["static_closure_complete"]["fit_role"] == "development_only"
    assert len(artifact["static_closure_complete"]["weights"]) == 16
    assert artifact["static_closure_complete"]["nonzero_weight_count"] == 16
    assert len({row["feedback_id"] for row in artifact["rows"]}) == len(artifact["rows"])
    assert artifact["acquisition_protocol_ready_score"] == 0
    assert (
        "results/experiment_7732_v673_causal_admission.json"
        in artifact["source_artifact_hashes"]["historical_disqualified_sources"]
    )


def test_7742_reader_rejects_changed_hash_summary_and_bank(
    private_run: tuple[dict, Path], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7742-REPLAY: independent reader rejects tampering."""
    artifact, _ = private_run
    for mutation in (
        "input_hash",
        "upstream_hash",
        "summary",
        "unknown_family",
        "raw_missing",
        "legacy_missing",
        "bank",
        "bank_malformed",
        "bank_hash_mismatch",
    ):
        changed = json.loads(json.dumps(artifact))
        if mutation == "input_hash":
            changed["rows"][0]["input_hash"] = "sha256:changed"
        elif mutation == "upstream_hash":
            key = next(iter(changed["source_artifact_hashes"]["eligible_producers"]))
            changed["source_artifact_hashes"]["eligible_producers"][key] = "sha256:changed"
        elif mutation == "summary":
            changed["sample_size_budget"]["effective_independent_n"] = 1
        elif mutation == "unknown_family":
            changed["rows"][0]["unit_id"] = "invented"
        elif mutation == "raw_missing":
            changed["raw_rows_path"] = str(tmp_path / "missing-rows.json")
        elif mutation == "legacy_missing":
            changed["legacy_path"] = str(tmp_path / "missing-legacy.json")
        elif mutation.startswith("bank_"):
            bank = tmp_path / f"{mutation}.json"
            if mutation == "bank_malformed":
                bank.write_text("{")
            else:
                state = json.loads(Path(changed["bank_state_paths"]["growth"]).read_text())
                state["budget"]["proposal_credits_spent"] += 1
                bank.write_text(json.dumps(state))
            changed["bank_state_paths"]["growth"] = str(bank)
        else:
            changed["bank_state_paths"]["growth"] = str(tmp_path / "absent.json")
        candidate = tmp_path / f"{mutation}.json"
        candidate.write_text(json.dumps(changed))
        assert exp.cold_reduce(candidate)["valid"] is False


def test_7742_private_cli_e2e(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7742-REPLAY: thin CLI and fresh cold reader agree."""
    candidate = tmp_path / "cli-candidate.json"
    raw = tmp_path / "raw"
    private_root = tmp_path / "inputs"
    from carnot.experiment_7732_v673_causal_admission import INPUTS

    for relative in [*INPUTS, str(exp.MANIFEST.relative_to(exp.ROOT))]:
        target = private_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp.ROOT / relative, target)
    command = (
        sys.executable,
        "-u",
        str(exp.ROOT / exp.CLI),
        "--date",
        "20260927",
        "--root",
        str(private_root),
        "--output",
        str(candidate),
        "--raw",
        str(raw),
        "--no-validate",
    )
    written = subprocess.run(
        command, cwd=exp.ROOT, capture_output=True, text=True, timeout=180, check=False
    )
    assert written.returncode == 0, written.stdout + written.stderr
    assert candidate.is_file()
    replay = subprocess.run(
        (sys.executable, "-u", str(exp.ROOT / exp.CLI), "--cold-reduce", str(candidate)),
        cwd=exp.ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)["valid"] is True


def test_7742_main_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7742-REPLAY: main propagates cold reader status."""
    calls: list[tuple] = []
    monkeypatch.setattr(exp, "cold_reduce", lambda _path: {"valid": True})
    assert exp.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 0
    monkeypatch.setattr(exp, "cold_reduce", lambda _path: {"valid": False})
    assert exp.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 1
    monkeypatch.setattr(exp, "run_experiment", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert exp.main(["--root", str(tmp_path), "--output", "candidate.json", "--no-validate"]) == 0
    assert calls[0][0][0] == tmp_path
    assert calls[0][1]["validate"] is False


@pytest.mark.parametrize("fail_terminal", [False, True])
def test_7742_validation_gate_uses_affected_checks(
    private_run: tuple[dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_terminal: bool,
) -> None:
    """SCENARIO-REPORT-7742-TERMINAL: current readers control readiness."""
    artifact, _ = private_run
    measured = {
        name: artifact[name]
        for name in (
            "rows",
            "legacy_path",
            "legacy_hash",
            "static_closure_complete",
            "bank_state_paths",
            "lifecycle_rows",
            "pending_high_water",
            "proposal_count",
            "exactly_once",
            "restart_exact_parity",
            "raw_rows_path",
            "raw_rows_sha256",
        )
    }
    monkeypatch.setattr(exp, "preflight", lambda _root: ([], {"eligible_producers": {}}, []))
    monkeypatch.setattr(exp, "measure", lambda *_args: measured)

    def fake_commands(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": item.name,
                "passed": not (fail_terminal and item.name == "adversarial_verify"),
                "exit_code": 1 if fail_terminal and item.name == "adversarial_verify" else 0,
                "log_path": str(tmp_path / f"{item.name}.log"),
                "log_sha256": "sha256:private",
            }
            for item in commands
        ]

    monkeypatch.setattr(exp, "run_commands", fake_commands)
    result = exp.run_experiment(
        tmp_path, "20260927", tmp_path / "terminal.json", raw=tmp_path / "raw", validate=True
    )
    assert result["acquisition_protocol_ready_score"] == (0 if fail_terminal else 1)
    assert result["flagged_adversarial"] is fail_terminal
    assert result["verdict_class"] == ("disqualified" if fail_terminal else "circular_positive")
    assert result["validation_receipts"]["exact_terminal_candidate_sha256"].startswith("sha256:")
