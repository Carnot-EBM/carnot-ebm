"""REQ-REPORT-7845: authenticate current supervisor outcomes before disposition."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from carnot.reporting.arc_supervisor_delta import inspect_sources, reduce_rows, verify_prior
from scripts.experiments import experiment_7845_v681_arc_supervisor_delta as cli


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, number: int, rows: list[dict], *, verdict: str = "null") -> Path:
    raw = Path(f"results/raw/experiment_{number}_v681_arc/rows.json")
    digest = _write(root / raw, {"rows": rows})
    producer = root / f"results/experiment_{number}_v681_arc.json"
    _write(
        producer,
        {
            "run_date": "20260929",
            "verdict_class": verdict,
            "flagged_adversarial": verdict == "disqualified",
            "source_artifact_hashes": {raw.as_posix(): {"sha256": digest}},
        },
    )
    return root / raw


def _row(game: str, helped: bool, *, shadow: bool = False, seed: int = 1) -> dict:
    receipt = {
        "mode": "shadow" if shadow else "applied",
        "enabled": not shadow,
        "stagnations_unredirected": 1,
        "redirects": []
        if shadow
        else [
            {
                "id": f"{game}-{seed}",
                "arm": "drop_goal_bias",
                "resolved_by_levelup": helped,
                "actions_to_levelup": 3 if helped else None,
            }
        ],
    }
    return {
        "game": game,
        "seed": seed,
        "termination": {"reason": "level_up" if helped else "stopped"},
        "trajectory_supervisor": receipt,
    }


# SCENARIO-REPORT-7845-EMPTY: no receipt after the baseline is a completed null.
def test_empty_delta_and_prior_hash_exclusion(tmp_path: Path) -> None:
    raw = _source(tmp_path, 9001, [_row("g1", True)])
    known = "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()
    found = inspect_sources(tmp_path, "20260928", {known})
    result = reduce_rows(found)
    assert result["new_source_count"] == 0
    assert result["honest_verdict"] == "complete_null_no_new_supervisor_outcomes"
    assert result["recommendation_rows"] == []
    assert found[0]["status"] == "excluded"
    assert found[0]["reason"] == "prior_hash_inventory"


# SCENARIO-REPORT-7845-RECEIPT: a shadow or disqualified source gives no firing.
def test_exclusions_and_censoring(tmp_path: Path) -> None:
    _source(tmp_path, 9001, [_row("g1", True), _row("g2", False), _row("g3", False, shadow=True)])
    _source(tmp_path, 9002, [_row("g4", True)], verdict="disqualified")
    found = inspect_sources(tmp_path, "20260928", set())
    result = reduce_rows(found)
    assert result["new_source_count"] == 1
    assert result["arm_statistics"]["drop_goal_bias"]["closed"] == 2
    assert result["arm_statistics"]["drop_goal_bias"]["progress"] == 1
    assert result["recommendation_rows"] == []
    assert any(row["reason"] == "disqualified_producer" for row in found)
    assert any(row["status"] == "shadow" for row in found)


# SCENARIO-REPORT-7845-RECEIPT: retirement needs twenty closed firings in three games.
def test_retirement_threshold_and_malformed_rows(tmp_path: Path) -> None:
    bad = _row("bad", False)
    del bad["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"]
    _source(tmp_path, 9001, [_row(f"g{i % 3}", False, seed=i) for i in range(20)] + [bad])
    found = inspect_sources(tmp_path, "20260928", set())
    result = reduce_rows(found)
    assert result["recommendation_rows"] == [
        {"arm": "drop_goal_bias", "kind": "shadow_retirement", "causal_claim": False}
    ]
    assert result["arm_statistics"]["drop_goal_bias"]["closed"] == 20
    assert any(row["reason"] == "malformed_redirect" for row in found)


# SCENARIO-REPORT-7845-GATE: mutation of a producer's raw bytes is excluded.
def test_hash_mismatch_is_visible(tmp_path: Path) -> None:
    raw = _source(tmp_path, 9001, [_row("g1", True)])
    raw.write_text("{}", encoding="utf-8")
    found = inspect_sources(tmp_path, "20260928", set())
    assert found[0]["reason"] == "raw_hash_mismatch"
    assert reduce_rows(found)["new_source_count"] == 0


# SCENARIO-REPORT-7845-RECEIPT: bad producer JSON remains excluded evidence.
def test_malformed_producer_is_excluded(tmp_path: Path) -> None:
    producer = tmp_path / "results/experiment_9001_v681_arc.json"
    producer.parent.mkdir(parents=True)
    producer.write_text("{} trailing", encoding="utf-8")
    found = inspect_sources(tmp_path, "20260928", set())
    assert found[0]["reason"] == "malformed_producer"
    assert found[0]["status"] == "excluded"


# SCENARIO-REPORT-7845-EMPTY: legacy malformed files are outside the new source window.
def test_historical_producer_is_not_current(tmp_path: Path) -> None:
    producer = tmp_path / "results/experiment_7000_old.json"
    producer.parent.mkdir(parents=True)
    producer.write_text("invalid", encoding="utf-8")
    assert inspect_sources(tmp_path, "20260928", set()) == []


# SCENARIO-REPORT-7845-RECEIPT: bad paths and incomplete firings remain visible.
def test_raw_path_and_receipt_exclusions(tmp_path: Path) -> None:
    good = _source(tmp_path, 9001, [_row("g1", True)])
    producer = tmp_path / "results/experiment_9001_v681_arc.json"
    document = json.loads(producer.read_text(encoding="utf-8"))
    document["source_artifact_hashes"]["results/raw/missing.json"] = "sha256:abc"
    document["source_artifact_hashes"]["results/raw/../../../outside.json"] = "sha256:abc"
    document["source_artifact_hashes"]["ignore.txt"] = "sha256:abc"
    _write(producer, document)
    found = inspect_sources(tmp_path, "20260928", set())
    assert any(row.get("reason") == "missing_raw" for row in found)
    assert all(row.get("source_path") != "ignore.txt" for row in found)
    assert any(row.get("status") == "completed" for row in found)
    assert good.is_file()


# SCENARIO-REPORT-7845-RECEIPT: duplicate, empty, and other modes cannot add N.
def test_duplicate_and_unfired_receipts(tmp_path: Path) -> None:
    duplicate = _row("g1", True)
    empty = _row("g2", False)
    empty["trajectory_supervisor"]["redirects"] = []
    other = _row("g3", False)
    other["trajectory_supervisor"]["mode"] = "unknown"
    absent = {"game": "g4", "seed": 1}
    _source(tmp_path, 9001, [duplicate, duplicate, empty, other, absent])
    found = inspect_sources(tmp_path, "20260928", set())
    assert {row.get("reason") for row in found} >= {"duplicate_retry", "no_firings", "not_applied"}
    assert reduce_rows(found)["arm_statistics"]["drop_goal_bias"]["closed"] == 1


# SCENARIO-REPORT-7845-RECEIPT: exhaustion asks for a reusable mechanism.
def test_exhaustion_and_censoring(tmp_path: Path) -> None:
    row = _row("g1", False)
    row["termination"]["reason"] = "action_limit"
    receipt = row["trajectory_supervisor"]
    receipt["arms_enabled"] = ["drop_goal_bias"]
    receipt["arms_used"] = ["drop_goal_bias"]
    _source(tmp_path, 9001, [row])
    result = reduce_rows(inspect_sources(tmp_path, "20260928", set()))
    assert result["arm_statistics"]["drop_goal_bias"]["censored"] == 1
    assert result["general_mechanism_requirement"] is not None


# SCENARIO-REPORT-7845-RECEIPT: a malformed source hash map adds no source.
def test_nonmapping_source_hashes(tmp_path: Path) -> None:
    producer = tmp_path / "results/experiment_9001_v681_arc.json"
    _write(producer, {"run_date": "20260929", "source_artifact_hashes": []})
    assert inspect_sources(tmp_path, "20260928", set()) == []


# SCENARIO-REPORT-7845-RECEIPT: normalize a declared digest without a prefix.
def test_unprefixed_declared_hash(tmp_path: Path) -> None:
    _source(tmp_path, 9001, [_row("g1", True)])
    producer = tmp_path / "results/experiment_9001_v681_arc.json"
    document = json.loads(producer.read_text(encoding="utf-8"))
    key = next(iter(document["source_artifact_hashes"]))
    document["source_artifact_hashes"][key] = document["source_artifact_hashes"][key][
        "sha256"
    ].removeprefix("sha256:")
    _write(producer, document)
    assert reduce_rows(inspect_sources(tmp_path, "20260928", set()))["new_source_count"] == 1


# SCENARIO-REPORT-7845-EMPTY: a new file without a supervisor outcome is not a new source.
def test_new_non_supervisor_file_is_not_a_source(tmp_path: Path) -> None:
    _source(tmp_path, 9001, [{"game": "g1", "seed": 1}])
    found = inspect_sources(tmp_path, "20260928", set())
    assert reduce_rows(found)["new_source_count"] == 0
    assert any(row.get("reason") == "no_supervisor_receipt" for row in found)


# SCENARIO-REPORT-7845-GATE: the inventory's old failure does not hide a bad operand.
def test_prior_gate_reports_missing_and_wrong_hash(tmp_path: Path) -> None:
    source = tmp_path / "prior.json"
    assert verify_prior(source, "sha256:abc")[0]["observed"] == "missing"
    source.write_text("{}", encoding="utf-8")
    failed = verify_prior(source, "sha256:abc")
    assert failed[0]["artifact_field"] == "sha256"
    assert failed[0]["expected"] == "sha256:abc"
    assert failed[0]["observed"].startswith("sha256:")


# SCENARIO-REPORT-7845-CLI: execute the reader in a private output root.
def test_cli_private_receipt_to_disposition(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7845_v681_arc_supervisor_delta.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--date",
            "20260929",
            "--reader-only",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["experiment_id"] == 7845
    assert result["task_id"] == "exp7845-arc-supervisor-delta"
    assert result["new_source_count"] == 0
    assert result["production_defaults_changed"] is False
    assert any(
        path.endswith("arc_supervisor_delta.py") for path in result["source_artifact_hashes"]
    )


# SCENARIO-REPORT-7845-GATE: inspect each missing or changed gate operand.
def test_cli_preconditions_preserve_exact_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "results/prior.json"
    registry = tmp_path / "ops/arc_solve_registry.yaml"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "PRIOR", prior)
    monkeypatch.setattr(cli, "PRIOR_HASH", "sha256:abc")
    _, failures, _ = cli.preconditions()
    assert {item["artifact_field"] for item in failures} == {"sha256", "levels_reproduced_positive"}
    expected = _write(
        prior,
        {"source_inventory": [{}, {"raw": "results/raw/missing.json", "raw_sha256": "sha256:abc"}]},
    )
    monkeypatch.setattr(cli, "PRIOR_HASH", expected)
    registry.parent.mkdir(parents=True)
    registry.write_text("levels_reproduced: 1\n", encoding="utf-8")
    _, failures, hashes = cli.preconditions()
    assert len(failures) == 1
    assert failures[0]["artifact_field"] == "raw_sha256"
    assert hashes == {"sha256:abc"}


# SCENARIO-REPORT-7845-CLI: a closed log's bytes survive a cold read.
def test_seal_and_cold_replay_detect_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "RAW", tmp_path / "results/raw/audit")
    source = tmp_path / "child.log"
    source.write_bytes(b"done\n")
    row = {"name": "unit", "log_path": str(source)}
    sealed = cli.seal([dict(row)])[0]
    assert cli.seal([dict(row)])[0]["log_sha256"] == sealed["log_sha256"]
    candidate = tmp_path / "candidate.json"
    _write(candidate, {"validation_receipts": [sealed]})
    assert cli.cold_replay(candidate) == []
    target = tmp_path / sealed["log_path"]
    target.write_bytes(b"changed\n")
    assert cli.cold_replay(candidate) == ["unit"]
    with pytest.raises(ValueError, match="sealed_log_collision"):
        cli.seal([dict(row)])


# SCENARIO-REPORT-7845-CLI: required failures close readiness while health stays separate.
@pytest.mark.parametrize("case", ["valid", "failed", "blocked"])
def test_scoped_validation_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    names = ("worktree_imports", "cold_replay", "adversarial_verify", "repository_health_180s")
    commands = [
        {
            "name": name,
            "argv": [sys.executable, name],
            "classification": "diagnostic" if name == "repository_health_180s" else "required",
            "deadline_s": 1,
        }
        for name in names
    ]
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"commands": commands})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "PRIOR", tmp_path / "results/prior.json")
    monkeypatch.setattr(cli, "MANIFEST", manifest)
    monkeypatch.setattr(cli, "PRIVATE", tmp_path / "private")
    previous = tmp_path / "previous.json"
    monkeypatch.setattr(cli, "DELIVERABLE", previous)
    monkeypatch.setattr(cli, "seal", lambda receipts: receipts)
    if case == "valid":
        health_log = tmp_path / "health.log"
        health_hash = _write(health_log, {"result": "failed_health"})
        _write(
            previous,
            {
                "validation_receipts": [
                    {
                        "name": "repository_health_180s",
                        "command_argv": commands[-1]["argv"],
                        "log_path": str(health_log),
                        "log_sha256": health_hash,
                        "passed": False,
                        "exit_code": -15,
                    }
                ]
            },
        )

    def fake_run(_root: Path, specs: list, **_kwargs: object) -> list[dict]:
        spec = specs[0]
        assert case != "valid" or spec.name != "repository_health_180s"
        success = case == "valid" or spec.name not in {"cold_replay", "adversarial_verify"}
        return [
            {
                "name": spec.name,
                "command_argv": list(spec.argv),
                "passed": success,
                "exit_code": 0 if success else 1,
                "resolved_imports": {
                    "carnot.reporting.arc_supervisor_delta": str(
                        cli.ROOT / "python/carnot/reporting/arc_supervisor_delta.py"
                    )
                }
                if case != "failed"
                else {},
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    gates = [{"artifact_field": "sha256"}] if case == "blocked" else []
    artifact = cli.base_artifact("20260929", time.monotonic_ns(), [], gates, [], reduce_rows([]))
    result = cli.validate(artifact, time.monotonic())
    assert len(result["validation_receipts"]) == 4
    candidate = tmp_path / "private/candidate.json"
    assert json.loads(candidate.read_text(encoding="utf-8"))["duration_s"] > 0
    if case == "valid":
        assert result["supervisor_delta_ready_score"] == 1
        assert result["verdict_class"] == "null"
        assert result["repository_health"]["status"] == "failed_health"
    else:
        assert result["supervisor_delta_ready_score"] == 0
        assert result["verdict_class"] == ("blocked" if case == "blocked" else "disqualified")


# SCENARIO-REPORT-7845-CLI: main writes a private current result without child recursion.
def test_main_success_path_uses_current_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"

    def fake_validate(artifact: dict, _started: float) -> dict:
        artifact["supervisor_delta_ready_score"] = 1
        return artifact

    monkeypatch.setattr(cli, "validate", fake_validate)
    monkeypatch.setattr(cli, "preconditions", lambda: ([], [], set()))
    monkeypatch.setattr(cli, "inspect_sources", lambda *_args: [])
    monkeypatch.setattr(sys, "argv", ["exp7845", "--date", "20260929", "--output", str(output)])
    assert cli.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 7845


# SCENARIO-REPORT-7845-CLI: the direct cold-replay mode returns failed bytes.
def test_main_cold_replay_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.json"
    _write(candidate, {})
    monkeypatch.setattr(cli, "cold_replay", lambda _path: ["mutated"])
    monkeypatch.setattr(sys, "argv", ["exp7845", "--cold-replay", str(candidate)])
    assert cli.main() == 1
