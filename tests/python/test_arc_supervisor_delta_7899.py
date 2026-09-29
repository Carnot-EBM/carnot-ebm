"""REQ-REPORT-7899-V685: private supervisor delta and direct CLI contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.arc_supervisor_v685_delta import reduce_receipts, replay_delta
from scripts.experiments import experiment_7899_v685_arc_supervisor_delta as cli


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _producer(root: Path, receipts: list[dict], name: str = "experiment_9001_arc") -> Path:
    raw = _write(root / "results/raw/arc/rows.json", {"rows": receipts})
    return _write(
        root / f"results/{name}.json",
        {
            "run_date": "20260929",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "source_artifact_hashes": {raw.relative_to(root).as_posix(): sha256_file(raw)},
        },
    )


def _receipt(
    identity: str, *, provenance: str = "live_agent_self_discovery", resolved: bool = True
) -> dict:
    return {
        "game": "g1",
        "seed": 7,
        "attempt": "run-1",
        "solve_provenance": provenance,
        "termination": {"reason": "action_limit"},
        "trajectory_supervisor": {
            "mode": "applied",
            "redirects": [
                {
                    "id": identity,
                    "arm": "drop_goal_bias",
                    "fired": True,
                    "resolved_by_levelup": resolved,
                    "actions_to_levelup": 4 if resolved else None,
                }
            ],
            "stagnations_unredirected": 2,
        },
    }


# SCENARIO-REPORT-7899-RECEIPTS: event identity and content, not mtime, govern N.
def test_delta_unseen_duplicate_revision_and_censor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    original = _receipt("old")
    producer = _producer(
        root,
        [
            original,
            _receipt("new", resolved=False),
            _receipt("new", resolved=False),
            _receipt("proxy", provenance="development_proxy"),
        ],
    )
    baseline = {"g1|7|run-1|old|drop_goal_bias": "sha256:old"}
    result = reduce_receipts(root, [producer], baseline, {})
    assert result["new_outcome_count"] == 1
    assert result["sample_size_budget"]["censored"] == 1
    assert result["per_game_results"]["g1"]["arms"]["drop_goal_bias"]["firings"] == 1
    assert {row["reason"] for row in result["rows"] if row["status"] == "excluded"} >= {
        "revised_prior_event",
        "duplicate_event",
        "non_live_provenance",
    }
    assert result["rows"][2]["source_sha256"] == sha256_file(root / "results/raw/arc/rows.json")
    assert replay_delta(result) == []
    result["new_outcome_count"] = 2
    assert replay_delta(result) == ["new_outcome_count"]


# SCENARIO-REPORT-7899-RECEIPTS: malformed and unauthenticated bytes stay excluded.
def test_malformed_missing_and_wrong_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    producer = _producer(root, [_receipt("good"), {"game": "g1", "seed": 1}])
    document = json.loads(producer.read_text())
    document["source_artifact_hashes"]["results/raw/arc/rows.json"] = "sha256:wrong"
    _write(producer, document)
    wrong = reduce_receipts(root, [producer], {}, {})
    assert wrong["new_outcome_count"] == 0
    assert wrong["rows"][0]["reason"] == "raw_hash_mismatch"
    raw = root / "results/raw/arc/rows.json"
    raw.write_text("{broken", encoding="utf-8")
    document["source_artifact_hashes"]["results/raw/arc/rows.json"] = sha256_file(raw)
    _write(producer, document)
    bad = reduce_receipts(root, [producer], {}, {})
    assert bad["rows"][0]["reason"] == "malformed_raw_json"
    raw.unlink()
    assert reduce_receipts(root, [producer], {}, {})["rows"][0]["reason"] == "missing_raw"


# SCENARIO-REPORT-7899-CLI: real private success, missing argv and cold replay.
def test_real_private_cli(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    producer = _producer(root, [])
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7899_v685_arc_supervisor_delta.py"
    )
    output = tmp_path / "delta.json"
    cmd = [
        sys.executable,
        str(script),
        "--reduce-ledger",
        str(root),
        "--producer",
        str(producer),
        "--output",
        str(output),
    ]
    good = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    assert json.loads(output.read_text())["new_outcome_count"] == 0
    missing = subprocess.run(cmd[:-2], capture_output=True, text=True, check=False)
    assert missing.returncode != 0 and "--output" in missing.stderr
    replay = subprocess.run(
        [sys.executable, str(script), "--cold-replay", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay.returncode == 0
    data = json.loads(output.read_text())
    data["new_outcome_count"] = 1
    _write(output, data)
    forged = subprocess.run(
        [sys.executable, str(script), "--cold-replay", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert forged.returncode == 1 and "new_outcome_count" in forged.stdout


# SCENARIO-REPORT-7899-CLI: absent private path is a terminal blocked operand.
def test_missing_producer_is_blocked(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with pytest.raises(FileNotFoundError):
        reduce_receipts(root, [root / "missing.json"], {}, {})


# SCENARIO-REPORT-7899-RECEIPTS: producer and redirect schema failures are rows.
def test_schema_exclusions_and_prior_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    producer = _producer(root, [_receipt("old")])
    raw = root / "results/raw/arc/rows.json"
    assert (
        reduce_receipts(root, [producer], {"raw:" + sha256_file(raw): sha256_file(raw)}, {})[
            "rows"
        ][0]["reason"]
        == "prior_raw_hash"
    )
    producer.write_text("{bad", encoding="utf-8")
    assert (
        reduce_receipts(root, [producer], {}, {})["rows"][0]["reason"] == "malformed_producer_json"
    )
    _write(producer, {"run_date": "20260929"})
    assert reduce_receipts(root, [producer], {}, {})["rows"][0]["reason"] == "missing_source_hashes"
    _write(producer, {"source_artifact_hashes": {}, "verdict_class": "disqualified"})
    assert reduce_receipts(root, [producer], {}, {})["rows"][0]["reason"] == "unqualified_producer"
    row = _receipt("x")
    row["trajectory_supervisor"]["redirects"][0].pop("arm")
    _producer(
        root,
        [row, {**_receipt("shadow"), "trajectory_supervisor": {"mode": "shadow", "redirects": []}}],
    )
    reasons = {r["reason"] for r in reduce_receipts(root, [producer], {}, {})["rows"]}
    assert {"malformed_redirect", "not_applied"} <= reasons


# SCENARIO-REPORT-7899-RECEIPTS: every distinct event and small-sample limit is replayable.
def test_identity_paths_and_recommendation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = _receipt("old")
    old_hash = canonical_hash(
        {"episode": first, "redirect": first["trajectory_supervisor"]["redirects"][0]}
    )
    previous = {"g1|7|run-1|old|drop_goal_bias": old_hash}
    revised = _receipt("new")
    revised["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"] = 3
    no_identity = _receipt("unknown")
    no_identity.pop("attempt")
    rows = [first, _receipt("new"), revised, no_identity, {"game": "g0"}]
    for i in range(20):
        item = _receipt(f"many-{i}")
        item.update(game=f"g{i % 3}", seed=i, attempt=f"run-{i}")
        rows.append(item)
    producer = _producer(root, rows)
    result = reduce_receipts(root, [producer], previous, {})
    assert result["new_outcome_count"] == 21
    assert {r["reason"] for r in result["rows"] if r["status"] == "excluded"} >= {
        "prior_event",
        "revised_duplicate_event",
        "missing_identity",
    }
    assert len(result["recommendation_rows"]) == 1
    assert result["recommendation_rows"][0]["causal_claim"] is False
    result["firings"] += 1
    assert replay_delta(result) == ["firings"]
    doc = json.loads(producer.read_text())
    doc["source_artifact_hashes"]["results/raw/../../../outside.json"] = "sha256:x"
    doc["source_artifact_hashes"]["other.json"] = "sha256:x"
    _write(producer, doc)
    assert "path_escape" in {
        r["reason"] for r in reduce_receipts(root, [producer], previous, {})["rows"]
    }


# SCENARIO-REPORT-7899-CLI: explicit manifest and negative-exit reason are frozen.
def test_manifest_and_sealed_expected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = cli.commands(tmp_path)
    negative = next(row for row in manifest if row["name"] == "cli_failure_coverage")
    assert negative["expected_exit"] == 2 and negative["expected_text"] == "--output"
    assert "--date" in next(row for row in manifest if row["name"] == "e2e_016_replay")["argv"]
    log = _write(tmp_path / "log.txt", "error: --output required")

    def fake_run(_root: Path, _specs: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": "negative",
                "log_path": str(log),
                "exit_code": 2,
                "timed_out": False,
                "output_tail": log.read_text(),
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    row = {
        "name": "negative",
        "argv": [sys.executable, "-V"],
        "classification": "required",
        "deadline_s": 10,
        "expected_exit": 2,
        "expected_text": "--output",
    }
    one = cli._run(row, tmp_path, 0.0)
    assert one["passed"] and Path(one["log_path"]).is_file()
    assert cli._run(row, tmp_path, 0.0)["log_sha256"] == one["log_sha256"]
    log.write_text("different error")
    assert cli._run(row, tmp_path, 0.0)["passed"] is False


# SCENARIO-REPORT-7899-CLI: prior inventory survives a failed historical verdict.
@pytest.mark.parametrize(
    "games", [{"g1": {"levels_reproduced": 2}}, [{"game": "g1", "levels_reproduced": 2}]]
)
def test_precheck_private_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, games: object
) -> None:
    prior = _write(
        tmp_path / "prior.json",
        {
            "verdict_class": "disqualified",
            "outcome_rows": [
                {"event_id": "old", "content_sha256": "sha256:event", "source_sha256": "sha256:raw"}
            ],
        },
    )
    registry = tmp_path / "registry.yaml"
    import yaml

    registry.write_text(yaml.safe_dump({"games": games}))
    agent = tmp_path / "agent.py"
    agent.write_text("pass\n")
    monkeypatch.setattr(cli, "PRIOR", prior)
    monkeypatch.setattr(cli, "REGISTRY", registry)
    monkeypatch.setattr(cli, "AGENT", agent)
    monkeypatch.setattr(
        cli,
        "EXPECTED",
        {prior: sha256_file(prior), registry: sha256_file(registry), agent: sha256_file(agent)},
    )
    checks, failures, baseline, levels = cli.precheck()
    assert failures == [] and len(checks) == 4
    assert baseline == {"old": "sha256:event", "raw:sha256:raw": "sha256:raw"}
    assert levels == {"g1": 2}
    prior.unlink()
    assert cli.precheck()[1][0]["observed"] == "missing"
    _write(prior, {"verdict_class": "null", "outcome_rows": []})
    monkeypatch.setattr(
        cli,
        "EXPECTED",
        {prior: sha256_file(prior), registry: sha256_file(registry), agent: sha256_file(agent)},
    )
    assert cli.precheck()[1][0]["artifact_field"] == "verdict_class"


# SCENARIO-REPORT-7899-CLI: direct in-process private modes also count for coverage.
def test_main_private_routes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    producer = _producer(root, [])
    output = tmp_path / "delta.json"
    assert (
        cli.main(
            ["--reduce-ledger", str(root), "--producer", str(producer), "--output", str(output)]
        )
        == 0
    )
    assert cli.main(["--cold-replay", str(output)]) == 0
    data = json.loads(output.read_text())
    data["firings"] = 1
    _write(output, data)
    assert cli.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--reduce-ledger", str(root), "--producer", str(producer)])


# SCENARIO-REPORT-7899-CLI: owned failure and external block have separate terminal classes.
@pytest.mark.parametrize("state", ["valid", "blocked", "required_failure", "terminal_failure"])
def test_dispatcher_terminal_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    private = tmp_path / "private"
    prior = _write(
        tmp_path / "prior.json",
        {
            "historical_required_failures": [],
            "honest_verdict": "complete_disqualified_terminal_verification",
            "validation_errors": [],
        },
    )
    monkeypatch.setattr(cli, "PRIVATE", private)
    monkeypatch.setattr(cli, "PRIOR", prior)
    monkeypatch.setattr(cli, "EXPECTED", {})
    failure = {
        "upstream_id": "prior",
        "path": str(tmp_path / "missing"),
        "sha256": None,
        "artifact_field": "sha256",
        "op": "==",
        "expected": "sha256:expected",
        "observed": "missing",
    }
    monkeypatch.setattr(
        cli, "precheck", lambda: ([], [failure] if state == "blocked" else [], {}, {})
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    _write(tmp_path / "results/experiment_arc_fixture.json", {})
    monkeypatch.setattr(
        cli,
        "commands",
        lambda _private: [
            {
                "name": "affected_pytest",
                "argv": [sys.executable, "-V"],
                "deadline_s": 30,
                "classification": "required",
                "expected_exit": 0,
                "expected_text": None,
            }
        ],
    )
    terminal_calls = 0

    def fake_run(row: dict, _private: Path, _started: float) -> dict:
        nonlocal terminal_calls
        if row["name"] == "terminal_adversarial":
            terminal_calls += 1
        passed = row["name"] != "affected_pytest" or state != "required_failure"
        if (
            state == "terminal_failure"
            and row["name"] == "terminal_adversarial"
            and terminal_calls == 1
        ):
            passed = False
        return {
            "name": row["name"],
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "command_argv": row["argv"],
        }

    monkeypatch.setattr(cli, "_run", fake_run)
    output = tmp_path / "output.json"
    rc = cli.main(["--date", "20260929", "--output", str(output)])
    data = json.loads(output.read_text())
    expected = {
        "valid": "null",
        "blocked": "blocked",
        "required_failure": "disqualified",
        "terminal_failure": "disqualified",
    }[state]
    assert data["verdict_class"] == expected
    assert rc == (0 if state == "valid" else 1)
    assert data["arc_delta_ready_score"] == int(state == "valid")
    if state == "blocked":
        assert data["gate_check_summary"] == [failure]
