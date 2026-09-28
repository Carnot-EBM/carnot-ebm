"""V679 capstone checks bound to immutable authority (REQ-REPORT-7822)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from carnot.experiment_7822_v679_capstone import (
    MANIFEST_SHA,
    account,
    arc_scores,
    authority,
    build,
    cold_replay,
    decisions,
    dispatch,
    load_manifest,
)
from carnot.reporting.current_work_receipt import sha256_file


ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "docs/research-notes/v679-authority-snapshots"
MANIFEST = ROOT / "results/raw/experiment_7822_v679_capstone/validation_command_manifest.json"
CLI = ROOT / "scripts/experiments/experiment_7822_v679_capstone.py"


def test_authority_rejects_changed_table_and_yaml(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7822-CUSTODY: independent contract bytes must agree."""
    design = (SNAP / "design.md").read_bytes()
    roadmap = (SNAP / "roadmap.yaml").read_bytes()
    assert authority(design, roadmap)["passed"] is True
    assert (
        authority(design.replace(b"Reconcile fourteen outcomes", b"Invent outcomes", 1), roadmap)[
            "passed"
        ]
        is False
    )
    changed = yaml.safe_load(roadmap)
    changed["tasks"][0]["title"] = "wrong"
    assert authority(design, yaml.safe_dump(changed).encode())["passed"] is False
    assert authority(b"bad", b"[]")["passed"] is False


def test_current_rows_keep_missing_producers_and_queue_receipts_separate() -> None:
    """SCENARIO-REPORT-7822-CUSTODY: a queue receipt cannot fill a producer."""
    tasks = yaml.safe_load((SNAP / "roadmap.yaml").read_bytes())["tasks"]
    rows, sources, failures = account(ROOT, tasks)
    assert [r["experiment_id"] for r in rows] == list(range(7809, 7823))
    assert len(sources) == 13
    assert rows[-1]["availability"] == "planned_output"
    assert rows[-1]["producer_hash"] is None
    assert [r["experiment_id"] for r in rows if r["availability"] == "absent"] == [7813, 7816, 7819]
    assert [r["experiment_id"] for r in rows if r["availability"] == "pre_gate_receipt"] == [
        7812,
        7815,
    ]
    assert rows[9]["flagged_adversarial"] is True
    assert rows[9]["producer_eligible"] is False
    assert all(r["benefit"] is None for r in rows)
    assert {int(f["upstream_id"][3:]) for f in failures if f["field"] == "producer_path"} == {
        7812,
        7813,
        7815,
        7816,
        7819,
    }
    assert all(s["path"] != tasks[-1]["deliverable"] for s in sources)


def test_repeated_exact_verdict_retires_only_listed_scope() -> None:
    """SCENARIO-REPORT-7822-TERMINAL: prior_failures drive retirement."""
    tasks = yaml.safe_load((SNAP / "roadmap.yaml").read_bytes())["tasks"]
    rows, _, _ = account(ROOT, tasks)
    outcome = decisions(rows, tasks)
    assert len(outcome) == 14
    assert outcome[2]["decision"] == "retire"
    assert "exp7797-training-runtime" in outcome[2]["matched_prior_ids"]
    assert outcome[3]["decision"] == "await_named_prerequisite"
    assert outcome[12]["decision"] == "await_named_prerequisite"
    assert all(item["trigger"] for item in outcome)


def test_blocked_result_has_fourteen_rows_and_distinct_claim_scopes() -> None:
    """SCENARIO-REPORT-7822-TERMINAL: missing science stays blocked."""
    publication = {"gates": {x: {"pass": True} for x in ("G1", "G2", "G3", "G4")}}
    publication.update(paper_ready=True, unmet_gates=[])
    result = build(ROOT, publication, {"required_checks_passed": True}, [], 0.1)
    assert result["honest_verdict"] == "complete_blocked_required_v679_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["paper_ready"] is True
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert len(result["task_dispositions"]) == 14
    assert result["claim_scope"]["source"].startswith("640 exposed")
    assert result["claim_scope"]["hidden_game"] == "unmeasured"
    assert result["MODEL_SPECS"] == []
    assert result["model_invocation_counts"]["loads"] == 0
    assert result["arc_sdk_score_recomputed"] == {
        "episode_count": 48,
        "independent_games": 8,
        "score_mismatches": [],
        "eligible_for_benefit": False,
    }
    assert result["hardware_complete_service_bound"] is None
    assert (
        result["source_artifact_hashes"][-1]["path"] != "results/experiment_7822_v679_capstone.json"
    )


def test_required_failure_disqualifies_and_zeros_all_readiness() -> None:
    """SCENARIO-REPORT-7822-TERMINAL: owned validation is a hard gate."""
    publication = {"gates": {x: {"pass": False} for x in ("G1", "G2", "G3", "G4")}}
    publication.update(paper_ready=False, unmet_gates=["G1", "G2", "G3", "G4"])
    result = build(ROOT, publication, {"required_checks_passed": False}, [], 0.1)
    assert result["verdict_class"] == "disqualified"
    assert set(result["acceptance_gate_results"].values()) == {0}


def test_cold_replay_detects_source_and_sealed_log_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7822-CUSTODY: a one-byte change is visible."""
    source = tmp_path / "source.json"
    source.write_text("{}")
    log = tmp_path / "sealed.log"
    log.write_text("ok")
    value = {
        "source_artifact_hashes": [{"path": "source.json", "sha256": sha256_file(source)}],
        "validation_receipts": {"checks": [{"log_path": str(log), "log_sha256": sha256_file(log)}]},
        "task_dispositions": [{"experiment_id": n} for n in range(7809, 7823)],
    }
    assert cold_replay(value, tmp_path) == []
    log.write_text("oK")
    assert "log_hash_mismatch" in cold_replay(value, tmp_path)
    log.write_text("ok")
    source.write_text("{ }")
    assert "source.json" in cold_replay(value, tmp_path)


def test_real_cli_dispatch_matches_frozen_manifest_and_rejects_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7822-DISPATCH: recording child sees the full sequence."""
    spec = importlib.util.spec_from_file_location("exp7822_cli", CLI)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    frozen = load_manifest(MANIFEST)
    assert sha256_file(MANIFEST) == f"sha256:{MANIFEST_SHA}"
    seen: list[tuple[str, list[str], str]] = []

    def record(root: Path, command: dict, durable: Path) -> dict:
        seen.append((command["name"], command["argv"], command["classification"]))
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "classification": command["classification"],
            "exit_code": 0,
            "passed": True,
        }

    assert cli.main(["--dispatch-only"], executor=record) == 0
    assert seen == [(c["name"], c["argv"], c["classification"]) for c in frozen["commands"]]
    mutated = tmp_path / "manifest.json"
    changed = json.loads(MANIFEST.read_text())
    changed["commands"].append(dict(changed["commands"][-1], name="undeclared"))
    mutated.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest"):
        cli.main(["--dispatch-only", "--manifest", str(mutated)], executor=record)


def test_real_cli_seals_recorded_run_and_disqualifies_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7822-DISPATCH: later children remain in final receipts."""
    import carnot.experiment_7822_v679_capstone as module
    from scripts.publication_gate import evaluate

    spec = importlib.util.spec_from_file_location("exp7822_cli_run", CLI)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    frozen = load_manifest(MANIFEST)
    frozen["candidate_path"] = str(tmp_path / "candidate.json")
    monkeypatch.setattr(module, "load_manifest", lambda path: frozen)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "result.json")

    def recorder(root: Path, command: dict, durable: Path) -> dict:
        name = command["name"]
        log = tmp_path / f"{name}.log"
        log.write_text(json.dumps(evaluate()) if name == "publication_gate" else name)
        return dict(
            name=name,
            command_argv=command["argv"],
            classification=command["classification"],
            exit_code=1 if name == "ruff_check" else 0,
            passed=name != "ruff_check",
            log_path=str(log),
            log_sha256=sha256_file(log),
            timed_out=False,
        )

    assert cli.main(["--date", "20260928"], executor=recorder) == 1
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["verdict_class"] == "disqualified"
    assert len(result["validation_receipts"]["checks"]) == len(frozen["commands"])
    assert result["validation_receipts"]["checks"][-1]["name"] == "cold_replay"
    assert set(result["acceptance_gate_results"].values()) == {0}
    assert cli.main(["--cold-replay", str(tmp_path / "result.json")], executor=recorder) == 0


def test_private_bad_producers_and_changed_arc_score(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7822-CUSTODY: malformed inputs and score edits fail."""
    tasks = yaml.safe_load((SNAP / "roadmap.yaml").read_bytes())["tasks"]
    with pytest.raises(ValueError, match="fourteen-task"):
        account(tmp_path, tasks[:-1])
    bad = tmp_path / tasks[0]["deliverable"]
    bad.parent.mkdir(parents=True)
    bad.write_text("[]")
    invalid = tmp_path / tasks[1]["deliverable"]
    invalid.write_text("{")
    _, _, failures = account(tmp_path, tasks)
    assert [f["field"] for f in failures if f["field"] == "schema"] == ["schema", "schema"]
    assert arc_scores(tmp_path, {}) is None
    raw = json.loads(
        (
            ROOT / "results/raw/experiment_7818_v679_arc_organic_measurement/raw_rows.json"
        ).read_bytes()
    )
    raw[0]["score_charged"] += 1
    sample = tmp_path / "arc.json"
    sample.write_text(json.dumps(raw[:1]))
    assert arc_scores(tmp_path, {"raw_rows_path": "arc.json"})["score_mismatches"] == [
        raw[0]["episode_id"]
    ]


def test_cold_replay_rejects_self_input_order_and_summary_change(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7822-CUSTODY: the output is never its own input."""
    base = {
        "source_artifact_hashes": [
            {"path": "results/experiment_7822_v679_capstone.json", "sha256": None}
        ],
        "task_dispositions": [],
    }
    assert cold_replay(base, tmp_path) == ["self_input", "task_order"]
    from scripts.publication_gate import evaluate

    value = build(ROOT, evaluate(), {"required_checks_passed": True}, [], 0.1)
    value["task_dispositions"][0]["benefit"] = 1
    assert "raw_to_summary_mismatch" in cold_replay(value, ROOT)


def test_manifest_schema_and_undeclared_child_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7822-DISPATCH: manifest and receipt agree exactly."""
    import carnot.experiment_7822_v679_capstone as module

    changed = json.loads(MANIFEST.read_text())
    changed["schema"] = "wrong"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(module, "sha256_file", lambda source: f"sha256:{MANIFEST_SHA}")
    with pytest.raises(ValueError, match="schema"):
        load_manifest(path)

    def wrong(root: Path, command: dict, durable: Path) -> dict:
        return dict(
            name="undeclared",
            command_argv=command["argv"],
            classification=command["classification"],
        )

    with pytest.raises(ValueError, match="undeclared child"):
        dispatch(ROOT, json.loads(MANIFEST.read_text()), wrong)


def test_private_authority_and_raw_mutations_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7822-CUSTODY: changed authority or raw score is named."""
    import carnot.experiment_7822_v679_capstone as module
    from scripts.publication_gate import evaluate

    publication = evaluate()
    original_authority = module.authority
    original_read = Path.read_bytes
    original_arc = module.arc_scores
    value = build(ROOT, publication, {"required_checks_passed": True}, [], 0.1)
    monkeypatch.setattr(
        module, "authority", lambda design, roadmap: {"passed": False, "errors": ["mutation"]}
    )
    assert "authority_mismatch" in cold_replay(value, ROOT)
    assert any(
        f["field"] == "table_json_yaml_match"
        for f in build(ROOT, publication, {"required_checks_passed": True}, [], 0.1)[
            "gate_check_summary"
        ]
    )
    with pytest.raises(ValueError, match="authority mismatch"):
        module.main(["--date", "20260928"])
    monkeypatch.setattr(module, "authority", original_authority)

    def changed_active(path: Path) -> bytes:
        if path == ROOT / "research-roadmap.yaml":
            return b"changed active roadmap"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_active)
    assert any(
        f["field"] == "authority_bytes"
        for f in build(ROOT, publication, {"required_checks_passed": True}, [], 0.1)[
            "gate_check_summary"
        ]
    )
    with pytest.raises(ValueError, match="roadmap differs"):
        module.main(["--date", "20260928"])

    def changed_design(path: Path) -> bytes:
        if path == ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md":
            return b"changed active design"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_design)
    with pytest.raises(ValueError, match="design differs"):
        module.main(["--date", "20260928"])
    monkeypatch.setattr(Path, "read_bytes", original_read)
    monkeypatch.setattr(module, "arc_scores", lambda root, arc: {"score_mismatches": ["changed"]})
    assert any(
        f["field"] == "sdk_score_recheck"
        for f in build(ROOT, publication, {"required_checks_passed": True}, [], 0.1)[
            "gate_check_summary"
        ]
    )
    monkeypatch.setattr(module, "arc_scores", original_arc)
    with pytest.raises(ValueError, match="run date"):
        module.main(["--date", "20260929"])
