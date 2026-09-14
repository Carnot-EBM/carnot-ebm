"""Tests for the V641 source contract receipt.

Spec refs: REQ-REPORT-7288 and SCENARIO-REPORT-7288-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest
import yaml

from carnot import experiment_7288_v641_source_contract as mod


ROOT = Path(__file__).resolve().parents[2]


def _copy(root: Path, relative: Path) -> None:
    """Copy one real input so private runs keep the repository unchanged."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def _private_root(tmp_path: Path, *, milestone: str = mod.MILESTONE) -> Path:
    """Create an authenticated private view for terminal-state tests."""

    root = tmp_path / "repo"
    for relative in (*mod.INPUT_PATHS, *mod.CODE_PATHS, mod.HISTORY_PATH):
        _copy(root, relative)
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text())
    roadmap["milestone"] = milestone
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    for relative in (mod.DEFAULT_OUTPUT_PATH.parent, mod.RAW_DIR, mod.CHECKPOINT_PATH.parent):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return deterministic metadata without treating it as local evidence."""

    records = {
        "2605.27494": ("Grounded Cache Routing", "v1"),
        "2609.10873": ("When Validation Stops Learning", "v1"),
        "sqlite.org/atomiccommit": ("Atomic Commit In SQLite", "2026-09-14"),
        "2503.21076": ("Kolmogorov-Arnold Classifier", "v1"),
    }
    key = next(candidate for candidate in records if candidate in url)
    title, version = records[key]
    return {
        "ok": True,
        "status_code": 200,
        "body": f"<title>{title}</title> [version:{version}]",
        "error": None,
    }


def _passing_receipts() -> list[dict[str, Any]]:
    """Represent successful scoped checks without recursively launching pytest."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "duration_s": 0.001,
            "timed_out": False,
            "log_sha256": mod.sha256_bytes(b"fixture log"),
            "passed": True,
            "baseline_failure": False,
        }
        for name in mod.VALIDATION_COMMAND_NAMES
    ]


def test_scenario_report_7288_contract_rejects_every_mutable_dimension() -> None:
    """SCENARIO-REPORT-7288-CONTRACT checks phase as well as shipped fields."""

    authority, roadmap, content, candidates = mod.select_yaml_authority(ROOT)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap is not None and content == (ROOT / authority).read_bytes()
    assert candidates[0]["available"] is True
    markdown = (ROOT / mod.DESIGN_PATH).read_text()
    exact = mod.evaluate_contract(markdown, roadmap)
    assert exact["passed"] is True
    assert [row["unit_id"] for row in exact["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert [row["markdown"]["phase"] for row in exact["contract_rows"]] == [
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
    ]

    mutations = []
    short = deepcopy(roadmap)
    short["tasks"].pop()
    mutations.append(short)
    for field, value in (
        ("id", "exp9999-mutated"),
        ("deliverable", "results/mutated.json"),
        ("phase", 9),
    ):
        changed = deepcopy(roadmap)
        changed["tasks"][5][field] = value
        mutations.append(changed)
    changed = deepcopy(roadmap)
    changed["tasks"][5]["gated_on"][0]["artifact_field"] = "mutated_ready_score"
    mutations.append(changed)
    assert all(
        mod.evaluate_contract(markdown, candidate)["passed"] is False for candidate in mutations
    )


def test_req_report_7288_authority_selection_is_milestone_bound(tmp_path: Path) -> None:
    """REQ-REPORT-7288 accepts active or staged V641 and nothing stale."""

    root = _private_root(tmp_path, milestone="2026.09.640")
    staged = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged, sort_keys=False))
    authority, roadmap, _content, candidates = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap is not None
    assert [row["available"] for row in candidates] == [False, True]
    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, candidates = mod.select_yaml_authority(root)
    assert authority is roadmap is content is None
    assert all(row["available"] is False for row in candidates)


def test_scenario_report_7288_gates_use_five_real_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7288-GATES retains evaluator and quarantine authority."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_replays(roadmap, tmp_path / "gates")
    assert len(rows) == 40
    assert mod.gate_replays_complete(rows) is True
    for edge_index in range(8):
        edge = [row for row in rows if row["edge_index"] == edge_index]
        assert [row["case"] for row in edge] == [
            "passing",
            "false",
            "missing_field",
            "missing_file",
            "quarantined",
        ]
        assert [row["conductor_passed"] for row in edge] == [
            True,
            False,
            False,
            False,
            True,
        ]
        assert [row["experiment_precondition_passed"] for row in edge] == [
            True,
            False,
            False,
            False,
            False,
        ]
        assert edge[1]["observed_value"] == 0
        assert "quarantined" in edge[-1]["experiment_precondition_reason"].lower()
        assert all(row["research_result"] is False for row in edge)
    assert mod.gate_replays_complete(rows[:-1]) is False
    changed = deepcopy(rows)
    changed[-1]["experiment_precondition_passed"] = True
    assert mod.gate_replays_complete(changed) is False
    for index, field, value in (
        (0, "case", "false"),
        (0, "conductor_passed", False),
        (1, "observed_value", 2),
    ):
        changed = deepcopy(rows)
        changed[index][field] = value
        assert mod.gate_replays_complete(changed) is False


def test_scenario_report_7288_sources_are_bounded_and_nonlocal() -> None:
    """SCENARIO-REPORT-7288-SOURCES preserves failed fetches as observations."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _fake_fetch(url)

    rows, access = mod.collect_source_rows(fetch)
    assert len(calls) == len(rows) == len(access) == 4
    assert {row["method_family"] for row in rows} == {
        "source_freshness",
        "fixed_budget_hypothesis_mixtures",
        "sqlite_persistent_journals",
        "deferred_kan_tsu",
    }
    assert {row["disposition"] for row in rows} == {"adapt", "implement", "defer"}
    assert all(row["publication_or_version_date"] for row in rows)
    assert all(row["adapted_mechanism"] for row in rows)
    assert all(row["falsifier"] for row in rows)
    assert all(row["deferred_boundary"] for row in rows)
    assert all(row["local_evidence_claimed"] is False for row in rows)
    assert all(row["observed_access"] == "http_success" for row in rows)

    def unavailable(_url: str) -> dict[str, Any]:
        raise OSError("offline")

    offline, observations = mod.collect_source_rows(unavailable)
    assert all(row["observed_access"] == "access_failed" for row in offline)
    assert all(row["error"] for row in observations)
    assert mod.source_rows_complete(offline) is True
    assert mod.observed_version("[version:v1] [version:v3]") == "v3"
    assert mod.observed_version("no marker") is None


def test_scenario_report_7288_artifact_is_evidence_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7288-ARTIFACT validates exact, forged, and blocked states."""

    root = _private_root(tmp_path)
    monkeypatch.setattr(mod, "run_required_validation", lambda *_args: _passing_receipts())
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["contract_complete_score"] == 1
    assert artifact["rows"] == artifact["contract_rows"]
    assert len(artifact["gate_control_rows"]) == 40
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert mod.validate_artifact(artifact, root=root) == []
    assert json.loads((root / mod.DEFAULT_OUTPUT_PATH).read_text()) == artifact

    for field, value in (
        ("contract_complete_score", 0),
        ("model_invoked", True),
        ("rows", []),
        ("reproducibility_checksum", "sha256:forged"),
        ("schema", "wrong"),
        ("experiment_id", "exp9999"),
        ("status", "running"),
        ("started_at_utc", "not-a-date"),
        ("completed_at_utc", "2026-09-14T00:00:00"),
        ("field_principles", {}),
        ("execution_host", "wrong-host"),
        ("duration_s", 0),
        ("random_seed", 0),
        ("source_artifact_hashes", {}),
        ("validation_receipts", []),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.validate_artifact(changed, root=root)
    changed = deepcopy(artifact)
    changed.pop("schema")
    assert mod.validate_artifact(changed, root=root) == ["missing_required_field:schema"]
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"missing.json": "sha256:missing"}
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=root)
    changed = deepcopy(artifact)
    first_hash_path = next(iter(changed["source_artifact_hashes"]))
    changed["source_artifact_hashes"][first_hash_path] = "sha256:wrong"
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=root)
    changed = deepcopy(artifact)
    changed["validation_receipts"][0]["command"] = ""
    assert "validation_receipt_shape_invalid" in mod.validate_artifact(changed, root=root)
    assert mod.validate_artifact([]) == ["artifact_mapping_required"]

    disqualified = deepcopy(artifact)
    disqualified["contract_rows"][0]["checks"]["phase"] = False
    disqualified["contract_rows"][0]["passed"] = False
    disqualified["rows"] = disqualified["contract_rows"]
    mod._apply_terminal_state(disqualified)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "contract_parity"
    inconsistent = deepcopy(disqualified)
    inconsistent["contract_rows"][0]["passed"] = True
    assert "contract_row_reduction" in mod.independent_reduce(inconsistent)

    stale = deepcopy(artifact)
    stale["markdown_milestone"] = "2026.09.640"
    assert mod._failure_summary(stale)["failed_check"] == "markdown_milestone"
    incomplete = deepcopy(artifact)
    incomplete["source_dispositions"] = []
    assert mod._failure_summary(incomplete)["failed_check"] == "acceptance:source_dispositions"

    (root / mod.HISTORY_PATH).unlink()
    blocked = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / "results/blocked.json",
        raw_dir=root / "results/raw/blocked",
        checkpoint_path=root / "results/checkpoints/blocked.json",
        fetcher=_fake_fetch,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["contract_complete_score"] == 0
    assert blocked["gate_check_summary"]["upstream"] == str(mod.HISTORY_PATH)
    assert blocked["gate_check_summary"]["observed_value"]
    assert mod.validate_artifact(blocked, root=root) == []


def test_req_report_7288_validation_scope_and_e2e(tmp_path: Path) -> None:
    """REQ-REPORT-7288 binds scoped checks and the file-to-gate receipt."""

    commands = mod.validation_commands(ROOT, ROOT / mod.RAW_CANDIDATE_PATH, mod.ACTIVE_ROADMAP_PATH)
    assert [name for name, _command in commands] == list(mod.VALIDATION_COMMAND_NAMES)
    assert mod.VALIDATION_COMMAND_NAMES[mod.PRETERMINAL_VALIDATION_COUNT] == (
        "independent_reducer"
    )
    assert dict(commands)["full_python_suite"][:2] == [
        str(ROOT / ".venv/bin/pytest"),
        "tests/python",
    ]
    assert "-n" in dict(commands)["focused_pytest"]
    assert dict(commands)["adversarial"][-1].endswith(str(mod.RAW_CANDIDATE_PATH))
    receipt = mod.e2e_contract_receipt(ROOT, tmp_path / "e2e")
    assert receipt["passed"] is True
    assert receipt["contract_rows"] == 14
    assert receipt["gate_control_rows"] == 40
    assert receipt["diagnostic_sha256"].startswith("sha256:")

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    contract = mod.evaluate_contract((ROOT / mod.DESIGN_PATH).read_text(), roadmap)
    gate_rows = mod.run_gate_replays(roadmap, tmp_path / "reduce-gates")
    source_rows, _access = mod.collect_source_rows(_fake_fetch)
    payload = {
        "contract_rows": contract["contract_rows"],
        "gate_control_rows": gate_rows,
        "source_dispositions": source_rows,
        "raw_authority_rows": [
            {"source_sha256": "sha256:a", "raw_sha256": "sha256:a"},
            {"source_sha256": "sha256:b", "raw_sha256": "sha256:b"},
        ],
    }
    assert mod.independent_reduce(payload) == []
    for section in ("contract_rows", "gate_control_rows", "source_dispositions"):
        changed = deepcopy(payload)
        changed[section].pop()
        assert mod.independent_reduce(changed)
    changed = deepcopy(payload)
    changed["raw_authority_rows"][0]["raw_sha256"] = "sha256:changed"
    assert "raw_authority_reduction" in mod.independent_reduce(changed)

    malformed_table = (
        "| Task ID | Deliverable | Phase |\n"
        "|---|---|---|\n"
        "| too-short |\n"
        "| exp7288-source-contract | results/a.json | not-a-phase |\n"
    )
    assert mod._markdown_phases(malformed_table) == {}
    assert mod._markdown_phases("no contract table") == {}

    bad_history_root = _private_root(tmp_path / "bad-history")
    (bad_history_root / mod.HISTORY_PATH).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        mod.build_sidecars(bad_history_root, bad_history_root / mod.RAW_DIR, gate_rows, [])

    missing_spec_root = _private_root(tmp_path / "missing-spec")
    (missing_spec_root / mod.SPEC_PATH).unlink()
    with mod._configured_runtime():
        preconditions, *_ = mod._preconditions(
            missing_spec_root,
            missing_spec_root / mod.DEFAULT_OUTPUT_PATH,
            missing_spec_root / mod.RAW_DIR,
            missing_spec_root / mod.CHECKPOINT_PATH,
        )
    requirement = next(row for row in preconditions if row["check"] == "driving_requirement")
    assert requirement["available"] is False
    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260913")
