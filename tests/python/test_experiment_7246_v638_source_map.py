"""Tests for the V638 source-ingestion and execution map.

Spec refs: REQ-REPORT-7246 and SCENARIO-REPORT-7246-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import pytest
import yaml

from carnot import experiment_7246_v638_source_map as mod


ROOT = Path(__file__).resolve().parents[2]


def _copy_file(root: Path, relative: Path) -> None:
    """Copy one declared input while keeping its repository-relative path."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def _private_root(tmp_path: Path, *, milestone: str = mod.MILESTONE) -> Path:
    """Create the smallest complete repository view used by artifact tests."""

    root = tmp_path / "repo"
    for relative in (*mod.SOURCE_PATHS, *mod.HISTORY_PATHS):
        _copy_file(root, relative)
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    roadmap["milestone"] = milestone
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    for relative in (mod.DEFAULT_OUTPUT_PATH.parent, mod.RAW_DIR, mod.CHECKPOINT_PATH.parent):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _matching_markdown(roadmap: dict[str, Any]) -> str:
    """Render only the independent table syntax consumed by the shipped parser."""

    lines = [
        "# V638 design",
        "",
        f"**Milestone:** {mod.MILESTONE}",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Title | Deliverable | Structured gates |",
        "|---|---|---|---|---|",
    ]
    for index, task in enumerate(roadmap["tasks"], 1):
        gates = task.get("gated_on", [])
        gate_text = "; ".join(
            f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {gate['value']}"
            for gate in gates
        )
        lines.append(
            f"| {index} | {task['id']} | {task['title']} | {task['deliverable']} | "
            f"{gate_text or 'None'} |"
        )
    return "\n".join(lines) + "\n"


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return deterministic bytes for all five bounded source requests."""

    titles = {
        "2608.15277": "Memory-Bounded Continuation of Greedy Sampling for Continual Anomaly Detection",
        "2603.29109": "SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization",
        "2608.00859": (
            "SparseKAN: Compressing Kolmogorov--Arnold Networks Across Basis Functions, "
            "Neurons, and Bits"
        ),
    }
    for source_id, title in titles.items():
        if source_id in url:
            return {
                "ok": True,
                "status_code": 200,
                "body": f"<title>{title}</title> [v1]",
                "error": None,
            }
    count = 35 if "2507.02092" in url else 8
    return {
        "ok": True,
        "status_code": 200,
        "body": json.dumps({"data": [{"citingPaper": {"title": str(i)}} for i in range(count)]}),
        "error": None,
    }


def _passing_receipts() -> list[dict[str, Any]]:
    """Return complete validation rows without launching nested pytest in unit tests."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_sha256": mod.sha256_bytes(b"fixture log"),
            "passed": True,
            "baseline_failure": False,
        }
        for name in mod.VALIDATION_COMMAND_NAMES
    ]


def test_scenario_report_7246_contract_records_current_markdown_mismatch() -> None:
    """SCENARIO-REPORT-7246-CONTRACT keeps the activated structural failure."""

    authority, roadmap, yaml_bytes, candidates = mod.select_yaml_authority(ROOT)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap is not None and yaml_bytes == (ROOT / authority).read_bytes()
    assert candidates[0]["available"] is True
    contract = mod.evaluate_contract((ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
    assert contract["passed"] is False
    assert len(contract["contract_rows"]) == 14
    assert contract["observed_id_order"] == list(mod.EXPECTED_ID_ORDER)
    assert contract["markdown_milestone"] == "2026.09.636"
    assert all(row["passed"] is False for row in contract["contract_rows"])


def test_req_report_7246_selects_only_matching_milestone(tmp_path: Path) -> None:
    """REQ-REPORT-7246 falls back to staged YAML and blocks without authority."""

    root = _private_root(tmp_path, milestone="2026.09.637")
    staged = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    (root / mod.NEXT_ROADMAP_PATH).write_text(
        yaml.safe_dump(staged, sort_keys=False), encoding="utf-8"
    )
    authority, roadmap, _content, candidates = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap is not None
    assert [row["available"] for row in candidates] == [False, True]
    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, candidates = mod.select_yaml_authority(root)
    assert authority is roadmap is content is None
    assert all(row["available"] is False for row in candidates)


def test_scenario_report_7246_gates_replay_all_four_cases(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7246-GATES uses the real evaluator on every edge."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    rows = mod.run_gate_replays(roadmap, tmp_path / "gates")
    assert len(rows) == 20
    assert mod.gate_replays_complete(rows) is True
    for edge_index in range(5):
        edge = [row for row in rows if row["edge_index"] == edge_index]
        assert [row["case"] for row in edge] == [
            "passing",
            "zero",
            "absent_field",
            "absent_file",
        ]
        assert [row["passed"] for row in edge] == [True, False, False, False]
        assert edge[1]["observed_value"] == 0
        assert "NO field" in edge[2]["reason"]
        assert "not found" in edge[3]["reason"]
    assert mod.gate_replays_complete(rows[:-1]) is False
    malformed = deepcopy(roadmap)
    malformed["tasks"][3]["gated_on"][0]["upstream"] = "bad-id"
    with pytest.raises(ValueError, match="invalid upstream task id"):
        mod.run_gate_replays(malformed, tmp_path / "bad")


def test_scenario_report_7246_sources_map_seven_methods_with_five_requests() -> None:
    """SCENARIO-REPORT-7246-SOURCES retains cost, controls, and access limits."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _fake_fetch(url)

    methods, access = mod.collect_source_evidence(fetch)
    assert len(methods) == 7 and len(access) == 5 and len(calls) == 5
    assert {row["method_id"] for row in methods} == {
        "contcore",
        "semloc",
        "grounding",
        "retrieval_control",
        "sparsekan",
        "ebt",
        "arm_ebm",
    }
    assert all(row["implementation_cost"] for row in methods)
    assert all(row["falsifying_control"] for row in methods)
    assert all("retired_overlap" in row for row in methods)
    assert all(row["access_status"] for row in methods)
    assert [row["observed_count"] for row in access[-2:]] == [35, 8]

    def unavailable(_url: str) -> dict[str, Any]:
        raise OSError("offline")

    limited_methods, limited_access = mod.collect_source_evidence(unavailable)
    assert all(
        row["access_status"] == "access_failed_cached_refresh_retained" for row in limited_access
    )
    assert sum(row["access_status"].startswith("access_failed") for row in limited_methods) == 5


def test_scenario_report_7246_sidecars_isolate_history_and_fixtures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7246-SIDECARS hashes model history and negative inputs."""

    root = _private_root(tmp_path)
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    gate_rows = mod.run_gate_replays(roadmap, tmp_path / "gate-inputs")
    _methods, access = mod.collect_source_evidence(_fake_fetch)
    metadata = mod.build_evidence_sidecars(root, root / mod.RAW_DIR, gate_rows, access)
    assert set(metadata) == {"history", "negative_fixtures", "source_access"}
    for receipt in metadata.values():
        path = root / receipt["path"]
        assert path.is_file() and receipt["sha256"] == mod.sha256(path)
    history = json.loads((root / metadata["history"]["path"]).read_text(encoding="utf-8"))
    assert len(history["historical_artifacts"]) == 13
    assert history["current_invocation"] == {
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
    }
    assert any(row["quarantined"] for row in history["historical_artifacts"])
    negative = json.loads(
        (root / metadata["negative_fixtures"]["path"]).read_text(encoding="utf-8")
    )
    assert len(negative["fixture_rows"]) == 15
    assert all(row["research_result"] is False for row in negative["fixture_rows"])


def test_req_report_7246_archive_state_is_observed() -> None:
    """REQ-REPORT-7246 records the actual V637 archive without rewriting it."""

    row = mod.archive_lag_row(ROOT)
    assert row["prompt_expected_latest_milestone"] == "2026.09.636"
    assert row["observed_latest_milestone"] == "2026.09.637"
    assert row["archive_advanced_since_prompt"] is True
    assert row["research_complete_rewritten"] is False


def test_scenario_report_7246_artifact_recomputes_success_and_disqualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7246-ARTIFACT derives both structural terminal classes."""

    root = _private_root(tmp_path)
    monkeypatch.setattr(
        mod, "run_validation_commands", lambda *_args, **_kwargs: _passing_receipts()
    )
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["source_ingestion_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "markdown_milestone"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["rows"] == artifact["contract_rows"]
    assert mod.validate_artifact(artifact, root=root) == []

    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    (root / mod.DESIGN_PATH).write_text(_matching_markdown(roadmap), encoding="utf-8")
    complete = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / "results/complete.json",
        raw_dir=root / "results/raw/complete",
        checkpoint_path=root / "results/checkpoints/complete.json",
        fetcher=_fake_fetch,
    )
    assert complete["verdict_class"] == "positive"
    assert complete["source_ingestion_complete_score"] == 1
    assert all(row["passed"] for row in complete["acceptance_gate_results"])
    assert mod.validate_artifact(complete, root=root) == []

    mutations = (
        lambda value: value.__setitem__("schema", "wrong"),
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.__setitem__("run_date", "20260911"),
        lambda value: value.__setitem__("execution_host", "wrong"),
        lambda value: value.__setitem__("duration_s", 0),
        lambda value: value.__setitem__("model_invoked", True),
        lambda value: value.__setitem__("verifier_is_oracle", True),
        lambda value: value.__setitem__("rows", []),
        lambda value: value.__setitem__("sample_size_budget", {}),
        lambda value: value.__setitem__("source_artifact_hashes", {}),
        lambda value: value.__setitem__("source_ingestion_complete_score", 0),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    )
    for mutate in mutations:
        changed = deepcopy(complete)
        mutate(changed)
        assert mod.validate_artifact(changed, root=root)
    assert mod.validate_artifact([], root=root) == ["artifact_mapping_required"]


def test_req_report_7246_missing_authority_blocks_without_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7246 uses blocked for an absent external prerequisite."""

    root = _private_root(tmp_path, milestone="2026.09.637")
    monkeypatch.setattr(
        mod, "run_validation_commands", lambda *_args, **_kwargs: _passing_receipts()
    )
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["field"] == "milestone"
    assert mod.validate_artifact(artifact, root=root) == []


def test_scenario_report_7246_validation_is_scoped_and_hashed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7246-VALIDATION preserves commands, exits, and log hashes."""

    commands = mod.validation_commands(ROOT, ROOT / mod.CHECKPOINT_PATH, mod.ACTIVE_ROADMAP_PATH)
    assert [name for name, _command in commands] == list(mod.VALIDATION_COMMAND_NAMES)
    assert dict(commands)["arc_floor"][-2:] == ["--min", "1"]
    receipts = mod.run_validation_commands(
        tmp_path,
        tmp_path / "checkpoint.json",
        {"yaml_authority_path": str(mod.ACTIVE_ROADMAP_PATH)},
        0.0,
        commands=(("probe", [str(ROOT / ".venv/bin/python"), "-u", "-c", "print('ok')"]),),
    )
    assert receipts[0]["exit_code"] == 0 and receipts[0]["passed"] is True
    assert receipts[0]["log_sha256"].startswith("sha256:")
    assert mod.date_argument("20260912") == "20260912"
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260911")
    with pytest.raises(ValueError):
        mod.date_argument("bad")


def test_scenario_report_7246_timeout_closes_descendant_output_pipe() -> None:
    """SCENARIO-REPORT-7246-VALIDATION bounds descendants that inherit stdout."""

    started = time.monotonic()
    receipt = mod._run_streaming_command(
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "-c",
            (
                "import subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)']);"
                "print('spawned',flush=True);time.sleep(3)"
            ),
        ],
        cwd=ROOT,
        timeout_s=0.1,
        heartbeat_s=0.05,
        operation="exp7246:timeout-regression",
    )
    assert time.monotonic() - started < 1.5
    assert receipt["timed_out"] is True and receipt["exit_code"] == 124
    assert "spawned" in receipt["stdout"]

    drained = mod._run_streaming_command(
        ["/bin/echo", "drained-after-fast-exit"],
        cwd=ROOT,
        timeout_s=1,
        heartbeat_s=1,
        operation="exp7246:fast-output-drain",
    )
    assert drained["exit_code"] == 0 and "drained-after-fast-exit" in drained["stdout"]

    closed = mod._run_streaming_command(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "import os,time;os.close(1);os.close(2);time.sleep(.05)",
        ],
        cwd=ROOT,
        timeout_s=1,
        heartbeat_s=1,
        operation="exp7246:closed-output",
    )
    assert closed["exit_code"] == 0 and closed["stdout"] == ""


def test_req_report_7246_defensive_branches_reject_malformed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7246 covers malformed rows, sources, hashes, and receipts."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    gates = mod.run_gate_replays(roadmap, tmp_path / "defensive-gates")
    for mutate in (
        lambda rows: rows[0].__setitem__("case", "wrong"),
        lambda rows: rows[0].__setitem__("passed", False),
        lambda rows: rows[1].__setitem__("observed_value", 2),
    ):
        changed = deepcopy(gates)
        mutate(changed)
        assert mod.gate_replays_complete(changed) is False

    assert mod._observed_version("no version marker") is None

    def malformed_citations(url: str) -> dict[str, Any]:
        receipt = _fake_fetch(url)
        if "citations" in url:
            receipt["body"] = "{"
        return receipt

    _methods, access = mod.collect_source_evidence(malformed_citations)
    assert all(row["source_delta"] == "citation_result_changed_or_malformed" for row in access[-2:])

    root = _private_root(tmp_path / "sidecar")
    first_history = root / mod.HISTORY_PATHS[0]
    first_history.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        mod.build_evidence_sidecars(root, root / mod.RAW_DIR, gates, access)

    monkeypatch.setattr(mod, "SOURCE_PATHS", (Path("missing-source"),))
    monkeypatch.setattr(mod, "HISTORY_PATHS", ())
    assert mod._hash_sources(root, None) == {"missing-source": None}
    monkeypatch.undo()

    root = _private_root(tmp_path / "preconditions")
    (root / mod.SPEC_PATH).unlink()
    (root / mod.HISTORY_PATHS[0]).unlink()
    rows, *_rest = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    assert next(row for row in rows if row["check"] == "driving_requirement")["available"] is False
    assert (
        next(row for row in rows if row["check"].startswith("v637_terminal:"))["available"] is False
    )
    assert mod._failed_precondition({"preconditions_checked": None}) is None

    artifact = mod._base_artifact(
        mod.RUN_DATE, tmp_path / "defensive-checkpoint.json", "2026-09-12T00:00:00+00:00"
    )
    artifact["markdown_milestone"] = mod.MILESTONE
    artifact["preconditions_checked"] = []
    artifact["contract_rows"] = [
        {
            "unit_id": mod.FIRST_TASK_ID,
            "markdown": {"id": "wrong"},
            "yaml": {"id": mod.FIRST_TASK_ID},
            "checks": {"id": False},
            "passed": False,
        }
    ]
    assert mod._failure_summary(artifact)["failed_check"] == "contract_parity"
    assert set(
        mod.independent_reduce(
            {
                "contract_rows": [],
                "gate_replay_rows": [],
                "source_method_rows": [],
                "raw_source_rows": [],
            }
        )
    ) == {
        "contract_row_count",
        "gate_replay_reduction",
        "source_method_count",
        "raw_hash_reduction",
    }
    bad_reduction = {
        "contract_rows": [{"passed": True, "checks": {"id": False}} for _ in range(14)],
        "gate_replay_rows": gates,
        "source_method_rows": list(mod.METHODS),
        "raw_source_rows": [
            {"source_sha256": "one", "raw_sha256": "one"},
            {"source_sha256": "two", "raw_sha256": "two"},
        ],
    }
    assert mod.independent_reduce(bad_reduction) == ["contract_row_reduction"]

    base = mod._base_artifact(
        mod.RUN_DATE, tmp_path / "validate-checkpoint.json", "2026-09-12T00:00:00+00:00"
    )
    missing = dict(base)
    missing.pop("schema")
    assert mod.validate_artifact(missing, root=tmp_path)[0].startswith("missing_required_field:")
    base.update(
        experiment_id="wrong",
        completed_at_utc="invalid",
        started_at_utc="2026-09-12T00:00:00",
        field_principles={},
        random_seed=0,
        source_artifact_hashes={"missing": "sha256:wrong"},
        validation_receipts=[{"name": "wrong"}],
    )
    errors = mod.validate_artifact(base, root=tmp_path)
    assert {
        "identity_invalid",
        "completed_at_utc_invalid",
        "started_at_utc_invalid",
        "field_principles_invalid",
        "random_seed_invalid",
        "source_hash_mismatch",
        "validation_receipts_invalid",
    }.issubset(errors)
    base["validation_receipts"] = [
        {
            "name": name,
            "command": "" if index == 0 else "fixture",
            "exit_code": 0,
            "log_sha256": "bad",
        }
        for index, name in enumerate(mod.VALIDATION_COMMAND_NAMES)
    ]
    assert "validation_receipt_shape_invalid" in mod.validate_artifact(base, root=tmp_path)

    runner_artifact = mod._base_artifact(
        mod.RUN_DATE, tmp_path / "runner-checkpoint.json", "2026-09-12T00:00:00+00:00"
    )
    runner_artifact["yaml_authority_path"] = str(mod.ACTIVE_ROADMAP_PATH)
    monkeypatch.setattr(
        mod,
        "_run_streaming_command",
        lambda *_args, **_kwargs: {
            "exit_code": 1,
            "duration_s": 0.01,
            "timed_out": False,
            "stdout": "failed",
            "stderr": "baseline",
        },
    )
    receipts = mod.run_validation_commands(
        tmp_path,
        tmp_path / "runner-checkpoint.json",
        runner_artifact,
        time.monotonic(),
        commands=(("full_python_suite", ["false"]),),
    )
    assert receipts[0]["baseline_failure"] is True
    assert runner_artifact["baseline_failures"] == receipts
