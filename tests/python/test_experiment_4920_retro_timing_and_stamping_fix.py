"""Tests for REQ-REPORT-4920 / SCENARIO-REPORT-4920."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from carnot import experiment_4920_retro_timing_and_stamping_fix as mod
from carnot.reporting import retro_timing_mtime_fallback as timing
from carnot.reporting import runtime_stamping


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mtime_ns(hour: int, minute: int, second: int) -> int:
    dt = datetime(2026, 6, 28, hour, minute, second, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _set_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _arm_payload(exp_id: int, *, duration_s: float | None, gpu: bool = False) -> JsonDict:
    payload: JsonDict = {
        "experiment_id": exp_id,
        "honest_verdict": f"complete_exp{exp_id}",
        "duration_s": duration_s,
        "inference_substrate": "live_llm_inference" if gpu else "aggregation_from_upstream_artifacts",
        "compute_bound": None,
    }
    if gpu:
        payload["generator_backend"] = "gpu0_cuda"
    return payload


def _make_v452_root(root: Path) -> dict[int, Path]:
    _write_text(root / "AGENTS.md", "# AGENTS\n")
    _write_text(root / "CODEX.md", "# CODEX\n")
    _write_text(root / "scripts" / "research_conductor.py", "# conductor placeholder\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "schema": "carnot.operational_retro.v64",
            "milestone": "2026.06.452",
            "experiments_completed": 0,
            "total_wall_time_minutes": 0,
            "compute_bound_experiments_count": 0,
            "gpu_idle_on_compute_bound_tasks": None,
            "summary": "false-zero despite on-disk artifacts exp4902-exp4912",
        },
    )

    names = {
        4902: "archive_451_activate_452",
        4903: "env_grounded_location_pruned_search",
        4904: "latent_action_interface",
        4905: "levelup_attempt",
        4906: "self_play_verifier_checkpoint",
        4907: "heldout_first_win_readiness",
        4908: "env_grounded_search_audit",
        4909: "submission_package_harden",
        4910: "kv260_continuity",
        4911: "sota_ingestion_v453_frontier",
        4912: "capstone_v452",
    }
    mtimes = {
        4902: _mtime_ns(5, 36, 2),
        4903: _mtime_ns(5, 52, 37),
        4904: _mtime_ns(6, 13, 19),
        4905: _mtime_ns(6, 31, 15),
        4906: _mtime_ns(6, 45, 11),
        4907: _mtime_ns(7, 55, 42),
        4908: _mtime_ns(8, 12, 29),
        4909: _mtime_ns(8, 31, 6),
        4910: _mtime_ns(8, 43, 55),
        4911: _mtime_ns(9, 5, 19),
        4912: _mtime_ns(9, 22, 5),
    }
    paths: dict[int, Path] = {}
    for exp_id, name in names.items():
        path = root / "results" / f"experiment_{exp_id}_{name}.json"
        paths[exp_id] = path
        _write_json(
            path,
            _arm_payload(
                exp_id,
                duration_s=None if exp_id in {4905, 4906} else 0.0001,
                gpu=exp_id in {4903, 4904, 4907},
            ),
        )
        _set_mtime(path, mtimes[exp_id])
    return paths


def test_req_report_4920_spec_declares_timing_and_stamping_contract() -> None:
    """REQ-REPORT-4920: OpenSpec declares the standalone fallback contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    for marker in (
        str(mod.OUTPUT_REL_PATH),
        "retro_timing_mtime_fallback.py",
        "runtime_stamping.py",
        "results/experiment_4905_levelup_attempt.json",
        "results/experiment_4906_self_play_verifier_checkpoint.json",
        "research_conductor_modified",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4920_pure_mtime_window_is_deterministic() -> None:
    """SCENARIO-REPORT-4920: pure mtime core reconstructs a non-zero window."""

    records = [
        timing.ArtifactMtimeRecord(
            path="results/experiment_4912_capstone_v452.json",
            mtime_ns=_mtime_ns(9, 22, 5),
            compute_bound=False,
        ),
        timing.ArtifactMtimeRecord(
            path="results/experiment_4904_latent_action_interface.json",
            mtime_ns=_mtime_ns(6, 13, 19),
            compute_bound=True,
        ),
        timing.ArtifactMtimeRecord(
            path="results/experiment_4902_archive_451_activate_452.json",
            mtime_ns=_mtime_ns(5, 36, 2),
            compute_bound=False,
        ),
        timing.ArtifactMtimeRecord(
            path="results/experiment_4907_heldout_first_win_readiness.json",
            mtime_ns=_mtime_ns(7, 55, 42),
            compute_bound=True,
        ),
        timing.ArtifactMtimeRecord(
            path="results/experiment_4903_env_grounded_location_pruned_search.json",
            mtime_ns=_mtime_ns(5, 52, 37),
            compute_bound=True,
        ),
    ]

    window = timing.reconstruct_mtime_window("2026.06.452", records)

    assert window["milestone"] == "2026.06.452"
    assert window["n_arms"] == 5
    assert window["window_start"] == "2026-06-28T05:36:02Z"
    assert window["window_end"] == "2026-06-28T09:22:05Z"
    assert window["wall_minutes"] == 226.05
    assert window["compute_bound_count"] == 3
    assert window == timing.reconstruct_mtime_window("2026.06.452", reversed(records))
    assert timing.reconstruct_mtime_window("2026.06.999", []) == {
        "milestone": "2026.06.999",
        "n_arms": 0,
        "window_start": None,
        "window_end": None,
        "wall_minutes": 0.0,
        "compute_bound_count": 0,
        "artifact_paths": [],
    }


def test_scenario_report_4920_stamping_helper_and_audit_find_missing_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4920: audit lists duration backfills for null fields."""

    stamped = runtime_stamping.stamp_runtime_metadata(
        {"honest_verdict": "success_demo"},
        started_s=10.0,
        finished_s=12.25,
        inference_substrate="live_llm_inference",
        compute_bound=True,
    )
    floored = runtime_stamping.stamp_runtime_metadata(
        {},
        started_s=1.0,
        finished_s=1.0,
        inference_substrate="aggregation_from_upstream_artifacts",
        compute_bound=False,
    )

    assert stamped["duration_s"] == 2.25
    assert stamped["inference_substrate"] == "live_llm_inference"
    assert stamped["compute_bound"] is True
    assert floored["duration_s"] == 0.0001
    assert floored["compute_bound"] is False

    exp4905 = tmp_path / "results" / "experiment_4905_levelup_attempt.json"
    exp4906 = tmp_path / "results" / "experiment_4906_self_play_verifier_checkpoint.json"
    exp4907 = tmp_path / "results" / "experiment_4907_heldout_first_win_readiness.json"
    _write_json(exp4905, {"duration_s": None, "inference_substrate": "offline", "compute_bound": None})
    _write_json(exp4906, {"duration_s": None, "inference_substrate": "live_llm_inference"})
    _write_json(exp4907, stamped)

    audit = runtime_stamping.audit_runtime_stamps([exp4907, exp4905, exp4906])

    assert audit["scanned_count"] == 3
    assert [row["experiment_id"] for row in audit["missing_by_field"]["duration_s"]] == [
        4905,
        4906,
    ]
    assert [row["experiment_id"] for row in audit["missing_by_field"]["compute_bound"]] == [
        4905,
        4906,
    ]
    assert audit["missing_by_field"]["inference_substrate"] == []
    assert [row["path"] for row in audit["missing_any"]] == [
        "results/experiment_4905_levelup_attempt.json",
        "results/experiment_4906_self_play_verifier_checkpoint.json",
    ]


def test_scenario_report_4920_run_writes_success_deliverable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4920: run writes the mtime/audit/proposal deliverable."""

    _make_v452_root(tmp_path)

    artifact = mod.run(root=tmp_path, started_s=100.0, now_s=100.25)

    assert artifact["honest_verdict"] == (
        "success_retro_timing_mtime_fallback_and_stamping_shipped"
    )
    assert artifact["mtime_fallback_window"]["n_arms"] == 11
    assert artifact["mtime_fallback_window"]["wall_minutes"] > 0
    assert artifact["mtime_fallback_window"]["compute_bound_count"] == 3
    assert [row["experiment_id"] for row in artifact["stamping_audit_missing_duration"]] == [
        4905,
        4906,
    ]
    assert artifact["mtime_fallback_module_path"] == str(mod.MTIME_FALLBACK_MODULE_REL_PATH)
    assert artifact["stamping_helper_path"] == str(mod.STAMPING_HELPER_REL_PATH)
    assert artifact["wiring_proposal_path"] == str(mod.WIRING_PROPOSAL_REL_PATH)
    assert (tmp_path / mod.WIRING_PROPOSAL_REL_PATH).exists()
    assert "assemble" in (tmp_path / mod.WIRING_PROPOSAL_REL_PATH).read_text(encoding="utf-8")
    assert artifact["research_conductor_modified"] is False
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_report_4920_missing_arm_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4920-BLOCKED-PRECONDITION: missing arm emits blocked_."""

    paths = _make_v452_root(tmp_path)
    paths[4906].unlink()

    artifact = mod.run(root=tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["honest_verdict"] == "blocked_missing_v452_arm_artifact"
    assert artifact["preconditions_checked"]["ok"] is False
    assert "results/experiment_4906_self_play_verifier_checkpoint.json" in (
        artifact["preconditions_checked"]["missing_arm_artifacts"]
    )
    assert artifact["mtime_fallback_window"]["n_arms"] == 0
    assert artifact["stamping_audit_missing_duration"] == []
    assert artifact["research_conductor_modified"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_report_4920_scanners_and_validation_cover_edge_cases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-REPORT-4920: scanners and schema validation reject malformed payloads."""

    _make_v452_root(tmp_path)
    records = timing.scan_milestone_records(tmp_path / "results", "2026.06.452")

    assert len(records) == 11
    assert sum(1 for record in records if record.compute_bound) == 3
    assert timing.compute_bound_from_artifact({"compute_bound": True}) is True
    assert timing.compute_bound_from_artifact({"compute_bound": False}) is False
    assert timing.compute_bound_from_artifact({"generator_backend": "gpu0_cuda"}) is True
    assert timing.compute_bound_from_artifact({"runtime": {"backend": "cuda"}}) is True
    assert timing.compute_bound_from_artifact({"runtime": [{"backend": "cuda"}]}) is True
    assert timing.compute_bound_from_artifact({"inference_substrate": "live_llm_inference"}) is False
    assert timing.find_milestone_arm_paths(tmp_path / "missing", "2026.06.452") == []
    assert timing.find_milestone_arm_paths(tmp_path / "results", "bad") == []
    assert timing._result_relative_path(Path("/tmp/root/results/a.json")) == "results/a.json"
    assert timing._result_relative_path(Path("/tmp/a.json")) == "/tmp/a.json"

    fallback_dir = tmp_path / "fallback_results"
    _write_json(fallback_dir / "experiment_1_probe_v452.json", {"duration_s": 1})
    assert [
        path.name for path in timing.find_milestone_arm_paths(fallback_dir, "2026.06.452")
    ] == ["experiment_1_probe_v452.json"]
    bad_range_dir = tmp_path / "bad_range_results"
    _write_json(bad_range_dir / "experiment_10_archive_451_activate_452.json", {})
    (bad_range_dir / "experiment_11_bad.json").write_text("{bad", encoding="utf-8")
    _write_json(bad_range_dir / "experiment_12_capstone_v452.json", {})
    assert len(timing.scan_milestone_records(bad_range_dir, "2026.06.452")) == 3

    bad_json = tmp_path / "results" / "experiment_4999_bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    plain_json = tmp_path / "plain.json"
    _write_json(plain_json, {"duration_s": None})
    list_json = tmp_path / "results" / "experiment_5000_list.json"
    list_json.write_text("[]", encoding="utf-8")
    audit = runtime_stamping.audit_runtime_stamps([bad_json, plain_json, list_json])
    assert audit["missing_any"][0]["path"] == "results/experiment_4999_bad.json"
    assert "json_error" in audit["missing_any"][0]["missing_fields"]
    assert any(row["path"] == str(plain_json) for row in audit["missing_any"])
    assert any(row["path"] == "results/experiment_5000_list.json" for row in audit["missing_any"])

    artifact = mod.run(root=tmp_path, started_s=2.0, now_s=2.5)
    assert mod.file_sha256(tmp_path / "missing.txt") == ""
    missing_retro = mod.run(root=tmp_path / "missing_retro", started_s=3.0, now_s=3.0)
    assert missing_retro["honest_verdict"] == "blocked_missing_operational_retro"
    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "done"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "missing_principle:mtime_fallback_window" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_research_conductor_modified" in mod.validate_artifact(
        {**artifact, "research_conductor_modified": True}
    )
    assert "invalid_mtime_fallback_window" in mod.validate_artifact(
        {**artifact, "mtime_fallback_window": {"n_arms": 0}}
    )
    assert "invalid_stamping_audit_missing_duration" in mod.validate_artifact(
        {**artifact, "stamping_audit_missing_duration": []}
    )
    assert "invalid_stamping_audit_missing_duration" in mod.validate_artifact(
        {**artifact, "stamping_audit_missing_duration": {}}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod.main(root=tmp_path) == 0
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced"])
    assert mod.main(root=tmp_path) == 2
