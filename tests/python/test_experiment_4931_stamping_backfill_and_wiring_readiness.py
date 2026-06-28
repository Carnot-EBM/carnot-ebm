"""Tests for REQ-REPORT-4931 / SCENARIO-REPORT-4931."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from carnot import experiment_4931_stamping_backfill_and_wiring_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mtime_ns(hour: int, minute: int) -> int:
    return int(datetime(2026, 6, 28, hour, minute, tzinfo=timezone.utc).timestamp() * 1e9)


def _set_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _seed_required_files(root: Path) -> None:
    _write_text(root / "scripts" / "research_conductor.py", "# conductor\n")
    _write_text(root / mod.MTIME_FALLBACK_MODULE_REL_PATH, "# mtime fallback shipped\n")
    _write_text(root / mod.STAMPING_HELPER_REL_PATH, "# runtime stamping shipped\n")
    _write_text(
        root / mod.WIRING_PROPOSAL_REL_PATH,
        "call `carnot.reporting.retro_timing_mtime_fallback.mtime_fallback_window` "
        "in the retro prompt-assembly TIMING DATA path for the operator wire\n",
    )
    _write_json(
        root / mod.EXP4920_RESULT_REL_PATH,
        {
            "honest_verdict": "success_retro_timing_mtime_fallback_and_stamping_shipped",
            "mtime_fallback_module_path": str(mod.MTIME_FALLBACK_MODULE_REL_PATH),
            "stamping_helper_path": str(mod.STAMPING_HELPER_REL_PATH),
            "wiring_proposal_path": str(mod.WIRING_PROPOSAL_REL_PATH),
            "research_conductor_modified": False,
        },
    )


def _arm_payload(
    exp_id: int,
    *,
    duration_s: float | None = 1.0,
    substrate: str = "aggregation_from_upstream_artifacts",
    compute_bound: bool | None = None,
    backend: str | None = None,
) -> JsonDict:
    payload: JsonDict = {
        "experiment": f"experiment_{exp_id}_demo",
        "experiment_id": exp_id,
        "honest_verdict": f"complete_exp{exp_id}",
        "inference_substrate": substrate,
        "random_seed": exp_id,
    }
    if duration_s is not None:
        payload["duration_s"] = duration_s
    if compute_bound is not None:
        payload["compute_bound"] = compute_bound
    if backend is not None:
        payload["model_specs"] = {"backend": backend}
    return payload


def _make_v454_arms(root: Path, exp_ids: tuple[int, ...]) -> dict[int, Path]:
    names = {
        4924: "archive_453_activate_454",
        4925: "levelup_attempt",
        4926: "levelup_attempt",
        4927: "self_play_verifier_checkpoint",
        4928: "heldout_first_win_readiness",
        4929: "bank_and_efficiency_audit",
        4930: "submission_package_harden",
        4932: "kv260_continuity",
        4933: "matm_similarity_retrieval_efficiency",
        4934: "capstone_v454",
    }
    paths: dict[int, Path] = {}
    for offset, exp_id in enumerate(exp_ids):
        path = root / "results" / f"experiment_{exp_id}_{names[exp_id]}.json"
        if exp_id == 4927:
            payload = _arm_payload(exp_id, duration_s=None, substrate="live_llm_inference")
        elif exp_id == 4928:
            payload = _arm_payload(
                exp_id,
                duration_s=673.601274,
                substrate="live_llm_inference",
                backend="gpu0_cuda",
            )
        elif exp_id == 4930:
            payload = _arm_payload(
                exp_id,
                duration_s=1.293081,
                backend="igpu_hip",
            )
        elif exp_id == 4933:
            payload = _arm_payload(
                exp_id,
                duration_s=12.0,
                substrate="live_llm_inference",
                backend="cuda",
            )
        elif exp_id in {4925, 4926}:
            payload = _arm_payload(
                exp_id,
                duration_s=None,
                substrate="offline_arcade_reproduction_gate_no_llm",
            )
        else:
            payload = _arm_payload(exp_id)
        _write_json(path, payload)
        _set_mtime(path, _mtime_ns(15 + (offset // 4), (offset * 7) % 60))
        paths[exp_id] = path
    return paths


def test_req_report_4931_spec_declares_backfill_contract() -> None:
    """REQ-REPORT-4931: OpenSpec declares the .454 stamping/readiness contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    for marker in (
        str(mod.OUTPUT_REL_PATH),
        "duration_s",
        "compute_bound_count>=3",
        "research_conductor_modified",
        "wiring_proposal_reconfirmed",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4931_success_backfills_and_preserves_mtime(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4931: missing runtime fields are backfilled idempotently."""

    _seed_required_files(tmp_path)
    arms = _make_v454_arms(
        tmp_path,
        (4924, 4925, 4926, 4927, 4928, 4929, 4930, 4932, 4933, 4934),
    )
    original_mtime = arms[4925].stat().st_mtime_ns

    artifact = mod.run(root=tmp_path, started_s=10.0, now_s=10.5)

    assert artifact["honest_verdict"] == (
        "success_v454_stamping_backfilled_and_mtime_window_confirmed"
    )
    assert artifact["mtime_fallback_window"]["n_arms"] == 10
    assert artifact["mtime_fallback_window"]["wall_minutes"] > 0
    assert artifact["mtime_fallback_window"]["compute_bound_count"] >= 3
    assert artifact["wiring_proposal_reconfirmed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["duration_s"] == 0.5
    assert mod.validate_artifact(artifact) == []
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact

    stamped_4925 = json.loads(arms[4925].read_text(encoding="utf-8"))
    assert stamped_4925["duration_s"] == mod.runtime_stamping.MIN_DURATION_S
    assert stamped_4925["compute_bound"] is False
    assert arms[4925].stat().st_mtime_ns == original_mtime
    assert any(row["path"].endswith("experiment_4927_self_play_verifier_checkpoint.json") for row in artifact["stamping_backfilled_arms"])

    second = mod.run(root=tmp_path, started_s=20.0, now_s=20.25)
    assert second["stamping_backfilled_arms"] == "none missing"
    assert second["honest_verdict"] == artifact["honest_verdict"]
    assert mod.validate_artifact(second) == []


def test_scenario_report_4931_blocks_when_window_gate_is_underpopulated(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4931-BLOCKED-PRECONDITION: too few arms blocks success."""

    _seed_required_files(tmp_path)
    _make_v454_arms(tmp_path, (4924, 4925, 4926, 4927, 4928, 4929, 4930))

    artifact = mod.run(root=tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["honest_verdict"] == "blocked_insufficient_v454_mtime_window"
    assert artifact["mtime_fallback_window"]["n_arms"] == 7
    assert artifact["preconditions_checked"]["window_gate"]["min_arms"] == 10
    assert artifact["preconditions_checked"]["window_gate"]["passed"] is False
    assert artifact["research_conductor_modified"] is False
    assert isinstance(artifact["stamping_backfilled_arms"], list)
    assert mod.validate_artifact(artifact) == []


def test_req_report_4931_classification_preconditions_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-4931: classifiers, blocked inputs, and validation fail closed."""

    assert mod.compute_bound_for_backfill({"compute_bound": True}) is True
    assert mod.compute_bound_for_backfill({"compute_bound": False}) is False
    assert mod.compute_bound_for_backfill({"model_specs": {"backend": "cuda"}}) is True
    assert mod.compute_bound_for_backfill({"generator_backend": "igpu_hip"}) is True
    assert mod.compute_bound_for_backfill({"cuda_128_server": True}) is True
    assert mod.compute_bound_for_backfill({"runtime": [{"backend": "cpu"}]}) is False
    assert mod.compute_bound_for_backfill({"inference_substrate": "live_llm_inference"}) is True
    assert mod.compute_bound_for_backfill({"inference_substrate": "offline"}) is False
    assert mod._relative_result_path(Path("/tmp/plain.json")) == "/tmp/plain.json"

    _seed_required_files(tmp_path)
    arms = _make_v454_arms(
        tmp_path,
        (4924, 4925, 4926, 4927, 4928, 4929, 4930, 4932, 4933, 4934),
    )
    arms[4934].write_text("[1, 2]", encoding="utf-8")
    with_bad_json = mod.run(root=tmp_path, started_s=1.0, now_s=1.25)
    assert with_bad_json["honest_verdict"] == "blocked_unreadable_v454_arm_artifact"
    assert "json_error" in with_bad_json["preconditions_checked"]["arm_artifact_errors"][0]["error"]
    bad_json = tmp_path / "results" / "experiment_4932_bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json_object(bad_json)[1].startswith("json_error")
    assert mod.backfill_runtime_stamps([bad_json]) == "none missing"

    assert mod._blocked_precondition_verdict(
        {"missing_modules": [], "exp4920_shipping_artifact_present": False}
    ) == "blocked_missing_exp4920_shipping_artifact"
    assert mod._blocked_precondition_verdict(
        {
            "missing_modules": [],
            "exp4920_shipping_artifact_present": True,
            "wiring_proposal_present": False,
            "wiring_proposal_current": False,
        }
    ) == "blocked_missing_wiring_proposal"
    assert mod._blocked_precondition_verdict(
        {
            "missing_modules": [],
            "exp4920_shipping_artifact_present": True,
            "wiring_proposal_present": True,
            "wiring_proposal_current": True,
            "arm_artifacts": [],
        }
    ) == "blocked_missing_v454_arm_artifacts"

    missing_root = tmp_path / "missing"
    artifact = mod.run(root=missing_root, started_s=3.0, now_s=3.0)
    assert artifact["honest_verdict"] == "blocked_missing_453_reporting_module"
    assert artifact["mtime_fallback_window"]["n_arms"] == 0
    assert mod.validate_artifact(artifact) == []
    assert mod.main(root=missing_root) == 0

    success_root = tmp_path / "success"
    _seed_required_files(success_root)
    _make_v454_arms(
        success_root,
        (4924, 4925, 4926, 4927, 4928, 4929, 4930, 4932, 4933, 4934),
    )
    success = mod.run(root=success_root, started_s=5.0, now_s=5.5)
    assert mod.file_sha256(success_root / "missing.txt") == ""
    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in success.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**success, "honest_verdict": "done"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**success, "inference_substrate": "live_llm_inference"}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**success, "field_principles": {}}
    )
    assert "invalid_research_conductor_modified" in mod.validate_artifact(
        {**success, "research_conductor_modified": True}
    )
    assert "invalid_wiring_proposal_reconfirmed" in mod.validate_artifact(
        {**success, "wiring_proposal_reconfirmed": False}
    )
    assert "invalid_success_mtime_fallback_window" in mod.validate_artifact(
        {**success, "mtime_fallback_window": {"n_arms": 0}}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**success, "reproducibility_checksum": ""}
    )
    original_validate = mod.validate_artifact
    try:
        mod.validate_artifact = lambda _artifact: ["forced"]
        assert mod.main(root=success_root) == 2
    finally:
        mod.validate_artifact = original_validate
