"""Focused tests for window-120 shadow-supervisor evidence accrual.

Spec refs: REQ-ARC-WMTE-6776 and SCENARIO-ARC-WMTE-6776-1..5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6776_arc_shadow_supervisor_accrual as exp
from carnot.agentic import arc_supervisor_refinement as refinement


def _models(tmp_path: Path) -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        path = tmp_path / spec["filename"]
        path.write_bytes(b"GGUF" + bytes([index]))
        rows.append(
            {
                **deepcopy(spec),
                "model_path": str(path),
                "model_sha256": spec["expected_sha256"],
                "model_size_bytes": path.stat().st_size,
                "resolved": True,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": True,
                    "detail": "fixture",
                },
            }
        )
    return rows


def _manifest() -> dict:
    cells = [
        {
            "cell_id": "science:tu93:2026083001:shadow",
            "cell_kind": "science",
            "game": "tu93",
            "seed": 2026083001,
            "model_id": exp.MODEL_SPECS[0]["model_id"],
            "model_role": exp.MODEL_SPECS[0]["role"],
            "action_budget": 399,
            "supervisor_observation": "shadow",
            "invariance_pair_id": "tu93:2026083001",
        },
        {
            "cell_id": "canary:tu93:2026083001:shadow_off",
            "cell_kind": "shadow_off_canary",
            "game": "tu93",
            "seed": 2026083001,
            "model_id": exp.MODEL_SPECS[0]["model_id"],
            "model_role": exp.MODEL_SPECS[0]["role"],
            "action_budget": 399,
            "supervisor_observation": "off",
            "invariance_pair_id": "tu93:2026083001",
        },
        {
            "cell_id": "transport:ar25:2026083099:shadow",
            "cell_kind": "transport_canary",
            "game": "ar25",
            "seed": 2026083099,
            "model_id": exp.MODEL_SPECS[1]["model_id"],
            "model_role": exp.MODEL_SPECS[1]["role"],
            "action_budget": 140,
            "supervisor_observation": "shadow",
            "invariance_pair_id": None,
        },
    ]
    return exp.freeze_manifest(cells=cells)


def _preflight(models: list[dict], *, passed: bool = True) -> dict:
    selected = {
        "index": 1,
        "uuid": exp.EXPECTED_GPU_UUIDS[1],
        "name": "NVIDIA GeForce RTX 3090",
        "memory_free_mb": 24_000,
        "memory_used_mb": 100,
        "active_compute_processes": [],
    }
    checks = [
        {
            "check": "least_used_eligible_rtx3090",
            "expected": {"free_vram_mb_at_least": 22_610},
            "observed": selected if passed else None,
            "passed": passed,
        }
    ]
    return {
        "all_passed": passed,
        "checks": checks,
        "models": deepcopy(models),
        "device_selection_receipt": {"selected_device": selected if passed else None},
        "source_receipts": {},
    }


def _trajectory_receipt(*, shadow: bool, fired_arm: str | None = None) -> dict:
    outcomes = {arm: {"fired": 0, "helped": 0} for arm in exp.ARM_ORDER}
    redirects = []
    if fired_arm is not None:
        outcomes[fired_arm]["fired"] = 1
        redirects.append(
            {
                "arm": fired_arm,
                "action_index": 120,
                "level": 0,
                "diagnosis": "fixture stagnation",
                "levelup_followed_without_redirect": False,
                "actions_to_levelup_without_redirect": None,
            }
        )
    if not shadow:
        return {"enabled": False, "mode": "off", "actions_observed": 399}
    return {
        "enabled": False,
        "mode": "shadow",
        "window": 120,
        "actions_observed": 399,
        "arms_used": [fired_arm] if fired_arm else [],
        "would_have_redirects": redirects,
        "would_have_arm_outcomes": outcomes,
        "stagnations_unredirected": 0,
        "observe_errors": 0,
    }


def _cell(spec: dict, *, action_hash: str, fired_arm: str | None = None) -> dict:
    shadow = spec["supervisor_observation"] == "shadow"
    return {
        **deepcopy(spec),
        "status": "complete",
        "worker_process": {"pid": 7100, "exit_code": 0, "absent_after_exit": True},
        "llama_server_process": {"pid": 7200, "exit_code": 0, "absent_after_exit": True},
        "llama_server_log": "/tmp/fixture-server.log",
        "live_model_invoked": True,
        "first_token_observed": True,
        "context_observed": exp.CONTEXT_REQUESTED,
        "trajectory_supervisor": _trajectory_receipt(
            shadow=shadow, fired_arm=fired_arm if shadow else None
        ),
        "scored_action_hash": action_hash,
        "scored_actions": 399,
        "levels": 0,
        "gpu_receipt": {
            "device_uuid": exp.EXPECTED_GPU_UUIDS[1],
            "gpu_layers": {"requested": 999, "offloaded": 66, "total": 66},
            "peak_vram_mb": 18_000,
            "lease_owner": {"pid": 7100, "owner_bound": True},
            "lease_release": {"released": True, "phase": "terminal_complete"},
            "vram_recovery": {"passed": True},
            "port_release": {"closed": True},
            "unrelated_processes_signaled": [],
        },
        "death_receipt": {
            "installed": True,
            "path": "/tmp/fixture-death.json",
            "signal_receipt": None,
        },
        "shard_receipts": [
            {
                "event": "action_block",
                "action_index": 20,
                "atomic_replace": True,
                "sha256": "sha256:" + "1" * 64,
            },
            {
                "event": "cell_complete",
                "action_index": 399,
                "atomic_replace": True,
                "sha256": "sha256:" + "2" * 64,
            },
        ],
        "teardown_passed": True,
        "failure_class": None,
        "solve_claim": False,
    }


def _refinement_receipt() -> dict:
    per_arm = [
        {
            "arm": arm,
            "fired": 2 if arm != exp.ARM_ORDER[-1] else 0,
            "helped": 1 if arm != exp.ARM_ORDER[-1] else 0,
            "meets_floor": False,
            "wilson_lower": 0.0,
            "wilson_upper": 1.0,
            "floor_shortfall": 8 if arm != exp.ARM_ORDER[-1] else 10,
        }
        for arm in exp.ARM_ORDER
    ]
    return {
        "tool": "scripts/arc_supervisor_refine.py",
        "ran": True,
        "exit_code": 0,
        "ledger_sha256_before": "sha256:" + "3" * 64,
        "ledger_sha256_after": "sha256:" + "4" * 64,
        "entry_ids_before": ["sha256:old"],
        "entry_ids_after": ["sha256:old"],
        "deduplicated": True,
        "ingest_counts": {"shadow_observed": 2, "applied_new": 0},
        "recommendation": {
            "status": refinement.STATUS_INSUFFICIENT,
            "per_arm": per_arm,
            "recommendation_only": True,
            "recommendations": [],
        },
    }


def test_scenario_6776_1_window_shadow_and_action_hash_invariance(tmp_path: Path) -> None:
    """SCENARIO-6776-1 pins window 120 and byte-equivalent scored actions."""
    models = _models(tmp_path)
    manifest = _manifest()
    digest = exp.scored_action_hash([{"kind": "ACTION1"}, {"kind": "ACTION2"}])
    cells = [
        _cell(manifest["cells"][0], action_hash=digest, fired_arm=exp.ARM_ORDER[0]),
        _cell(manifest["cells"][1], action_hash=digest),
        _cell(manifest["cells"][2], action_hash="sha256:" + "5" * 64),
    ]
    invariance = exp.action_hash_invariance(cells, manifest)
    assert exp.SUPERVISOR_WINDOW == 120
    assert exp.SUPERVISOR_MODE == "shadow"
    assert invariance["passed"] is True
    assert invariance["pairs"][0]["shadow_action_hash"] == digest

    artifact = exp.build_artifact(
        manifest=manifest,
        models=models,
        preflight=_preflight(models),
        cells=cells,
        refinement_receipt=_refinement_receipt(),
        duration_s=61.0,
    )
    assert artifact["shadow_supervisor_transport_ready"] is True
    assert artifact["verdict_class"] == "partial"
    assert artifact["solve_claim"] is False
    assert exp.validate_artifact(artifact) == []


def test_scenario_6776_2_atomic_shards_death_install_and_firing_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-6776-2 checkpoints blocks/firings and installs death evidence."""
    shard = tmp_path / "cell" / "rows.json"
    first = exp.write_progress_shard(
        shard,
        cell_id="science:tu93:1:shadow",
        action_index=20,
        event="action_block",
        trajectory_supervisor=_trajectory_receipt(shadow=True),
    )
    second = exp.write_progress_shard(
        shard,
        cell_id="science:tu93:1:shadow",
        action_index=120,
        event="supervisor_firing",
        trajectory_supervisor=_trajectory_receipt(shadow=True, fired_arm=exp.ARM_ORDER[0]),
    )
    stored = json.loads(shard.read_text())
    assert first["atomic_replace"] is True
    assert second["atomic_replace"] is True
    assert stored["event"] == "supervisor_firing"
    assert not list(shard.parent.glob(f".{shard.name}.*"))

    calls = []
    monkeypatch.setattr(
        exp.long_run_receipt,
        "install",
        lambda path, progress=None: calls.append((path, progress)),
    )
    death = exp.install_death_receipt(tmp_path / "death.json", lambda: {"actions": 120})
    assert death["installed"] is True
    assert calls[0][0] == tmp_path / "death.json"

    manifest = _manifest()
    rows = exp.expand_cell_rows(
        [_cell(manifest["cells"][0], action_hash="sha256:" + "6" * 64, fired_arm=exp.ARM_ORDER[0])]
    )
    assert [row["supervisor_arm"] for row in rows] == list(exp.ARM_ORDER)
    assert rows[0]["arm_fired"] == 1
    assert all(row["solve_claim"] is False for row in rows)


def test_scenario_6776_3_refinement_dedupes_and_excludes_shadow(tmp_path: Path) -> None:
    """SCENARIO-6776-3 preserves applied-only ledger counts and deduplication."""
    ledger_path = tmp_path / "ledger.json"
    applied = {
        "game": "tu93",
        "seed": 1,
        "trajectory_supervisor": {
            "mode": "applied",
            "enabled": True,
            "window": 120,
            "redirects": [
                {
                    "arm": exp.ARM_ORDER[0],
                    "action_index": 120,
                    "level": 0,
                    "resolved_by_levelup": False,
                    "actions_to_levelup": None,
                }
            ],
        },
    }
    shadow = {
        "game": "tu93",
        "seed": 2,
        "trajectory_supervisor": _trajectory_receipt(shadow=True, fired_arm=exp.ARM_ORDER[0]),
    }
    source_a = tmp_path / "a" / "rows.json"
    source_b = tmp_path / "b" / "rows.json"
    exp.write_json_atomic(source_a, {"rows": [applied, shadow]})
    exp.write_json_atomic(source_b, {"rows": [applied, shadow]})

    receipt = exp.refine_shards(
        ledger_path=ledger_path,
        inputs=[source_a, source_b],
        now_iso="2026-08-30T12:00:00+00:00",
    )
    assert receipt["ingest_counts"]["applied_new"] == 1
    assert receipt["ingest_counts"]["applied_duplicate"] == 1
    assert receipt["ingest_counts"]["shadow_observed"] == 2
    assert receipt["deduplicated"] is True
    assert receipt["recommendation"]["per_arm"][0]["fired"] == 1


def test_scenario_6776_4_model_pins_and_teardown_are_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-6776-4 validates exact GGUF pins and complete owned teardown."""
    models = _models(tmp_path)
    assert exp.model_pin_errors(models) == []
    models[0]["model_sha256"] = "sha256:wrong"
    assert exp.model_pin_errors(models) == [f"model_sha256:{exp.MODEL_SPECS[0]['model_id']}"]

    manifest = _manifest()
    cell = _cell(manifest["cells"][0], action_hash="sha256:" + "7" * 64)
    assert exp.teardown_errors(cell) == []
    cell["llama_server_process"]["absent_after_exit"] = False
    assert "llama_server_process" in exp.teardown_errors(cell)


def test_named_cached_model_path_preserves_gguf_identity(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6776 retains the named snapshot link, not its hash-only blob target."""

    model_root = tmp_path / "models--unsloth--example"
    blob = model_root / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    named = model_root / "snapshots" / "revision" / "model.gguf"
    named.parent.mkdir(parents=True)
    named.symlink_to(blob)

    assert exp.named_cached_model_path(blob, "model.gguf") == str(named)
    assert exp.named_cached_model_path(named, "model.gguf") == str(named)
    assert exp.named_cached_model_path(blob, "missing.gguf") == str(blob)


def test_scenario_6776_5_blocked_preflight_writes_full_denominator(tmp_path: Path) -> None:
    """SCENARIO-6776-5 starts no worker and keeps a complete blocked receipt."""
    models = _models(tmp_path)
    manifest = _manifest()
    calls = []
    artifact = exp.run(
        result_path=tmp_path / "result.json",
        manifest=manifest,
        preflight_fn=lambda: _preflight(models, passed=False),
        worker_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        clock=iter((1_000_000_000, 3_000_000_000)).__next__,
    )
    assert calls == []
    assert artifact["honest_verdict"].startswith("complete_blocked_shadow_supervisor_accrual")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_detail"] == exp.INFERENCE_SUBSTRATE_DETAIL
    assert artifact["shadow_supervisor_transport_ready"] is False
    assert len(artifact["rows"]) == len(manifest["cells"]) * len(exp.ARM_ORDER)
    assert artifact["gate_check_summary"]["failed_check"] == ("least_used_eligible_rtx3090")
    assert all(row["solve_claim"] is False for row in artifact["rows"])
    assert exp.validate_artifact(artifact) == []


def test_artifact_rejects_action_change_and_solve_claim(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6776 disqualifies behavior drift and rejects solve inflation."""
    models = _models(tmp_path)
    manifest = _manifest()
    cells = [
        _cell(manifest["cells"][0], action_hash="sha256:" + "8" * 64),
        _cell(manifest["cells"][1], action_hash="sha256:" + "9" * 64),
        _cell(manifest["cells"][2], action_hash="sha256:" + "a" * 64),
    ]
    artifact = exp.build_artifact(
        manifest=manifest,
        models=models,
        preflight=_preflight(models),
        cells=cells,
        refinement_receipt=_refinement_receipt(),
        duration_s=61.0,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["shadow_supervisor_transport_ready"] is False
    artifact["solve_claim"] = True
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "solve_claim" in exp.validate_artifact(artifact)


def test_defensive_contract_branches_and_default_manifest(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-6776 covers every fail-closed reducer branch in new code."""
    selected_failure = exp.failed_preflight_check(
        {
            "checks": [
                {"check": "least_used_eligible_rtx3090", "passed": False, "observed": None},
                {
                    "check": "exclusive_gpu_without_unrelated_compute",
                    "passed": False,
                    "observed": ["inventory"],
                },
            ]
        }
    )
    assert selected_failure["observed"] == ["inventory"]

    default = exp.freeze_manifest()
    assert len(default["cells"]) == len(exp.PANEL_GAMES) * len(exp.RANDOM_SEEDS) + 2
    assert default["cells"][-1]["model_id"] == exp.MODEL_SPECS[1]["model_id"]

    broken_shard = tmp_path / "broken" / "rows.json"
    broken_shard.parent.mkdir()
    broken_shard.write_text("not json")
    receipt = exp.write_progress_shard(
        broken_shard,
        cell_id="broken",
        action_index=1,
        event="action_block",
        trajectory_supervisor={},
    )
    assert receipt["atomic_replace"] is True

    models = _models(tmp_path)
    malformed = list(reversed(deepcopy(models)))
    malformed[0]["role"] = "wrong"
    malformed[0]["model_path"] = "/tmp/wrong.gguf"
    malformed[0]["resolved"] = False
    malformed[0]["tokenizer"] = None
    malformed.pop()
    pin_errors = exp.model_pin_errors(malformed)
    assert "model_order_or_denominator" in pin_errors
    assert any(value.startswith("model_missing:") for value in pin_errors)
    assert any(value.startswith("model_role:") for value in pin_errors)
    assert any(value.startswith("model_filename:") for value in pin_errors)
    assert any(value.startswith("model_resolution:") for value in pin_errors)
    assert any(value.startswith("embedded_tokenizer:") for value in pin_errors)

    manifest = _manifest()
    bad_teardown = _cell(manifest["cells"][0], action_hash="sha256:" + "b" * 64)
    bad_teardown["worker_process"] = {}
    bad_teardown["gpu_receipt"]["lease_release"] = {}
    bad_teardown["gpu_receipt"]["vram_recovery"] = {}
    bad_teardown["gpu_receipt"]["port_release"] = {}
    bad_teardown["gpu_receipt"]["unrelated_processes_signaled"] = [999]
    assert set(exp.teardown_errors(bad_teardown)) == {
        "worker_process",
        "lease_release",
        "vram_recovery",
        "port_release",
        "unrelated_process_signal",
    }

    digest = "sha256:" + "c" * 64
    cells = [
        _cell(manifest["cells"][0], action_hash=digest),
        _cell(manifest["cells"][1], action_hash=digest),
        _cell(manifest["cells"][2], action_hash="sha256:" + "d" * 64),
    ]
    damaged = deepcopy(cells)
    damaged.append({"cell_id": "unexpected", "status": "failed"})
    damaged[0].update(
        {
            "live_model_invoked": False,
            "first_token_observed": False,
            "context_observed": 1,
            "solve_claim": True,
            "death_receipt": {},
            "shard_receipts": [],
            "worker_process": {},
        }
    )
    damaged[1]["scored_action_hash"] = "sha256:" + "e" * 64
    damaged[0]["gpu_receipt"]["gpu_layers"]["offloaded"] = 0
    damaged[0]["trajectory_supervisor"].update(
        {"mode": "applied", "window": 400, "redirects": [], "arm_outcomes": {}}
    )
    bad_refinement = deepcopy(_refinement_receipt())
    bad_refinement.update({"ran": False, "exit_code": 1, "deduplicated": False})
    bad_refinement["ingest_counts"] = {"shadow_observed": 0, "applied_new": 1}
    transport_errors = exp._transport_errors(
        manifest=manifest,
        models=models,
        cells=damaged,
        preflight=_preflight(models, passed=False),
        refinement_receipt=bad_refinement,
    )
    assert {
        "preconditions",
        "cell_denominator_or_order",
        f"first_token:{damaged[0]['cell_id']}",
        f"context:{damaged[0]['cell_id']}",
        f"solve_claim:{damaged[0]['cell_id']}",
        f"cuda_offload:{damaged[0]['cell_id']}",
        f"death_receipt:{damaged[0]['cell_id']}",
        f"shards:{damaged[0]['cell_id']}",
        f"shadow_receipt:{damaged[0]['cell_id']}",
        f"applied_keys_in_shadow:{damaged[0]['cell_id']}",
        "ledger_deduplication",
        "shadow_relabelled_applied",
    } <= set(transport_errors)

    partial_refinement = deepcopy(_refinement_receipt())
    partial_refinement["ingest_counts"]["shadow_observed"] = 0
    partial = exp.build_artifact(
        manifest=manifest,
        models=models,
        preflight=_preflight(models),
        cells=cells,
        refinement_receipt=partial_refinement,
        duration_s=61,
    )
    assert partial["status"] == "complete_partial_shadow_supervisor_accrual"

    floor_refinement = deepcopy(_refinement_receipt())
    for row in floor_refinement["recommendation"]["per_arm"]:
        row["fired"] = 10
        row["meets_floor"] = True
    complete = exp.build_artifact(
        manifest=manifest,
        models=models,
        preflight=_preflight(models),
        cells=cells,
        refinement_receipt=floor_refinement,
        duration_s=61,
    )
    assert complete["status"] == "complete_shadow_supervisor_transport_and_evidence_ready"
    assert complete["inference_substrate"] == "live_llm_inference"

    corrupt = deepcopy(complete)
    corrupt.pop("title")
    corrupt["field_principles"] = {}
    corrupt["schema"] = "wrong"
    corrupt["inference_substrate"] = "cpu"
    corrupt["supervisor_window"] = 400
    corrupt["supervisor_mode"] = "applied"
    corrupt["solve_claim"] = True
    corrupt["verifier_is_oracle"] = True
    corrupt["verdict_class"] = "unknown"
    corrupt["honest_verdict"] = "nonterminal"
    corrupt["frozen_manifest"]["manifest_sha256"] = "wrong"
    corrupt["model_specs"] = []
    corrupt["rows"] = list(reversed(corrupt["rows"]))
    corrupt["rows"][0]["solve_claim"] = True
    corrupt["rows"][0]["row_sha256"] = "wrong"
    corrupt["firings_before_by_arm"] = {}
    corrupt["firings_after_by_arm"] = {}
    corrupt["evidence_floor_met_by_arm"] = {}
    corrupt["action_hash_invariance"]["passed"] = True
    corrupt["action_hash_invariance"]["pairs"][0]["shadow_action_hash"] = "sha256:wrong"
    corrupt["shadow_supervisor_transport_ready"] = True
    validation = exp.validate_artifact(corrupt)
    for expected in (
        "required_fields",
        "field_principles",
        "schema",
        "inference_substrate",
        "supervisor_window",
        "supervisor_mode",
        "solve_claim",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "manifest_sha256",
        "model_order_or_denominator",
        "row_denominator_or_order",
        "firings_before_by_arm",
        "firings_after_by_arm",
        "evidence_floor_met_by_arm",
        "action_hash_invariance",
        "transport_ready_without_invariance",
        "reproducibility_checksum",
    ):
        assert expected in validation

    blocked_corrupt = deepcopy(partial)
    blocked_corrupt["verdict_class"] = "blocked"
    blocked_corrupt["shadow_supervisor_transport_ready"] = True
    blocked_validation = exp.validate_artifact(blocked_corrupt)
    assert "blocked_transport_ready" in blocked_validation
    assert "blocked_rows" in blocked_validation

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["fixture-invalid"])
    with pytest.raises(ValueError, match="fixture-invalid"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            manifest=manifest,
            preflight_fn=lambda: _preflight(models, passed=False),
            clock=iter((1_000_000_000, 2_000_000_000)).__next__,
        )
