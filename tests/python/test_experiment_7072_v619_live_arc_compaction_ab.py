"""Tests for REQ-ARC-7072 claim-grade live compaction evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7072_v619_live_arc_compaction_ab as experiment


ROOT = Path(__file__).resolve().parents[2]


def _models(tmp_path: Path) -> list[dict[str, object]]:
    qwen = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    gemma = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    qwen.write_bytes(b"qwen")
    gemma.write_bytes(b"gemma")
    return [
        {
            "key": "qwen",
            "name": "Qwen3.8-27B",
            "hf_id": experiment.QWEN_HF_ID,
            "model_path": str(qwen),
            "gpu": 0,
        },
        {
            "key": "gemma",
            "name": "Gemma4-26B-A4B-it",
            "hf_id": experiment.GEMMA_HF_ID,
            "model_path": str(gemma),
            "gpu": 1,
        },
    ]


def _units(n: int = 30) -> list[dict[str, object]]:
    return [
        {
            "unit_id": f"rotation-{index:02d}",
            "game_id": f"hidden-{index % 6}",
            "source_group": f"rotation-group-{index % 3}",
            "seed": 707_200 + index,
            "start_state_hash": f"start-{index:02d}",
        }
        for index in range(n)
    ]


def _manifest(tmp_path: Path) -> dict[str, object]:
    return experiment.freeze_paired_cell_manifest(
        units=_units(),
        model_specs=_models(tmp_path),
        qwen_pair_count=30,
        gemma_pair_count=8,
        action_budget=120,
        token_budget=4096,
        context_budget=98_304,
        time_budget_s=2400,
        random_seed=experiment.RANDOM_SEED,
    )


def _rows(manifest: dict[str, object], *, fired: int = 30) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    treatment_index = {"qwen": 0, "gemma": 0}
    for pair in manifest["pairs"]:
        assert isinstance(pair, dict)
        for arm in pair["arm_order"]:
            is_treatment = arm == "treatment"
            model_key = str(pair["model_key"])
            compactions = int(is_treatment and treatment_index[model_key] < fired)
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "unit_id": pair["unit_id"],
                    "source_group": pair["source_group"],
                    "model_key": pair["model_key"],
                    "model_hf_id": pair["model_hf_id"],
                    "model_path": pair["model_path"],
                    "model_file_hash": pair["model_file_hash"],
                    "arm": arm,
                    "seed": pair["seed"],
                    "start_state_hash": pair["start_state_hash"],
                    "action_budget": pair["action_budget"],
                    "token_budget": pair["token_budget"],
                    "context_budget": pair["context_budget"],
                    "time_budget_s": pair["time_budget_s"],
                    "status": "complete",
                    "tool_loop_reachable": True,
                    "eligible_for_compaction": True,
                    "compactions": compactions,
                    "refetches": compactions,
                    "parse_failures": int(is_treatment and pair["pair_id"] == "qwen:rotation-00"),
                    "tool_calls": 4,
                    "exact_progress": 2,
                    "levels_reached": 1,
                    "solves": 0,
                    "actions": 60,
                    "tokens": 1000,
                    "peak_context": 20_000 if not is_treatment else 17_000,
                    "p95_context": 19_000 if not is_treatment else 16_000,
                    "wall_time_s": 100.0 if not is_treatment else 105.0,
                    "crashes": 0,
                    "duplicate_submissions": 0,
                    "solve_provenance": "live_agent_self_discovery",
                    "config_bytes": experiment.arm_config_bytes(pair, str(arm)).decode("ascii"),
                    "imported_production_symbols": list(experiment.PRODUCTION_SYMBOLS),
                }
            )
        treatment_index[str(pair["model_key"])] += 1
    return rows


def test_req_arc_7072_spec_exists_before_implementation() -> None:
    """REQ-ARC-7072 and every required scenario exist before code."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7072" in text
    for suffix in (
        "FLAG-ISOLATION",
        "ACTIVATION",
        "PARSER-FAILURE",
        "PAIRING",
        "CHECKPOINT",
        "CLEANUP",
        "PROVENANCE",
        "AGGREGATION",
    ):
        assert f"SCENARIO-ARC-7072-{suffix}" in text


def test_manifest_freezes_population_order_seed_and_model_identity(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-PAIRING freezes both model populations before outcomes."""

    manifest = _manifest(tmp_path)
    pairs = manifest["pairs"]
    qwen = [row for row in pairs if row["model_key"] == "qwen"]
    gemma = [row for row in pairs if row["model_key"] == "gemma"]

    assert len(qwen) == 30
    assert len(gemma) == 8
    assert len({row["source_group"] for row in pairs}) >= 2
    assert abs(sum(row["arm_order"][0] == "treatment" for row in pairs) - len(pairs) / 2) <= 1
    assert all(
        row["seed"] == next(u["seed"] for u in _units() if u["unit_id"] == row["unit_id"])
        for row in pairs
    )
    assert manifest["manifest_hash"] == experiment.hash_without_field(manifest, "manifest_hash")


def test_flag_isolation_changes_only_compaction_master_switch(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-FLAG-ISOLATION keeps all other bytes paired."""

    manifest = _manifest(tmp_path)
    rows = experiment.flag_isolation_rows(manifest)
    control = json.loads(experiment.arm_config_bytes(manifest["pairs"][0], "control"))
    treatment = json.loads(experiment.arm_config_bytes(manifest["pairs"][0], "treatment"))

    assert rows and all(row["passed"] is True for row in rows)
    assert control["environment"].get(experiment.COMPACTION_FLAG) is None
    assert treatment["environment"][experiment.COMPACTION_FLAG] == "1"
    assert experiment.GROWTH_FLAG not in treatment["environment"]
    assert experiment.STATE_BUDGET_FLAG not in treatment["environment"]


@pytest.mark.parametrize(
    ("fired", "control_fire", "reachable", "expected"),
    [(30, False, True, 1), (23, False, True, 0), (30, True, True, 0), (30, False, False, 0)],
)
def test_treatment_activation_blocks_non_fire_and_unreachable_loops(
    tmp_path: Path, fired: int, control_fire: bool, reachable: bool, expected: int
) -> None:
    """SCENARIO-ARC-7072-ACTIVATION disqualifies inert or unreachable treatment."""

    manifest = _manifest(tmp_path)
    rows = _rows(manifest, fired=fired)
    qwen = [row for row in rows if row["model_key"] == "qwen"]
    if control_fire:
        next(row for row in qwen if row["arm"] == "control")["compactions"] = 1
    if not reachable:
        qwen[0]["tool_loop_reachable"] = False
    scores = experiment.activation_scores(qwen)

    assert scores["compaction_treatment_activated_score"] == expected
    assert scores["tool_loop_reachable_score"] == int(reachable)
    if expected == 0:
        assert experiment.classify_value(rows)["verdict_class"] == "disqualified"


def test_parser_failure_is_charged_to_rows_and_safety_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-PARSER-FAILURE retains failed transport in aggregation."""

    rows = _rows(_manifest(tmp_path))
    result = experiment.aggregate_rows(rows)

    assert sum(row["parse_failures"] for row in rows) == 1
    assert result["parse_failure_rows"]
    assert result["by_model"]["qwen"]["treatment"]["parse_failures"] == 1
    assert experiment.classify_value(rows)["verdict_class"] == "null"


def test_checkpoint_resume_rejects_other_manifest_and_does_not_duplicate(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-CHECKPOINT resumes only matching completed cells."""

    path = tmp_path / "checkpoint.json"
    store = experiment.CellCheckpointStore(path, manifest_hash="manifest-a")
    row = {"cell_id": "pair-a:control", "status": "complete", "actions": 3}
    store.save(row)

    assert store.pending(["pair-a:control", "pair-a:treatment"]) == ["pair-a:treatment"]
    store.save(row)
    assert len(store.rows) == 1
    with pytest.raises(ValueError, match="manifest"):
        experiment.CellCheckpointStore(path, manifest_hash="manifest-b")


def test_model_identity_fails_closed_on_path_hash_or_hub_change(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-PAIRING requires the exact selected GGUF identity."""

    spec = _models(tmp_path)[0]
    identity = experiment.model_identity_row(spec)
    assert experiment.validate_model_identity(identity, spec) == []
    for field in ("model_hf_id", "model_path", "model_file_hash"):
        attacked = deepcopy(identity)
        attacked[field] = "changed"
        assert experiment.validate_model_identity(attacked, spec)


def test_registry_precheck_excludes_public_adapter_and_source_reading() -> None:
    """SCENARIO-ARC-7072-PROVENANCE accepts hidden self-discovery only."""

    rows = experiment.registry_precheck_rows(_units(3), reproduced_game_ids={"public-1"})
    assert all(row["passed"] is True for row in rows)
    attacked = deepcopy(_units(1)[0])
    attacked.update({"game_id": "public-1", "source_read": True, "per_game_adapter": True})
    rejected = experiment.registry_precheck_rows([attacked], reproduced_game_ids={"public-1"})

    assert rejected[0]["passed"] is False
    assert {"already_reproduced", "source_read", "per_game_adapter"} <= set(rejected[0]["reasons"])


def test_cleanup_signals_only_owned_resources() -> None:
    """SCENARIO-ARC-7072-CLEANUP leaves unattributed resources untouched."""

    stopped: list[int] = []
    released: list[str] = []
    resources = [
        {"kind": "process", "identity": "pid-7", "pid": 7, "owned": True},
        {"kind": "process", "identity": "pid-8", "pid": 8, "owned": False},
        {"kind": "gpu_lease", "identity": "gpu-0", "owned": True},
        {"kind": "port_lease", "identity": "port-9", "owned": True},
    ]
    rows = experiment.cleanup_owned_resources(
        resources,
        stop_process=lambda pid: stopped.append(pid) or True,
        release_resource=lambda identity: released.append(identity) or True,
    )

    assert stopped == [7]
    assert released == ["gpu-0", "port-9"]
    assert (
        next(row for row in rows if row["identity"] == "pid-8")["action"] == "blocked_unattributed"
    )
    assert all(row["terminal"] is True for row in rows)


def test_row_aggregation_recomputes_paired_release_gates(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-AGGREGATION derives every headline from paired rows."""

    rows = _rows(_manifest(tmp_path))
    for row in rows:
        row["parse_failures"] = 0
    result = experiment.aggregate_rows(rows)
    decision = experiment.classify_value(rows)

    assert result["qwen_pair_count"] == 30
    assert result["gemma_pair_count"] == 8
    assert result["by_model"]["qwen"]["paired"]["exact_quality_mean_delta"] == 0.0
    assert result["by_model"]["qwen"]["paired"]["p95_context_ratio"] < 0.9
    assert result["by_model"]["qwen"]["paired"]["wall_time_ratio"] == 1.05
    assert decision["verdict_class"] == "positive"
    assert decision["compaction_value_ready_score"] == 1


def test_blocked_artifact_is_schema_complete_and_names_exact_gate(tmp_path: Path) -> None:
    """REQ-ARC-7072 writes exact terminal evidence when a precondition fails."""

    checks = [experiment.gate_row("owned_idle_rtx3090", True, False)]
    artifact = experiment.blocked_artifact(
        execution_date="20260906",
        checks=checks,
        source_hashes={"upstream": "sha256:abc"},
        duration_s=0.2,
    )

    assert experiment.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "owned_idle_rtx3090"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert set(experiment.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_fail_closed_helpers_cover_missing_models_and_invalid_manifests(tmp_path: Path) -> None:
    """REQ-ARC-7072 rejects absent pins, undersized populations, and malformed arms."""

    models = _models(tmp_path)
    resolved = experiment.resolve_model_specs(
        qwen_resolver=lambda _hf_id: str(models[0]["model_path"]),
        pair_resolver=lambda **_kwargs: [models[1]],
        gpu_indices=(3, 4),
    )
    assert [row["hf_id"] for row in resolved] == [experiment.QWEN_HF_ID, experiment.GEMMA_HF_ID]
    assert [row["gpu"] for row in resolved] == [3, 4]
    assert (
        experiment.resolve_model_specs(
            qwen_resolver=lambda _hf_id: None,
            pair_resolver=lambda **_kwargs: [models[1]],
        )
        == []
    )
    assert experiment.sha256_file(tmp_path / "missing.gguf") is None

    kwargs = {
        "units": _units(),
        "model_specs": models,
        "qwen_pair_count": 30,
        "gemma_pair_count": 8,
        "action_budget": 120,
        "token_budget": 4096,
        "context_budget": 98_304,
        "time_budget_s": 2400,
        "random_seed": experiment.RANDOM_SEED,
    }
    with pytest.raises(ValueError, match="minimum"):
        experiment.freeze_paired_cell_manifest(**{**kwargs, "qwen_pair_count": 29})
    with pytest.raises(ValueError, match="eligible units"):
        experiment.freeze_paired_cell_manifest(**{**kwargs, "units": _units(29)})
    with pytest.raises(ValueError, match="both pinned"):
        experiment.freeze_paired_cell_manifest(**{**kwargs, "model_specs": models[:1]})
    with pytest.raises(ValueError, match="unknown arm"):
        experiment.arm_config(_manifest(tmp_path)["pairs"][0], "placebo")


def test_checkpoint_reload_and_small_sample_interval(tmp_path: Path) -> None:
    """SCENARIO-ARC-7072-CHECKPOINT restores rows and keeps small-N bounds explicit."""

    path = tmp_path / "checkpoint.json"
    row = {"cell_id": "pair-a:control", "status": "complete"}
    experiment.CellCheckpointStore(path, manifest_hash="same").save(row)
    resumed = experiment.CellCheckpointStore(path, manifest_hash="same")

    assert resumed.rows == {"pair-a:control": row}
    assert experiment._one_sided_lower_95([2.0]) == 2.0
    assert experiment._one_sided_lower_95([]) == float("-inf")


def test_catalog_selection_keeps_only_new_hidden_or_rotation_units() -> None:
    """SCENARIO-ARC-7072-PROVENANCE never substitutes reproduced public units."""

    catalog = [
        {"game_id": "public-1", "tags": [], "private_tags": [], "source_group": "public"},
        {"game_id": "hidden-1", "tags": ["hidden"], "private_tags": [], "source_group": "h"},
        {"game_id": "rotation-1", "tags": [], "private_tags": ["rotation"], "source_group": "r"},
        {"game_id": "novel-1", "tags": [], "private_tags": [], "source_group": "n"},
    ]
    units = experiment.eligible_catalog_units(catalog, reproduced_game_ids={"public"})

    assert len(units) == 15
    assert {row["game_id"] for row in units} == {"hidden-1", "rotation-1", "novel-1"}
    assert {row["source_group"].split(":", 1)[0] for row in units} == {
        "hidden",
        "rotation",
        "unregistered_hidden_candidate",
    }


@pytest.mark.memory_watchdog_skip
def test_production_symbols_are_reachable_through_scored_factory() -> None:
    """REQ-ARC-7072 checks the real E3 policy and factory imports."""

    rows = experiment.production_reachability_rows()

    assert len(rows) == 3
    assert all(row["passed"] is True for row in rows)


def test_artifact_validator_rejects_semantic_attacks_and_writer_fails_closed(
    tmp_path: Path,
) -> None:
    """REQ-ARC-7072 rejects schema, provenance, verdict, and checksum attacks."""

    artifact = experiment.blocked_artifact(
        execution_date="20260906",
        checks=[experiment.gate_row("eligible_units", ">=30", 0)],
        source_hashes={},
        duration_s=0.1,
    )
    output = tmp_path / "artifact.json"
    experiment.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"
    assert experiment.validate_artifact([]) == ["artifact_not_object"]

    attacked = deepcopy(artifact)
    attacked.pop("rows")
    attacked["field_principles"] = {}
    attacked["inference_substrate"] = "cached_fixture"
    attacked["verifier_is_oracle"] = True
    attacked["verdict_class"] = "invented"
    attacked["honest_verdict"] = "success"
    attacked["gate_check_summary"] = {}
    errors = experiment.validate_artifact(attacked)
    assert "missing_field:rows" in errors
    assert "field_principles_incomplete" in errors
    assert "inference_substrate_invalid" in errors
    assert "verifier_is_oracle_invalid" in errors
    assert "verdict_prefix_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors

    bad_block = deepcopy(artifact)
    bad_block["gate_check_summary"] = {}
    bad_block["reproducibility_checksum"] = experiment.hash_without_field(
        bad_block, "reproducibility_checksum"
    )
    assert "blocked_gate_summary_invalid" in experiment.validate_artifact(bad_block)
    with pytest.raises(ValueError, match="invalid Exp7072"):
        experiment.write_artifact(tmp_path / "bad.json", attacked)


def test_gate_receipts_are_json_serializable() -> None:
    """REQ-ARC-7072 keeps exact expected and observed values in JSON-safe form."""

    row = experiment.gate_row(
        "model_pins",
        sorted((experiment.QWEN_HF_ID, experiment.GEMMA_HF_ID)),
        sorted((experiment.QWEN_HF_ID, experiment.GEMMA_HF_ID)),
    )

    assert json.loads(experiment.canonical_json_bytes(row))["passed"] is True
