"""Tests for Exp 1524 FR-11 live policy promotion.

Spec: REQ-LEARN-1524, SCENARIO-LEARN-1524, SCENARIO-LEARN-1525.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_live_policy_promotion as mod


def test_req_learn_1524_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1524-1/8: bootstrap artifact exposes the required contract."""

    output = tmp_path / mod.OUTPUT_FILE
    manifest = tmp_path / mod.MANIFEST_FILE

    artifact = mod.write_in_progress_artifact(
        output,
        manifest_path=manifest,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["live_policy_promotion_ready"] is False
    assert artifact["policy_promotion_manifest_path"] == mod.MANIFEST_FILE
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_req_learn_1524_filters_only_rollback_passing_packaged_updates() -> None:
    """REQ-LEARN-1524-3/4: unsafe or unproven updates are rejected."""

    policy_rows = [
        _policy_row("eligible"),
        _policy_row("rolled-back"),
        _policy_row("unreachable"),
        _policy_row("stale"),
        _policy_row("no-validator"),
        _policy_row("false-accept"),
        _policy_row("soundness"),
        _policy_row("not-packaged"),
        _policy_row("eligible-two"),
        dict(_policy_row("not-accepted"), accepted=False),
        dict(_policy_row("missing-skill"), skill_id=""),
    ]
    rollback_rows = [
        _rollback_row("eligible"),
        _rollback_row("rolled-back", decision="rollback", rollback_reasons=["x"]),
        _rollback_row("unreachable", reachable=False),
        _rollback_row("stale", stale=True),
        _rollback_row("no-validator", deterministic=False),
        _rollback_row("false-accept", false_accept_delta=1),
        _rollback_row("soundness", soundness_mistakes=1),
        _rollback_row("not-packaged"),
        _rollback_row("eligible-two"),
        _rollback_row("not-accepted"),
        dict(_rollback_row("missing-skill"), skill_id=""),
    ]
    pack_manifest = _pack_manifest(["eligible", "eligible-two", "not-accepted", "missing-skill"])

    selection = mod.select_promotable_updates(
        policy_rows=policy_rows,
        rollback_rows=rollback_rows,
        pack_manifest=pack_manifest,
        limit=1,
    )
    promoted = selection["promoted"]
    rejected_by_id = {row["source_event_id"]: row for row in selection["rejected"]}

    assert [row["source_event_id"] for row in promoted] == ["daily_eval:eligible"]
    assert (
        "rollback_decision_not_keep"
        in rejected_by_id["daily_eval:rolled-back"]["rejection_reasons"]
    )
    assert "rollback_reason:x" in rejected_by_id["daily_eval:rolled-back"]["rejection_reasons"]
    assert (
        "source_evidence_unreachable"
        in rejected_by_id["daily_eval:unreachable"]["rejection_reasons"]
    )
    assert "source_evidence_stale" in rejected_by_id["daily_eval:stale"]["rejection_reasons"]
    assert (
        "missing_deterministic_validator_support"
        in rejected_by_id["daily_eval:no-validator"]["rejection_reasons"]
    )
    assert (
        "false_accept_delta_positive"
        in rejected_by_id["daily_eval:false-accept"]["rejection_reasons"]
    )
    assert "soundness_mistake" in rejected_by_id["daily_eval:soundness"]["rejection_reasons"]
    assert (
        "missing_portable_provenance"
        in rejected_by_id["daily_eval:not-packaged"]["rejection_reasons"]
    )
    assert rejected_by_id["daily_eval:eligible-two"]["rejection_reasons"] == [
        "bounded_live_set_limit"
    ]
    assert "exp1512_not_accepted" in rejected_by_id["daily_eval:not-accepted"]["rejection_reasons"]
    assert "missing_skill_id" in rejected_by_id["daily_eval:missing-skill"]["rejection_reasons"]


def test_req_learn_1524_selects_contract_cases_and_parses_edge_outputs(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1524-6: contract cases and model outputs are normalized."""

    manifest = tmp_path / "runtime.jsonl"
    _write_jsonl(
        manifest,
        [
            {"row_type": "summary"},
            _contract_case("unlabeled", expected_label=None, final_accept=False),
            _contract_case("accept-ok", expected_label=True, final_accept=True),
            _contract_case("reject", expected_label=False, final_accept=False),
            _contract_case("marginal", expected_label=True, final_accept=False),
        ],
    )
    cases = mod.select_contract_cases(manifest, limit=2)
    case = cases[0]
    model = {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen"}
    update = {"source_event_id": "daily_eval:eligible"}

    by_bool = mod.validate_policy_output(
        case,
        model_spec=model,
        update=update,
        mode="promoted",
        raw_output='broken {not-json} {"contract_case_id":"reject","final_deterministic_accept":false}',
    )
    mismatch = mod.validate_policy_output(
        case,
        model_spec=model,
        update=update,
        mode="promoted",
        raw_output='{"contract_case_id":"other","final_deterministic_decision":"reject"}',
    )
    missing_decision = mod.validate_policy_output(
        case,
        model_spec=model,
        update=update,
        mode="promoted",
        raw_output='{"contract_case_id":"reject"}',
    )

    assert [row["contract_case_id"] for row in cases] == ["reject", "marginal"]
    assert by_bool["task_success"] is True
    assert mismatch["parse_status"] == "contract_case_id_mismatch"
    assert missing_decision["parse_status"] == "missing_final_decision"
    with pytest.raises(ValueError, match="unknown evaluation mode"):
        mod.build_policy_prompt(case, update=update, mode="bogus")


def test_req_learn_1524_resolves_cache_fallback_after_pair_exception() -> None:
    """REQ-LEARN-1524-5: single mandated GGUF resolver can unblock runtime."""

    def broken_pair(**_: Any) -> None:
        raise RuntimeError("pair probe failed")

    models = mod._resolve_runtime_models(
        broken_pair,
        lambda hf_id: (
            f"/models/{hf_id.rsplit('/', 1)[-1]}.gguf"
            if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"
            else None
        ),
        max_models=1,
    )

    assert models == [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe_primary_live_policy_evaluation",
            "gpu": 0,
            "model_path": "/models/Qwen3.6-35B-A3B-GGUF.gguf",
        }
    ]


def test_scenario_learn_1524_runner_promotes_live_with_injected_sota(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1524: promoted policy is evaluated through Exp 1520."""

    paths = _write_sources(tmp_path)

    def fake_generate(
        prompt: str,
        model: dict[str, Any],
        mode: str,
        update: dict[str, Any],
        case: dict[str, Any],
    ) -> str:
        del prompt, model, update
        if mode == "baseline":
            return "The contract should probably be rejected."
        return json.dumps(
            {
                "contract_case_id": case["contract_case_id"],
                "final_deterministic_decision": "reject",
            },
            sort_keys=True,
        )

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["promotion_manifest"],
        policy_cache_artifact_path=paths["policy_artifact"],
        policy_cache_manifest_path=paths["policy_manifest"],
        rollback_artifact_path=paths["rollback_artifact"],
        rollback_manifest_path=paths["rollback_manifest"],
        portable_pack_artifact_path=paths["pack_artifact"],
        portable_pack_manifest_path=paths["pack_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=fake_generate,
        gpu_probe_fn=lambda: {"cuda_available": True, "gpu_count": 1},
        update_limit=1,
        case_limit=1,
    )
    rows = _read_jsonl(paths["promotion_manifest"])

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["live_policy_promotion_ready"] is True
    assert artifact["rollback_passing_updates_loaded"] == 1
    assert artifact["promoted_policy_updates"] == ["daily_eval:eligible"]
    assert artifact["baseline_task_success_rate"] == pytest.approx(0.0)
    assert artifact["promoted_task_success_rate"] == pytest.approx(1.0)
    assert artifact["utility_delta"] == pytest.approx(1.0)
    assert artifact["false_accept_delta"] == 0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 2
    assert rows[0]["row_type"] == "policy_promotion_evaluation"
    assert rows[0]["runtime_contract_validation"]["promoted"]["false_accept"] is False
    assert rows[1]["row_type"] == "summary"
    mod.validate_artifact(artifact, manifest_path=paths["promotion_manifest"])


def test_scenario_learn_1525_blocks_without_mandated_sota_runtime(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1525: missing SOTA GGUFs never fall back to tiny models."""

    paths = _write_sources(tmp_path)
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["promotion_manifest"],
        policy_cache_artifact_path=paths["policy_artifact"],
        policy_cache_manifest_path=paths["policy_manifest"],
        rollback_artifact_path=paths["rollback_artifact"],
        rollback_manifest_path=paths["rollback_manifest"],
        portable_pack_artifact_path=paths["pack_artifact"],
        portable_pack_manifest_path=paths["pack_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: None,
        resolver_fn=lambda _hf_id: None,
        generator_fn=lambda *_args, **_kwargs: "must not be called",
        gpu_probe_fn=lambda: {"cuda_available": False, "gpu_count": 0},
        update_limit=1,
        case_limit=1,
    )
    rows = _read_jsonl(paths["promotion_manifest"])

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["live_policy_promotion_ready"] is False
    assert artifact["models_used"] == []
    assert artifact["no_model_weight_mutation"] is True
    assert "no_mandated_sota_gguf_runtime" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "summary"
    assert rows[-1]["live_sota_model_inference_used"] is False


def test_req_learn_1524_false_accept_blocks_readiness(tmp_path: Path) -> None:
    """REQ-LEARN-1524-7: promoted false accepts prevent readiness."""

    paths = _write_sources(tmp_path)
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["promotion_manifest"],
        policy_cache_artifact_path=paths["policy_artifact"],
        policy_cache_manifest_path=paths["policy_manifest"],
        rollback_artifact_path=paths["rollback_artifact"],
        rollback_manifest_path=paths["rollback_manifest"],
        portable_pack_artifact_path=paths["pack_artifact"],
        portable_pack_manifest_path=paths["pack_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=lambda _prompt, _model, mode, _update, case: json.dumps(
            {
                "contract_case_id": case["contract_case_id"],
                "final_deterministic_decision": "reject" if mode == "baseline" else "accept",
            }
        ),
        gpu_probe_fn=lambda: {"cuda_available": True, "gpu_count": 1},
        update_limit=1,
        case_limit=1,
    )

    assert artifact["live_policy_promotion_ready"] is False
    assert artifact["false_accept_delta"] == 1
    assert artifact["soundness_mistakes"] == 1
    assert "false_accept_delta_positive" in artifact["blockers"]
    assert "soundness_mistakes_nonzero" in artifact["blockers"]


def test_req_learn_1524_blocks_empty_candidates_cases_and_rows(tmp_path: Path) -> None:
    """REQ-LEARN-1524-7: empty bounded sets never become ready."""

    paths = _write_sources(tmp_path)
    _write_jsonl(paths["rollback_manifest"], [_rollback_row("not-packaged")])
    _write_json(paths["pack_manifest"], _pack_manifest([]))
    _write_jsonl(
        paths["runtime_manifest"],
        [_contract_case("accept-ok", expected_label=True, final_accept=True)],
    )
    no_candidates = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["promotion_manifest"],
        policy_cache_artifact_path=paths["policy_artifact"],
        policy_cache_manifest_path=paths["policy_manifest"],
        rollback_artifact_path=paths["rollback_artifact"],
        rollback_manifest_path=paths["rollback_manifest"],
        portable_pack_artifact_path=paths["pack_artifact"],
        portable_pack_manifest_path=paths["pack_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=lambda *_args, **_kwargs: "must not be called",
    )
    _write_sources(tmp_path)
    no_rows = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["promotion_manifest"],
        policy_cache_artifact_path=paths["policy_artifact"],
        policy_cache_manifest_path=paths["policy_manifest"],
        rollback_artifact_path=paths["rollback_artifact"],
        rollback_manifest_path=paths["rollback_manifest"],
        portable_pack_artifact_path=paths["pack_artifact"],
        portable_pack_manifest_path=paths["pack_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=lambda *_args, **_kwargs: "must not be called",
        update_limit=0,
        case_limit=1,
    )

    assert "no_rollback_passing_reachable_updates" in no_candidates["blockers"]
    assert "no_exp1520_explicit_contract_cases" in no_candidates["blockers"]
    assert "no_live_policy_promotion_rows" in no_rows["blockers"]


def test_req_learn_1524_defensive_contract_and_source_blockers(tmp_path: Path) -> None:
    """REQ-LEARN-1524-2/8: malformed gates fail with explicit blockers."""

    artifact = mod.write_in_progress_artifact(
        tmp_path / mod.OUTPUT_FILE,
        manifest_path=tmp_path / mod.MANIFEST_FILE,
        project_root=tmp_path,
    )
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(artifact, status="weird"))
    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(dict(artifact, honest_verdict="not_terminal"))

    ready = dict(
        artifact,
        status="complete",
        live_policy_promotion_ready=True,
        live_sota_model_inference_used=True,
        soundness_mistakes=0,
        false_accept_delta=0,
        no_model_weight_mutation=True,
        promoted_policy_updates=["daily_eval:eligible"],
        blockers=[],
    )
    ready_checks = [
        ("live_sota_model_inference_used", False, "live SOTA inference"),
        ("soundness_mistakes", 1, "zero soundness mistakes"),
        ("false_accept_delta", 1, "increase false accepts"),
        ("no_model_weight_mutation", False, "frozen model weights"),
        ("promoted_policy_updates", [], "promoted policy updates"),
    ]
    for key, value, message in ready_checks:
        with pytest.raises(AssertionError, match=message):
            mod.validate_artifact(dict(ready, **{key: value}), manifest_path=tmp_path)
    with pytest.raises(AssertionError, match="promotion manifest"):
        mod.validate_artifact(ready, manifest_path=tmp_path / "missing.jsonl")

    bad_pack = mod.select_promotable_updates(
        policy_rows=[_policy_row("bad-pack")],
        rollback_rows=[_rollback_row("bad-pack")],
        pack_manifest={
            "entries": [
                "not-a-mapping",
                {
                    "source_event_id": "daily_eval:bad-pack",
                    "skill_id": "fr11_v10_trace2skill/bad-pack",
                    "promotion_status": "rejected_not_promoted",
                },
            ]
        },
    )
    no_entries = mod.select_promotable_updates(
        policy_rows=[_policy_row("no-pack")],
        rollback_rows=[_rollback_row("no-pack")],
        pack_manifest={"entries": "bad"},
    )
    assert "portable_provenance_not_packaged" in bad_pack["rejected"][0]["rejection_reasons"]
    assert "missing_portable_provenance" in no_entries["rejected"][0]["rejection_reasons"]

    paths = _write_sources(tmp_path)
    _write_json(paths["policy_artifact"], {"policy_cache_ready": False})
    _write_json(paths["rollback_artifact"], {"rollback_audit_passed": False})
    _write_json(paths["pack_artifact"], {"portable_skill_pack_ready": False})
    _write_json(paths["runtime_artifact"], {"runtime_contract_e2e_ready": False})
    paths["policy_manifest"].unlink()
    _sources, blockers = mod._load_required_sources(
        {
            "policy_artifact": paths["policy_artifact"],
            "policy_manifest": paths["policy_manifest"],
            "rollback_artifact": paths["rollback_artifact"],
            "rollback_manifest": paths["rollback_manifest"],
            "pack_artifact": paths["pack_artifact"],
            "pack_manifest": paths["pack_manifest"],
            "runtime_artifact": paths["runtime_artifact"],
            "runtime_manifest": paths["runtime_manifest"],
        }
    )
    assert "exp1512_policy_cache_not_ready" in blockers
    assert "exp1513_rollback_audit_not_passed" in blockers
    assert "exp1514_portable_pack_not_ready" in blockers
    assert "exp1520_runtime_contract_not_ready" in blockers
    assert any(blocker.startswith("missing_policy_manifest:") for blocker in blockers)

    missing_blockers: list[str] = []
    malformed_blockers: list[str] = []
    assert mod._load_json_or_blocker(tmp_path / "missing.json", missing_blockers) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert mod._load_json_or_blocker(malformed, malformed_blockers) is None
    assert any(blocker.startswith("missing_artifact:") for blocker in missing_blockers)
    assert any(blocker.startswith("malformed_artifact:") for blocker in malformed_blockers)

    assert mod._display_path(Path("/outside/repo/file.json"), project_root=tmp_path) == (
        "/outside/repo/file.json"
    )
    with pytest.raises(AssertionError, match="JSON artifact"):
        mod._read_json(malformed)
    malformed_jsonl = tmp_path / "malformed.jsonl"
    malformed_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="JSONL row"):
        mod._read_jsonl(malformed_jsonl)


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "output": tmp_path / mod.OUTPUT_FILE,
        "promotion_manifest": tmp_path / mod.MANIFEST_FILE,
        "policy_artifact": tmp_path / "experiment_1512.json",
        "policy_manifest": tmp_path / "policy.jsonl",
        "rollback_artifact": tmp_path / "experiment_1513.json",
        "rollback_manifest": tmp_path / "rollback.jsonl",
        "pack_artifact": tmp_path / "experiment_1514.json",
        "pack_manifest": tmp_path / "pack.json",
        "runtime_artifact": tmp_path / "experiment_1520.json",
        "runtime_manifest": tmp_path / "runtime.jsonl",
    }
    _write_json(paths["policy_artifact"], {"status": "complete", "policy_cache_ready": True})
    _write_jsonl(paths["policy_manifest"], [_policy_row("eligible")])
    _write_json(paths["rollback_artifact"], {"status": "complete", "rollback_audit_passed": True})
    _write_jsonl(paths["rollback_manifest"], [_rollback_row("eligible")])
    _write_json(
        paths["pack_artifact"],
        {"status": "complete", "portable_skill_pack_ready": True},
    )
    _write_json(paths["pack_manifest"], _pack_manifest(["eligible"]))
    _write_json(
        paths["runtime_artifact"],
        {"status": "complete", "runtime_contract_e2e_ready": True},
    )
    _write_jsonl(
        paths["runtime_manifest"],
        [_contract_case("contract-reject", expected_label=False, final_accept=False)],
    )
    return paths


def _policy_row(case_id: str) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_cache_event_v1",
        "spec": ["REQ-LEARN-1512"],
        "source_event_id": f"daily_eval:{case_id}",
        "source_case_id": case_id,
        "source_kind": "daily_eval",
        "skill_id": f"fr11_v10_trace2skill/{case_id}",
        "policy_action": "retrieval_boost",
        "accepted": True,
        "quarantined": False,
        "deterministic_validation_observed": True,
    }


def _rollback_row(
    case_id: str,
    *,
    decision: str = "keep",
    reachable: bool = True,
    stale: bool = False,
    deterministic: bool = True,
    soundness_mistakes: int = 0,
    false_accept_delta: int = 0,
    rollback_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_rollback_replay_row_v1",
        "spec": ["REQ-LEARN-1513"],
        "source_event_id": f"daily_eval:{case_id}",
        "source_case_id": case_id,
        "source_kind": "daily_eval",
        "skill_id": f"fr11_v10_trace2skill/{case_id}",
        "policy_action": "retrieval_boost",
        "decision": decision,
        "source_evidence_reachable": reachable,
        "source_evidence_stale": stale,
        "deterministic_validator_supported": deterministic,
        "soundness_mistakes": soundness_mistakes,
        "false_accept_delta": false_accept_delta,
        "utility_delta": 1,
        "rollback_reasons": list(rollback_reasons or []),
    }


def _pack_manifest(case_ids: list[str]) -> dict[str, Any]:
    return {
        "schema": "trace2skill_portable_skill_pack_v1",
        "entries": [
            {
                "skill_id": f"fr11_v10_trace2skill/{case_id}",
                "source_event_id": f"daily_eval:{case_id}",
                "resolver_key": f"daily_eval:{case_id}",
                "promotion_status": "packaged_rollback_passed",
                "source_artifact": "results/fr11_policy_rollback_replay_1513.jsonl",
                "verifier_evidence": {
                    "rollback_decision": "keep",
                    "source_evidence_reachable": True,
                    "source_evidence_stale": False,
                    "deterministic_validator_supported": True,
                    "soundness_mistakes": 0,
                    "false_accept_delta": 0,
                },
                "created_date": "20260508",
            }
            for case_id in case_ids
        ],
        "rejected_entries": [],
    }


def _contract_case(
    contract_case_id: str,
    *,
    expected_label: bool | None,
    final_accept: bool,
) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": contract_case_id,
        "prompt_or_case_id": contract_case_id,
        "proposed_output": "candidate output",
        "certificate_parse_result": {"linked": False},
        "safe_dsl_verifier_result": {"linked": False},
        "monitor_event_result": {"linked": False},
        "structural_contract_result": {"linked": True, "contract_family": "unit"},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": "structural_contract",
        "source_path": "source.jsonl",
        "source_line": 1,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
