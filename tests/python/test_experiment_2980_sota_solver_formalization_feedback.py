"""Tests for Exp 2980 SOTA solver formalization feedback.

Spec refs: REQ-VERIFY-2980, SCENARIO-VERIFY-2980.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import logic_frontier_materializer as exp2966
from carnot.eval import sota_solver_formalization_feedback_v2 as exp


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _spec(hf_id: str = MANDATED) -> dict[str, Any]:
    return {
        "name": "Gemma4-26B-A4B-it" if hf_id == MANDATED else "LegacyTiny",
        "hf_id": hf_id,
        "gpu": 0,
        "model_path": "/tmp/model.gguf",
    }


def _frontier_rows(limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in exp2966.build_logic_frontier_items()[:limit]:
        record = item.to_manifest_record()
        rows.append(
            {
                "item_id": record["item_id"],
                "prompt": record["prompt"],
                "skill_label": record["skill_labels"][0],
                "skill_labels": record["skill_labels"],
                "expected_solver_status": record["expected_solver_status"],
                "accepted_reference_formalization": {
                    "format": "smt2",
                    "assertions": record["reference_z3"]["assertions"],
                    "expected_solver_status": record["expected_solver_status"],
                    "expected_answer_values": record["expected_answer_values"],
                },
                "solver_feedback": {
                    "parse_error": "no_json_object",
                    "z3_exception": None,
                    "model_counterexample": {"solver_status": "sat"}
                    if record["expected_solver_status"] == "sat"
                    else None,
                    "unsat_core_or_mus": {"unsat_core": ["premise", "negated_goal"]}
                    if record["expected_solver_status"] == "unsat"
                    else None,
                    "minimal_correction_hint": "Use the exact predicate inventory.",
                    "skill_label": record["skill_labels"][0],
                    "accepted_reference_formalization": {
                        "format": "smt2",
                        "assertions": record["reference_z3"]["assertions"],
                        "expected_solver_status": record["expected_solver_status"],
                        "expected_answer_values": record["expected_answer_values"],
                    },
                },
            }
        )
    return rows


def _write_frontier(path: Path, *, ready: bool = True, limit: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: fixture" if ready else "blocked: fixture",
                "mcs_feedback_schema_ready": ready,
                "frontier_items": _frontier_rows(limit),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        feedback_frontier_path=tmp_path / "results" / exp.EXP2979_FILENAME,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "raw",
        started_at=10.0,
        clock=lambda: 15.0,
        monotonic_clock=lambda: 20.0,
    )


def _collect_unparseable(
    spec: dict[str, Any],
    items: list[exp.FeedbackFrontierItem],
    _config_arg: exp.ExperimentConfig,
) -> dict[str, Any]:
    return {
        "summary": {
            "hf_id": spec["hf_id"],
            "model_name": spec["name"],
            "model_path": spec.get("model_path"),
            "model_used": True,
            "blocker": None,
        },
        "rows": [
            {
                "item_id": item.item_id,
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec.get("model_path"),
                "gpu_index": spec.get("gpu"),
                "prompt_hash": exp.sha256_text(item.prompt),
                "per_item_seed": exp.RANDOM_SEED + index,
                "generation_source": "live_feedback_aware_formalization",
                "output_text": "not json",
                "raw_response_path": f"/tmp/{item.item_id}.json",
                "elapsed_seconds": 0.1,
                "blocker": None,
            }
            for index, item in enumerate(items)
        ],
    }


def _collect_reference(
    spec: dict[str, Any],
    items: list[exp.FeedbackFrontierItem],
    _config_arg: exp.ExperimentConfig,
) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "item_id": item.item_id,
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec.get("model_path"),
                "gpu_index": spec.get("gpu"),
                "prompt_hash": exp.sha256_text(item.prompt),
                "per_item_seed": exp.RANDOM_SEED + index,
                "generation_source": "live_feedback_aware_formalization",
                "output_text": json.dumps(exp.repair_proposal_from_feedback(item).to_dict()),
                "raw_response_path": f"/tmp/{item.item_id}.json",
                "elapsed_seconds": 0.1,
                "blocker": None,
            }
        )
    return {
        "summary": {
            "hf_id": spec["hf_id"],
            "model_name": spec["name"],
            "model_path": spec.get("model_path"),
            "model_used": True,
            "blocker": None,
        },
        "rows": rows,
    }


def _collect_no_rows(
    spec: dict[str, Any],
    _items: list[exp.FeedbackFrontierItem],
    _config_arg: exp.ExperimentConfig,
) -> dict[str, Any]:
    return {
        "summary": {
            "hf_id": spec["hf_id"],
            "model_name": spec["name"],
            "model_path": spec.get("model_path"),
            "model_used": False,
            "blocker": "no_usable_generations",
        },
        "rows": [],
    }


def test_req_verify_2980_spec_anchor_exists() -> None:
    """REQ-VERIFY-2980: the feedback-aware formalizer is OpenSpec anchored."""
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-2980" in spec
    assert "SCENARIO-VERIFY-2980" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference_plus_z3"' in spec
    assert "tautology_flag_rate==0" in spec


def test_scenario_verify_2980_repairs_feedback_rows_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2980: one deterministic feedback repair pass clears gates."""
    _write_frontier(tmp_path / "results" / exp.EXP2979_FILENAME)
    calls: list[dict[str, Any]] = []

    artifact = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **kwargs: calls.append(kwargs) or [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_unparseable,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert calls == [{"gpu_indices": (0, 1)}]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["formalization_feedback_clean"] is True
    assert artifact["headline_result"] is True
    assert artifact["n_items"] == 3
    assert artifact["models_used"] == [MANDATED]
    assert artifact["mandatory_headline_model_ids"] == list(exp.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["parseability_rate"] == pytest.approx(1.0)
    assert artifact["z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["solver_verified_accuracy"] == pytest.approx(1.0)
    assert artifact["answer_accuracy"] == pytest.approx(1.0)
    assert artifact["tautology_flag_rate"] == pytest.approx(0.0)
    assert artifact["feedback_repair_delta"] == pytest.approx(1.0)
    assert artifact["failure_categories"]["solver_verified_correct"] == 3
    assert artifact["solver_feedback_examples"]
    assert artifact["per_skill_metrics"]["symbolization"]["n_items"] >= 1
    assert artifact["per_item_results"][0]["initial_result"]["failure_category"] == "unparseable"
    assert artifact["per_item_results"][0]["repair_attempted"] is True
    assert artifact["per_item_results"][0]["final_result"]["failure_category"] == "solver_verified_correct"
    exp.validate_artifact(artifact)


def test_req_verify_2980_preconditions_legacy_policy_and_validation(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2980: blocked and legacy-only runs cannot claim clean rows."""
    _write_frontier(tmp_path / "results" / exp.EXP2979_FILENAME, ready=False)
    not_ready = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_reference,
    )
    _write_frontier(tmp_path / "results" / exp.EXP2979_FILENAME, ready=True)
    no_z3 = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_reference,
        z3_module=None,
    )
    no_model = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_reference,
    )
    no_rows = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_no_rows,
    )
    legacy = exp.run_experiment(
        replace(_config(tmp_path), allow_legacy_tiny_fallback=True),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=_collect_reference,
    )

    assert not_ready["honest_verdict"] == "blocked_precondition: exp2979_feedback_not_ready"
    assert no_z3["honest_verdict"] == "blocked_precondition: z3_import_failed"
    assert no_model["honest_verdict"] == "blocked_precondition: headline_gguf_missing"
    assert no_rows["headline_result"] is False
    assert no_rows["honest_verdict"] == "complete: no mandated local SOTA model produced usable proposal rows"
    assert legacy["headline_result"] is False
    assert legacy["legacy_models_only_for_smoke"] is True
    assert legacy["formalization_feedback_clean"] is False
    assert legacy["parseability_rate"] == pytest.approx(1.0)
    assert (
        exp._honest_verdict(False, True, False)
        == "complete: feedback-aware local SOTA formalization did not clear .280 Z3 gates"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "x"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(legacy | {"inference_substrate": "deterministic"})
    with pytest.raises(ValueError, match="formalization_feedback_clean"):
        exp.validate_artifact(legacy | {"formalization_feedback_clean": True})
    with pytest.raises(ValueError, match="headline_result"):
        exp.validate_artifact(legacy | {"headline_result": True})
    with pytest.raises(ValueError, match="mandated model_specs"):
        exp.validate_artifact(
            legacy
            | {
                "headline_result": True,
                "legacy_models_only_for_smoke": False,
                "models_used": [MANDATED],
                "model_specs": [{"hf_id": "legacy"}],
            }
        )

    recovered_single = exp.check_preconditions(
        _config(tmp_path),
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("cache down")),
        individual_model_resolver=lambda hf_id: "/tmp/gemma.gguf" if hf_id == MANDATED else None,
    )
    llama_missing = exp.check_preconditions(
        _config(tmp_path),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        module_importer=lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    assert recovered_single.block_reason is None
    assert recovered_single.model_specs == [{**_spec(), "model_path": "/tmp/gemma.gguf"}]
    assert any("cache down" in row["detail"] for row in recovered_single.rows)
    assert llama_missing.block_reason == "llama_cpp_import_failed"


def test_req_verify_2980_parser_tautology_repair_and_collector_edges(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2980: parser, tautology guard, and live collector edges are deterministic."""
    _write_frontier(tmp_path / "results" / exp.EXP2979_FILENAME, limit=1)
    cfg = _config(tmp_path)
    item = exp.load_feedback_frontier(cfg)[0]
    repair = exp.repair_proposal_from_feedback(item)
    tautology, _ = exp.parse_structured_proposal(
        json.dumps(
            {
                "variables": [],
                "predicates": [],
                "assertions": ["(assert true)"],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": {},
            }
        )
    )

    assert repair.expected_status == item.expected_solver_status
    assert exp.repair_proposal_from_feedback(
        exp.FeedbackFrontierItem(
            item_id=item.item_id,
            prompt=item.prompt,
            skill_label=item.skill_label,
            skill_labels=item.skill_labels,
            expected_solver_status=item.expected_solver_status,
            expected_answer_values=item.expected_answer_values,
            accepted_reference_formalization=item.accepted_reference_formalization,
            solver_feedback={},
        )
    ).expected_status == item.expected_solver_status
    feedback_only_path = tmp_path / "results" / "feedback_only.json"
    feedback_only_rows = _frontier_rows(1)
    feedback_only_rows[0].pop("accepted_reference_formalization")
    feedback_only_path.write_text(
        json.dumps({"mcs_feedback_schema_ready": True, "frontier_items": feedback_only_rows}),
        encoding="utf-8",
    )
    assert exp.load_feedback_frontier(replace(cfg, feedback_frontier_path=feedback_only_path))[0].accepted_reference_formalization
    empty_path = tmp_path / "results" / "empty.json"
    empty_path.write_text(json.dumps({"mcs_feedback_schema_ready": True, "frontier_items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="exp2979_frontier_items_missing"):
        exp.load_feedback_frontier(replace(cfg, feedback_frontier_path=empty_path))
    array_path = tmp_path / "results" / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp.load_feedback_frontier(replace(cfg, feedback_frontier_path=array_path))
    assert exp.tautology_flag(tautology) is True
    assert exp.tautology_flag(repair) is False
    assert exp.tautology_flag_from_dict(None) is False
    assert exp.tautology_flag_from_dict({"assertions": []}) is True
    assert exp.tautology_flag_from_dict({"assertions": ["(declare-const x Int)"]}) is True
    assert exp.formalization_feedback_clean(
        {
            "parseability_rate": 0.5,
            "z3_execution_rate": 0.5,
            "solver_verified_accuracy": 0.4,
            "answer_accuracy": 0.4,
            "tautology_flag_rate": 0.0,
        },
        headline_result=True,
    )
    assert exp.completion_text({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert exp.completion_text({"choices": [{"text": "plain"}]}) == "plain"
    assert exp.completion_text({"choices": []}) == ""
    assert exp.completion_text(17) == ""

    class FakeLlama:
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"text": json.dumps(repair.to_dict())}]}

        def close(self) -> None:
            type(self).closed = True

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("generation failed")

        def close(self) -> None:
            return None

    class EmptyGeneration:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"text": ""}]}

        def close(self) -> None:
            return None

    ok = exp.collect_live_feedback_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    missing_path = exp.collect_live_feedback_outputs(
        {**_spec(), "model_path": ""},
        [item],
        cfg,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    import_failed = exp.collect_live_feedback_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
    )
    load_failed = exp.collect_live_feedback_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, LoadFails, None),
    )
    gen_failed = exp.collect_live_feedback_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, GenerateFails, None),
    )
    empty = exp.collect_live_feedback_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, EmptyGeneration, None),
    )

    assert ok["summary"]["model_used"] is True
    assert ok["rows"][0]["output_text"].startswith("{")
    assert ok["rows"][0]["raw_response_sha256"] == exp.sha256_text(ok["rows"][0]["output_text"])
    assert FakeLlama.closed is True
    assert missing_path["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert load_failed["summary"]["blocker"] == "RuntimeError: load failed"
    assert gen_failed["summary"]["model_used"] is False
    assert gen_failed["rows"][0]["blocker"] == "ValueError: generation failed"
    assert empty["rows"][0]["blocker"] == "empty_generation"
