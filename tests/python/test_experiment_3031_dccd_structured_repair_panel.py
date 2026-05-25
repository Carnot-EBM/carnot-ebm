"""Tests for Exp 3031 tiny DCCD structured repair panel.

Spec: REQ-CODE-3031, SCENARIO-CODE-3031.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import dccd_structured_repair_panel_3031 as exp
from carnot.eval import hard_code_stress_manifest as hard


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _controller_rule() -> dict[str, bool]:
    return {
        "require_schema_valid": True,
        "require_syntax_success": True,
        "require_entry_point_present": True,
        "require_false_accept_probe_clean": True,
        "require_no_intent_drift": True,
        "require_original_passed": True,
        "require_metamorphic_passed_all": True,
        "require_tautology_probe_clean": True,
    }


def _write_ready_sources(tmp_path: Path, *, n_cases: int = 2) -> Path:
    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"tiny-test-model-placeholder")
    items = [dict(item) for item in hard.default_items()[:n_cases]]
    _write_jsonl(tmp_path / exp.HARD_MANIFEST_REL_PATH, items)
    _write_json(
        tmp_path / exp.CONTROLLER_CONFIG_REL_PATH,
        {
            "policy_type": "transparent_grid_rule",
            "selected_rule": _controller_rule(),
            "llm_judge_used": False,
        },
    )
    _write_json(
        tmp_path / exp.EXP3015_REL_PATH,
        {
            "acceptance_controller_ready": True,
            "controller_config_path": exp.CONTROLLER_CONFIG_REL_PATH.as_posix(),
        },
    )
    return model_path


def test_req_code_3031_spec_anchor_and_required_fields_exist() -> None:
    """REQ-CODE-3031: the DCCD panel is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = REPO_ROOT / "scripts/experiment_3031_dccd_structured_repair_panel_v1.py"

    assert "REQ-CODE-3031" in spec
    assert "SCENARIO-CODE-3031" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) >= {
        "dccd_panel_ready",
        "n_cases",
        "model_specs",
        "legacy_smoke_only_used",
        "baseline_acceptance_metrics",
        "dccd_metrics",
        "intent_drift_delta",
        "false_accept_delta",
        "changed_files",
        "tests_run",
        "inference_substrate",
        "honest_verdict",
    }
    assert script.is_file()


def test_req_code_3031_schema_projection_extracts_json_and_fences() -> None:
    """REQ-CODE-3031: constrained projection keeps the draft intent and patch separate."""

    patch = "def clamp_score(x, lo, hi):\n    return max(lo, min(x, hi))\n"
    raw_json = json.dumps(
        {"draft_intent": "Clamp x into inclusive bounds.", "final_patch": patch}
    )
    projected = exp.parse_schema_candidate(
        raw_json,
        fallback_draft="fallback draft",
        entry_point="clamp_score",
    )
    fenced = exp.extract_python_candidate(
        "analysis\n```python\n" + patch + "```\nfinal",
        entry_point="clamp_score",
    )

    assert projected.schema_valid is True
    assert projected.draft_intent == "Clamp x into inclusive bounds."
    assert projected.final_patch == patch
    assert projected.schema_errors == []
    assert fenced == patch

    bad = exp.parse_schema_candidate("{}", fallback_draft="", entry_point="clamp_score")

    assert bad.schema_valid is False
    assert "final_patch missing" in bad.schema_errors


def test_scenario_code_3031_metrics_compare_acceptance_and_dccd() -> None:
    """SCENARIO-CODE-3031: metric deltas compare DCCD against acceptance-only."""

    acceptance_rows = [
        {
            "accepted": True,
            "passed": True,
            "strict_valid": True,
            "schema_valid": True,
            "syntax_success": True,
            "false_accept": False,
            "intent_drift": False,
        },
        {
            "accepted": False,
            "passed": False,
            "strict_valid": False,
            "schema_valid": True,
            "syntax_success": True,
            "false_accept": False,
            "intent_drift": True,
        },
    ]
    dccd_rows = [
        {**acceptance_rows[0], "condition": exp.DCCD_MODE},
        {
            **acceptance_rows[1],
            "condition": exp.DCCD_MODE,
            "accepted": True,
            "passed": True,
            "strict_valid": True,
            "intent_drift": False,
        },
    ]

    baseline = exp.condition_metrics(acceptance_rows, n_cases=2)
    dccd = exp.condition_metrics(dccd_rows, n_cases=2)
    deltas = exp.metric_deltas(baseline, dccd)

    assert baseline["accepted_count"] == 1
    assert baseline["pass_rate"] == pytest.approx(0.5)
    assert baseline["intent_drift_count"] == 0
    assert dccd["accepted_count"] == 2
    assert dccd["pass_rate"] == pytest.approx(1.0)
    assert deltas["intent_drift_delta"] == pytest.approx(0.0)
    assert deltas["false_accept_delta"] == pytest.approx(0.0)


def test_scenario_code_3031_builds_ready_panel_with_injected_generator(tmp_path: Path) -> None:
    """SCENARIO-CODE-3031: a live-generation substitute can produce a complete panel."""

    model_path = _write_ready_sources(tmp_path, n_cases=2)
    items = list(hard.default_items()[:2])

    def generator(
        case: exp.PanelCase,
        mode: str,
        draft_text: str | None,
        model_spec: dict[str, Any],
    ) -> exp.GenerationResult:
        assert model_spec["hf_id"] == HEADLINE_MODEL
        if mode == exp.UNCONSTRAINED_MODE and case.item_id == items[0]["item_id"]:
            return exp.GenerationResult(
                raw_text=(
                    "Clamp x into the inclusive range.\n"
                    f"```python\n{items[0]['reference_solution']}```"
                ),
                duration_s=0.1,
                tokens_generated=16,
            )
        if mode == exp.UNCONSTRAINED_MODE:
            return exp.GenerationResult(
                raw_text=items[1]["baseline_candidate"],
                duration_s=0.1,
                tokens_generated=16,
            )
        assert draft_text
        return exp.GenerationResult(
            raw_text=json.dumps(
                {
                    "draft_intent": case.expected_behavior,
                    "final_patch": case.reference_solution,
                }
            ),
            duration_s=0.1,
            tokens_generated=32,
        )

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            selected_model_path=model_path,
            selected_model_id=HEADLINE_MODEL,
            started_at=10.0,
            clock=lambda: 12.0,
            tests_run=("SCENARIO-CODE-3031-focused",),
            changed_files=("python/carnot/eval/dccd_structured_repair_panel_3031.py",),
        ),
        generator_fn=generator,
    )
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["dccd_panel_ready"] is True
    assert artifact["n_cases"] == 2
    assert artifact["model_specs"] == [
        {
            "hf_id": HEADLINE_MODEL,
            "model_path": str(model_path),
            "role": "headline_live_generation",
        }
    ]
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["baseline_acceptance_metrics"]["accepted_count"] == 1
    assert artifact["dccd_metrics"]["accepted_count"] == 2
    assert artifact["dccd_metrics"]["pass_rate"] == pytest.approx(1.0)
    assert artifact["intent_drift_delta"] == pytest.approx(0.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["model_load_attempted"] is True
    assert artifact["inference_substrate"]["live_generation_succeeded"] is True
    assert artifact["case_results"][1]["acceptance_only"]["accepted"] is False
    assert artifact["case_results"][1]["draft_conditioned_constrained"]["accepted"] is True


def test_req_code_3031_blocks_when_headline_model_unavailable(tmp_path: Path) -> None:
    """REQ-CODE-3031: no loadable headline GGUF yields the mandated blocked verdict."""

    _write_ready_sources(tmp_path, n_cases=1)

    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            selected_model_path=tmp_path / "missing.gguf",
            selected_model_id=HEADLINE_MODEL,
            started_at=1.0,
            clock=lambda: 1.5,
        ),
        generator_fn=None,
    )

    assert artifact["dccd_panel_ready"] is False
    assert artifact["n_cases"] == 0
    assert artifact["model_specs"] == []
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["honest_verdict"].startswith("blocked_sota_headline_model_unavailable")
    assert artifact["inference_substrate"]["selected_headline_model"]["available"] is False


def test_req_code_3031_helper_edges_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-3031: deterministic helper edges fail closed without live loading."""

    explicit_controller = tmp_path / "controller.json"
    explicit_meta = tmp_path / "meta.jsonl"
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        controller_config_path=explicit_controller,
        metamorphic_manifest_path=explicit_meta,
    )
    item = dict(hard.default_items()[0])
    passing_baseline = {**item, "baseline_candidate": item["reference_solution"]}
    case = exp.select_panel_cases([item], [item["item_id"]])[0]

    assert config.resolved_controller_config_path() == explicit_controller
    assert config.resolved_metamorphic_manifest_path() == explicit_meta
    assert exp.select_panel_cases([passing_baseline], [item["item_id"]]) == []
    assert exp.select_panel_cases([item], ["missing-item"]) == []
    assert exp.extract_python_candidate("", entry_point=case.entry_point) == ""
    assert exp.extract_python_candidate(
        json.dumps({"final_patch": item["reference_solution"]}),
        entry_point=case.entry_point,
    ) == item["reference_solution"]
    embedded = (
        "draft text\n"
        f"def {case.entry_point}(x, lo, hi):\n"
        "    return max(lo, min(x, hi))\n"
        "not code"
    )
    assert exp.extract_python_candidate(embedded, entry_point=case.entry_point).startswith(
        f"def {case.entry_point}"
    )
    projected = exp.parse_schema_candidate(
        "not json\n```python\n" + item["reference_solution"] + "```",
        fallback_draft="fallback draft",
        entry_point=case.entry_point,
    )
    assert projected.schema_valid is False
    assert projected.final_patch == item["reference_solution"]

    false_accept_variant = {
        **item,
        "source_item_id": item["item_id"],
        "source_entry_point": item["entry_point"],
        "tests": [
            {
                "test_id": "SCENARIO-CODE-3031-false",
                "code": "assert clamp_score(1, 0, 2) == 999",
            }
        ],
    }
    evaluated = exp.evaluate_candidate(
        case=case,
        patch_text=item["reference_solution"],
        draft_intent=item["expected_behavior"],
        schema_valid=True,
        schema_errors=[],
        condition=exp.DCCD_MODE,
        variants=[false_accept_variant],
        accepted=True,
    )
    assert evaluated["false_accept"] is True
    assert evaluated["metamorphic_variant_count"] == 1
    assert exp.syntax_diagnostics("")[0] is False
    assert exp.syntax_diagnostics("def broken(:\n")[0] is False
    assert exp.intent_preserved("", item["expected_behavior"]) is False
    assert exp.intent_preserved("only stop words", "and the") is True
    assert exp.content_tokens("finaltoken") == ["finaltoken"]
    assert exp._load_hard_items(exp.ExperimentConfig(repo_root=tmp_path / "missing")) == []
    bad_hard = tmp_path / "bad-hard.jsonl"
    bad_hard.write_text("{bad\n", encoding="utf-8")
    assert (
        exp._load_hard_items(exp.ExperimentConfig(repo_root=tmp_path, hard_manifest_path=bad_hard))
        == []
    )
    explicit_meta.write_text(json.dumps(false_accept_variant) + "\n", encoding="utf-8")
    assert exp._load_metamorphic_variants(config)[0]["source_item_id"] == item["item_id"]
    explicit_meta.write_text("{bad\n", encoding="utf-8")
    assert exp._load_metamorphic_variants(config) == []
    monkeypatch.setattr(exp, "_hf_cache_candidates", lambda _model_id: [])
    assert exp._select_headline_model(exp.ExperimentConfig(repo_root=tmp_path)) == {
        "hf_id": None,
        "path": None,
        "available": False,
        "source": "not_found",
    }

    def raising_generator(
        _case: exp.PanelCase,
        _mode: str,
        _draft_text: str | None,
        _model_spec: dict[str, Any],
    ) -> exp.GenerationResult:
        raise RuntimeError("boom")

    assert exp._safe_generate(raising_generator, case, exp.UNCONSTRAINED_MODE, None, {}).error == (
        "RuntimeError: boom"
    )

    def error_generator(
        _case: exp.PanelCase,
        _mode: str,
        _draft_text: str | None,
        _model_spec: dict[str, Any],
    ) -> exp.GenerationResult:
        return exp.GenerationResult("", error="model failed")

    failed_case, errors = exp._run_case(
        case=case,
        generator_fn=error_generator,
        model_spec={},
        controller_rule={},
        variants=[],
    )
    assert failed_case["item_id"] == item["item_id"]
    assert errors == [
        f"{item['item_id']}:{exp.UNCONSTRAINED_MODE}:model failed",
        f"{item['item_id']}:{exp.DCCD_MODE}:model failed",
    ]
    assert exp._json_object_from_text("") is None
    assert exp._json_object_from_text("```json\n{\"x\": 1}\n```") == {"x": 1}
    assert exp._json_object_from_text("```json\n{bad}\n```") is None
    assert exp._patch_from_json_text("{}") == ""
    assert exp._entry_point_present("def broken(:\n", "broken") is False
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad\n", encoding="utf-8")
    assert exp._read_json_if_present(bad_json) == {}
    assert exp._path_string(tmp_path, tmp_path.parent / "outside.txt").endswith("outside.txt")
    assert exp._honest_verdict(ready=False, blocked=False, n_cases=1).startswith(
        "complete_flagged:"
    )
