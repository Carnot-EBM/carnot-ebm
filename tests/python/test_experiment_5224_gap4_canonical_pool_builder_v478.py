"""Tests for Exp 5224 GAP-4 canonical pool builder.

Spec refs: REQ-REPORT-5224, SCENARIO-REPORT-5224-REGENERATE,
SCENARIO-REPORT-5224-REPAIR-ONLY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5223_gap4_flagged_pool_authenticity_audit_v478 as exp5223
from carnot import experiment_5224_gap4_canonical_pool_builder_v478 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _install_schema(root: Path) -> None:
    source = REPO / exp5223.CANONICAL_SCHEMA_RELATIVE_PATH
    target = root / exp5223.CANONICAL_SCHEMA_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _audit(root: Path, *, repairable: bool) -> None:
    _write_json(
        root / exp5223.RESULT_RELATIVE_PATH,
        {
            "gap4_pool_repairable": repairable,
            "canonical_schema_path": exp5223.CANONICAL_SCHEMA_RELATIVE_PATH,
            "preflight_reasons": [] if repairable else ["missing_random_seed"],
        },
    )


def _tasks(n: int) -> list[JsonDict]:
    return [
        {
            "task_id": f"human_replay:unit:{idx:04d}",
            "source": "unit_human_replay",
            "demos": [{"input": [[idx % 10]], "output": [[idx % 10]]}],
            "test_input": [[idx % 10]],
            "test_shape": [1, 1],
        }
        for idx in range(n)
    ]


def _model_specs() -> list[JsonDict]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        }
    ]


def _certificate(_prompt: str, model_spec: JsonDict, seed: int) -> mod.GenerationCertificate:
    return mod.GenerationCertificate(
        model_id=str(model_spec["hf_id"]),
        model_path_or_digest=str(model_spec["model_path"]),
        started_at="2026-07-04T12:00:00Z",
        duration_s=65.0,
        completion_text='{"template":"identity_same_shape","confidence":"readiness_only"}',
        random_seed=seed,
        backend="unit_constrained_generation",
    )


def _adversarial_clean(path: Path) -> JsonDict:
    return {
        "passed": True,
        "reports": [{"artifact": str(path), "flag_count": 0, "flags": []}],
    }


def _canonical_record(idx: int = 0, **overrides: Any) -> JsonDict:
    row: JsonDict = {
        "candidate_id": f"gap4:repaired:{idx:04d}",
        "source_task_id": f"human_replay:repaired:{idx:04d}",
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_path_or_digest": "sha256:" + "a" * 64,
        "prompt_digest": "sha256:" + "b" * 64,
        "random_seed": 5224 + idx,
        "generation_started_at": "2026-07-04T00:00:00Z",
        "generation_duration_s": 61.0 + (idx / 1000.0),
        "decoding_protocol": {
            "method": "verified_repair",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 384,
        },
        "pass_at_1_fields": {
            "vote_top1": False,
            "gated_top1": False,
            "scoring_protocol": mod.SCORING_PROTOCOL,
        },
        "pass_at_2_fields": {
            "vote_top2": False,
            "gated_top2": False,
            "scoring_protocol": mod.SCORING_PROTOCOL,
        },
        "validation_inputs_digest": "sha256:" + "c" * 64,
        "provenance_kind": "canonical_pool_builder_repair",
    }
    row.update(overrides)
    return row


def test_req_report_5224_spec_declares_builder_and_terminal_fields() -> None:
    """REQ-REPORT-5224: OpenSpec names the builder, scenarios, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5224",
        "SCENARIO-REPORT-5224-REGENERATE",
        "SCENARIO-REPORT-5224-REPAIR-ONLY",
        mod.RESULT_RELATIVE_PATH,
        "gap4_canonical_pool_usable",
        "canonical_pool_n",
    ):
        assert marker in spec


def test_scenario_report_5224_regenerates_nonrepairable_audit_to_canonical_pool(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5224-REGENERATE: nonrepairable exp5223 input regenerates n>=120."""

    _install_schema(tmp_path)
    _audit(tmp_path, repairable=False)
    ticks = iter([0.0, 65.0])

    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=_model_specs,
        task_loader=lambda _root, _limit: _tasks(_limit),
        generation_certificate_func=_certificate,
        adversarial_verify_runner=_adversarial_clean,
        now=lambda: next(ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    pool = json.loads((tmp_path / artifact["canonical_pool_path"]).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["gap4_canonical_pool_usable"] is True
    assert artifact["canonical_pool_n"] == mod.CANONICAL_POOL_TARGET_N
    assert artifact["repaired_rows"] == 0
    assert artifact["regenerated_rows"] == mod.CANONICAL_POOL_TARGET_N
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["model_specs"] == _model_specs()
    assert artifact["protocol_fields_complete"] is True
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("success:")
    assert "significance" not in artifact["honest_verdict"].lower()
    assert len(pool["candidate_rows"]) == mod.CANONICAL_POOL_TARGET_N
    assert all(exp5223.canonical_candidate_record_errors(row) == [] for row in pool["candidate_rows"])
    assert mod.artifact_schema_errors(artifact, pool["candidate_rows"]) == []


def test_scenario_report_5224_repair_only_preserves_verified_rows_without_models_used(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5224-REPAIR-ONLY: repair-only path does not invent generation metadata."""

    _install_schema(tmp_path)
    _audit(tmp_path, repairable=True)
    rows = [_canonical_record(idx) for idx in range(mod.CANONICAL_POOL_TARGET_N)]
    _write_json(tmp_path / exp5223.EXP5211_RELATIVE_PATH, {"candidate_rows": rows})
    ticks = iter([4.0, 5.25])

    artifact = mod.run(
        root=tmp_path,
        cached_pair_loader=lambda: (_ for _ in ()).throw(AssertionError("no regeneration")),
        task_loader=lambda _root, _limit: (_ for _ in ()).throw(AssertionError("no tasks")),
        generation_certificate_func=lambda _prompt, _spec, _seed: pytest.fail("no generator"),
        adversarial_verify_runner=_adversarial_clean,
        now=lambda: next(ticks),
    )

    pool = json.loads((tmp_path / artifact["canonical_pool_path"]).read_text(encoding="utf-8"))
    assert artifact["gap4_canonical_pool_usable"] is True
    assert artifact["canonical_pool_n"] == mod.CANONICAL_POOL_TARGET_N
    assert artifact["repaired_rows"] == mod.CANONICAL_POOL_TARGET_N
    assert artifact["regenerated_rows"] == 0
    assert artifact["models_used"] == []
    assert artifact["model_specs"] == []
    assert artifact["exp5223_guard_passed"] is True
    assert pool["candidate_rows"] == rows
    assert mod.artifact_schema_errors(artifact, pool["candidate_rows"]) == []


def test_req_report_5224_regeneration_blocks_without_mandated_sota_model(tmp_path: Path) -> None:
    """REQ-REPORT-5224: regenerated rows require a mandated local SOTA GGUF model."""

    _install_schema(tmp_path)
    _audit(tmp_path, repairable=False)
    with pytest.raises(ValueError, match="cached_sota_pair"):
        mod.run(
            root=tmp_path,
            cached_pair_loader=lambda: None,
            task_loader=lambda _root, _limit: _tasks(_limit),
            generation_certificate_func=_certificate,
            adversarial_verify_runner=_adversarial_clean,
        )


def test_req_report_5224_artifact_schema_rejects_wrapped_or_overclaimed_fields() -> None:
    """REQ-REPORT-5224: terminal fields stay bare and cannot overclaim usability."""

    row = _canonical_record()
    artifact = {
        "experiment": mod.EXPERIMENT,
        "experiment_id": mod.EXPERIMENT_ID,
        "schema": mod.SCHEMA,
        "spec_refs": list(mod.SPEC_REFS),
        "result_path": mod.RESULT_RELATIVE_PATH,
        "gap4_canonical_pool_usable": {"value": True, "principle": "bad"},
        "canonical_pool_n": "120",
        "canonical_pool_path": mod.CANONICAL_POOL_RELATIVE_PATH,
        "repaired_rows": 0,
        "regenerated_rows": 120,
        "model_specs": [],
        "models_used": [],
        "random_seed": mod.RANDOM_SEED,
        "protocol_fields_complete": True,
        "adversarial_verify_passed": True,
        "inference_substrate": "live_llm_inference",
        "honest_verdict": "usable",
        "exp5223_guard_passed": False,
        "canonical_schema_path": exp5223.CANONICAL_SCHEMA_RELATIVE_PATH,
        "duration_s": 0.0,
        "field_principles": {},
        "reproducibility_checksum": "bad",
    }

    errors = mod.artifact_schema_errors(artifact, [row])
    for reason in (
        "gap4_canonical_pool_usable_bare_bool",
        "canonical_pool_n_bare_int",
        "field_principles",
        "inference_substrate",
        "honest_verdict_terminal_prefix",
        "usable_requires_adversarial_schema_and_protocol",
        "regenerated_rows_require_mandated_sota",
        "reproducibility_checksum",
    ):
        assert reason in errors


def test_req_report_5224_negative_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-5224: malformed repair/schema/count paths fail closed."""

    assert mod._candidate_rows(None) == []
    with pytest.raises(ValueError, match="canonical schema"):
        mod._load_canonical_schema(tmp_path)
    assert mod._dedupe_rows([{}, _canonical_record(1), _canonical_record(1)]) == [
        _canonical_record(1)
    ]
    _audit(tmp_path, repairable=True)
    _write_json(
        tmp_path / exp5223.EXP5211_RELATIVE_PATH,
        {"candidate_rows": [_canonical_record(2), {"candidate_id": "bad"}]},
    )
    assert mod.repair_rows_from_sources(tmp_path, {"gap4_pool_repairable": True}) == [
        _canonical_record(2)
    ]
    with pytest.raises(ValueError, match="not enough source tasks"):
        mod.regenerate_rows(
            tasks=[],
            deficit=1,
            model_spec=_model_specs()[0],
            generation_certificate_func=_certificate,
        )

    bad_row = _canonical_record(3)
    bad_row.pop("pass_at_2_fields")
    assert mod._schema_errors([bad_row]) == ["row_0:missing_pass_at_2_fields"]
    assert mod._verdict(False, 0).startswith("complete:")
    assert mod._adversarial_passed({"reports": [{"flag_count": 0}]}) is True
    assert mod._adversarial_passed({"reports": [{"flag_count": 1}]}) is False

    artifact = mod.build_artifact(
        rows=[_canonical_record(4)],
        repaired_rows=1,
        regenerated_rows=0,
        model_specs=[],
        model_spec_used=None,
        duration_s=1.0,
        adversarial_verify_passed=True,
    )
    malformed = dict(artifact)
    malformed["canonical_pool_n"] = 2
    malformed["repaired_rows"] = "1"
    malformed["regenerated_rows"] = 1
    malformed["protocol_fields_complete"] = False
    malformed["canonical_pool_path"] = "bad.json"
    malformed["canonical_schema_path"] = "bad.schema.json"
    malformed["reproducibility_checksum"] = "bad"

    errors = mod.artifact_schema_errors(malformed, [_canonical_record(4)])
    for reason in (
        "canonical_pool_n",
        "repaired_rows_bare_int",
        "protocol_fields_complete",
        "canonical_pool_path",
        "canonical_schema_path",
        "reproducibility_checksum",
    ):
        assert reason in errors
    bad_counts = dict(artifact)
    bad_counts["regenerated_rows"] = 1
    bad_counts["reproducibility_checksum"] = "bad"
    assert "repaired_plus_regenerated_rows" in mod.artifact_schema_errors(
        bad_counts, [_canonical_record(4)]
    )
    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, malformed)
