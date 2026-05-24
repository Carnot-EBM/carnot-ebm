"""Tests for Exp 3002 metamorphic hard-set repair oracle audit.

Spec: REQ-CODE-3002, SCENARIO-CODE-3002.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
REQUIRED_FIELDS = {
    "metamorphic_oracle_ready",
    "hard_set_manifest_path",
    "metamorphic_manifest_path",
    "n_source_items",
    "n_metamorphic_variants",
    "relation_types",
    "false_accept_probe_ready",
    "tautology_probe_ready",
    "rejected_variants",
    "verifier_transcript_paths",
    "honest_verdict",
}


def _write_hard_sources(tmp_path: Path, *, n_items: int = 24) -> None:
    hard.write_artifact(
        hard.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=hard.default_items()[:n_items],
            started_at=10.0,
            clock=lambda: 11.0,
            tests_run=("focused-exp2990",),
        )
    )


def _write_exp2991_cache(
    tmp_path: Path,
    *,
    item_ids: list[str] | None = None,
) -> None:
    item_ids = item_ids or [row["item_id"] for row in hard.default_items()]
    patch_dir = tmp_path / "results/raw/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1/patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    candidate_evaluations: list[dict[str, Any]] = []
    patch_paths: list[str] = []
    for item in hard.default_items():
        if item["item_id"] not in item_ids:
            continue
        patch_path = patch_dir / f"{item['item_id']}_cached.py"
        patch_path.write_text(str(item["reference_solution"]), encoding="utf-8")
        rel_path = str(patch_path.relative_to(tmp_path))
        patch_paths.append(rel_path)
        candidate_evaluations.append(
            {
                "item_id": item["item_id"],
                "candidate_patch_path": rel_path,
                "model_hf_id": "cached-exp2991",
            }
        )
    artifact = {
        "artifact": "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1",
        "candidate_evaluations": candidate_evaluations,
        "candidate_patch_paths": patch_paths,
        "selected_item_ids": item_ids,
        "source_artifacts": [{"path": str(hard.DEFAULT_MANIFEST_REL_PATH), "present": True}],
    }
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / exp.EXP2991_ARTIFACT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_code_3002_spec_anchor_exists() -> None:
    """REQ-CODE-3002: the metamorphic oracle audit is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-3002" in spec
    assert "SCENARIO-CODE-3002" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "without running live LLM repair" in spec


def test_scenario_code_3002_writes_replayable_metamorphic_oracle(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-3002: variants and probe evidence are replayable."""

    _write_hard_sources(tmp_path)
    _write_exp2991_cache(tmp_path)

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=20.0,
            clock=lambda: 23.0,
            tests_run=("focused-exp3002",),
        )
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    variants = _load_jsonl(tmp_path / artifact["metamorphic_manifest_path"])
    transcript_rows = [
        row
        for rel_path in artifact["verifier_transcript_paths"]
        for row in _load_jsonl(tmp_path / rel_path)
    ]

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["metamorphic_oracle_ready"] is True
    assert artifact["hard_set_manifest_path"] == str(hard.DEFAULT_MANIFEST_REL_PATH)
    assert artifact["n_source_items"] == 24
    assert artifact["n_metamorphic_variants"] == len(variants)
    assert artifact["n_metamorphic_variants"] > artifact["n_source_items"]
    assert set(artifact["relation_types"]) == set(exp.RELATION_TYPES)
    assert artifact["false_accept_probe_ready"] is True
    assert artifact["tautology_probe_ready"] is True
    assert artifact["live_llm_repair_run"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 3.0
    assert artifact["validation_summary"]["reference_variant_failures"] == 0
    assert artifact["validation_summary"]["cached_exp2991_candidates_seen"] == 24
    assert artifact["false_accept_probe_summary"]["metamorphic_catches_count"] > 0
    assert artifact["tautology_probe_summary"]["baseline_passes_vacuous_probe"] is True
    assert artifact["honest_verdict"].startswith("flagged:")
    assert all((tmp_path / rel_path).is_file() for rel_path in artifact["verifier_transcript_paths"])
    assert all(row["tests"] for row in variants)
    assert all(row["reference_verification"]["passed"] is True for row in variants)
    assert any(row["candidate_key"] == "cached_exp2991_repair_0" for row in transcript_rows)
    assert any(row["case_kind"] == "metamorphic_variant" for row in transcript_rows)
    assert any(row["case_kind"] == "false_accept_probe" for row in transcript_rows)


def test_req_code_3002_reconstructs_inspectable_manifest_when_source_moved(
    tmp_path: Path,
) -> None:
    """REQ-CODE-3002: moved hard-set manifests are reconstructed inspectably."""

    selected = [row["item_id"] for row in hard.default_items()[:20]]
    _write_exp2991_cache(tmp_path, item_ids=selected)

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))
    manifest_path = tmp_path / artifact["hard_set_manifest_path"]
    rows = _load_jsonl(manifest_path)

    assert artifact["metamorphic_oracle_ready"] is True
    assert artifact["hard_set_manifest_path"] == str(exp.RECONSTRUCTED_MANIFEST_REL_PATH)
    assert manifest_path.is_file()
    assert [row["item_id"] for row in rows] == selected
    assert artifact["n_source_items"] == len(selected)
    assert artifact["source_manifest_resolution"]["mode"] == "reconstructed_from_exp2991"


def test_req_code_3002_reports_rejected_semantic_and_tautology_variants(
    tmp_path: Path,
) -> None:
    """REQ-CODE-3002: invalid generated variants are rejected with reasons."""

    _write_hard_sources(tmp_path)
    _write_exp2991_cache(tmp_path)

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))
    reasons = {row["reason"] for row in artifact["rejected_variants"]}

    assert "reference_failed_semantics_changed" in reasons
    assert "tautological_oracle_rejected" in reasons
    assert any(row["relation_type"] == "input_permutation" for row in artifact["rejected_variants"])
    assert any(row["relation_type"] == "tautology_probe" for row in artifact["rejected_variants"])


def test_req_code_3002_blocks_when_no_source_items_are_available(tmp_path: Path) -> None:
    """REQ-CODE-3002: no manifest and no Exp 2991 fallback blocks honestly."""

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))

    assert artifact["metamorphic_oracle_ready"] is False
    assert artifact["n_source_items"] == 0
    assert artifact["n_metamorphic_variants"] == 0
    assert artifact["false_accept_probe_ready"] is False
    assert artifact["tautology_probe_ready"] is False
    assert artifact["honest_verdict"] == "blocked: hard-set source items unavailable"


def test_req_code_3002_helper_guards_keep_invalid_cases_outside_oracle(tmp_path: Path) -> None:
    """REQ-CODE-3002: helper guardrails reject malformed relations and syntax."""

    bad_reference = {
        **hard.default_items()[0],
        "reference_solution": hard.default_items()[0]["baseline_candidate"],
    }
    _accepted, rejected = exp._generate_variants([bad_reference])
    no_refactor = {**hard.default_items()[0], "tests": [{"test_id": "bad", "code": "x = 1"}]}

    assert rejected
    assert exp._semantic_change_reject_probe([hard.default_items()[0]]) is None
    assert exp._refactor_variant(no_refactor) is None
    assert exp._freeze_literal({"b": [2], "a": 1}) == (("a", 1), ("b", (2,)))
    assert exp._literal_call_case("f", "x = 1") is None
    assert exp._literal_call_case("f", "assert True") is None
    assert exp._literal_call_case("f", "assert f(1) > 1") is None
    assert exp._literal_call_case("f", "assert g(1) == 1") is None
    assert exp._literal_call_case("f", "assert f(x) == 1") is None
    assert exp._refactor_assert_code("x = 1") is None
    assert exp._refactor_assert_code("assert True") is None
    assert exp._adapt_candidate("def bad(:\n", "old", "new") == "def bad(:\n"
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.txt").is_absolute()
