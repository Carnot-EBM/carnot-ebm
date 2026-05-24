"""Tests for Exp 3014 cached repair-failure taxonomy.

Spec: REQ-CODE-3014, SCENARIO-CODE-3014.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import repair_failure_taxonomy as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
REQUIRED_FIELDS = {
    "repair_failure_taxonomy_ready",
    "taxonomy_table_path",
    "n_cached_candidates_audited",
    "syntax_failure_count",
    "schema_failure_count",
    "false_accept_count",
    "tautology_failure_count",
    "intent_drift_count",
    "recommended_acceptance_rules",
    "halluguard_ntk_claim_made",
    "honest_verdict",
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_sources(tmp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [dict(row) for row in hard.default_items()[:2]]
    variants, rejected = metamorphic._generate_variants(items)
    manifest_path = tmp_path / hard.DEFAULT_MANIFEST_REL_PATH
    metamorphic_path = tmp_path / metamorphic.METAMORPHIC_MANIFEST_REL_PATH
    _write_jsonl(manifest_path, items)
    _write_jsonl(metamorphic_path, variants)
    _write_json(
        tmp_path / "results" / metamorphic.ARTIFACT_FILENAME,
        {
            "artifact": "experiment_3002_metamorphic_repair_oracle_audit_v1",
            "metamorphic_oracle_ready": True,
            "metamorphic_manifest_path": str(metamorphic.METAMORPHIC_MANIFEST_REL_PATH),
            "n_metamorphic_variants": len(variants),
            "rejected_variants": [
                *rejected,
                {
                    "item_id": "tautology-probe",
                    "relation_type": "tautology_probe",
                    "reason": "tautological_oracle_rejected",
                },
            ],
            "tautology_probe_ready": True,
            "validation_summary": {"reference_variant_failures": 0},
        },
    )
    return items, variants


def _candidate(
    tmp_path: Path,
    *,
    item: dict[str, Any],
    index: int,
    patch_code: str,
    raw_response: str,
    schema_valid: bool,
) -> dict[str, Any]:
    token = f"{item['item_id']}_{HEADLINE_MODEL.replace('/', '_')}_{index}"
    patch_path = (
        tmp_path
        / "results/raw/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1/patches"
        / f"{token}.py"
    )
    transcript_path = (
        tmp_path
        / "results/raw/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1/transcripts"
        / f"{token}.json"
    )
    source_transcript_path = (
        tmp_path
        / "results/raw/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1/transcripts"
        / f"{token}.json"
    )
    verifier_path = tmp_path / "results/verifier_transcripts/experiment_3003" / f"{token}.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    source_transcript_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_code, encoding="utf-8")
    _write_json(
        source_transcript_path,
        {
            "item_id": item["item_id"],
            "model_hf_id": HEADLINE_MODEL,
            "raw_response": raw_response,
            "prompt": f"repair {item['item_id']}",
        },
    )
    _write_json(
        transcript_path,
        {
            "item_id": item["item_id"],
            "model_hf_id": HEADLINE_MODEL,
            "source_transcript_path": str(source_transcript_path.relative_to(tmp_path)),
            "prompt": f"repair {item['item_id']}",
        },
    )
    _write_json(verifier_path, {"item_id": item["item_id"], "passed": False})
    return {
        "item_id": item["item_id"],
        "model_hf_id": HEADLINE_MODEL,
        "candidate_patch_path": str(patch_path.relative_to(tmp_path)),
        "transcript_path": str(transcript_path.relative_to(tmp_path)),
        "live_transcript_path": str(source_transcript_path.relative_to(tmp_path)),
        "verifier_log_path": str(verifier_path.relative_to(tmp_path)),
        "schema_valid": schema_valid,
    }


def _write_exp3003_candidates(tmp_path: Path, items: list[dict[str, Any]]) -> None:
    prompt_pressure = _candidate(
        tmp_path,
        item=items[0],
        index=0,
        patch_code="<|channel>thought0. analyze the repair\n",
        raw_response="<|channel>thought0. analyze the repair",
        schema_valid=False,
    )
    visible_test_overfit = _candidate(
        tmp_path,
        item=items[0],
        index=1,
        patch_code=(
            "def clamp_score(x, lo, hi):\n"
            "    if (x, lo, hi) == (12, 0, 10):\n"
            "        return 10\n"
            "    if (x, lo, hi) == (-3, 0, 10):\n"
            "        return 0\n"
            "    if (x, lo, hi) == (5, 0, 10):\n"
            "        return 5\n"
            "    return None\n"
        ),
        raw_response=json.dumps(
            {
                "draft_intent": items[0]["expected_behavior"],
                "final_patch": "def clamp_score(x, lo, hi):\n    return 5\n",
            }
        ),
        schema_valid=True,
    )
    parser_schema_mismatch = _candidate(
        tmp_path,
        item=items[0],
        index=2,
        patch_code=str(items[0]["reference_solution"]),
        raw_response=json.dumps({"final_patch": items[0]["reference_solution"]}),
        schema_valid=False,
    )
    intent_drift = _candidate(
        tmp_path,
        item=items[1],
        index=3,
        patch_code=str(items[1]["baseline_candidate"]),
        raw_response=json.dumps(
            {
                "draft_intent": "Return a sorted unique list.",
                "final_patch": items[1]["baseline_candidate"],
            }
        ),
        schema_valid=True,
    )
    _write_json(
        tmp_path / "results" / exp.EXP3003_FILENAME,
        {
            "artifact": "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1",
            "candidate_evaluations": [
                prompt_pressure,
                visible_test_overfit,
                parser_schema_mismatch,
                intent_drift,
            ],
            "candidate_patch_paths": [
                row["candidate_patch_path"]
                for row in [
                    prompt_pressure,
                    visible_test_overfit,
                    parser_schema_mismatch,
                    intent_drift,
                ]
            ],
            "verifier_log_paths": [
                row["verifier_log_path"]
                for row in [
                    prompt_pressure,
                    visible_test_overfit,
                    parser_schema_mismatch,
                    intent_drift,
                ]
            ],
        },
    )


def test_req_code_3014_spec_anchor_exists() -> None:
    """REQ-CODE-3014: the taxonomy builder is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-3014" in spec
    assert "SCENARIO-CODE-3014" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "without running live LLM inference" in spec
    assert "halluguard_ntk_claim_made=false" in spec


def test_scenario_code_3014_classifies_cached_candidate_and_validator_failures(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-3014: cached failures are replayed and classified."""

    items, _variants = _write_sources(tmp_path)
    _write_exp3003_candidates(tmp_path, items)

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 12.0,
            tests_run=("focused-exp3014",),
        )
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    table_rows = _read_jsonl(tmp_path / artifact["taxonomy_table_path"])

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_failure_taxonomy_ready"] is True
    assert artifact["n_cached_candidates_audited"] == 4
    assert artifact["syntax_failure_count"] == 1
    assert artifact["schema_failure_count"] == 2
    assert artifact["false_accept_count"] == 1
    assert artifact["tautology_failure_count"] == 1
    assert artifact["intent_drift_count"] == 1
    assert artifact["halluguard_ntk_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 2.0
    assert any("schema" in rule for rule in artifact["recommended_acceptance_rules"])
    assert any(
        row["row_type"] == "validator" and row["failure_mode"] == "tautology" for row in table_rows
    )
    assert any(row["primary_root_cause"] == "prompt-format pressure" for row in table_rows)
    assert any(row["primary_root_cause"] == "parser/schema mismatch" for row in table_rows)
    assert any(row["primary_root_cause"] == "false accept" for row in table_rows)
    assert any(row["primary_root_cause"] == "intent drift" for row in table_rows)


def test_req_code_3014_blocks_without_cached_exp3003_candidates(tmp_path: Path) -> None:
    """REQ-CODE-3014: missing cached candidates blocks honestly."""

    artifact = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path))

    assert artifact["repair_failure_taxonomy_ready"] is False
    assert artifact["n_cached_candidates_audited"] == 0
    assert artifact["taxonomy_table_path"] == ""
    assert artifact["honest_verdict"] == "blocked: exp3003 cached candidates unavailable"


def test_req_code_3014_helper_classification_branches(tmp_path: Path) -> None:
    """REQ-CODE-3014: helper branches keep root-cause labels deterministic."""

    assert exp._looks_like_prompt_format_pressure("```json\n{}\n```") is True
    assert exp._looks_like_prompt_format_pressure("def ok():\n    return 1\n") is False
    assert (
        exp._primary_candidate_root_cause(
            schema_valid=True,
            syntax_success=False,
            false_accept=False,
            intent_drift=False,
            prompt_pressure=False,
            entry_present=False,
        )
        == "invalid patch shape"
    )
    assert (
        exp._primary_candidate_root_cause(
            schema_valid=True,
            syntax_success=True,
            false_accept=False,
            intent_drift=False,
            prompt_pressure=False,
            entry_present=True,
        )
        == "passed"
    )
    assert (
        exp._validator_taxonomy_rows(
            {},
            [{"source_item_id": "repair-hard-x", "reference_verification": {"passed": False}}],
        )[0]["primary_root_cause"]
        == "oracle ambiguity"
    )
    explicit_meta = tmp_path / "explicit.jsonl"
    assert (
        exp._metamorphic_manifest_path(
            exp.ExperimentConfig(repo_root=tmp_path, metamorphic_manifest_path=explicit_meta),
            {},
        )
        == explicit_meta
    )
    assert exp._entry_point_present("def broken(:\n", "wanted") is False
    assert exp._entry_point_present("def other():\n    return 1\n", "wanted") is False
    assert exp._entry_point_present("def wanted():\n    return 1\n", "wanted") is True
    assert exp._token_overlap("", "anything") == 0.0
    assert exp._token_overlap("Clamp the score", "Clamp score into range") > 0.0
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.json").is_absolute()
