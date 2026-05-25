"""Tests for Exp 3030 validator-frontier corrigendum.

Spec refs: REQ-REPORT-3030, SCENARIO-REPORT-3030.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import validator_frontier_corrigendum_3030 as mod


REQUIRED_FIELDS = {
    "validator_frontier_corrigendum_ready",
    "verified_region_count",
    "irrelevant_region_count",
    "unresolved_region_count",
    "fallback_only_count",
    "missing_authority_count",
    "frontier_rows",
    "cited_upstream_artifacts",
    "inference_substrate",
    "honest_verdict",
}
FORBIDDEN_TOP_LEVEL = {
    "model_specs",
    "target_model",
    "cuda",
    "gpu",
    "gpu_inventory",
    "gguf",
    "headline_models_used",
    "model_checksums",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validator_row(item_id: str = "case-1") -> dict[str, Any]:
    return {
        "item_id": item_id,
        "all_authoritative_nodes_exact_checked": True,
        "known_good_validation": {"accepted": True, "llm_judge_used": False},
        "known_bad_validation": {"accepted": False, "llm_judge_used": False},
        "validator_tree": {
            "nodes": [
                {
                    "node_id": f"{item_id}:json_required_fields:0",
                    "kind": "json_required_fields",
                    "authority": "runtime_json_parser",
                    "authoritative": True,
                    "exact_checked": True,
                },
                {
                    "node_id": f"{item_id}:semantic_boundary:1",
                    "kind": "semantic_boundary",
                    "authority": "semantic_boundary_non_authoritative",
                    "authoritative": False,
                    "exact_checked": False,
                },
            ]
        },
    }


def _certified_row(row_id: str, role: str, status: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_type": "candidate_frontier",
        "item_id": "case-1",
        "candidate_role": role,
        "certificate_status": status,
        "deterministic_validator_outcome": {
            "accepted": status == "certified_safe",
            "failing_node_ids": [] if status == "certified_safe" else ["case-1:json_required_fields:0"],
            "llm_judge_used": False,
            "rejection_reasons": [] if status == "certified_safe" else ["missing_required_field"],
        },
        "live_llm_evidence_used": False,
        "enumerator_fallback_used": False,
        "probability_bound_placeholder": {
            "exact_probability_computed": False,
            "bound_type": "placeholder",
            "reason": "no token-trie or model-probability frontier was computed",
        },
        "transcript_path": "results/beaver_style_validator_frontier_certificate_3018/transcripts/case-1.json",
        "transcript_sha256": "abc",
    }


def _write_sources(root: Path) -> None:
    validator_manifest = [_validator_row()]
    certificate_rows = [
        _certified_row("case-1:known_good", "known_good", "certified_safe"),
        _certified_row("case-1:known_bad", "known_bad", "certified_violating"),
        {
            "row_id": "case-1:non_prefix:case-1:semantic_boundary:1",
            "row_type": "non_prefix_closed_node",
            "item_id": "case-1",
            "node_id": "case-1:semantic_boundary:1",
            "certificate_status": "non_prefix_closed",
            "frontier_exploration": {"bounded": False, "mode": "not_applicable"},
            "live_llm_evidence_used": False,
            "enumerator_fallback_used": False,
        },
        {
            "row_id": "rejected-random:unresolved",
            "row_type": "source_rejected_item",
            "item_id": "rejected-random",
            "certificate_status": "unresolved",
            "source_rejection_reason": "nondeterministic_validator",
            "frontier_exploration": {"bounded": False, "mode": "not_available"},
            "live_llm_evidence_used": False,
            "enumerator_fallback_used": False,
        },
        {
            "row_id": "rejected-llm:unresolved",
            "row_type": "source_rejected_item",
            "item_id": "rejected-llm",
            "certificate_status": "unresolved",
            "source_rejection_reason": "llm_only_label",
            "frontier_exploration": {"bounded": False, "mode": "not_available"},
            "live_llm_evidence_used": False,
            "enumerator_fallback_used": False,
        },
    ]
    _write_json(root, mod.EXP3017_REL_PATH, _exp3017_artifact())
    _write_jsonl(root, mod.EXP3017_MANIFEST_REL_PATH, validator_manifest)
    _write_json(root, mod.EXP3018_REL_PATH, _exp3018_artifact())
    _write_jsonl(root, mod.EXP3018_MANIFEST_REL_PATH, certificate_rows)
    _write_json(root, mod.EXP3027_REL_PATH, _exp3027_artifact())


def _exp3017_artifact() -> dict[str, Any]:
    return {
        "artifact": "experiment_3017_nsvif_instruction_validator_tree_expansion_v1",
        "instruction_validator_tree_ready": True,
        "validator_manifest_path": mod.EXP3017_MANIFEST_REL_PATH.as_posix(),
        "n_instruction_items": 1,
        "n_validator_trees": 1,
        "all_authoritative_nodes_exact_checked": True,
        "llm_judge_used": False,
        "honest_verdict": "complete: exact validator-tree fixture",
    }


def _exp3018_artifact() -> dict[str, Any]:
    return {
        "artifact": "experiment_3018_beaver_style_validator_frontier_certificate_v1",
        "frontier_certificate_ready": True,
        "certificate_manifest_path": mod.EXP3018_MANIFEST_REL_PATH.as_posix(),
        "n_frontier_items": 5,
        "certified_safe_count": 1,
        "certified_violating_count": 1,
        "non_prefix_closed_count": 1,
        "unresolved_count": 2,
        "live_llm_evidence_used": False,
        "enumerator_fallback_separated": True,
        "probability_bound_policy": {
            "exact_probability_computed": False,
            "bound_type": "placeholder",
        },
        "enumerator_fallback_provenance": {
            "present": True,
            "paths": ["results/raw/exp3004/fallback.json"],
            "separated_from_live": True,
        },
        "honest_verdict": "complete: validator frontier certificate ready with explicit unresolved bounds",
    }


def _exp3027_artifact() -> dict[str, Any]:
    return {
        "artifact": "experiment_3027_adversarial_flag_methodology_corrigendum",
        "methodology_corrigendum_ready": True,
        "unresolved_bound_rows": [
            {
                "row_id": "exp3018_beaver_frontier_certificate",
                "classification": "unresolved_bound",
                "source_artifact_path": mod.EXP3018_REL_PATH.as_posix(),
                "supporting_fields": [{"field": "unresolved_count", "value": 2}],
            }
        ],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": "complete: methodology_corrigendum_ready=true",
    }


def _row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["row_id"] == row_id)


def test_req_report_3030_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3030: OpenSpec declares the validator-frontier corrigendum."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    script = Path("scripts/experiment_3030_validator_frontier_corrigendum_v2.py")

    assert "REQ-REPORT-3030" in spec
    assert "SCENARIO-REPORT-3030" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert script.is_file()


def test_scenario_report_3030_separates_frontier_regions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3030: every frontier row receives inspectable accounting."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["validator_frontier_corrigendum_ready"] is True
    assert artifact["verified_region_count"] == 2
    assert artifact["irrelevant_region_count"] == 1
    assert artifact["unresolved_region_count"] == 2
    assert artifact["fallback_only_count"] == 1
    assert artifact["missing_authority_count"] == 0
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == {
        "kind": mod.INFERENCE_SUBSTRATE_KIND,
        "no_live_llm_inference": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_top_level_live_model_metadata": True,
        "source_metadata_location": "cited_upstream_artifacts[].source_field_summary",
    }
    assert not FORBIDDEN_TOP_LEVEL.intersection(artifact)

    rows = artifact["frontier_rows"]
    assert len(rows) == 6
    for row in rows:
        assert {
            "row_id",
            "source_artifact_path",
            "authority_type",
            "classification",
            "bound_status",
            "allowed_claim_wording",
        } <= row.keys()

    verified = _row_by_id(rows, "case-1:known_good")
    assert verified["classification"] == "verified"
    assert verified["exact_authorities"] == ["runtime_json_parser"]
    assert "Do not claim full BEAVER" in verified["allowed_claim_wording"]

    irrelevant = _row_by_id(rows, "case-1:non_prefix:case-1:semantic_boundary:1")
    assert irrelevant["classification"] == "irrelevant"
    assert irrelevant["bound_status"] == "clipped_irrelevant_to_exact_authority"

    unresolved = _row_by_id(rows, "rejected-llm:unresolved")
    assert unresolved["classification"] == "unresolved"
    assert unresolved["authority_type"] == "live_llm_dependency"
    assert "Do not promote" in unresolved["allowed_claim_wording"]

    fallback = _row_by_id(rows, "exp3004_enumerator_fallback:0")
    assert fallback["classification"] == "fallback_only"
    assert fallback["source_artifact_path"] == "results/raw/exp3004/fallback.json"
    assert "cannot be promoted" in fallback["allowed_claim_wording"]

    citations = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(citations) == {"exp3017", "exp3017_manifest", "exp3018", "exp3018_manifest", "exp3027"}
    assert citations["exp3018"]["sha256"] == _sha256(tmp_path / mod.EXP3018_REL_PATH)
    assert artifact["status_updates_written"] is False


def test_req_report_3030_missing_authority_and_fallback_rows_do_not_promote() -> None:
    """REQ-REPORT-3030: live or fallback evidence blocks exact authority wording."""

    validator_rows = {"case-1": _validator_row()}
    live_row = _certified_row("case-live:known_good", "known_good", "certified_safe")
    live_row["live_llm_evidence_used"] = True
    fallback_row = _certified_row("case-fallback:known_good", "known_good", "certified_safe")
    fallback_row["enumerator_fallback_used"] = True
    missing_row = _certified_row("case-missing:known_good", "known_good", "certified_safe")
    missing_row.pop("deterministic_validator_outcome")

    assert mod.classify_certificate_row(live_row, validator_rows)["classification"] == "missing_authority"
    assert mod.classify_certificate_row(fallback_row, validator_rows)["classification"] == "fallback_only"
    missing = mod.classify_certificate_row(missing_row, validator_rows)
    assert missing["classification"] == "missing_authority"
    assert missing["bound_status"] == "missing_exact_authority_or_provenance"


def test_req_report_3030_blocks_when_required_source_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3030: missing required artifacts fail closed."""

    _write_json(tmp_path, mod.EXP3017_REL_PATH, _exp3017_artifact())

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["validator_frontier_corrigendum_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["frontier_rows"] == []
    assert artifact["required_source_errors"] == [
        {
            "experiment_id": "exp3017_manifest",
            "path": mod.EXP3017_MANIFEST_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3018",
            "path": mod.EXP3018_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3018_manifest",
            "path": mod.EXP3018_MANIFEST_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3027",
            "path": mod.EXP3027_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
    ]


def test_req_report_3030_write_main_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3030: persistence and malformed inputs stay deterministic."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["validator_frontier_corrigendum_ready"] is True
    assert mod.main(tmp_path) == 0

    malformed = tmp_path / "bad.json"
    list_payload = tmp_path / "list.json"
    bad_jsonl = tmp_path / "bad.jsonl"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1]\n", encoding="utf-8")
    bad_jsonl.write_text("{bad-json}\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.read_jsonl_objects(bad_jsonl) == []
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._mapping("bad") == {}
    assert mod._mapping_list([{"a": 1}, "bad"]) == [{"a": 1}]
    assert mod._string_list(["a", 1, ""]) == ["a", "1"]
    assert mod._string_list("bad") == []
    assert mod._counts_by_class([]) == {
        "verified": 0,
        "irrelevant": 0,
        "unresolved": 0,
        "fallback_only": 0,
        "missing_authority": 0,
    }
    assert mod._honest_verdict(False, mod._counts_by_class([])) == "blocked_required_upstream_missing"
    assert mod._path_for_row(tmp_path, str((tmp_path / "inside.json").resolve())) == "inside.json"
    assert mod._path_for_row(tmp_path, "/outside/root.json") == "/outside/root.json"

    source_rejected = {
        "row_id": "rejected-ambiguous:non_prefix_closed",
        "row_type": "source_rejected_item",
        "item_id": "rejected-ambiguous",
        "certificate_status": "non_prefix_closed",
    }
    assert (
        mod.classify_certificate_row(source_rejected, {})["authority_type"]
        == "ambiguous_instruction_no_deterministic_boundary"
    )
    unresolved_unknown = {
        "row_id": "rejected-other:unresolved",
        "row_type": "source_rejected_item",
        "item_id": "rejected-other",
        "certificate_status": "unresolved",
        "source_rejection_reason": "other",
    }
    assert (
        mod.classify_certificate_row(unresolved_unknown, {})["authority_type"]
        == "unresolved_validator_authority"
    )
    unknown_status = {
        "row_id": "case-unknown",
        "row_type": "candidate_frontier",
        "item_id": "case-1",
        "certificate_status": "unexpected",
    }
    assert mod.classify_certificate_row(unknown_status, {})["classification"] == "missing_authority"
