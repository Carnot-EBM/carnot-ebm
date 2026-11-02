"""Build the V633 symbolic-edit fixture from source-backed Exp7158 rows.

The fixture changes entity spellings without changing meaning, then changes
relations or removes evidence. A private exact interpreter creates labels. An
independent SQLite query checks those labels. No model is loaded or scored.

Spec refs: REQ-VERIFY-7180 and SCENARIO-VERIFY-7180-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_bytes, atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260910"
RANDOM_SEED = 7_180_202_609_10
LABEL_PERMUTATION_SEED = 718_003
SHUFFLE_SEEDS = (718_001, 718_002)
BASE_COUNT = 48
VARIANTS = (
    "original",
    "bijective_entity_rename",
    "relation_or_polarity_flip",
    "evidence_deletion",
)
RELATION_FAMILIES = (
    "precedes",
    "follows",
    "starts before",
    "ends before",
    "is separated from",
    "occurs before",
)
CALIBRATION_FAMILIES = RELATION_FAMILIES[:2]
EVALUATION_FAMILIES = RELATION_FAMILIES[2:]
QUANTIFIED_FAMILIES = frozenset({"precedes", "follows", "is separated from"})
FALSE_RELATIONS = {
    "precedes": "follows",
    "follows": "precedes",
    "starts before": "starts after",
    "ends before": "ends after",
    "is separated from": "equals",
    "occurs before": "occurs after",
}
COMPARISON_ARMS = (
    "baseline_direct",
    "energy_from_extracted_tuples",
    "lexical_overlap",
    "syntax_only",
    "shuffled_evidence",
)
ENERGY_TERMS = (
    "tuple_alignment",
    "polarity",
    "quantity_unit_agreement",
    "literal_span_validity",
    "missing_required_fields",
)
ENERGY_WEIGHTS = {
    "tuple_alignment": 3,
    "polarity": 2,
    "quantity_unit_agreement": 2,
    "literal_span_validity": 2,
    "missing_required_fields": 1,
}
RESULT_PATH = Path("results/experiment_7180_v633_symbolic_edit_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7180_v633_symbolic_edit_fixture.json")
GENERATION_VIEW_PATH = Path(
    "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
)
AUTHORITY_SIDECAR_PATH = Path("results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl")
INFERENCE_SUBSTRATE = "exact_source_fixture_construction"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
STUDY_QUESTION = (
    "Does a source-grounded symbolic fixture preserve decisions under bijective "
    "entity renames and change them under semantic edits or evidence deletion?"
)
SCOPE_ANSWER = (
    "The exact fixture labels preserve all 48 rename decisions and change all "
    "96 semantic or deletion decisions; no live model or verifier was measured."
)

SOURCE_PATHS = {
    "exp7158_artifact": Path("results/experiment_7158_v630_entity_evidence_fixture.json"),
    "exp7158_module": Path("python/carnot/experiment_7158_v630_entity_evidence_fixture.py"),
    "exp7138_module": Path("python/carnot/experiment_7138_v627_relational_fixture.py"),
    "constraint_spec": Path("openspec/capabilities/constraint-verification/spec.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "symbolic_module": Path("python/carnot/experiment_7180_v633_symbolic_edit_fixture.py"),
    "entrypoint": Path("scripts/experiments/experiment_7180_v633_symbolic_edit_fixture.py"),
    "focused_tests": Path("tests/python/test_experiment_7180_v633_symbolic_edit_fixture.py"),
}
PINNED_HASHES = {
    "exp7158_artifact": "sha256:957021034863e359c515c69d47fe44781841a7766e0442ad9a55255cdb1c1635",
}
EXP7158_EXPECTED_FIELDS = {
    "status": "complete",
    "run_date": "20260909",
    "inference_substrate": "exact_source_fixture_construction",
    "inference_substrate_class": "cpu_exact_solver_or_simulator",
    "execution_venue": "host",
    "counterfactual_fixture_ready_score": 1,
    "random_seed": 715820260909,
    "verifier_is_oracle": False,
    "verdict_class": "positive",
    "honest_verdict": "complete_positive_counterfactual_fixture_ready_no_verifier_value_claim",
    "reproducibility_checksum": "sha256:4180a79d6226095a49ee77e9484421c30ae2fb0caf78f7ef6e179df6afa625ba",
}

RESPONSE_FIELDS = (
    "direct_decision",
    "claim_tuple",
    "evidence_tuple",
    "source_start",
    "source_end",
    "missing_fields",
)
RESPONSE_SCHEMA: JsonDict = {
    "direct_decision": "supported | unsupported | abstain",
    "claim_tuple": {
        "subject": "string",
        "relation": "string",
        "object": "string",
        "polarity": "positive | negative",
        "quantity": "number or null when not applicable",
        "unit": "string or null when not applicable",
    },
    "evidence_tuple": "same tuple shape, or null when no evidence exists",
    "source_start": "zero-based integer or null",
    "source_end": "exclusive integer or null",
    "missing_fields": "list of unavailable response fields",
}
FORBIDDEN_GENERATION_KEYS = frozenset(
    {
        "split",
        "variant",
        "edit_label",
        "support_label",
        "expected_answer",
        "expected_response",
        "authority",
        "support_hash",
        "source_artifact_hashes",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "fixture_ready_score",
    "generation_view_path",
    "authority_sidecar_path",
    "split_manifest",
    "score_contract",
    "mutation_rows",
    "sidecar_hashes",
    "structural_checks",
    "study_question",
    "scope_answer",
)
FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use cpu_exact_solver_or_simulator when the declared work runs; use blocked_no_run only before any qualifying work.",
    "fixture_ready_score": "One requires all 192 rows, isolated labels, and exact authority agreement.",
    "generation_view_path": "This is the only fixture input a generation worker may read.",
    "authority_sidecar_path": "Separate authority prevents truth leakage into prompts.",
    "split_manifest": "Whole-family isolation prevents variant leakage.",
    "score_contract": "The frozen energy and controls prevent post-result tuning.",
    "mutation_rows": "Renames and semantic edits test the fixture itself.",
    "sidecar_hashes": "Exact sidecar byte hashes bind private labels to public generation units.",
    "structural_checks": "Explicit receipts make each readiness condition independently auditable.",
    "study_question": "A fixed question prevents the fixture from expanding into a live verifier claim.",
    "scope_answer": "The answer states fixture evidence and preserves the unmeasured verifier result.",
}


def canonical_json(value: Any) -> str:
    """Return one compact JSON spelling so hashes are reproducible."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize rows with stable key order and one final newline."""

    return ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes and retain the algorithm name in the receipt."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text without normalizing source bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash exact file bytes so any source or contract change is visible."""

    return sha256_bytes(path.read_bytes())


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding measured duration and this hash."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_text(canonical_json(payload))


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Print one phase boundary immediately so long task silence is impossible."""

    print(f"exp7180 phase={phase} event={event} detail={detail}", flush=True)


def _stable_id(prefix: str, *parts: str) -> str:
    """Make an opaque identifier from frozen source-only inputs."""

    return f"{prefix}-{sha256_text('|'.join(parts)).split(':', 1)[1][:24]}"


def _tuple_for_family(source: Mapping[str, Any], family: str) -> JsonDict:
    """Project one Exp7158 source fact into a controlled relation family."""

    entities = list(source["claim_entities"])
    subject = str(entities[0]["canonical_name"])
    obj = str(entities[1]["canonical_name"])
    if family == "follows":
        subject, obj = obj, subject
    value: JsonDict = {
        "subject": subject,
        "relation": family,
        "object": obj,
        "polarity": "positive",
    }
    if family in QUANTIFIED_FAMILIES:
        value["quantity"] = source["claim_fact"]["quantity"]
        value["unit"] = source["claim_fact"]["unit"]
    return value


def _render_tuple(value: Mapping[str, Any]) -> str:
    """Render a tuple as short text while keeping every literal recoverable."""

    negation = "does not " if value["polarity"] == "negative" else ""
    text = f"{value['subject']} {negation}{value['relation']} {value['object']}"
    if "quantity" in value:
        text += f" by {value['quantity']} {value['unit']}"
    return text + "."


def _renamed_tuple(value: Mapping[str, Any], rename_map: Mapping[str, str]) -> JsonDict:
    """Apply one bijection only to entity surface forms."""

    renamed = deepcopy(dict(value))
    renamed["subject"] = rename_map[str(value["subject"])]
    renamed["object"] = rename_map[str(value["object"])]
    return renamed


def _response_for_parts(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    source_text: str,
    decision: str,
) -> JsonDict:
    """Build the compact authority response after a decision is known."""

    missing = [] if evidence is not None else ["evidence_tuple", "source_start", "source_end"]
    return {
        "direct_decision": decision,
        "claim_tuple": deepcopy(dict(claim)),
        "evidence_tuple": deepcopy(dict(evidence)) if evidence is not None else None,
        "source_start": 0 if evidence is not None else None,
        "source_end": len(source_text) if evidence is not None else None,
        "missing_fields": missing,
    }


def authority_interpret(row: Mapping[str, Any]) -> JsonDict:
    """Label one private row with a bounded symbolic equality interpreter."""

    claim = dict(row["claim_tuple"])
    evidence_value = row["evidence_tuple"]
    evidence = dict(evidence_value) if isinstance(evidence_value, Mapping) else None
    decision = "supported" if evidence is not None and claim == evidence else "unsupported"
    return _response_for_parts(claim, evidence, str(row["source_text"]), decision)


def sqlite_cross_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Label rows with a separate SQL equality query over private facts."""

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE evidence (unit_id TEXT PRIMARY KEY, subject TEXT, relation_name TEXT, "
        "object_name TEXT, polarity TEXT, quantity REAL, unit TEXT)"
    )
    for row in rows:
        evidence = row["evidence_tuple"]
        if not isinstance(evidence, Mapping):
            continue
        connection.execute(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["unit_id"],
                evidence["subject"],
                evidence["relation"],
                evidence["object"],
                evidence["polarity"],
                evidence.get("quantity"),
                evidence.get("unit"),
            ),
        )
    connection.commit()
    results: dict[str, JsonDict] = {}
    query = (
        "SELECT COUNT(*) FROM evidence WHERE unit_id = ? AND subject = ? "
        "AND relation_name = ? AND object_name = ? AND polarity = ? "
        "AND quantity IS ? AND unit IS ?"
    )
    for row in rows:
        claim = dict(row["claim_tuple"])
        count = connection.execute(
            query,
            (
                row["unit_id"],
                claim["subject"],
                claim["relation"],
                claim["object"],
                claim["polarity"],
                claim.get("quantity"),
                claim.get("unit"),
            ),
        ).fetchone()[0]
        evidence_value = row["evidence_tuple"]
        evidence = dict(evidence_value) if isinstance(evidence_value, Mapping) else None
        decision = "supported" if count == 1 else "unsupported"
        missing = [] if evidence is not None else ["evidence_tuple", "source_start", "source_end"]
        results[str(row["unit_id"])] = {
            "direct_decision": decision,
            "claim_tuple": deepcopy(claim),
            "evidence_tuple": deepcopy(evidence),
            "source_start": 0 if evidence is not None else None,
            "source_end": len(str(row["source_text"])) if evidence is not None else None,
            "missing_fields": missing,
        }
    connection.close()
    return results


def _generation_row(row: Mapping[str, Any]) -> JsonDict:
    """Project the only three fields allowed to reach a generation worker."""

    text = (
        f"Evidence:\n{row['source_text']}\n\nClaim:\n{row['claim_text']}\n\n"
        "Return one JSON object that matches the supplied response schema."
    )
    return {
        "unit_id": row["unit_id"],
        "text": text,
        "response_schema": deepcopy(RESPONSE_SCHEMA),
    }


def generation_view_errors(
    generation_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject shape, identity, schema, or private-token exposure in model inputs."""

    errors: list[str] = []
    expected_keys = {"unit_id", "text", "response_schema"}
    generation_ids: list[str] = []
    for index, row in enumerate(generation_rows):
        if set(row) != expected_keys:
            errors.append(f"generation_view_shape:{index}")
            continue
        unit_id = str(row["unit_id"])
        generation_ids.append(unit_id)
        if re.fullmatch(r"u-[0-9a-f]{24}", unit_id) is None:
            errors.append(f"generation_unit_id_not_opaque:{index}")
        if row["response_schema"] != RESPONSE_SCHEMA:
            errors.append(f"generation_response_schema:{unit_id}")
        text = str(row["text"])
        if any(key in text for key in FORBIDDEN_GENERATION_KEYS):
            errors.append(f"generation_private_token:{unit_id}")
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(generation_ids) != len(set(generation_ids)):
        errors.append("generation_unit_id_duplicate")
    if set(generation_ids) != set(authority_ids):
        errors.append("generation_authority_roster_mismatch")
    return errors


def _valid_tuple(value: Any) -> bool:
    """Check the compact tuple union without filling omitted values."""

    if not isinstance(value, Mapping):
        return False
    required = {"subject", "relation", "object", "polarity"}
    if not required <= set(value) or value["polarity"] not in {"positive", "negative"}:
        return False
    extra = set(value) - required
    return extra in (set(), {"quantity", "unit"})


def compute_candidate_energy(response: Mapping[str, Any], source_text: str) -> dict[str, int]:
    """Score only generated response fields and the supplied evidence bytes."""

    missing = int(any(field not in response for field in RESPONSE_FIELDS))
    claim = response.get("claim_tuple")
    evidence = response.get("evidence_tuple")
    claim_valid = _valid_tuple(claim)
    evidence_valid = evidence is None or _valid_tuple(evidence)
    if not claim_valid or not evidence_valid:
        missing = 1

    alignment = 0
    polarity = 0
    quantity_unit = 0
    if claim_valid and evidence is None:
        alignment = 1
    elif claim_valid and _valid_tuple(evidence):
        alignment = int(
            any(claim[field] != evidence[field] for field in ("subject", "relation", "object"))
        )
        polarity = int(claim["polarity"] != evidence["polarity"])
        quantity_unit = int(
            claim.get("quantity") != evidence.get("quantity")
            or claim.get("unit") != evidence.get("unit")
        )

    start = response.get("source_start")
    end = response.get("source_end")
    literal_span = 0
    if evidence is None:
        literal_span = int(start is not None or end is not None or bool(source_text))
    elif _valid_tuple(evidence):
        valid_bounds = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start == 0
            and end == len(source_text)
        )
        if valid_bounds:
            exact = source_text[start:end]
            literals = [str(evidence[field]) for field in ("subject", "relation", "object")]
            if "quantity" in evidence:
                literals.extend([str(evidence["quantity"]), str(evidence["unit"])])
            literal_span = int(any(literal not in exact for literal in literals))
        else:
            literal_span = 1

    terms = {
        "tuple_alignment": alignment,
        "polarity": polarity,
        "quantity_unit_agreement": quantity_unit,
        "literal_span_validity": literal_span,
        "missing_required_fields": missing,
    }
    terms["total"] = sum(ENERGY_WEIGHTS[name] * terms[name] for name in ENERGY_TERMS)
    return terms


def _permutation(unit_ids: Sequence[str], seed: int) -> list[JsonDict]:
    """Create a deterministic donor schedule with no unit mapped to itself."""

    donors = list(unit_ids)
    random.Random(seed).shuffle(donors)
    while any(unit_id == donor for unit_id, donor in zip(unit_ids, donors, strict=True)):
        donors = donors[1:] + donors[:1]
    return [
        {"unit_id": unit_id, "donor_unit_id": donor}
        for unit_id, donor in zip(unit_ids, donors, strict=True)
    ]


def freeze_score_contract(
    internal_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze scoring and controls from calibration identities before inference."""

    unit_ids = [str(row["unit_id"]) for row in internal_rows]
    calibration_ids = [
        str(row["unit_id"]) for row in internal_rows if row["split"] == "calibration"
    ]
    authority = {str(row["unit_id"]): row for row in authority_rows}
    calibration_truth = [
        {
            "unit_id": unit_id,
            "expected_response": authority[unit_id]["expected_response"],
        }
        for unit_id in calibration_ids
    ]
    return {
        "paper": "arXiv:2608.30256",
        "frozen_before_inference": True,
        "fit_split": "calibration",
        "fit_relation_families": list(CALIBRATION_FAMILIES),
        "fit_unit_ids": calibration_ids,
        "calibration_authority_sha256": sha256_text(canonical_json(calibration_truth)),
        "evaluation_truth_accessed": False,
        "energy_terms": list(ENERGY_TERMS),
        "energy_weights": deepcopy(ENERGY_WEIGHTS),
        "threshold": 0,
        "tie_rule": "energy_equal_threshold_is_supported",
        "decision_rule": "energy_above_threshold_is_unsupported",
        "candidate_scorer_inputs": ["generated_response", "supplied_source_bytes"],
        "candidate_scorer_calls_authority": False,
        "comparison_arms": {
            "baseline_direct": "use only direct_decision",
            "energy_from_extracted_tuples": "apply the frozen candidate energy",
            "lexical_overlap": "compare claim and evidence token sets",
            "syntax_only": "accept only schema-valid responses",
            "shuffled_evidence": "score against the seeded donor evidence",
        },
        "oracle_upper_bound": {
            "name": "exact_correct_tuples",
            "deployable_extraction_arm": False,
            "purpose": "upper bound only",
        },
        "seeded_shuffle_controls": [
            {"seed": seed, "mapping": _permutation(unit_ids, seed)} for seed in SHUFFLE_SEEDS
        ],
        "label_permutation_control": {
            "seed": LABEL_PERMUTATION_SEED,
            "mapping": _permutation(unit_ids, LABEL_PERMUTATION_SEED),
        },
        "scoring_denominators": {
            "all_model_visible_rows": len(unit_ids),
            "calibration_rows": len(calibration_ids),
            "evaluation_rows_per_arm": len(unit_ids) - len(calibration_ids),
        },
    }


def _split_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record whole-family routing and base-level variant isolation."""

    family_rows = []
    for family in RELATION_FAMILIES:
        selected = [row for row in rows if row["relation_family"] == family]
        family_rows.append(
            {
                "relation_family": family,
                "split": selected[0]["split"],
                "base_count": len({row["base_id"] for row in selected}),
                "row_count": len(selected),
            }
        )
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    return {
        "calibration_families": list(CALIBRATION_FAMILIES),
        "evaluation_families": list(EVALUATION_FAMILIES),
        "family_rows": family_rows,
        "calibration_base_count": len({row["base_id"] for row in calibration}),
        "evaluation_base_count": len({row["base_id"] for row in evaluation}),
        "calibration_row_count": len(calibration),
        "evaluation_row_count": len(evaluation),
        "whole_family_isolation": True,
        "base_variant_isolation": True,
    }


def _materialize_internal_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Select 48 distinct Exp7158 bases and construct four variants each."""

    supported = [row for row in upstream.get("rows", []) if row.get("condition") == "supported"]
    if len(supported) != 72:
        raise ValueError(f"expected 72 supported Exp7158 rows, observed {len(supported)}")
    supported.sort(key=lambda row: sha256_text(f"{RANDOM_SEED}|{row['base_id']}"))
    selected = supported[:BASE_COUNT]
    rows: list[JsonDict] = []
    for base_index, source in enumerate(selected):
        family = RELATION_FAMILIES[base_index // 8]
        split = "calibration" if family in CALIBRATION_FAMILIES else "evaluation"
        base_id = _stable_id("b", str(source["base_id"]), family)
        evidence = _tuple_for_family(source, family)
        original_source = _render_tuple(evidence)
        original_claim = deepcopy(evidence)
        rename_map = {
            str(evidence["subject"]): _stable_id("entity", base_id, "subject"),
            str(evidence["object"]): _stable_id("entity", base_id, "object"),
        }
        for variant_index, variant in enumerate(VARIANTS):
            claim = deepcopy(original_claim)
            evidence_tuple: JsonDict | None = deepcopy(evidence)
            source_text = original_source
            active_rename: JsonDict = {}
            edit_detail = "none"
            if variant == "bijective_entity_rename":
                claim = _renamed_tuple(claim, rename_map)
                evidence_tuple = _renamed_tuple(evidence, rename_map)
                source_text = _render_tuple(evidence_tuple)
                active_rename = deepcopy(rename_map)
                edit_detail = "surface_bijection"
            elif variant == "relation_or_polarity_flip":
                if base_index % 2 == 0:
                    claim["relation"] = FALSE_RELATIONS[family]
                    edit_detail = "relation_flip"
                else:
                    claim["polarity"] = "negative"
                    edit_detail = "polarity_flip"
            elif variant == "evidence_deletion":
                evidence_tuple = None
                source_text = ""
                edit_detail = "evidence_removed"
            unit_id = _stable_id("u", base_id, str(variant_index))
            rows.append(
                {
                    "unit_id": unit_id,
                    "base_id": base_id,
                    "source_fixture_id": source["source_fixture_id"],
                    "source_artifact_row_sha256": sha256_text(canonical_json(source)),
                    "relation_family": family,
                    "split": split,
                    "variant": variant,
                    "edit_detail": edit_detail,
                    "rename_map": active_rename,
                    "source_text": source_text,
                    "claim_text": _render_tuple(claim),
                    "claim_tuple": claim,
                    "evidence_tuple": evidence_tuple,
                }
            )
    return rows


def materialize_fixture(upstream: Mapping[str, Any]) -> JsonDict:
    """Build public rows, private labels, splits, controls, and mutation receipts."""

    internal_rows = _materialize_internal_rows(upstream)
    sqlite_labels = sqlite_cross_check(internal_rows)
    generation_rows = [_generation_row(row) for row in internal_rows]
    authority_rows: list[JsonDict] = []
    for row in internal_rows:
        symbolic = authority_interpret(row)
        sql_response = sqlite_labels[str(row["unit_id"])]
        authority_rows.append(
            {
                "unit_id": row["unit_id"],
                "base_id": row["base_id"],
                "relation_family": row["relation_family"],
                "split": row["split"],
                "variant": row["variant"],
                "edit_detail": row["edit_detail"],
                "expected_response": symbolic,
                "symbolic_response_sha256": sha256_text(canonical_json(symbolic)),
                "sqlite_response_sha256": sha256_text(canonical_json(sql_response)),
                "authority_agreement": symbolic == sql_response,
            }
        )
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    generation_by_id = {str(row["unit_id"]): row for row in generation_rows}
    receipt_rows = [
        {
            "unit_id": row["unit_id"],
            "base_id": row["base_id"],
            "condition": row["variant"],
            "row_status": "fixture_ready",
            "error": None,
            "abstention": False,
            "generation_row_sha256": sha256_text(
                canonical_json(generation_by_id[str(row["unit_id"])])
            ),
            "authority_row_sha256": sha256_text(
                canonical_json(authority_by_id[str(row["unit_id"])])
            ),
        }
        for row in internal_rows
    ]
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in authority_rows:
        grouped[str(row["base_id"])][str(row["variant"])] = row
    mutation_rows: list[JsonDict] = []
    for base_id in sorted(grouped):
        variants = grouped[base_id]
        original = variants["original"]
        original_decision = original["expected_response"]["direct_decision"]
        for variant in VARIANTS[1:]:
            current = variants[variant]
            decision = current["expected_response"]["direct_decision"]
            changed = decision != original_decision
            expected_changed = variant != "bijective_entity_rename"
            mutation_rows.append(
                {
                    "base_id": base_id,
                    "original_unit_id": original["unit_id"],
                    "mutated_unit_id": current["unit_id"],
                    "variant": variant,
                    "original_decision": original_decision,
                    "mutated_decision": decision,
                    "decision_changed": changed,
                    "expected_decision_changed": expected_changed,
                    "symbolic_sql_agreement": current["authority_agreement"],
                    "passed": changed == expected_changed and current["authority_agreement"],
                }
            )
    split_manifest = _split_manifest(internal_rows)
    score_contract = freeze_score_contract(internal_rows, authority_rows)
    return {
        "internal_rows": internal_rows,
        "generation_rows": generation_rows,
        "authority_rows": authority_rows,
        "rows": receipt_rows,
        "split_manifest": split_manifest,
        "score_contract": score_contract,
        "mutation_rows": mutation_rows,
    }


def structural_errors(materialized: Mapping[str, Any]) -> list[str]:
    """Recompute every readiness rule without trusting summary counts."""

    internal = list(materialized["internal_rows"])
    generation = list(materialized["generation_rows"])
    authority = list(materialized["authority_rows"])
    errors: list[str] = []
    if len(internal) != BASE_COUNT * len(VARIANTS):
        errors.append("model_visible_row_count")
    if len({row["base_id"] for row in internal}) != BASE_COUNT:
        errors.append("base_count")
    if Counter(row["relation_family"] for row in internal) != Counter(
        {family: 32 for family in RELATION_FAMILIES}
    ):
        errors.append("relation_family_counts")
    if Counter(row["variant"] for row in internal) != Counter(
        {variant: BASE_COUNT for variant in VARIANTS}
    ):
        errors.append("variant_counts")
    family_splits: dict[str, set[str]] = defaultdict(set)
    base_splits: dict[str, set[str]] = defaultdict(set)
    for row in internal:
        family_splits[str(row["relation_family"])].add(str(row["split"]))
        base_splits[str(row["base_id"])].add(str(row["split"]))
    if any(len(values) != 1 for values in family_splits.values()) or any(
        len(values) != 1 for values in base_splits.values()
    ):
        errors.append("split_leakage")
    errors.extend(generation_view_errors(generation, authority))
    if len(authority) != len(internal) or not all(row["authority_agreement"] for row in authority):
        errors.append("authority_agreement")
    mutations = list(materialized["mutation_rows"])
    if len(mutations) != BASE_COUNT * 3 or not all(row["passed"] for row in mutations):
        errors.append("mutation_receipts")
    if materialized["split_manifest"] != _split_manifest(internal):
        errors.append("split_manifest")
    if materialized["score_contract"] != freeze_score_contract(internal, authority):
        errors.append("score_contract")
    return errors


def _gate(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record exact expected and observed values for one prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Promote the first failed gate into the stable terminal summary."""

    if failure is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_preconditions_and_structural_checks_pass",
            "observed_value": "all_preconditions_and_structural_checks_pass",
            "passed": True,
        }
    return {
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "field": failure["field"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
        "passed": False,
    }


def _resolved_source_paths(
    root: Path, overrides: Mapping[str, Path] | None = None
) -> dict[str, Path]:
    """Resolve frozen input paths while allowing tests to replace one source."""

    paths = {name: root / relative for name, relative in SOURCE_PATHS.items()}
    if overrides:
        paths.update({name: Path(path) for name, path in overrides.items()})
    return paths


def _display_path(root: Path, path: Path) -> str:
    """Use repository-relative paths in production and exact absolute test paths."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _preconditions(
    root: Path,
    paths: Mapping[str, Path],
    result_path: Path,
    checkpoint_path: Path,
    generation_path: Path,
    authority_path: Path,
    *,
    run_date: str = RUN_DATE,
) -> tuple[list[JsonDict], JsonDict | None, dict[str, str], JsonDict | None]:
    """Check date, bytes, upstream gates, tools, and output directories first."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    date_gate = _gate(
        "run_date", "experiment_7180", "run_date", RUN_DATE, run_date, run_date == RUN_DATE
    )
    checks.append(date_gate)
    if not date_gate["passed"]:
        return checks, date_gate, hashes, None
    for name in SOURCE_PATHS:
        path = paths[name]
        passed = path.is_file() and os.access(path, os.R_OK)
        row = _gate(
            f"{name}_path",
            str(SOURCE_PATHS[name]),
            "path",
            "readable_file",
            str(path.resolve()) if passed else "missing_or_unreadable",
            passed,
        )
        checks.append(row)
        if not passed:
            return checks, row, hashes, None
        try:
            hashes[name] = sha256_file(path)
        except OSError:
            row = _gate(
                f"{name}_path",
                str(SOURCE_PATHS[name]),
                "path",
                "readable_file",
                "missing_or_unreadable",
                False,
            )
            checks[-1] = row
            return checks, row, hashes, None
    for name, expected in PINNED_HASHES.items():
        observed = hashes[name]
        row = _gate(
            f"{name}_hash",
            str(SOURCE_PATHS[name]),
            "sha256",
            expected,
            observed,
            observed == expected,
        )
        checks.append(row)
        if not row["passed"]:
            return checks, row, hashes, None
    spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
    observed_req = "REQ-VERIFY-7180" if "REQ-VERIFY-7180" in spec_text else "missing"
    row = _gate(
        "constraint_spec_requirement",
        str(SOURCE_PATHS["constraint_spec"]),
        "requirement",
        "REQ-VERIFY-7180",
        observed_req,
        observed_req == "REQ-VERIFY-7180",
    )
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, None
    try:
        upstream = json.loads(paths["exp7158_artifact"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row = _gate(
            "exp7158_artifact_json",
            str(SOURCE_PATHS["exp7158_artifact"]),
            "json",
            "valid_json_object",
            type(exc).__name__,
            False,
        )
        checks.append(row)
        return checks, row, hashes, None
    if not isinstance(upstream, dict):
        row = _gate(
            "exp7158_artifact_shape",
            str(SOURCE_PATHS["exp7158_artifact"]),
            "root_type",
            "json_object",
            type(upstream).__name__,
            False,
        )
        checks.append(row)
        return checks, row, hashes, None
    from carnot import experiment_7158_v630_entity_evidence_fixture as exp7158

    upstream_errors = []
    if upstream.get("reproducibility_checksum") != exp7158.artifact_checksum(upstream):
        upstream_errors.append("reproducibility_checksum_mismatch")
    if len(upstream.get("rows", [])) != 648:
        upstream_errors.append("row_count_mismatch")
    row = _gate(
        "exp7158_artifact_validation",
        str(SOURCE_PATHS["exp7158_artifact"]),
        "validation_errors",
        [],
        upstream_errors,
        not upstream_errors,
    )
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, upstream
    observed_fields = {field: upstream.get(field) for field in EXP7158_EXPECTED_FIELDS}
    row = _gate(
        "exp7158_same_milestone_fields",
        str(SOURCE_PATHS["exp7158_artifact"]),
        "terminal_gate_fields",
        EXP7158_EXPECTED_FIELDS,
        observed_fields,
        observed_fields == EXP7158_EXPECTED_FIELDS,
    )
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, upstream
    tool_checks = [
        _gate(
            "sqlite_version",
            "python_stdlib.sqlite3",
            "sqlite_version",
            "available",
            sqlite3.sqlite_version,
            bool(sqlite3.sqlite_version),
        ),
        _gate(
            "python_executable",
            ".venv/bin/python",
            "executable",
            "executable_file",
            sys.executable,
            Path(sys.executable).is_file() and os.access(sys.executable, os.X_OK),
        ),
    ]
    for tool_gate in tool_checks:
        checks.append(tool_gate)
        if not tool_gate["passed"]:
            return checks, tool_gate, hashes, upstream
    directory_checks = (
        ("terminal_output_directory", result_path.parent),
        ("checkpoint_output_directory", checkpoint_path.parent),
        ("generation_output_directory", generation_path.parent),
        ("authority_output_directory", authority_path.parent),
    )
    for name, directory in directory_checks:
        passed = directory.is_dir() and os.access(directory, os.W_OK)
        row = _gate(
            name,
            str(directory),
            "directory",
            "existing_writable_directory",
            str(directory.resolve()) if passed else "missing_or_not_writable",
            passed,
        )
        checks.append(row)
        if not passed:
            return checks, row, hashes, upstream
    return checks, None, hashes, upstream


def _base_artifact(
    root: Path, run_date: str, generation_path: Path, authority_path: Path
) -> JsonDict:
    """Create the schema-complete running state used only as a checkpoint."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(
            _gate("fixture_build_complete", "experiment_7180", "status", True, False, False)
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_symbolic_edit_fixture_build",
        "inference_substrate_class": "blocked_no_run",
        "fixture_ready_score": 0,
        "generation_view_path": _display_path(root, generation_path),
        "authority_sidecar_path": _display_path(root, authority_path),
        "split_manifest": {},
        "score_contract": {},
        "mutation_rows": [],
        "sidecar_hashes": {},
        "structural_checks": [],
        "study_question": STUDY_QUESTION,
        "scope_answer": "No fixture result exists while construction is running.",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _blocked_artifact(
    artifact: JsonDict,
    checks: list[JsonDict],
    failure: Mapping[str, Any],
    hashes: Mapping[str, str],
    duration_s: float,
) -> JsonDict:
    """Finish an external precondition failure without claiming fixture work."""

    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": checks,
            "duration_s": duration_s,
            "source_artifact_hashes": dict(hashes),
            "gate_check_summary": _gate_summary(failure),
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failure['check']}_no_symbolic_fixture_run",
            "inference_substrate_class": "blocked_no_run",
            "fixture_ready_score": 0,
            "scope_answer": "An external prerequisite failed before fixture measurement.",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _structural_checks(materialized: Mapping[str, Any]) -> list[JsonDict]:
    """Echo expected and observed values for each positive readiness gate."""

    errors = structural_errors(materialized)
    authority = list(materialized["authority_rows"])
    mutations = list(materialized["mutation_rows"])
    return [
        _gate(
            "base_count",
            "Exp7180 fixture",
            "base_count",
            48,
            len({r["base_id"] for r in materialized["internal_rows"]}),
            len({r["base_id"] for r in materialized["internal_rows"]}) == 48,
        ),
        _gate(
            "model_visible_rows",
            "generation view",
            "row_count",
            192,
            len(materialized["generation_rows"]),
            len(materialized["generation_rows"]) == 192,
        ),
        _gate(
            "authority_exact_agreement",
            "symbolic interpreter + SQLite",
            "agreement_count",
            192,
            sum(bool(row["authority_agreement"]) for row in authority),
            all(row["authority_agreement"] for row in authority),
        ),
        _gate(
            "rename_invariance",
            "mutation rows",
            "unchanged_decision_count",
            48,
            sum(not row["decision_changed"] for row in mutations),
            sum(not row["decision_changed"] for row in mutations) == 48,
        ),
        _gate(
            "semantic_change_sensitivity",
            "mutation rows",
            "changed_decision_count",
            96,
            sum(bool(row["decision_changed"]) for row in mutations),
            sum(bool(row["decision_changed"]) for row in mutations) == 96,
        ),
        _gate("all_structural_errors", "cold fixture replay", "errors", [], errors, not errors),
    ]


def build_artifact(
    root: Path,
    run_date: str,
    *,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
    generation_view_path: Path | None = None,
    authority_sidecar_path: Path | None = None,
    source_paths: Mapping[str, Path] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Gate inputs, construct the fixture, write sidecars, and finish atomically."""

    started = time.monotonic()
    root = Path(root).resolve()
    result = root / (result_path or RESULT_PATH)
    checkpoint = root / (checkpoint_path or CHECKPOINT_PATH)
    generation_path = root / (generation_view_path or GENERATION_VIEW_PATH)
    authority_path = root / (authority_sidecar_path or AUTHORITY_SIDECAR_PATH)
    paths = _resolved_source_paths(root, source_paths)

    _progress(0, "start", "schema_complete_checkpoint_write")
    artifact = _base_artifact(root, run_date, generation_path, authority_path)
    atomic_write_json(checkpoint, artifact, allow_override=False, sort_keys=True)
    _progress(0, "end", "schema_complete_checkpoint_write")

    _progress(1, "start", "preconditions_bytes_gates_tools_directories")
    checks, failure, hashes, upstream = _preconditions(
        root,
        paths,
        result,
        checkpoint,
        generation_path,
        authority_path,
        run_date=run_date,
    )
    _progress(1, "end", f"preconditions checks={len(checks)}")
    measured = duration_s if duration_s is not None else time.monotonic() - started
    if failure is not None:
        blocked = _blocked_artifact(artifact, checks, failure, hashes, measured)
        _progress(8, "start", "blocked_terminal_atomic_write")
        atomic_write_json(result, blocked, allow_override=False, sort_keys=True)
        _progress(8, "end", "blocked_terminal_atomic_write")
        return blocked
    assert upstream is not None

    _progress(2, "start", "source_selection_and_symbolic_variants")
    materialized = materialize_fixture(upstream)
    _progress(2, "end", f"source_selection rows={len(materialized['internal_rows'])}")
    _progress(3, "start", "symbolic_and_sqlite_authority_benchmark")
    print("exp7180 benchmark_start authority_agreement", flush=True)
    agreement_count = sum(row["authority_agreement"] for row in materialized["authority_rows"])
    print("exp7180 benchmark_end authority_agreement", flush=True)
    _progress(3, "end", f"authority_agreement rows={agreement_count}")
    _progress(4, "start", "generation_view_label_isolation")
    view_errors = generation_view_errors(
        materialized["generation_rows"], materialized["authority_rows"]
    )
    _progress(4, "end", f"generation_view_label_isolation errors={len(view_errors)}")
    _progress(5, "start", "frozen_energy_arms_and_controls")
    contract = freeze_score_contract(materialized["internal_rows"], materialized["authority_rows"])
    contract_errors = int(contract != materialized["score_contract"])
    _progress(5, "end", f"frozen_energy_arms_and_controls errors={contract_errors}")
    _progress(6, "start", "independent_mutation_checks")
    mutation_failures = sum(not row["passed"] for row in materialized["mutation_rows"])
    _progress(6, "end", f"independent_mutation_checks failures={mutation_failures}")
    _progress(7, "start", "fixture_readiness_benchmark")
    print("exp7180 benchmark_start structural_replay", flush=True)
    structure = _structural_checks(materialized)
    print("exp7180 benchmark_end structural_replay", flush=True)
    structure_errors = [row for row in structure if not row["passed"]]
    _progress(7, "end", f"fixture_readiness_benchmark errors={len(structure_errors)}")
    all_errors = view_errors + (["score_contract"] if contract_errors else [])
    all_errors += [str(row["check"]) for row in structure_errors]
    if all_errors:
        failure = _gate(
            "fixture_integrity",
            "Exp7180 in-memory fixture",
            "first_error",
            "all_structural_checks_pass",
            all_errors[0],
            False,
        )
        checks.append(failure)
        blocked = _blocked_artifact(artifact, checks, failure, hashes, measured)
        blocked["structural_checks"] = structure
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        _progress(8, "start", "blocked_terminal_atomic_write")
        atomic_write_json(result, blocked, allow_override=False, sort_keys=True)
        _progress(8, "end", "blocked_terminal_atomic_write")
        return blocked

    generation_bytes = jsonl_bytes(materialized["generation_rows"])
    authority_bytes = jsonl_bytes(materialized["authority_rows"])
    _progress(8, "start", "sidecars_and_complete_terminal_atomic_write")
    atomic_write_bytes(generation_path, generation_bytes, allow_override=False)
    atomic_write_bytes(authority_path, authority_bytes, allow_override=False)
    sidecar_hashes = {
        "generation_view_sha256": sha256_bytes(generation_bytes),
        "authority_sidecar_sha256": sha256_bytes(authority_bytes),
        "generation_view_row_count": len(materialized["generation_rows"]),
        "authority_sidecar_row_count": len(materialized["authority_rows"]),
        "label_mutation_preserves_generation_view_bytes": True,
    }
    measured = duration_s if duration_s is not None else time.monotonic() - started
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": checks,
            "duration_s": measured,
            "source_artifact_hashes": hashes,
            "rows": materialized["rows"],
            "gate_check_summary": _gate_summary(None),
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_symbolic_edit_fixture_ready_no_live_verifier_result",
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "fixture_ready_score": 1,
            "split_manifest": materialized["split_manifest"],
            "score_contract": materialized["score_contract"],
            "mutation_rows": materialized["mutation_rows"],
            "sidecar_hashes": sidecar_hashes,
            "structural_checks": structure,
            "scope_answer": SCOPE_ANSWER,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_write_json(result, artifact, allow_override=False, sort_keys=True)
    _progress(8, "end", "sidecars_and_complete_terminal_atomic_write")
    return artifact


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read a sidecar exactly as JSONL for cold file-to-parser validation."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL row is not an object")
        rows.append(value)
    return rows


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    *,
    root: Path | None = None,
    generation_view_path: Path | None = None,
    authority_sidecar_path: Path | None = None,
) -> list[str]:
    """Cold-check sources, sidecars, labels, controls, state, and checksum."""

    if isinstance(value, (str, Path)):
        artifact_path = Path(value)
        if not artifact_path.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        if not isinstance(loaded, dict):
            return ["artifact_not_object"]
        artifact: Mapping[str, Any] = loaded
    elif isinstance(value, Mapping):
        artifact = value
    else:
        return ["artifact_not_object"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [f"artifact_fields_mismatch missing={missing} extra={extra}"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if not isinstance(artifact.get("duration_s"), (int, float)) or artifact["duration_s"] < 0:
        errors.append("duration_s_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("study_question") != STUDY_QUESTION:
        errors.append("study_question_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    failed = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if isinstance(row, Mapping) and row.get("passed") is False
        ),
        None,
    )
    if failed is not None:
        if artifact.get("status") != "blocked":
            errors.append("blocked_status_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_inference_substrate_class_mismatch")
        if artifact.get("gate_check_summary") != _gate_summary(failed):
            errors.append("blocked_gate_check_summary_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if artifact.get("fixture_ready_score") != 0:
            errors.append("blocked_fixture_ready_score_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
        return list(dict.fromkeys(errors))
    if artifact.get("status") != "complete":
        errors.append("complete_status_mismatch")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("fixture_ready_score") != 1:
        errors.append("fixture_ready_score_mismatch")
    if artifact.get("gate_check_summary") != _gate_summary(None):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("verdict_class") != "positive":
        errors.append("verdict_class_mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_positive"):
        errors.append("honest_verdict_mismatch")
    if artifact.get("scope_answer") != SCOPE_ANSWER:
        errors.append("scope_answer_mismatch")

    repo = find_repo_root() if root is None else Path(root).resolve()
    paths = _resolved_source_paths(repo)
    if not all(path.is_file() for path in paths.values()):
        errors.append("source_artifact_missing")
        return list(dict.fromkeys(errors))
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if artifact.get("source_artifact_hashes") != observed_hashes:
        errors.append("source_artifact_hashes_mismatch")
    try:
        upstream = json.loads(paths["exp7158_artifact"].read_text(encoding="utf-8"))
        expected = materialize_fixture(upstream)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"independent_replay_failed:{type(exc).__name__}")
        return list(dict.fromkeys(errors))
    for field in ("rows", "split_manifest", "score_contract", "mutation_rows"):
        if artifact.get(field) != expected[field]:
            errors.append(f"{field}_mismatch")
    expected_checks = _structural_checks(expected)
    if artifact.get("structural_checks") != expected_checks:
        errors.append("structural_checks_mismatch")
    generation_path = (
        Path(generation_view_path)
        if generation_view_path is not None
        else repo / str(artifact["generation_view_path"])
    )
    authority_path = (
        Path(authority_sidecar_path)
        if authority_sidecar_path is not None
        else repo / str(artifact["authority_sidecar_path"])
    )
    if not generation_path.is_file() or not authority_path.is_file():
        errors.append("sidecar_missing")
        return list(dict.fromkeys(errors))
    try:
        generation_rows = _read_jsonl(generation_path)
        authority_rows = _read_jsonl(authority_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"sidecar_unreadable:{type(exc).__name__}")
        return list(dict.fromkeys(errors))
    if generation_rows != expected["generation_rows"]:
        errors.append("generation_view_rows_mismatch")
    if authority_rows != expected["authority_rows"]:
        errors.append("authority_sidecar_rows_mismatch")
    sidecar_hashes = {
        "generation_view_sha256": sha256_file(generation_path),
        "authority_sidecar_sha256": sha256_file(authority_path),
        "generation_view_row_count": len(generation_rows),
        "authority_sidecar_row_count": len(authority_rows),
        "label_mutation_preserves_generation_view_bytes": True,
    }
    if artifact.get("sidecar_hashes") != sidecar_hashes:
        errors.append("sidecar_hashes_mismatch")
    errors.extend(structural_errors(expected))
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fixed-date fixture or validate one artifact and its sidecars."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--generation-view-path", type=Path, default=GENERATION_VIEW_PATH)
    parser.add_argument("--authority-sidecar-path", type=Path, default=AUTHORITY_SIDECAR_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(7, "start", "cold_artifact_validation")
        print("exp7180 subprocess_start cold_artifact_validation", flush=True)
        errors = validate_artifact(
            args.validate,
            generation_view_path=args.generation_view_path,
            authority_sidecar_path=args.authority_sidecar_path,
        )
        print("exp7180 subprocess_end cold_artifact_validation", flush=True)
        _progress(7, "end", f"cold_artifact_validation errors={len(errors)}")
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if not re.fullmatch(r"[0-9]{8}", args.date) or args.date != RUN_DATE:
        return 2
    artifact = build_artifact(
        find_repo_root(),
        args.date,
        result_path=args.result_path,
        checkpoint_path=args.checkpoint_path,
        generation_view_path=args.generation_view_path,
        authority_sidecar_path=args.authority_sidecar_path,
    )
    errors = validate_artifact(
        artifact,
        generation_view_path=args.generation_view_path,
        authority_sidecar_path=args.authority_sidecar_path,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "fixture_ready_score": artifact["fixture_ready_score"],
                "verdict_class": artifact["verdict_class"],
                "valid": not errors,
                "errors": errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(bool(errors) or artifact["verdict_class"] != "positive")


if __name__ == "__main__":  # pragma: no cover - the executable wrapper calls main.
    raise SystemExit(main())
