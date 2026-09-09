"""Build the V630 counterfactual entity-to-evidence fixture.

This module turns the frozen Exp7138 relational source rows into exact,
source-grounded claim/evidence pairs.  It never loads a model.  The exact
labels come from controlled construction rules, while the separately frozen
energy vector is only a candidate verifier for later work.

Spec refs: REQ-VERIFY-7158 and SCENARIO-VERIFY-7158-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
RANDOM_SEED = 7_158_202_609_09
BASE_COUNT = 72
RESULT_PATH = Path("results/experiment_7158_v630_entity_evidence_fixture.json")
INFERENCE_SUBSTRATE = "exact_source_fixture_construction"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

SOURCE_FAMILIES = ("CNN/DM", "Recent News", "MARCO", "Yelp")
FAMILY_SPLITS = {
    "CNN/DM": "train",
    "Recent News": "calibration",
    "MARCO": "evaluation",
    "Yelp": "evaluation",
}
CONDITIONS = (
    "supported",
    "entity_substitution",
    "relation_reversal",
    "evidence_removal",
    "irrelevant_evidence_added",
    "irrelevant_evidence_only",
    "negation",
    "numeric_unit_change",
    "duplicate_alias",
)
CONDITION_LABELS = {
    "supported": "supported",
    "entity_substitution": "unsupported",
    "relation_reversal": "unsupported",
    "evidence_removal": "unsupported",
    "irrelevant_evidence_added": "supported",
    "irrelevant_evidence_only": "unsupported",
    "negation": "unsupported",
    "numeric_unit_change": "unsupported",
    "duplicate_alias": "supported",
}
PERTURBATION_DESCRIPTIONS = {
    "supported": "unchanged source-grounded evidence",
    "entity_substitution": "claim subject replaced by a source-absent entity",
    "relation_reversal": "claim entity roles reversed while evidence stays fixed",
    "evidence_removal": "all evidence removed while the claim stays fixed",
    "irrelevant_evidence_added": "irrelevant same-family evidence added after support",
    "irrelevant_evidence_only": "support replaced by irrelevant same-family evidence",
    "negation": "claim polarity negated while evidence stays fixed",
    "numeric_unit_change": "claim quantity unit changed while evidence stays fixed",
    "duplicate_alias": "duplicate entity aliases added without changing identity",
}

ENERGY_TERMS = (
    "entity_presence",
    "relation_role_agreement",
    "polarity",
    "quantity_unit_agreement",
    "counterfactual_sensitivity",
)
ENERGY_INPUT_FIELDS = (
    "claim_fact",
    "claim_entities",
    "evidence_entities",
    "evidence_facts",
    "base_support_fact_hash",
    "evidence_fact_hashes",
)
SEALED_FIELDS = (
    "condition",
    "perturbation",
    "support_label",
    "expected_answer",
    "exact_label_rule",
)
# Quoted JSON keys catch aliases without rejecting ordinary prose that happens
# to use a word such as "condition".
SEALED_TOKENS = (
    "support_label",
    "expected_answer",
    "exact_label_rule",
    '"condition"',
    '"perturbation"',
    '"split"',
)

SOURCE_PATHS = {
    "exp7138_artifact": Path("results/experiment_7138_v627_relational_fixture.json"),
    "ragtruth_license": Path("data/ragtruth/LICENSE"),
    "constraint_spec": Path("openspec/capabilities/constraint-verification/spec.md"),
    "reporting_spec": Path("openspec/capabilities/research-reporting/spec.md"),
    "relational_module": Path("python/carnot/experiment_7138_v627_relational_fixture.py"),
    "verify_package": Path("python/carnot/verify/__init__.py"),
}
PINNED_HASHES = {
    "exp7138_artifact": "sha256:c9239d41c038df709d9f339b5e7e2e81a1473de1267814161fdb5da86e0d4f27",
    "ragtruth_license": "sha256:b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "source_family_rows",
    "entity_evidence_rows",
    "perturbation_rows",
    "split_rows",
    "sealed_field_rows",
    "energy_term_contract",
    "mutation_test_rows",
    "frozen_fixture_ids",
    "counterfactual_fixture_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Field reasons distinguish fixture integrity from future verifier performance.",
    "status": "A terminal state prevents an unfinished corpus from gating generation.",
    "preconditions_checked": "Input and license receipts prove the fixture was built from available admissible sources.",
    "run_date": "The date binds the corpus to the source and split versions used.",
    "inference_substrate": "Use exact_source_fixture_construction for deterministic labeled data work.",
    "inference_substrate_class": "Use cpu_exact_solver_or_simulator, or blocked_no_run before construction, because no model is loaded.",
    "execution_venue": "Host execution prevents any model or board attribution.",
    "duration_s": "Measured build time reveals truncated fixture generation.",
    "source_artifact_hashes": "Hashes bind every derived row to its source inputs.",
    "rows": "One row per base example, condition, and perturbation makes all comparisons reconstructable.",
    "source_family_rows": "Family coverage prevents one template from carrying the result.",
    "entity_evidence_rows": "Entity and span anchors expose localization failures hidden by aggregate labels.",
    "perturbation_rows": "Paired mutations provide the causal test of evidence sensitivity.",
    "split_rows": "Explicit split membership prevents calibration leakage into evaluation.",
    "sealed_field_rows": "Sealing receipts prove evaluation truth cannot enter prompts or scores.",
    "energy_term_contract": "A frozen auditable energy prevents post-outcome score tuning.",
    "mutation_test_rows": "Term-level mutations show which constraint fired and why.",
    "frozen_fixture_ids": "Stable IDs let later tasks reproduce the exact paired schedule.",
    "counterfactual_fixture_ready_score": "The exact field gates generation on corpus integrity, not expected performance.",
    "random_seed": "A fixed seed makes split and perturbation construction reproducible.",
    "reproducibility_checksum": "The checksum catches source, split, or contract drift.",
    "gate_check_summary": "A blocked artifact records the missing input and observed value.",
    "verifier_is_oracle": "False records that exact labels are separate from the frozen candidate energy.",
    "verdict_class": "Use the closed enum positive | circular_positive | null | blocked | disqualified | partial for structural aggregation.",
    "honest_verdict": "The terminal text reports fixture readiness only and makes no verifier-value claim.",
}

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were",
    "are", "has", "have", "had", "into", "over", "under", "about", "after",
    "before", "your", "you", "its", "their", "they", "but", "not", "our",
}
_MUTATION_EXPECTATIONS = {
    "entity_substitution": ["entity_presence"],
    "relation_reversal": ["relation_role_agreement"],
    "evidence_removal": ["entity_presence", "counterfactual_sensitivity"],
    "irrelevant_evidence_added": [],
    "irrelevant_evidence_only": ["entity_presence", "counterfactual_sensitivity"],
    "negation": ["polarity"],
    "numeric_unit_change": ["quantity_unit_agreement"],
    "duplicate_alias": [],
}


def canonical_json(value: Any) -> str:
    """Return the one compact JSON spelling used by every content hash."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text and include the digest algorithm in the value."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without normalizing whitespace or line endings."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding runtime duration and this digest."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_text(canonical_json(payload))


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Print one machine-readable phase boundary and flush it immediately."""

    print(f"exp7158 phase={phase} event={event} detail={detail}", flush=True)


def _span(text: str, start: int, end: int, *, document: str) -> JsonDict:
    """Describe one zero-based, end-exclusive character slice with its hash."""

    exact = text[start:end]
    return {
        "document": document,
        "start": start,
        "end": end,
        "text": exact,
        "text_sha256": sha256_text(exact),
    }


def _stable_id(prefix: str, *parts: str) -> str:
    """Make a stable opaque identifier from source-only strings."""

    return f"{prefix}-{sha256_text('|'.join(parts)).split(':', 1)[1][:20]}"


def _source_fact(row: Mapping[str, Any]) -> JsonDict:
    """Extract a literal ordered-token relation and exact source offsets."""

    source_text = str(row["source_text"])
    candidates = [
        match
        for match in _WORD_RE.finditer(source_text)
        if len(match.group()) >= 3 and match.group().casefold() not in _STOPWORDS
    ]
    first = candidates[0]
    second = next(
        match for match in candidates[1:] if match.group().casefold() != first.group().casefold()
    )
    source_fixture_id = str(row["fixture_id"])
    subject_id = _stable_id("entity", source_fixture_id, "subject")
    object_id = _stable_id("entity", source_fixture_id, "object")
    source_span = _span(source_text, first.start(), second.end(), document="source")
    entities = [
        {
            "entity_id": subject_id,
            "canonical_name": first.group(),
            "aliases": [first.group()],
            "source_start": first.start(),
            "source_end": first.end(),
        },
        {
            "entity_id": object_id,
            "canonical_name": second.group(),
            "aliases": [second.group()],
            "source_start": second.start(),
            "source_end": second.end(),
        },
    ]
    fact = {
        "subject_entity_id": subject_id,
        "relation": "precedes",
        "object_entity_id": object_id,
        "polarity": "positive",
        "quantity": second.start() - first.end(),
        "unit": "code_points",
    }
    return {
        "source_fixture_id": source_fixture_id,
        "source_id": str(row["source_id"]),
        "source_family": str(row["source_family"]),
        "source_text": source_text,
        "source_text_sha256": str(row["source_text_sha256"]),
        "source_span": source_span,
        "evidence_text": source_span["text"],
        "entities": entities,
        "fact": fact,
        "fact_hash": sha256_text(canonical_json(fact)),
    }


def _entity_public(entity: Mapping[str, Any], *, duplicate: bool = False) -> JsonDict:
    """Copy the model-visible identity fields, optionally retaining duplicate aliases."""

    canonical = str(entity["canonical_name"])
    aliases = [canonical]
    if duplicate:
        aliases.extend([canonical, canonical.casefold()])
    return {
        "entity_id": str(entity["entity_id"]),
        "canonical_name": canonical,
        "aliases": aliases,
    }


def _claim_for_condition(base: Mapping[str, Any], condition: str) -> tuple[str, JsonDict, list[JsonDict]]:
    """Render one controlled claim and its structured relational meaning."""

    entities = list(base["entities"])
    subject = _entity_public(entities[0], duplicate=condition == "duplicate_alias")
    obj = _entity_public(entities[1], duplicate=condition == "duplicate_alias")
    fact = deepcopy(base["fact"])
    if condition == "entity_substitution":
        decoy_name = "Counterfactual" + sha256_text(str(base["source_fixture_id"]))[-8:]
        subject = {
            "entity_id": _stable_id("entity", str(base["source_fixture_id"]), "decoy"),
            "canonical_name": decoy_name,
            "aliases": [decoy_name],
        }
        fact["subject_entity_id"] = subject["entity_id"]
    elif condition == "relation_reversal":
        subject, obj = obj, subject
        fact["subject_entity_id"], fact["object_entity_id"] = (
            fact["object_entity_id"],
            fact["subject_entity_id"],
        )
    elif condition == "negation":
        fact["polarity"] = "negative"
    elif condition == "numeric_unit_change":
        fact["unit"] = "bytes"

    subject_name = str(subject["canonical_name"])
    object_name = str(obj["canonical_name"])
    quantity = fact["quantity"]
    unit = fact["unit"]
    if condition == "negation":
        claim = f"{subject_name} does not precede {object_name} by {quantity} {unit}."
    elif condition == "duplicate_alias":
        claim = (
            f"{subject_name} ({subject_name}) precedes "
            f"{object_name} ({object_name}) by {quantity} {unit}."
        )
    else:
        claim = f"{subject_name} precedes {object_name} by {quantity} {unit}."
    return claim, fact, [subject, obj]


def _evidence_for_condition(
    base: Mapping[str, Any], irrelevant: Mapping[str, Any], condition: str
) -> tuple[str, list[JsonDict], list[JsonDict], dict[str, int]]:
    """Return exact evidence components and their local starting offsets."""

    base_text = str(base["evidence_text"])
    irrelevant_text = str(irrelevant["evidence_text"])
    if condition == "evidence_removal":
        return "", [], [], {}
    if condition == "irrelevant_evidence_only":
        return (
            irrelevant_text,
            [_entity_public(entity) for entity in irrelevant["entities"]],
            [deepcopy(irrelevant["fact"])],
            {str(irrelevant["source_fixture_id"]): 0},
        )
    if condition == "irrelevant_evidence_added":
        separator = "\n---\n"
        return (
            base_text + separator + irrelevant_text,
            [_entity_public(entity) for entity in base["entities"]]
            + [_entity_public(entity) for entity in irrelevant["entities"]],
            [deepcopy(base["fact"]), deepcopy(irrelevant["fact"])],
            {
                str(base["source_fixture_id"]): 0,
                str(irrelevant["source_fixture_id"]): len(base_text) + len(separator),
            },
        )
    return (
        base_text,
        [_entity_public(entity) for entity in base["entities"]],
        [deepcopy(base["fact"])],
        {str(base["source_fixture_id"]): 0},
    )


def _all_occurrences(text: str, needle: str, document: str) -> list[JsonDict]:
    """Return exact spans for every non-overlapping spelling of one entity."""

    spans: list[JsonDict] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return spans
        spans.append(_span(text, found, found + len(needle), document=document))
        start = found + len(needle)


def _entity_receipts(
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    irrelevant: Mapping[str, Any],
    evidence_offsets: Mapping[str, int],
) -> list[JsonDict]:
    """Build entity-level source, claim, and evidence localization receipts."""

    owners = {
        str(entity["entity_id"]): (base, entity)
        for entity in base["entities"]
    }
    owners.update(
        {
            str(entity["entity_id"]): (irrelevant, entity)
            for entity in irrelevant["entities"]
        }
    )
    claim_entities = {str(entity["entity_id"]): entity for entity in row["claim_entities"]}
    evidence_entities = {
        str(entity["entity_id"]): entity for entity in row["evidence_entities"]
    }
    entity_ids = sorted(set(claim_entities) | set(evidence_entities))
    receipts: list[JsonDict] = []
    for entity_id in entity_ids:
        entity = claim_entities.get(entity_id) or evidence_entities[entity_id]
        canonical = str(entity["canonical_name"])
        owner = owners.get(entity_id)
        source_spans: list[JsonDict] = []
        evidence_spans: list[JsonDict] = []
        source_fixture_id: str | None = None
        if owner is not None:
            owner_base, owner_entity = owner
            source_fixture_id = str(owner_base["source_fixture_id"])
            source_span = _span(
                str(owner_base["source_text"]),
                int(owner_entity["source_start"]),
                int(owner_entity["source_end"]),
                document="source",
            )
            source_span["source_fixture_id"] = source_fixture_id
            source_spans.append(source_span)
            if source_fixture_id in evidence_offsets:
                local = evidence_offsets[source_fixture_id] + (
                    int(owner_entity["source_start"]) - int(owner_base["source_span"]["start"])
                )
                evidence_spans.append(
                    _span(
                        str(row["evidence_text"]),
                        local,
                        local + len(canonical),
                        document="evidence",
                    )
                )
        claim_spans = (
            _all_occurrences(str(row["claim_text"]), canonical, "claim")
            if entity_id in claim_entities
            else []
        )
        receipts.append(
            {
                "fixture_id": row["fixture_id"],
                "entity_id": entity_id,
                "canonical_name": canonical,
                "aliases": list(entity["aliases"]),
                "source_fixture_id": source_fixture_id,
                "source_spans": source_spans,
                "claim_spans": claim_spans,
                "evidence_spans": evidence_spans,
            }
        )
    return receipts


def render_generation_prompt(row: Mapping[str, Any]) -> str:
    """Render a neutral future-generation view with no exact truth metadata."""

    return (
        f"Evidence:\n{row['evidence_text']}\n\nClaim:\n{row['claim_text']}\n\n"
        "Return one answer word: yes if the evidence states the claim, otherwise no."
    )


def energy_input(row: Mapping[str, Any]) -> JsonDict:
    """Project only the six fields admitted by the frozen candidate scorer."""

    return {field: deepcopy(row[field]) for field in ENERGY_INPUT_FIELDS}


def compute_energy_vector(value: Mapping[str, Any]) -> dict[str, int]:
    """Compute five deterministic mismatch terms without reading a truth label."""

    claim = value["claim_fact"]
    evidence_facts = list(value["evidence_facts"])
    evidence_ids = {str(entity["entity_id"]) for entity in value["evidence_entities"]}
    subject_id = str(claim["subject_entity_id"])
    object_id = str(claim["object_entity_id"])
    entities_present = subject_id in evidence_ids and object_id in evidence_ids
    same_roles = [
        fact
        for fact in evidence_facts
        if fact["subject_entity_id"] == subject_id
        and fact["object_entity_id"] == object_id
        and fact["relation"] == claim["relation"]
    ]
    same_polarity = [fact for fact in same_roles if fact["polarity"] == claim["polarity"]]
    return {
        "entity_presence": int(not entities_present),
        "relation_role_agreement": int(entities_present and not same_roles),
        "polarity": int(bool(same_roles) and not same_polarity),
        "quantity_unit_agreement": int(
            bool(same_polarity)
            and not any(
                fact["quantity"] == claim["quantity"] and fact["unit"] == claim["unit"]
                for fact in same_polarity
            )
        ),
        "counterfactual_sensitivity": int(
            value["base_support_fact_hash"] not in value["evidence_fact_hashes"]
        ),
    }


def changed_energy_terms(
    baseline: Mapping[str, int], current: Mapping[str, int]
) -> list[str]:
    """List changed energy terms in the contract's fixed presentation order."""

    return [term for term in ENERGY_TERMS if baseline[term] != current[term]]


def freeze_energy_contract(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Fit weights and threshold from calibration rows, never evaluation truth."""

    calibration = [row for row in rows if row["split"] == "calibration"]
    candidates = [
        (weights, threshold)
        for weights in itertools.product((1, 2), repeat=len(ENERGY_TERMS))
        for threshold in range(0, 3)
    ]
    ranked: list[tuple[int, tuple[int, ...], int]] = []
    for weights, threshold in candidates:
        correct = 0
        for row in calibration:
            vector = compute_energy_vector(energy_input(row))
            score = sum(weights[index] * vector[term] for index, term in enumerate(ENERGY_TERMS))
            prediction = "supported" if score <= threshold else "unsupported"
            correct += int(prediction == row["support_label"])
        ranked.append((-correct, weights, threshold))
    _, weights, threshold = min(ranked)
    calibration_truth = [
        {"fixture_id": row["fixture_id"], "support_label": row["support_label"]}
        for row in calibration
    ]
    return {
        "candidate_verifier_only": True,
        "term_order": list(ENERGY_TERMS),
        "term_semantics": {
            "entity_presence": "one when either claim entity ID is absent from evidence",
            "relation_role_agreement": "one when present entities lack the claimed directed relation",
            "polarity": "one when matching directed evidence has the opposite polarity",
            "quantity_unit_agreement": "one when matching directed evidence disagrees on quantity or unit",
            "counterfactual_sensitivity": "one when the base support fact is absent from evidence",
        },
        "weights": {term: weights[index] for index, term in enumerate(ENERGY_TERMS)},
        "threshold": threshold,
        "tie_rule": "energy_equal_threshold_is_supported",
        "decision_rule": "weighted_energy_above_threshold_is_unsupported",
        "fit_split": "calibration",
        "fit_source_families": sorted({str(row["source_family"]) for row in calibration}),
        "fit_fixture_ids": [str(row["fixture_id"]) for row in calibration],
        "calibration_row_count": len(calibration),
        "calibration_base_count": len({str(row["base_id"]) for row in calibration}),
        "calibration_truth_sha256": sha256_text(canonical_json(calibration_truth)),
        "candidate_grid_sha256": sha256_text(canonical_json(candidates)),
        "evaluation_truth_accessed": False,
        "verifier_is_oracle": False,
    }


def materialize_fixture(upstream: Mapping[str, Any]) -> JsonDict:
    """Construct all paired rows and integrity receipts in memory."""

    upstream_rows = list(upstream.get("rows", []))
    if len(upstream_rows) != BASE_COUNT:
        raise ValueError(f"expected {BASE_COUNT} Exp7138 rows, observed {len(upstream_rows)}")
    bases = [_source_fact(row) for row in upstream_rows]
    family_bases: dict[str, list[JsonDict]] = defaultdict(list)
    for base in bases:
        family_bases[str(base["source_family"])].append(base)
    if tuple(sorted(family_bases)) != tuple(sorted(SOURCE_FAMILIES)):
        raise ValueError("Exp7138 source families do not match the frozen schedule")
    irrelevant_by_fixture: dict[str, JsonDict] = {}
    for family in SOURCE_FAMILIES:
        ordered = sorted(family_bases[family], key=lambda item: str(item["source_fixture_id"]))
        for index, base in enumerate(ordered):
            irrelevant_by_fixture[str(base["source_fixture_id"])] = ordered[(index + 1) % len(ordered)]

    rows: list[JsonDict] = []
    receipt_context: dict[str, tuple[JsonDict, JsonDict, dict[str, int]]] = {}
    for base in bases:
        irrelevant = irrelevant_by_fixture[str(base["source_fixture_id"])]
        base_id = _stable_id("base", str(base["source_fixture_id"]), str(base["source_text_sha256"]))
        pair_id = _stable_id("pair", base_id)
        for condition in CONDITIONS:
            claim_text, claim_fact, claim_entities = _claim_for_condition(base, condition)
            evidence_text, evidence_entities, evidence_facts, offsets = _evidence_for_condition(
                base, irrelevant, condition
            )
            fixture_id = _stable_id("eef", base_id, condition)
            label = CONDITION_LABELS[condition]
            evidence_fact_hashes = [sha256_text(canonical_json(fact)) for fact in evidence_facts]
            row = {
                "fixture_id": fixture_id,
                "base_id": base_id,
                "pair_id": pair_id,
                "source_fixture_id": base["source_fixture_id"],
                "source_id": base["source_id"],
                "source_family": base["source_family"],
                "split": FAMILY_SPLITS[str(base["source_family"])],
                "condition": condition,
                "perturbation": PERTURBATION_DESCRIPTIONS[condition],
                "source_text_sha256": base["source_text_sha256"],
                "source_span": deepcopy(base["source_span"]),
                "evidence_text": evidence_text,
                "evidence_text_sha256": sha256_text(evidence_text),
                "claim_text": claim_text,
                "claim_text_sha256": sha256_text(claim_text),
                "claim_span": _span(claim_text, 0, len(claim_text), document="claim"),
                "claim_fact": claim_fact,
                "claim_entities": claim_entities,
                "evidence_entities": evidence_entities,
                "evidence_facts": evidence_facts,
                "base_support_fact_hash": base["fact_hash"],
                "evidence_fact_hashes": evidence_fact_hashes,
                "expected_answer": "yes" if label == "supported" else "no",
                "support_label": label,
                "exact_label_rule": f"controlled_fixture_condition:{condition}",
            }
            rows.append(row)
            receipt_context[fixture_id] = (base, irrelevant, offsets)
    condition_rank = {condition: index for index, condition in enumerate(CONDITIONS)}
    rows.sort(key=lambda row: (str(row["base_id"]), condition_rank[str(row["condition"])]))

    entity_rows: list[JsonDict] = []
    for row in rows:
        base, irrelevant, offsets = receipt_context[str(row["fixture_id"])]
        entity_rows.extend(_entity_receipts(row, base, irrelevant, offsets))

    by_base: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        by_base[str(row["base_id"])][str(row["condition"])] = row
    perturbation_rows: list[JsonDict] = []
    mutation_rows: list[JsonDict] = []
    for base_id in sorted(by_base):
        group = by_base[base_id]
        baseline = group["supported"]
        baseline_vector = compute_energy_vector(energy_input(baseline))
        for condition in CONDITIONS[1:]:
            current = group[condition]
            current_vector = compute_energy_vector(energy_input(current))
            changed = changed_energy_terms(baseline_vector, current_vector)
            perturbation_rows.append(
                {
                    "base_id": base_id,
                    "pair_id": baseline["pair_id"],
                    "supported_fixture_id": baseline["fixture_id"],
                    "perturbed_fixture_id": current["fixture_id"],
                    "condition": condition,
                    "support_label": current["support_label"],
                    "claim_text_sha256": current["claim_text_sha256"],
                    "evidence_text_sha256": current["evidence_text_sha256"],
                }
            )
            expected = _MUTATION_EXPECTATIONS[condition]
            mutation_rows.append(
                {
                    "base_id": base_id,
                    "condition": condition,
                    "baseline_fixture_id": baseline["fixture_id"],
                    "mutated_fixture_id": current["fixture_id"],
                    "baseline_vector": baseline_vector,
                    "mutated_vector": current_vector,
                    "changed_terms": changed,
                    "expected_changed_terms": expected,
                    "unchanged_terms": [term for term in ENERGY_TERMS if term not in changed],
                    "passed": changed == expected,
                }
            )

    family_rows = []
    split_rows = []
    for family in SOURCE_FAMILIES:
        selected = [row for row in rows if row["source_family"] == family]
        base_ids = sorted({str(row["base_id"]) for row in selected})
        source_ids = sorted({str(row["source_id"]) for row in selected})
        family_rows.append(
            {
                "source_family": family,
                "split": FAMILY_SPLITS[family],
                "base_count": len(base_ids),
                "row_count": len(selected),
                "condition_counts": dict(sorted(Counter(row["condition"] for row in selected).items())),
            }
        )
        split_rows.append(
            {
                "source_family": family,
                "split": FAMILY_SPLITS[family],
                "base_count": len(base_ids),
                "row_count": len(selected),
                "source_ids": source_ids,
                "base_ids_sha256": sha256_text(canonical_json(base_ids)),
                "frozen_before_threshold_fitting": True,
            }
        )

    sealed_rows: list[JsonDict] = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        prompt = render_generation_prompt(row)
        scorer_input = energy_input(row)
        sealed_truth = {field: row[field] for field in SEALED_FIELDS}
        sealed_rows.append(
            {
                "fixture_id": row["fixture_id"],
                "sealed_fields": list(SEALED_FIELDS),
                "sealed_truth_sha256": sha256_text(canonical_json(sealed_truth)),
                "model_view_fields": ["fixture_id", "evidence_text", "claim_text", "prompt"],
                "prompt_sha256": sha256_text(prompt),
                "energy_input_fields": list(ENERGY_INPUT_FIELDS),
                "energy_input_sha256": sha256_text(canonical_json(scorer_input)),
                "passed": True,
            }
        )

    return {
        "rows": rows,
        "source_family_rows": family_rows,
        "entity_evidence_rows": entity_rows,
        "perturbation_rows": perturbation_rows,
        "split_rows": split_rows,
        "sealed_field_rows": sealed_rows,
        "energy_term_contract": freeze_energy_contract(rows),
        "mutation_test_rows": mutation_rows,
        "frozen_fixture_ids": [row["fixture_id"] for row in rows],
    }


def _span_matches(text: str, span: Mapping[str, Any]) -> bool:
    """Return whether one receipt exactly names a valid text slice."""

    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        return False
    exact = text[start:end]
    return (
        end <= len(text)
        and span.get("text") == exact
        and span.get("text_sha256") == sha256_text(exact)
    )


def span_errors(
    rows: Sequence[Mapping[str, Any]],
    entity_rows: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
) -> list[str]:
    """Replay every row-level and entity-level character offset from source."""

    errors: list[str] = []
    sources = {str(row["fixture_id"]): row for row in upstream.get("rows", [])}
    primary = {str(row["fixture_id"]): row for row in rows}
    for row in rows:
        fixture_id = str(row["fixture_id"])
        source = sources.get(str(row["source_fixture_id"]))
        if source is None or not _span_matches(str(source["source_text"]), row["source_span"]):
            errors.append(f"source_span_mismatch:{fixture_id}")
        if not _span_matches(str(row["claim_text"]), row["claim_span"]):
            errors.append(f"claim_span_mismatch:{fixture_id}")
        if row.get("evidence_text_sha256") != sha256_text(str(row["evidence_text"])):
            errors.append(f"evidence_text_hash_mismatch:{fixture_id}")
    for receipt in entity_rows:
        fixture_id = str(receipt["fixture_id"])
        row = primary.get(fixture_id)
        if row is None:
            errors.append(f"entity_parent_missing:{fixture_id}")
            continue
        for span in receipt.get("source_spans", []):
            source = sources.get(str(span.get("source_fixture_id")))
            if source is None or not _span_matches(str(source["source_text"]), span):
                errors.append(f"entity_source_span_mismatch:{fixture_id}:{receipt['entity_id']}")
        for span in receipt.get("claim_spans", []):
            if not _span_matches(str(row["claim_text"]), span):
                errors.append(f"entity_claim_span_mismatch:{fixture_id}:{receipt['entity_id']}")
        for span in receipt.get("evidence_spans", []):
            if not _span_matches(str(row["evidence_text"]), span):
                errors.append(f"entity_evidence_span_mismatch:{fixture_id}:{receipt['entity_id']}")
    return errors


def split_errors(
    rows: Sequence[Mapping[str, Any]], split_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Check frozen family routing, evaluation size, and family isolation."""

    errors: list[str] = []
    if {str(row["source_family"]) for row in split_rows} != set(SOURCE_FAMILIES):
        errors.append("split_family_roster_mismatch")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[str(row["source_family"])].add(str(row["split"]))
    if any(len(values) != 1 for values in family_splits.values()):
        errors.append("source_family_split_leakage")
    if any(family_splits.get(family) != {split} for family, split in FAMILY_SPLITS.items()):
        errors.append("frozen_family_split_mismatch")
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(evaluation) < 48:
        errors.append("evaluation_row_shortfall")
    if len({str(row["base_id"]) for row in evaluation}) < 30:
        errors.append("evaluation_pair_shortfall")
    return errors


def sealing_errors(
    rows: Sequence[Mapping[str, Any]], sealed_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Prove evaluation truth is absent from future prompts and energy inputs."""

    errors: list[str] = []
    evaluation = {str(row["fixture_id"]): row for row in rows if row["split"] == "evaluation"}
    receipts = {str(row["fixture_id"]): row for row in sealed_rows}
    if set(receipts) != set(evaluation):
        errors.append("sealed_evaluation_roster_mismatch")
    for fixture_id, row in evaluation.items():
        receipt = receipts.get(fixture_id)
        if receipt is None:
            continue
        if set(receipt.get("energy_input_fields", [])) != set(ENERGY_INPUT_FIELDS):
            errors.append(f"sealed_energy_fields:{fixture_id}")
            continue
        prompt = render_generation_prompt(row)
        scorer_input = energy_input(row)
        serialized = (prompt + canonical_json(scorer_input)).lower()
        if any(token in serialized for token in SEALED_TOKENS):
            errors.append(f"sealed_token_exposure:{fixture_id}")
        if receipt.get("prompt_sha256") != sha256_text(prompt):
            errors.append(f"sealed_prompt_hash:{fixture_id}")
        if receipt.get("energy_input_sha256") != sha256_text(canonical_json(scorer_input)):
            errors.append(f"sealed_energy_hash:{fixture_id}")
        truth = {field: row[field] for field in SEALED_FIELDS}
        if receipt.get("sealed_truth_sha256") != sha256_text(canonical_json(truth)):
            errors.append(f"sealed_truth_hash:{fixture_id}")
    return errors


def _gate(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one exact precondition receipt with no inferred repair."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failure into the stable automation summary shape."""

    if failure is None:
        return {
            "failed_check": None,
            "expected_value": "all_fixture_integrity_checks_pass",
            "observed_value": "all_fixture_integrity_checks_pass",
            "passed": True,
        }
    return {
        "failed_check": failure["check"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
        "passed": False,
    }


def _resolved_source_paths(
    root: Path, overrides: Mapping[str, Path] | None = None
) -> dict[str, Path]:
    """Resolve the frozen source roster, allowing tests to replace one input."""

    paths = {name: root / relative for name, relative in SOURCE_PATHS.items()}
    if overrides:
        paths.update({name: Path(path) for name, path in overrides.items()})
    return paths


def _preconditions(
    root: Path,
    paths: Mapping[str, Path],
    result_path: Path,
    *,
    run_date: str = RUN_DATE,
) -> tuple[list[JsonDict], JsonDict | None, dict[str, str], JsonDict | None]:
    """Check path, byte, license, and Exp7138 integrity in fail-first order."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    date_row = _gate("run_date", RUN_DATE, run_date, run_date == RUN_DATE)
    checks.append(date_row)
    if not date_row["passed"]:
        return checks, date_row, hashes, None
    for name in SOURCE_PATHS:
        path = paths[name]
        passed = path.is_file()
        row = _gate(
            f"{name}_path",
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
            row = _gate(f"{name}_path", "readable_file", "missing_or_unreadable", False)
            checks[-1] = row
            return checks, row, hashes, None
    for name, expected in PINNED_HASHES.items():
        observed = hashes[name]
        row = _gate(f"{name}_hash", expected, observed, observed == expected)
        checks.append(row)
        if not row["passed"]:
            return checks, row, hashes, None

    license_text = paths["ragtruth_license"].read_text(encoding="utf-8")
    expected_license = {"spdx_id": "MIT", "upstream": "ParticleMedia/RAGTruth"}
    observed_license = {
        "spdx_id": "MIT" if license_text.startswith("MIT License") else "unknown",
        "upstream": "ParticleMedia/RAGTruth"
        if "Copyright (c) 2023 Particle Media" in license_text
        else "unknown",
    }
    row = _gate("ragtruth_license", expected_license, observed_license, observed_license == expected_license)
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, None

    constraint_text = paths["constraint_spec"].read_text(encoding="utf-8")
    row = _gate(
        "constraint_spec_requirement",
        "REQ-VERIFY-7158",
        "REQ-VERIFY-7158" if "REQ-VERIFY-7158" in constraint_text else "missing",
        "REQ-VERIFY-7158" in constraint_text,
    )
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, None
    try:
        upstream = json.loads(paths["exp7138_artifact"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row = _gate("exp7138_artifact_json", "valid_json_object", type(exc).__name__, False)
        checks.append(row)
        return checks, row, hashes, None
    valid_shape = isinstance(upstream, dict)
    row = _gate("exp7138_artifact_shape", "json_object", type(upstream).__name__, valid_shape)
    checks.append(row)
    if not valid_shape:
        return checks, row, hashes, None
    from carnot import experiment_7138_v627_relational_fixture as exp7138

    prior_errors = exp7138.validate_artifact(upstream)
    row = _gate("exp7138_fixture_validation", [], prior_errors, not prior_errors)
    checks.append(row)
    if prior_errors:
        return checks, row, hashes, upstream
    row = _gate(
        "exp7138_fixture_ready_score",
        1,
        upstream.get("source_grounding_fixture_ready_score"),
        upstream.get("source_grounding_fixture_ready_score") == 1,
    )
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, upstream
    families = Counter(str(item.get("source_family")) for item in upstream.get("rows", []))
    expected_families = {family: 18 for family in SOURCE_FAMILIES}
    row = _gate("exp7138_source_families", expected_families, dict(families), dict(families) == expected_families)
    checks.append(row)
    if not row["passed"]:
        return checks, row, hashes, upstream
    output_observed = {
        "path": str(result_path.resolve()),
        "schema_complete_first_write": result_path.is_file(),
        "atomic_terminal_write": True,
    }
    row = _gate(
        "output_path",
        {"schema_complete_first_write": True, "atomic_terminal_write": True},
        output_observed,
        output_observed["schema_complete_first_write"],
    )
    checks.append(row)
    return checks, None if row["passed"] else row, hashes, upstream


def _base_artifact(run_date: str) -> JsonDict:
    """Return the complete first-write schema before any input is inspected."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "source_family_rows": [],
        "entity_evidence_rows": [],
        "perturbation_rows": [],
        "split_rows": [],
        "sealed_field_rows": [],
        "energy_term_contract": {},
        "mutation_test_rows": [],
        "frozen_fixture_ids": [],
        "counterfactual_fixture_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(
            _gate("fixture_build_complete", True, False, False)
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_counterfactual_fixture_build",
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
    """Convert the already-persisted running shape into an exact terminal block."""

    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": checks,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": duration_s,
            "source_artifact_hashes": dict(hashes),
            "counterfactual_fixture_ready_score": 0,
            "gate_check_summary": _gate_summary(failure),
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failure['check']}_no_fixture_run",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_errors(materialized: Mapping[str, Any], upstream: Mapping[str, Any]) -> list[str]:
    """Check all readiness gates without trusting producer summary counters."""

    rows = list(materialized["rows"])
    errors: list[str] = []
    if len(rows) != BASE_COUNT * len(CONDITIONS):
        errors.append("row_count_mismatch")
    if Counter(str(row["condition"]) for row in rows) != Counter(
        {condition: BASE_COUNT for condition in CONDITIONS}
    ):
        errors.append("condition_count_mismatch")
    if any(row["support_label"] != CONDITION_LABELS[str(row["condition"])] for row in rows):
        errors.append("exact_label_mismatch")
    errors.extend(span_errors(rows, materialized["entity_evidence_rows"], upstream))
    errors.extend(split_errors(rows, materialized["split_rows"]))
    errors.extend(sealing_errors(rows, materialized["sealed_field_rows"]))
    mutations = list(materialized["mutation_test_rows"])
    if len(mutations) != BASE_COUNT * (len(CONDITIONS) - 1) or not all(
        row.get("passed") is True for row in mutations
    ):
        errors.append("mutation_tests_mismatch")
    if materialized["energy_term_contract"] != freeze_energy_contract(rows):
        errors.append("energy_term_contract_mismatch")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    *,
    result_path: Path | None = None,
    source_paths: Mapping[str, Path] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Write first, gate inputs, build exact rows, and atomically finish."""

    started = time.monotonic()
    root = Path(root).resolve()
    output_path = root / RESULT_PATH if result_path is None else Path(result_path)
    paths = _resolved_source_paths(root, source_paths)

    _progress(0, "start", "schema_complete_first_write")
    artifact = _base_artifact(run_date)
    atomic_write_json(output_path, artifact, allow_override=False, sort_keys=True)
    _progress(0, "end", "schema_complete_first_write")

    _progress(1, "start", "preconditions_and_license")
    checks, failure, hashes, upstream = _preconditions(
        root, paths, output_path, run_date=run_date
    )
    _progress(1, "end", "preconditions_and_license")
    measured = duration_s if duration_s is not None else time.monotonic() - started
    if failure is not None:
        blocked = _blocked_artifact(artifact, checks, failure, hashes, measured)
        _progress(8, "start", "blocked_terminal_write")
        atomic_write_json(output_path, blocked, allow_override=False, sort_keys=True)
        _progress(8, "end", "blocked_terminal_write")
        return blocked
    assert upstream is not None

    _progress(2, "start", "exp7138_source_replay")
    source_count = len(upstream["rows"])
    _progress(2, "end", f"exp7138_source_replay rows={source_count}")
    _progress(3, "start", "paired_rows_and_spans")
    materialized = materialize_fixture(upstream)
    _progress(3, "end", f"paired_rows_and_spans rows={len(materialized['rows'])}")
    _progress(4, "start", "frozen_splits_and_sealing")
    split_and_seal_errors = split_errors(materialized["rows"], materialized["split_rows"])
    split_and_seal_errors += sealing_errors(
        materialized["rows"], materialized["sealed_field_rows"]
    )
    _progress(4, "end", f"frozen_splits_and_sealing errors={len(split_and_seal_errors)}")
    _progress(5, "start", "calibration_only_energy_contract")
    materialized["energy_term_contract"] = freeze_energy_contract(materialized["rows"])
    _progress(5, "end", "calibration_only_energy_contract")
    _progress(6, "start", "per_unit_mutation_tests")
    mutation_failures = sum(not row["passed"] for row in materialized["mutation_test_rows"])
    _progress(6, "end", f"per_unit_mutation_tests failures={mutation_failures}")
    _progress(7, "start", "fixture_benchmark")
    print("exp7158 subprocess_start fixture_benchmark", flush=True)  # pragma: no cover
    benchmark_errors = _fixture_errors(materialized, upstream)
    print("exp7158 subprocess_end fixture_benchmark", flush=True)  # pragma: no cover
    _progress(7, "end", f"fixture_benchmark errors={len(benchmark_errors)}")

    all_errors = split_and_seal_errors + benchmark_errors
    if all_errors:
        failure = _gate(
            "fixture_integrity",
            "all_fixture_integrity_checks_pass",
            all_errors[0],
            False,
        )
        checks.append(failure)
        blocked = _blocked_artifact(artifact, checks, failure, hashes, measured)
        _progress(8, "start", "blocked_terminal_write")
        atomic_write_json(output_path, blocked, allow_override=False, sort_keys=True)
        _progress(8, "end", "blocked_terminal_write")
        return blocked

    artifact.update(materialized)
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": checks,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "duration_s": measured,
            "source_artifact_hashes": hashes,
            "counterfactual_fixture_ready_score": 1,
            "gate_check_summary": _gate_summary(None),
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_counterfactual_fixture_ready_no_verifier_value_claim",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(8, "start", "complete_terminal_atomic_write")
    atomic_write_json(output_path, artifact, allow_override=False, sort_keys=True)
    _progress(8, "end", "complete_terminal_atomic_write")
    return artifact


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    *,
    root: Path | None = None,
) -> list[str]:
    """Cold-check schema, sources, deterministic replay, seals, and terminal state."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
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
    if not isinstance(artifact.get("duration_s"), (int, float)) or artifact.get("duration_s") < 0:
        errors.append("duration_s_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
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
        if artifact.get("counterfactual_fixture_ready_score") != 0:
            errors.append("blocked_readiness_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
        return list(dict.fromkeys(errors))

    if artifact.get("status") != "complete":
        errors.append("complete_status_mismatch")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("complete_inference_substrate_class_mismatch")
    if artifact.get("counterfactual_fixture_ready_score") != 1:
        errors.append("counterfactual_fixture_ready_score_mismatch")
    if artifact.get("gate_check_summary") != _gate_summary(None):
        errors.append("complete_gate_check_summary_mismatch")
    if artifact.get("verdict_class") != "positive":
        errors.append("verdict_class_mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_positive"):
        errors.append("honest_verdict_mismatch")

    repo = find_repo_root() if root is None else Path(root).resolve()
    paths = _resolved_source_paths(repo)
    if all(path.is_file() for path in paths.values()):
        observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
        if artifact.get("source_artifact_hashes") != observed_hashes:
            errors.append("source_artifact_hashes_mismatch")
        try:
            upstream = json.loads(paths["exp7138_artifact"].read_text(encoding="utf-8"))
            expected = materialize_fixture(upstream)
            for field in (
                "rows",
                "source_family_rows",
                "entity_evidence_rows",
                "perturbation_rows",
                "split_rows",
                "sealed_field_rows",
                "energy_term_contract",
                "mutation_test_rows",
                "frozen_fixture_ids",
            ):
                if artifact.get(field) != expected[field]:
                    errors.append(f"{field}_mismatch")
            errors.extend(_fixture_errors(artifact, upstream))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"independent_replay_failed:{type(exc).__name__}")
    else:
        errors.append("source_artifact_missing")
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fixed-date fixture or validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(7, "start", "artifact_validator")
        print("exp7158 subprocess_start artifact_validator", flush=True)
        errors = validate_artifact(args.validate)
        print("exp7158 subprocess_end artifact_validator", flush=True)
        _progress(7, "end", f"artifact_validator errors={len(errors)}")
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if not re.fullmatch(r"[0-9]{8}", args.date) or args.date != RUN_DATE:
        return 2
    root = find_repo_root()
    artifact = build_artifact(root, args.date, result_path=args.result_path)
    errors = validate_artifact(artifact, root=root)
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "counterfactual_fixture_ready_score": artifact[
                    "counterfactual_fixture_ready_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "valid": not errors,
                "errors": errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if errors or artifact["verdict_class"] != "positive" else 0


if __name__ == "__main__":  # pragma: no cover - use the thin experiment wrapper.
    raise SystemExit(main())
