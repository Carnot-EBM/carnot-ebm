"""Qualify model-produced relations without running or repairing a model.

Spec refs: REQ-VERIFY-6901 and SCENARIO-VERIFY-6901-*.

The reducer freezes thresholds on calibration labels. It opens held labels once
only after all public-source admission checks pass. Clingo remains the exact
authority, so a successful result is circular rather than an independent moat.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot import asp_energy
from carnot import experiment_6888_independent_relation_qualification as prior
from carnot import experiment_6900_authentic_anchored_relation_corpus as acquisition


JsonDict = dict[str, Any]
Solver = Callable[[asp_energy.ASPProgram], list[list[str]]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6901_independent_model_relation_qualification.json")
EXP6274_PATH = Path("results/experiment_6274_asp_energy_semantic_compiler.json")
EXP6886_PATH = Path("results/experiment_6886_enoki_exact_relation_fixture.json")
EXP6900_PATH = Path("results/experiment_6900_authentic_anchored_relation_corpus.json")
COMPILER_PATH = Path("python/carnot/asp_energy.py")
CALIBRATION_SIDECAR_PATH = prior.CALIBRATION_SIDECAR_PATH
HELD_SIDECAR_PATH = prior.HELD_SIDECAR_PATH
SIDECAR_SCHEMA = prior.SIDECAR_SCHEMA
SCHEMA = "carnot.exp6901.independent_model_relation_qualification.v1"
INFERENCE_SUBSTRATE = "fresh_process_sealed_relation_reduction_no_llm"
RANDOM_SEED = 6901
SOLVER_TIMEOUT_S = 2.0
MINIMUM_MODEL_EVENTS = 90

PROPOSAL_ARMS = acquisition.PROPOSAL_ARMS
MODEL_ARMS = (*acquisition.GGUF_ARMS, acquisition.ENOKI_ARM)
GGUF_SEEDS = acquisition.SEEDS

EXPECTED_HASHES = {
    "exp6274": "sha256:b02c88963c4815aa0e26d451ffd60fdd9f1014d32e76f638592ac114c611e96b",
    "exp6886": "sha256:602250fbfe172f08458ea279787d992e89835f12005ba6ef59ec02f3b411d500",
    "exp6900": "sha256:beb442dfa3743bc3271150eb88d35cf0e31ac8b657e00664ed143611ed7d0c0c",
    "compiler": "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1",
    "calibration_sidecar": prior.EXPECTED_HASHES["calibration_sidecar"],
    "held_sidecar": prior.EXPECTED_HASHES["held_sidecar"],
}

EXPECTED_EXP6900_SOURCE_HASHES = {
    "canary_prompt_manifest": {
        "sha256": "sha256:82bcb73b549baceae597bb5cded834d7f61197629bb4c0d0a6503d1bddcb7181"
    },
    "corpus_sources": {
        "sha256": "sha256:3190e7f7e8ba85ef82d91fc394bee154068bb0c92f2d6da147c53c4a5fb74ab7"
    },
    "exp6886": {
        "path": "results/experiment_6886_enoki_exact_relation_fixture.json",
        "sha256": EXPECTED_HASHES["exp6886"],
    },
    "exp6899": {
        "path": "results/experiment_6899_live_relation_acquisition_canary.json",
        "sha256": "sha256:7c24282585cf771a56af627d0ed42e31f081d4a014c3ecc55a4f33d5f58dbb82",
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "adversarial_admission_rows",
    "sealed_sidecar_hashes",
    "frozen_thresholds",
    "rows",
    "span_metric_rows",
    "tuple_metric_rows",
    "parse_coverage_rows",
    "abstention_rows",
    "family_rows",
    "perturbation_rows",
    "asp_compilation_rows",
    "solver_parity_rows",
    "completeness_blind_spot_rows",
    "reported_vs_recomputed_metrics",
    "independent_solver_receipts",
    "held_leakage_count",
    "model_eligible_arm_rows",
    "rule_control_rows",
    "qualified_model_relation_event_count",
    "model_relation_qualification_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why its evidence exists.",
    "preconditions_checked": "Admission failures stop the reducer before held labels open.",
    "inference_substrate": "This exact value records that no model inference ran.",
    "duration_s": "Measured wall time proves that the reducer executed.",
    "source_artifact_hashes": "Exact hashes bind all public frozen inputs.",
    "adversarial_admission_rows": "Fresh flags prevent quarantined acquisition from qualifying.",
    "sealed_sidecar_hashes": "Exact hashes bind calibration and held authority.",
    "frozen_thresholds": "Calibration-only thresholds prevent held-set tuning.",
    "rows": "Per-relation rows retain successes, errors, omissions, and exact checks.",
    "span_metric_rows": "Span rows score UTF-8 source grounding separately.",
    "tuple_metric_rows": "Tuple rows expose precision, recall, and duplicate credit.",
    "parse_coverage_rows": "Coverage keeps every frozen cell in the denominator.",
    "abstention_rows": "Abstention cost remains separate from tuple precision.",
    "family_rows": "Family floors prevent easy families from hiding weak families.",
    "perturbation_rows": "Perturbation floors preserve the weakest robustness slice.",
    "asp_compilation_rows": "Compilation rows expose unsupported atoms and errors.",
    "solver_parity_rows": "Exact rows separate compiler parity from held validity.",
    "completeness_blind_spot_rows": "Missed proposals cannot become verifier-completeness claims.",
    "reported_vs_recomputed_metrics": "Row replay detects aggregate drift.",
    "independent_solver_receipts": "Solver receipts retain calls, timeouts, and disagreements.",
    "held_leakage_count": "Zero proves held formal content did not enter proposal inputs.",
    "model_eligible_arm_rows": "Only GGUF and Enoki arms can satisfy readiness.",
    "rule_control_rows": "The lexical rule stays visible but cannot satisfy model gates.",
    "qualified_model_relation_event_count": "Only exact-admitted model events feed downstream work.",
    "model_relation_qualification_ready_score": "One requires a model arm and at least 90 exact events.",
    "random_seed": "A fixed reducer identity supports deterministic replay.",
    "reproducibility_checksum": "One digest detects silent terminal artifact drift.",
    "gate_check_summary": "Expected and observed values make each block actionable.",
    "verifier_is_oracle": "True discloses that exact execution defines validity.",
    "verdict_class": "The closed class prevents an oracle result from claiming positive.",
    "honest_verdict": "A complete prefix marks a terminal result.",
}

QualificationError = prior.QualificationError
HeldSidecarReader = prior.HeldSidecarReader
canonical_json = prior.canonical_json
sha256_bytes = prior.sha256_bytes
sha256_text = prior.sha256_text
sha256_json = prior.sha256_json
sha256_file = prior.sha256_file
gate_check = prior.gate_check
gate_summary = prior.gate_summary
check_exact_hashes = prior.check_exact_hashes


def verify_artifact(path: str) -> JsonDict:
    """Load the on-disk adversarial verifier without a package-path assumption."""

    loader = SourceFileLoader(
        "carnot_exp6901_adversarial_verify", str(REPO_ROOT / "scripts/adversarial_verify.py")
    )
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module.verify_artifact(path)


def adversarial_admission_rows(
    source: Mapping[str, Any], report: Mapping[str, Any]
) -> list[JsonDict]:
    """Turn the source stamp and fresh verifier report into exact gates."""

    flags = [row for row in report.get("flags", []) if isinstance(row, Mapping)]
    critical = [row for row in flags if str(row.get("severity", "")).lower() == "critical"]
    return [
        gate_check("unflagged_exp6900", False, source.get("flagged_adversarial") is True),
        gate_check("fresh_adversarial_critical_count", 0, len(critical)),
        {
            **gate_check("fresh_adversarial_verifier_loaded", True, report.get("loaded") is True),
            "gate_version": report.get("gate_version"),
            "flags": [deepcopy(dict(row)) for row in flags],
        },
    ]


def _expected_cell_identities(
    sources: Sequence[Mapping[str, Any]], arms: Sequence[str], gguf_seeds: Sequence[int]
) -> set[str]:
    identities: set[str] = set()
    for source in sources:
        fixture_id = str(source["fixture_id"])
        for arm in arms:
            if arm.startswith("gguf:"):
                model_id = arm.removeprefix("gguf:")
                identities.update(f"{model_id}::{int(seed)}::{fixture_id}" for seed in gguf_seeds)
            else:
                identities.add(f"{arm}::deterministic::{fixture_id}")
    return identities


def _cell_identity_valid(cell: Mapping[str, Any], gguf_seeds: Sequence[int]) -> bool:
    arm = str(cell.get("arm", ""))
    fixture_id = str(cell.get("fixture_id", ""))
    if arm.startswith("gguf:"):
        model_id = arm.removeprefix("gguf:")
        seed = cell.get("seed")
        return (
            cell.get("hf_id") == model_id
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in gguf_seeds
            and cell.get("cell_identity") == f"{model_id}::{seed}::{fixture_id}"
        )
    return (
        arm in {acquisition.ENOKI_ARM, acquisition.RULE_ARM, "rule:anchored_lexical_v1"}
        and cell.get("hf_id") in {None, ""}
        and cell.get("seed") is None
        and cell.get("cell_identity") == f"{arm}::deterministic::{fixture_id}"
    )


def validate_acquisition_matrix(
    cells: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    gguf_seeds: Sequence[int],
) -> JsonDict:
    """Validate all frozen record, arm, model, seed, and terminal identities."""

    source_by_id = {str(row["fixture_id"]): row for row in sources}
    identities = [str(row.get("cell_identity")) for row in cells]
    expected = _expected_cell_identities(sources, arms, gguf_seeds)
    identity_valid = all(_cell_identity_valid(row, gguf_seeds) for row in cells)
    content_valid = True
    for cell in cells:
        source = source_by_id.get(str(cell.get("fixture_id")))
        if source is None or any(
            (
                cell.get("terminal") is not True,
                cell.get("source_text_hash") != source.get("source_text_hash"),
                cell.get("split") != source.get("split"),
                cell.get("family") != source.get("family"),
                cell.get("group_id") != source.get("group_id"),
            )
        ):
            content_valid = False
            break
    checks = [
        gate_check("unique_cell_identity", len(identities), len(set(identities))),
        gate_check("complete_terminal_cells", len(expected), len(identities)),
        gate_check("record_arm_model_seed_identity", True, identity_valid),
        gate_check("cell_identity_set", expected, set(identities)),
        gate_check("terminal_cell_content", True, content_valid),
    ]
    return gate_summary(checks)


def _raw_output(cell: Mapping[str, Any]) -> str:
    if "raw_output" in cell:
        return str(cell.get("raw_output", ""))
    try:
        return base64.b64decode(str(cell.get("raw_output_b64", "")), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def detect_held_leakage(
    proposal_artifact: Mapping[str, Any], held_payload: Mapping[str, Any]
) -> list[JsonDict]:
    """Check only proposal prompts and raw outputs for held formal content."""

    prompt_values: list[str] = []

    def collect_strings(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                collect_strings(item)
        elif isinstance(value, list):
            for item in value:
                collect_strings(item)
        elif isinstance(value, str):
            prompt_values.append(value)

    collect_strings(proposal_artifact.get("prompt_manifest", {}))
    view = {
        "prompt_manifest": {},
        "rows": [
            {
                "cell_identity": "prompt_manifest",
                "raw_output": "\n".join(prompt_values),
            },
            *[
                {
                    "cell_identity": row.get("cell_identity"),
                    "raw_output": _raw_output(row),
                }
                for row in proposal_artifact.get("cell_manifest", [])
                if isinstance(row, Mapping)
            ],
        ],
    }
    return prior.detect_held_leakage(view, held_payload)


def _identity_metadata(cell: Mapping[str, Any]) -> JsonDict:
    arm = str(cell["arm"])
    return {
        "arm": arm,
        "cell_identity": str(cell["cell_identity"]),
        "model_id": (
            str(cell.get("hf_id"))
            if cell.get("hf_id")
            else arm
            if arm == acquisition.ENOKI_ARM
            else None
        ),
        "seed_id": cell.get("seed") if cell.get("seed") is not None else "deterministic",
        "record_id": str(cell["fixture_id"]),
        "family": str(cell["family"]),
    }


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str],
    extra: str,
) -> list[JsonDict]:
    result = []
    for arm in arms:
        selected = [row for row in rows if metadata[str(row["arm"])]["arm"] == arm]
        totals = [
            sum(int(row.get(name, 0)) for row in selected)
            for name in ("true_positive", "false_positive", "false_negative")
        ]
        result.append(
            {
                "arm": arm,
                **prior._metrics(*totals),
                extra: sum(int(row.get(extra, 0)) for row in selected),
            }
        )
    return result


def _subgroup_metrics(
    tuple_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str],
    formal_by_id: Mapping[str, Mapping[str, Any]],
    field: str,
) -> list[JsonDict]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tuple_rows:
        meta = metadata[str(row["arm"])]
        value = (
            meta["family"]
            if field == "family"
            else str(formal_by_id[meta["record_id"]].get("expected_case", "unknown"))
        )
        grouped[(str(meta["arm"]), str(value))].append(row)
    result = []
    for arm in arms:
        values = sorted(value for candidate, value in grouped if candidate == arm)
        for value in values:
            selected = grouped[(arm, value)]
            totals = [
                sum(int(row.get(name, 0)) for row in selected)
                for name in ("true_positive", "false_positive", "false_negative")
            ]
            result.append(
                {"arm": arm, field: value, "cell_count": len(selected), **prior._metrics(*totals)}
            )
    return result


def score_partition(
    *,
    cells: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    solver: Solver,
    solver_timeout_s: float,
) -> JsonDict:
    """Score one split and retain model and seed identity on every exact row."""

    synthetic_cells = []
    metadata: dict[str, JsonDict] = {}
    for cell in cells:
        identity = str(cell["cell_identity"])
        copied = deepcopy(dict(cell))
        copied["arm"] = identity
        synthetic_cells.append(copied)
        metadata[identity] = _identity_metadata(cell)
    base_report = prior.score_partition(
        cells=synthetic_cells,
        sources=sources,
        formal_rows=formal_rows,
        vocabulary=vocabulary,
        arms=sorted(metadata),
        solver=solver,
        solver_timeout_s=solver_timeout_s,
    )
    formal_by_id = {str(row["fixture_id"]): row for row in formal_rows}
    parity_rows = []
    exact_by_identity: dict[str, JsonDict] = {}
    for row in base_report["solver_parity_rows"]:
        identity = str(row["arm"])
        meta = metadata[identity]
        formal = formal_by_id[meta["record_id"]]
        exact_validity = bool(
            row.get("parity") is True
            and row.get("answer_sets_hash") == sha256_json(formal.get("answer_sets", []))
        )
        enriched = {
            **deepcopy(dict(row)),
            **meta,
            "compiler_solver_parity": row.get("parity") is True,
            "exact_validity": exact_validity,
            "held_answer_sets_hash": sha256_json(formal.get("answer_sets", [])),
        }
        parity_rows.append(enriched)
        exact_by_identity[identity] = {
            "status": row.get("status"),
            "compiler_solver_parity": row.get("parity") is True,
            "exact_validity": exact_validity,
        }
    rows = []
    identities_with_rows: set[str] = set()
    for row in base_report["rows"]:
        identity = str(row["arm"])
        identities_with_rows.add(identity)
        rows.append(
            {
                **deepcopy(dict(row)),
                **metadata[identity],
                "exact_check": deepcopy(exact_by_identity[identity]),
            }
        )
    for identity, meta in metadata.items():
        if identity not in identities_with_rows:
            formal = formal_by_id[meta["record_id"]]
            rows.append(
                {
                    **meta,
                    "relation": None,
                    "perturbation": str(formal.get("expected_case", "unknown")),
                    "parse_status": "missing",
                    "outcome": "missing_proposal",
                    "reason": "no_relation_row",
                    "no_headroom": True,
                    "exact_check": deepcopy(exact_by_identity[identity]),
                }
            )
    tuple_rows = _aggregate_metrics(
        base_report["tuple_metric_rows"], metadata, arms, "duplicate_proposal_count"
    )
    span_rows = _aggregate_metrics(
        base_report["span_metric_rows"], metadata, arms, "offset_mismatch_count"
    )
    parse_rows = []
    abstention_rows = []
    for arm in arms:
        parse_selected = [
            row
            for row in base_report["parse_coverage_rows"]
            if metadata[str(row["arm"])]["arm"] == arm
        ]
        abstain_selected = [
            row for row in base_report["abstention_rows"] if metadata[str(row["arm"])]["arm"] == arm
        ]
        cell_count = sum(int(row["cell_count"]) for row in parse_selected)
        parsed_count = sum(int(row["parsed_cell_count"]) for row in parse_selected)
        abstention_count = sum(int(row["abstention_count"]) for row in abstain_selected)
        parse_rows.append(
            {
                "arm": arm,
                "cell_count": cell_count,
                "parsed_cell_count": parsed_count,
                "parse_coverage": parsed_count / cell_count if cell_count else None,
            }
        )
        cost = abstention_count / cell_count if cell_count else None
        abstention_rows.append(
            {
                "arm": arm,
                "cell_count": cell_count,
                "abstention_count": abstention_count,
                "abstention_rate": cost,
                "abstention_cost": cost,
                "timeout_count": sum(int(row["timeout_count"]) for row in abstain_selected),
            }
        )
    compile_rows = [
        {**deepcopy(dict(row)), **metadata[str(row["arm"])]}
        for row in base_report["asp_compilation_rows"]
    ]
    tuple_by_identity = {str(row["arm"]): row for row in base_report["tuple_metric_rows"]}
    blind_spots = []
    for identity, meta in metadata.items():
        missed = int(tuple_by_identity[identity].get("false_negative", 0))
        if missed:
            blind_spots.append(
                {
                    **meta,
                    "missed_gold_relation_count": missed,
                    "verifier_complete": False,
                    "reason": "proposal_false_negative",
                }
            )
    event_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        relation = row.get("relation")
        if (
            row.get("outcome") == "true_positive"
            and isinstance(relation, Mapping)
            and relation.get("span_exact") is True
            and row.get("exact_check", {}).get("exact_validity") is True
        ):
            event_counts[str(row["arm"])] += 1
    return {
        "rows": rows,
        "span_metric_rows": span_rows,
        "tuple_metric_rows": tuple_rows,
        "parse_coverage_rows": parse_rows,
        "abstention_rows": abstention_rows,
        "family_rows": _subgroup_metrics(
            base_report["tuple_metric_rows"], metadata, arms, formal_by_id, "family"
        ),
        "perturbation_rows": _subgroup_metrics(
            base_report["tuple_metric_rows"], metadata, arms, formal_by_id, "perturbation"
        ),
        "asp_compilation_rows": compile_rows,
        "solver_parity_rows": parity_rows,
        "completeness_blind_spot_rows": blind_spots,
        "exact_admitted_event_count_by_arm": dict(event_counts),
        "independent_solver_receipts": deepcopy(base_report["independent_solver_receipts"]),
    }


def _by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["arm"]): row for row in rows}


def freeze_thresholds(calibration: Mapping[str, Any]) -> JsonDict:
    """Select one calibration reference and freeze all held thresholds."""

    tuples = _by_arm(calibration.get("tuple_metric_rows", []))
    spans = _by_arm(calibration.get("span_metric_rows", []))
    coverage = _by_arm(calibration.get("parse_coverage_rows", []))
    abstention = _by_arm(calibration.get("abstention_rows", []))
    if not tuples:
        raise QualificationError("calibration_metrics_missing")

    def rank(arm: str) -> tuple[float, float, float, float]:
        return (
            float(tuples[arm].get("f1") or 0.0),
            float(spans.get(arm, {}).get("f1") or 0.0),
            float(tuples[arm].get("recall") or 0.0),
            float(coverage.get(arm, {}).get("parse_coverage") or 0.0),
        )

    reference = max(sorted(tuples), key=rank)
    family_values = [
        float(row.get("f1") or 0.0)
        for row in calibration.get("family_rows", [])
        if row.get("arm") == reference
    ]
    perturbation_values = [
        float(row.get("f1") or 0.0)
        for row in calibration.get("perturbation_rows", [])
        if row.get("arm") == reference
    ]
    exact_values = [
        bool(row.get("exact_validity"))
        for row in calibration.get("solver_parity_rows", [])
        if row.get("arm") == reference
    ]
    return {
        "threshold_source_split": "calibration",
        "reference_arm": reference,
        "minimum_span_f1": float(spans[reference]["f1"]),
        "minimum_tuple_precision": float(tuples[reference].get("precision") or 0.0),
        "minimum_tuple_recall": float(tuples[reference].get("recall") or 0.0),
        "minimum_parse_coverage": float(coverage[reference].get("parse_coverage") or 0.0),
        "minimum_exact_validity": sum(exact_values) / len(exact_values),
        "minimum_family_floor": min(family_values),
        "minimum_perturbation_floor": min(perturbation_values),
        "maximum_abstention_cost": float(abstention[reference].get("abstention_cost") or 0.0),
        "minimum_exact_admitted_events": MINIMUM_MODEL_EVENTS,
        "calibration_metrics_sha256": sha256_json(
            {
                key: calibration.get(key, [])
                for key in (
                    "span_metric_rows",
                    "tuple_metric_rows",
                    "parse_coverage_rows",
                    "abstention_rows",
                    "family_rows",
                    "perturbation_rows",
                    "solver_parity_rows",
                )
            }
        ),
    }


def evaluate_model_readiness(
    held: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    event_count_by_arm: Mapping[str, int],
    *,
    model_arms: Sequence[str],
) -> JsonDict:
    """Apply held thresholds while excluding rule controls from outgoing gates."""

    tuples = _by_arm(held.get("tuple_metric_rows", []))
    spans = _by_arm(held.get("span_metric_rows", []))
    coverage = _by_arm(held.get("parse_coverage_rows", []))
    abstention = _by_arm(held.get("abstention_rows", []))
    rows = []
    for arm in sorted(tuples):
        family_values = [
            float(row.get("f1") or 0.0)
            for row in held.get("family_rows", [])
            if row.get("arm") == arm
        ]
        perturbation_values = [
            float(row.get("f1") or 0.0)
            for row in held.get("perturbation_rows", [])
            if row.get("arm") == arm
        ]
        exact_values = [
            bool(row.get("exact_validity"))
            for row in held.get("solver_parity_rows", [])
            if row.get("arm") == arm
        ]
        observed = {
            "span_f1": spans.get(arm, {}).get("f1"),
            "tuple_precision": tuples[arm].get("precision"),
            "tuple_recall": tuples[arm].get("recall"),
            "parse_coverage": coverage.get(arm, {}).get("parse_coverage"),
            "exact_validity": sum(exact_values) / len(exact_values) if exact_values else None,
            "family_floor": min(family_values) if family_values else None,
            "perturbation_floor": min(perturbation_values) if perturbation_values else None,
            "abstention_cost": abstention.get(arm, {}).get("abstention_cost"),
        }
        floors = {
            "span_f1": thresholds["minimum_span_f1"],
            "tuple_precision": thresholds["minimum_tuple_precision"],
            "tuple_recall": thresholds["minimum_tuple_recall"],
            "parse_coverage": thresholds["minimum_parse_coverage"],
            "exact_validity": thresholds["minimum_exact_validity"],
            "family_floor": thresholds["minimum_family_floor"],
            "perturbation_floor": thresholds["minimum_perturbation_floor"],
        }
        failed = [
            name
            for name, floor in floors.items()
            if observed[name] is None or float(observed[name]) < float(floor)
        ]
        if observed["abstention_cost"] is None or float(observed["abstention_cost"]) > float(
            thresholds["maximum_abstention_cost"]
        ):
            failed.append("abstention_cost")
        model_eligible = arm in set(model_arms)
        event_count = int(event_count_by_arm.get(arm, 0))
        if model_eligible and event_count < int(thresholds["minimum_exact_admitted_events"]):
            failed.append("minimum_exact_admitted_events")
        rows.append(
            {
                "arm": arm,
                "model_eligible": model_eligible,
                "observed": observed,
                "thresholds": deepcopy(dict(thresholds)),
                "failed_thresholds": failed,
                "threshold_passed": not [
                    name for name in failed if name != "minimum_exact_admitted_events"
                ],
                "passed": model_eligible and not failed,
                "exact_admitted_event_count": event_count,
            }
        )
    model_rows = [row for row in rows if row["model_eligible"]]
    rule_rows = [row for row in rows if not row["model_eligible"]]
    passing = [row for row in model_rows if row["passed"]]
    return {
        "model_eligible_arm_rows": model_rows,
        "rule_control_rows": rule_rows,
        "qualified_model_relation_event_count": sum(
            int(row["exact_admitted_event_count"]) for row in passing
        ),
        "model_relation_qualification_ready_score": int(bool(passing)),
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
        }
    )


def _attach_principles(artifact: JsonDict) -> None:
    principles = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves required Exp6901 evidence.")
        for key in artifact
    }
    for row in artifact.get("gate_check_summary", {}).get("checks", []):
        principles[f"gate:{row.get('check')}"] = (
            "This exact expectation prevents inadmissible evidence from reaching held scoring."
        )
    artifact["field_principles"] = principles


def blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
    adversarial_admission_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the full terminal schema without opening held labels."""

    summary = gate_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6901,
        "run_date": date,
        "status": "blocked",
        "field_principles": {},
        "preconditions_checked": {
            "held_sidecar_open_count": 0,
            "no_model_inference": True,
            "no_output_repair": True,
            "gate_check_summary": summary,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "adversarial_admission_rows": [deepcopy(dict(row)) for row in adversarial_admission_rows],
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "frozen_thresholds": {},
        "rows": [],
        "span_metric_rows": [],
        "tuple_metric_rows": [],
        "parse_coverage_rows": [],
        "abstention_rows": [],
        "family_rows": [],
        "perturbation_rows": [],
        "asp_compilation_rows": [],
        "solver_parity_rows": [],
        "completeness_blind_spot_rows": [],
        "reported_vs_recomputed_metrics": {
            "reported_ready_score": 0,
            "recomputed_ready_score": 0,
            "reported_qualified_event_count": 0,
            "recomputed_qualified_event_count": 0,
            "agreement": True,
        },
        "independent_solver_receipts": {},
        "held_leakage_count": 0,
        "model_eligible_arm_rows": [],
        "rule_control_rows": [],
        "qualified_model_relation_event_count": 0,
        "model_relation_qualification_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_independent_model_relation_qualification",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def reduce_frozen_outputs(
    *,
    date: str,
    proposal_artifact: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    calibration_sidecar: Mapping[str, Any],
    held_reader: HeldSidecarReader,
    vocabulary: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    model_arms: Sequence[str],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
    adversarial_admission_rows: Sequence[Mapping[str, Any]],
    solver: Solver,
    solver_timeout_s: float,
    duration_s: float,
    precondition_checks: Sequence[Mapping[str, Any]] = (),
    minimum_model_events: int = MINIMUM_MODEL_EVENTS,
) -> JsonDict:
    """Freeze calibration, open held once, and build the terminal artifact."""

    cells = [row for row in proposal_artifact.get("cell_manifest", []) if isinstance(row, Mapping)]
    calibration = score_partition(
        cells=[row for row in cells if row.get("split") == "calibration"],
        sources=[row for row in sources if row.get("split") == "calibration"],
        formal_rows=calibration_sidecar.get("rows", []),
        vocabulary=vocabulary,
        arms=arms,
        solver=solver,
        solver_timeout_s=solver_timeout_s,
    )
    thresholds = freeze_thresholds(calibration)
    thresholds["minimum_exact_admitted_events"] = int(minimum_model_events)
    if held_reader.open_count != 0:
        raise QualificationError("held_opened_before_threshold_freeze")
    held_sidecar = held_reader.open_once()
    leaks = detect_held_leakage(proposal_artifact, held_sidecar)
    held = score_partition(
        cells=[row for row in cells if row.get("split") == "held"],
        sources=[row for row in sources if row.get("split") == "held"],
        formal_rows=held_sidecar.get("rows", []),
        vocabulary=vocabulary,
        arms=arms,
        solver=solver,
        solver_timeout_s=solver_timeout_s,
    )
    eligibility = evaluate_model_readiness(
        held,
        thresholds,
        held["exact_admitted_event_count_by_arm"],
        model_arms=model_arms,
    )
    if leaks:
        eligibility["model_eligible_arm_rows"] = [
            {
                **row,
                "passed": False,
                "failed_thresholds": [*row["failed_thresholds"], "held_leakage"],
            }
            for row in eligibility["model_eligible_arm_rows"]
        ]
        eligibility["qualified_model_relation_event_count"] = 0
        eligibility["model_relation_qualification_ready_score"] = 0
    ready = int(eligibility["model_relation_qualification_ready_score"])
    qualified = int(eligibility["qualified_model_relation_event_count"])
    checks = [
        *precondition_checks,
        gate_check("held_sidecar_open_count", 1, held_reader.open_count),
        gate_check("held_leakage_count", 0, len(leaks)),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6901,
        "run_date": date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": {
            "calibration_scored_before_held_open": True,
            "held_sidecar_open_count": held_reader.open_count,
            "no_model_inference": True,
            "no_output_repair": True,
            "gate_check_summary": gate_summary(precondition_checks),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "adversarial_admission_rows": [deepcopy(dict(row)) for row in adversarial_admission_rows],
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "frozen_thresholds": thresholds,
        "rows": held["rows"],
        "span_metric_rows": held["span_metric_rows"],
        "tuple_metric_rows": held["tuple_metric_rows"],
        "parse_coverage_rows": held["parse_coverage_rows"],
        "abstention_rows": held["abstention_rows"],
        "family_rows": held["family_rows"],
        "perturbation_rows": held["perturbation_rows"],
        "asp_compilation_rows": held["asp_compilation_rows"],
        "solver_parity_rows": held["solver_parity_rows"],
        "completeness_blind_spot_rows": held["completeness_blind_spot_rows"],
        "reported_vs_recomputed_metrics": {
            "reported_ready_score": ready,
            "recomputed_ready_score": ready,
            "reported_qualified_event_count": qualified,
            "recomputed_qualified_event_count": qualified,
            "agreement": True,
        },
        "independent_solver_receipts": held["independent_solver_receipts"],
        "held_leakage_rows": leaks,
        "held_leakage_count": len(leaks),
        "model_eligible_arm_rows": eligibility["model_eligible_arm_rows"],
        "rule_control_rows": eligibility["rule_control_rows"],
        "qualified_model_relation_event_count": qualified,
        "model_relation_qualification_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "disqualified" if leaks else "circular_positive" if ready else "null",
        "honest_verdict": (
            "complete_disqualified_held_relation_leakage"
            if leaks
            else "complete_circular_positive_independent_model_relation_qualification"
            if ready
            else "complete_null_no_model_relation_arm_passed_frozen_thresholds"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay terminal schema, principles, aggregate metrics, and checksum."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    principles = artifact.get("field_principles")
    gate_names = {
        f"gate:{row.get('check')}"
        for row in artifact.get("gate_check_summary", {}).get("checks", [])
        if isinstance(row, Mapping)
    }
    if (
        not isinstance(principles, Mapping)
        or not set(artifact) <= set(principles)
        or not gate_names <= set(principles)
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    reported = artifact.get("reported_vs_recomputed_metrics")
    if not isinstance(reported, Mapping):
        errors.append("reported_vs_recomputed_metrics")
    elif (
        reported.get("agreement") is not True
        or reported.get("reported_ready_score")
        != artifact.get("model_relation_qualification_ready_score")
        or reported.get("recomputed_ready_score")
        != artifact.get("model_relation_qualification_ready_score")
        or reported.get("reported_qualified_event_count")
        != artifact.get("qualified_model_relation_event_count")
        or reported.get("recomputed_qualified_event_count")
        != artifact.get("qualified_model_relation_event_count")
    ):
        errors.append("aggregate_vs_row_disagreement")
    if artifact.get("verdict_class") != "blocked" and artifact.get("frozen_thresholds"):
        event_counts: dict[str, int] = defaultdict(int)
        for row in artifact.get("rows", []):
            relation = row.get("relation") if isinstance(row, Mapping) else None
            if (
                isinstance(row, Mapping)
                and row.get("outcome") == "true_positive"
                and isinstance(relation, Mapping)
                and relation.get("span_exact") is True
                and row.get("exact_check", {}).get("exact_validity") is True
            ):
                event_counts[str(row.get("arm"))] += 1
        model_arms = [str(row.get("arm")) for row in artifact.get("model_eligible_arm_rows", [])]
        recomputed = evaluate_model_readiness(
            artifact, artifact["frozen_thresholds"], event_counts, model_arms=model_arms
        )
        if recomputed["model_relation_qualification_ready_score"] != artifact.get(
            "model_relation_qualification_ready_score"
        ) or recomputed["qualified_model_relation_event_count"] != artifact.get(
            "qualified_model_relation_event_count"
        ):
            errors.append("aggregate_vs_row_disagreement")
    for row in artifact.get("rows", []):
        if not {"arm", "record_id", "relation", "family", "perturbation", "exact_check"} <= set(
            row
        ):
            errors.append("row_schema")
            break
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hash_rows(observed: Mapping[str, str]) -> JsonDict:
    paths = {
        "exp6274": EXP6274_PATH,
        "exp6886": EXP6886_PATH,
        "exp6900": EXP6900_PATH,
        "compiler": COMPILER_PATH,
    }
    return {name: {"path": paths[name].as_posix(), "sha256": observed[name]} for name in paths}


def run_experiment(*, date: str, output_path: Path | str = RESULT_PATH) -> JsonDict:
    """Run fresh public admission and write one terminal Exp6901 artifact."""

    started = time.perf_counter()
    source_paths = {
        "exp6274": REPO_ROOT / EXP6274_PATH,
        "exp6886": REPO_ROOT / EXP6886_PATH,
        "exp6900": REPO_ROOT / EXP6900_PATH,
        "compiler": REPO_ROOT / COMPILER_PATH,
    }
    observed = {
        name: sha256_file(path) if path.is_file() else "missing"
        for name, path in source_paths.items()
    }
    checks = list(
        check_exact_hashes(observed, {name: EXPECTED_HASHES[name] for name in source_paths})[
            "checks"
        ]
    )
    proposal = _read_json(source_paths["exp6900"]) if source_paths["exp6900"].is_file() else {}
    fixture = _read_json(source_paths["exp6886"]) if source_paths["exp6886"].is_file() else {}
    compiler = _read_json(source_paths["exp6274"]) if source_paths["exp6274"].is_file() else {}
    fresh_report = (
        verify_artifact(str(source_paths["exp6900"]))
        if source_paths["exp6900"].is_file()
        else {"loaded": False, "flags": []}
    )
    admission = adversarial_admission_rows(proposal, fresh_report)
    sources = acquisition.reconstruct_source_records()
    matrix = validate_acquisition_matrix(
        proposal.get("cell_manifest", []), sources, PROPOSAL_ARMS, GGUF_SEEDS
    )
    checks.extend(
        [
            gate_check(
                "relation_corpus_complete_score", 1, proposal.get("relation_corpus_complete_score")
            ),
            gate_check(
                "exp6900_source_artifact_hashes",
                EXPECTED_EXP6900_SOURCE_HASHES,
                proposal.get("source_artifact_hashes"),
            ),
            *admission,
            *matrix["checks"],
            gate_check(
                "relation_fixture_ready_score", 1, fixture.get("relation_fixture_ready_score")
            ),
            gate_check("qualified_compiler", 1.0, compiler.get("asp_energy_semantic_ready_score")),
            gate_check("compiler_parity_failure_count", 0, compiler.get("parity_failure_count")),
            gate_check(
                "independent_solver_available",
                True,
                not asp_energy.solver_name_version().endswith(":missing"),
            ),
        ]
    )
    calibration_hash = (
        sha256_file(CALIBRATION_SIDECAR_PATH) if CALIBRATION_SIDECAR_PATH.is_file() else "missing"
    )
    held_hash = sha256_file(HELD_SIDECAR_PATH) if HELD_SIDECAR_PATH.is_file() else "missing"
    checks.extend(
        [
            gate_check(
                "calibration_sidecar_hash", EXPECTED_HASHES["calibration_sidecar"], calibration_hash
            ),
            gate_check("held_sidecar_hash", EXPECTED_HASHES["held_sidecar"], held_hash),
        ]
    )
    source_hashes = _source_hash_rows(observed)
    sidecar_hashes = {
        "calibration": {"path": str(CALIBRATION_SIDECAR_PATH), "sha256": calibration_hash},
        "held": {"path": str(HELD_SIDECAR_PATH), "sha256": held_hash},
    }
    if gate_summary(checks)["passed"]:
        calibration = _read_json(CALIBRATION_SIDECAR_PATH)
        reader = HeldSidecarReader(HELD_SIDECAR_PATH, EXPECTED_HASHES["held_sidecar"])
        artifact = reduce_frozen_outputs(
            date=date,
            proposal_artifact=proposal,
            sources=sources,
            calibration_sidecar=calibration,
            held_reader=reader,
            vocabulary=fixture.get("closed_vocabulary_manifest", {}).get("entries", []),
            arms=PROPOSAL_ARMS,
            model_arms=MODEL_ARMS,
            source_artifact_hashes=source_hashes,
            sealed_sidecar_hashes=sidecar_hashes,
            adversarial_admission_rows=admission,
            solver=asp_energy.solve_with_clingo,
            solver_timeout_s=SOLVER_TIMEOUT_S,
            duration_s=0.0,
            precondition_checks=checks,
        )
        artifact["duration_s"] = time.perf_counter() - started
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    else:
        artifact = blocked_artifact(
            date=date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            source_artifact_hashes=source_hashes,
            sealed_sidecar_hashes=sidecar_hashes,
            adversarial_admission_rows=admission,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise QualificationError("artifact_validation:" + ",".join(errors))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6901 from the required command-line wrapper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260902")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = run_experiment(date=args.date, output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper executes this path.
    raise SystemExit(main())
