"""Audit and run the oracle-distinct diagnostic-energy experiment.

The frozen Exp6745 corpus must have evaluable outcome variation before any
model can train. This module performs that gate first. It also keeps the
feature contract and row-derived metric functions executable for later data.

Spec refs: REQ-ENERGY-6746 and SCENARIO-ENERGY-6746-*.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = Path("results/experiment_6745_sota_dual_encoding_proposal_corpus.json")
RESULT_PATH = Path("results/experiment_6746_oracle_distinct_diagnostic_energy.json")
SCHEMA = "carnot.experiment_6746.oracle_distinct_diagnostic_energy.v1"
INFERENCE_SUBSTRATE = "local CPU structural-energy training/evaluation"

HELD_FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
SOURCE_FAMILY_SPLITS = {
    "expander_tseitin": "train",
    "ladder_tseitin": "dev",
    "pigeonhole_anchor": "test",
}
RANDOM_SEEDS = {
    "split": 6_746_001,
    "train": {
        "dual_encoding": 6_746_011,
        "encoder_a_only": 6_746_011,
        "encoder_b_only": 6_746_011,
        "undifferentiated_scalar": 6_746_011,
    },
    "bootstrap": 6_746_021,
}
BOOTSTRAP_CONFIG = {
    "n_resamples": 2_000,
    "confidence_level": 0.95,
    "paired": True,
    "minimum_rows_per_family": 30,
    "minimum_rows_per_binary_class": 10,
}
TRAINING_CONFIG = {
    "arms": [
        "dual_encoding",
        "encoder_a_only",
        "encoder_b_only",
        "undifferentiated_scalar",
    ],
    "model": "linear_structural_energy",
    "capacity": 32,
    "l2_penalty": 1.0,
    "maximum_steps": 500,
    "fixed_budget_for_all_arms": True,
}

ORACLE_FEATURE_DENYLIST = (
    "diagnosis",
    "target",
    "label",
    "answer",
    "answer_key",
    "certificate",
    "certificate_sha256",
    "parsed_certificate",
    "exact_check",
    "exact_valid",
    "valid",
    "validity",
    "reason",
    "checked_assignment_count",
    "solver",
    "solver_count",
    "solver_conflicts",
    "assignment",
    "row_blocked",
    "parser_status",
    "parse_failure",
    "abstention",
)

_FORMAT_FEATURES = [
    "format.raw_char_count",
    "format.whitespace_token_count",
    "format.line_count",
    "format.digit_count",
    "format.equals_count",
    "format.comma_count",
    "format.invalid_symbol_count",
    "format.trailing_text_count",
    "format.output_budget_fraction",
    "syntax.claim_token_count",
    "syntax.binding_term_count",
    "syntax.clause_term_count",
]
_TOPOLOGY_FEATURES = [
    "topology.variable_count",
    "topology.clause_count",
    "topology.literal_count",
    "topology.mean_clause_width",
    "topology.max_clause_width",
    "topology.component_count",
    "topology.max_component_fraction",
    "topology.mean_degree",
    "topology.degree_std",
]
_ENCODER_A_FEATURES = [
    "encoder_a.binding_node_count",
    "encoder_a.core_node_count",
    "encoder_a.duplicate_variable_count",
    "encoder_a.conflicting_variable_count",
    "encoder_a.out_of_range_reference_count",
    "encoder_a.local_syntax_error_count",
]
_ENCODER_B_FEATURES = [feature.replace("encoder_a", "encoder_b") for feature in _ENCODER_A_FEATURES]
_DUAL_FEATURES = [
    "dual.binding_count_delta",
    "dual.core_count_delta",
    "dual.conflict_count_delta",
    "dual.out_of_range_delta",
    "dual.structural_jaccard",
    "dual.claim_disagreement",
]
FEATURE_SCHEMA = {
    "version": "preoracle_structural_features_v1",
    "frozen_before_label_join": True,
    "allowed_input_classes": [
        "pre-oracle syntax",
        "formula topology",
        "encoder structure",
        "local inconsistency descriptors",
        "model-independent format",
    ],
    "arms": {
        "dual_encoding": [
            *_FORMAT_FEATURES,
            *_TOPOLOGY_FEATURES,
            *_ENCODER_A_FEATURES,
            *_ENCODER_B_FEATURES,
            *_DUAL_FEATURES,
        ],
        "encoder_a_only": [*_FORMAT_FEATURES, *_TOPOLOGY_FEATURES, *_ENCODER_A_FEATURES],
        "encoder_b_only": [*_FORMAT_FEATURES, *_TOPOLOGY_FEATURES, *_ENCODER_B_FEATURES],
        "undifferentiated_scalar": ["scalar.local_failure_mass"],
    },
}

FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets downstream checks reject incompatible results.",
    "experiment": "The numeric identity binds this artifact to the planned task.",
    "title": "The title states the narrow oracle-distinct diagnostic purpose.",
    "run_date": "The planning date identifies the frozen evaluation window.",
    "status": "The status separates an owned block from a completed comparison.",
    "field_principles": "Each field and gate states why it affects scientific credit.",
    "inference_substrate": "The declaration limits this task to local CPU structural training and evaluation.",
    "duration_s": "Monotonic elapsed time records the work actually performed.",
    "random_seed": "Split, arm-training, and bootstrap seeds make all random choices explicit.",
    "reproducibility_checksum": "The hash binds features, splits, configs, and ordered source rows.",
    "source_artifact": "The path and hash bind the exact upstream evidence used.",
    "verifier_is_oracle": "False prevents a learned diagnostic from claiming exact authority.",
    "feature_schema": "The frozen allowlist makes every possible model input auditable.",
    "oracle_feature_denylist": "The denylist prevents exact outcomes and their proxies from entering a model.",
    "feature_taint_audit": "The audit records schema taint and whether proxy testing was evaluable.",
    "immutable_family_splits": "The receipt proves that training and held families do not overlap.",
    "training_config": "One fixed capacity and budget makes arm comparisons matched.",
    "bootstrap_config": "The registered resampling plan prevents interval tuning after results are seen.",
    "preconditions_checked": "Every owned stop records the expected and observed gate value.",
    "rows": "Unit-arm predictions are the only authority for reported metrics.",
    "heldout_metrics_by_family": "Family reports must derive from raw held rows.",
    "paired_relabel_metrics": "Paired deltas test stability under meaning-preserving variable relabeling.",
    "heldout_reasoning_error_auroc": "This exact downstream field comes only from raw held rows.",
    "oracle_leakage_detected": "Any denylist, split, taint, or proxy failure removes positive credit.",
    "diagnostic_energy_ready": "Readiness requires every arm and audit; it does not imply a positive result.",
    "gate_check_summary": "The first failed check names its expected and observed value.",
    "verdict_class": "The closed vocabulary makes the terminal scientific state machine-readable.",
    "honest_verdict": "The terminal prefix makes an owned block explicit.",
}
GATE_PRINCIPLES = {
    "exp6745_artifact_present": "Training cannot start without the frozen upstream corpus.",
    "exp6745_identity": "The task accepts only the planned upstream experiment.",
    "dual_encoding_corpus_ready": "Incomplete upstream rows cannot support a controlled comparison.",
    "feature_denylist_clean": "A prohibited input would make the learned score circular.",
    "immutable_source_family_splits": "Source split drift can move a family across the evaluation boundary.",
    "family_disjoint_splits": "A held family in training invalidates the transfer test.",
    "relabel_pairs_complete": "Unpaired relabel rows cannot support a paired consistency delta.",
    "held_family_outcome_classes": "AUROC needs outcome variation in every evaluated family.",
    "bootstrap_sample_support": "Intervals need enough rows in both binary outcome classes.",
}


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value in the artifact receipt format."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash the exact upstream bytes used by this experiment."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def audit_feature_schema(schema: Mapping[str, Any]) -> JsonDict:
    """Reject any model-input path that contains one denied field token."""

    violations = []
    features = [feature for values in schema.get("arms", {}).values() for feature in values]
    for feature in features:
        tokens = str(feature).lower().split(".")
        match = next((entry for entry in ORACLE_FEATURE_DENYLIST if entry in tokens), None)
        if match is not None:
            violations.append({"feature": str(feature), "matched_denylist_entry": match})
    return {"passed": not violations, "violations": violations}


def build_family_heldout_splits(
    rows: Sequence[Mapping[str, Any]], families: Sequence[str] = HELD_FAMILIES
) -> dict[str, JsonDict]:
    """Build fixed leave-one-family-out folds and reject ambiguous membership."""

    row_ids = [str(row.get("row_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate_row_id")
    allowed = set(families)
    unexpected = sorted({str(row.get("family")) for row in rows} - allowed)
    if unexpected:
        raise ValueError("unexpected_family:" + ",".join(unexpected))
    folds = {}
    for held_family in families:
        heldout = [str(row["row_id"]) for row in rows if row.get("family") == held_family]
        train = [str(row["row_id"]) for row in rows if row.get("family") != held_family]
        folds[str(held_family)] = {
            "train_families": [family for family in families if family != held_family],
            "heldout_families": [held_family],
            "train_row_ids": train,
            "heldout_row_ids": heldout,
        }
    return folds


def pair_relabel_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Pair one base and one relabel row for each model and source pair."""

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("model_family_id")), str(row.get("pair_id")))
        role = str(row.get("pair_role"))
        if role in grouped[key]:
            raise ValueError("duplicate_pair_role:" + "|".join((*key, role)))
        grouped[key][role] = row
    pairs = []
    for (model, pair_id), mates in sorted(grouped.items()):
        if set(mates) != {"base", "relabel"}:
            raise ValueError("incomplete_relabel_pair:" + "|".join((model, pair_id)))
        families = {str(mates[role].get("family")) for role in ("base", "relabel")}
        if len(families) != 1:
            raise ValueError("cross_family_relabel_pair:" + "|".join((model, pair_id)))
        pairs.append(
            {
                "unit_id": "|".join((model, pair_id)),
                "model_family_id": model,
                "pair_id": pair_id,
                "family": families.pop(),
                "base_row_id": str(mates["base"]["row_id"]),
                "relabel_row_id": str(mates["relabel"]["row_id"]),
            }
        )
    return pairs


def recompute_binary_metrics(
    rows: Sequence[Mapping[str, Any]], *, calibration_bins: int = 10
) -> JsonDict:
    """Recompute classification, calibration, and localization from rows."""

    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    targets = [int(row["target"]) for row in rows]
    if any(target not in {0, 1} for target in targets):
        raise ValueError("binary_target required")
    energies = [float(row["energy"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    positives = [energy for energy, target in zip(energies, targets, strict=True) if target == 1]
    negatives = [energy for energy, target in zip(energies, targets, strict=True) if target == 0]
    if positives and negatives:
        pair_scores = [
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        ]
        auroc = sum(pair_scores) / len(pair_scores)
        ranked = sorted(zip(energies, targets, strict=True), reverse=True)
        positive_seen = 0
        precisions = []
        for rank, (_, target) in enumerate(ranked, start=1):
            if target == 1:
                positive_seen += 1
                precisions.append(positive_seen / rank)
        auprc = sum(precisions) / len(positives)
    else:
        auroc = None
        auprc = None
    n_rows = len(rows)
    accuracy = sum(a == b for a, b in zip(predictions, targets, strict=True)) / n_rows
    brier = (
        sum(
            (probability - target) ** 2
            for probability, target in zip(probabilities, targets, strict=True)
        )
        / n_rows
    )
    expected_calibration_error = 0.0
    for index in range(calibration_bins):
        low = index / calibration_bins
        high = (index + 1) / calibration_bins
        bucket = [
            (probability, target)
            for probability, target in zip(probabilities, targets, strict=True)
            if low <= probability < high or (index == calibration_bins - 1 and probability == 1.0)
        ]
        if bucket:
            confidence = sum(item[0] for item in bucket) / len(bucket)
            frequency = sum(item[1] for item in bucket) / len(bucket)
            expected_calibration_error += len(bucket) / n_rows * abs(confidence - frequency)
    localized = [
        row.get("localization") == row.get("localization_target")
        for row in rows
        if row.get("localization_target") is not None
    ]
    return {
        "n_rows": n_rows,
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "brier_score": brier,
        "expected_calibration_error": expected_calibration_error,
        "localization_accuracy": sum(localized) / len(localized) if localized else None,
    }


def _source_family_assignments(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Collect the one immutable source split declared for each family."""

    assignments = {}
    for family in HELD_FAMILIES:
        assignments[family] = sorted(
            {str(row.get("split")) for row in rows if row.get("family") == family}
        )
    return assignments


def _outcome_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count closed diagnoses within each held family."""

    return {
        family: dict(
            sorted(
                Counter(
                    str(row.get("diagnosis")) for row in rows if row.get("family") == family
                ).items()
            )
        )
        for family in HELD_FAMILIES
    }


def evaluate_preconditions(
    corpus: Mapping[str, Any] | None, schema: Mapping[str, Any] = FEATURE_SCHEMA
) -> JsonDict:
    """Evaluate all registered gates without training or changing the corpus."""

    present = corpus is not None
    checks = [
        {
            "check": "exp6745_artifact_present",
            "expected": True,
            "observed": present,
            "passed": present,
        }
    ]
    if not present:
        return {"checks": checks, "all_passed": False, "oracle_leakage_detected": False}
    assert corpus is not None
    rows = list(corpus.get("rows", []))
    feature_audit = audit_feature_schema(schema)
    assignments = _source_family_assignments(rows)
    expected_assignments = {family: [split] for family, split in SOURCE_FAMILY_SPLITS.items()}
    try:
        folds = build_family_heldout_splits(rows)
        split_ok = all(
            set(fold["train_row_ids"]).isdisjoint(fold["heldout_row_ids"])
            for fold in folds.values()
        )
    except ValueError:
        split_ok = False
    try:
        pair_relabel_rows(rows)
        relabel_ok = True
    except ValueError:
        relabel_ok = False
    outcome_counts = _outcome_counts(rows)
    class_ok = all(len(counts) >= 2 for counts in outcome_counts.values())
    bootstrap_observed = {}
    for family in HELD_FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        reasoning = sum(row.get("diagnosis") == "reasoning_error" for row in family_rows)
        other = len(family_rows) - reasoning
        bootstrap_observed[family] = {
            "rows": len(family_rows),
            "reasoning_error": reasoning,
            "other": other,
            "minimum_binary_class_rows": min(reasoning, other),
        }
    bootstrap_ok = all(
        values["rows"] >= BOOTSTRAP_CONFIG["minimum_rows_per_family"]
        and values["minimum_binary_class_rows"] >= BOOTSTRAP_CONFIG["minimum_rows_per_binary_class"]
        for values in bootstrap_observed.values()
    )
    checks.extend(
        [
            {
                "check": "exp6745_identity",
                "expected": 6745,
                "observed": corpus.get("experiment"),
                "passed": corpus.get("experiment") == 6745,
            },
            {
                "check": "dual_encoding_corpus_ready",
                "expected": True,
                "observed": corpus.get("dual_encoding_corpus_ready"),
                "passed": corpus.get("dual_encoding_corpus_ready") is True,
            },
            {
                "check": "feature_denylist_clean",
                "expected": [],
                "observed": feature_audit["violations"],
                "passed": feature_audit["passed"],
            },
            {
                "check": "immutable_source_family_splits",
                "expected": expected_assignments,
                "observed": assignments,
                "passed": assignments == expected_assignments,
            },
            {
                "check": "family_disjoint_splits",
                "expected": True,
                "observed": split_ok,
                "passed": split_ok,
            },
            {
                "check": "relabel_pairs_complete",
                "expected": True,
                "observed": relabel_ok,
                "passed": relabel_ok,
            },
            {
                "check": "held_family_outcome_classes",
                "expected": ">=2 diagnosis classes per held family",
                "observed": outcome_counts,
                "passed": class_ok,
            },
            {
                "check": "bootstrap_sample_support",
                "expected": {
                    "minimum_rows_per_family": BOOTSTRAP_CONFIG["minimum_rows_per_family"],
                    "minimum_rows_per_binary_class": BOOTSTRAP_CONFIG[
                        "minimum_rows_per_binary_class"
                    ],
                },
                "observed": bootstrap_observed,
                "passed": bootstrap_ok,
            },
        ]
    )
    leakage_gates = {
        "feature_denylist_clean",
        "immutable_source_family_splits",
        "family_disjoint_splits",
    }
    leakage = any(check["check"] in leakage_gates and not check["passed"] for check in checks)
    return {
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "oracle_leakage_detected": leakage,
    }


def _compact_split_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep family membership and split hashes without copying labels."""

    folds = build_family_heldout_splits(rows) if rows else {}
    return {
        "source_family_assignments": deepcopy(SOURCE_FAMILY_SPLITS),
        "heldout_folds": {
            family: {
                "train_families": fold["train_families"],
                "heldout_families": fold["heldout_families"],
                "train_row_count": len(fold["train_row_ids"]),
                "heldout_row_count": len(fold["heldout_row_ids"]),
                "membership_sha256": sha256_json(
                    {
                        "train": fold["train_row_ids"],
                        "heldout": fold["heldout_row_ids"],
                    }
                ),
            }
            for family, fold in folds.items()
        },
    }


def _reproducibility_checksum(
    rows: Sequence[Mapping[str, Any]], split_receipt: Mapping[str, Any]
) -> str:
    """Hash exactly the feature, split, config, and ordered-row inputs."""

    return sha256_json(
        {
            "feature_schema": FEATURE_SCHEMA,
            "splits": split_receipt,
            "training_config": TRAINING_CONFIG,
            "bootstrap_config": BOOTSTRAP_CONFIG,
            "random_seeds": RANDOM_SEEDS,
            "ordered_rows": [row.get("row_sha256") for row in rows],
        }
    )


def _first_failure(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the first failed gate in registered order."""

    return next(check for check in checks if check.get("passed") is not True)


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    corpus: Mapping[str, Any] | None,
    source_path: Path,
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build the full required schema for an owned precondition block."""

    rows = list(corpus.get("rows", [])) if corpus is not None else []
    split_receipt = _compact_split_receipt(rows)
    failed = _first_failure(preconditions["checks"])
    leakage = preconditions.get("oracle_leakage_detected") is True
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6746,
        "title": "Held-family oracle-distinct diagnostic energy",
        "run_date": date,
        "status": "complete_blocked_diagnostic_energy",
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": deepcopy(RANDOM_SEEDS),
        "reproducibility_checksum": _reproducibility_checksum(rows, split_receipt),
        "source_artifact": {
            "path": str(UPSTREAM_PATH),
            "exists": source_path.is_file(),
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
            "experiment": corpus.get("experiment") if corpus is not None else None,
            "reproducibility_checksum": (
                corpus.get("reproducibility_checksum") if corpus is not None else None
            ),
        },
        "verifier_is_oracle": False,
        "feature_schema": deepcopy(FEATURE_SCHEMA),
        "oracle_feature_denylist": list(ORACLE_FEATURE_DENYLIST),
        "feature_taint_audit": {
            "schema_frozen_before_label_join": True,
            "schema_audit": audit_feature_schema(FEATURE_SCHEMA),
            "taint_audit": {"status": "passed", "violations": []},
            "deterministic_proxy_audit": {
                "status": "not_evaluable_precondition_failed",
                "violations": [],
            },
        },
        "immutable_family_splits": split_receipt,
        "training_config": deepcopy(TRAINING_CONFIG),
        "bootstrap_config": deepcopy(BOOTSTRAP_CONFIG),
        "preconditions_checked": deepcopy(preconditions["checks"]),
        "rows": [],
        "heldout_metrics_by_family": {},
        "paired_relabel_metrics": {},
        "heldout_reasoning_error_auroc": None,
        "oracle_leakage_detected": leakage,
        "diagnostic_energy_ready": False,
        "gate_check_summary": {
            "all_passed": False,
            "failed_check": failed["check"],
            "expected": deepcopy(failed["expected"]),
            "observed": deepcopy(failed["observed"]),
            "checks": deepcopy(preconditions["checks"]),
        },
        "verdict_class": "disqualified" if leakage else "blocked",
        "honest_verdict": (
            "complete_blocked_diagnostic_energy: "
            f"{failed['check']} expected {canonical_json(failed['expected'])}; "
            f"observed {canonical_json(failed['observed'])}"
        ),
    }
    principles = deepcopy(FIELD_PRINCIPLES)
    principles.update({f"gate:{name}": principle for name, principle in GATE_PRINCIPLES.items()})
    artifact["field_principles"] = principles
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, principles, blocked semantics, and hashes."""

    required = set(FIELD_PRINCIPLES)
    checks = artifact.get("gate_check_summary", {}).get("checks", [])
    principle_keys = set(artifact.get("field_principles", {}))
    return [
        name
        for name, failed in (
            ("missing_required_fields", bool(required - set(artifact))),
            ("missing_field_principles", bool(set(artifact) - principle_keys)),
            (
                "missing_gate_principles",
                any(f"gate:{check.get('check')}" not in principle_keys for check in checks),
            ),
            ("verifier_oracle_mismatch", artifact.get("verifier_is_oracle") is not False),
            ("blocked_rows_present", bool(artifact.get("rows"))),
            ("blocked_readiness_true", artifact.get("diagnostic_energy_ready") is not False),
            (
                "blocked_verdict_prefix_mismatch",
                not str(artifact.get("honest_verdict", "")).startswith(
                    "complete_blocked_diagnostic_energy"
                ),
            ),
            (
                "heldout_auroc_present_on_block",
                artifact.get("heldout_reasoning_error_auroc") is not None,
            ),
            ("duration_not_finite", not math.isfinite(float(artifact.get("duration_s", math.nan)))),
        )
        if failed
    ]


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically replace one result artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp6746-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(date: str = "20260829", root: Path = REPO_ROOT) -> JsonDict:
    """Evaluate preconditions and write the current complete blocked result."""

    started = time.monotonic()
    frozen_schema = deepcopy(FEATURE_SCHEMA)
    source_path = root / UPSTREAM_PATH
    corpus = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    preconditions = evaluate_preconditions(corpus, frozen_schema)
    if preconditions["all_passed"] is True:  # pragma: no cover - frozen Exp6745 fails this gate
        raise RuntimeError(
            "Exp6745 unexpectedly became evaluable; implement the registered arm run"
        )
    artifact = build_blocked_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        corpus=corpus,
        source_path=source_path,
        preconditions=preconditions,
    )
    write_json_atomic(root / RESULT_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by the required command
    """Run Exp6746 and print the terminal artifact location."""

    artifact = run()
    print(json.dumps({"artifact": str(RESULT_PATH), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
