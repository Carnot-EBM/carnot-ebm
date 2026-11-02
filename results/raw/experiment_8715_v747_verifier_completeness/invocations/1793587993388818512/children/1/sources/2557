"""Independently audit compatibility scores for shortcut explanations.

Spec refs: REQ-CONSTRAINT-6852 and SCENARIO-CONSTRAINT-6852-*.

The producer already summarized its model scores. This module does not trust
those summaries. It reads the raw token receipts, joins them to score-free
semantic identities, and rebuilds every margin on deterministic CPU code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6852_compatibility_shortcut_authority_audit.json")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6852_compatibility_shortcut_authority_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6852_compatibility_shortcut_authority_audit.py")
TEST_PATH = Path("tests/python/test_experiment_6852_compatibility_shortcut_authority_audit.py")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
SOURCE_PATHS = {
    "exp6487": Path("results/experiment_6487_representation_integrity_audit.json"),
    "exp6847": Path("results/experiment_6847_v598_independent_capstone.json"),
    "exp6849": Path("results/experiment_6849_typed_program_isomorphic_authority_audit.json"),
    "exp6850": Path("results/experiment_6850_three_family_scoring_admission_canary.json"),
    "exp6851": Path("results/experiment_6851_three_family_isomorphic_compatibility_stream.json"),
}
UPSTREAM_IDS = ("exp6849", "exp6850", "exp6851")
REQUIRED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_REPEATS = (0, 1)
ATTACK_KINDS = (
    "identifier_only",
    "prompt_length",
    "candidate_length",
    "token_count",
    "label_position",
    "row_order",
    "normalization",
    "surface_form",
    "model_scale",
    "model_family",
)
INFERENCE_SUBSTRATE = "deterministic CPU independent reduction"
RANDOM_SEED = 6852
RUN_DATE = "20260901"
ARTIFACT_SCHEMA = "carnot.experiment_6852.compatibility_shortcut_authority_audit.v1"
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
SPEC_REFS = (
    "REQ-CONSTRAINT-6852",
    "SCENARIO-CONSTRAINT-6852-MISSING-PRODUCER",
    "SCENARIO-CONSTRAINT-6852-PARTIAL-MODELS",
    "SCENARIO-CONSTRAINT-6852-NULL-SCORES",
    "SCENARIO-CONSTRAINT-6852-DUPLICATE-IDENTITY",
    "SCENARIO-CONSTRAINT-6852-LABEL-INVERSION",
    "SCENARIO-CONSTRAINT-6852-ROW-REORDER",
    "SCENARIO-CONSTRAINT-6852-STALE-HASH",
    "SCENARIO-CONSTRAINT-6852-SHORTCUTS",
    "SCENARIO-CONSTRAINT-6852-ISOMORPHIC",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "upstream_state_manifest",
    "conductor_skip_manifest",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "recomputed_margin_rows",
    "isomorphic_invariance_results",
    "shortcut_attack_results",
    "missing_model_manifest",
    "control_explanation_results",
    "authority_failure_witnesses",
    "compatibility_audit_complete_score",
    "compatibility_claim_eligible_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
TASK_TITLE_FRAGMENTS = {
    "exp6849": "Independent typed-program isomorphic authority aud",
    "exp6850": "Three-family forced-sequence scoring admission can",
    "exp6851": "Three-family isomorphic fixed-sequence compatibili",
}


def canonical_json(value: object) -> bytes:
    """Return stable JSON bytes so row order cannot change a checksum."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix hashes so algorithms remain explicit in durable artifacts."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash the exact source bytes used by the independent audit."""

    return sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _gate(check: str, expected: object, observed: object, passed: bool) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def parse_conductor_records(text: str) -> list[JsonDict]:
    """Extract every completion, failure, block, or skip for the three tasks."""

    records: list[JsonDict] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        for task_id, title_fragment in TASK_TITLE_FRAGMENTS.items():
            if title_fragment not in cells[1]:
                continue
            status = cells[2]
            record_type = {
                "GATE_BLOCK": "conductor_skip",
                "FAIL": "failure",
                "OK": "completion",
            }.get(status, "other")
            records.append(
                {
                    "task_id": task_id,
                    "timestamp": cells[0],
                    "task_title": cells[1],
                    "status": status,
                    "detail": cells[3],
                    "record_type": record_type,
                    "is_skip_record": status == "GATE_BLOCK",
                }
            )
            break
    return records


def _manifest_authority(
    authority: Mapping[str, Any],
) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    pairs: dict[str, JsonDict] = {}
    candidates: dict[str, JsonDict] = {}
    for pair in authority.get("sanitized_candidate_pair_manifest", []):
        if not isinstance(pair, dict):
            continue
        semantic_identity = pair.get("semantic_identity")
        if isinstance(semantic_identity, str):
            pairs[semantic_identity] = pair
        for candidate in pair.get("candidates", []):
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str):
                candidates[candidate["candidate_id"]] = candidate
    return pairs, candidates


def _expected_transforms(authority: Mapping[str, Any]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = defaultdict(lambda: {"base"})
    for row in authority.get("isomorphic_transform_manifest", []):
        if isinstance(row, dict) and isinstance(row.get("pair_id"), str):
            expected[row["pair_id"]].add(str(row.get("transform_kind")))
    return expected


def _identity_witnesses(
    rows: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    witnesses: list[JsonDict] = []
    row_counts = Counter(str(row.get("row_identity")) for row in rows)
    for identity, count in sorted(row_counts.items()):
        if count > 1:
            witnesses.append(
                {
                    "failure_kind": "duplicate_row_identity",
                    "row_identity": identity,
                    "occurrences": count,
                }
            )
    semantic_counts = Counter(
        (
            str(row.get("model_hf_id")),
            str(row.get("semantic_pair_identity")),
            str(row.get("transform_kind")),
            row.get("repeat"),
        )
        for row in rows
    )
    for identity, count in sorted(semantic_counts.items(), key=lambda item: str(item[0])):
        if count > 1:
            witnesses.append(
                {
                    "failure_kind": "duplicate_semantic_scoring_unit",
                    "semantic_scoring_unit": list(identity),
                    "occurrences": count,
                }
            )
    receipt_counts = Counter(
        (str(row.get("row_identity")), str(row.get("candidate_id"))) for row in receipts
    )
    for identity, count in sorted(receipt_counts.items()):
        if count > 1:
            witnesses.append(
                {
                    "failure_kind": "duplicate_receipt_identity",
                    "receipt_identity": list(identity),
                    "occurrences": count,
                }
            )
    return witnesses


def _finite_scores(receipt: Mapping[str, Any]) -> tuple[list[float] | None, str | None]:
    scores = receipt.get("token_logprobs")
    if not isinstance(scores, list) or not scores:
        return None, "missing_token_scores"
    if all(value is None for value in scores):
        return None, "null_token_score"
    if any(value is None for value in scores):
        return None, "null_token_score"
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
        return None, "non_finite_token_score"
    token_ids = receipt.get("candidate_token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != len(scores):
        return None, "token_score_length_mismatch"
    return [float(value) for value in scores], None


def _recompute_rows(
    authority: Mapping[str, Any],
    admission: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    rows = [row for row in producer.get("rows", []) if isinstance(row, dict)]
    receipts = [row for row in producer.get("token_score_receipts", []) if isinstance(row, dict)]
    failures = _identity_witnesses(rows, receipts)
    pairs, candidates = _manifest_authority(authority)
    receipt_groups: dict[str, list[JsonDict]] = defaultdict(list)
    for receipt in receipts:
        receipt_groups[str(receipt.get("row_identity"))].append(receipt)
    model_specs = {
        row.get("hf_id"): row
        for row in admission.get("model_specs", [])
        if isinstance(row, dict) and isinstance(row.get("hf_id"), str)
    }
    recomputed: list[JsonDict] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda value: str(value.get("row_identity"))):
        row_identity = str(row.get("row_identity"))
        if row_identity in seen:
            continue
        seen.add(row_identity)
        semantic_pair_identity = row.get("semantic_pair_identity")
        pair = pairs.get(str(semantic_pair_identity))
        if pair is None or pair.get("pair_id") != row.get("pair_id"):
            failures.append(
                {
                    "failure_kind": "semantic_pair_identity_mismatch",
                    "row_identity": row_identity,
                    "expected": sorted(pairs),
                    "observed": semantic_pair_identity,
                }
            )
            continue
        paired = receipt_groups.get(row_identity, [])
        if len(paired) != 2:
            failures.append(
                {
                    "failure_kind": "candidate_receipt_count",
                    "row_identity": row_identity,
                    "expected": 2,
                    "observed": len(paired),
                }
            )
            continue
        prompt_tokens = paired[0].get("prompt_token_ids")
        if (
            not isinstance(prompt_tokens, list)
            or paired[1].get("prompt_token_ids") != prompt_tokens
        ):
            failures.append(
                {
                    "failure_kind": "paired_prompt_token_mismatch",
                    "row_identity": row_identity,
                    "expected": "identical non-null prompt token ids",
                    "observed": [
                        paired[0].get("prompt_token_ids"),
                        paired[1].get("prompt_token_ids"),
                    ],
                }
            )
            continue
        scored: dict[bool, JsonDict] = {}
        invalid = False
        for receipt in paired:
            candidate = candidates.get(str(receipt.get("candidate_id")))
            if candidate is None:
                failures.append(
                    {
                        "failure_kind": "unknown_candidate_identity",
                        "row_identity": row_identity,
                        "observed": receipt.get("candidate_id"),
                    }
                )
                invalid = True
                continue
            expected_label = candidate.get("exact_label")
            observed_label = receipt.get("exact_label")
            if expected_label is not observed_label:
                failures.append(
                    {
                        "failure_kind": "label_inversion",
                        "row_identity": row_identity,
                        "candidate_id": receipt.get("candidate_id"),
                        "expected": expected_label,
                        "observed": observed_label,
                    }
                )
                invalid = True
                continue
            if candidate.get("semantic_identity") != receipt.get("candidate_semantic_identity"):
                failures.append(
                    {
                        "failure_kind": "candidate_semantic_identity_mismatch",
                        "row_identity": row_identity,
                        "candidate_id": receipt.get("candidate_id"),
                        "expected": candidate.get("semantic_identity"),
                        "observed": receipt.get("candidate_semantic_identity"),
                    }
                )
                invalid = True
                continue
            scores, score_failure = _finite_scores(receipt)
            if score_failure is not None:
                failures.append(
                    {
                        "failure_kind": score_failure,
                        "row_identity": row_identity,
                        "candidate_id": receipt.get("candidate_id"),
                        "observed": receipt.get("token_logprobs"),
                    }
                )
                invalid = True
                continue
            assert scores is not None
            scored[bool(expected_label)] = {
                "sum": sum(scores),
                "mean": sum(scores) / len(scores),
                "token_count": len(scores),
            }
        if invalid or set(scored) != {False, True}:
            continue
        sum_margin = scored[True]["sum"] - scored[False]["sum"]
        mean_margin = scored[True]["mean"] - scored[False]["mean"]
        producer_sum = row.get("sum_log_likelihood_margin")
        producer_mean = row.get("mean_token_log_likelihood_margin")
        margin_matches = (
            isinstance(producer_sum, (int, float))
            and isinstance(producer_mean, (int, float))
            and math.isclose(float(producer_sum), sum_margin, abs_tol=1e-6)
            and math.isclose(float(producer_mean), mean_margin, abs_tol=1e-6)
        )
        if not margin_matches:
            failures.append(
                {
                    "failure_kind": "producer_margin_mismatch",
                    "row_identity": row_identity,
                    "expected": {
                        "sum_log_likelihood_margin": sum_margin,
                        "mean_token_log_likelihood_margin": mean_margin,
                    },
                    "observed": {
                        "sum_log_likelihood_margin": producer_sum,
                        "mean_token_log_likelihood_margin": producer_mean,
                    },
                }
            )
        model_spec = model_specs.get(row.get("model_hf_id"), {})
        recomputed.append(
            {
                "row_identity": row_identity,
                "model_hf_id": row.get("model_hf_id"),
                "model_family": model_spec.get("family", row.get("model_family")),
                "model_size_bytes": model_spec.get("model_size_bytes"),
                "pair_id": pair.get("pair_id"),
                "semantic_pair_identity": semantic_pair_identity,
                "transform_id": row.get("transform_id"),
                "transform_kind": row.get("transform_kind"),
                "repeat": row.get("repeat"),
                "prompt_token_count": len(prompt_tokens),
                "compatible_candidate_length": row.get("compatible_candidate_length"),
                "violation_candidate_length": row.get("violation_candidate_length"),
                "compatible_token_count": scored[True]["token_count"],
                "violation_token_count": scored[False]["token_count"],
                "compatible_label_position": row.get("compatible_label_position"),
                "recomputed_sum_margin": sum_margin,
                "recomputed_mean_token_margin": mean_margin,
                "producer_sum_margin": producer_sum,
                "producer_mean_token_margin": producer_mean,
                "producer_margin_matches": margin_matches,
                "direction": _direction(mean_margin),
            }
        )
    failures.sort(key=lambda value: canonical_json(value))
    return recomputed, failures


def _unit_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("model_hf_id")),
        str(row.get("semantic_pair_identity")),
        int(row.get("repeat", -1)),
    )


def _isomorphic_results(
    recomputed: Sequence[JsonDict], expected: Mapping[str, set[str]]
) -> list[JsonDict]:
    grouped: dict[tuple[str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in recomputed:
        grouped[_unit_key(row)][str(row.get("transform_kind"))] = row
    results: list[JsonDict] = []
    for key, cells in sorted(grouped.items()):
        model, semantic_pair_identity, repeat = key
        base = cells.get("base")
        pair_id = (
            str(base.get("pair_id")) if base else str(next(iter(cells.values())).get("pair_id"))
        )
        transform_names = sorted(expected.get(pair_id, {"base"}) - {"base"})
        comparisons: list[JsonDict] = []
        for transform_kind in transform_names:
            transformed = cells.get(transform_kind)
            comparisons.append(
                {
                    "transform_kind": transform_kind,
                    "base_margin": base.get("recomputed_mean_token_margin") if base else None,
                    "transformed_margin": (
                        transformed.get("recomputed_mean_token_margin") if transformed else None
                    ),
                    "direction_survives": bool(
                        base
                        and transformed
                        and base.get("direction") == transformed.get("direction")
                        and base.get("direction") != "zero"
                    ),
                }
            )
        results.append(
            {
                "model_hf_id": model,
                "semantic_pair_identity": semantic_pair_identity,
                "repeat": repeat,
                "base_present": base is not None,
                "expected_transform_count": len(transform_names),
                "observed_transform_count": sum(
                    transform_kind in cells for transform_kind in transform_names
                ),
                "direction_survives": bool(comparisons)
                and all(row["direction_survives"] for row in comparisons),
                "comparisons": comparisons,
            }
        )
    return results


def _numeric_delta(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    scale = max(abs(float(numerator)), abs(float(denominator)), 1.0)
    return abs(float(numerator) - float(denominator)) / scale


def _shortcut_results(recomputed: Sequence[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in recomputed:
        grouped[_unit_key(row)][str(row.get("transform_kind"))] = row
    family_bases: dict[tuple[str, int], list[JsonDict]] = defaultdict(list)
    for key, cells in grouped.items():
        base = cells.get("base")
        if base:
            family_bases[(key[1], key[2])].append(base)
    results: list[JsonDict] = []
    transform_controls = {
        "identifier_only": "identifier_permutation",
        "label_position": "label_swap",
        "row_order": "row_reordering",
        "surface_form": "surface_paraphrase",
    }
    for key, cells in sorted(grouped.items()):
        model, semantic_pair_identity, repeat = key
        base = cells.get("base")
        base_margin = float(base["recomputed_mean_token_margin"]) if base else None
        scientific_magnitude = abs(base_margin) if base_margin is not None else None
        controls: dict[str, tuple[float | None, str]] = {}
        for attack_kind, transform_kind in transform_controls.items():
            transformed = cells.get(transform_kind)
            control = (
                abs(float(transformed["recomputed_mean_token_margin"]) - base_margin)
                if transformed and base_margin is not None
                else None
            )
            controls[attack_kind] = (control, f"{transform_kind}_minus_base_margin")
        prompt_controls = [
            abs(float(row["recomputed_mean_token_margin"]) - base_margin)
            for row in cells.values()
            if base_margin is not None
            and base
            and row.get("prompt_token_count") != base.get("prompt_token_count")
        ]
        controls["prompt_length"] = (
            max(prompt_controls, default=0.0) if base is not None else None,
            "largest_margin_shift_with_changed_prompt_token_count",
        )
        controls["candidate_length"] = (
            _numeric_delta(
                base.get("compatible_candidate_length") if base else None,
                base.get("violation_candidate_length") if base else None,
            ),
            "normalized_character_length_difference",
        )
        controls["token_count"] = (
            _numeric_delta(
                base.get("compatible_token_count") if base else None,
                base.get("violation_token_count") if base else None,
            ),
            "normalized_candidate_token_count_difference",
        )
        if base:
            max_tokens = max(
                int(base["compatible_token_count"]), int(base["violation_token_count"]), 1
            )
            normalized_sum = float(base["recomputed_sum_margin"]) / max_tokens
            normalization = abs(normalized_sum - float(base["recomputed_mean_token_margin"]))
        else:
            normalization = None
        controls["normalization"] = (
            normalization,
            "difference_between_mean_margin_and_max_token_normalized_sum_margin",
        )
        peer_bases = family_bases.get((semantic_pair_identity, repeat), [])
        peer_margins = [float(row["recomputed_mean_token_margin"]) for row in peer_bases]
        family_range = max(peer_margins) - min(peer_margins) if len(peer_margins) > 1 else None
        controls["model_family"] = (family_range, "cross_family_base_margin_range")
        sized = [row for row in peer_bases if isinstance(row.get("model_size_bytes"), (int, float))]
        scale_range = (
            max(float(row["recomputed_mean_token_margin"]) for row in sized)
            - min(float(row["recomputed_mean_token_margin"]) for row in sized)
            if len(sized) > 1
            else None
        )
        controls["model_scale"] = (scale_range, "cross_scale_base_margin_range")
        for attack_kind in ATTACK_KINDS:
            control_margin, source = controls[attack_kind]
            explains = (
                control_margin is not None
                and scientific_magnitude is not None
                and control_margin + 1e-12 >= scientific_magnitude
            )
            results.append(
                {
                    "model_hf_id": model,
                    "semantic_pair_identity": semantic_pair_identity,
                    "repeat": repeat,
                    "attack_kind": attack_kind,
                    "control_source": source,
                    "scientific_margin_abs": scientific_magnitude,
                    "observed_control_margin": control_margin,
                    "explains_same_or_greater": explains if control_margin is not None else None,
                    "passed": control_margin is not None and not explains,
                }
            )
    return results


def _principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "inference_substrate": "Names the CPU-only reduction method and prevents an implied model call.",
        "source_artifact_hashes": "Binds every authority decision to the exact bytes that were read.",
        "rows": "Preserves each scored or missing model unit and each shortcut attack.",
        "recomputed_margin_rows": "Uses raw token receipts instead of producer aggregates.",
        "missing_model_manifest": "Keeps missing evidence null instead of converting it to zero.",
        "compatibility_claim_eligible_score": "Opens only after every authority and identifiability gate passes.",
        "verifier_is_oracle": "States that this reducer audits evidence but does not define semantic truth.",
        "honest_verdict": "States the terminal row-supported scientific disposition.",
    }
    return {
        field: specific.get(
            field, f"Records the audit's {field.replace('_', ' ')} without omission."
        )
        for field in fields
    }


def reduce_evidence(
    *,
    authority: Mapping[str, Any] | None,
    admission: Mapping[str, Any] | None,
    producer: Mapping[str, Any] | None,
    source_artifact_hashes: Mapping[str, Any],
    upstream_state_manifest: Sequence[Mapping[str, Any]],
    conductor_skip_manifest: Sequence[Mapping[str, Any]],
    run_date: str,
    duration_s: float,
    historical_evidence: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Reduce all evidence, including valid blocked and partial inputs."""

    authority_payload = authority or {}
    admission_payload = admission or {}
    producer_payload = producer or {}
    recomputed, failures = _recompute_rows(authority_payload, admission_payload, producer_payload)
    expected_transforms = _expected_transforms(authority_payload)
    invariance = _isomorphic_results(recomputed, expected_transforms)
    attacks = _shortcut_results(recomputed)
    model_rows: list[JsonDict] = []
    missing_models: list[JsonDict] = []
    expected_per_model = sum(len(transforms) for transforms in expected_transforms.values()) * len(
        REQUIRED_REPEATS
    )
    for model in REQUIRED_MODELS:
        available = [row for row in recomputed if row.get("model_hf_id") == model]
        model_margins = [float(row["recomputed_mean_token_margin"]) for row in available]
        if available:
            model_rows.append(
                {
                    "row_kind": "scored_scientific_unit",
                    "model_hf_id": model,
                    "expected_row_count": expected_per_model,
                    "observed_row_count": len(available),
                    "complete": len(available) == expected_per_model,
                    "mean_margin": _mean(model_margins),
                }
            )
        else:
            model_rows.append(
                {
                    "row_kind": "missing_scientific_unit",
                    "model_hf_id": model,
                    "expected_row_count": expected_per_model,
                    "observed_row_count": 0,
                    "complete": False,
                    "mean_margin": None,
                }
            )
            missing_models.append(
                {
                    "model_hf_id": model,
                    "state": "missing",
                    "expected_row_count": expected_per_model,
                    "observed_row_count": 0,
                    "margin": None,
                }
            )
    rows: list[JsonDict] = [dict(row) for row in historical_evidence]
    rows.extend(model_rows)
    rows.extend({"row_kind": "shortcut_attack", **row} for row in attacks)
    rows.sort(key=canonical_json)
    stale_hashes = sorted(
        source_id
        for source_id, receipt in source_artifact_hashes.items()
        if isinstance(receipt, Mapping) and receipt.get("hash_match") is not True
    )
    state_by_id = {str(row.get("task_id")): row for row in upstream_state_manifest}
    readable_upstreams = all(
        state_by_id.get(task_id, {}).get("artifact_readable") is True for task_id in UPSTREAM_IDS
    )
    authority_ready = (
        all(
            authority_payload.get(field) == 1
            for field in (
                "authority_audit_complete_score",
                "typed_program_authority_ready_score",
                "isomorphic_fixture_ready_score",
            )
        )
        and admission_payload.get("three_family_scoring_admission_ready_score") == 1
    )
    scientific_rows_exist = bool(recomputed)
    models_complete = all(row["complete"] for row in model_rows)
    shortcuts_clear = bool(attacks) and all(row["passed"] for row in attacks)
    invariance_clear = bool(invariance) and all(row["direction_survives"] for row in invariance)
    checks = [
        _gate("upstream_artifacts_readable", True, readable_upstreams, readable_upstreams),
        _gate("authority_and_admission_ready", True, authority_ready, authority_ready),
        _gate("scientific_rows_present", ">0", len(recomputed), scientific_rows_exist),
        _gate(
            "artifact_hash_authority",
            "all recorded hashes match current bytes",
            stale_hashes,
            not stale_hashes,
        ),
        _gate(
            "row_and_receipt_authority",
            [],
            sorted({row["failure_kind"] for row in failures}),
            not failures,
        ),
        _gate(
            "all_models_complete",
            {model: expected_per_model for model in REQUIRED_MODELS},
            {row["model_hf_id"]: row["observed_row_count"] for row in model_rows},
            models_complete,
        ),
        _gate("shortcut_controls_clear", True, shortcuts_clear, shortcuts_clear),
        _gate("isomorphic_direction_survives", True, invariance_clear, invariance_clear),
    ]
    claim_eligible = int(all(row["passed"] for row in checks))
    authority_failure = bool(failures or stale_hashes or not authority_ready)
    if not scientific_rows_exist:
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_compatibility_audit_no_scientific_rows"
    elif authority_failure:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_compatibility_audit_authority_failure"
    elif not models_complete:
        verdict_class = "partial"
        honest_verdict = "complete_partial_compatibility_audit_missing_model_rows"
    elif claim_eligible:
        means = [row["mean_margin"] for row in model_rows]
        if all(isinstance(value, float) and value > 0 for value in means):
            verdict_class = "positive"
            honest_verdict = "complete_positive_model_compatibility_identified"
        else:
            verdict_class = "null"
            honest_verdict = "complete_null_eligible_compatibility_direction_not_positive"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_model_compatibility_not_identifiable"
    preconditions = {
        "inventory_complete": len(state_by_id) == len(UPSTREAM_IDS),
        "upstream_artifacts_readable": readable_upstreams,
        "scientific_row_count": len(recomputed),
        "authority_failure_count": len(failures),
        "stale_artifact_ids": stale_hashes,
        "no_llm_invoked": True,
    }
    control_explanations = [
        {
            "model_hf_id": row["model_hf_id"],
            "semantic_pair_identity": row["semantic_pair_identity"],
            "repeat": row["repeat"],
            "attack_kind": row["attack_kind"],
            "outcome": (
                "missing_control"
                if row["observed_control_margin"] is None
                else "explains_same_or_greater"
                if row["explains_same_or_greater"]
                else "does_not_explain"
            ),
            "scientific_margin_abs": row["scientific_margin_abs"],
            "observed_control_margin": row["observed_control_margin"],
        }
        for row in attacks
    ]
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "exp6852-compatibility-shortcut-authority-audit",
        "run_date": run_date,
        "result_path": RESULT_PATH.as_posix(),
        "status": "complete",
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_artifact_hashes),
        "upstream_state_manifest": [dict(row) for row in upstream_state_manifest],
        "conductor_skip_manifest": [dict(row) for row in conductor_skip_manifest],
        "random_seed": RANDOM_SEED,
        "rows": rows,
        "recomputed_margin_rows": recomputed,
        "isomorphic_invariance_results": invariance,
        "shortcut_attack_results": attacks,
        "missing_model_manifest": missing_models,
        "control_explanation_results": control_explanations,
        "authority_failure_witnesses": failures,
        "compatibility_audit_complete_score": 1,
        "compatibility_claim_eligible_score": claim_eligible,
        "gate_check_summary": {
            "passed": claim_eligible == 1,
            "checks": checks,
            "failed_checks": [row for row in checks if not row["passed"]],
        },
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    checksum_payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "field_principles"}
    }
    artifact["reproducibility_checksum"] = sha256_bytes(canonical_json(checksum_payload))
    artifact["field_principles"] = _principles((*artifact, "field_principles"))
    return artifact


def _load_optional(path: Path) -> JsonDict | None:
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _upstream_states(repo_root: Path, payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    states: list[JsonDict] = []
    for task_id in UPSTREAM_IDS:
        path = repo_root / SOURCE_PATHS[task_id]
        payload = payloads.get(task_id)
        row_count = (
            len(payload.get("rows", []))
            if task_id == "exp6851" and isinstance(payload, dict)
            else 0
        )
        states.append(
            {
                "task_id": task_id,
                "artifact_path": SOURCE_PATHS[task_id].as_posix(),
                "artifact_present": path.is_file(),
                "artifact_readable": payload is not None,
                "artifact_sha256": sha256_file(path) if path.is_file() else None,
                "state": payload.get("status", "missing") if payload else "missing",
                "verdict_class": payload.get("verdict_class") if payload else "blocked",
                "honest_verdict": payload.get("honest_verdict") if payload else None,
                "scientific_row_count": row_count,
            }
        )
    return states


def _source_hash_manifest(
    repo_root: Path, payloads: Mapping[str, JsonDict | None]
) -> dict[str, JsonDict]:
    authority = payloads.get("exp6849") or {}
    producer = payloads.get("exp6851") or {}
    producer_sources = producer.get("source_artifact_hashes", {})
    authority_sources = authority.get("source_artifact_hashes", {})
    recorded = {
        "exp6847": (authority_sources.get("exp6847") or {}).get("file_sha256"),
        "exp6849": (producer_sources.get("exp6849") or {}).get("sha256"),
        "exp6850": (producer_sources.get("exp6850") or {}).get("sha256"),
    }
    paths = {
        **SOURCE_PATHS,
        "conductor_log": CONDUCTOR_PATH,
        "module": MODULE_PATH,
        "spec": SPEC_PATH,
        "test": TEST_PATH,
        "wrapper": WRAPPER_PATH,
    }
    manifest: dict[str, JsonDict] = {}
    for source_id, relative_path in paths.items():
        path = repo_root / relative_path
        current = sha256_file(path) if path.is_file() else None
        expected = recorded.get(source_id, current)
        manifest[source_id] = {
            "path": relative_path.as_posix(),
            "current_sha256": current,
            "recorded_sha256": expected,
            "hash_match": current is not None and current == expected,
        }
    return manifest


def _historical_evidence(payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    exp6487 = payloads.get("exp6487") or {}
    exp6847 = payloads.get("exp6847") or {}
    shortcuts = (exp6487.get("aggregate_row_recomputation") or {}).get("surviving_shortcuts", [])
    disposition = exp6847.get("typed_compatibility_disposition") or {}
    return [
        {
            "row_kind": "historical_authority_unit",
            "source_id": "exp6487",
            "verdict": exp6487.get("honest_verdict"),
            "surviving_shortcuts": shortcuts,
        },
        {
            "row_kind": "historical_authority_unit",
            "source_id": "exp6847",
            "verdict": exp6847.get("honest_verdict"),
            "typed_compatibility_verdict_class": disposition.get("verdict_class"),
            "typed_compatibility_blocking_criteria": disposition.get("blocking_criteria", []),
        },
    ]


def build_artifact(repo_root: Path, *, run_date: str, duration_s: float) -> JsonDict:
    """Load repository evidence and build one complete terminal artifact."""

    payloads = {
        source_id: _load_optional(repo_root / path) for source_id, path in SOURCE_PATHS.items()
    }
    states = _upstream_states(repo_root, payloads)
    conductor_path = repo_root / CONDUCTOR_PATH
    conductor_records = parse_conductor_records(
        conductor_path.read_text(encoding="utf-8") if conductor_path.is_file() else ""
    )
    return reduce_evidence(
        authority=payloads["exp6849"],
        admission=payloads["exp6850"],
        producer=payloads["exp6851"],
        source_artifact_hashes=_source_hash_manifest(repo_root, payloads),
        upstream_state_manifest=states,
        conductor_skip_manifest=conductor_records,
        historical_evidence=_historical_evidence(payloads),
        run_date=run_date,
        duration_s=duration_s,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Name schema failures so a malformed blocked result cannot look valid."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact):
        errors.append("field_principles must cover every top-level field")
    for field in ("compatibility_audit_complete_score", "compatibility_claim_eligible_score"):
        if artifact.get(field) not in {0, 1}:
            errors.append(f"{field} must be 0 or 1")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must name deterministic CPU independent reduction")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class is outside the closed vocabulary")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith("complete_"):
        errors.append("honest_verdict must be terminal")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write only the requested result path after validation succeeds."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E command
    """Run the deterministic reducer; no model process is created."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    started = time.monotonic()
    artifact = build_artifact(REPO_ROOT, run_date=args.date, duration_s=0.0)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    write_artifact(REPO_ROOT / args.result_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - module is normally called through the wrapper
    raise SystemExit(main())
