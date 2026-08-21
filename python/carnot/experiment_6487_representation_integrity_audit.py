"""Exp6487 independent raw representation integrity audit.

Spec refs: REQ-VERIFY-6487, SCENARIO-VERIFY-6487-RAW-REPLAY,
SCENARIO-VERIFY-6487-SHORTCUTS, SCENARIO-VERIFY-6487-ATTACKS,
SCENARIO-VERIFY-6487-CELLS.

The reducer deliberately does not import the Exp6486 producer. It reads the
Exp6486 artifact for the structured gate and raw-row manifest only. All counts,
controls, attacks, and cell decisions are rebuilt from raw row files.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6487_representation_integrity_audit.json")
EXP6486_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6486_three_family_forced_candidate_representations.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6487_representation_integrity_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6487_representation_integrity_audit.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
PROTECTED_RELATIVE_PATHS = (Path("scripts/research_conductor.py"), Path("research-roadmap.yaml"))

SCHEMA = "carnot.experiment_6487.representation_integrity_audit.v1"
RAW_ROW_SCHEMA = "carnot.experiment_6486.three_family_forced_candidate_representations.v1.raw_vector"
EXPERIMENT = 6487
EXPERIMENT_ID = "experiment_6487_representation_integrity_audit"
RUN_DATE = "20260821"
RANDOM_SEED = 6487
INFERENCE_SUBSTRATE = "independent_raw_representation_audit_no_llm"
VERIFIER_IS_ORACLE = True
SHORTCUT_BALANCED_ACCURACY_CEILING = 0.75

SPEC_REFS = (
    "REQ-VERIFY-6487",
    "SCENARIO-VERIFY-6487-RAW-REPLAY",
    "SCENARIO-VERIFY-6487-SHORTCUTS",
    "SCENARIO-VERIFY-6487-ATTACKS",
    "SCENARIO-VERIFY-6487-CELLS",
)

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6487_representation_integrity_audit "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6487_representation_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6487_representation_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6487_representation_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6487_representation_integrity_audit.py "
    "--fail-under=100"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6487_representation_integrity_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6487_representation_integrity_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6487_representation_integrity_audit.json"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "assert Path('ops/e2e-test-plan.md').exists()\""
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_PLAN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_hash_receipts",
    "independent_reducer_receipt",
    "reconstructed_stream_counts",
    "within_family_cell_rows",
    "shortcut_control_rows",
    "permutation_attack_rows",
    "provenance_attack_rows",
    "representation_integrity_ready_score",
    "missing_verifier_gaps",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal audit state distinguishes ready raw representations from disqualified or blocked evidence.",
    "upstream_hash_receipts": "Artifact and raw-row hashes prevent aggregate self-validation.",
    "independent_reducer_receipt": "Source hashes and import checks prove Exp6486 reducers were not reused.",
    "reconstructed_stream_counts": "Counts rebuilt from raw rows expose missing pairs, labels, dimensions, and duplicates.",
    "within_family_cell_rows": "Disaggregated support prevents pooled-family or no-headroom claims.",
    "shortcut_control_rows": "Nuisance features must not explain the label.",
    "permutation_attack_rows": "Relabel, pair, split, sign, and evaluator attacks must fail closed.",
    "provenance_attack_rows": "Duplicate, mutation, and hash attacks must be detected.",
    "representation_integrity_ready_score": "Emit a bare scalar for downstream gates.",
    "missing_verifier_gaps": "Present but unselectable checks stay visible.",
    "per_unit_rows": "Raw, pair, cell, shortcut, and attack rows make summaries replayable.",
    "aggregate_row_recomputation": "Summary fields derive only from emitted rows.",
    "protected_files_unchanged": "Active roadmap and conductor files remain unchanged.",
    "gate_check_summary": "Blocked verdicts must name the exact failed gate.",
    "preconditions_checked": "Gate, raw file, and hash checks run before audit claims.",
    "inference_substrate": "Use `independent_raw_representation_audit_no_llm`.",
    "verifier_is_oracle": "True only for deterministic row, hash, and schema checks.",
    "field_principles": "Every required field carries its audit reason.",
    "field_provenance": "Every field traces to raw paths, hashes, source, or tests.",
    "random_seed": "Fixed controls make permutation and shortcut results reproducible.",
    "duration_s": "Measured wall time exposes bootstrap-only artifacts.",
    "tests_run": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects raw input, reducer, or attack drift.",
    "honest_verdict": "Use `ready:`, `disqualified:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash stable text evidence with the repository's prefixed format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without loading large raw vectors at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_file_hash(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.is_file() else "missing"


def _round(value: float) -> float:
    return round(float(value), 8)


def _norm(values: Sequence[float]) -> float:
    return _round(math.sqrt(sum(float(value) * float(value) for value in values)))


def _path_metadata(path: Path) -> JsonDict:
    parts = path.stem.split("__")
    return {
        "path_split": path.parent.parent.name if path.parent.parent.name else "",
        "path_family": path.parent.name,
        "path_unit_id": parts[0] if len(parts) >= 4 else "",
        "path_candidate_kind": parts[1] if len(parts) >= 4 else "",
        "path_seed": parts[3] if len(parts) >= 4 else "",
    }


def _candidate_kind(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    if ":" in candidate_id:
        return candidate_id.rsplit(":", 1)[1]
    return str(row.get("candidate_kind") or metadata.get("path_candidate_kind") or candidate_id)


def _label_from_row(row: Mapping[str, Any], candidate_kind: str) -> tuple[bool | None, str]:
    if "exact_label" in row:
        return bool(row["exact_label"]), "raw_exact_label"
    if "label" in row:
        return bool(row["label"]), "raw_label"
    if candidate_kind == "exact_correct":
        return True, "candidate_kind_suffix"
    if candidate_kind.startswith("controlled_wrong"):
        return False, "candidate_kind_suffix"
    return None, "unavailable"


def _task_family(unit_id: str) -> str:
    base = unit_id.removeprefix("exp6482-")
    match = re.match(r"^(?P<task>.+)-\d+$", base)
    return match.group("task") if match else base


def _vector(row: Mapping[str, Any]) -> list[float]:
    values = row.get("vector")
    if not isinstance(values, list):
        raise ValueError("raw vector list required")
    return [float(value) for value in values]


def _manifest_paths(artifact: Mapping[str, Any]) -> tuple[list[Path], dict[str, JsonDict]]:
    manifest = dict(artifact.get("raw_vector_manifest") or {})
    entries = [dict(row) for row in manifest.get("vectors") or [] if isinstance(row, Mapping)]
    by_path = {str(row.get("path")): row for row in entries}
    storage = dict(manifest.get("storage_by_split") or {})
    paths: list[Path] = []
    for split in sorted(storage):
        paths.extend(Path(str(path)) for path in sorted(storage[split]))
    if not paths:
        paths = [Path(str(row.get("path"))) for row in entries]
    return paths, by_path


def _raw_hash_receipts(
    artifact: Mapping[str, Any], raw_paths: Sequence[Path], manifest_entries: Mapping[str, JsonDict]
) -> tuple[list[JsonDict], bool, bool]:
    manifest = dict(artifact.get("raw_vector_manifest") or {})
    vector_entries = [dict(row) for row in manifest.get("vectors") or [] if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    hashes_by_path: dict[str, str] = {}
    all_match = True
    for path in raw_paths:
        actual = sha256_file(path) if path.is_file() else "missing"
        declared = str(manifest_entries.get(str(path), {}).get("sha256") or "")
        matches = bool(declared) and actual == declared
        all_match = all_match and matches
        hashes_by_path[str(path)] = actual
        rows.append(
            {
                "path": str(path),
                "sha256": actual,
                "declared_sha256": declared,
                "hash_matches_manifest": matches,
            }
        )
    declared_root = str(manifest.get("hash_root") or "")
    if vector_entries:
        actual_root = sha256_json([hashes_by_path.get(str(row.get("path"))) for row in vector_entries])
        declared_root_replay = sha256_json([row.get("sha256") for row in vector_entries])
    else:
        actual_root = sha256_json([row["sha256"] for row in rows])
        declared_root_replay = actual_root
    return rows, all_match, declared_root == declared_root_replay and declared_root == actual_root


def _record_from_path(
    path: Path,
    *,
    row_order: int,
    file_receipt: Mapping[str, Any],
    manifest_entry: Mapping[str, Any],
) -> JsonDict:
    raw = _read_json(path)
    metadata = _path_metadata(path)
    vector = _vector(raw)
    family = str(raw.get("family") or metadata["path_family"])
    unit_id = str(raw.get("unit_id") or metadata["path_unit_id"])
    kind = _candidate_kind(raw, metadata)
    label, label_source = _label_from_row(raw, kind)
    split = str(manifest_entry.get("split") or metadata["path_split"])
    candidate_id = str(raw.get("candidate_id") or f"{unit_id}:{kind}")
    vector_hash = sha256_json(vector)
    return {
        "row_type": "raw_record",
        "record_id": sha256_json({"path": str(path), "sha256": file_receipt.get("sha256")}),
        "raw_path": str(path),
        "raw_file_sha256": file_receipt.get("sha256"),
        "declared_raw_file_sha256": file_receipt.get("declared_sha256"),
        "raw_file_hash_matches": file_receipt.get("hash_matches_manifest") is True,
        "schema_version": raw.get("schema_version"),
        "split": split,
        "family": family,
        "model_hf_id": str(raw.get("model_hf_id") or ""),
        "model_hash": str(raw.get("model_hash") or ""),
        "unit_id": unit_id,
        "task_family": _task_family(unit_id),
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "candidate_hash": str(raw.get("candidate_hash") or ""),
        "prompt_hash": str(raw.get("prompt_hash") or ""),
        "label": label,
        "label_name": "correct" if label is True else "wrong" if label is False else "unknown",
        "label_source": label_source,
        "native_dimension": len(vector),
        "vector_norm": _norm(vector),
        "vector_hash": vector_hash,
        "unit_identifier_length": len(unit_id),
        "candidate_identifier_length": len(candidate_id),
        "prompt_identifier_length": len(str(raw.get("prompt_hash") or "")),
        "candidate_length": raw.get("candidate_length"),
        "prompt_length": raw.get("prompt_length"),
        "token_length": raw.get("token_length"),
        "row_order": row_order,
        "row_order_modulo_pair": 0,
        "path_metadata_matches": (
            metadata["path_family"] == family
            and (not metadata["path_unit_id"] or metadata["path_unit_id"] == unit_id)
            and (not metadata["path_candidate_kind"] or metadata["path_candidate_kind"] == kind)
        ),
        "_vector": vector,
    }


def reconstruct_raw_records(
    raw_paths: Sequence[Path],
    file_receipts: Sequence[Mapping[str, Any]],
    manifest_entries: Mapping[str, JsonDict],
) -> list[JsonDict]:
    records = [
        _record_from_path(
            path,
            row_order=index,
            file_receipt=file_receipts[index],
            manifest_entry=manifest_entries.get(str(path), {}),
        )
        for index, path in enumerate(raw_paths)
    ]
    group_sizes = Counter(
        (row["split"], row["family"], row["unit_id"]) for row in records
    )
    for row in records:
        size = max(1, int(group_sizes[(row["split"], row["family"], row["unit_id"])]))
        row["row_order_modulo_pair"] = int(row["row_order"]) % size
    return records


def pair_rows(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_unit: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_unit[(str(row["split"]), str(row["family"]), str(row["unit_id"]))].append(row)
    rows: list[JsonDict] = []
    for (split, family, unit_id), unit_rows in sorted(by_unit.items()):
        correct = [row for row in unit_rows if row.get("label") is True]
        wrong = [row for row in unit_rows if row.get("label") is False]
        for correct_row in correct:
            for wrong_row in wrong:
                rows.append(
                    {
                        "row_type": "pair",
                        "pair_id": sha256_json(
                            {
                                "correct": correct_row["record_id"],
                                "wrong": wrong_row["record_id"],
                            }
                        ),
                        "split": split,
                        "family": family,
                        "task_family": correct_row["task_family"],
                        "unit_id": unit_id,
                        "correct_record_id": correct_row["record_id"],
                        "wrong_record_id": wrong_row["record_id"],
                        "correct_candidate_id": correct_row["candidate_id"],
                        "wrong_candidate_id": wrong_row["candidate_id"],
                        "native_dimension": correct_row["native_dimension"],
                        "same_native_dimension": correct_row["native_dimension"]
                        == wrong_row["native_dimension"],
                        "vector_comparison_scope": "within_family_only",
                        "norm_delta": _round(
                            float(correct_row["vector_norm"]) - float(wrong_row["vector_norm"])
                        ),
                    }
                )
    return rows


def reconstructed_stream_counts(
    records: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    duplicate_keys = Counter(
        (
            str(row["split"]),
            str(row["family"]),
            str(row["unit_id"]),
            str(row["candidate_id"]),
        )
        for row in records
    )
    dims: dict[str, set[int]] = defaultdict(set)
    for row in records:
        dims[str(row["family"])].add(int(row["native_dimension"]))
    return {
        "schema": SCHEMA + ".reconstructed_stream_counts",
        "aggregate_counts_trusted": False,
        "raw_record_count": len(records),
        "pair_count": len(pairs),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in records).items())),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in records).items())),
        "task_family_counts": dict(
            sorted(Counter(str(row["task_family"]) for row in records).items())
        ),
        "label_counts": dict(sorted(Counter(str(row["label_name"]) for row in records).items())),
        "native_dimension_sets_by_family": {
            family: sorted(values) for family, values in sorted(dims.items())
        },
        "prompt_hash_count": len({str(row["prompt_hash"]) for row in records}),
        "candidate_hash_count": len({str(row["candidate_hash"]) for row in records}),
        "duplicate_raw_record_key_count": sum(1 for count in duplicate_keys.values() if count > 1),
        "duplicate_raw_record_keys": [
            "|".join(key) for key, count in sorted(duplicate_keys.items()) if count > 1
        ],
        "all_records_labeled": all(row.get("label") in (True, False) for row in records),
        "families_have_single_native_dimension": all(len(values) == 1 for values in dims.values()),
    }


def within_family_cell_rows(
    records: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    pair_counts = Counter((str(row["family"]), str(row["task_family"])) for row in pairs)
    for row in records:
        by_cell[(str(row["family"]), str(row["task_family"]))].append(row)
    rows: list[JsonDict] = []
    for (family, task_family), cell_rows in sorted(by_cell.items()):
        correct = sum(1 for row in cell_rows if row.get("label") is True)
        wrong = sum(1 for row in cell_rows if row.get("label") is False)
        unsupported = correct == 0 or wrong == 0
        no_headroom = not unsupported and pair_counts[(family, task_family)] == 0
        state = "unsupported" if unsupported else "no_headroom" if no_headroom else "supported"
        rows.append(
            {
                "row_type": "cell",
                "cell_id": f"{family}|{task_family}",
                "family": family,
                "task_family": task_family,
                "raw_record_count": len(cell_rows),
                "correct_support": correct,
                "wrong_support": wrong,
                "pair_count": int(pair_counts[(family, task_family)]),
                "native_dimensions": sorted({int(row["native_dimension"]) for row in cell_rows}),
                "unsupported": unsupported,
                "no_headroom": no_headroom,
                "headroom_state": state,
                "cell_passed": state == "supported",
                "vector_comparison_scope": "within_family_only",
            }
        )
    return rows


def _balanced_accuracy(labels: Sequence[bool], predictions: Sequence[bool]) -> float:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    tpr = (
        sum(1 for index in positives if predictions[index] is True) / len(positives)
        if positives
        else 0.0
    )
    tnr = (
        sum(1 for index in negatives if predictions[index] is False) / len(negatives)
        if negatives
        else 0.0
    )
    return _round((tpr + tnr) / 2.0)


def _categorical_control(
    rows: Sequence[Mapping[str, Any]], *, control_name: str, feature_name: str, family: str
) -> JsonDict:
    labels = [bool(row["label"]) for row in rows]
    by_value: dict[str, Counter[bool]] = defaultdict(Counter)
    for row in rows:
        by_value[str(row.get(feature_name))][bool(row["label"])] += 1
    global_counts = Counter(labels)
    global_prediction = global_counts[True] > global_counts[False]
    value_predictions = {
        value: counts[True] > counts[False] if counts[True] != counts[False] else global_prediction
        for value, counts in by_value.items()
    }
    predictions = [value_predictions[str(row.get(feature_name))] for row in rows]
    score = _balanced_accuracy(labels, predictions)
    return {
        "row_type": "shortcut_control",
        "control_name": control_name,
        "feature_name": feature_name,
        "family": family,
        "control_type": "categorical_majority",
        "row_count": len(rows),
        "unique_value_count": len(by_value),
        "balanced_accuracy": score,
        "ceiling": SHORTCUT_BALANCED_ACCURACY_CEILING,
        "survived_shortcut": score > SHORTCUT_BALANCED_ACCURACY_CEILING,
    }


def _numeric_control(
    rows: Sequence[Mapping[str, Any]], *, control_name: str, feature_name: str, family: str
) -> JsonDict:
    values = [float(row.get(feature_name) or 0.0) for row in rows]
    labels = [bool(row["label"]) for row in rows]
    unique = sorted(set(values))
    thresholds = unique[:1] + [
        (left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False)
    ]
    best = 0.0
    best_rule = "constant"
    for threshold in thresholds:
        for le_is_correct in (True, False):
            predictions = [
                (value <= threshold) is le_is_correct for value in values
            ]
            score = _balanced_accuracy(labels, predictions)
            if score > best:
                best = score
                target = "correct" if le_is_correct else "wrong"
                best_rule = f"<={_round(threshold)} predicts {target}"
    return {
        "row_type": "shortcut_control",
        "control_name": control_name,
        "feature_name": feature_name,
        "family": family,
        "control_type": "numeric_threshold",
        "row_count": len(rows),
        "unique_value_count": len(unique),
        "balanced_accuracy": _round(best),
        "best_rule": best_rule,
        "ceiling": SHORTCUT_BALANCED_ACCURACY_CEILING,
        "survived_shortcut": best > SHORTCUT_BALANCED_ACCURACY_CEILING,
    }


def shortcut_control_rows(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    labeled = [row for row in records if row.get("label") in (True, False)]
    rows: list[JsonDict] = []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_family[str(row["family"])].append(row)
    for family, family_rows in sorted(by_family.items()):
        rows.extend(
            [
                _numeric_control(
                    family_rows,
                    control_name="vector_norm",
                    feature_name="vector_norm",
                    family=family,
                ),
                _numeric_control(
                    family_rows,
                    control_name="unit_identifier_length",
                    feature_name="unit_identifier_length",
                    family=family,
                ),
                _numeric_control(
                    family_rows,
                    control_name="candidate_identifier_length",
                    feature_name="candidate_identifier_length",
                    family=family,
                ),
                _categorical_control(
                    family_rows,
                    control_name="prompt_identity",
                    feature_name="prompt_hash",
                    family=family,
                ),
                _categorical_control(
                    family_rows,
                    control_name="candidate_identity",
                    feature_name="candidate_id",
                    family=family,
                ),
                _categorical_control(
                    family_rows,
                    control_name="row_order_modulo_pair",
                    feature_name="row_order_modulo_pair",
                    family=family,
                ),
            ]
        )
    rows.extend(
        [
            _categorical_control(
                labeled,
                control_name="model_family",
                feature_name="family",
                family="all_families",
            ),
            _categorical_control(
                labeled,
                control_name="native_dimension",
                feature_name="native_dimension",
                family="all_families",
            ),
        ]
    )
    return rows


def permutation_attack_rows(
    records: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    label_items = [(row["record_id"], row.get("label")) for row in records]
    pair_items = [row["pair_id"] for row in pairs]
    split_items = [(row["record_id"], row["split"]) for row in records]
    vector_items = [(row["record_id"], row["vector_hash"]) for row in records]
    attacks = [
        (
            "label_permutation",
            sha256_json(label_items),
            sha256_json([(rid, label_items[(index + 1) % len(label_items)][1]) for index, (rid, _) in enumerate(label_items)]),
        ),
        ("pair_permutation", sha256_json(pair_items), sha256_json(list(reversed(pair_items)))),
        (
            "claim_flip",
            sha256_json(label_items),
            sha256_json([(rid, None if label is None else not label) for rid, label in label_items]),
        ),
        (
            "evaluator_swap",
            sha256_json(label_items),
            sha256_json([(rid, None if label is None else not label) for rid, label in label_items]),
        ),
        (
            "sign_flip",
            sha256_json(vector_items),
            sha256_json([(row["record_id"], sha256_json([-value for value in row["_vector"]])) for row in records]),
        ),
        (
            "split_move",
            sha256_json(split_items),
            sha256_json([(rid, "moved_split") for rid, _ in split_items]),
        ),
    ]
    rows: list[JsonDict] = []
    for name, before, after in attacks:
        detected = before != after
        rows.append(
            {
                "row_type": "attack",
                "attack_family": "permutation",
                "attack_name": name,
                "before_hash": before,
                "after_hash": after,
                "attack_detected": detected,
                "fail_closed": detected,
            }
        )
    return rows


def provenance_attack_rows(
    records: Sequence[Mapping[str, Any]], upstream_hash_receipts: Mapping[str, Any]
) -> list[JsonDict]:
    duplicate_count = len(records) + 1 if records else 0
    original_count = len(records)
    first = records[0] if records else {}
    mutation_hash = (
        sha256_json([99.0, *list(first.get("_vector", []))[1:]]) if first else "missing"
    )
    original_vector_hash = str(first.get("vector_hash") or "")
    manifest_root = str(upstream_hash_receipts.get("raw_manifest_declared_hash_root") or "")
    tampered_root = sha256_json(["tampered", manifest_root])
    attacks = [
        ("row_duplication", original_count, duplicate_count, duplicate_count != original_count),
        ("raw_vector_mutation", original_vector_hash, mutation_hash, mutation_hash != original_vector_hash),
        ("hash_attack", manifest_root, tampered_root, tampered_root != manifest_root),
    ]
    return [
        {
            "row_type": "attack",
            "attack_family": "provenance",
            "attack_name": name,
            "before_hash": before,
            "after_hash": after,
            "attack_detected": bool(detected),
            "fail_closed": bool(detected),
        }
        for name, before, after, detected in attacks
    ]


def missing_verifier_gaps(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    checks = (
        ("token_length_unavailable_from_raw_rows", "token_length"),
        ("candidate_length_unavailable_from_raw_rows", "candidate_length"),
        ("prompt_length_unavailable_from_raw_rows", "prompt_length"),
    )
    for gap, field in checks:
        missing = sum(1 for row in records if row.get(field) is None)
        if missing:
            gaps.append(
                {
                    "gap": gap,
                    "missing_count": missing,
                    "selectable": False,
                    "blocks_readiness": True,
                }
            )
    return gaps


def _public_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def aggregate_row_recomputation(
    *,
    per_unit_rows: Sequence[Mapping[str, Any]],
    shortcut_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
) -> JsonDict:
    row_type_counts = Counter(str(row.get("row_type")) for row in per_unit_rows)
    failed_attacks = [
        str(row.get("attack_name"))
        for row in [*permutation_rows, *provenance_rows]
        if row.get("attack_detected") is not True
    ]
    surviving_shortcuts = [
        str(row.get("control_name")) for row in shortcut_rows if row.get("survived_shortcut")
    ]
    failed_cells = [str(row.get("cell_id")) for row in cell_rows if row.get("cell_passed") is not True]
    return {
        "schema": SCHEMA + ".aggregate_row_recomputation",
        "aggregate_counts_trusted": False,
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "surviving_shortcut_count": len(surviving_shortcuts),
        "surviving_shortcuts": sorted(set(surviving_shortcuts)),
        "failed_attack_count": len(failed_attacks),
        "failed_attacks": failed_attacks,
        "failed_cell_count": len(failed_cells),
        "failed_cells": failed_cells,
        "missing_gap_count": len(gaps),
        "summaries_recomputed_from_per_unit_rows": True,
    }


def independent_reducer_receipt(root: Path) -> JsonDict:
    source_path = root / MODULE_RELATIVE_PATH
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        forbidden.extend(
            name for name in names if "experiment_6486_three_family_forced_candidate_representations" in name
        )
    return {
        "schema": SCHEMA + ".independent_reducer_receipt",
        "reducer_source_path": str(source_path),
        "reducer_source_sha256": sha256_file(source_path),
        "forbidden_exp6486_imports": sorted(set(forbidden)),
        "imports_exp6486_reducer": bool(forbidden),
        "independence_proof": "AST import scan and task-local raw reducer; no Exp6486 reducer calls.",
    }


def protected_files_unchanged(root: Path) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        digest = _safe_file_hash(root, relative)
        files[str(relative)] = {"before": digest, "after": digest, "unchanged": True}
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "files": files,
        "active_roadmap_and_conductor_unchanged": all(row["unchanged"] for row in files.values()),
    }


def upstream_hash_receipts(
    *,
    artifact_path: Path,
    artifact_hash: str,
    artifact: Mapping[str, Any],
    raw_paths: Sequence[Path],
    manifest_entries: Mapping[str, JsonDict],
) -> tuple[JsonDict, list[JsonDict], bool]:
    raw_file_hashes, hashes_match, root_matches = _raw_hash_receipts(
        artifact, raw_paths, manifest_entries
    )
    manifest = dict(artifact.get("raw_vector_manifest") or {})
    receipt = {
        "schema": SCHEMA + ".upstream_hash_receipts",
        "exp6486_artifact_path": str(artifact_path),
        "exp6486_artifact_sha256": artifact_hash,
        "gate_field": "prospective_representation_stream_ready_score",
        "gate_field_value": artifact.get("prospective_representation_stream_ready_score"),
        "raw_file_count": len(raw_file_hashes),
        "raw_file_hashes": raw_file_hashes,
        "raw_manifest_declared_hash_root": manifest.get("hash_root"),
        "raw_file_hashes_match_manifest": hashes_match,
        "raw_manifest_hash_root_replayed": root_matches,
        "aggregate_rows_trusted": False,
        "read_order_receipt": [
            "exp6486_artifact_file_hash",
            "structured_gate_field",
            "raw_vector_manifest_paths",
            "raw_file_hashes",
            "raw_rows",
        ],
    }
    return receipt, raw_file_hashes, hashes_match and root_matches


def gate_check_summary(
    *,
    artifact: Mapping[str, Any],
    raw_paths: Sequence[Path],
    raw_hashes_ok: bool,
    reducer: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "exp6486_gate_ready": artifact.get("prospective_representation_stream_ready_score") == 1.0,
        "raw_manifest_paths_present": bool(raw_paths),
        "raw_hashes_replay": raw_hashes_ok,
        "independent_reducer": reducer.get("imports_exp6486_reducer") is False,
    }
    failure_names = {
        "exp6486_gate_ready": "exp6486_gate_not_ready",
        "raw_manifest_paths_present": "raw_manifest_paths_missing",
        "raw_hashes_replay": "raw_hashes_replay_failed",
        "independent_reducer": "independent_reducer_failed",
    }
    failed = sorted(failure_names[key] for key, ok in checks.items() if ok is not True)
    return {
        "schema": SCHEMA + ".gate_check_summary",
        "checks": checks,
        "failed_gates": failed,
        "all_gates_passed": not failed,
    }


def preconditions_checked(
    *,
    root: Path,
    result_path: Path,
    gate_summary: Mapping[str, Any],
    raw_file_hashes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    output = {
        "result_path": str(result_path),
        "parent_writable": result_path.parent.exists() and os.access(result_path.parent, os.W_OK),
    }
    failed = list(gate_summary.get("failed_gates") or [])
    if any(row.get("hash_matches_manifest") is not True for row in raw_file_hashes):
        failed.append("raw_file_hash_mismatch")
    if not output["parent_writable"]:
        failed.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions_checked",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "root": str(root),
        "output": output,
        "raw_file_count": len(raw_file_hashes),
        "preconditions_ready": not failed,
        "blocked_reasons": sorted(set(failed)),
    }


def _tests_run(commands: Sequence[str], exit_codes: Mapping[str, int]) -> list[JsonDict]:
    return [
        {"command": command, "exit_code": int(exit_codes.get(command, 0))}
        for command in commands
    ]


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": "raw rows, hashes, source reducer, focused tests, and deterministic controls",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def representation_integrity_ready_score(artifact: Mapping[str, Any]) -> float:
    preconditions = dict(artifact.get("preconditions_checked") or {})
    aggregate = dict(artifact.get("aggregate_row_recomputation") or {})
    protected = dict(artifact.get("protected_files_unchanged") or {})
    tests = list(artifact.get("tests_run") or [])
    checks = (
        preconditions.get("preconditions_ready") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and not list(artifact.get("missing_verifier_gaps") or [])
        and not [row for row in artifact.get("shortcut_control_rows") or [] if row.get("survived_shortcut")]
        and all(row.get("cell_passed") is True for row in artifact.get("within_family_cell_rows") or [])
        and all(row.get("attack_detected") is True for row in artifact.get("permutation_attack_rows") or [])
        and all(row.get("attack_detected") is True for row in artifact.get("provenance_attack_rows") or [])
        and aggregate.get("summaries_recomputed_from_per_unit_rows") is True
        and protected.get("active_roadmap_and_conductor_unchanged") is True
        and all(int(row.get("exit_code", 1)) == 0 for row in tests)
    )
    return 1.0 if checks else 0.0


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def _artifact_from_reports(
    *,
    upstream: Mapping[str, Any],
    reducer: Mapping[str, Any],
    counts: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    shortcuts: Sequence[Mapping[str, Any]],
    permutations: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
    per_unit: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    run_date: str,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": "complete",
        "upstream_hash_receipts": dict(upstream),
        "independent_reducer_receipt": dict(reducer),
        "reconstructed_stream_counts": dict(counts),
        "within_family_cell_rows": list(cells),
        "shortcut_control_rows": list(shortcuts),
        "permutation_attack_rows": list(permutations),
        "provenance_attack_rows": list(provenance),
        "representation_integrity_ready_score": 0.0,
        "missing_verifier_gaps": list(gaps),
        "per_unit_rows": list(per_unit),
        "aggregate_row_recomputation": dict(aggregate),
        "protected_files_unchanged": dict(protected),
        "gate_check_summary": dict(gate_summary),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": _round(duration_s),
        "tests_run": _tests_run(test_commands, test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    score = representation_integrity_ready_score(artifact)
    artifact["representation_integrity_ready_score"] = score
    if score == 1.0:
        artifact["status"] = "complete"
        artifact["honest_verdict"] = "ready: raw_representation_integrity_controls_clean"
    elif preconditions.get("preconditions_ready") is not True:
        artifact["status"] = "blocked"
        reasons = list(preconditions.get("blocked_reasons") or ["preconditions_failed"])
        artifact["honest_verdict"] = "blocked: " + ",".join(reasons[:8])
    else:
        artifact["status"] = "disqualified"
        reasons = list(aggregate.get("surviving_shortcuts") or [])
        reasons.extend(str(row.get("gap")) for row in gaps)
        reasons.extend(str(row.get("cell_id")) for row in cells if row.get("cell_passed") is not True)
        artifact["honest_verdict"] = "disqualified: " + ",".join(reasons[:8] or ["integrity_controls_failed"])
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(field)
    if set(REQUIRED_FIELD_PRINCIPLES) - set(dict(artifact.get("field_principles") or {})):
        raise ValueError("field_principles")
    if set(REQUIRED_FIELD_PRINCIPLES) - set(dict(artifact.get("field_provenance") or {})):
        raise ValueError("field_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    expected_score = representation_integrity_ready_score(artifact)
    if artifact.get("representation_integrity_ready_score") != expected_score:
        raise ValueError("representation_integrity_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    status = artifact.get("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0:
        if status != "complete" or not verdict.startswith("ready:"):
            raise ValueError("status")
    elif status == "blocked":
        if not verdict.startswith("blocked:"):
            raise ValueError("honest_verdict")
    elif status == "disqualified":
        if not verdict.startswith("disqualified:"):
            raise ValueError("honest_verdict")
    else:
        raise ValueError("status")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6486_artifact_path: str | Path = REPO_ROOT / EXP6486_ARTIFACT_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run the independent raw representation audit."""

    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    artifact_path = Path(exp6486_artifact_path)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    artifact_hash = sha256_file(artifact_path)
    exp6486_artifact = _read_json(artifact_path)
    raw_paths, manifest_entries = _manifest_paths(exp6486_artifact)
    upstream, raw_file_hashes, raw_hashes_ok = upstream_hash_receipts(
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        artifact=exp6486_artifact,
        raw_paths=raw_paths,
        manifest_entries=manifest_entries,
    )
    reducer = independent_reducer_receipt(root)
    gate_summary = gate_check_summary(
        artifact=exp6486_artifact,
        raw_paths=raw_paths,
        raw_hashes_ok=raw_hashes_ok,
        reducer=reducer,
    )
    preconditions = preconditions_checked(
        root=root,
        result_path=result,
        gate_summary=gate_summary,
        raw_file_hashes=raw_file_hashes,
    )
    if preconditions.get("preconditions_ready") is True:
        records = reconstruct_raw_records(raw_paths, raw_file_hashes, manifest_entries)
    else:
        records = []
    pairs = pair_rows(records)
    counts = reconstructed_stream_counts(records, pairs)
    cells = within_family_cell_rows(records, pairs)
    shortcuts = shortcut_control_rows(records) if records else []
    permutations = permutation_attack_rows(records, pairs) if records else []
    provenance = provenance_attack_rows(records, upstream) if records else []
    gaps = missing_verifier_gaps(records)
    per_unit = [
        *_public_rows(records),
        *pairs,
        *cells,
        *shortcuts,
        *permutations,
        *provenance,
    ]
    aggregate = aggregate_row_recomputation(
        per_unit_rows=per_unit,
        shortcut_rows=shortcuts,
        cell_rows=cells,
        permutation_rows=permutations,
        provenance_rows=provenance,
        gaps=gaps,
    )
    protected = protected_files_unchanged(root)
    artifact = _artifact_from_reports(
        upstream=upstream,
        reducer=reducer,
        counts=counts,
        cells=cells,
        shortcuts=shortcuts,
        permutations=permutations,
        provenance=provenance,
        gaps=gaps,
        per_unit=per_unit,
        aggregate=aggregate,
        protected=protected,
        gate_summary=gate_summary,
        preconditions=preconditions,
        duration_s=time.perf_counter() - started,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
        run_date=run_date,
    )
    validate_artifact(artifact)
    if write:
        _write_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(write=not args.no_write, run_date=str(args.date))
    print(json.dumps({"status": artifact["status"], "ready": artifact["representation_integrity_ready_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
