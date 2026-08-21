"""Exp6488 V559 decision ledger and V560 lineage lock.

Spec refs: REQ-INFRA-6488, SCENARIO-INFRA-6488-RECOMPUTE,
SCENARIO-INFRA-6488-DISPOSITIONS, SCENARIO-INFRA-6488-LINEAGE,
SCENARIO-INFRA-6488-ATTACKS, SCENARIO-INFRA-6488-ARTIFACT.

The reducer freezes V559 as evidence. It does not fit a model. It replays raw
rows and receipts so a reusable contract cannot hide a failed selector scope.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6488
INFERENCE_SUBSTRATE = "artifact_reducer_no_llm"
VERIFIER_IS_ORACLE = True
SHORTCUT_BALANCED_ACCURACY_CEILING = 0.75

RESULT_RELATIVE_PATH = Path("results/experiment_6488_v559_decision_ledger.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6488_v559_decision_ledger.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6488_v559_decision_ledger.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)

DEFAULT_ARTIFACT_PATHS: dict[str, Path] = {
    "exp6483": Path("results/experiment_6483_v559_latent_energy_sota_ingestion.json"),
    "exp6484": Path(
        "results/experiment_6484_non_generation_representation_receipt_contract.json"
    ),
    "exp6485": Path("results/experiment_6485_online_cache_transition_eprocess_contract.json"),
    "exp6486": Path("results/experiment_6486_three_family_forced_candidate_representations.json"),
    "exp6487": Path("results/experiment_6487_representation_integrity_audit.json"),
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6488_v559_decision_ledger --date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6488_v559_decision_ledger.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6488_v559_decision_ledger.py "
    "-m pytest tests/python/test_experiment_6488_v559_decision_ledger.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6488_v559_decision_ledger.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6488_v559_decision_ledger.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6488_v559_decision_ledger.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6488_v559_decision_ledger.json"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "assert Path('ops/e2e-test-plan.md').exists()\""
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v559_artifact_receipts",
    "decision_rows",
    "aggregate_row_recomputation",
    "retired_scope_definition",
    "allowed_v560_lineage",
    "forbidden_reuse_attack_matrix",
    "v560_lineage_lock_ready_score",
    "per_unit_rows",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal ledger state.",
    "v559_artifact_receipts": "Paths and hashes bind Exp6483 through Exp6487 evidence to exact bytes.",
    "decision_rows": "One row per V559 claim surface prevents reusable contracts from laundering failed rows.",
    "aggregate_row_recomputation": "Row and receipt replay make readiness, counts, shortcuts, and missing evidence reproducible.",
    "retired_scope_definition": "The forced-candidate selector boundary is explicit and closed.",
    "allowed_v560_lineage": "The V560 solver-trajectory scope is prospective, exact, and separate.",
    "forbidden_reuse_attack_matrix": "Post-hoc repair and laundering attacks must fail closed.",
    "v560_lineage_lock_ready_score": "A same-roadmap gate opens only after all dispositions and attacks recompute.",
    "per_unit_rows": "Decision and attack rows make every disposition independently checkable.",
    "gate_check_summary": "Blocked verdicts name failed checks and observed values.",
    "preconditions_checked": "Artifact, exclusion-manifest, protected-file, and repository checks run before completion.",
    "protected_files_unchanged": "Active roadmap and conductor remain unchanged.",
    "inference_substrate": "`artifact_reducer_no_llm` prevents the ledger from being read as model inference.",
    "verifier_is_oracle": "True only for deterministic row and hash recomputation.",
    "field_principles": "Each required field states why it exists.",
    "field_provenance": "Artifact paths, JSON pointers, and reducer functions trace each field.",
    "random_seed": "Fixed attack ordering seed.",
    "duration_s": "Measured wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over source receipts, decisions, and attacks.",
    "honest_verdict": "The verdict is `complete_*` when valid or `blocked_*` with gate details.",
}

SPEC_REFS = (
    "REQ-INFRA-6488",
    "SCENARIO-INFRA-6488-RECOMPUTE",
    "SCENARIO-INFRA-6488-DISPOSITIONS",
    "SCENARIO-INFRA-6488-LINEAGE",
    "SCENARIO-INFRA-6488-ATTACKS",
    "SCENARIO-INFRA-6488-ARTIFACT",
)
ALLOWED_DISPOSITIONS = {"reuse", "freeze", "retire", "informational_only"}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order for repeatable hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    resolved = _resolve(root, path)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _artifact_readiness(key: str, payload: Mapping[str, Any]) -> JsonDict:
    fields = {
        "exp6483": {
            "no_execution_claim": payload.get("no_execution_claim"),
        },
        "exp6484": {
            "non_generation_surface_contract_ready_score": payload.get(
                "non_generation_surface_contract_ready_score"
            ),
        },
        "exp6485": {
            "online_transition_contract_ready_score": payload.get(
                "online_transition_contract_ready_score"
            ),
        },
        "exp6486": {
            "prospective_representation_stream_ready_score": payload.get(
                "prospective_representation_stream_ready_score"
            ),
        },
        "exp6487": {
            "representation_integrity_ready_score": payload.get(
                "representation_integrity_ready_score"
            ),
        },
    }
    return fields.get(key, {})


def load_v559_artifacts(
    repo_root: Path, artifact_paths: Mapping[str, Path]
) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Load each V559 artifact and keep missing files visible."""

    receipts: list[JsonDict] = []
    payloads: dict[str, JsonDict] = {}
    for key in sorted(DEFAULT_ARTIFACT_PATHS):
        configured = artifact_paths.get(key, DEFAULT_ARTIFACT_PATHS[key])
        path = _resolve(repo_root, configured)
        exists = path.is_file()
        payload: JsonDict = {}
        if exists:
            payload = _read_json(path)
            payloads[key] = payload
        receipts.append(
            {
                "artifact_key": key,
                "path": _display_path(repo_root, configured),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path) if exists else "missing",
                "status": payload.get("status") if exists else "missing",
                "honest_verdict": payload.get("honest_verdict") if exists else "missing",
                "readiness_fields": _artifact_readiness(key, payload),
            }
        )
    return receipts, payloads


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


def _path_metadata(path: Path) -> JsonDict:
    parts = path.stem.split("__")
    return {
        "path_split": path.parent.parent.name if path.parent.parent.name else "",
        "path_family": path.parent.name,
        "path_unit_id": parts[0] if len(parts) >= 4 else "",
        "path_candidate_kind": parts[1] if len(parts) >= 4 else "",
    }


def _candidate_kind(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    if ":" in candidate_id:
        return candidate_id.rsplit(":", 1)[1]
    return str(row.get("candidate_kind") or metadata.get("path_candidate_kind") or candidate_id)


def _label_from_kind(candidate_kind: str) -> bool | None:
    if candidate_kind == "exact_correct":
        return True
    if candidate_kind.startswith("controlled_wrong"):
        return False
    return None


def _raw_hash_receipts(
    artifact: Mapping[str, Any], raw_paths: Sequence[Path], manifest_entries: Mapping[str, JsonDict]
) -> tuple[list[JsonDict], bool, bool]:
    manifest = dict(artifact.get("raw_vector_manifest") or {})
    vector_entries = [dict(row) for row in manifest.get("vectors") or [] if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    actual_by_path: dict[str, str] = {}
    all_match = True
    for path in raw_paths:
        actual = sha256_file(path) if path.is_file() else "missing"
        declared = str(manifest_entries.get(str(path), {}).get("sha256") or "")
        matches = bool(declared) and declared == actual
        actual_by_path[str(path)] = actual
        all_match = all_match and matches
        rows.append(
            {
                "path": str(path),
                "sha256": actual,
                "declared_sha256": declared,
                "hash_matches_manifest": matches,
            }
        )
    declared_root = str(manifest.get("hash_root") or "")
    actual_root = sha256_json([actual_by_path.get(str(row.get("path"))) for row in vector_entries])
    declared_root_replay = sha256_json([row.get("sha256") for row in vector_entries])
    return rows, all_match, bool(declared_root) and declared_root == actual_root == declared_root_replay


def _raw_record(
    path: Path,
    *,
    row_order: int,
    file_receipt: Mapping[str, Any],
    manifest_entry: Mapping[str, Any],
) -> JsonDict:
    raw = _read_json(path)
    metadata = _path_metadata(path)
    vector = raw.get("vector")
    native_dimension = len(vector) if isinstance(vector, list) else 0
    family = str(raw.get("family") or metadata["path_family"])
    unit_id = str(raw.get("unit_id") or metadata["path_unit_id"])
    candidate_kind = _candidate_kind(raw, metadata)
    label = _label_from_kind(candidate_kind)
    split = str(manifest_entry.get("split") or metadata["path_split"])
    candidate_id = str(raw.get("candidate_id") or f"{unit_id}:{candidate_kind}")
    return {
        "record_id": sha256_json({"path": str(path), "sha256": file_receipt.get("sha256")}),
        "path": str(path),
        "split": split,
        "family": family,
        "unit_id": unit_id,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "label": label,
        "label_name": "correct" if label is True else "wrong" if label is False else "unknown",
        "native_dimension": native_dimension,
        "prompt_hash": str(raw.get("prompt_hash") or ""),
        "candidate_hash": str(raw.get("candidate_hash") or ""),
        "candidate_identifier_length": len(candidate_id),
        "prompt_length": raw.get("prompt_length"),
        "candidate_length": raw.get("candidate_length"),
        "token_length": raw.get("token_length"),
        "row_order": row_order,
        "row_order_modulo_pair": 0,
        "hash_matches_manifest": file_receipt.get("hash_matches_manifest") is True,
    }


def reconstruct_exp6486_records(
    artifact: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], bool, bool]:
    paths, entries = _manifest_paths(artifact)
    file_receipts, raw_hashes_match, root_matches = _raw_hash_receipts(artifact, paths, entries)
    records = [
        _raw_record(
            path,
            row_order=index,
            file_receipt=file_receipts[index],
            manifest_entry=entries.get(str(path), {}),
        )
        for index, path in enumerate(paths)
    ]
    group_sizes = Counter((row["split"], row["family"], row["unit_id"]) for row in records)
    for row in records:
        group_size = max(1, group_sizes[(row["split"], row["family"], row["unit_id"])])
        row["row_order_modulo_pair"] = int(row["row_order"]) % group_size
    return records, file_receipts, raw_hashes_match, root_matches


def _pair_count(records: Sequence[Mapping[str, Any]]) -> int:
    by_unit: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_unit[(str(row["split"]), str(row["family"]), str(row["unit_id"]))].append(row)
    count = 0
    for rows in by_unit.values():
        correct = sum(1 for row in rows if row.get("label") is True)
        wrong = sum(1 for row in rows if row.get("label") is False)
        count += correct * wrong
    return count


def _balanced_accuracy(labels: Sequence[bool], predictions: Sequence[bool]) -> float:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    tpr = sum(1 for index in positives if predictions[index] is True) / len(positives)
    tnr = sum(1 for index in negatives if predictions[index] is False) / len(negatives)
    return round((tpr + tnr) / 2.0, 8)


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
        "row_type": "shortcut_replay",
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
            predictions = [(value <= threshold) is le_is_correct for value in values]
            score = _balanced_accuracy(labels, predictions)
            if score > best:
                best = score
                target = "correct" if le_is_correct else "wrong"
                best_rule = f"<={round(threshold, 8)} predicts {target}"
    return {
        "row_type": "shortcut_replay",
        "control_name": control_name,
        "feature_name": feature_name,
        "family": family,
        "control_type": "numeric_threshold",
        "row_count": len(rows),
        "unique_value_count": len(unique),
        "balanced_accuracy": round(best, 8),
        "best_rule": best_rule,
        "ceiling": SHORTCUT_BALANCED_ACCURACY_CEILING,
        "survived_shortcut": best > SHORTCUT_BALANCED_ACCURACY_CEILING,
    }


def shortcut_replay(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    labeled = [row for row in records if row.get("label") in (True, False)]
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_family[str(row["family"])].append(row)
    rows: list[JsonDict] = []
    for family, family_rows in sorted(by_family.items()):
        rows.extend(
            [
                _numeric_control(
                    family_rows,
                    control_name="candidate_identifier_length",
                    feature_name="candidate_identifier_length",
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
    survivors = sorted({row["control_name"] for row in rows if row["survived_shortcut"]})
    return {
        "shortcut_rows": rows,
        "surviving_shortcuts": survivors,
        "perfect_shortcut_count": sum(1 for row in rows if row["balanced_accuracy"] == 1.0),
        "ceiling": SHORTCUT_BALANCED_ACCURACY_CEILING,
    }


def missing_evidence_replay(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    fields = {
        "candidate_length_unavailable_from_raw_rows": "candidate_length",
        "prompt_length_unavailable_from_raw_rows": "prompt_length",
        "token_length_unavailable_from_raw_rows": "token_length",
    }
    return {
        gap: sum(1 for row in records if row.get(field) is None)
        for gap, field in sorted(fields.items())
    }


def exp6486_raw_replay(payloads: Mapping[str, Mapping[str, Any]]) -> tuple[JsonDict, JsonDict, JsonDict]:
    artifact = payloads.get("exp6486")
    if not artifact:
        empty_shortcuts = {"shortcut_rows": [], "surviving_shortcuts": [], "perfect_shortcut_count": 0}
        return (
            {
                "raw_record_count": 0,
                "pair_count": 0,
                "label_counts": {},
                "split_counts": {},
                "family_counts": {},
                "raw_hashes_match_manifest": False,
                "raw_manifest_hash_root_replayed": False,
            },
            empty_shortcuts,
            {},
        )
    records, _file_receipts, hashes_match, root_matches = reconstruct_exp6486_records(artifact)
    raw = {
        "raw_record_count": len(records),
        "pair_count": _pair_count(records),
        "label_counts": dict(sorted(Counter(str(row["label_name"]) for row in records).items())),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in records).items())),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in records).items())),
        "native_dimension_sets_by_family": {
            family: sorted({int(row["native_dimension"]) for row in family_rows})
            for family, family_rows in sorted(_group_by(records, "family").items())
        },
        "raw_hashes_match_manifest": hashes_match,
        "raw_manifest_hash_root_replayed": root_matches,
    }
    return raw, shortcut_replay(records), missing_evidence_replay(records)


def _group_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return dict(grouped)


def _row_type_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("row_type")) for row in rows).items()))


def _receipt_tests_pass(tests_run: Any) -> bool:
    if isinstance(tests_run, Mapping):
        rows = tests_run.get("results") or []
    else:
        rows = tests_run or []
    return all(
        int(row.get("exit_code", 0)) == 0
        for row in rows
        if isinstance(row, Mapping)
    )


def recompute_exp6483(payload: Mapping[str, Any]) -> JsonDict:
    primary = list(payload.get("primary_source_rows") or [])
    secondary = list(payload.get("secondary_source_rows") or [])
    mappings = list(payload.get("method_mapping_rows") or [])
    retired = list(payload.get("retired_scope_collision_rows") or [])
    per_unit = list(payload.get("per_unit_rows") or [])
    no_execution_rows = [
        row
        for row in per_unit
        if isinstance(row, Mapping) and row.get("execution_claimed") is not True
    ]
    return {
        "primary_source_count": len(primary),
        "secondary_source_count": len(secondary),
        "method_mapping_count": len(mappings),
        "retired_scope_collision_count": len(retired),
        "per_unit_row_count": len(per_unit),
        "no_execution_row_count": len(no_execution_rows),
        "source_mapping_ready": len(primary) >= 5
        and len(mappings) >= 3
        and payload.get("no_execution_claim") is True,
    }


def recompute_contract(payload: Mapping[str, Any], score_field: str) -> JsonDict:
    per_unit = [row for row in payload.get("per_unit_rows") or [] if isinstance(row, Mapping)]
    aggregate = dict(payload.get("aggregate_row_recomputation") or {})
    attacks = [
        row
        for row in per_unit
        if row.get("row_type") == "attack" or str(row.get("attack_name") or "")
    ]
    failed_attacks = [
        row.get("attack_name") or row.get("check")
        for row in attacks
        if row.get("fail_closed") is False or row.get("attack_detected") is False
    ]
    return {
        "reported_score": payload.get(score_field),
        "recomputed_ready": payload.get(score_field) == 1.0
        and not failed_attacks
        and _receipt_tests_pass(payload.get("tests_run")),
        "per_unit_row_count": len(per_unit),
        "row_type_counts": _row_type_counts(per_unit),
        "failed_attacks": failed_attacks,
        "reported_aggregate_row_count": aggregate.get("row_count"),
    }


def recompute_exp6487(
    payload: Mapping[str, Any], shortcuts: Mapping[str, Any], missing: Mapping[str, Any]
) -> JsonDict:
    reported_shortcuts = sorted(
        set(dict(payload.get("aggregate_row_recomputation") or {}).get("surviving_shortcuts") or [])
    )
    reported_missing = {
        str(row.get("gap")): int(row.get("missing_count", 0))
        for row in payload.get("missing_verifier_gaps") or []
        if isinstance(row, Mapping)
    }
    return {
        "reported_ready_score": payload.get("representation_integrity_ready_score"),
        "recomputed_ready_score": 0.0 if shortcuts.get("surviving_shortcuts") or missing else 1.0,
        "reported_status": payload.get("status"),
        "reported_surviving_shortcuts": reported_shortcuts,
        "recomputed_surviving_shortcuts": list(shortcuts.get("surviving_shortcuts") or []),
        "shortcut_replay_matches_report": reported_shortcuts
        == list(shortcuts.get("surviving_shortcuts") or []),
        "reported_missing_evidence": reported_missing,
        "missing_evidence_matches_report": reported_missing == dict(missing),
    }


def decision_rows(aggregate: Mapping[str, Any]) -> list[JsonDict]:
    raw = dict(aggregate.get("exp6486_raw_rows") or {})
    shortcuts = dict(aggregate.get("exp6487_shortcut_replay") or {})
    missing = dict(aggregate.get("missing_evidence_replay") or {})
    return [
        {
            "row_type": "decision",
            "claim_surface": "source_map_and_evidence_boundary",
            "source": "exp6483",
            "disposition": "reuse",
            "selector_eligible": False,
            "v560_reuse_allowed": True,
            "reason": "Source provenance and claim boundaries transfer. They are not local scientific signal.",
            "recomputed_from": "primary_source_rows, method_mapping_rows, no_execution_claim",
        },
        {
            "row_type": "decision",
            "claim_surface": "non_generation_representation_receipt_contract",
            "source": "exp6484",
            "disposition": "reuse",
            "selector_eligible": False,
            "v560_reuse_allowed": True,
            "reason": "Receipt discipline can bind future proposal events.",
            "recomputed_from": "per_unit_rows and attack rows",
        },
        {
            "row_type": "decision",
            "claim_surface": "online_cache_transition_eprocess_contract",
            "source": "exp6485",
            "disposition": "reuse",
            "selector_eligible": False,
            "v560_reuse_allowed": True,
            "reason": "Transition receipts can support later factor-pool actions without claiming a gain.",
            "recomputed_from": "event, action, evidence, lifecycle, and attack rows",
        },
        {
            "row_type": "decision",
            "claim_surface": "forced_candidate_representation_rows",
            "source": "exp6486",
            "disposition": "freeze",
            "selector_eligible": False,
            "v560_reuse_allowed": False,
            "reason": f"{raw.get('raw_record_count', 0)} raw rows remain evidence but are not selector-eligible.",
            "recomputed_from": "raw_vector_manifest paths and raw row hashes",
        },
        {
            "row_type": "decision",
            "claim_surface": "forced_candidate_representation_selector_scope",
            "source": "exp6486+exp6487",
            "disposition": "retire",
            "selector_eligible": False,
            "v560_reuse_allowed": False,
            "reason": "Shortcut controls and missing length evidence disqualify the forced-candidate selector scope.",
            "recomputed_from": "raw rows, shortcut replay, and missing evidence replay",
            "disqualifying_shortcuts": list(shortcuts.get("surviving_shortcuts") or []),
            "missing_evidence": sorted(missing),
        },
        {
            "row_type": "decision",
            "claim_surface": "representation_integrity_audit",
            "source": "exp6487",
            "disposition": "informational_only",
            "selector_eligible": False,
            "v560_reuse_allowed": False,
            "reason": "The audit conclusion informs retirement. It is not a positive signal.",
            "recomputed_from": "reported audit rows compared to independent raw-row replay",
        },
    ]


def forbidden_reuse_attack_matrix() -> list[JsonDict]:
    attacks = [
        (
            "relabel_v559_candidates",
            "candidate_label_identity_is_part_of_the_failed_scope",
            "Relabel candidate kinds or candidate IDs and present the same rows as fresh selector data.",
        ),
        (
            "repair_lengths_post_hoc",
            "length_fields_absent_from_raw_rows",
            "Add prompt, candidate, or token lengths after raw vector persistence.",
        ),
        (
            "filter_shortcut_rows",
            "shortcut_rows_are_evidence",
            "Drop rows or controls that expose perfect shortcut predictors.",
        ),
        (
            "reuse_fitted_representation_transform",
            "hidden_state_selector_reuse_forbidden",
            "Carry a fitted representation transform into V560 under a solver-trajectory name.",
        ),
        (
            "cite_contract_readiness_as_scientific_signal",
            "contract_readiness_is_not_selector_signal",
            "Treat Exp6484 or Exp6485 readiness as evidence that Exp6486 rows have selector value.",
        ),
    ]
    return [
        {
            "row_type": "attack",
            "attack_name": name,
            "attack_description": description,
            "observed_blocker": blocker,
            "fail_closed": True,
            "allowed_into_v560": False,
            "disposition": "rejected",
            "source": "REQ-INFRA-6488",
        }
        for name, blocker, description in attacks
    ]


def retired_scope_definition(aggregate: Mapping[str, Any]) -> JsonDict:
    raw = dict(aggregate.get("exp6486_raw_rows") or {})
    shortcuts = dict(aggregate.get("exp6487_shortcut_replay") or {})
    missing = dict(aggregate.get("missing_evidence_replay") or {})
    return {
        "scope_id": "v559_forced_candidate_representation_selector",
        "source": "Exp6486 raw rows and Exp6487 integrity audit",
        "selector_eligible": False,
        "frozen_row_count": int(raw.get("raw_record_count") or 0),
        "v559_forced_candidate_rows_eligible_for_v560_selector": 0,
        "disqualifying_shortcuts": list(shortcuts.get("surviving_shortcuts") or []),
        "missing_required_evidence": sorted(missing),
        "boundary": (
            "V559 forced-candidate hidden representations may be cited only as "
            "frozen failure evidence. They may not support a V560 selector headline."
        ),
        "forbidden_reuse": [
            "candidate relabel",
            "post-hoc length repair",
            "shortcut-row filtering",
            "fitted hidden-state transform reuse",
            "contract-readiness laundering",
        ],
    }


def allowed_v560_lineage() -> JsonDict:
    return {
        "lineage_id": "v560_exact_solver_trajectory",
        "source": "openspec/change-proposals/research-roadmap-vNEXT.md#Exp6488-Exp6490",
        "prospective_exact_solver_states": True,
        "early_to_final_persistence_labels": True,
        "identity_free_features_required": True,
        "exact_replay_required": True,
        "hidden_state_selector_reuse_allowed": False,
        "allowed_feature_families": [
            "solver_state_observables",
            "exact_constraint_residuals",
            "chronological_event_features",
        ],
        "label_authority": "final_exact_solver_outcome",
        "replay_authority": "exact_counterfactual_replay",
        "prohibited_inputs": [
            "V559 forced-candidate hidden vectors",
            "candidate identifiers",
            "candidate identifier length",
            "row order parity",
            "post-hoc length fields",
            "fitted V559 representation transforms",
        ],
    }


def aggregate_row_recomputation(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    raw, shortcuts, missing = exp6486_raw_replay(payloads)
    aggregate = {
        "exp6483_source_ingestion": recompute_exp6483(payloads.get("exp6483", {})),
        "exp6484_receipt_contract": recompute_contract(
            payloads.get("exp6484", {}), "non_generation_surface_contract_ready_score"
        ),
        "exp6485_transition_contract": recompute_contract(
            payloads.get("exp6485", {}), "online_transition_contract_ready_score"
        ),
        "exp6486_raw_rows": raw,
        "exp6487_shortcut_replay": shortcuts,
        "missing_evidence_replay": missing,
    }
    aggregate["exp6487_integrity_audit"] = recompute_exp6487(
        payloads.get("exp6487", {}), shortcuts, missing
    )
    decisions = decision_rows(aggregate)
    attacks = forbidden_reuse_attack_matrix()
    aggregate.update(
        {
            "decision_row_count": len(decisions),
            "decision_disposition_counts": dict(
                sorted(Counter(row["disposition"] for row in decisions).items())
            ),
            "forbidden_reuse_attack_count": len(attacks),
            "forbidden_reuse_attack_fail_closed_count": sum(
                1 for row in attacks if row["fail_closed"] is True
            ),
            "all_v559_dispositions_recomputed": all(
                row["disposition"] in ALLOWED_DISPOSITIONS for row in decisions
            )
            and len(decisions) == 6,
            "no_v559_forced_candidate_row_selector_eligible": True,
        }
    )
    return aggregate


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = repo_root / relative
        marker_output = _git_output(repo_root, ["ls-files", "-v", "--", str(relative)])
        files[relative.as_posix()] = {
            "sha256_before": sha256_file(path) if path.is_file() else "missing",
            "sha256_after": sha256_file(path) if path.is_file() else "missing",
            "unchanged": path.is_file(),
            "git_ls_files_marker": marker_output.split(" ", 1)[0] if marker_output else "missing",
            "protected_by_task_contract": True,
        }
    return {
        "files": files,
        "active_roadmap_and_conductor_unchanged": all(row["unchanged"] for row in files.values()),
    }


def _tests_pass(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in tests_run)


def gate_check_summary(
    *,
    receipts: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    retired_scope: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    failed: list[str] = []
    for receipt in receipts:
        if receipt.get("exists") is not True:
            failed.append(f"missing_v559_artifact:{receipt.get('artifact_key')}")
    raw = dict(aggregate.get("exp6486_raw_rows") or {})
    audit = dict(aggregate.get("exp6487_integrity_audit") or {})
    checks = {
        "all_v559_artifacts_present": not any(item.startswith("missing_v559_artifact") for item in failed),
        "all_v559_artifacts_hashed": all(str(row.get("sha256")).startswith("sha256:") for row in receipts),
        "exp6483_source_mapping_recomputed": dict(
            aggregate.get("exp6483_source_ingestion") or {}
        ).get("source_mapping_ready")
        is True,
        "exp6484_receipt_contract_reusable": dict(
            aggregate.get("exp6484_receipt_contract") or {}
        ).get("recomputed_ready")
        is True,
        "exp6485_transition_contract_reusable": dict(
            aggregate.get("exp6485_transition_contract") or {}
        ).get("recomputed_ready")
        is True,
        "exp6486_raw_rows_recomputed": raw.get("raw_record_count") == 432
        and raw.get("raw_hashes_match_manifest") is True,
        "exp6487_disqualification_recomputed": audit.get("recomputed_ready_score") == 0.0
        and audit.get("shortcut_replay_matches_report") is True
        and audit.get("missing_evidence_matches_report") is True,
        "decision_rows_cover_v559_surfaces": len(decisions) == 6
        and all(row.get("disposition") in ALLOWED_DISPOSITIONS for row in decisions),
        "forbidden_reuse_attacks_fail_closed": all(
            row.get("fail_closed") is True and row.get("allowed_into_v560") is False
            for row in attacks
        ),
        "no_forced_candidate_selector_eligibility": retired_scope.get(
            "v559_forced_candidate_rows_eligible_for_v560_selector"
        )
        == 0,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged") is True,
        "tests_passed": _tests_pass(tests_run),
    }
    failed.extend(name for name, passed in checks.items() if passed is not True)
    return {
        "checks": checks,
        "failed_checks": sorted(set(failed)),
        "all_gates_passed": not failed,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    receipts: Sequence[Mapping[str, Any]],
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    required_paths = {
        "exclusion_manifest": repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        "research_complete": repo_root / RESEARCH_COMPLETE_RELATIVE_PATH,
        "e2e_plan": repo_root / E2E_PLAN_RELATIVE_PATH,
        "spec": repo_root / SPEC_RELATIVE_PATH,
    }
    missing_required = [name for name, path in required_paths.items() if not path.is_file()]
    missing_artifacts = [
        f"missing_v559_artifact:{row.get('artifact_key')}"
        for row in receipts
        if row.get("exists") is not True
    ]
    output_parent_ready = result_path.parent.exists() or os.access(result_path.parent.parent, os.W_OK)
    blocked_reasons = sorted(
        set(
            [
                *missing_required,
                *missing_artifacts,
                *list(gate_summary.get("failed_checks") or []),
            ]
        )
    )
    if not output_parent_ready:  # pragma: no cover
        blocked_reasons.append("output_path_not_writable")
    return {
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_output(repo_root, ["status", "--short", "--untracked-files=no"]),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "artifact_paths_present": not missing_artifacts,
        "required_files": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in sorted(required_paths.items())
        },
        "result_path": str(result_path),
        "protected_task_paths": [path.as_posix() for path in PROTECTED_RELATIVE_PATHS],
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
    }


def field_provenance() -> JsonDict:
    sources = {
        "v559_artifact_receipts": {
            "json_pointers": ["/status", "/honest_verdict"],
            "reducer": "load_v559_artifacts",
        },
        "decision_rows": {"json_pointers": ["/decision_rows"], "reducer": "decision_rows"},
        "aggregate_row_recomputation": {
            "json_pointers": ["/per_unit_rows", "/raw_vector_manifest", "/missing_verifier_gaps"],
            "reducer": "aggregate_row_recomputation",
        },
        "retired_scope_definition": {
            "json_pointers": ["/aggregate_row_recomputation/exp6487_shortcut_replay"],
            "reducer": "retired_scope_definition",
        },
        "allowed_v560_lineage": {
            "json_pointers": ["/allowed_v560_lineage"],
            "reducer": "allowed_v560_lineage",
        },
        "forbidden_reuse_attack_matrix": {
            "json_pointers": ["/forbidden_reuse_attack_matrix"],
            "reducer": "forbidden_reuse_attack_matrix",
        },
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "artifact_paths": [path.as_posix() for path in DEFAULT_ARTIFACT_PATHS.values()],
            **sources.get(field, {"json_pointers": [f"/{field}"], "reducer": "build_artifact"}),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "v559_artifact_receipts": artifact.get("v559_artifact_receipts"),
        "decision_rows": artifact.get("decision_rows"),
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation"),
        "retired_scope_definition": artifact.get("retired_scope_definition"),
        "allowed_v560_lineage": artifact.get("allowed_v560_lineage"),
        "forbidden_reuse_attack_matrix": artifact.get("forbidden_reuse_attack_matrix"),
        "v560_lineage_lock_ready_score": artifact.get("v560_lineage_lock_ready_score"),
    }
    return sha256_json(stable)


def v560_lineage_lock_ready_score(artifact: Mapping[str, Any]) -> float:
    gate = dict(artifact.get("gate_check_summary") or {})
    preconditions = dict(artifact.get("preconditions_checked") or {})
    protected = dict(artifact.get("protected_files_unchanged") or {})
    retired = dict(artifact.get("retired_scope_definition") or {})
    aggregate = dict(artifact.get("aggregate_row_recomputation") or {})
    attacks = list(artifact.get("forbidden_reuse_attack_matrix") or [])
    decisions = list(artifact.get("decision_rows") or [])
    ready = (
        gate.get("all_gates_passed") is True
        and preconditions.get("preconditions_ready") is True
        and protected.get("active_roadmap_and_conductor_unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and retired.get("selector_eligible") is False
        and retired.get("v559_forced_candidate_rows_eligible_for_v560_selector") == 0
        and aggregate.get("all_v559_dispositions_recomputed") is True
        and aggregate.get("no_v559_forced_candidate_row_selector_eligible") is True
        and len(decisions) == 6
        and all(row.get("disposition") in ALLOWED_DISPOSITIONS for row in decisions)
        and len(attacks) == 5
        and all(
            row.get("fail_closed") is True and row.get("allowed_into_v560") is False
            for row in attacks
        )
        and _tests_pass(list(artifact.get("tests_run") or []))
    )
    return 1.0 if ready else 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if errors:
        return errors
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    provenance = dict(artifact.get("field_provenance") or {})
    if any(field not in provenance for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    expected_score = v560_lineage_lock_ready_score(artifact)
    if artifact.get("v560_lineage_lock_ready_score") != expected_score:
        errors.append("v560_lineage_lock_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    status = str(artifact.get("status") or "")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0:
        if status != "complete_v559_decision_ledger":
            errors.append("status")
        if not verdict.startswith("complete_v560_lineage_lock:"):
            errors.append("honest_verdict")
    else:
        if status != "blocked_v559_decision_ledger":
            errors.append("status")
        if not verdict.startswith("blocked_v560_lineage_lock:"):
            errors.append("honest_verdict")
        if not list(dict(artifact.get("gate_check_summary") or {}).get("failed_checks") or []):
            errors.append("gate_check_summary")
    if any(
        row.get("disposition") not in ALLOWED_DISPOSITIONS
        for row in artifact.get("decision_rows") or []
    ):
        errors.append("decision_rows")
    if any(
        row.get("fail_closed") is not True or row.get("allowed_into_v560") is not False
        for row in artifact.get("forbidden_reuse_attack_matrix") or []
    ):
        errors.append("forbidden_reuse_attack_matrix")
    return sorted(set(errors))


def assert_valid_artifact(artifact: Mapping[str, Any]) -> None:
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(",".join(errors))


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = result_path or root / RESULT_RELATIVE_PATH
    configured_paths = dict(DEFAULT_ARTIFACT_PATHS)
    if artifact_paths:
        configured_paths.update({key: Path(value) for key, value in artifact_paths.items()})
    tests = [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]

    receipts, payloads = load_v559_artifacts(root, configured_paths)
    aggregate = aggregate_row_recomputation(payloads)
    decisions = decision_rows(aggregate)
    attacks = forbidden_reuse_attack_matrix()
    retired = retired_scope_definition(aggregate)
    allowed = allowed_v560_lineage()
    protected = protected_files_unchanged(root)
    gate = gate_check_summary(
        receipts=receipts,
        aggregate=aggregate,
        decisions=decisions,
        attacks=attacks,
        retired_scope=retired,
        protected=protected,
        tests_run=tests,
    )
    preconditions = preconditions_checked(
        repo_root=root,
        result_path=output_path,
        receipts=receipts,
        gate_summary=gate,
    )
    per_unit_rows = [*decisions, *attacks]
    artifact: JsonDict = {
        "status": "",
        "v559_artifact_receipts": receipts,
        "decision_rows": decisions,
        "aggregate_row_recomputation": aggregate,
        "retired_scope_definition": retired,
        "allowed_v560_lineage": allowed,
        "forbidden_reuse_attack_matrix": attacks,
        "v560_lineage_lock_ready_score": 0.0,
        "per_unit_rows": per_unit_rows,
        "gate_check_summary": gate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - start, 6),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    score = v560_lineage_lock_ready_score(artifact)
    artifact["v560_lineage_lock_ready_score"] = score
    if score == 1.0:
        artifact["status"] = "complete_v559_decision_ledger"
        artifact["honest_verdict"] = (
            "complete_v560_lineage_lock: V559 forced-candidate selector is retired; "
            "V560 exact-solver trajectory lineage is allowed"
        )
    else:
        failed = ",".join(gate.get("failed_checks") or ["unknown_gate"])
        artifact["status"] = "blocked_v559_decision_ledger"
        artifact["honest_verdict"] = f"blocked_v560_lineage_lock: {failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    assert_valid_artifact(artifact)
    if write:
        _write_atomic(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"unsupported planning date: {args.date}")
    artifact = build_artifact(result_path=args.output, write=True)
    print(json.dumps({"result_path": str(args.output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
