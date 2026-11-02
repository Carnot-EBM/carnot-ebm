"""Assemble Exp 3272 prompt-injection v4 full corpus and split audit.

Spec refs: REQ-REPORT-3272, SCENARIO-REPORT-3272.

This module is deliberately data-only: it does not invoke a teacher model, train
a KAN, or run Garak. Its job is to make the downstream evaluation substrate
boring and auditable by freezing exactly which examples are trainable, which are
evaluation-only, and which signatures were checked for split leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_v4_full_corpus_assembly_leakage_audit.v1"
EXPERIMENT_ID = "exp3272"
TASK_ID = "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1"
ARTIFACT = "experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1"
MILESTONE = "2026.05.303"
RUN_DATE = "20260528"
RANDOM_SEED = 3272

OUTPUT_REL_PATH = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.py"
)
EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")
EXP3269_REL_PATH = Path(
    "results/experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.json"
)
EXP3270_REL_PATH = Path("results/experiment_3270_prompt_injection_teacher_label_shards_2_4_v1.json")
EXP3271_REL_PATH = Path(
    "results/experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1.json"
)
SHARD_INPUT_DIR = Path("data/prompt_injection_v4/teacher_label_shards")
GARAK_SEED_REL_PATH = Path("data/prompt_injection_v4/splits/garak_adaptive_seed_v1.jsonl")
GARAK_SEED_SHARD_ID = "v4-garak-adaptive-seed"
CORPUS_REL_PATH = Path(
    "data/prompt_injection_v4/full_corpus/prompt_injection_v4_full_15k_corpus_v1.jsonl"
)
SPLIT_OUTPUT_DIR = Path("data/prompt_injection_v4/frozen_splits")

NORMAL_SHARD_NUMBERS = (2, 3, 4, 5, 6, 7)
SPLIT_ORDER = ("train", "eval", "holdout", "garak")
NORMAL_SPLITS = ("train", "eval", "holdout")
DEFAULT_SPLIT_TARGETS = {"train": 10000, "eval": 2000, "holdout": 2000, "garak": 1000}
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
ALLOWED_LABELS = ("benign", "injection")

REQUIRED_ARTIFACT_FIELDS = {
    "full_15k_corpus_ready",
    "assembled_example_count",
    "train_count",
    "eval_count",
    "holdout_count",
    "garak_count",
    "leakage_audit_passed",
    "duplicate_count_removed",
    "split_distribution",
    "output_paths",
    "checksums",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def shard_input_rel_path(shard_number: int) -> Path:
    """Return the teacher-label JSONL path for one normal v4 shard."""

    return SHARD_INPUT_DIR / f"v4_shard_{int(shard_number):03d}_teacher_labels_v1.jsonl"


def split_output_rel_path(split: str) -> Path:
    """Return the frozen JSONL path for one downstream split."""

    return SPLIT_OUTPUT_DIR / f"prompt_injection_v4_{split}_v1.jsonl"


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path = OUTPUT_REL_PATH,
    split_targets: Mapping[str, int] | None = None,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-REPORT-3272: assemble, freeze, audit, and persist the v4 corpus."""

    start = monotonic()
    root = Path(project_root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root / out_path
    targets = normalize_split_targets(split_targets)
    target_total = sum(targets.values())

    source_payloads = read_source_payloads(root)
    preconditions = precondition_checks(root=root, source_payloads=source_payloads, targets=targets)
    blocked_reason = first_blocked_reason(preconditions)

    rows: list[JsonDict] = []
    removed_duplicates: list[JsonDict] = []
    output_paths = [
        Path(output_path).as_posix() if not Path(output_path).is_absolute() else str(out_path)
    ]
    output_checksums: dict[str, str] = {}
    split_error = ""
    leakage_audit = empty_leakage_audit()

    if not blocked_reason:
        raw_rows = load_all_rows(root=root, source_payloads=source_payloads)
        deduped_rows, removed_duplicates = remove_cross_source_duplicates(raw_rows)
        rows, split_error = freeze_splits(
            deduped_rows,
            targets=targets,
            random_seed=int(random_seed),
        )
        if split_error:
            blocked_reason = split_error
            rows = []
        else:
            rows = assign_canonical_ids(rows)
            leakage_audit = audit_split_leakage(rows)
            output_paths.extend(write_corpus_and_split_files(root, rows))
            output_checksums = output_file_checksums(root, output_paths[1:])

    split_distribution = compute_split_distribution(rows)
    counts = {split: split_distribution.get(split, {}).get("total", 0) for split in SPLIT_ORDER}
    assembled_count = len(rows)
    leakage_passed = bool(leakage_audit.get("leakage_audit_passed"))
    ready = (
        not blocked_reason
        and assembled_count == target_total
        and counts["train"] == targets["train"]
        and counts["eval"] == targets["eval"]
        and counts["holdout"] == targets["holdout"]
        and counts["garak"] == targets["garak"]
        and leakage_passed
    )

    checksums = {
        "output_files": output_checksums,
        "source_artifacts": source_checksums(root),
    }
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "full_15k_corpus_ready": ready,
        "blocked_reason": blocked_reason,
        "target_total_examples": target_total,
        "split_targets": dict(targets),
        "assembled_example_count": assembled_count,
        "raw_example_count": raw_input_count(source_payloads) if not blocked_reason else 0,
        "train_count": counts["train"],
        "eval_count": counts["eval"],
        "holdout_count": counts["holdout"],
        "garak_count": counts["garak"],
        "leakage_audit_passed": leakage_passed,
        "duplicate_count_removed": len(removed_duplicates),
        "duplicate_rows_removed_sample": duplicate_sample(removed_duplicates),
        "within_source_duplicate_count": within_source_duplicate_count(rows),
        "split_distribution": split_distribution,
        "leakage_audit": leakage_audit,
        "preconditions_checked": preconditions,
        "output_paths": output_paths,
        "checksums": checksums,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration(start, monotonic()),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def normalize_split_targets(split_targets: Mapping[str, int] | None) -> dict[str, int]:
    """Normalize split targets so tests and production use the same code path."""

    raw = dict(DEFAULT_SPLIT_TARGETS if split_targets is None else split_targets)
    return {split: max(0, safe_int(raw.get(split))) for split in SPLIT_ORDER}


def read_source_payloads(root: Path) -> dict[str, JsonDict]:
    """Read small upstream JSON artifacts once for gate checks and row loading."""

    return {
        "exp3264": read_json_object(root / EXP3264_REL_PATH),
        "exp3269": read_json_object(root / EXP3269_REL_PATH),
        "exp3270": read_json_object(root / EXP3270_REL_PATH),
        "exp3271": read_json_object(root / EXP3271_REL_PATH),
    }


def precondition_checks(
    *,
    root: Path,
    source_payloads: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, int],
) -> list[JsonDict]:
    """Check the label and seed gates before writing any downstream corpus files."""

    exp3264 = source_payloads.get("exp3264") or {}
    exp3269 = source_payloads.get("exp3269") or {}
    exp3270 = source_payloads.get("exp3270") or {}
    exp3271 = source_payloads.get("exp3271") or {}
    normal_target = int(targets["train"]) + int(targets["eval"]) + int(targets["holdout"])
    exp3270_required = min(8000, normal_target)
    exp3271_required = min(14000, normal_target)
    missing_inputs = missing_input_paths(root)
    return [
        {
            "name": "exp3269_full_corpus_manifest",
            "passed": exp3269.get("full_corpus_manifest_ready") is True,
            "path": EXP3269_REL_PATH.as_posix(),
            "target_total_examples": safe_int(exp3269.get("target_total_examples")),
        },
        {
            "name": "exp3264_seed_shard",
            "passed": exp3264.get("teacher_label_shard_ready") is True
            and exp3264.get("teacher_label_shard_v3_ready") is not False
            and len(list(exp3264.get("per_example_labels") or [])) > 0,
            "path": EXP3264_REL_PATH.as_posix(),
            "row_count": len(list(exp3264.get("per_example_labels") or [])),
        },
        {
            "name": "exp3270_teacher_label_shards_2_4",
            "passed": exp3270.get("teacher_label_shards_2_4_ready") is True
            and safe_int(exp3270.get("cumulative_label_count")) >= exp3270_required,
            "path": EXP3270_REL_PATH.as_posix(),
            "cumulative_label_count": safe_int(exp3270.get("cumulative_label_count")),
            "required_min_count": exp3270_required,
        },
        {
            "name": "exp3271_garak_seed",
            "passed": exp3271.get("teacher_label_shards_5_7_garak_seed_ready") is True
            and safe_int(exp3271.get("garak_seed_count")) >= int(targets["garak"]),
            "path": EXP3271_REL_PATH.as_posix(),
            "garak_seed_count": safe_int(exp3271.get("garak_seed_count")),
            "required_min_count": int(targets["garak"]),
        },
        {
            "name": "exp3271_teacher_label_shards_5_7",
            "passed": exp3271.get("teacher_label_shards_5_7_garak_seed_ready") is True
            and safe_int(exp3271.get("cumulative_label_count")) >= exp3271_required,
            "path": EXP3271_REL_PATH.as_posix(),
            "cumulative_label_count": safe_int(exp3271.get("cumulative_label_count")),
            "required_min_count": exp3271_required,
        },
        {
            "name": "source_jsonl_files_present",
            "passed": not missing_inputs,
            "missing_paths": missing_inputs,
        },
    ]


def missing_input_paths(root: Path) -> list[str]:
    """List JSONL inputs that must exist before assembly can proceed."""

    rel_paths = [*(shard_input_rel_path(number) for number in NORMAL_SHARD_NUMBERS), GARAK_SEED_REL_PATH]
    return [rel_path.as_posix() for rel_path in rel_paths if not (root / rel_path).is_file()]


def first_blocked_reason(preconditions: Sequence[Mapping[str, Any]]) -> str:
    """Map the first failed gate to the operator-facing blocker."""

    mapping = {
        "exp3269_full_corpus_manifest": "gated_exp3269_full_corpus_manifest_not_ready",
        "exp3264_seed_shard": "gated_exp3264_seed_shard_not_ready",
        "exp3270_teacher_label_shards_2_4": "gated_exp3270_teacher_label_shards_2_4_not_ready",
        "exp3271_garak_seed": "gated_exp3271_garak_seed_not_ready",
        "exp3271_teacher_label_shards_5_7": "gated_exp3271_teacher_label_shards_5_7_not_ready",
        "source_jsonl_files_present": "missing_source_jsonl_files",
    }
    for row in preconditions:
        if row.get("passed") is not True:
            return mapping.get(str(row.get("name") or ""), "precondition_failed")
    return ""


def load_all_rows(*, root: Path, source_payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Load and normalize the seed shard, normal shards, and Garak seed rows."""

    rows: list[JsonDict] = []
    for index, row in enumerate(source_payloads["exp3264"].get("per_example_labels") or []):
        rows.append(normalize_seed_row(row, row_index=index))
    for shard_number in NORMAL_SHARD_NUMBERS:
        rel_path = shard_input_rel_path(shard_number)
        for index, row in enumerate(read_jsonl(root / rel_path)):
            rows.append(normalize_jsonl_row(row, source_path=rel_path, row_index=index))
    for index, row in enumerate(read_jsonl(root / GARAK_SEED_REL_PATH)):
        rows.append(normalize_jsonl_row(row, source_path=GARAK_SEED_REL_PATH, row_index=index))
    for source_order, row in enumerate(rows):
        row["source_order"] = source_order
    return rows


def normalize_seed_row(row: Mapping[str, Any], *, row_index: int) -> JsonDict:
    """Normalize the `.302` seed shard into the same schema as JSONL shards."""

    text = str(row.get("text") or "")
    label = normalize_label(row.get("teacher_label") or row.get("source_label"))
    source_example_id = str(row.get("example_id") or f"v4-shard-001-{int(row_index):06d}")
    return enrich_normalized_row(
        {
            "source_example_id": source_example_id,
            "source_shard_id": "v4-shard-001",
            "source_path": str(row.get("source_path") or EXP3264_REL_PATH.as_posix()),
            "source_row_index": int(row_index),
            "source": str(row.get("source") or "exp3264_seed_shard"),
            "source_label": normalize_label(row.get("source_label") or label),
            "teacher_label": label,
            "teacher_label_source": str(row.get("teacher_label_source") or "exp3264_teacher_label"),
            "category_id": str(row.get("category_id") or seed_category_id(label)),
            "instruction_alignment": str(
                row.get("instruction_alignment") or default_alignment(label)
            ),
            "source_split": "train_eval_holdout_candidate",
            "text": text,
            "raw_output": str(row.get("raw_output") or label),
            "parse_status": str(row.get("parse_status") or "parsed"),
            "provenance": dict(row.get("provenance") or {}),
        }
    )


def normalize_jsonl_row(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    row_index: int,
) -> JsonDict:
    """Normalize one JSONL shard row while preserving its source identity."""

    text = str(row.get("text") or "")
    label = normalize_label(row.get("teacher_label") or row.get("source_label"))
    source_shard_id = str(row.get("shard_id") or source_path.stem)
    is_garak = source_shard_id == GARAK_SEED_SHARD_ID
    return enrich_normalized_row(
        {
            "source_example_id": str(row.get("example_id") or f"{source_shard_id}-{row_index:06d}"),
            "source_shard_id": source_shard_id,
            "source_path": source_path.as_posix(),
            "source_row_index": int(row.get("row_index") if row.get("row_index") is not None else row_index),
            "source": str(row.get("source") or "jsonl_shard"),
            "source_label": normalize_label(row.get("source_label") or label),
            "teacher_label": label,
            "teacher_label_source": str(row.get("teacher_label_source") or "jsonl_teacher_label"),
            "category_id": str(row.get("category_id") or "unknown"),
            "instruction_alignment": str(
                row.get("instruction_alignment") or default_alignment(label)
            ),
            "source_split": "garak_adaptive_seed"
            if is_garak
            else str(row.get("split") or "train_eval_holdout_candidate"),
            "text": text,
            "raw_output": str(row.get("raw_output") or label),
            "parse_status": str(row.get("parse_status") or "parsed"),
            "provenance": dict(row.get("provenance") or {}),
        }
    )


def enrich_normalized_row(row: JsonDict) -> JsonDict:
    """Attach stable text signatures used for de-duplication and leakage audit."""

    normalized = normalize_text(str(row["text"]))
    near = near_duplicate_signature(str(row["text"]))
    template = template_family_signature(str(row["text"]))
    row.update(
        {
            "text_sha256": sha256_text(str(row["text"])),
            "normalized_text": normalized,
            "normalized_text_sha256": sha256_text(normalized),
            "near_duplicate_signature": near,
            "near_duplicate_sha256": sha256_text(near),
            "template_family_signature": template,
            "template_family_sha256": sha256_text(template),
        }
    )
    return row


def remove_cross_source_duplicates(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Remove exact/near duplicates only when they cross source shard boundaries.

    The seed shard contains repeated prompts that are part of its recorded source
    distribution. Dropping those would silently shrink the 15k gate, so repeated
    rows from the same source shard are kept and later forced into one split.
    """

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("near_duplicate_sha256") or "")].append(row)

    remove_ids: set[tuple[str, int]] = set()
    for group_rows in groups.values():
        source_order = sorted(
            {str(row.get("source_shard_id") or "") for row in group_rows},
            key=lambda source: (source != GARAK_SEED_SHARD_ID, source),
        )
        if len(source_order) <= 1:
            continue
        keep_source = source_order[0]
        for row in group_rows:
            if str(row.get("source_shard_id") or "") != keep_source:
                remove_ids.add((str(row.get("source_example_id") or ""), safe_int(row.get("source_order"))))

    kept: list[JsonDict] = []
    removed: list[JsonDict] = []
    for row in rows:
        key = (str(row.get("source_example_id") or ""), safe_int(row.get("source_order")))
        target = removed if key in remove_ids else kept
        target.append(dict(row))
    return kept, removed


def freeze_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, int],
    random_seed: int,
) -> tuple[list[JsonDict], str]:
    """Assign rows to frozen splits without splitting normal template families."""

    normal_rows = [dict(row) for row in rows if row.get("source_shard_id") != GARAK_SEED_SHARD_ID]
    garak_rows = [dict(row) for row in rows if row.get("source_shard_id") == GARAK_SEED_SHARD_ID]
    normal_target = int(targets["train"]) + int(targets["eval"]) + int(targets["holdout"])
    if len(normal_rows) != normal_target:
        return [], f"assembled_normal_count_{len(normal_rows)}_does_not_match_target_{normal_target}"
    if len(garak_rows) != int(targets["garak"]):
        return [], f"assembled_garak_count_{len(garak_rows)}_does_not_match_target_{targets['garak']}"

    groups = group_normal_rows(normal_rows)
    assignments: dict[str, str] = {}
    counts = {split: 0 for split in NORMAL_SPLITS}
    for signature, group_rows in sorted_groups(groups, random_seed=random_seed):
        size = len(group_rows)
        candidates = [split for split in NORMAL_SPLITS if counts[split] + size <= int(targets[split])]
        if not candidates:
            return [], f"template_family_split_capacity_exhausted_size_{size}"
        split = min(
            candidates,
            key=lambda candidate: (
                counts[candidate] / max(1, int(targets[candidate])),
                int(targets[candidate]),
                stable_hash(f"{random_seed}:{signature}:{candidate}"),
            ),
        )
        assignments[signature] = split
        counts[split] += size

    if counts != {split: int(targets[split]) for split in NORMAL_SPLITS}:  # pragma: no cover
        return [], f"split_counts_do_not_match_targets_{counts}"

    frozen: list[JsonDict] = []
    for row in normal_rows:
        split = assignments[str(row["template_family_sha256"])]
        row["split"] = split
        row["training_eligible"] = split == "train"
        frozen.append(row)
    for row in garak_rows:
        row["split"] = "garak"
        row["training_eligible"] = False
        frozen.append(row)
    frozen.sort(key=lambda row: (SPLIT_ORDER.index(str(row["split"])), safe_int(row["source_order"])))
    return frozen, ""


def group_normal_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    """Group normal-corpus rows by template signature for leakage-safe splitting."""

    groups: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        groups[str(row["template_family_sha256"])].append(dict(row))
    return groups


def sorted_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    random_seed: int,
) -> list[tuple[str, list[JsonDict]]]:
    """Sort larger template families first with a reproducible hash tie-breaker."""

    return sorted(
        ((signature, [dict(row) for row in rows]) for signature, rows in groups.items()),
        key=lambda item: (-len(item[1]), stable_hash(f"{random_seed}:{item[0]}")),
    )


def assign_canonical_ids(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attach stable canonical IDs after final split assignment."""

    split_indices = {split: 0 for split in SPLIT_ORDER}
    canonical_rows: list[JsonDict] = []
    for global_index, row in enumerate(rows):
        split = str(row["split"])
        split_index = split_indices[split]
        split_indices[split] += 1
        canonical = dict(row)
        canonical["canonical_index"] = global_index
        canonical["split_index"] = split_index
        canonical["canonical_id"] = f"pi-v4-{split}-{split_index:06d}"
        canonical_rows.append(canonical)
    return canonical_rows


def audit_split_leakage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Audit exact, near, and normal template-family leakage across frozen splits."""

    exact = signature_overlap(rows, "normalized_text_sha256", splits=SPLIT_ORDER)
    near = signature_overlap(rows, "near_duplicate_sha256", splits=SPLIT_ORDER)
    template = signature_overlap(rows, "template_family_sha256", splits=NORMAL_SPLITS)
    garak_rows = [row for row in rows if row.get("split") == "garak"]
    normal_template_signatures = {
        str(row.get("template_family_sha256") or "")
        for row in rows
        if row.get("split") in NORMAL_SPLITS
    }
    garak_template_overlap_count = sum(
        1
        for row in garak_rows
        if str(row.get("template_family_sha256") or "") in normal_template_signatures
    )
    garak_training_eligible_false = all(row.get("training_eligible") is False for row in garak_rows)
    passed = (
        exact["overlap_group_count"] == 0
        and near["overlap_group_count"] == 0
        and template["overlap_group_count"] == 0
        and garak_training_eligible_false
    )
    return {
        "leakage_audit_passed": passed,
        "exact_duplicate_overlap": exact,
        "near_duplicate_overlap": near,
        "normal_template_family_overlap": template,
        "garak_template_family_overlap_count": garak_template_overlap_count,
        "garak_training_eligible_false": garak_training_eligible_false,
    }


def signature_overlap(
    rows: Sequence[Mapping[str, Any]],
    signature_key: str,
    *,
    splits: Sequence[str],
) -> JsonDict:
    """Return split-spanning signature groups for one audit key."""

    split_set = set(splits)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") in split_set:
            groups[str(row.get(signature_key) or "")].append(row)
    overlap_groups = [
        group_rows
        for group_rows in groups.values()
        if len({str(row.get("split") or "") for row in group_rows}) > 1
    ]
    return {
        "signature_key": signature_key,
        "overlap_group_count": len(overlap_groups),
        "overlap_row_count": sum(len(group) for group in overlap_groups),
        "sample": [
            {
                "signature": str(group[0].get(signature_key) or ""),
                "splits": sorted({str(row.get("split") or "") for row in group}),
                "example_ids": [str(row.get("canonical_id") or row.get("source_example_id")) for row in group[:5]],
            }
            for group in overlap_groups[:5]
        ],
    }


def empty_leakage_audit() -> JsonDict:
    """Return the audit shape for gated-skip artifacts."""

    empty = {"signature_key": "", "overlap_group_count": 0, "overlap_row_count": 0, "sample": []}
    return {
        "leakage_audit_passed": False,
        "exact_duplicate_overlap": dict(empty),
        "near_duplicate_overlap": dict(empty),
        "normal_template_family_overlap": dict(empty),
        "garak_template_family_overlap_count": 0,
        "garak_training_eligible_false": False,
    }


def compute_split_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Compute class, instruction-form, and taxonomy balance for every split."""

    return {
        split: distribution_counts([row for row in rows if row.get("split") == split])
        for split in SPLIT_ORDER
    }


def distribution_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the shared balance counters for a row collection."""

    labels = Counter(str(row.get("teacher_label") or "") for row in rows)
    alignments = Counter(str(row.get("instruction_alignment") or "") for row in rows)
    categories = Counter(str(row.get("category_id") or "") for row in rows)
    training_eligible = sum(1 for row in rows if row.get("training_eligible") is True)
    return {
        "total": len(rows),
        "benign": int(labels.get("benign", 0)),
        "injection": int(labels.get("injection", 0)),
        "aligned_instruction": int(alignments.get("aligned_instruction", 0)),
        "misaligned_instruction": int(alignments.get("misaligned_instruction", 0)),
        "non_instruction": int(alignments.get("non_instruction", 0)),
        "training_eligible": int(training_eligible),
        "by_category": dict(sorted(categories.items())),
    }


def write_corpus_and_split_files(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Write canonical full corpus and each frozen split JSONL file."""

    output_paths = [CORPUS_REL_PATH.as_posix()]
    write_jsonl(root / CORPUS_REL_PATH, rows)
    for split in SPLIT_ORDER:
        rel_path = split_output_rel_path(split)
        write_jsonl(rel_path=root / rel_path, rows=[row for row in rows if row.get("split") == split])
        output_paths.append(rel_path.as_posix())
    return output_paths


def write_jsonl(rel_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL deterministically with sorted object keys."""

    rel_path.parent.mkdir(parents=True, exist_ok=True)
    rel_path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def output_file_checksums(root: Path, output_paths: Sequence[str]) -> dict[str, str]:
    """Checksum written corpus and split files."""

    return {rel_path: sha256_file(root / rel_path) for rel_path in output_paths}


def source_checksums(root: Path) -> dict[str, str]:
    """Checksum every upstream artifact and JSONL source consumed by assembly."""

    rel_paths = [
        EXP3264_REL_PATH,
        EXP3269_REL_PATH,
        EXP3270_REL_PATH,
        EXP3271_REL_PATH,
        *(shard_input_rel_path(number) for number in NORMAL_SHARD_NUMBERS),
        GARAK_SEED_REL_PATH,
    ]
    return {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in rel_paths
        if (root / rel_path).is_file()
    }


def raw_input_count(source_payloads: Mapping[str, Mapping[str, Any]]) -> int:
    """Return metadata-only raw count evidence for the result artifact."""

    exp3264 = source_payloads.get("exp3264") or {}
    exp3270 = source_payloads.get("exp3270") or {}
    exp3271 = source_payloads.get("exp3271") or {}
    seed_count = len(list(exp3264.get("per_example_labels") or []))
    return (
        seed_count
        + safe_int(exp3270.get("new_label_count"))
        + safe_int(exp3271.get("new_label_count"))
        + safe_int(exp3271.get("garak_seed_count"))
    )


def within_source_duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count repeated near signatures that remain within the same source shard."""

    groups: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        groups[(str(row.get("source_shard_id") or ""), str(row.get("near_duplicate_sha256") or ""))] += 1
    return sum(count - 1 for count in groups.values() if count > 1)


def duplicate_sample(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return a small duplicate-removal sample for operator audit."""

    return [
        {
            "source_example_id": str(row.get("source_example_id") or ""),
            "source_shard_id": str(row.get("source_shard_id") or ""),
            "near_duplicate_sha256": str(row.get("near_duplicate_sha256") or ""),
        }
        for row in rows[:10]
    ]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict string required by REQ-REPORT-3272."""

    if artifact.get("full_15k_corpus_ready") is True:
        return (
            "complete: full_15k_corpus_ready=true; "
            f"assembled_example_count={artifact.get('assembled_example_count')}; "
            f"train_count={artifact.get('train_count')}; "
            f"eval_count={artifact.get('eval_count')}; "
            f"holdout_count={artifact.get('holdout_count')}; "
            f"garak_count={artifact.get('garak_count')}"
        )
    return (
        "complete: full_15k_corpus_ready=false; "
        f"blocked_reason={artifact.get('blocked_reason')}; "
        f"leakage_audit_passed={artifact.get('leakage_audit_passed')}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable result evidence while excluding timing and the hash itself."""

    stable_keys = [
        "experiment_id",
        "task_id",
        "full_15k_corpus_ready",
        "blocked_reason",
        "target_total_examples",
        "split_targets",
        "assembled_example_count",
        "train_count",
        "eval_count",
        "holdout_count",
        "garak_count",
        "leakage_audit_passed",
        "duplicate_count_removed",
        "within_source_duplicate_count",
        "split_distribution",
        "leakage_audit",
        "output_paths",
        "checksums",
        "random_seed",
    ]
    payload = {key: artifact.get(key) for key in stable_keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that omit required fields or use a non-terminal verdict."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3272")
    if not terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for absent or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSONL rows from a trusted local artifact path."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows.append(dict(payload) if isinstance(payload, Mapping) else {})
    return rows


def normalize_label(value: Any) -> str:
    """Coerce unknown labels into benign so malformed rows fail conservatively."""

    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else "benign"


def seed_category_id(label: str) -> str:
    """Assign a coarse taxonomy bucket when the seed shard lacks category IDs."""

    return "seed_injection" if label == "injection" else "seed_benign"


def default_alignment(label: str) -> str:
    """Infer instruction alignment from the binary label when a source omits it."""

    return "misaligned_instruction" if label == "injection" else "aligned_instruction"


def normalize_text(text: str) -> str:
    """Lowercase and whitespace-normalize text for exact duplicate checks."""

    return " ".join(text.lower().split())


def near_duplicate_signature(text: str) -> str:
    """Build a conservative near-duplicate key that ignores punctuation only."""

    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def template_family_signature(text: str) -> str:
    """Build the template-family key used to keep prompt variants in one split."""

    value = normalize_text(text)
    value = re.sub(r"\[(case|variant) [^\]]+\]", "[id]", value)
    value = re.sub(r"\b\d+\b", "{n}", value)
    return value


def stable_hash(value: str) -> str:
    """Return a SHA-256 hex digest for deterministic tie-breaking."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the full SHA-256 digest for a local artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_int(value: Any) -> int:
    """Convert JSON-ish counts without raising on malformed upstream values."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def duration(start: float, now: float) -> float:
    """Return a non-negative rounded duration."""

    return round(max(0.0, float(now) - float(start)), 6)


def terminal_prefix_ok(value: str) -> bool:
    """Check the terminal honest-verdict prefix contract."""

    return any(value.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def main() -> int:  # pragma: no cover
    """Run the reporting task from the command line."""

    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
