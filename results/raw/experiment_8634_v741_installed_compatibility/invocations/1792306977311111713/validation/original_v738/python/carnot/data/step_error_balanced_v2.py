"""Build the Exp 3858 balanced step-error corpus.

Spec: REQ-DATA-3858, SCENARIO-DATA-3858, SCENARIO-DATA-3858-FALLBACK.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_PRMBENCH_DATASET_ID = "hitsmy/PRMBench_Preview"
DEFAULT_PRMBENCH_FILENAME = "prmbench_preview.jsonl"
DEFAULT_PRMBENCH_REACHABILITY_URL = (
    "https://huggingface.co/api/datasets/hitsmy/PRMBench_Preview"
)
DEFAULT_RANDOM_SEED = 3858
DEFAULT_TARGET_N = 1000
DEFAULT_MIN_INCORRECT_STEPS = 100
FOVER_FALLBACK_FILES = (
    "fover_corpus_v3.json",
    "fover_test_v3.json",
    "fover_train_v3.json",
    "fover_corpus_expanded.json",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "n_items": (
        "Total corpus size; N>=1000 so downstream residual-catch CIs are not "
        "small-sample noise."
    ),
    "n_incorrect_steps": (
        "THE GATE field - emit as a BARE int (gated-fields-must-be-bare). The "
        "gold-incorrect set size; the scissor's residual is a SUBSET of this, "
        "so >=100 is required for a meaningful residual CI."
    ),
    "incorrect_fraction": (
        "Balance check - must be materially > the v4 corpus's 1.7% so a real "
        "residual set exists."
    ),
    "primary_source": (
        "prmbench | fover_v3_fallback - provenance + whether 9-axis labels are present."
    ),
    "error_axis_coverage": (
        "Which of the 9 PRMBench axes are represented (null for fover fallback) - "
        "enables exp3860's per-axis independence decomposition."
    ),
    "label_mapping_note": (
        "How PRMBench step-correctness mapped to the binary gold label - auditability."
    ),
    "schema_compatible_with_2837": (
        "Bare bool - confirms 1 item scored end-to-end through the frozen ensemble path."
    ),
    "preconditions_checked": (
        "Records source availability checked before building - pre-empts fabricating a corpus."
    ),
    "random_seed": (
        "Determinism for the balanced draw - a third party must reproduce the exact subset."
    ),
    "reproducibility_checksum": (
        "Content hash of (source, N, seed, balance) catches silent drift vs any replication."
    ),
    "inference_substrate": "aggregation substrate (no model loaded)",
    "duration_s": "real wall-clock of the ETL",
}


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for the deterministic Exp 3858 corpus build."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    target_n: int = DEFAULT_TARGET_N
    min_incorrect_steps: int = DEFAULT_MIN_INCORRECT_STEPS
    target_incorrect_fraction: float = 0.5
    prmbench_dataset_id: str = DEFAULT_PRMBENCH_DATASET_ID
    prmbench_filename: str = DEFAULT_PRMBENCH_FILENAME
    prmbench_reachability_url: str = DEFAULT_PRMBENCH_REACHABILITY_URL

    @property
    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "data" / "step_error_balanced_v2.json"


@dataclass(frozen=True)
class SourceAvailability:
    """Source precondition result captured before any corpus rows are built."""

    prmbench_reachable: bool
    fover_v3_paths: list[Path]
    preconditions_checked: list[dict[str, Any]]


SchemaValidator = Callable[[dict[str, Any]], bool]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def check_source_availability(
    config: BuildConfig,
    *,
    command_runner: CommandRunner | None = None,
) -> SourceAvailability:
    """Check PRMBench reachability and FoVer fallback files before building."""

    runner = command_runner or subprocess.run
    try:
        result = runner(
            ["curl", "-sf", "-o", "/dev/null", config.prmbench_reachability_url],
            capture_output=True,
            text=True,
            timeout=20,
        )
        prmbench_reachable = result.returncode == 0
        prmbench_detail = (
            "reachable" if prmbench_reachable else f"curl_exit_{result.returncode}"
        )
    except Exception as exc:  # pragma: no cover - defensive around host tools.
        prmbench_reachable = False
        prmbench_detail = f"{type(exc).__name__}: {exc}"

    fover_paths = [
        config.repo_root / "data" / filename
        for filename in FOVER_FALLBACK_FILES
        if (config.repo_root / "data" / filename).exists()
    ]
    preconditions_checked = [
        {
            "resource": "prmbench_hf",
            "available": prmbench_reachable,
            "detail": prmbench_detail,
        },
        {
            "resource": "fover_v3_fallback",
            "available": bool(fover_paths),
            "detail": [str(path.relative_to(config.repo_root)) for path in fover_paths],
        },
    ]
    return SourceAvailability(
        prmbench_reachable=prmbench_reachable,
        fover_v3_paths=fover_paths,
        preconditions_checked=preconditions_checked,
    )


def load_prmbench_records(config: BuildConfig) -> list[dict[str, Any]]:
    """Download PRMBench via Hugging Face Hub and return decoded JSONL rows."""

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=config.prmbench_dataset_id,
            repo_type="dataset",
            filename=config.prmbench_filename,
        )
    )
    return _read_jsonl(path)


def parse_prmbench_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map PRMBench modified-process steps to exp2837 scoreable step rows."""

    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        process = row.get("modified_process")
        if not isinstance(process, list):
            continue
        error_steps = _int_set(row.get("error_steps", []))
        base_id = str(row.get("idx") or f"prmbench_{row_index}")
        question = str(
            row.get("question")
            or row.get("modified_question")
            or row.get("original_question")
            or ""
        )
        axis = str(row.get("classification") or "").strip() or None
        for step_index, raw_step in enumerate(process, start=1):
            step_text = str(raw_step).strip()
            if not step_text:
                continue
            items.append(
                {
                    "question_id": f"{base_id}:step_{step_index:04d}",
                    "question": question,
                    "step_text": step_text,
                    "label": "incorrect" if step_index in error_steps else "correct",
                    "error_axis": axis,
                    "source": "prmbench",
                }
            )
    return items


def load_fover_fallback_items(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load, normalize, and dedupe FoVer v3 fallback rows."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)

    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        step_text = str(row.get("step_text") or "").strip()
        if not step_text:
            continue
        try:
            label = normalize_label(row.get("label"))
        except ValueError:
            continue
        question_id = str(row.get("question_id") or f"fover_fallback_{row_index}")
        item = {
            "question_id": question_id,
            "question": str(row.get("question") or ""),
            "step_text": step_text,
            "label": label,
            "error_axis": None,
            "source": "fover_v3_fallback",
        }
        items.append(item)
    return dedupe_items(items)


def normalize_label(raw_label: Any) -> str:
    """Normalize FoVer/PRMBench labels to exp2837-compatible strings."""

    if isinstance(raw_label, bool):
        return "correct" if raw_label else "incorrect"
    if raw_label in {"incorrect", 1, "1"}:
        return "incorrect"
    if raw_label in {"correct", 0, "0"}:
        return "correct"
    raise ValueError(f"unsupported step-error label: {raw_label!r}")


def dedupe_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe scoreable rows without changing their labels."""

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("question_id") or item.get("question") or ""),
            _sha256_text(str(item.get("step_text") or "")),
            str(item.get("label") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def select_balanced_items(items: Sequence[dict[str, Any]], config: BuildConfig) -> list[dict[str, Any]]:
    """Draw a deterministic balanced subset, preferring all available PRMBench axes."""

    incorrect = [item for item in items if item.get("label") == "incorrect"]
    correct = [item for item in items if item.get("label") == "correct"]
    if not incorrect or not correct:
        return []

    n_incorrect = min(len(incorrect), config.target_n // 2)
    n_correct = min(len(correct), config.target_n - n_incorrect)
    n_incorrect = min(n_incorrect, n_correct)
    n_correct = min(n_correct, n_incorrect)
    rng = random.Random(config.random_seed)
    selected_incorrect = _sample_with_axis_coverage(incorrect, n_incorrect, rng)
    selected_correct = rng.sample(correct, n_correct)
    selected = [*selected_incorrect, *selected_correct]
    rng.shuffle(selected)
    return selected


def validate_schema_compatible_with_2837(
    item: dict[str, Any],
    *,
    label_to_int: Callable[[Any], int] | None = None,
    score_text_verifiers: Callable[[list[str]], dict[str, list[float]]] | None = None,
) -> bool:
    """Validate one row through the frozen exp2837 label and verifier hooks."""

    if not (item.get("question_id") or item.get("question")):
        return False
    step_text = str(item.get("step_text") or "")
    if not step_text:
        return False
    if item.get("label") not in {"correct", "incorrect"}:
        return False
    if label_to_int is None or score_text_verifiers is None:
        from carnot.eval.fover_memory_leakage_v3 import (  # noqa: PLC0415
            _label_to_int,
            _score_text_verifiers,
        )

        label_to_int = label_to_int or _label_to_int
        score_text_verifiers = score_text_verifiers or _score_text_verifiers
    try:
        label_to_int(item["label"])
        scores = score_text_verifiers([step_text])
    except Exception:
        return False
    return bool(scores) and all(len(values) == 1 for values in scores.values())


def build_corpus_artifact(
    config: BuildConfig,
    *,
    availability: SourceAvailability | None = None,
    prmbench_records: Iterable[dict[str, Any]] | None = None,
    schema_validator: SchemaValidator | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the in-memory artifact for the best available source."""

    start = time.perf_counter() if started_s is None else started_s
    availability = availability or check_source_availability(config)
    validator = schema_validator or validate_schema_compatible_with_2837

    if availability.prmbench_reachable:
        records = list(prmbench_records) if prmbench_records is not None else load_prmbench_records(config)
        all_items = parse_prmbench_records(records)
        primary_source = "prmbench"
        label_mapping_note = (
            "PRMBench error_steps are 1-based indices into modified_process; "
            "label='incorrect' iff the step index is listed in error_steps, "
            "otherwise label='correct'. error_axis is copied from classification."
        )
    elif availability.fover_v3_paths:
        all_items = load_fover_fallback_items(availability.fover_v3_paths)
        primary_source = "fover_v3_fallback"
        label_mapping_note = (
            "FoVer v3 fallback rows already carry binary labels; labels are normalized "
            "to correct/incorrect strings and error_axis is null because FoVer v3 does "
            "not expose PRMBench's nine-axis taxonomy."
        )
    else:
        return _artifact(
            config=config,
            items=[],
            primary_source="none",
            label_mapping_note="No source available; no label mapping applied.",
            preconditions_checked=availability.preconditions_checked,
            schema_compatible=False,
            started_s=start,
            finished_s=time.perf_counter() if now_s is None else now_s,
            honest_verdict="blocked_no_step_error_source",
        )

    selected = select_balanced_items(all_items, config)
    schema_compatible = bool(selected) and validator(selected[0])
    finished = time.perf_counter() if now_s is None else now_s
    honest_verdict = _honest_verdict(
        primary_source=primary_source,
        n_items=len(selected),
        n_incorrect_steps=_count_incorrect(selected),
        schema_compatible=schema_compatible,
        config=config,
    )
    return _artifact(
        config=config,
        items=selected,
        primary_source=primary_source,
        label_mapping_note=label_mapping_note,
        preconditions_checked=availability.preconditions_checked,
        schema_compatible=schema_compatible,
        started_s=start,
        finished_s=finished,
        honest_verdict=honest_verdict,
    )


def write_corpus_artifact(config: BuildConfig, **kwargs: Any) -> dict[str, Any]:
    """Build and write `data/step_error_balanced_v2.json`."""

    artifact = build_corpus_artifact(config, **kwargs)
    output_path = config.resolved_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    """CLI entrypoint for `scripts/experiments/experiment_3858...py`."""

    repo_root = Path(__file__).resolve().parents[3]
    artifact = write_corpus_artifact(BuildConfig(repo_root=repo_root))
    print(json.dumps({"output_path": str(BuildConfig(repo_root=repo_root).resolved_output_path), "honest_verdict": artifact["honest_verdict"]}, indent=2))
    return 0


def _artifact(
    *,
    config: BuildConfig,
    items: list[dict[str, Any]],
    primary_source: str,
    label_mapping_note: str,
    preconditions_checked: list[dict[str, Any]],
    schema_compatible: bool,
    started_s: float,
    finished_s: float,
    honest_verdict: str,
) -> dict[str, Any]:
    n_incorrect = _count_incorrect(items)
    return {
        "artifact": "step_error_balanced_v2",
        "schema": "carnot.step_error_balanced.v2",
        "honest_verdict": honest_verdict,
        "n_items": len(items),
        "n_incorrect_steps": n_incorrect,
        "incorrect_fraction": round(n_incorrect / len(items), 6) if items else 0.0,
        "primary_source": primary_source,
        "error_axis_coverage": _error_axis_coverage(items),
        "label_mapping_note": label_mapping_note,
        "schema_compatible_with_2837": bool(schema_compatible),
        "preconditions_checked": preconditions_checked,
        "random_seed": config.random_seed,
        "reproducibility_checksum": _reproducibility_checksum(
            source=primary_source,
            seed=config.random_seed,
            items=items,
        ),
        "inference_substrate": "aggregation_etl_no_model_loaded",
        "duration_s": round(finished_s - started_s, 6),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "items": items,
    }


def _honest_verdict(
    *,
    primary_source: str,
    n_items: int,
    n_incorrect_steps: int,
    schema_compatible: bool,
    config: BuildConfig,
) -> str:
    if not schema_compatible:
        return "blocked_schema_compatible_with_2837"
    if primary_source == "prmbench" and n_items >= config.target_n and n_incorrect_steps >= config.min_incorrect_steps:
        return (
            "complete: balanced_step_error_corpus_v2_"
            f"n{n_items}_nincorrect{n_incorrect_steps}_sourceprmbench_9axistrue_schema_ok"
        )
    if primary_source == "fover_v3_fallback" and n_items >= config.target_n and n_incorrect_steps >= config.min_incorrect_steps:
        return (
            "complete: balanced_step_error_corpus_v2_"
            f"n{n_items}_nincorrect{n_incorrect_steps}_sourcefover_v3_fallback_"
            "9axisfalse_schema_ok"
        )
    return (
        "complete: balanced_corpus_v2_fover_fallback_"
        f"n{n_items}_nincorrect{n_incorrect_steps}_"
        "below_target_scissor_will_widen_or_inconclusive"
    )


def _sample_with_axis_coverage(
    items: Sequence[dict[str, Any]],
    n_items: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        axis = item.get("error_axis")
        if axis:
            by_axis.setdefault(str(axis), []).append(item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for axis in sorted(by_axis):
        if len(selected) >= n_items:
            break
        representative = rng.choice(by_axis[axis])
        selected.append(representative)
        selected_ids.add(id(representative))

    remaining = [item for item in items if id(item) not in selected_ids]
    selected.extend(rng.sample(remaining, n_items - len(selected)))
    return selected


def _honest_item_signature(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": item.get("question_id"),
        "label": item.get("label"),
        "error_axis": item.get("error_axis"),
        "step_text_sha256": _sha256_text(str(item.get("step_text") or "")),
    }


def _reproducibility_checksum(*, source: str, seed: int, items: Sequence[dict[str, Any]]) -> str:
    payload = {
        "source": source,
        "seed": seed,
        "n": len(items),
        "n_incorrect": _count_incorrect(items),
        "items": [_honest_item_signature(item) for item in items],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error_axis_coverage(items: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(item["error_axis"]) for item in items if item.get("error_axis")})


def _count_incorrect(items: Sequence[dict[str, Any]]) -> int:
    return sum(1 for item in items if item.get("label") == "incorrect")


def _int_set(values: Any) -> set[int]:
    result: set[int] = set()
    if not isinstance(values, list):
        return result
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
