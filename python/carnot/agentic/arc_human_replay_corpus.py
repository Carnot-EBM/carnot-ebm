"""Staged ARC Public Demo replay shards for frame-change training.

Spec refs: REQ-ARC-FCP-4495, SCENARIO-ARC-FCP-4495.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
SHARD_DIRNAME = "shards"
MANIFEST_NAME = "manifest.json"
SHARD_SCHEMA = "carnot.arc_human_replay.frame_action_delta.v1"


def decode_trajectory(value: Any) -> list[Any]:
    """REQ-ARC-FCP-4495: decode mirror trajectories without trusting one exact encoding.

    The reachable mirrors have used JSON strings nested inside another JSON
    string.  Decoding a few stable layers keeps the staging path robust while
    still failing closed to an empty trajectory for malformed rows.
    """

    decoded = value.decode("utf-8") if isinstance(value, bytes) else value
    for _ in range(4):
        if not isinstance(decoded, str):
            break
        stripped = decoded.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    return list(decoded) if isinstance(decoded, list) else []


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    if any(len(row) != width for row in value):
        return False
    return all(isinstance(cell, int | float) and not isinstance(cell, bool) for row in value for cell in row)


def _normalize_grid(value: Any) -> list[list[int]]:
    return [[int(cell) for cell in row] for row in value]


def extract_frame(value: Any) -> list[list[int]] | None:
    """REQ-ARC-FCP-4495: find the rendered frame/grid in a replay event."""

    if _is_grid(value):
        return _normalize_grid(value)
    if isinstance(value, list) and value and all(_is_grid(item) for item in value):
        return _normalize_grid(value[-1])
    if not isinstance(value, Mapping):
        return None
    for key in ("frame", "grid", "screen", "observation", "cells", "board"):
        if key in value:
            found = extract_frame(value[key])
            if found is not None:
                return found
    for key in ("data", "state", "payload", "result"):
        if key in value:
            found = extract_frame(value[key])
            if found is not None:
                return found
    return None


def extract_action(event: Any) -> Any:
    """REQ-ARC-FCP-4495: recover the action adjacent to a frame transition."""

    if not isinstance(event, Mapping):
        return {"missing": True}
    for key in ("action", "action_input", "input", "command", "player_action", "move"):
        if key in event:
            return event[key]
    data = event.get("data")
    if isinstance(data, Mapping):
        for key in ("action", "action_input", "input", "command", "player_action", "move"):
            if key in data:
                return data[key]
    return {"missing": True}


def frame_delta(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> float:
    """REQ-ARC-FCP-4495: compute the changed-cell fraction between two frames."""

    if not before or not after:
        return 0.0
    if len(before) != len(after) or len(before[0]) != len(after[0]):
        return 1.0
    total = len(before) * len(before[0])
    if total <= 0:
        return 0.0
    changed = sum(
        1
        for y, row in enumerate(before)
        for x, cell in enumerate(row)
        if int(cell) != int(after[y][x])
    )
    return float(changed) / float(total)


def _flatten_level_thresholds(actions_by_level: Any) -> list[tuple[int, int]]:
    thresholds: list[tuple[int, int]] = []

    def visit(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
        ):
            thresholds.append((int(value[0]), int(value[1])))
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(actions_by_level)
    return sorted((level, count) for level, count in thresholds if level > 0 and count > 0)


def level_progress(row: Mapping[str, Any], step_index: int) -> float:
    """REQ-ARC-FCP-4495: map a transition index onto replay level completion progress."""

    thresholds = _flatten_level_thresholds(row.get("actions_by_level"))
    if thresholds:
        max_level = max(level for level, _count in thresholds)
        completed = max((level for level, count in thresholds if int(step_index) >= count), default=0)
        return min(1.0, float(completed) / float(max_level))
    total_actions = int(row.get("total_actions") or 0)
    if total_actions <= 0:
        return 0.0
    return min(1.0, float(step_index) / float(total_actions))


def iter_training_examples(rows: Iterable[Mapping[str, Any]]) -> Iterator[JsonDict]:
    """REQ-ARC-FCP-4495: yield local ``frame, action -> frame_delta, level_progress`` rows."""

    for row_index, row in enumerate(rows):
        events = decode_trajectory(row.get("trajectory"))
        frame_events: list[tuple[Any, list[list[int]]]] = []
        for event in events:
            frame = extract_frame(event)
            if frame is not None:
                frame_events.append((event, frame))
        for transition_index, ((before_event, before_frame), (after_event, after_frame)) in enumerate(
            zip(frame_events, frame_events[1:], strict=False),
            start=1,
        ):
            action = extract_action(after_event)
            if action == {"missing": True}:
                action = extract_action(before_event)
            yield {
                "schema": SHARD_SCHEMA,
                "env": str(row.get("env") or ""),
                "guid": str(row.get("guid") or ""),
                "source_row_index": row_index,
                "step_index": transition_index,
                "frame": before_frame,
                "action": action,
                "frame_delta": frame_delta(before_frame, after_frame),
                "level_progress": level_progress(row, transition_index),
            }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_shards(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    max_examples_per_shard: int = 4096,
    max_examples: int | None = None,
) -> JsonDict:
    """SCENARIO-ARC-FCP-4495: write reusable JSONL shards plus a manifest."""

    output_path = Path(output_dir)
    shard_dir = output_path / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in shard_dir.glob("train-*.jsonl"):
        stale.unlink()

    shard_infos: list[JsonDict] = []
    current_handle: Any | None = None
    current_path: Path | None = None
    current_count = 0
    shard_index = -1
    example_count = 0

    def close_current() -> None:
        nonlocal current_handle, current_path, current_count
        if current_handle is None or current_path is None:
            return
        current_handle.close()
        shard_infos.append(
            {
                "path": str(current_path.relative_to(output_path)),
                "rows": current_count,
                "sha256": _sha256_file(current_path),
            }
        )
        current_handle = None
        current_path = None
        current_count = 0

    try:
        for example in iter_training_examples(rows):
            if max_examples is not None and example_count >= int(max_examples):
                break
            if current_handle is None or current_count >= int(max_examples_per_shard):
                close_current()
                shard_index += 1
                current_path = shard_dir / f"train-{shard_index:05d}.jsonl"
                current_handle = current_path.open("w", encoding="utf-8")
            current_handle.write(json.dumps(example, sort_keys=True) + "\n")
            current_count += 1
            example_count += 1
    finally:
        close_current()

    manifest: JsonDict = {
        "schema": SHARD_SCHEMA,
        "example_count": example_count,
        "shard_count": len(shard_infos),
        "shards": shard_infos,
        "source_metadata": dict(source_metadata or {}),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(data_dir: Path | str) -> JsonDict:
    """SCENARIO-ARC-FCP-4495: load the staging manifest without upstream access."""

    path = Path(data_dir) / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_shards(data_dir: Path | str, *, limit: int | None = None) -> Iterator[JsonDict]:
    """SCENARIO-ARC-FCP-4495: stream staged training examples from local shards."""

    base = Path(data_dir)
    manifest = load_manifest(base)
    emitted = 0
    for shard in manifest.get("shards", []):
        shard_path = base / str(shard["path"])
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and emitted >= int(limit):
                    return
                emitted += 1
                yield json.loads(line)


def iter_parquet_rows(paths: Iterable[Path | str]) -> Iterator[JsonDict]:  # pragma: no cover
    """Read reachable Hugging Face mirror parquet rows in small batches."""

    import pyarrow.parquet as pq

    for path in paths:
        parquet_file = pq.ParquetFile(Path(path))
        for batch in parquet_file.iter_batches(batch_size=8):
            for row in batch.to_pylist():
                yield dict(row)


def write_training_shards_from_parquet(
    parquet_paths: Iterable[Path | str],
    output_dir: Path | str,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    max_examples_per_shard: int = 4096,
    max_examples: int | None = None,
) -> JsonDict:  # pragma: no cover
    """Stage parquet mirror rows into local JSONL shards."""

    return write_training_shards(
        iter_parquet_rows(parquet_paths),
        output_dir,
        source_metadata=source_metadata,
        max_examples_per_shard=max_examples_per_shard,
        max_examples=max_examples,
    )
