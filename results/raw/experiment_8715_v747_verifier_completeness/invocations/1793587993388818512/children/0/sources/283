"""FoVer corpus loader for REQ-KONA-019 Boltzmann-GPT training."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FoVerItem:
    """Single FoVer trace with binary correctness label."""

    text: str
    label: int


class FoVerDataset:
    """Load FoVer correct/incorrect traces from rows or a JSONL corpus."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        rows: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        source_rows = list(rows) if rows is not None else self._read_rows(self._default_path(path))
        self._items = [self._item_from_row(row) for row in source_rows]
        labels = {item.label for item in self._items}
        if labels != {0, 1}:
            raise ValueError("FoVerDataset requires both correct and incorrect labels")

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self._items]

    @property
    def labels(self) -> list[int]:
        return [item.label for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> FoVerItem:
        return self._items[index]

    def __iter__(self) -> Iterator[FoVerItem]:
        return iter(self._items)

    @staticmethod
    def _default_path(path: str | Path | None) -> Path:
        default = Path(__file__).resolve().parents[3] / "data" / "fover_corpus.jsonl"
        return Path(path) if path is not None else default

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _item_from_row(row: dict[str, Any]) -> FoVerItem:
        text = str(row.get("step_text") or row.get("response"))
        raw_label = row.get("label", row.get("is_correct"))
        if isinstance(raw_label, bool):
            label = int(raw_label)
        else:
            label = 1 if str(raw_label).lower() == "correct" else 0
        return FoVerItem(text=text, label=label)
