"""REQ-REPORT-9167-V781: reuse repeated artifact classification, never a verdict.

The checker repeatedly asks the same structural question for numeric pairs.
Each lookup binds the full content, current checker bytes and marker settings.
The cache exists only while one actual checker invocation executes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from scripts import adversarial_verify as av


@contextmanager
def preprocessing() -> Iterator[None]:
    """Keep repeated preprocessing inside one actual checker call."""
    original = av._is_arc_generation_or_exploration_artifact
    cache: dict[tuple[str, str, str], bool] = {}

    def classify(value: dict[str, Any]) -> bool:
        key = (
            canonical_hash(value),
            sha256_file(Path(av.__file__)),
            canonical_hash(
                [
                    sorted(av.OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS),
                    av._LEVER_EXERCISE_CONTEXT_MARKERS,
                    av._ARC_GAME_IDS,
                ]
            ),
        )
        if key not in cache:
            cache[key] = original(value)
        return cache[key]

    with patch.object(av, "_is_arc_generation_or_exploration_artifact", classify):
        yield


def current_report(path: Path) -> dict[str, Any]:
    """Execute all checks while reusing only identical structural preprocessing."""
    with preprocessing():
        return dict(av._verify_artifact_impl(path))
