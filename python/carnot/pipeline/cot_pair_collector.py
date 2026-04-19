"""CoTPairCollector — collect and atomically flush Chain-of-Thought reasoning pairs.

**Why this exists (Exp 478, RETRO-038):**

    The 200q live benchmark (Exp 467 and Exp 478) collects reasoning traces
    (CoT pairs) during the pipeline pass for downstream JEPA retrain experiments.
    Previously these pairs were held in a list in memory and written in a single
    json.dump call at the end of the experiment.  If the experiment was interrupted,
    all collected pairs were lost.

    CoTPairCollector fixes this by:
    1. Accumulating pairs in memory as they are collected.
    2. Offering ``flush()`` which writes the pairs atomically to disk using a
       tmp-rename pattern, so a crash mid-write never leaves a partial file.
    3. Returning the count of pairs written so callers can embed it in the artifact.

**Why atomic write (tmp-rename):**

    json.dump writes the file byte-by-byte.  A crash mid-write leaves a truncated
    file that is not valid JSON.  Writing to a .tmp file and renaming atomically
    guarantees that any reader either sees the full file or no file — never a
    partial one.  POSIX rename is atomic when source and target are on the same
    filesystem, which is guaranteed here.

Spec: REQ-BENCH-027, SCENARIO-BENCH-046 (Exp 478)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class CoTPairCollector:
    """Accumulate Chain-of-Thought pairs and flush them atomically to a JSON file.

    Each pair represents one question's reasoning trace from the pipeline pass.
    Required keys: model, question, cot_text, correct.

    Parameters
    ----------
    output_path : str
        Path where the JSON file will be written when ``flush()`` is called.
        The directory will be created if it does not exist.

    Spec: REQ-BENCH-027, SCENARIO-BENCH-046
    """

    def __init__(self, output_path: str) -> None:
        self._output_path = Path(output_path)
        self._pairs: list[dict[str, Any]] = []

    def add(self, model: str, question: str, cot_text: str, correct: bool) -> None:
        """Add one CoT pair to the in-memory collection.

        Parameters
        ----------
        model : str
            Human-readable model name (e.g. 'Gemma4-E4B-it').
        question : str
            The original question text.
        cot_text : str
            The model's chain-of-thought response (may include intermediate steps).
        correct : bool
            Whether the model's final answer was correct after the pipeline pass.
        """
        self._pairs.append({
            "model": model,
            "question": question,
            "cot_text": cot_text,
            "correct": correct,
        })

    def flush(self) -> int:
        """Write all accumulated pairs atomically to disk and return the count.

        Creates the parent directory if it does not exist.  Writes to a .tmp
        file and renames atomically — POSIX rename guarantees the reader sees
        either the complete file or no file.

        Returns
        -------
        int
            Number of pairs written.  0 if no pairs were accumulated.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._output_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._pairs, indent=2))
        tmp_path.replace(self._output_path)
        n = len(self._pairs)
        _log.info("CoTPairCollector: flushed %d pairs to %s", n, self._output_path)
        return n

    def __len__(self) -> int:
        """Return the number of pairs currently accumulated (before flush)."""
        return len(self._pairs)
