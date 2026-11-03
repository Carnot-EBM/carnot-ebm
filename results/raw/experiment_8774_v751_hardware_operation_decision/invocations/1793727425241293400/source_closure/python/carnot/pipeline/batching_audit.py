"""BatchingEnforcementAudit — scan experiment scripts for sequential question loops.

**Why sequential inference is slow (RETRO-041, resolved as sub-item):**
    Running inference one question at a time (``for q in questions: infer(q)``) is
    3-5x slower than batched inference at the GPU utilization patterns seen in Carnot
    benchmarks.  The root cause is GPU underutilization: at batch_size=1, the GPU
    spends most of its wall time on kernel launch overhead and memory transfers,
    while the actual compute SMs (Streaming Multiprocessors) are starved.  At
    batch_size=8 for GSM8K (longer prompts) and batch_size=4 for HumanEval (variable
    length, code generation), SM utilization rises dramatically and throughput scales
    near-linearly.

    The .35 retrospective estimated 5% wall time savings from batching enforcement
    alone — roughly 250 minutes recovered per 5000-minute milestone.

**What this module provides:**
    ``BatchingViolation`` — a single detected sequential-loop violation in a script,
    including the file path, line number, matched pattern, and severity.

    ``BatchingEnforcementAudit`` — scans a directory of scripts for ``for q in``-style
    loops without a nearby ``BatchedInferenceRunner`` instantiation.  Produces a list
    of violations the conductor can act on.

    ``BatchingEnforcementAudit.recommended_batch_size(task_type)`` — returns the
    standard batch size for a given task type so experiments use consistent values:
    - gsm8k: 8 (longer prompts, lower per-question throughput)
    - humaneval: 4 (code generation, variable output length)
    - default: 8

**Severity semantics:**
    - ``'high'``   — sequential for-loop iterating over 50+ question variables without
                     any BatchedInferenceRunner in the same file.  These are the
                     primary bottleneck scripts.
    - ``'medium'`` — sequential for-loop pattern detected but fewer question variables
                     or BatchedInferenceRunner is present (already partially migrated).

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# Patterns that indicate sequential question-by-question inference loops.
# These are heuristic: they look for ``for <var> in <questions_var>:`` constructs
# where the iterable name contains "question", "sample", "prompt", or "item".
_SEQUENTIAL_LOOP_PATTERN = re.compile(
    r"for\s+\w+\s+in\s+\w*(question|sample|prompt|item|prob)s?\w*\s*:",
    re.IGNORECASE,
)

# Minimum number of sequential-loop pattern matches in a file to classify as high severity.
# A file with one loop might be iterating over a tiny list; the concern is bulk inference.
_HIGH_SEVERITY_THRESHOLD = 1

# Standard batch sizes by task type — see module docstring for rationale.
_BATCH_SIZE_STANDARDS: dict[str, int] = {
    "gsm8k": 8,
    "humaneval": 4,
    "default": 8,
}


@dataclass
class BatchingViolation:
    """One detected sequential-inference violation in a script.

    A violation means: the script iterates over questions one-by-one without using
    ``BatchedInferenceRunner``, which causes 3-5x slower GPU throughput due to kernel
    launch overhead and SM underutilization.

    Fields
    ------
    script_path : str
        Absolute or relative path to the script that contains the violation.
    line_no : int
        1-based line number of the sequential loop pattern.
    pattern : str
        The matched source line (stripped) that triggered the violation.
    severity : str
        ``'high'`` when no ``BatchedInferenceRunner`` is found anywhere in the file;
        ``'medium'`` when ``BatchedInferenceRunner`` is present but sequential loops
        also exist (partially migrated script).

    Spec: REQ-INFRA-047, SCENARIO-INFRA-055
    """

    script_path: str
    line_no: int
    pattern: str
    severity: str

    @property
    def is_high_severity(self) -> bool:
        """Return True when this violation has severity='high'.

        High-severity violations are scripts with sequential for-loops over question
        variables where no BatchedInferenceRunner is found anywhere in the file.
        These are the primary bottleneck scripts — migrating them recovers the most
        wall time per milestone.

        Spec: REQ-INFRA-047
        """
        return self.severity == "high"


class BatchingEnforcementAudit:
    """Scan a directory of scripts for sequential question loops without BatchedInferenceRunner.

    **Why this audit matters:**
        Before Exp 437 introduced BatchedInferenceRunner, every experiment iterated
        over questions with a plain ``for q in questions:`` loop, achieving batch_size=1
        throughput.  Even after the runner was available, most scripts were never
        migrated — they continued to call inference one question at a time.  The .35
        retro estimated 5% wall time savings (≈250 min/milestone) from enforcing
        batching across all scripts.

        This audit closes the visibility gap: it tells the conductor exactly which
        scripts are still sequential so they can be prioritized for migration.

    Parameters
    ----------
    scripts_dir : str
        Path to the directory containing experiment scripts (typically ``scripts/``).

    Spec: REQ-INFRA-047, REQ-INFRA-048,
          SCENARIO-INFRA-055, SCENARIO-INFRA-056
    """

    def __init__(self, scripts_dir: str) -> None:
        self.scripts_dir = scripts_dir

    def scan(self) -> list[BatchingViolation]:
        """Scan all Python scripts in scripts_dir for sequential question loops.

        For each ``.py`` file in ``scripts_dir``:
        1. Read its source.
        2. Check whether ``BatchedInferenceRunner`` appears anywhere in the file.
        3. Find every line matching the sequential-loop pattern.
        4. Emit a ``BatchingViolation`` for each match:
           - severity='high' when BatchedInferenceRunner is absent (no migration at all)
           - severity='medium' when BatchedInferenceRunner is present but loops exist
             (partial migration — some questions still sequential)

        Returns
        -------
        list[BatchingViolation]
            One entry per sequential-loop line detected across all scripts.

        Spec: REQ-INFRA-047, SCENARIO-INFRA-055
        """
        violations: list[BatchingViolation] = []
        scripts_path = Path(self.scripts_dir)
        if not scripts_path.exists():
            _log.warning("scripts_dir does not exist: %s", self.scripts_dir)
            return violations

        for py_file in sorted(scripts_path.glob("*.py")):
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError as exc:
                _log.warning("Cannot read %s: %s", py_file, exc)
                continue

            has_batched_runner = "BatchedInferenceRunner" in source
            severity = "medium" if has_batched_runner else "high"

            lines = source.splitlines()
            for line_no, line in enumerate(lines, start=1):
                if _SEQUENTIAL_LOOP_PATTERN.search(line):
                    violations.append(
                        BatchingViolation(
                            script_path=str(py_file),
                            line_no=line_no,
                            pattern=line.strip(),
                            severity=severity,
                        )
                    )

        _log.info(
            "BatchingEnforcementAudit scanned %s: %d violations found",
            self.scripts_dir,
            len(violations),
        )
        return violations

    def recommended_batch_size(self, task_type: str) -> int:
        """Return the standard batch size for a given task type.

        Standard batch sizes were chosen based on observed GPU throughput:
        - gsm8k=8: GSM8K prompts are longer (multi-step arithmetic), so each forward
          pass is slower per token.  batch_size=8 balances VRAM usage and throughput.
        - humaneval=4: Code generation has variable output length; longer outputs mean
          more memory pressure per sequence.  batch_size=4 avoids OOM while still
          batching.
        - default=8: For unknown task types, 8 is the safe default — it's the batch
          size that recovers most of the throughput gap vs. batch_size=1 without
          risking OOM on 24 GB GPUs.

        Parameters
        ----------
        task_type : str
            One of ``'gsm8k'``, ``'humaneval'``, or any string (falls back to default).

        Returns
        -------
        int
            Recommended batch size.

        Spec: REQ-INFRA-048, SCENARIO-INFRA-056
        """
        return _BATCH_SIZE_STANDARDS.get(task_type.lower(), _BATCH_SIZE_STANDARDS["default"])
