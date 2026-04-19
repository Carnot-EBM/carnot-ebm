"""DualGPUHarness and HarnessAudit — enforce cuda:1 assignment for dual-model benchmark scripts.

**Why this exists (RETRO-041):**
    GPU 1 (RTX 3090, 24 GB VRAM) has been idle for THREE CONSECUTIVE MILESTONES despite
    DualGPURunner existing in experiment_template.py since RETRO-034.  The adoption gap:
    DualGPURunner was never wired into actual benchmark harness scripts.  Every dual-model
    experiment (Exp 476, 478, etc.) loads both models onto GPU 0, causing VRAM saturation
    and sequential inference.  GPU 1 sits idle while GPU 0 sweats through 48 GB of work
    that could be split 24+24.

    This module closes the adoption gap with two components:

    1. ``DualGPUHarness`` — a wrapper that patches any model_specs list to assign
       cuda:0 to the first model and cuda:1 to the second, without requiring each
       benchmark script to be manually rewritten.  Any script that instantiates this
       and calls apply() gets correct dual-GPU assignment automatically.

    2. ``HarnessAudit`` — a static scanner that reads benchmark scripts in scripts/,
       detects which ones load two or more models, and flags those that do NOT already
       have an explicit cuda:1 assignment.  Produces AuditFinding records the conductor
       can act on in future milestones.

**Why device_map={'': 'cuda:N'} not device_map='auto':**
    device_map='auto' spreads a single model across ALL visible GPUs for tensor
    parallelism / offloading.  With two models, both models fight over both GPUs —
    neither gets a clean forward pass on dedicated silicon.  {'': 'cuda:N'} pins
    every layer of model N to GPU N, which is the RETRO-025 lesson applied correctly.

**Eligibility gate (live_mode + n_gpus >= 2):**
    In CI there are no physical GPUs, so apply() must be a no-op that leaves specs
    unchanged.  The live_mode flag (set True when CARNOT_FORCE_LIVE=1 is in env)
    gates the actual device injection so CI never tries to reference cuda:1.

Spec: REQ-INFRA-045, REQ-INFRA-046,
      SCENARIO-INFRA-053, SCENARIO-INFRA-054
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AuditFinding
# ---------------------------------------------------------------------------


@dataclass
class AuditFinding:
    """One finding from HarnessAudit.scan().

    Fields
    ------
    script_path : str
        Path to the script that was inspected.
    has_dual_model_load : bool
        True when the script contains two or more model-load calls
        (heuristic: two or more calls to _load_* functions, or two or more
        MODEL_SPECS entries, or two or more 'hf_id' string literals).
    has_cuda1_assignment : bool
        True when the script contains an explicit 'cuda:1' string literal,
        indicating at least one model is pinned to GPU 1.
    needs_fix : bool
        True when has_dual_model_load=True and has_cuda1_assignment=False.
        These are the scripts that silently waste GPU 1.

    Spec: REQ-INFRA-046, SCENARIO-INFRA-054
    """

    script_path: str
    has_dual_model_load: bool
    has_cuda1_assignment: bool
    needs_fix: bool


# ---------------------------------------------------------------------------
# HarnessAudit
# ---------------------------------------------------------------------------


class HarnessAudit:
    """Scan benchmark scripts for dual-model loads missing explicit cuda:1 assignment.

    This is a static analysis tool — it reads Python source files and looks for
    patterns that indicate a script loads two or more models but has NOT pinned
    any model to GPU 1.  It does NOT execute the scripts.

    Why static analysis rather than runtime introspection:
        The scripts load large language models (7B+ parameters).  Loading them
        just to audit GPU assignment would be expensive and require hardware.
        Source pattern matching is zero-cost and runs in CI.

    Heuristic for "dual model load":
        A script is flagged as having a dual-model load when ANY of these
        conditions are met:
        - It contains >= 2 occurrences of 'hf_id' (HuggingFace model ID key)
        - It contains >= 2 calls to functions named _load_* (common naming pattern
          in Carnot benchmark scripts for model loaders)

    Heuristic for "has cuda:1 assignment":
        The source contains the literal string 'cuda:1'.

    Parameters
    ----------
    scripts_dir : str
        Directory to scan.  Scans all *.py files non-recursively.

    Spec: REQ-INFRA-046, SCENARIO-INFRA-054
    """

    def __init__(self, scripts_dir: str) -> None:
        self._scripts_dir = scripts_dir

    def scan(self) -> list[AuditFinding]:
        """Scan all Python files in scripts_dir and return AuditFinding per file.

        Only returns findings for files that have at least one model-load indicator.
        Files with no model loads are omitted (they are not benchmark harnesses).

        Returns
        -------
        list[AuditFinding]
            One entry per script that appears to load at least one model.
            Findings with needs_fix=True are the actionable ones.
        """
        findings: list[AuditFinding] = []
        scripts_path = Path(self._scripts_dir)

        if not scripts_path.is_dir():
            _log.warning("HarnessAudit: scripts_dir '%s' does not exist", self._scripts_dir)
            return findings

        for py_file in sorted(scripts_path.glob("*.py")):
            finding = self._audit_file(py_file)
            if finding is not None:
                findings.append(finding)

        _log.info(
            "HarnessAudit: scanned %d scripts — %d dual-model, %d need cuda:1 fix",
            len(list(scripts_path.glob("*.py"))),
            sum(1 for f in findings if f.has_dual_model_load),
            sum(1 for f in findings if f.needs_fix),
        )
        return findings

    def _audit_file(self, py_file: Path) -> AuditFinding | None:
        """Audit a single Python file.  Returns None if the file has no model loads."""
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _log.warning("HarnessAudit: could not read '%s': %s", py_file, exc)
            return None

        # Heuristic: count 'hf_id' occurrences as proxy for model load calls.
        hf_id_count = source.count("hf_id")
        # Heuristic: count calls to functions matching _load_* pattern.
        load_fn_count = self._count_load_fn_calls(source)

        # A script is a benchmark harness if it has at least one model-load indicator.
        has_any_load = hf_id_count >= 1 or load_fn_count >= 1
        if not has_any_load:
            return None  # not a benchmark harness, skip

        has_dual_model_load = hf_id_count >= 2 or load_fn_count >= 2
        has_cuda1_assignment = "cuda:1" in source

        needs_fix = has_dual_model_load and not has_cuda1_assignment

        return AuditFinding(
            script_path=str(py_file),
            has_dual_model_load=has_dual_model_load,
            has_cuda1_assignment=has_cuda1_assignment,
            needs_fix=needs_fix,
        )

    @staticmethod
    def _count_load_fn_calls(source: str) -> int:
        """Count function calls to names matching _load_* via AST parsing.

        Falls back to 0 if the source cannot be parsed (e.g. syntax errors in
        very early experiment drafts).
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Direct call: _load_something(...)
                if isinstance(func, ast.Name) and func.id.startswith("_load_"):
                    count += 1
                # Method call: obj._load_something(...)
                elif isinstance(func, ast.Attribute) and func.attr.startswith("_load_"):
                    count += 1
        return count


# ---------------------------------------------------------------------------
# DualGPUHarness
# ---------------------------------------------------------------------------


class DualGPUHarness:
    """Patch a model_specs list to assign cuda:0 and cuda:1 automatically.

    This mixin/wrapper removes the need for each benchmark script to manually
    set 'gpu' and 'device_map' per model.  Any script that calls apply() gets
    correct dual-GPU pinning without a harness rewrite.

    Why this is a class, not a function:
        The eligibility check (n_gpus and live_mode) belongs with the apply logic
        so callers can introspect is_eligible before loading any model.  Having
        both in one class also makes the state explicit and testable.

    Parameters
    ----------
    n_gpus : int
        Number of CUDA GPUs detected at runtime.  Pass 0 in CI/CPU environments.
        Detected at instantiation; does not auto-detect (caller owns detection
        so the class stays unit-testable without hardware).
    live_mode : bool
        True when CARNOT_FORCE_LIVE=1 is set in the environment.  In CI,
        live_mode=False so apply() is a safe no-op.

    Spec: REQ-INFRA-045, SCENARIO-INFRA-053
    """

    def __init__(self, n_gpus: int = 2, live_mode: bool = False) -> None:
        self._n_gpus = n_gpus
        self._live_mode = live_mode

    @classmethod
    def from_env(cls) -> "DualGPUHarness":
        """Construct from current environment (CARNOT_FORCE_LIVE + torch.cuda.device_count).

        Convenience factory for production use.  Unit tests should use the
        constructor directly to avoid torch import and GPU detection.
        """
        live_mode = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        n_gpus = 0
        if live_mode:
            try:
                import torch  # noqa: PLC0415

                n_gpus = torch.cuda.device_count()
            except ImportError:
                n_gpus = 0
        return cls(n_gpus=n_gpus, live_mode=live_mode)

    @property
    def is_eligible(self) -> bool:
        """True when dual-GPU assignment should be applied.

        Conditions (ALL must hold):
        1. n_gpus >= 2  — at least 2 physical GPUs present
        2. live_mode    — CARNOT_FORCE_LIVE=1 is set (CI skips this)

        Why gate on live_mode: in CI there are no GPUs.  Without this gate,
        apply() would inject cuda:N device maps that crash immediately on a
        CPU-only machine.  live_mode=False in CI means apply() is always a no-op.
        """
        return self._n_gpus >= 2 and self._live_mode

    def apply(self, model_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign cuda:0 to the first model and cuda:1 to the second (when eligible).

        When eligible (is_eligible=True):
            - model_specs[0]['gpu'] = 0, model_specs[0]['device_map'] = {'': 'cuda:0'}
            - model_specs[1]['gpu'] = 1, model_specs[1]['device_map'] = {'': 'cuda:1'}
            - Additional models are assigned to the last available GPU with a warning.

        When NOT eligible (CI mode, single GPU, or live_mode=False):
            Returns model_specs unchanged.  No device maps are injected.

        Parameters
        ----------
        model_specs : list[dict]
            Each spec should have at minimum a 'name' key.  The method mutates
            the dicts in-place and returns the same list.

        Returns
        -------
        list[dict]
            The same model_specs list with 'gpu' and 'device_map' injected (or
            unchanged if not eligible).

        Spec: REQ-INFRA-045, SCENARIO-INFRA-053
        """
        if not self.is_eligible:
            _log.debug(
                "DualGPUHarness: not eligible (n_gpus=%d, live_mode=%s) — specs unchanged",
                self._n_gpus,
                self._live_mode,
            )
            return model_specs

        for i, spec in enumerate(model_specs):
            gpu_idx = min(i, self._n_gpus - 1)
            if i >= self._n_gpus:
                _log.warning(
                    "DualGPUHarness: model %d ('%s') exceeds GPU count %d — "
                    "assigning to GPU %d (last available).  "
                    "True parallelism requires reducing model count to match GPU count.",
                    i,
                    spec.get("name", f"model_{i}"),
                    self._n_gpus,
                    gpu_idx,
                )
            spec["gpu"] = gpu_idx
            spec["device_map"] = {"": f"cuda:{gpu_idx}"}

        _log.info(
            "DualGPUHarness: assigned %d models — cuda:0..cuda:%d",
            len(model_specs),
            min(len(model_specs) - 1, self._n_gpus - 1),
        )
        return model_specs
