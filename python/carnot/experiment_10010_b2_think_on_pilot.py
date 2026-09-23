"""Experiment 10010: replay live think-ON world-model induction on the B2 windows.

REQ-ARC-WMTE-10010. This is the harness for the pilot that
`docs/research-notes/b2-positive-control-2026-09-23.md` pre-registers (section
"Pre-registered design for the pending think-ON pilot"). The note fixes the ten
windows, the grid masks, the held-out row exclusions, the metrics, the controls,
and the codeonly baseline. This module follows it and changes none of it.

What one run does, in order:

1. PRECONDITIONS. Model file, llama-server binary, GPU 1 by UUID and idle, the
   process name, positive-control evidence, and the kernel's induce flags (no
   unlisted CARNOT_ARC_* flag). A miss writes `blocked_<resource>` and stops
   before any server starts. The shard is read first, so a rebuild needs no GPU.
2. WINDOWS. Load the ten recorded first-call windows from the positive-control
   reports, with a sha256 per file.
3. PROMPT FIDELITY. Build the think-ON prompt with the live builder and prove it
   equals the recorded prompt minus the codeonly directive and one fence. Prove
   no held-out answer appears in it.
4. CONTROLS. Score identity, expert, and the recorded codeonly first shots in
   the same wrapper. A miss means the metric is broken, so the run stops.
5. GENERATION (real run only). One live `generate()` call per window on GPU 1.
6. SCORING and the artifact. Each held-out row runs in its own new process that
   holds only that row's input, never an answer or another row
   (experiment_10010_engine_child.py).

The scored agent runs vLLM with NVFP4 weights; this card cannot. The pilot runs
llama.cpp with Q4_K_M and records every such difference in the artifact.

`--dry-run` does steps 1-4 and 6 with the recorded codeonly first shots standing
in for model output. It starts no server and touches no GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from carnot import experiment_10010_engine_child as engine_child
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_engine_call_guard import default_rss_delta_bytes, default_timeout_s
from carnot.agentic.arc_llm_reinduction import _proposal_prefix
from carnot.agentic.arc_world_model_trust_energy import (
    _split_prefix_heldout,
    cegis_accept_split_enabled,
)
from carnot.paths import repo_root as _repo_root

EXPERIMENT_ID = 10010
REQUIREMENT_ID = "REQ-ARC-WMTE-10010"
SCHEMA = "carnot.experiment_10010_b2_think_on_pilot.v1"

# ---- Pre-registered design (research note, 2026-09-23). Do not edit to fit a result. ----
PILOT_WINDOWS: tuple[str, ...] = (
    "su15",
    "sp80",
    "ft09",
    "g50t",
    "m0r0",
    "dc22",
    "wa30",
    "ka59",
    "sb26",
    "ar25",
)
# Grid rows that show a hidden step counter. They are zeroed before any comparison.
MASK_GRID_ROWS: dict[str, tuple[int, ...]] = {
    "g50t": (63,),
    "m0r0": (0, 63),
    "dc22": (63,),
    "wa30": (63,),
    "ka59": (63,),
}
# Held-out rows whose answer depends on hidden undo history. Left out of the masked metrics.
EXCLUDED_HELDOUT_ROWS: dict[str, tuple[int, ...]] = {"sb26": (20,), "ar25": (19,)}
# Only these windows can pass the unmasked live gate with a correct engine.
LIVE_GATE_MEANINGFUL: tuple[str, ...] = ("su15", "sp80", "ft09")
PREREGISTERED_BASELINE_MEAN = 0.13
BASELINE_TOLERANCE = 0.01
# The evidence file stores each window's mean to 3 decimals.
PER_WINDOW_BASELINE_TOLERANCE = 0.0005
RECORDED_SEEDS: tuple[str, ...] = ("7491001", "7491002", "7491003")
FIRST_CALL_SEED = "7491001"
# The pre-registration calls 3-4 changing rows "coarse"; below 5 is flagged in the artifact.
FEW_CHANGING_ROWS = 5

# ---- Evidence locations, relative to the repository root. ----
POSITIVE_CONTROL_REL = Path("results/raw/b2_positive_control_2026_09_23")
B2_RAW_REL = Path("results/raw/experiment_10009_b2_induction_gate_measurement_v3")
DEFAULT_ARTIFACT_REL = Path("results/experiment_10010_b2_think_on_pilot.json")
DEFAULT_OUTPUT_REL = Path("results/raw/experiment_10010_b2_think_on_pilot")

# ---- Measured serving setup (outer loop, 2026-09-23). GPU 0 belongs to the conductor. ----
GPU1_UUID = "GPU-7971baff-9583-eaa6-2292-393f930a28f9"
GPU0_UUID = "GPU-b52387a2-c625-de87-8d34-e6f64e684bab"
GPU1_INDEX = "1"
GPU1_MAX_USED_MIB = 500
MIN_RESIDENCY_MIB = 15000
DEFAULT_LLAMA_SERVER = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
DEFAULT_HF_CACHE = Path.home() / ".cache/huggingface/hub"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
N_CTX = 131072
SERVER_FLAGS: tuple[str, ...] = (
    "-ngl",
    "999",
    "-c",
    str(N_CTX),
    "--parallel",
    "1",
    "--cache-type-k",
    "q8_0",
    "--cache-type-v",
    "q8_0",
    "-fit",
    "off",
    "--ctx-checkpoints",
    "2",
)
# An explicit port. 8919 is the live default and is held by another server on this box.
DEFAULT_PORT = 8996
SERVER_HEALTH_WAIT_S = 1800.0

# ---- Live generation settings. ----
LIVE_TRIES = 3
MAX_TOKENS = 131072
# Pre-registered live budget: 2,400 s per call, the kernel's CARNOT_ARC_INDUCE_TIMEOUT.
# This card decodes at 40.8 tok/s, close to the scored vLLM card's 40.0 at k=8, so the
# timeout binds at about 98k tokens here, before the local 108,720-token cap.
CALL_TIMEOUT_S = 2400
REQUIRED = ("engine", "is_level_complete")
SEED_BASE = 10010
HEARTBEAT_S = 60.0

# Measured decode rates on the scored card (tok/s per stream). The live-ladder rescore
# asks, per rate, whether a draw would have ended in the 2,400 s timeout on Kaggle.
LIVE_TIMEOUT_S = 2400
SCORED_DECODE_RATES_TOK_S: dict[str, float] = {
    "llamacpp_k1_52p2": 52.2,  # ops/known-issues.md, llama.cpp single stream
    "vllm_k8_40p0": 40.0,  # _vllm_max_seqs docstring, the shipped k=8
}

# Always set by the Kaggle kernel (scripts/kaggle/submission_kernel/main.py).
KAGGLE_KERNEL_ENV: dict[str, str] = {
    "CARNOT_ARC_INDUCE_TOOL_LOOP": "repair",
    "CARNOT_ARC_INDUCE_TOOL_TURNS": "8",
    "CARNOT_ARC_INDUCE_MAX_TOKENS": "131072",
    "CARNOT_ARC_INDUCE_TIMEOUT": "2400",
    "CARNOT_ARC_EXPLORE_DIVERSITY": "1",
}
# Also set by the kernel, but only on some paths. They choose the backend and weights,
# which this card cannot reproduce, so the pilot never applies them. Recorded only.
KAGGLE_KERNEL_CONDITIONAL_ENV: dict[str, str] = {
    "CARNOT_ARC_LLM_BACKEND": "vllm, when the NVFP4 safetensors dataset is attached (it is)",
    "CARNOT_ARC_VLLM_MODEL_DIR": "the attached safetensors directory, with the vLLM backend",
    "CARNOT_ARC_GGUF_PATH": "the Qwen3.8-27B-NVFP4 GGUF, for the llama.cpp fallback",
    "CARNOT_ARC_MTP": "1 or 0, from the draft-head probe (llama.cpp fallback)",
    "CARNOT_ARC_SERVER_LOG_DIR": "/kaggle/working (log location only)",
    "CARNOT_ARC_E3_DIR": "/kaggle/working/arc_e3 (output location only)",
}
# What the scored agent actually runs, for the artifact. Not run here: sm_86 has no FP4.
SCORED_PATH_WEIGHTS: dict[str, str] = {
    "primary": "vLLM, Qwen3.8-27B NVFP4 safetensors, fp8 KV cache, --max-num-seqs 8",
    "fallback": "llama.cpp, Qwen3.8-27B-NVFP4 GGUF",
    "this_pilot": "llama.cpp, Qwen3.8-27B Q4_K_M GGUF, q8_0 KV cache, one stream",
}
# The only CARNOT_ARC_* names the shell may carry into a real run (fail closed). A value
# of None allows any value. The harness sets CARNOT_ARC_GENERATOR_SEED itself per window.
INDUCTION_ENV_ALLOWLIST: dict[str, Optional[str]] = {
    **KAGGLE_KERNEL_ENV,
    "CARNOT_ARC_SERVER_LOG_DIR": None,
    "CARNOT_ARC_E3_DIR": None,
    "CARNOT_ARC_GENERATOR_SEED": None,
}

IDENTITY_ENGINE_SOURCE = "def engine(grid, action, data=None):\n    return grid\n"

_CODEONLY_FENCE = "\n```python\n"


def progress(message: str) -> None:
    """One flushed line. A long GPU run that prints nothing cannot be watched."""
    print(f"[exp{EXPERIMENT_ID} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


class WindowError(RuntimeError):
    """A pre-registered window could not be loaded as the reports describe it."""


class PromptTransformError(RuntimeError):
    """The recorded prompt does not have the codeonly shape this harness strips."""


@dataclass(frozen=True)
class EvidencePaths:
    """Where the recorded evidence lives. Tests point this at `tmp_path` copies."""

    repo_root: Path
    positive_control: Path
    b2_raw: Path

    @classmethod
    def under(cls, repo_root: Path) -> "EvidencePaths":
        root = Path(repo_root)
        return cls(root, root / POSITIVE_CONTROL_REL, root / B2_RAW_REL)

    def workflow_result(self) -> Path:
        return self.positive_control / "workflow_result.json"

    def pilot_baseline(self) -> Path:
        return self.positive_control / "synth" / "pilot_baseline.json"

    def expert_engine(self, game: str) -> Path:
        return self.positive_control / game / "expert_engine.py"

    def request(self, game: str, seed: str) -> Path:
        return self.b2_raw / f"{game}__seed-{seed}" / "requests" / "00_request.json"

    def response(self, game: str, seed: str) -> Path:
        return self.b2_raw / f"{game}__seed-{seed}" / "requests" / "00_response.json"


@dataclass
class WindowSpec:
    """One pre-registered window: the recorded rows plus the fixed masks and exclusions."""

    game: str
    index: int
    window_file: Path
    window_sha256: str
    rows: list
    n_prefix: int
    heldout_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]
    mask_rows: tuple[int, ...]
    cell: int
    reported_digest: dict[str, Any]
    report_split_text: str

    @property
    def visible_rows(self) -> list:
        return list(self.rows[: self.n_prefix])

    def summary(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "window_file": str(self.window_file),
            "window_sha256": self.window_sha256,
            "n_rows": len(self.rows),
            "visible_indices": [0, self.n_prefix - 1],
            "heldout_indices": list(self.heldout_indices),
            "excluded_heldout_indices": list(self.excluded_indices),
            "mask_grid_rows": list(self.mask_rows),
            "cell": self.cell,
            "reported_digest": dict(self.reported_digest),
        }


def load_control_reports(paths: EvidencePaths) -> dict[str, dict[str, Any]]:
    """Map game -> its positive-control report (the `control` block of the workflow)."""
    data = json.loads(paths.workflow_result().read_text())
    reports: dict[str, dict[str, Any]] = {}
    for entry in data.get("results") or []:
        control = entry.get("control")
        if isinstance(control, dict) and entry.get("game"):
            reports[str(entry["game"])] = control
    return reports


def _rebase_evidence_path(recorded: str, repo_root: Path) -> Path:
    # The reports name absolute paths in the main checkout. Rebase onto this repository,
    # so a worktree or a test fixture reads its own copy.
    marker = "results/raw/"
    idx = recorded.find(marker)
    if idx >= 0:
        return Path(repo_root) / recorded[idx:]
    return Path(recorded)


def parse_window_file(report_text: str, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    """The report's `window_file` is prose: a path, then notes in parentheses."""
    first = str(report_text).split(" (", 1)[0].strip()
    if not first.endswith(".jsonl"):
        raise WindowError(f"window_file does not start with a .jsonl path: {first[:120]!r}")
    digest: dict[str, Any] = {}
    m = re.search(r"sha256 ([0-9a-f]{64})", str(report_text))
    if m:
        digest["sha256"] = m.group(1)
    m = re.search(r"md5 ([0-9a-f]{32})", str(report_text))
    if m:
        digest["md5"] = m.group(1)
    return _rebase_evidence_path(first, repo_root), digest


# A range like "17-24". A sentence may end right after it ("17-24."), so only a word
# character or a decimal part ("24.5") next to the numbers rejects the match.
_RANGE = re.compile(r"(?<!\w)(?<!\d\.)(\d{1,3})\s*-\s*(\d{1,3})(?!\w)(?!\.\d)")


def parse_split(report_text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Read "visible rows 0-16 ... held-out rows 17-24" from the report prose."""
    pairs = [(int(a), int(b)) for a, b in _RANGE.findall(str(report_text))]
    visible = next(((a, b) for a, b in pairs if a == 0 and b > 0), None)
    if visible is None:
        raise WindowError("split text names no visible range starting at row 0")
    heldout = next(((a, b) for a, b in pairs if a == visible[1] + 1 and b >= a), None)
    if heldout is None:
        raise WindowError("split text names no held-out range after the visible range")
    return visible, heldout


def load_rows(path: Path) -> list:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(
            e3.Transition(
                np.asarray(d["grid"]),
                int(d["action"]),
                d.get("data"),
                np.asarray(d["next_grid"]),
                int(d["level_before"]),
                int(d["level_after"]),
            )
        )
    return rows


def load_window(
    game: str, index: int, report: Mapping[str, Any], paths: EvidencePaths
) -> WindowSpec:
    """Load one window and cross-check the report's split against the live split code."""
    window_file, digest = parse_window_file(str(report.get("window_file", "")), paths.repo_root)
    if not window_file.exists():
        raise WindowError(f"window file missing: {window_file}")
    raw = window_file.read_bytes()
    file_sha = sha256_bytes(raw)
    if "sha256" in digest and digest["sha256"] != file_sha:
        raise WindowError(f"window sha256 {file_sha} differs from the report's {digest['sha256']}")
    if "md5" in digest and hashlib.md5(raw).hexdigest() != digest["md5"]:
        raise WindowError("window md5 differs from the report")
    rows = load_rows(window_file)
    split_text = str(report.get("split", ""))
    (v0, v1), (h0, h1) = parse_split(split_text)
    prefix, heldout = _split_prefix_heldout(rows)
    proposal = _proposal_prefix(rows)
    n_prefix = v1 + 1
    # Three sources must agree on the cut: the report, the live gate split, and the
    # live proposer prefix. A disagreement means the replay is not the recorded call.
    if not (len(prefix) == n_prefix == len(proposal) and h0 == n_prefix and h1 == len(rows) - 1):
        raise WindowError(
            f"split disagreement: report visible 0-{v1}, held-out {h0}-{h1}; live prefix "
            f"{len(prefix)}, proposal prefix {len(proposal)}, rows {len(rows)}"
        )
    width = int(np.asarray(rows[0].grid).shape[1])
    # ARC-AGI-3 frames are 64 pixels wide, so the logical cell is 64 // width. The
    # prompt states this number, so the fidelity gate catches a wrong value.
    cell = max(1, 64 // width)
    return WindowSpec(
        game=game,
        index=index,
        window_file=window_file,
        window_sha256=file_sha,
        rows=rows,
        n_prefix=n_prefix,
        heldout_indices=tuple(range(n_prefix, len(rows))),
        excluded_indices=tuple(EXCLUDED_HELDOUT_ROWS.get(game, ())),
        mask_rows=tuple(MASK_GRID_ROWS.get(game, ())),
        cell=cell,
        reported_digest=digest,
        report_split_text=split_text[:200],
    )


# ---------------------------------------------------------------------------------------
# Prompt fidelity: the replayed prompt must be the recorded prompt, byte for byte.
# ---------------------------------------------------------------------------------------


def recorded_prompt(paths: EvidencePaths, game: str, seed: str = FIRST_CALL_SEED) -> str:
    return str(json.loads(paths.request(game, seed).read_text())["prompt"])


def think_prompt_from_recorded(recorded: str) -> tuple[str, list[str]]:
    """Undo the codeonly wrapping that B2's shim added inside `generate()`.

    B2 switched think mode off only inside `generate()`. So `induce()` had already built
    the think-ON prompt, and `generate()` then added the codeonly directive in front and
    one pre-opened fence behind. Removing exactly those two gives the think-ON prompt.
    """
    directive = e3._L2_CODEONLY_DIRECTIVE
    if not recorded.startswith(directive):
        raise PromptTransformError("recorded prompt does not start with the codeonly directive")
    if not recorded.endswith(_CODEONLY_FENCE):
        raise PromptTransformError("recorded prompt does not end with one pre-opened fence")
    body = recorded[len(directive) : -len(_CODEONLY_FENCE)]
    transform = [
        f"removed the codeonly directive prefix ({len(directive)} chars)",
        f"removed one trailing pre-opened fence {_CODEONLY_FENCE!r}",
    ]
    return body, transform


class _PromptCaptured(Exception):
    """Raised by the capture stub so `induce()` stops before any network call."""


def _live_proposer(
    *,
    port: int = DEFAULT_PORT,
    model_path: Optional[str] = None,
) -> Any:
    """A proposer built the way the scored agent builds it (arc_competition_agent `_proposer`).

    `ffn_cpu_layers=0` is the Kaggle value, passed explicitly so construction never
    probes a GPU. max_tokens and timeout are the pre-registered local values.
    """
    return e3.LocalGGUFProposer(
        repo_substr=e3.ARC_LIVE_GENERATOR_REPO_SUBSTR,
        model_path=model_path,
        mtp=False,
        kv_quant="q8_0",
        no_think_prefix=e3.ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
        port=int(port),
        n_gpu_layers=999,
        ffn_cpu_layers=0,
        n_ctx=N_CTX,
        max_tokens=MAX_TOKENS,
        timeout=CALL_TIMEOUT_S,
    )


def capture_live_induce_call(game: str, visible_rows: Sequence[Any], cell: int) -> dict[str, Any]:
    """Run the live `induce()` up to its first `generate()` call and keep that call's arguments.

    This uses the live prompt builder, suffix, transition cap, and defect-gate rows as they
    are, instead of copies of them. The stub raises before any request is sent.
    """
    tool_loop = os.environ.get("CARNOT_ARC_INDUCE_TOOL_LOOP")
    if tool_loop in ("1", "selfparse"):
        # These values make induce() call the model before the combined prompt is built.
        raise PromptTransformError(f"CARNOT_ARC_INDUCE_TOOL_LOOP={tool_loop} would call the model")
    proposer = _live_proposer()
    captured: dict[str, Any] = {}

    def _capture(
        self: Any,
        prompt: str,
        required: tuple = REQUIRED,
        validate: Any = None,
        tries: int = LIVE_TRIES,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        captured.update(
            prompt=prompt,
            required=tuple(required),
            validate=validate,
            tries=int(tries),
            kwargs=dict(kwargs),
        )
        raise _PromptCaptured()

    proposer.generate = MethodType(_capture, proposer)
    try:
        proposer.induce(game, list(visible_rows), int(cell))
    except _PromptCaptured:
        pass
    if "prompt" not in captured:
        raise PromptTransformError("induce() returned before calling generate()")
    return captured


def _first_diff(a: str, b: str) -> Optional[int]:
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def prompt_fidelity(spec: WindowSpec, paths: EvidencePaths) -> dict[str, Any]:
    """Build the think-ON prompt live and compare it with the recorded first-call prompt."""
    result: dict[str, Any] = {"game": spec.game, "match": False}
    recorded = recorded_prompt(paths, spec.game)
    result["recorded_prompt_sha256"] = sha256_text(recorded)
    per_seed = {}
    for seed in RECORDED_SEEDS:
        p = paths.request(spec.game, seed)
        per_seed[seed] = (
            sha256_text(recorded_prompt(paths, spec.game, seed)) if p.exists() else None
        )
    result["recorded_prompt_sha256_per_seed"] = per_seed
    result["recorded_prompts_identical_across_seeds"] = len(set(per_seed.values())) == 1
    try:
        expected, transform = think_prompt_from_recorded(recorded)
    except PromptTransformError as exc:
        result["reason"] = f"recorded prompt shape: {exc}"
        return result
    result["transform"] = transform
    result["expected_think_prompt_sha256"] = sha256_text(expected)
    try:
        captured = capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    except PromptTransformError as exc:
        result["reason"] = f"live builder: {exc}"
        return result
    built = str(captured["prompt"])
    result["built_prompt_sha256"] = sha256_text(built)
    result["built_prompt_chars"] = len(built)
    result["captured_required"] = list(captured["required"])
    result["captured_tries"] = captured["tries"]
    kwargs = captured["kwargs"]
    result["captured_codeonly_eligible"] = kwargs.get("codeonly_eligible")
    shown = list(kwargs.get("engine_transitions") or [])
    # The defect-gate rows must be the visible row objects themselves, not copies.
    result["defect_gate_rows_are_visible_rows"] = len(shown) == spec.n_prefix and all(
        a is b for a, b in zip(shown, spec.visible_rows)
    )
    diff_at = _first_diff(built, expected)
    result["first_diff_index"] = diff_at
    if diff_at is not None:
        lo = max(0, diff_at - 60)
        result["built_excerpt"] = built[lo : diff_at + 120]
        result["expected_excerpt"] = expected[lo : diff_at + 120]
        result["reason"] = "built think-ON prompt differs from the recorded prompt"
    elif not result["defect_gate_rows_are_visible_rows"]:
        result["reason"] = "live induce passed rows other than the visible rows to generate()"
    elif tuple(captured["required"]) != REQUIRED or captured["tries"] != LIVE_TRIES:
        result["reason"] = "live induce call shape differs from the pre-registered call"
    else:
        result["match"] = True
    result["_captured"] = captured
    return result


_TRANSITION_LINE = re.compile(
    r"^--- ACTION(\d+)((?: data=.*?)?) \(level (\d+)->(\d+)\): "
    r"changed cells \(FULL, run-length\) = (.*)$",
    re.M,
)


def transition_line_key(row: Any) -> tuple:
    """The key of the line `_transitions_block` renders for one row."""
    click = f" data={row.data}" if row.data else ""
    return (
        int(row.action),
        click,
        int(row.level_before),
        int(row.level_after),
        e3._rle_delta_compact(np.asarray(row.grid), np.asarray(row.next_grid)),
    )


def heldout_leak_check(prompt: str, rows: Sequence[Any], n_prefix: int) -> dict[str, Any]:
    """Assert no held-out answer is in the prompt.

    Two checks. Every transition line in the prompt must be explained by a visible row;
    a line only a held-out row explains is a leak. And no held-out next grid may appear
    as a full grid unless a visible row shows the same grid.
    """
    visible = list(rows[:n_prefix])
    parsed = Counter(
        (int(a), click, int(lb), int(la), delta)
        for a, click, lb, la, delta in _TRANSITION_LINE.findall(prompt)
    )
    expected = Counter(transition_line_key(r) for r in visible)
    visible_keys = set(expected)
    heldout_keys = {i: transition_line_key(rows[i]) for i in range(n_prefix, len(rows))}
    # `extra` counts lines shown more often than the visible rows produce them. Any such
    # line a held-out row would render is a leak, even if a visible row renders it too.
    extra = parsed - expected
    line_leaks = sorted(i for i, k in heldout_keys.items() if extra.get(k, 0) > 0)
    heldout_values = set(heldout_keys.values())
    unexplained = [
        [*k[:4], k[4][:80]] for k in extra if k not in visible_keys and k not in heldout_values
    ]
    duplicated_visible = sum(
        n for k, n in extra.items() if k in visible_keys and k not in heldout_values
    )
    shown = [np.asarray(r.grid) for r in visible] + [np.asarray(r.next_grid) for r in visible]
    grid_leaks = []
    for i in range(n_prefix, len(rows)):
        nxt = np.asarray(rows[i].next_grid)
        if e3._rle_grid(nxt) in prompt and not any(np.array_equal(nxt, g) for g in shown):
            grid_leaks.append(i)
    missing = expected - parsed
    return {
        "passed": not line_leaks and not grid_leaks and not unexplained,
        "prompt_transition_lines": int(sum(parsed.values())),
        "visible_rows": len(visible),
        "visible_lines_missing_from_prompt": int(sum(missing.values())),
        "visible_lines_shown_twice": int(duplicated_visible),
        "heldout_line_leaks": line_leaks,
        "heldout_full_grid_leaks": grid_leaks,
        "unexplained_lines": unexplained[:10],
    }


# ---------------------------------------------------------------------------------------
# Scoring wrapper. Computed here, not from WorldModelVerifier's graded fields, because the
# verifier drops raised rows from every graded metric (defect 2 in the research note).
# ---------------------------------------------------------------------------------------


def build_mask(spec: WindowSpec) -> Optional[np.ndarray]:
    if not spec.mask_rows:
        return None
    mask = np.zeros(np.asarray(spec.rows[0].grid).shape, dtype=bool)
    for r in spec.mask_rows:
        mask[int(r), :] = True
    return mask


def _compile_engine(source: Optional[str], name: str) -> tuple[Any, Optional[str]]:
    if not source or not str(source).strip():
        return None, "no_engine_source"
    try:
        return compile(str(source), name, "exec"), None
    except SyntaxError as exc:
        return None, f"syntax_error: {exc.msg} line {exc.lineno}"


ENGINE_CHILD_PATH = Path(__file__).with_name("experiment_10010_engine_child.py")
# Recorded in every engine run. It names the defense: one new interpreter per row,
# holding only that row's input (spec REQ-ARC-WMTE-10010, amendment 2).
ROW_ISOLATION = "new process per row (exec, not fork); each holds one row's input only"
ROW_STDERR_TAIL_BYTES = 4096
# Tokens that suggest an engine looks at its own process or the disk. Recorded per
# engine; the per-row process makes them useless, but a reader should see them.
INTROSPECTION_IO_TOKENS: tuple[str, ...] = (
    "_getframe",
    "f_back",
    "f_locals",
    "inspect",
    "gc.",
    "builtins",
    "sys.modules",
    "ctypes",
    "open(",
    "__subclasses__",
    "subprocess",
    "socket",
    "os.environ",
    "pathlib",
)


@dataclass
class RowPrediction:
    """One held-out row's prediction from the child process."""

    grid: Optional[np.ndarray]
    error: str = ""


Predictor = Callable[[str, Sequence[dict[str, Any]], str], tuple[list[RowPrediction], dict]]


def introspection_tokens(source: Optional[str]) -> list[str]:
    text = str(source or "")
    return [t for t in INTROSPECTION_IO_TOKENS if t in text]


def _child_env() -> dict[str, str]:
    # No CARNOT_* flags and no PYTHONPATH reach the child. One BLAS thread keeps the
    # numbers the same as the earlier fork-based scorer, which also ran one thread.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    env.update(PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env["MKL_NUM_THREADS"] = "1"
    return env


def _row_input(row: Any) -> dict[str, Any]:
    """Only what the engine may see. The held-out answer never leaves this process."""
    return {
        "grid": np.asarray(row.grid).tolist(),
        "action": int(row.action),
        "data": copy.deepcopy(row.data),
    }


def _parse_child_row(raw: Any) -> RowPrediction:
    # The child's output is untrusted: accept only a plain list of integer rows.
    if not isinstance(raw, dict):
        return RowPrediction(None, "RowProcessError: malformed row result")
    if not raw.get("ok"):
        return RowPrediction(None, str(raw.get("error") or "RowProcessError: no error text")[:200])
    try:
        arr = np.asarray(raw.get("grid"))
    except Exception as exc:
        return RowPrediction(None, f"RowProcessError: {type(exc).__name__}"[:200])
    if arr.dtype.kind not in "iu" or arr.ndim != 2:
        return RowPrediction(None, f"OutputTypeError: parent refused dtype {arr.dtype.str!r}")
    return RowPrediction(arr.astype(np.int64), "")


def engine_child_sha256() -> Optional[str]:
    """Hash of the per-row scoring program, or None when it is missing (a crash test)."""
    try:
        return sha256_file(ENGINE_CHILD_PATH)
    except OSError:
        return None


def _row_deadline_s(timeout_s: Optional[float]) -> float:
    # Interpreter start, then module load and one call (each under the timer), then margin.
    load_and_call = 2 * float(timeout_s) if timeout_s is not None else engine_child.HARD_CAP_S
    return engine_child.STARTUP_S + load_and_call + engine_child.KILL_MARGIN_S


def _pump_row_process(
    proc: subprocess.Popen, payload: bytes, deadline: float
) -> tuple[bytes, bytes, str]:
    """Write the job and read both pipes until EOF, the deadline, or the size cap."""
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    in_fd, out_fd = proc.stdin.fileno(), proc.stdout.fileno()
    os.set_blocking(in_fd, False)
    pending = memoryview(payload)
    open_fds = {out_fd, proc.stderr.fileno()}
    out: list[bytes] = []
    err = b""
    size = 0
    while open_fds:
        left = deadline - time.monotonic()
        if left <= 0:
            return b"".join(out), err, "killed at the row deadline"
        writers = [in_fd] if pending else []
        readable, writable, _ = select.select(sorted(open_fds), writers, [], left)
        if writable:
            try:
                pending = pending[os.write(in_fd, pending[:65536]) :]
            except BlockingIOError:
                pass
            except OSError:
                pending = pending[:0]  # the process is gone; its read side is closed
            if not pending:
                proc.stdin.close()
        for fd in readable:
            chunk = os.read(fd, 65536)
            if not chunk:
                open_fds.discard(fd)
            elif fd == out_fd:
                out.append(chunk)
                size += len(chunk)
            else:
                err = (err + chunk)[-ROW_STDERR_TAIL_BYTES:]
        if size > engine_child.MAX_RESULT_BYTES:
            return b"".join(out), err, "killed: result too large"
    return b"".join(out), err, ""


def run_row_process(job: Mapping[str, Any], deadline_s: float) -> tuple[dict, Optional[dict]]:
    """Score one row in a NEW interpreter (exec, not fork) that holds only that row.

    Returns the row result and the process's own report. A crash, a kill, or an
    unreadable result becomes an error result, so the row scores 0.
    """
    payload = json.dumps(job).encode()
    with tempfile.TemporaryDirectory(prefix="exp10010_row_") as cwd:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-s", "-P", str(ENGINE_CHILD_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=_child_env(),
                close_fds=True,
            )
        except OSError as exc:
            return {"ok": False, "error": f"ChildProcessError: {exc}"[:200]}, None
        deadline = time.monotonic() + deadline_s
        try:
            out, err, killed = _pump_row_process(proc, payload, deadline)
            if not killed:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    killed = "killed at the row deadline after closing its pipe"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()
    if killed:
        return {"ok": False, "error": f"EngineCallTimeout: {killed}"}, None
    tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
    lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, ValueError):
        msg = f"RowProcessError: no valid result (exit {proc.returncode}): {tail[0]}"
        return {"ok": False, "error": msg[:200]}, None
    result = report.get("result") if isinstance(report, dict) else None
    if not isinstance(result, dict):
        return {"ok": False, "error": "RowProcessError: result is not an object"}, None
    info = report.get("child")
    return result, info if isinstance(info, dict) else None


def predict_rows_in_child(
    source: str, rows: Sequence[dict[str, Any]], label: str
) -> tuple[list[RowPrediction], dict[str, Any]]:
    """Run one engine over the held-out inputs, one NEW process per row.

    Each process gets one row's input and nothing else. The rows are consecutive, so
    row k+1's grid is row k's answer: a process holding two rows leaks an answer.
    A crashed, killed, or silent row process scores its row 0.
    """
    timeout_s = default_timeout_s()
    rss = default_rss_delta_bytes()
    rss_mb = None if rss is None else rss / (1024 * 1024)
    deadline_s = _row_deadline_s(timeout_s)
    meta: dict[str, Any] = {
        "isolation": ROW_ISOLATION,
        "timeout_s": timeout_s,
        "row_deadline_s": deadline_s,
        "engine_child_sha256": engine_child_sha256(),
        "n_row_processes": len(rows),
        "n_row_process_failures": 0,
    }
    preds: list[RowPrediction] = []
    for i, row in enumerate(rows):
        # The same per-row seed as the fork-based scorer: SEED_BASE plus the row's position.
        job = {
            "source": source,
            "name": label,
            "row": dict(row),
            "timeout_s": timeout_s,
            "rss_delta_mb": rss_mb,
            "seed": SEED_BASE + i,
        }
        result, info = run_row_process(job, deadline_s)
        if info is None:
            meta["n_row_process_failures"] += 1
            meta.setdefault("child_error", str(result.get("error"))[:200])
        else:
            meta.setdefault("child", info)
        preds.append(_parse_child_row(result))
    return preds, meta


def _is_levelup(row: Any) -> bool:
    return int(row.level_after) > int(row.level_before)


def score_engine(
    source: Optional[str],
    spec: WindowSpec,
    label: str,
    predictor: Optional[Predictor] = None,
) -> dict[str, Any]:
    """Pre-registered metrics for one engine on one window.

    primary: masked symmetric-union change fidelity, mean over held-out changing rows;
      a raised, wrong-type, or wrong-shape row scores 0.
    guard: no-op hallucination rate; a raised no-op row counts as hallucinated.
    secondary: masked exact accuracy; live unmasked exact accuracy and its 1.0 pass.
    Each held-out row runs in its own new process that holds only that row (child module).
    """
    predictor = predictor or predict_rows_in_child
    code_obj, compile_error = _compile_engine(source, f"<{label} {spec.game}>")
    mask = build_mask(spec)
    graded = [i for i in spec.heldout_indices if not _is_levelup(spec.rows[i])]
    preds: dict[int, RowPrediction] = {}
    run_meta: dict[str, Any] = {}
    if code_obj is not None and graded:
        got, run_meta = predictor(
            str(source), [_row_input(spec.rows[i]) for i in graded], f"{label}_{spec.game}"
        )
        preds = dict(zip(graded, got))
    fids: list[float] = []
    n_noop = n_halluc = n_exact = n_scored = 0
    n_raised = n_raised_changing = n_raised_noop = 0
    n_levelup = 0
    live_rows = live_exact = 0
    per_row: list[dict[str, Any]] = []
    for i in spec.heldout_indices:
        row = spec.rows[i]
        entry: dict[str, Any] = {"row": int(i)}
        if _is_levelup(row):
            # Same rule as the live verifier: a level-up row shows the next level's board.
            n_levelup += 1
            entry["status"] = "levelup_excluded"
            per_row.append(entry)
            continue
        p = preds.get(i) or RowPrediction(None, "no_engine")
        pred, err = p.grid, p.error
        nxt = np.asarray(row.next_grid)
        ok_shape = pred is not None and pred.shape == nxt.shape
        live_rows += 1
        try:
            live_exact += int(ok_shape and bool(np.array_equal(pred, nxt)))
        except Exception as exc:
            ok_shape = False  # a prediction that cannot be compared is never right
            entry["compare_error"] = f"{type(exc).__name__}: {exc}"[:160]
        if i in spec.excluded_indices:
            entry["status"] = "excluded_preregistered"
            per_row.append(entry)
            continue
        n_scored += 1
        g0 = e3.apply_hud_mask(np.asarray(row.grid), mask)
        g1 = e3.apply_hud_mask(nxt, mask)
        changed = not np.array_equal(g0, g1)
        entry["changing"] = bool(changed)
        if err:
            n_raised += 1
            entry["raised"] = err
        exact = False
        fid = 0.0
        if ok_shape:
            try:
                pg = e3.apply_hud_mask(pred, mask)
                exact = bool(np.array_equal(pg, g1))
                if changed:
                    union = (g0 != g1) | (pg != g0)
                    fid = float(((pg == g1) & union).sum() / union.sum())
            except Exception as exc:
                # A comparison that fails is the engine's fault, never a free pass.
                exact, fid = False, 0.0
                entry["compare_error"] = f"{type(exc).__name__}: {exc}"[:160]
        n_exact += int(exact)
        entry["exact"] = exact
        if changed:
            n_raised_changing += int(bool(err))
            fids.append(fid)
            entry["fidelity"] = fid
        else:
            n_noop += 1
            halluc = not exact
            n_raised_noop += int(bool(err))
            n_halluc += int(halluc)
            entry["hallucinated"] = halluc
        per_row.append(entry)
    live_acc = (live_exact / live_rows) if live_rows else None
    return {
        "engine_status": compile_error or "compiled",
        "primary_change_fidelity": float(np.mean(fids)) if fids else None,
        "n_changing_rows": len(fids),
        "noop_hallucination_rate": (n_halluc / n_noop) if n_noop else None,
        "n_noop_rows": n_noop,
        "masked_exact_accuracy": (n_exact / n_scored) if n_scored else None,
        "n_scored_rows": n_scored,
        "n_raised_rows": n_raised,
        "n_raised_changing_rows": n_raised_changing,
        "n_raised_noop_rows": n_raised_noop,
        "n_levelup_rows_excluded": n_levelup,
        "live_unmasked_exact_accuracy": live_acc,
        "live_unmasked_pass_1p0": bool(live_acc is not None and live_acc >= 1.0),
        "live_gate_meaningful": spec.game in LIVE_GATE_MEANINGFUL,
        "introspection_io_tokens": introspection_tokens(source),
        "engine_run": run_meta,
        "per_row": per_row,
    }


def _compact(score: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in score.items() if k != "per_row"}


# ---------------------------------------------------------------------------------------
# Controls: scored in the same wrapper, before any model output. A miss stops the run.
# ---------------------------------------------------------------------------------------


def response_content(response: Mapping[str, Any]) -> str:
    if isinstance(response.get("content"), str):
        return str(response["content"])
    try:
        return str(response["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def recorded_first_shot_source(
    paths: EvidencePaths, game: str, seed: str
) -> tuple[str, dict[str, Any]]:
    """The engine code the live codeonly path took from one recorded first-call response."""
    raw = paths.response(game, seed).read_bytes()
    response = json.loads(raw)
    content = response_content(response)
    # The same two steps as generate(): take the fenced block, else the raw text,
    # because the codeonly stop sequence consumed the closing fence.
    code = e3._extract_python(content) or content.strip()
    return code, {
        "seed": seed,
        "response_sha256": sha256_bytes(raw),
        "code_sha256": sha256_text(code),
        "tokens_predicted": response.get("tokens_predicted"),
        "stop_type": response.get("stop_type"),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def run_controls(specs: Sequence[WindowSpec], paths: EvidencePaths) -> dict[str, Any]:
    """Identity must score 0.0, expert 1.0, and codeonly first shots must reproduce 0.13."""
    progress("phase controls: begin (identity, expert, recorded codeonly first shots)")
    baseline_file = json.loads(paths.pilot_baseline().read_text())
    file_windows = baseline_file.get("per_window_fid_s1_s2_s3_mean") or {}
    per_window: dict[str, Any] = {}
    failures: list[str] = []
    window_means: list[float] = []
    for spec in specs:
        identity = score_engine(IDENTITY_ENGINE_SOURCE, spec, "identity")
        expert_path = paths.expert_engine(spec.game)
        expert_source = expert_path.read_text()
        expert = score_engine(expert_source, spec, "expert")
        shots = []
        for seed in RECORDED_SEEDS:
            code, meta = recorded_first_shot_source(paths, spec.game, seed)
            s = score_engine(code, spec, f"codeonly_{seed}")
            primary = s["primary_change_fidelity"]
            shots.append({**meta, **_compact(s), "primary_or_zero": primary or 0.0})
        window_mean = float(np.mean([s["primary_or_zero"] for s in shots]))
        window_means.append(window_mean)
        id_p = identity["primary_change_fidelity"]
        ex_p = expert["primary_change_fidelity"]
        # All three channels, not just the primary: a broken no-op or exact channel
        # would otherwise pass the controls unseen.
        id_ok = (
            id_p is not None
            and abs(id_p) < 1e-12
            and identity["noop_hallucination_rate"] in (0, 0.0, None)
        )
        ex_ok = (
            ex_p is not None
            and abs(ex_p - 1.0) < 1e-9
            and expert["noop_hallucination_rate"] in (0, 0.0, None)
            and expert["masked_exact_accuracy"] is not None
            and abs(expert["masked_exact_accuracy"] - 1.0) < 1e-9
        )
        if not id_ok:
            failures.append(f"identity_{spec.game}")
        if not ex_ok:
            failures.append(f"expert_{spec.game}")
        file_row = file_windows.get(spec.game)
        file_mean = None if not file_row else float(file_row[-1])
        # The evidence file rounds each window to 3 decimals, so 0.0005 is the widest
        # honest gap. A shift here that cancels in the mean would pass the mean check.
        if file_mean is None or abs(window_mean - file_mean) > PER_WINDOW_BASELINE_TOLERANCE:
            failures.append(f"codeonly_window_{spec.game}")
        per_window[spec.game] = {
            "identity": _compact(identity),
            "identity_ok": id_ok,
            "expert": _compact(expert),
            "expert_ok": ex_ok,
            "expert_engine_path": _relative(expert_path, paths.repo_root),
            "expert_engine_sha256": sha256_text(expert_source),
            "codeonly_first_shots": shots,
            "codeonly_window_mean": window_mean,
            "baseline_file_window_mean": file_mean,
        }
        progress(
            f"controls {spec.game}: identity={id_p} expert={ex_p} codeonly_mean={window_mean:.3f}"
        )
    reproduced = float(np.mean(window_means)) if window_means else None
    # The pre-registered 0.13 is the mean over all ten windows. A subset (tests only; the
    # CLI always runs all ten) is held to the evidence file's own per-window means instead.
    if {s.game for s in specs} == set(PILOT_WINDOWS):
        expected: Optional[float] = PREREGISTERED_BASELINE_MEAN
        expected_source = "preregistered mean over the ten windows"
    else:
        subset = [file_windows[s.game][-1] for s in specs if file_windows.get(s.game)]
        ok_subset = len(subset) == len(specs) and bool(subset)
        expected = float(np.mean(subset)) if ok_subset else None
        expected_source = "evidence-file mean over this window subset"
    baseline_ok = (
        reproduced is not None
        and expected is not None
        and abs(reproduced - expected) <= BASELINE_TOLERANCE
    )
    if not baseline_ok:
        failures.append("codeonly_baseline")
    diffs = [
        abs(v["codeonly_window_mean"] - v["baseline_file_window_mean"])
        for v in per_window.values()
        if v["baseline_file_window_mean"] is not None
    ]
    result = {
        "passed": not failures,
        "failures": failures,
        "identity_all_zero": all(v["identity_ok"] for v in per_window.values()),
        "expert_all_one": all(v["expert_ok"] for v in per_window.values()),
        "codeonly_baseline_mean_reproduced": reproduced,
        "codeonly_baseline_mean_preregistered": PREREGISTERED_BASELINE_MEAN,
        "codeonly_baseline_mean_file": baseline_file.get("mean"),
        "codeonly_baseline_expected": expected,
        "codeonly_baseline_expected_source": expected_source,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "codeonly_baseline_ok": baseline_ok,
        "max_abs_diff_vs_file_per_window": max(diffs) if diffs else None,
        "per_window_baseline_tolerance": PER_WINDOW_BASELINE_TOLERANCE,
        "per_window": per_window,
    }
    progress(
        f"phase controls: end passed={result['passed']} baseline={reproduced} failures={failures}"
    )
    return result


# ---------------------------------------------------------------------------------------
# Environment: the replay must run under the flags the scored kernel runs under.
# ---------------------------------------------------------------------------------------

_SAMPLING_ENV = (
    "CARNOT_ARC_INDUCE_TEMPERATURE",
    "CARNOT_ARC_INDUCE_TOP_P",
    "CARNOT_ARC_INDUCE_TOP_K",
    "CARNOT_ARC_INDUCE_THINKING_BUDGET",
    "CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK",
    "CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION",
)


def induction_resolution() -> dict[str, Any]:
    """Each setting that changes the induce prompt or its sampling, as the code resolves it."""
    return {
        "think_on": bool(e3.induce_think_on()),
        "transitions_k": e3._induce_transitions_k(),
        "prompt_enrichment": bool(e3.induce_prompt_enrichment_enabled()),
        "repeat_penalty": float(e3._induce_repeat_penalty()),
        "defect_reasks": int(e3._induce_defect_reasks()),
        "goal_defect_reasks": int(e3._goal_defect_reasks()),
        "defect_gate_owns_attempts": bool(e3._defect_gate_owns_attempts()),
        "vllm_backend": bool(e3._vllm_backend_active()),
        "tool_loop_enters_induce": os.environ.get("CARNOT_ARC_INDUCE_TOOL_LOOP")
        in ("1", "selfparse"),
        "sampling_overrides": {k: os.environ.get(k) for k in _SAMPLING_ENV},
    }


@contextmanager
def _kaggle_kernel_env() -> Iterator[None]:
    # Drop every CARNOT_ARC_* flag, then set only what the scored kernel sets.
    saved = dict(os.environ)
    try:
        for key in [k for k in os.environ if k.startswith("CARNOT_ARC_")]:
            del os.environ[key]
        os.environ.update(KAGGLE_KERNEL_ENV)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def induction_env_parity() -> dict[str, Any]:
    """Compare this shell's induce settings with the kernel's llama.cpp-shaped settings.

    It does NOT prove parity with the scored backend. The kernel runs vLLM with NVFP4
    weights when the safetensors dataset is attached, which this card cannot run. That
    difference is declared (`backend_parity: false`), never passed off as a match.
    """
    current = induction_resolution()
    with _kaggle_kernel_env():
        kernel = induction_resolution()
    diffs = {
        k: {"current": current[k], "kernel": kernel[k]} for k in current if current[k] != kernel[k]
    }
    return {
        "matches_kernel_llamacpp_shape": not diffs,
        "backend_parity": False,
        "backend_parity_note": (
            "Declared deviation. Scored path: "
            + SCORED_PATH_WEIGHTS["primary"]
            + ". This pilot: "
            + SCORED_PATH_WEIGHTS["this_pilot"]
            + ". Only the llama.cpp-shaped induce settings are compared."
        ),
        "kernel_conditional_env_not_applied": dict(KAGGLE_KERNEL_CONDITIONAL_ENV),
        "current": current,
        "kernel_llamacpp_shape": kernel,
        "differences": diffs,
    }


def induction_env_unlisted_flags() -> dict[str, Any]:
    """Fail closed on any CARNOT_ARC_* flag the scored kernel does not set.

    Some flags change generate() without changing the prompt (the goal-defect check,
    inert-engine rejection), so neither the parity check nor the prompt check sees them.
    """
    unlisted = {}
    wrong_value = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("CARNOT_ARC_"):
            continue
        if key not in INDUCTION_ENV_ALLOWLIST:
            unlisted[key] = value
        elif INDUCTION_ENV_ALLOWLIST[key] is not None and value != INDUCTION_ENV_ALLOWLIST[key]:
            wrong_value[key] = {"value": value, "kernel": INDUCTION_ENV_ALLOWLIST[key]}
    return {
        "ok": not unlisted and not wrong_value,
        "unlisted": unlisted,
        "wrong_value": wrong_value,
        "allowlist": sorted(INDUCTION_ENV_ALLOWLIST),
    }


def gate_env_flags() -> dict[str, Any]:
    """The live gate flags in effect. None of them changes a pilot metric."""
    metric = os.environ.get("CARNOT_ARC_TRUST_METRIC")
    return {
        "CARNOT_ARC_TRUST_METRIC": {
            "raw": metric,
            "effective": "cell_recall" if metric == "cell_recall" else "exact",
        },
        "CARNOT_ARC_WM_HUD_MASK": {
            "raw": os.environ.get("CARNOT_ARC_WM_HUD_MASK"),
            "effective": bool(e3.world_model_hud_mask_enabled()),
        },
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": {
            "raw": os.environ.get("CARNOT_ARC_CEGIS_ACCEPT_SPLIT"),
            "effective": bool(cegis_accept_split_enabled()),
        },
        "think_mode": {
            "raw": os.environ.get("CARNOT_ARC_INDUCE_THINK"),
            "effective_think_on": bool(e3.induce_think_on()),
            "scored_default": e3.ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT,
        },
        "engine_call_timeout_s": default_timeout_s(),
        "note": (
            "The pilot metrics come from this module's wrapper, so these gate flags change "
            "no pilot number. They are recorded because the live gate reads them."
        ),
    }


# --- GPU and server machinery --------------------------------------------------------
# Copied, not imported, from scripts/experiments/experiment_6440_qwen38_generator_h2h_arm.py.
# That script puts the main checkout on sys.path at import, and it splits across both
# cards. Changes here: GPU 1 only, pinned by UUID, and the measured single-card flags.

SmiRunner = Callable[[Sequence[str]], Optional[str]]


def run_nvidia_smi(args: Sequence[str]) -> Optional[str]:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=25)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_inventory(smi: SmiRunner) -> dict[str, dict[str, Any]]:
    out = smi(["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]) or ""
    inv: dict[str, dict[str, Any]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            inv[parts[1]] = {"index": parts[0], "memory_used_mib": int(parts[2])}
    return inv


def compute_apps(smi: SmiRunner) -> list[dict[str, Any]]:
    out = (
        smi(["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
        or ""
    )
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            used = int(parts[2]) if parts[2].isdigit() else None
            apps.append({"pid": int(parts[0]), "gpu_uuid": parts[1], "used_mib": used})
    return apps


def pid_residency_mib(pid: int, uuid: str, smi: SmiRunner) -> Optional[int]:
    """MiB this PID holds on this card. Residency is how we prove which card we got."""
    total = sum(
        a["used_mib"] or 0 for a in compute_apps(smi) if a["pid"] == pid and a["gpu_uuid"] == uuid
    )
    return total or None


def port_bound(port: int) -> bool:
    """True when something already holds the port. A bind test sees any listener, not
    only a healthy llama-server; a server still loading answers /health with an error."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return True
    return False


def _set_parent_death_signal() -> None:  # pragma: no cover - runs in the forked child
    # Linux only: if this harness dies, even by SIGKILL, the kernel sends the server
    # SIGTERM, so a killed run cannot leave 21 GB on GPU 1.
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    pr_set_pdeathsig = 1
    libc.prctl(pr_set_pdeathsig, int(signal.SIGTERM), 0, 0, 0)


def health_ok(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def props_model_path(port: int, timeout: float = 20.0) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=timeout) as r:
            d = json.loads(r.read())
        return str(
            d.get("model_path") or d.get("default_generation_settings", {}).get("model") or ""
        )
    except Exception as exc:
        return f"PROPS_ERROR {type(exc).__name__}: {exc}"


def completion_alive(port: int, timeout: float = 120.0) -> dict[str, Any]:
    """A real bounded /completion. A 200 from /health does not prove generation works."""
    body = json.dumps({"prompt": "2+2=", "n_predict": 8, "temperature": 0.0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return {
            "alive": True,
            "s": round(time.time() - t0, 2),
            "content": str(d.get("content"))[:80],
        }
    except Exception as exc:
        return {"alive": False, "s": round(time.time() - t0, 2), "error": f"{exc!r}"[:200]}


class ServerError(RuntimeError):
    """The owned server failed a launch gate. The run stops rather than measure an unknown."""


@dataclass
class OwnedServer:
    proc: Any
    pid: int
    port: int
    argv: list[str]
    meta: dict[str, Any]


def resolve_gguf(hf_cache: Path) -> Optional[Path]:
    """Exact repo folder and filename. Never fall back to some other cached .gguf."""
    root = Path(hf_cache) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
    hits = sorted(root.glob(f"*/{MODEL_FILENAME}")) if root.exists() else []
    return hits[-1] if hits else None


def terminate_owned_server(server: Optional[OwnedServer]) -> dict[str, Any]:
    """Stop our server by its PID. Never a name pattern, which could match other processes."""
    if server is None or server.proc is None:
        return {"terminated": False, "reason": "no owned server"}
    proc = server.proc
    try:
        proc.terminate()
        try:
            rc = proc.wait(timeout=45)
            method = "terminate"
        except Exception:
            proc.kill()
            rc = proc.wait(timeout=45)
            method = "kill"
        pid_file = (server.meta or {}).get("pid_file")
        if pid_file:
            Path(pid_file).unlink(missing_ok=True)
        return {"terminated": True, "pid": server.pid, "method": method, "returncode": rc}
    except Exception as exc:
        return {"terminated": False, "pid": server.pid, "error": f"{exc!r}"[:200]}


def launch_owned_server(
    gguf: Path,
    port: int,
    llama_server: Path,
    log_dir: Path,
    smi: SmiRunner = run_nvidia_smi,
    health_wait_s: float = SERVER_HEALTH_WAIT_S,
) -> OwnedServer:
    argv = [
        str(llama_server),
        "-m",
        str(gguf),
        *SERVER_FLAGS,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    # A UUID, not an index: CUDA device order can differ from nvidia-smi order.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU1_UUID)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama_server_{port}.log"
    pid_path = log_dir / f"llama_server_{port}.pid"
    progress(f"server launch: CUDA_VISIBLE_DEVICES={GPU1_UUID} {' '.join(argv)}")
    fh = log_path.open("ab")
    proc = subprocess.Popen(
        argv,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=_set_parent_death_signal,
    )
    # The PID file lets a person find and stop our server if this process is killed.
    pid_path.write_text(f"{proc.pid}\n")
    server = OwnedServer(
        proc, int(proc.pid), int(port), argv, {"log": str(log_path), "pid_file": str(pid_path)}
    )
    t0 = time.time()
    last_note = t0
    last_gpu_check = 0.0
    while True:
        if proc.poll() is not None:
            raise ServerError(f"llama-server exited early rc={proc.returncode}; log {log_path}")
        if time.time() - last_gpu_check >= 10:
            # Stop at once if the model starts to load on the conductor's card.
            last_gpu_check = time.time()
            if pid_residency_mib(server.pid, GPU0_UUID, smi):
                terminate_owned_server(server)
                raise ServerError("server holds memory on GPU 0 during load")
        if health_ok(port):
            break
        if time.time() - t0 > health_wait_s:
            terminate_owned_server(server)
            raise ServerError(f"llama-server not healthy within {health_wait_s:.0f}s")
        if time.time() - last_note >= 30:
            progress(f"server launch: waiting for health, {time.time() - t0:.0f}s")
            last_note = time.time()
        time.sleep(2)
    meta = {
        "pid": server.pid,
        "port": port,
        "argv": argv,
        "log": str(log_path),
        "pid_file": str(pid_path),
        "health_wait_s": round(time.time() - t0, 1),
        "residency_mib_gpu1": pid_residency_mib(server.pid, GPU1_UUID, smi),
        "residency_mib_gpu0": pid_residency_mib(server.pid, GPU0_UUID, smi),
        "props_model_path": props_model_path(port),
        "completion_probe": completion_alive(port),
        "gpu_uuid_proven": GPU1_UUID,
    }
    server.meta = meta
    problem = None
    if (meta["residency_mib_gpu1"] or 0) < MIN_RESIDENCY_MIB:
        problem = f"no real GPU 1 offload: {meta['residency_mib_gpu1']} MiB"
    elif meta["residency_mib_gpu0"]:
        problem = f"server holds {meta['residency_mib_gpu0']} MiB on GPU 0"
    elif Path(gguf).name not in meta["props_model_path"]:
        problem = f"/props names {meta['props_model_path']!r}"
    elif not meta["completion_probe"].get("alive"):
        problem = f"/health ok but /completion dead: {meta['completion_probe']}"
    if problem:
        terminate_owned_server(server)
        raise ServerError(problem)
    progress(
        f"server ready pid={server.pid} gpu1={meta['residency_mib_gpu1']} MiB "
        f"health {meta['health_wait_s']}s"
    )
    return server


# ---------------------------------------------------------------------------------------
# Generation (real run only): one live generate() call per window on the owned server.
# ---------------------------------------------------------------------------------------


def slots_n_decoded(port: int, timeout: float = 5.0) -> Optional[int]:
    """Best effort: tokens decoded so far on the busy slot. None when the server hides it."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    found: list[int] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "n_decoded" and isinstance(v, int):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return max(found) if found else None


@contextmanager
def heartbeat(
    label: str, interval_s: float, probe: Optional[Callable[[], Any]] = None
) -> Iterator[None]:
    """Print a line every `interval_s` while a model call is in flight."""
    stop = threading.Event()
    t0 = time.monotonic()

    def beat() -> None:
        while not stop.wait(interval_s):
            extra = ""
            if probe is not None:
                try:
                    extra = f" decoded={probe()}"
                except Exception:
                    extra = " decoded=unknown"
            progress(f"heartbeat {label} elapsed={time.monotonic() - t0:.0f}s{extra}")

    thread = threading.Thread(target=beat, name="exp10010-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()


@dataclass
class CallRecorder:
    """Per-call records for the window in progress. Reset per window, installed once."""

    game: str = ""
    heartbeat_s: float = HEARTBEAT_S
    raw_dir: Optional[Path] = None
    probe: Optional[Callable[[], Any]] = None
    run_id: str = "run"
    calls: list[dict[str, Any]] = field(default_factory=list)


def install_call_recorder(proposer: Any, recorder: CallRecorder) -> None:
    """Wrap the chat request so every model call is timed, counted, and announced.

    Think mode sends every induce call through `_chat_complete_request`, so this one
    seam sees each try and each defect re-ask. The wrapper changes no argument.
    """
    original = proposer._chat_complete_request

    def recorded(self: Any, prompt: str, **kwargs: Any) -> tuple[dict, str]:
        idx = len(recorder.calls)
        attempt = int(kwargs.get("attempt", 0))
        rec: dict[str, Any] = {
            "call_index": idx,
            "attempt": attempt,
            "seed": self.sampling_seed(attempt),
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
            "repeat_penalty": kwargs.get("repeat_penalty"),
            "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt),
        }
        progress(
            f"model call BEFORE game={recorder.game} call={idx} attempt={attempt} "
            f"seed={rec['seed']} max_tokens={rec['max_tokens']}"
        )
        t0 = time.monotonic()
        try:
            with heartbeat(
                f"game={recorder.game} call={idx}", recorder.heartbeat_s, recorder.probe
            ):
                normalized, extraction = original(prompt, **kwargs)
        except BaseException as exc:
            rec.update(
                wall_s=round(time.monotonic() - t0, 2),
                error=f"{exc!r}"[:300],
                stop_type="error",
                censored=True,
                censor_reason="wall_timeout" if _is_timeout(exc) else "transport_error",
            )
            recorder.calls.append(rec)
            progress(
                f"model call AFTER game={recorder.game} call={idx} "
                f"{rec['censor_reason']} wall={rec['wall_s']}s"
            )
            raise
        timings = normalized.get("timings") or {}
        stop_type = str(normalized.get("stop_type") or "")
        rec.update(
            wall_s=round(time.monotonic() - t0, 2),
            stop_type=stop_type,
            completion_tokens=timings.get("predicted_n"),
            prompt_tokens=timings.get("prompt_n"),
            final_chars=len(self.last_final_content or ""),
            reasoning_chars=len(self.last_reasoning_content or ""),
            extraction_chars=len(extraction or ""),
            truncated=bool(normalized.get("truncated")),
            censored=stop_type == "limit",
            censor_reason="token_limit" if stop_type == "limit" else None,
        )
        if recorder.raw_dir is not None:
            # One directory per run, so a restarted window never overwrites the
            # reasoning of an interrupted attempt.
            call_dir = recorder.raw_dir / recorder.game / recorder.run_id
            call_dir.mkdir(parents=True, exist_ok=True)
            raw_path = call_dir / f"call{idx:02d}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "record": rec,
                        "final_content": self.last_final_content,
                        "reasoning_content": self.last_reasoning_content,
                    },
                    indent=1,
                )
            )
            rec["raw_path"] = str(raw_path)
        recorder.calls.append(rec)
        progress(
            f"model call AFTER game={recorder.game} call={idx} stop={stop_type} "
            f"tokens={rec['completion_tokens']} wall={rec['wall_s']}s"
        )
        return normalized, extraction

    proposer._chat_complete_request = MethodType(recorded, proposer)


def pin_to_owned_server(proposer: Any, port: int) -> None:
    """Let `_ensure_server()` confirm our server, never launch one.

    The live version relaunches a dead server on whatever card it picks. Here that
    could land on GPU 0, which belongs to the conductor. A dead server fails the call.
    """

    def ensure(self: Any) -> bool:
        return int(self.port) == int(port) and bool(self._healthy()) and bool(self._reusable())

    proposer._ensure_server = MethodType(ensure, proposer)


def window_seed(spec: WindowSpec) -> int:
    """Per-window seed base. The proposer sends base * 1000 + attempt to the sampler."""
    return SEED_BASE * 100 + spec.index


def vllm_shape_extraction(engine_source: str, reasoning: str, final: str) -> dict[str, Any]:
    """The engine the scored vLLM path would extract from this same completion.

    vLLM runs with no reasoning parser, so its `content` holds the reasoning, then
    `</think>`, then the answer, and `_extract_python` takes the FIRST python block. A
    draft written while thinking would then win over the final answer.
    """
    text = f"{reasoning}\n</think>\n\n{final}" if reasoning else final
    code = e3._extract_python(text)
    differs = code.strip() != str(engine_source).strip()
    return {
        "differs": bool(differs),
        "reasoning_has_python_fence": "```python" in reasoning,
        "engine_source": code if differs else None,
        "engine_sha256": sha256_text(code),
    }


def generate_window(
    proposer: Any, captured: Mapping[str, Any], spec: WindowSpec, recorder: CallRecorder
) -> dict[str, Any]:
    """One live generate() call with the captured prompt and arguments."""
    if not e3.induce_think_on():
        raise RuntimeError("think mode resolved OFF at generation time")
    seed = window_seed(spec)
    previous = os.environ.get("CARNOT_ARC_GENERATOR_SEED")
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    recorder.game = spec.game
    recorder.calls = []
    before = (
        int(proposer.n_induce_defect_reasks),
        int(proposer.n_goal_defect_reasks),
        int(proposer.n_content_failures),
        int(proposer.n_server_failures),
    )
    t0 = time.monotonic()
    try:
        ok, out = proposer.generate(
            captured["prompt"],
            tuple(captured["required"]),
            validate=captured.get("validate"),
            tries=int(captured["tries"]),
            **dict(captured["kwargs"]),
        )
    finally:
        if previous is None:
            os.environ.pop("CARNOT_ARC_GENERATOR_SEED", None)
        else:
            os.environ["CARNOT_ARC_GENERATOR_SEED"] = previous
    calls = list(recorder.calls)
    return {
        "ok": bool(ok),
        "engine_source": str(out) if ok else None,
        "engine_sha256": sha256_text(str(out)) if ok else None,
        "vllm_shape_extraction": (
            vllm_shape_extraction(
                str(out),
                str(getattr(proposer, "last_reasoning_content", "") or ""),
                str(getattr(proposer, "last_final_content", "") or ""),
            )
            if ok
            else None
        ),
        "run_id": recorder.run_id,
        "failure_note": None if ok else str(out)[:400],
        "window_seed_base": seed,
        "effective_seeds": [c.get("seed") for c in calls],
        "wall_s": round(time.monotonic() - t0, 2),
        "n_calls": len(calls),
        "calls": calls,
        "requested_n_predict": int(getattr(proposer, "last_requested_n_predict", -1)),
        "defect_reasks": int(proposer.n_induce_defect_reasks) - before[0],
        "goal_defect_reasks": int(proposer.n_goal_defect_reasks) - before[1],
        "content_failures": int(proposer.n_content_failures) - before[2],
        "server_failures": int(proposer.n_server_failures) - before[3],
        "censored": any(bool(c.get("censored")) for c in calls),
    }


# ---------------------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    repo_root: Path
    output_dir: Path
    artifact_path: Path
    dry_run: bool = False
    port: int = DEFAULT_PORT
    llama_server: Path = DEFAULT_LLAMA_SERVER
    hf_cache: Path = DEFAULT_HF_CACHE
    heartbeat_s: float = HEARTBEAT_S
    windows: tuple[str, ...] = PILOT_WINDOWS


def process_comm() -> str:
    try:
        return Path("/proc/self/comm").read_text().strip()
    except OSError:
        return ""


# This host's orphan janitor kills python3 and pytest processes older than 2 h. The
# pilot runs for hours, so it must start under a name the janitor leaves alone.
JANITOR_REAPED_COMMS: tuple[str, ...] = ("python3", "pytest")


@dataclass
class RunHooks:
    """Seams for tests. The defaults are the real machinery."""

    smi: SmiRunner = run_nvidia_smi
    port_in_use: Callable[[int], bool] = port_bound
    launch_server: Callable[..., OwnedServer] = launch_owned_server
    terminate_server: Callable[[Optional[OwnedServer]], dict[str, Any]] = terminate_owned_server
    make_proposer: Callable[..., Any] = _live_proposer
    props_model_path: Callable[[int], str] = props_model_path
    completion_alive: Callable[[int], dict[str, Any]] = completion_alive
    slots_probe: Callable[[int], Optional[int]] = slots_n_decoded
    process_comm: Callable[[], str] = process_comm


def evidence_files(paths: EvidencePaths, windows: Sequence[str]) -> list[Path]:
    files = [paths.workflow_result(), paths.pilot_baseline()]
    for game in windows:
        files.append(paths.expert_engine(game))
        for seed in RECORDED_SEEDS:
            files.append(paths.request(game, seed))
            files.append(paths.response(game, seed))
    return files


def check_preconditions(
    cfg: RunConfig, hooks: RunHooks, paths: EvidencePaths, needs_generation: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any], Optional[Path]]:
    """Step 0. Every check runs; the first failure names the blocked verdict.

    The GPU, port, and process-name checks apply only when a window still needs a
    model call. A rerun that only rebuilds the artifact from the shard needs no GPU.
    """
    checks: list[dict[str, Any]] = []

    def add(resource: str, ok: Optional[bool], detail: str, blocked: Optional[str]) -> None:
        checks.append(
            {"resource": resource, "available": ok, "detail": detail, "blocked_verdict": blocked}
        )

    gguf = resolve_gguf(cfg.hf_cache)
    add(
        "gguf_cached_qwen38_27b_q4_k_m",
        gguf is not None,
        str(gguf) if gguf else f"{MODEL_FILENAME} not under {cfg.hf_cache}",
        "blocked_gguf_not_cached_qwen38_27b",
    )
    binary_ok = cfg.llama_server.exists() and os.access(cfg.llama_server, os.X_OK)
    add("llama_server_binary", binary_ok, str(cfg.llama_server), "blocked_llama_server_missing")
    gpu_checks = (
        "gpu1_present_by_uuid",
        "gpu1_under_500mib_used",
        "gpu1_no_other_process",
        "server_port_free",
        "process_name_not_reaped_by_janitor",
    )
    if cfg.dry_run or not needs_generation:
        why = (
            "skipped: --dry-run touches no GPU and starts no server"
            if cfg.dry_run
            else "skipped: every window is already in the shard; no model call is needed"
        )
        for name in gpu_checks:
            add(name, None, why, None)
    else:
        inv = gpu_inventory(hooks.smi)
        g1 = inv.get(GPU1_UUID)
        add("gpu1_present_by_uuid", g1 is not None, f"{GPU1_UUID}: {g1}", "blocked_gpu1_absent")
        used = None if g1 is None else int(g1["memory_used_mib"])
        add(
            "gpu1_under_500mib_used",
            used is not None and used < GPU1_MAX_USED_MIB,
            f"memory.used={used} MiB",
            "blocked_gpu1_memory_in_use",
        )
        apps = [a for a in compute_apps(hooks.smi) if a["gpu_uuid"] == GPU1_UUID]
        add(
            "gpu1_no_other_process",
            g1 is not None and not apps,
            f"processes on GPU 1: {apps}",
            "blocked_gpu1_other_process",
        )
        add(
            "server_port_free",
            not hooks.port_in_use(cfg.port),
            f"port {cfg.port}",
            "blocked_server_port_in_use",
        )
        comm = hooks.process_comm()
        add(
            "process_name_not_reaped_by_janitor",
            comm not in JANITOR_REAPED_COMMS,
            f"/proc/self/comm={comm!r}; start with .venv/bin/python, not python3 or pytest",
            "blocked_process_name_reaped_by_janitor",
        )
    missing = [p for p in evidence_files(paths, cfg.windows) if not p.exists()]
    add(
        "positive_control_evidence",
        not missing,
        f"{len(missing)} missing" + (f": {[str(m) for m in missing[:4]]}" if missing else ""),
        "blocked_positive_control_evidence_missing",
    )
    parity = induction_env_parity()
    add(
        "induction_env_matches_kernel_llamacpp_shape",
        parity["matches_kernel_llamacpp_shape"],
        json.dumps(parity["differences"], default=str)[:300],
        "blocked_induction_env_not_kernel_llamacpp_shape",
    )
    unlisted = induction_env_unlisted_flags()
    parity["unlisted_flags"] = unlisted
    add(
        "induction_env_no_unlisted_flags",
        unlisted["ok"],
        json.dumps({"unlisted": unlisted["unlisted"], "wrong": unlisted["wrong_value"]})[:300],
        "blocked_induction_env_unlisted_flag",
    )
    return checks, parity, gguf


# Row statuses that are a finished model outcome. Only these are kept in the shard.
FINISHED_STATUSES = ("generated", "generation_failed", "dry_run_stand_in")


def read_shard(path: Path) -> dict[str, dict[str, Any]]:
    """Finished windows only. A prompt-fidelity stop is re-checked on every run."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("game") and row.get("status") in FINISHED_STATUSES:
            done[str(row["game"])] = row
    return done


def append_shard(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def window_guard(
    server: OwnedServer, hooks: RunHooks, gguf: Path, port: int
) -> Optional[dict[str, Any]]:
    """Before each window: card present, alone, model resident, identity, and generation.

    After a client-side timeout, llama-server cancels the abandoned request within about
    a second (it polls the connection), so the /completion probe is not held up by it.
    """
    if GPU1_UUID not in gpu_inventory(hooks.smi):
        return {"kind": "gpu1_lost"}
    others = [
        a
        for a in compute_apps(hooks.smi)
        if a["gpu_uuid"] == GPU1_UUID and int(a["pid"]) != int(server.pid)
    ]
    if others:
        # A second process on the card slows decoding and changes which calls time out.
        return {"kind": "gpu1_foreign_process", "processes": others}
    res = pid_residency_mib(server.pid, GPU1_UUID, hooks.smi)
    if (res or 0) < MIN_RESIDENCY_MIB:
        return {"kind": "residency_collapsed", "residency_mib": res}
    props = hooks.props_model_path(port)
    if Path(gguf).name not in props:
        return {"kind": "props_identity_lost", "props": props[:200]}
    probe = hooks.completion_alive(port)
    if not probe.get("alive"):
        return {"kind": "completion_dead", "probe": probe}
    return None


def _stand_in_row(spec: WindowSpec, paths: EvidencePaths, prompt_sha: str) -> dict[str, Any]:
    """--dry-run: the recorded codeonly first shot stands in for the model's output."""
    code, meta = recorded_first_shot_source(paths, spec.game, FIRST_CALL_SEED)
    return {
        "game": spec.game,
        "index": spec.index,
        "status": "dry_run_stand_in",
        "window_sha256": spec.window_sha256,
        "built_prompt_sha256": prompt_sha,
        "stand_in": {
            "source": f"recorded codeonly first shot, seed {FIRST_CALL_SEED}",
            "engine_source": code,
            **meta,
        },
        "generation": None,
    }


def _row_engine_source(row: Mapping[str, Any]) -> Optional[str]:
    gen = row.get("generation") or {}
    if gen:
        return gen.get("engine_source")
    return (row.get("stand_in") or {}).get("engine_source")


def score_window_row(row: dict[str, Any], spec: WindowSpec) -> dict[str, Any]:
    """Score (or re-score) one finished window from its stored engine source.

    Resumed rows go through here too, so every row in one artifact is scored by the
    same code. The stored module hash says which code generated the row.
    """
    label = "stand_in" if row.get("status") == "dry_run_stand_in" else "think_on"
    score = score_engine(_row_engine_source(row), spec, label)
    primary = score["primary_change_fidelity"]
    row["score"] = _compact(score)
    row["per_row"] = score["per_row"]
    row["live_ladder"] = live_ladder(row.get("generation"), 0.0 if primary is None else primary)
    vshape = (row.get("generation") or {}).get("vllm_shape_extraction") or {}
    if vshape.get("differs"):
        alt = score_engine(vshape.get("engine_source"), spec, "vllm_shape")
        row["vllm_shape_score"] = _compact(alt)
    elif vshape:
        row["vllm_shape_score"] = {"same_engine_as_run": True}
    row["scored_by_module_sha256"] = sha256_file(Path(__file__))
    # The engine child runs the engine, so a child-only fix must show here too.
    row["scored_by_engine_child_sha256"] = engine_child_sha256()
    return row


def live_ladder(gen: Optional[Mapping[str, Any]], primary: float) -> dict[str, Any]:
    """Would this window survive the scored path's 2,400 s timeout? One entry per rate.

    Locally a long draw can hit the token cap and be retried; on Kaggle the timeout
    fires first and the window fails with no retry. Rule A: a local cap hit fails the
    window. Rule B: a draw longer than 2,400 s times the rate fails it. The ladder can
    only lower a score. It cannot raise a window that timed out here.
    """
    calls = list((gen or {}).get("calls") or [])
    out: dict[str, Any] = {"timeout_s": LIVE_TIMEOUT_S, "rates": {}}
    for name, rate in SCORED_DECODE_RATES_TOK_S.items():
        ceiling = LIVE_TIMEOUT_S * rate
        failed_at: Optional[int] = None
        reason: Optional[str] = None
        for c in calls:
            tokens = c.get("completion_tokens")
            if c.get("censor_reason") == "wall_timeout":
                failed_at, reason = c.get("call_index"), "timed out here as well"
            elif c.get("stop_type") == "limit":
                failed_at, reason = c.get("call_index"), "rule A: hit the local token cap"
            elif isinstance(tokens, (int, float)) and tokens > ceiling:
                failed_at, reason = c.get("call_index"), f"rule B: {tokens} > {ceiling:.0f}"
            if failed_at is not None:
                break
        out["rates"][name] = {
            "tok_s": rate,
            "ceiling_tokens": round(ceiling),
            "survives": failed_at is None,
            "failed_at_call": failed_at,
            "reason": reason,
            "primary": primary if failed_at is None else 0.0,
        }
    return out


def _infra_failure(gen: Mapping[str, Any]) -> Optional[str]:
    """A server or transport failure is not a model outcome, so it is never scored.

    Live generate() returns (False, msg) on any request error with no retry. Kept as a
    finished row, a crashed server would count as a model failure forever, because
    resume skips finished rows.
    """
    for c in gen.get("calls") or []:
        if c.get("censor_reason") == "transport_error":
            return f"transport error on call {c.get('call_index')}: {c.get('error')}"[:300]
    if not gen.get("ok") and not gen.get("calls"):
        return f"no model call was made: {gen.get('failure_note')}"[:300]
    return None


def _new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_pid{os.getpid()}"


def run_pilot(cfg: RunConfig, hooks: Optional[RunHooks] = None) -> dict[str, Any]:
    """Run the pilot (or its dry run) and write the artifact. Returns the artifact."""
    hooks = hooks or RunHooks()
    t0 = time.monotonic()
    paths = EvidencePaths.under(cfg.repo_root)
    run_id = _new_run_id()
    st: dict[str, Any] = {
        "rows": {},
        "run_id": run_id,
        "server_launched": False,
        "server": None,
        "teardown": None,
    }
    mode = "dry_run" if cfg.dry_run else "real"
    progress(f"start mode={mode} run_id={run_id} windows={len(cfg.windows)} out={cfg.output_dir}")

    # The shard is read first, so a rerun that only rebuilds the artifact needs no GPU.
    shard = cfg.output_dir / ("shard_dry_run.jsonl" if cfg.dry_run else "shard.jsonl")
    done = {g: r for g, r in read_shard(shard).items() if g in cfg.windows}
    needs_generation = any(g not in done for g in cfg.windows)

    progress("phase preconditions: begin")
    checks, parity, gguf = check_preconditions(cfg, hooks, paths, needs_generation)
    st.update(preconditions=checks, induction_env=parity, gguf=gguf)
    failed = [c for c in checks if c["available"] is False]
    progress(f"phase preconditions: end failed={[c['resource'] for c in failed]}")
    if failed:
        return finish(cfg, st, t0, failed[0]["blocked_verdict"])

    progress("phase windows: begin")
    specs: list[WindowSpec] = []
    try:
        reports = load_control_reports(paths)
        for i, game in enumerate(cfg.windows):
            if game not in reports:
                raise WindowError(f"no positive-control report for {game}")
            spec = load_window(game, i, reports[game], paths)
            specs.append(spec)
            progress(f"window {game}: rows={len(spec.rows)} sha256={spec.window_sha256[:16]}")
    except (WindowError, OSError, ValueError) as exc:
        st["window_error"] = f"{exc}"[:400]
        return finish(cfg, st, t0, "blocked_window_load_failed")
    st["specs"] = specs
    by_game = {s.game: s for s in specs}
    progress("phase windows: end")

    # A shard row generated on a different window file is not this window's result.
    for game, row in done.items():
        if row.get("window_sha256") != by_game[game].window_sha256:
            st["shard_mismatch"] = {
                "game": game,
                "shard_window_sha256": row.get("window_sha256"),
                "current_window_sha256": by_game[game].window_sha256,
            }
            return finish(cfg, st, t0, f"blocked_shard_window_mismatch_{game}")

    progress("phase prompt fidelity: begin")
    fidelity: dict[str, Any] = {}
    captured: dict[str, dict[str, Any]] = {}
    for spec in specs:
        f = prompt_fidelity(spec, paths)
        cap = f.pop("_captured", None)
        if cap is not None:
            f["heldout_leak_check"] = heldout_leak_check(cap["prompt"], spec.rows, spec.n_prefix)
        else:
            f["heldout_leak_check"] = {"passed": False, "reason": "no prompt was built"}
        f["usable"] = bool(f["match"] and f["heldout_leak_check"]["passed"])
        if not f["usable"] and not f.get("reason"):
            f["reason"] = "held-out leak check failed"
        fidelity[spec.game] = f
        if f["usable"] and cap is not None:
            captured[spec.game] = cap
        progress(f"fidelity {spec.game}: match={f['match']} usable={f['usable']}")
    st["fidelity"] = fidelity
    progress(f"phase prompt fidelity: end usable={len(captured)}/{len(specs)}")

    controls = run_controls(specs, paths)
    st["controls"] = controls
    if not controls["passed"]:
        return finish(cfg, st, t0, f"blocked_control_failed_{controls['failures'][0]}")
    if not captured:
        return finish(cfg, st, t0, "blocked_prompt_fidelity_failed_all_windows")

    rows: dict[str, dict[str, Any]] = {}
    st["rows"] = rows
    st["resumed_windows"] = sorted(done)
    progress(f"phase rescore resumed: begin n={len(done)}")
    for game, row in done.items():
        rows[game] = score_window_row(dict(row), by_game[game])
        rows[game]["resumed"] = True
        progress(f"resumed {game}: primary={rows[game]['score']['primary_change_fidelity']}")
    progress("phase rescore resumed: end")
    pending = [s for s in specs if s.game not in rows]
    progress(f"phase generation: begin mode={mode} finished={len(rows)} pending={len(pending)}")
    server: Optional[OwnedServer] = None
    try:
        proposer = None
        recorder = CallRecorder(
            heartbeat_s=cfg.heartbeat_s, raw_dir=cfg.output_dir / "calls", run_id=run_id
        )
        if not cfg.dry_run and any(s.game in captured for s in pending):
            if gguf is None:
                raise ServerError("no GGUF path")
            server = hooks.launch_server(
                gguf, cfg.port, cfg.llama_server, cfg.output_dir / "server_logs", hooks.smi
            )
            st["server_launched"] = True
            st["server"] = server.meta
            proposer = hooks.make_proposer(port=cfg.port, model_path=str(gguf))
            pin_to_owned_server(proposer, cfg.port)
            if not proposer._ensure_server():
                raise ServerError("the proposer could not confirm the owned server")
            port = cfg.port
            recorder.probe = lambda: hooks.slots_probe(port)
            install_call_recorder(proposer, recorder)
        for spec in pending:
            f = fidelity[spec.game]
            if spec.game not in captured:
                # Kept in the artifact, never in the shard: a code fix must re-check it.
                rows[spec.game] = {
                    "game": spec.game,
                    "index": spec.index,
                    "status": "stopped_prompt_fidelity",
                    "reason": f.get("reason"),
                    "window_sha256": spec.window_sha256,
                }
                progress(f"window {spec.game}: stopped by prompt fidelity")
                continue
            if cfg.dry_run:
                row = _stand_in_row(spec, paths, f["built_prompt_sha256"])
            else:
                assert server is not None and gguf is not None and proposer is not None
                wedge = window_guard(server, hooks, gguf, cfg.port)
                if wedge is not None:
                    st["wedge"] = {**wedge, "game": spec.game}
                    progress(f"WEDGE before {spec.game}: {wedge}")
                    break
                progress(f"window {spec.game}: generation begin seed_base={window_seed(spec)}")
                gen = generate_window(proposer, captured[spec.game], spec, recorder)
                infra = _infra_failure(gen)
                timed_out = any(c.get("censor_reason") == "wall_timeout" for c in gen["calls"])
                if infra is None and timed_out:
                    # A timeout is a model outcome only if the server is still healthy.
                    after = window_guard(server, hooks, gguf, cfg.port)
                    if after is not None:
                        infra = f"server unhealthy after a timeout: {after}"
                if infra is not None:
                    st["wedge"] = {"kind": "infra_failure", "game": spec.game, "error": infra}
                    st.setdefault("infra_failed_generations", []).append(
                        {"game": spec.game, "generation": gen}
                    )
                    progress(f"INFRA FAILURE {spec.game}: {infra}")
                    break
                row = {
                    "game": spec.game,
                    "index": spec.index,
                    "status": "generated" if gen["ok"] else "generation_failed",
                    "window_sha256": spec.window_sha256,
                    "built_prompt_sha256": f["built_prompt_sha256"],
                    "generation": gen,
                }
            row["run_id"] = run_id
            row["generated_by_module_sha256"] = sha256_file(Path(__file__))
            score_window_row(row, spec)
            append_shard(shard, row)
            rows[spec.game] = row
            primary = row["score"].get("primary_change_fidelity")
            progress(f"window {spec.game}: done status={row['status']} primary={primary}")
    except ServerError as exc:
        st["wedge"] = {"kind": "server_error", "error": f"{exc}"[:400]}
        progress(f"server error: {exc}")
    except Exception as exc:
        st["wedge"] = {"kind": "exception", "error": f"{type(exc).__name__}: {exc}"[:400]}
        progress(f"exception: {type(exc).__name__}: {exc}")
    finally:
        if server is not None:
            st["teardown"] = hooks.terminate_server(server)
            progress(f"server teardown: {st['teardown']}")
    n_finished = sum(1 for r in rows.values() if r.get("status") in FINISHED_STATUSES)
    progress(f"phase generation: end finished={n_finished}/{len(cfg.windows)}")
    if st.get("wedge") or len(rows) < len(cfg.windows):
        kind = (st.get("wedge") or {}).get("kind", "incomplete")
        verdict = f"partial_think_on_pilot_{n_finished}_of_{len(cfg.windows)}_windows_{kind}"
        return finish(cfg, st, t0, verdict)
    stopped = sorted(g for g, r in rows.items() if r.get("status") == "stopped_prompt_fidelity")
    if stopped:
        # A pilot mean over a hand-picked subset is not the pre-registered pilot.
        return finish(cfg, st, t0, f"blocked_prompt_fidelity_failed_{len(stopped)}_windows")
    return finish(cfg, st, t0, None)


# ---------------------------------------------------------------------------------------
# Artifact.
# ---------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def compare_to_baseline(rows: Mapping[str, Any], controls: Mapping[str, Any]) -> dict[str, Any]:
    """Pilot against the codeonly baseline, always over the SAME set of windows.

    A window with no score (stopped by prompt fidelity) must not drop out of the pilot
    mean while staying in the baseline mean. So there are two views: matched (both
    means over the scored windows) and all-windows (a stopped window counts as 0).
    """
    per_window: dict[str, Any] = {}
    pilot_vals: list[float] = []
    matched_base: list[float] = []
    all_windows_pilot: list[float] = []
    halluc: list[float] = []
    exact: list[float] = []
    live_passes: list[str] = []
    ladder: dict[str, list[float]] = {name: [] for name in SCORED_DECODE_RATES_TOK_S}
    vllm_shape: list[float] = []
    vllm_differs: list[str] = []
    base_windows = (controls.get("per_window") or {}) if controls else {}
    for game, row in rows.items():
        base = base_windows.get(game, {}).get("codeonly_window_mean")
        score = row.get("score")
        if not score:
            all_windows_pilot.append(0.0)
            per_window[game] = {"pilot_primary": None, "codeonly_baseline": base, "delta": None}
            continue
        p = score.get("primary_change_fidelity")
        p = 0.0 if p is None else float(p)
        per_window[game] = {
            "pilot_primary": p,
            "codeonly_baseline": base,
            "delta": None if base is None else p - float(base),
        }
        pilot_vals.append(p)
        all_windows_pilot.append(p)
        if base is not None:
            matched_base.append(float(base))
        if score.get("noop_hallucination_rate") is not None:
            halluc.append(float(score["noop_hallucination_rate"]))
        if score.get("masked_exact_accuracy") is not None:
            exact.append(float(score["masked_exact_accuracy"]))
        if score.get("live_unmasked_pass_1p0") and game in LIVE_GATE_MEANINGFUL:
            live_passes.append(game)
        for name, entry in ((row.get("live_ladder") or {}).get("rates") or {}).items():
            if name in ladder:
                ladder[name].append(float(entry.get("primary") or 0.0))
        alt = row.get("vllm_shape_score") or {}
        if alt.get("same_engine_as_run"):
            vllm_shape.append(p)
        elif "primary_change_fidelity" in alt:
            vllm_shape.append(float(alt.get("primary_change_fidelity") or 0.0))
            vllm_differs.append(game)
    deltas = [v["delta"] for v in per_window.values() if v["delta"] is not None]
    reproduced = controls.get("codeonly_baseline_mean_reproduced") if controls else None
    pilot_mean = _mean(pilot_vals)
    matched_mean = _mean(matched_base) if len(matched_base) == len(pilot_vals) else None
    return {
        "preregistered_baseline_mean": PREREGISTERED_BASELINE_MEAN,
        "baseline_mean_reproduced": reproduced,
        "pilot_mean_primary_change_fidelity": pilot_mean,
        "baseline_mean_over_scored_windows": matched_mean,
        "delta_vs_matched_baseline": (
            None if pilot_mean is None or matched_mean is None else pilot_mean - matched_mean
        ),
        "pilot_mean_all_windows_stopped_as_zero": _mean(all_windows_pilot),
        "delta_all_windows_vs_reproduced_baseline": (
            None
            if not all_windows_pilot or reproduced is None
            else float(np.mean(all_windows_pilot)) - float(reproduced)
        ),
        "delta_vs_preregistered_baseline": (
            None if pilot_mean is None else pilot_mean - PREREGISTERED_BASELINE_MEAN
        ),
        "n_windows_scored": len(pilot_vals),
        "n_windows_in_run": len(rows),
        "n_windows_above_baseline": sum(1 for d in deltas if d > 1e-12),
        "n_windows_below_baseline": sum(1 for d in deltas if d < -1e-12),
        "n_windows_equal_baseline": sum(1 for d in deltas if abs(d) <= 1e-12),
        "guard_mean_noop_hallucination_rate": _mean(halluc),
        "secondary_mean_masked_exact_accuracy": _mean(exact),
        "live_gate_1p0_passes": live_passes,
        "live_gate_meaningful_windows": list(LIVE_GATE_MEANINGFUL),
        "live_ladder_mean_primary": {name: _mean(v) for name, v in ladder.items()},
        "live_ladder_note": (
            "The same windows rescored as the scored path's 2,400 s timeout would end them. "
            "A draw that hit the local cap, or ran past 2,400 s times the rate, scores 0. "
            "Report this next to the as-run mean; it can only lower a window."
        ),
        "vllm_shape_mean_primary": _mean(vllm_shape),
        "vllm_shape_windows_with_a_different_engine": sorted(vllm_differs),
        "per_window": per_window,
        "note": "Descriptive only. 10 windows x 1 draw supports no significance claim.",
    }


def false_negative_risk(rows: Mapping[str, Any], controls: Mapping[str, Any]) -> dict[str, Any]:
    censored: list[dict[str, Any]] = []
    censored_windows: list[str] = []
    failed: list[str] = []
    stopped: list[str] = []
    for game, row in rows.items():
        gen = row.get("generation") or {}
        for call in gen.get("calls") or []:
            if call.get("censored"):
                censored.append(
                    {
                        "game": game,
                        "call_index": call.get("call_index"),
                        "censor_reason": call.get("censor_reason"),
                        "wall_s": call.get("wall_s"),
                        "completion_tokens": call.get("completion_tokens"),
                    }
                )
        if gen.get("censored"):
            censored_windows.append(game)
        if row.get("status") == "generation_failed":
            failed.append(game)
        if row.get("status") == "stopped_prompt_fidelity":
            stopped.append(game)
    few = {
        g: v["expert"]["n_changing_rows"]
        for g, v in (controls.get("per_window") or {}).items()
        if v["expert"]["n_changing_rows"] < FEW_CHANGING_ROWS
    }
    return {
        "censored_calls": censored,
        "windows_with_censored_generation": sorted(censored_windows),
        "windows_generation_failed": sorted(failed),
        "windows_stopped_prompt_fidelity": sorted(stopped),
        "windows_with_few_changing_rows": few,
        "explanation": (
            "A censored call cannot tell 'the model cannot induce this window' from 'the "
            "budget ran out'. On a window with few changing held-out rows, one row moves "
            "the primary metric by 0.25 or more. Read those windows as coarse."
        ),
    }


def _checksum(st: Mapping[str, Any]) -> str:
    specs = st.get("specs") or []
    controls = (st.get("controls") or {}).get("per_window") or {}
    fidelity = st.get("fidelity") or {}
    payload = {
        "module_sha256": sha256_file(Path(__file__)),
        "engine_child_sha256": sha256_file(ENGINE_CHILD_PATH),
        "windows": {s.game: s.window_sha256 for s in specs},
        "recorded_prompts": {g: f.get("recorded_prompt_sha256") for g, f in fidelity.items()},
        "built_prompts": {g: f.get("built_prompt_sha256") for g, f in fidelity.items()},
        "experts": {g: v.get("expert_engine_sha256") for g, v in controls.items()},
        "recorded_responses": {
            g: [s.get("response_sha256") for s in v.get("codeonly_first_shots", [])]
            for g, v in controls.items()
        },
        "config": {
            "server_flags": SERVER_FLAGS,
            "tries": LIVE_TRIES,
            "max_tokens": MAX_TOKENS,
            "call_timeout_s": CALL_TIMEOUT_S,
            "seed_base": SEED_BASE,
            "masks": MASK_GRID_ROWS,
            "exclusions": EXCLUDED_HELDOUT_ROWS,
        },
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


DEVIATIONS_FROM_LIVE_PATH = (
    "generate() is called directly with the captured prompt and arguments. If the combined "
    "call fails, live induce() would then try an engine-only and a goal-only call. That "
    "fallback is not replayed; the pilot is one round, as pre-registered.",
    "Round 1 only. The pilot measures the round-1 stall-path engine on unseen rows. Live "
    "continues with refactor rounds and a default-branch induce that see the held-out rows "
    "while the CEGIS split is off (REQ-ARC-WMTE-6090), plus the kernel's tool-loop repair "
    "resample (CARNOT_ARC_INDUCE_TOOL_LOOP=repair). None of that is replayed, so the pilot "
    "is not an estimate of the engine the live agent finally accepts.",
    "_ensure_server() is pinned to the owned server on GPU 1. The live path would relaunch a "
    "dead server; here a dead server fails the call instead of risking GPU 0.",
    "Sampling seeds are pinned per window through CARNOT_ARC_GENERATOR_SEED. The live path "
    "sends no seed.",
    "Backend and weights. The scored kernel attaches the NVFP4 safetensors dataset, so it "
    "runs vLLM with NVFP4 weights, an fp8 KV cache, and --max-num-seqs 8; its fallback is "
    "llama.cpp with the NVFP4 GGUF. This card (sm_86) cannot run NVFP4. The pilot runs "
    "llama.cpp with the Q4_K_M GGUF, a q8_0 KV cache, and one stream. Quantisation, KV "
    "precision, server, and sampler all differ. The environment check compares only the "
    "llama.cpp-shaped induce settings (backend_parity: false).",
    "Sampler. Every think-ON call goes through _chat_complete_request, which sends "
    "repeat_penalty 1.1 and repeat_last_n 256 under their llama.cpp names. llama-server "
    "applies them. vLLM reads only repetition_penalty and accepts unknown fields, so it drops "
    "both silently and uses 1.0 (vLLM 0.29.0, chat_completion/protocol.py). min_p is 0.05 "
    "here (llama.cpp default) and 0.0 on vLLM. Repetition loops were the dominant induce "
    "failure, so the pilot's censoring and yield may be better than the scored agent's.",
    "Answer extraction. llama-server splits the reasoning into reasoning_content, so the "
    "engine comes from the answer alone. The scored vLLM server has no reasoning parser, so "
    "its content holds the reasoning too, and _extract_python takes the first python block, "
    "which can be a draft. vllm_shape_* fields score the engine that path would extract "
    "from the same accepted completion. They do not model a different retry being accepted.",
    "Token budget. The local pool is 131072 tokens, so generate() clamps n_predict to 108,720 "
    "(pool minus the worst-case prompt reserve). The scored pool does not clamp. At 40.8 tok/s "
    "the 2,400 s timeout binds at about 98k tokens, before the clamp. Each window records "
    "its requested_n_predict.",
    "Timeout. The per-call timeout is 2,400 s, as pre-registered. CORRECTION 2026-09-23: the "
    "first version of this harness used 4,800 s and called it pre-registered. It was not. The "
    "note says 2,400 s, and the stated reason (this card decodes about half as fast as the "
    "Kaggle card) was wrong: 40.8 tok/s here against 40.0 at the scored k=8.",
    "Decode rate. The local run is closest to the scored vLLM path at k=8 (40.0 tok/s). At "
    "llama.cpp k=1 (52.2 tok/s) the live ceiling is about 125k tokens, so a draw that timed "
    "out here might complete there. live_ladder can only lower a window's score, never raise it.",
)


def _substrate(st: Mapping[str, Any], rows: Mapping[str, Any], verdict: str) -> dict[str, Any]:
    """Declare the compute from what the rows and this process actually show.

    A rebuilt or resumed artifact still reports model generation when its rows hold
    model calls, even if this process made none.
    """
    calls = sum(len((r.get("generation") or {}).get("calls") or []) for r in rows.values())
    gen_wall = sum(float((r.get("generation") or {}).get("wall_s") or 0.0) for r in rows.values())
    infra_calls = sum(
        len((f.get("generation") or {}).get("calls") or [])
        for f in st.get("infra_failed_generations") or []
    )
    # A model call that reached the server counts, even when its window was not kept.
    generated = calls > 0 or infra_calls > 0
    loaded = bool(st.get("server_launched"))
    if verdict.startswith("blocked"):
        cls = "blocked_no_run"
    elif generated:
        cls = "model_full_generation"
    elif loaded:
        cls = "model_load_no_generation"
    else:
        cls = "no_model_load"
    if generated:
        name = "live_llm_inference"
    elif loaded:
        name = "live_llm_server_loaded_liveness_probe_only"
    else:
        name = "verifier_ensemble_against_cached_candidates"
    return {
        "class": cls,
        "name": name,
        "n_model_calls_in_rows": calls,
        "n_model_calls_in_unkept_windows": infra_calls,
        "generation_wall_s_in_rows": round(gen_wall, 2),
        "invoked": generated or loaded,
    }


def _write_artifact(cfg: RunConfig, artifact: dict[str, Any], verdict: str) -> Path:
    """Write the artifact. A blocked rerun never replaces an artifact that has rows."""
    target = cfg.artifact_path
    if verdict.startswith("blocked") and target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, ValueError):
            existing = {}
        if existing.get("per_window_rows"):
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            target = cfg.output_dir / "blocked_attempts" / f"{stamp}_{verdict[:80]}.json"
            artifact["kept_existing_artifact"] = str(cfg.artifact_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact["written_to"] = str(target)
    target.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    return target


def finish(
    cfg: RunConfig, st: Mapping[str, Any], t0: float, verdict: Optional[str]
) -> dict[str, Any]:
    """Assemble and write the artifact. `verdict=None` means every window finished."""
    rows = dict(st.get("rows") or {})
    controls = dict(st.get("controls") or {})
    specs: list[WindowSpec] = list(st.get("specs") or [])
    comparison = compare_to_baseline(rows, controls) if controls else None
    if verdict is None:
        mean = (comparison or {}).get("pilot_mean_primary_change_fidelity")
        n_scored = (comparison or {}).get("n_windows_scored", 0)
        mean_txt = "none" if mean is None else f"{mean:.3f}"
        if cfg.dry_run:
            verdict = (
                f"complete_dry_run_harness_verified_{n_scored}_windows_stand_in_mean_{mean_txt}"
            )
        else:
            base = controls.get("codeonly_baseline_mean_reproduced")
            base_txt = "none" if base is None else f"{base:.3f}"
            verdict = (
                f"complete_think_on_pilot_{n_scored}_windows_mean_change_fidelity_{mean_txt}"
                f"_vs_codeonly_{base_txt}"
            )
    gguf = st.get("gguf")
    sub = _substrate(st, rows, verdict)
    process_wall = time.monotonic() - t0
    resumed_wall = sum(
        float((r.get("generation") or {}).get("wall_s") or 0.0)
        for r in rows.values()
        if r.get("resumed")
    )
    artifact: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "requirement_id": REQUIREMENT_ID,
        "schema": SCHEMA,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": st.get("run_id"),
        "run_mode": "dry_run" if cfg.dry_run else "real",
        "honest_verdict": verdict,
        "inference_substrate": sub["name"],
        "inference_substrate_class": sub["class"],
        "planned_inference_substrate": "live_llm_inference",
        "substrate_evidence": {
            "n_model_calls_in_rows": sub["n_model_calls_in_rows"],
            "n_model_calls_in_unkept_windows": sub["n_model_calls_in_unkept_windows"],
            "generation_wall_s_in_rows": sub["generation_wall_s_in_rows"],
            "server_launched_this_process": bool(st.get("server_launched")),
            "rule": "Declared from the rows, so a resumed or rebuilt artifact keeps its class.",
        },
        "model_specs": [
            {
                "name": "Qwen3.8-27B",
                "hf_id": MODEL_ID,
                "file": MODEL_FILENAME,
                "path": None if gguf is None else str(gguf),
                "quantisation": "Q4_K_M",
                "kv_cache": "q8_0",
                "server": "llama.cpp llama-server, one stream",
                "invoked": sub["invoked"],
            }
        ],
        "scored_path_weights_not_run_here": dict(SCORED_PATH_WEIGHTS),
        "random_seed": SEED_BASE,
        "window_seed_bases": {s.game: window_seed(s) for s in specs},
        "reproducibility_checksum": _checksum(st),
        "preconditions_checked": list(st.get("preconditions") or []),
        # Total wall time of the work the artifact reports: this process plus the
        # generation time of rows resumed from earlier runs.
        "duration_s": round(process_wall + resumed_wall, 3),
        "duration_basis": {
            "this_process_s": round(process_wall, 3),
            "resumed_rows_generation_s": round(resumed_wall, 2),
        },
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "The pilot scores induced engines against recorded transitions. No executable "
            "oracle picks or repairs an engine. The expert engines are a ceiling control only."
        ),
        "gate_env_flags": gate_env_flags(),
        "induction_env": st.get("induction_env"),
        "preregistration": {
            "note": "docs/research-notes/b2-positive-control-2026-09-23.md",
            "section": "Pre-registered design for the pending think-ON pilot",
            "windows": list(PILOT_WINDOWS),
            "mask_grid_rows": MASK_GRID_ROWS,
            "excluded_heldout_rows": EXCLUDED_HELDOUT_ROWS,
            "baseline_mean": PREREGISTERED_BASELINE_MEAN,
            "live_budget": {"max_tokens": MAX_TOKENS, "call_timeout_s": 2400},
        },
        "live_settings": {
            "think_mode": "as induce_think_on() resolves",
            "tries": LIVE_TRIES,
            "max_tokens": MAX_TOKENS,
            "required": list(REQUIRED),
            "call_timeout_s": CALL_TIMEOUT_S,
            "server_flags": list(SERVER_FLAGS),
            "gpu_uuid": GPU1_UUID,
        },
        "deviations_from_live_path": list(DEVIATIONS_FROM_LIVE_PATH),
        "deviations_from_preregistration": [
            "None in the windows, masks, exclusions, metrics, controls, baseline, or budget.",
            "Added, not changed: the live_ladder and vllm_shape rescores are extra views. "
            "The pre-registered primary is the as-run value.",
        ],
        "scoring_wrapper": {
            "primary": "masked symmetric-union change fidelity over held-out changing rows; "
            "a raised, wrong-type, or wrong-shape row scores 0",
            "guard": "no-op hallucination rate; a raised no-op row counts as hallucinated",
            "secondary": "masked exact accuracy; live unmasked exact accuracy and its 1.0 pass",
            "levelup_rows": "excluded, as the live verifier excludes them",
            "isolation": (
                "each held-out row runs in its own new interpreter (exec, not fork) that "
                "receives only that row's grid, action, and a copy of data, plus the engine "
                "source; no other row exists in that process, because row k+1's input is "
                "row k's answer. The process reseeds random and numpy, runs in an empty temp "
                "directory, and installs an audit hook that refuses reads outside the Python "
                "install, all writes, and process, socket, and ctypes use"
            ),
            "output_type": "integer arrays only (integral floats are cast); object and "
            "structured arrays are refused",
            "engine_call_guard_s": default_timeout_s(),
            "engine_child": _relative(ENGINE_CHILD_PATH, cfg.repo_root),
        },
        "windows": [s.summary() for s in specs],
        "window_error": st.get("window_error"),
        "shard_mismatch": st.get("shard_mismatch"),
        "prompt_fidelity": st.get("fidelity"),
        "controls": controls or None,
        "per_window_rows": [rows[g] for g in cfg.windows if g in rows],
        "resumed_windows": st.get("resumed_windows", []),
        "comparison_to_baseline": comparison,
        "false_negative_risk": false_negative_risk(rows, controls) if controls else None,
        "infra_failed_generations": st.get("infra_failed_generations"),
        "sample_size_caveat": (
            "10 windows x 1 draw is a pilot. Each window has one prompt, so more draws would "
            "measure sampling variance only. No significance claim is made."
        ),
        "server": st.get("server"),
        "server_teardown": st.get("teardown"),
        "wedge": st.get("wedge"),
        "field_provenance": {
            "honest_verdict": "Lets a reader tell complete, partial, and blocked apart.",
            "preconditions_checked": "Shows which resources were verified before any call.",
            "duration_s": "Real generation takes minutes per window; seconds means no model ran.",
            "reproducibility_checksum": "Binds windows, prompts, controls, code, and config.",
            "verifier_is_oracle": "False: no executable oracle chooses the engine.",
        },
    }
    target = _write_artifact(cfg, artifact, verdict)
    progress(f"artifact {target}: {verdict}")
    return artifact


def _raise_system_exit(signum: int, frame: Any) -> None:
    # The default SIGTERM and SIGHUP actions skip `finally`, which would leave the
    # server running on GPU 1. Raising SystemExit runs the teardown.
    raise SystemExit(128 + signum)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--dry-run", action="store_true", help="no server, no GPU, stand-in output")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--artifact", type=Path, default=None)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--heartbeat-s", type=float, default=HEARTBEAT_S)
    args = ap.parse_args(argv)
    if args.dry_run and args.output_dir is None:
        # A dry run must not write into the research record by default.
        ap.error("--dry-run needs --output-dir (use a scratch directory)")
    root = Path(args.repo_root) if args.repo_root else _repo_root(start=Path(__file__))
    output_dir = Path(args.output_dir) if args.output_dir else root / DEFAULT_OUTPUT_REL
    if args.artifact:
        artifact_path = Path(args.artifact)
    elif args.dry_run:
        artifact_path = output_dir / "experiment_10010_dry_run.json"
    else:
        artifact_path = root / DEFAULT_ARTIFACT_REL
    cfg = RunConfig(
        repo_root=root,
        output_dir=output_dir,
        artifact_path=artifact_path,
        dry_run=bool(args.dry_run),
        port=int(args.port),
        heartbeat_s=float(args.heartbeat_s),
    )
    previous = {
        sig: signal.signal(sig, _raise_system_exit) for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        artifact = run_pilot(cfg)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    verdict = str(artifact["honest_verdict"])
    if verdict.startswith("complete"):
        return 0
    return 2 if verdict.startswith("blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
