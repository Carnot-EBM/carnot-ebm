"""The submission kernel's AGENT_SRC must actually RUN, not merely parse.

Spec refs: REQ-ARC-WMTE-4744, SCENARIO-ARC-WMTE-4744.

WHY THIS FILE EXISTS
====================
`scripts/kaggle/submission_kernel/main.py` carries the whole scored agent bootstrap as a raw
string literal (`AGENT_SRC`) which it writes to `my_agent.py` at runtime on Kaggle. Nothing
imports it, nothing executes it, and until now the only validation it had was:

  * `ast.parse(agent_src)` in `test_arc_scored_path_liveness_witness.py`, and
  * a handful of substring greps.

`ast.parse` proves the string is syntactically valid Python. It says NOTHING about whether the
names in it are defined. A typo like `_mains[0]` where the variable was renamed to `_mains_MUT`
parses perfectly and then dies with `NameError` on the scored run -- which, on a submission whose
whole value is the LLM tier, means a zero-score import failure discovered after the fact.

That exact shape was transiently present during the 2026-07-28 migration session
(`_heads_MUT`/`_mains_MUT` defined while `_mains[0]`/`_heads[0]` were read). It was restored with
the mtime preserved, so even a file-timestamp check gave no warning. The existing grep test happens
to catch that PARTICULAR shape because it greps for `_mains`; a differently-shaped typo ships.

WHAT THIS FILE ADDS
===================
1. A real `exec()` of the GGUF-resolution block against a FAKE `/kaggle/input` tree, asserting the
   main model and the draft head are told apart correctly -- including the adversarial case the
   block was written for (both files match a loose filter, and rglob order is undefined).
2. A static undefined-name sweep (`ruff --select F821`) over the extracted source, which catches
   the general typo class rather than one instance of it.

NOTE ON SCOPE. We do not exec the WHOLE of AGENT_SRC -- it imports jax, launches a subprocess, and
talks to a GPU. The GGUF-resolution block is extracted and run in isolation because it is the part
that (a) decides which weights the scored run loads, (b) is pure filesystem logic with no
dependencies, and (c) fails silently rather than loudly when wrong.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_KERNEL = Path(_REPO) / "scripts" / "kaggle" / "submission_kernel" / "main.py"


def _agent_src() -> str:
    """The `my_agent.py` source the kernel authors at runtime, as a string."""
    src = _KERNEL.read_text()
    ast.parse(src)  # the kernel itself must be valid Python
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "AGENT_SRC"
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    pytest.fail("AGENT_SRC not found in the submission kernel")
    raise AssertionError("unreachable")  # pragma: no cover - pytest.fail raises


def _gguf_resolution_block(agent_src: str) -> str:
    """The statements from `_HEAD_SUBSTR = ...` through the resolution print, verbatim.

    Sliced by SOURCE LINES rather than re-typed, so this test executes the shipped bytes. If the
    block is renamed or removed the slice fails loudly instead of silently testing a stale copy.
    """
    lines = agent_src.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("_HEAD_SUBSTR")), None)
    if start is None:
        pytest.fail("AGENT_SRC no longer defines _HEAD_SUBSTR -- the GGUF resolution block moved")
    end = next(
        (i for i, ln in enumerate(lines[start:], start) if "LLM TIER GGUF RESOLUTION" in ln),
        None,
    )
    if end is None:
        pytest.fail("AGENT_SRC no longer prints 'LLM TIER GGUF RESOLUTION'")
    return "\n".join(lines[start : end + 1])


def _run_resolution(tmp_path: Path, filenames: list[str]) -> dict:
    """exec the real resolution block against a fake /kaggle/input containing `filenames`."""
    inp = tmp_path / "kaggle_input"
    # Nest the files the way Kaggle actually mounts datasets (.../datasets/<owner>/<slug>/),
    # because the block uses rglob and a flat directory would not exercise that.
    for i, name in enumerate(filenames):
        d = inp / "datasets" / "iancblenke" / f"ds{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"")
    block = _gguf_resolution_block(_agent_src())
    ns: dict = {"inp": inp, "print": lambda *a, **k: None}
    exec(compile(block, "<AGENT_SRC gguf-resolution>", "exec"), ns)  # noqa: S102
    return ns


def test_agent_src_gguf_resolution_executes_and_separates_head_from_main(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4744: the resolution block RUNS and binds the 18.3GB main weights, not the
    491MB draft head.

    This is the adversarial case the block exists for: BOTH files are `*.gguf`, BOTH contain
    "gemma-4-31B", and rglob order between them is undefined. Binding the head as the generator
    loads, serves, and answers nonsense -- a silent failure with no error anywhere.
    """
    ns = _run_resolution(
        tmp_path,
        ["mtp-gemma-4-31B-it-Q8_0.gguf", "gemma-4-31B-it-Q4_K_M.gguf"],
    )
    assert ns["gguf"] is not None, "no main GGUF resolved"
    assert ns["gguf"].name == "gemma-4-31B-it-Q4_K_M.gguf"
    assert ns["mtp_head"] is not None, "no MTP head resolved"
    assert ns["mtp_head"].name == "mtp-gemma-4-31B-it-Q8_0.gguf"


def test_agent_src_gguf_resolution_is_order_independent(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4744: reversing the on-disk order must not change which file is the model.

    The pre-fix filter picked the main model by rglob order between two matching files. Creating
    the two datasets in the opposite order is the cheapest way to assert the fix is a real
    disambiguation and not a re-statement of the same accident.
    """
    ns = _run_resolution(
        tmp_path,
        ["gemma-4-31B-it-Q4_K_M.gguf", "mtp-gemma-4-31B-it-Q8_0.gguf"],
    )
    assert ns["gguf"].name == "gemma-4-31B-it-Q4_K_M.gguf"
    assert ns["mtp_head"].name == "mtp-gemma-4-31B-it-Q8_0.gguf"


def test_agent_src_gguf_resolution_head_absent_is_not_an_error(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4744: with only the main model attached, `mtp_head` is None and the main
    model still resolves.

    A missing head is a DEGRADED but valid scored run (no speculative decoding, ~1.4x slower
    decode). It must not take the generator down with it, and it must never fall back to using the
    main weights as their own draft -- llama.cpp accepts that and then silently disables
    speculation.
    """
    ns = _run_resolution(tmp_path, ["gemma-4-31B-it-Q4_K_M.gguf"])
    assert ns["gguf"].name == "gemma-4-31B-it-Q4_K_M.gguf"
    assert ns["mtp_head"] is None


def test_agent_src_has_no_undefined_names() -> None:
    """REQ-ARC-WMTE-4744: `ruff --select F821` over AGENT_SRC catches the typo class `ast.parse`
    cannot see.

    This is the general form of the `_mains_MUT` incident: a renamed variable, a dropped import, a
    misspelled attribute -- all parse, all raise NameError on the scored run. Skipping when ruff is
    unavailable would make this an invisible failure, so its absence FAILS instead: the tool is a
    declared dev dependency of this repo and a missing linter is a broken environment, not a
    reason to stop checking.
    """
    ruff = shutil.which("ruff")
    assert ruff, "ruff is required to validate AGENT_SRC for undefined names"
    src = _agent_src()
    # F821 needs a real file to report against; the name is arbitrary but must end in .py.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "my_agent.py"
        p.write_text(src)
        out = subprocess.run(
            [ruff, "check", "--isolated", "--select", "F821", "--output-format", "concise", str(p)],
            capture_output=True,
            text=True,
        )
    assert out.returncode == 0, (
        "AGENT_SRC references undefined names -- this ships as a zero-score NameError on the "
        "scored run, and `ast.parse` cannot see it:\n"
        + textwrap.indent(out.stdout + out.stderr, "    ")
    )
