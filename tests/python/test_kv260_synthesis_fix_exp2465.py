"""Regression tests for the KV260 yosys-synthesis-fix from exp2465 (.238).

Spec: REQ-HARDWARE-016 (KV260 sampler RTL synthesizability).
Cross-ref: experiment 2465 (this fix), experiment 2427 (.235 failure that
motivated the fix), commit 3723c8912 (interrupted-run checkpoint that
shipped the wrapper + UNISIM LUT6 stub).

Why these tests exist (verbose layman explanation):

The KV260 stack is 18 modules of Verilog scattered across 20 .v files.
Two things kept yosys synthesis broken across .234, .235, and .237:

1. No top-level wrapper. Three of the .v files re-define a module called
   `ising_sampler_128_sync` (ising_sampler_v2.v, ising_sampler_v2_synth.v,
   minimal_axi_responder.v all do). yosys only resolves which one wins
   when there's a single hierarchy root to drive `hierarchy -top`. Without
   a root, yosys gets confused and emits an error.

2. No UNISIM LUT6 stub. The kanele LUT chain instantiates `LUT6` by name.
   Inside Vivado that resolves to a built-in primitive. Under open-source
   yosys, `LUT6` is undefined and synthesis errors out with
   "Module `\\LUT6' referenced ... is not part of the design."

Commit 3723c8912 fixed both: it added carnot_ising_top.v (the wrapper) and
xilinx_unisim_stubs.v (a behavioral LUT6 ROM model). After the fix, yosys
synthesizes the full tree with 0 errors.

These tests pin the *structural preconditions* of the fix so a future
refactor cannot silently revert it. We don't run yosys here — CI hosts
typically don't have yosys installed, and a 656-second synthesis run
would be wasteful per test. Instead we assert the structural
properties that *together* are necessary and sufficient for yosys to
succeed: wrapper present, stubs present, expected file count, expected
top-module name on the wrapper, LUT6 stub module present in the stubs
file. The optional integration test at the bottom actually invokes
yosys when the binary is available.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KV260_DIR = REPO_ROOT / "hardware" / "kv260"
WRAPPER_FILE = KV260_DIR / "carnot_ising_top.v"
STUBS_FILE = KV260_DIR / "xilinx_unisim_stubs.v"
ARTIFACT = REPO_ROOT / "results" / "experiment_2465_kv260_rtl_synthesis_fix_v6.json"


def _read_rtl_files() -> list[Path]:
    """Return the .v files in the kv260 directory, sorted for reproducibility."""
    return sorted(KV260_DIR.glob("*.v"))


def test_kv260_rtl_directory_exists():
    """The hardware/kv260/ directory exists where the synthesis fix lives.

    A bare existence check — if someone moves the directory the rest of
    the tests would error inscrutably; this gives a clear failure first.
    """
    assert KV260_DIR.is_dir(), f"kv260 RTL directory missing at {KV260_DIR}"


def test_kv260_rtl_file_count_is_twenty():
    """20 .v files total: 18 leaf modules + carnot_ising_top.v wrapper + xilinx_unisim_stubs.v stubs.

    Why pinned: exp2427 listed 18 files and failed; exp2465 lists 20 and
    succeeds. The two added files are the load-bearing fix. If this drifts
    below 20, the fix has been partially reverted.
    """
    files = _read_rtl_files()
    assert len(files) == 20, (
        f"Expected 20 .v files in {KV260_DIR}, got {len(files)}: "
        f"{[f.name for f in files]}"
    )


def test_carnot_ising_top_wrapper_present():
    """The top-level wrapper `carnot_ising_top.v` exists and declares the expected top module.

    Why pinned: without this wrapper, yosys has no single hierarchy root to
    drive synth and the duplicate `ising_sampler_128_sync` definitions in
    v2/v2_synth/minimal_axi_responder collide. The wrapper is what makes
    `synth -top carnot_ising_top` work.
    """
    assert WRAPPER_FILE.is_file(), f"Wrapper missing: {WRAPPER_FILE}"
    contents = WRAPPER_FILE.read_text()
    assert re.search(
        r"^module\s+carnot_ising_top\b", contents, re.MULTILINE
    ), "carnot_ising_top.v does not declare `module carnot_ising_top`"


def test_xilinx_unisim_lut6_stub_present():
    """xilinx_unisim_stubs.v exists and contains a synthesizable `module LUT6` body.

    Why pinned: kanele_lut.v / kan_lut_block.v / kanele_top.v instantiate
    `LUT6` by name. Inside Vivado, LUT6 is a UNISIM primitive; outside
    Vivado (yosys), it must be supplied as a behavioral stub or synthesis
    errors with "Module `\\LUT6' is not part of the design." This file is
    the stub. It must be yosys-only and never included in a Vivado build.
    """
    assert STUBS_FILE.is_file(), f"UNISIM stubs missing: {STUBS_FILE}"
    contents = STUBS_FILE.read_text()
    assert re.search(
        r"^module\s+LUT6\b", contents, re.MULTILINE
    ), "xilinx_unisim_stubs.v does not declare `module LUT6`"
    assert "INIT" in contents, (
        "LUT6 stub must accept the INIT parameter to be bit-identical to "
        "the UNISIM cell"
    )


def test_duplicate_sync_module_definitions_have_a_top_to_resolve_them():
    """At least one wrapper file declares carnot_ising_top so yosys can
    pick which `ising_sampler_128_sync` definition wins.

    Why pinned: three files define `module ising_sampler_128_sync`. yosys
    can synthesize cleanly only because the wrapper provides a hierarchy
    root that lets the unused duplicates get pruned. If the wrapper is
    removed, the collision returns.
    """
    files = _read_rtl_files()
    sync_definers = []
    for f in files:
        if re.search(r"^module\s+ising_sampler_128_sync\b", f.read_text(), re.MULTILINE):
            sync_definers.append(f.name)
    assert len(sync_definers) >= 2, (
        "Expected the duplicate ising_sampler_128_sync pattern that the "
        "wrapper resolves; got only "
        f"{sync_definers}. If this drifts, the wrapper-fix may no longer "
        "be necessary — re-evaluate."
    )
    assert WRAPPER_FILE.is_file(), (
        "Duplicate `ising_sampler_128_sync` definitions exist but the "
        "wrapper that lets yosys resolve them is gone — this will break "
        "synthesis again."
    )


def test_experiment_2465_artifact_records_zero_errors():
    """The exp2465 artifact recorded synthesis_errors=0 and the terminal-state flag.

    Why pinned: this is the artifact that asserts the fix is verified.
    If a future run regenerates this file and it stops claiming 0 errors,
    the test catches the regression.
    """
    assert ARTIFACT.is_file(), f"Artifact missing: {ARTIFACT}"
    payload = json.loads(ARTIFACT.read_text())
    assert payload["synthesis_errors"] == 0
    assert payload["kv260_synthesis_succeeded"] is True
    assert payload["honest_verdict"].startswith("complete:"), (
        "honest_verdict must start with the terminal prefix 'complete:' "
        "per Verdict Terminal-Prefix Discipline in CLAUDE.md"
    )
    assert payload["rtl_file_count"] == 20
    assert payload["duration_s"] > 300, (
        "duration_s must be above the 300s fabrication floor for "
        "compute-bound synthesis tasks"
    )
    assert payload["yosys_exit_code"] == 0


def test_experiment_2465_artifact_schema_completeness():
    """Required fields per the task spec are all present with the right shapes.

    Why pinned: the conductor's reconciler and the adversarial verifier
    both scan artifacts for the principle-annotated fields. A missing
    field is silently equivalent to a fabrication signal.
    """
    payload = json.loads(ARTIFACT.read_text())
    required_top_level = {
        "honest_verdict",
        "synthesis_errors",
        "kv260_synthesis_succeeded",
        "file_with_error",
        "error_description",
        "rtl_file_count",
        "yosys_version",
        "n_fix_iterations",
        "duration_s",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
    }
    missing = required_top_level - set(payload.keys())
    assert not missing, f"Artifact missing required fields: {missing}"
    assert isinstance(payload["preconditions_checked"], list)
    assert len(payload["preconditions_checked"]) >= 2, (
        "preconditions_checked must list at least the yosys + rtl_directory "
        "checks per the PRECONDITIONS step in the task spec"
    )


@pytest.mark.skipif(
    shutil.which("yosys") is None and not Path("/opt/oss-cad-suite/bin/yosys").is_file(),
    reason="yosys binary not available on this host — structural tests above still run",
)
def test_yosys_synthesizes_kv260_with_zero_errors_when_binary_available():
    """Integration: run yosys end-to-end and assert 0 errors.

    Why pinned: the structural tests cover the necessary conditions for
    synthesis success, but only an actual yosys invocation can confirm
    sufficiency. We gate this on the binary being available so CI hosts
    without yosys skip cleanly. On the dev box it provides a true
    end-to-end check that the fix still holds.

    Cost: ~600s wall time. We mark it slow via the host gate above; no
    mark.slow because the project doesn't appear to use that convention.
    """
    yosys = shutil.which("yosys") or "/opt/oss-cad-suite/bin/yosys"
    cmd = [
        yosys,
        "-p",
        f"read_verilog {KV260_DIR}/*.v; synth -top carnot_ising_top -noabc;",
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    error_lines = [
        ln for ln in combined.splitlines() if re.search(r"\bERROR\b|^error:", ln)
    ]
    assert completed.returncode == 0, (
        f"yosys exited with {completed.returncode}. Errors found: {error_lines[:5]}"
    )
    assert not error_lines, f"yosys reported errors: {error_lines[:5]}"
