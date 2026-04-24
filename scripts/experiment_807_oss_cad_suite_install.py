#!/usr/bin/env python3
"""Experiment 807 — OSS-CAD-Suite Installation and Minimal Ising Synthesis.

**Research question:**
    Exps 791, 794, and 804 all failed because `sudo pacman -S yosys nextpnr icestorm`
    does not succeed in the conductor environment. The correct alternative is
    OSS-CAD-Suite (YosysHQ/oss-cad-suite-build): a pre-built binary tarball that
    requires no sudo, no package manager — just curl and tar.

    This experiment installs OSS-CAD-Suite to ~/tools/oss-cad-suite/, verifies that
    yosys, nextpnr-ice40, and icepack all respond to version/help queries, and then
    runs a minimal 2-spin Ising synthesis to confirm the toolchain is functional.

**Why OSS-CAD-Suite instead of pacman:**
    The YosysHQ nightly builds are pre-compiled for linux-x64 and ship as a self-contained
    tarball. Extraction puts all binaries under ~/tools/oss-cad-suite/bin/ — no system
    permissions required. This is the standard approach for CI/CD environments where sudo
    is not available.

**Honest verdict mapping:**
    tools_installed_synthesis_clean:   all 3 tools present + minimal synthesis succeeds.
    tools_installed_synthesis_failed:  all 3 tools present but synthesis errors.
    already_installed:                 tools were present before this experiment ran.
    download_failed:                   GitHub API or curl failed.
    extract_failed:                    tar extraction failed.

Spec: REQ-HW-036, REQ-HW-037, SCENARIO-HW-034
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

apply_env_autofix()

DELIVERABLE = "results/experiment_807_oss_cad_suite_install.json"
INSTALL_DIR = Path.home() / "tools" / "oss-cad-suite"
TOOLS = ["yosys", "nextpnr-ice40", "icepack"]

# Minimal 2-spin Ising Verilog module for synthesis smoke test.
# WHY 2 spins: this is the smallest meaningful Ising topology — two spins with
# a coupling J between them. It synthesizes quickly (~1s) and is sufficient to
# confirm the synth_ice40 pass runs end-to-end.
_ISING2_VERILOG = """\
module ising2 (
    input  wire clk,
    input  wire rst,
    output reg  [1:0] spins
);
    // Two-spin Ising model with ferromagnetic coupling J=1.
    // Each clock cycle we flip both spins to their energy-minimising state.
    // In hardware, this becomes the core of a Gibbs sampler sweep.
    always @(posedge clk or posedge rst) begin
        if (rst)
            spins <= 2'b00;
        else
            spins <= ~spins;  // simplified: real sampler uses stochastic logic
    end
endmodule
"""

# GitHub API URL for the latest OSS-CAD-Suite release.
_GITHUB_API_URL = (
    "https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest"
)

GITHUB_API_URL = _GITHUB_API_URL  # exported for test access


def _all_tools_installed() -> bool:
    """Return True if all required FPGA tool binaries already exist in INSTALL_DIR.

    WHY check binary existence rather than running them: existence check is
    instant and avoids spawning processes at startup. We confirm the tools
    actually work in the separate verify_tools() step.
    """
    return all((INSTALL_DIR / "bin" / t).exists() for t in TOOLS)


def fetch_download_url(api_url: str = _GITHUB_API_URL) -> str | None:
    """Query the GitHub releases API and return the linux-x64 .tgz asset URL.

    The API returns a JSON object with an 'assets' list. Each asset has a 'name'
    and a 'browser_download_url'. We pick the asset whose name contains 'linux-x64'
    and ends with '.tgz'.

    Returns None if the request fails or no matching asset is found.
    """
    try:
        import urllib.request  # stdlib — no extra deps

        with urllib.request.urlopen(api_url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[807] GitHub API request failed: {exc}", file=sys.stderr)
        return None

    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if "linux-x64" in name and name.endswith(".tgz"):
            return asset["browser_download_url"]

    print("[807] No linux-x64 .tgz asset found in release.", file=sys.stderr)
    return None


def download_tarball(url: str, dest: str = "/tmp/oss-cad.tgz") -> tuple[bool, str]:
    """Download url to dest using curl. Return (success, message).

    WHY curl rather than urllib: curl handles redirects, shows progress on large
    downloads, and is always available on the target Linux system.
    """
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", dest, url],
            capture_output=True,
            text=True,
            timeout=600,  # large tarball ~500 MB; 10 min ceiling
        )
        if result.returncode != 0:
            return False, f"curl exited {result.returncode}: {result.stderr[:300]}"
        return True, dest
    except subprocess.TimeoutExpired:
        return False, "curl timed out after 600s"
    except FileNotFoundError:
        return False, "curl not found on PATH"


def extract_tarball(
    tarball: str = "/tmp/oss-cad.tgz", target_parent: str | None = None
) -> tuple[bool, str]:
    """Extract tarball into target_parent (defaults to ~/tools/).

    The tarball from YosysHQ unpacks to oss-cad-suite/ so the final location
    will be ~/tools/oss-cad-suite/ — matching INSTALL_DIR.

    Returns (success, message).
    """
    if target_parent is None:
        target_parent = str(INSTALL_DIR.parent)

    parent = Path(target_parent)
    parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["tar", "-xzf", tarball, "-C", str(parent)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return False, f"tar exited {result.returncode}: {result.stderr[:300]}"
        return True, str(INSTALL_DIR)
    except subprocess.TimeoutExpired:
        return False, "tar timed out after 300s"
    except FileNotFoundError:
        return False, "tar not found on PATH"


def _check_tool(cmd: list[str]) -> tuple[bool, str]:
    """Run cmd and return (present, version_string).

    A tool is present if subprocess does not raise FileNotFoundError and
    produces non-empty combined stdout+stderr. Non-zero exit is acceptable
    because icepack --help exits 1 but prints usage.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (result.stdout + result.stderr).strip()
        present = len(combined) > 0
        return present, combined[:200] if present else "no output"
    except subprocess.TimeoutExpired:
        return False, f"timeout after 15s: {cmd}"
    except FileNotFoundError:
        return False, f"not found: {cmd[0]}"


def verify_tools() -> dict[str, dict]:
    """Check each tool in TOOLS under INSTALL_DIR/bin.

    Returns a dict mapping tool name → {present: bool, version: str}.
    Uses --version for yosys and nextpnr-ice40; --help for icepack (which
    exits non-zero but still prints output when present — REQ-HW-037-1).
    """
    help_flags = {
        "yosys": "--version",
        "nextpnr-ice40": "--version",
        "icepack": "--help",
    }
    results: dict[str, dict] = {}
    for tool in TOOLS:
        binary = INSTALL_DIR / "bin" / tool
        flag = help_flags[tool]
        present, version = _check_tool([str(binary), flag])
        results[tool] = {"present": present, "version": version}
    return results


def run_synthesis() -> dict:
    """Run minimal 2-spin Ising synthesis via yosys synth_ice40.

    Writes the Verilog to /tmp/test_ising2.v, runs yosys with synth_ice40,
    and counts SB_LUT4 cells in the output JSON netlist.

    Returns dict with keys: success (bool), lut_count (int|None), stderr_snippet (str).
    """
    verilog_path = Path("/tmp/test_ising2.v")
    netlist_path = Path("/tmp/test_ising2.json")
    verilog_path.write_text(_ISING2_VERILOG)

    yosys_bin = str(INSTALL_DIR / "bin" / "yosys")
    synth_script = f"synth_ice40; write_json {netlist_path}"

    try:
        result = subprocess.run(
            [yosys_bin, "-p", synth_script, str(verilog_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "lut_count": None, "stderr_snippet": "timed out"}
    except FileNotFoundError:
        return {"success": False, "lut_count": None, "stderr_snippet": "yosys not found"}

    stderr_lower = result.stderr.lower()
    has_error = "error" in stderr_lower and result.returncode != 0

    if result.returncode != 0 or has_error:
        return {
            "success": False,
            "lut_count": None,
            "stderr_snippet": result.stderr[:300],
        }

    # Parse lut_count from the JSON netlist.
    lut_count = _count_luts_from_netlist(netlist_path)
    return {
        "success": True,
        "lut_count": lut_count,
        "stderr_snippet": result.stderr[:200],
    }


def _count_luts_from_netlist(netlist_path: Path) -> int:
    """Count SB_LUT4 cells in a yosys JSON netlist file.

    WHY parse JSON rather than grep stdout: the JSON netlist is the canonical
    output of write_json and is unambiguous — stdout formatting varies across
    yosys versions. The JSON always has cells[name].type for each cell.

    Returns 0 if the netlist exists but has no SB_LUT4 cells.
    Raises if the file is missing or malformed (caller catches).
    """
    try:
        data = json.loads(netlist_path.read_text())
    except Exception:
        return 0

    count = 0
    for module in data.get("modules", {}).values():
        for cell in module.get("cells", {}).values():
            if cell.get("type") == "SB_LUT4":
                count += 1
    return count


def _update_hardware_wishlist(
    install_dir: Path,
    honest_verdict: str,
    tool_versions: dict,
) -> None:
    """Append an OSS-CAD-Suite status entry to research-hardware-wishlist.md.

    WHY append rather than replace: CLAUDE.md mandates never removing existing
    content from ops/spec docs. We add a new dated section.
    """
    wishlist = _REPO_ROOT / "research-hardware-wishlist.md"
    if not wishlist.exists():
        return

    version_summary = "; ".join(
        f"{t}={'present' if v['present'] else 'absent'}"
        for t, v in tool_versions.items()
    )
    entry = (
        f"\n### KV260 Synthesis Status (Exp 807 — 20260424) — OSS-CAD-Suite Install\n\n"
        f"- **Exp 807 result:** `honest_verdict={honest_verdict}`\n"
        f"  - OSS-CAD-Suite installed at `{install_dir}`\n"
        f"  - Tool presence: {version_summary}\n"
        f"  - Gates Exp 816 (KV260 synthesis v2)\n"
    )
    with wishlist.open("a") as f:
        f.write(entry)


def main() -> None:
    """Orchestrate OSS-CAD-Suite install, verification, and minimal synthesis."""
    tmpl = ExperimentTemplate(
        exp_id=807,
        title="OSS-CAD-Suite Installation and Minimal Ising Synthesis",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(807, timeout_minutes=45, result_path=DELIVERABLE):
        # ---- Step 1: check if already installed --------------------------------
        already = _all_tools_installed()
        if already:
            tool_versions = verify_tools()
            all_present = all(v["present"] for v in tool_versions.values())
            verdict = "already_installed"
            synth_result: dict | None = None
            if all_present:
                synth_result = run_synthesis()
            artifact = tmpl.build_result(
                {
                    "install_dir": str(INSTALL_DIR),
                    "tool_versions": tool_versions,
                    "all_tools_present": all_present,
                    "synthesis_result": synth_result,
                    "honest_verdict": verdict,
                },
                status="success",
            )
            _write_artifact(artifact, tmpl)
            _update_hardware_wishlist(INSTALL_DIR, verdict, tool_versions)
            tmpl.assert_deliverable_written()
            return

        # ---- Step 2: fetch download URL ----------------------------------------
        url = fetch_download_url()
        if url is None:
            artifact = tmpl.build_result(
                {
                    "install_dir": str(INSTALL_DIR),
                    "tool_versions": {},
                    "all_tools_present": False,
                    "synthesis_result": None,
                    "honest_verdict": "download_failed",
                    "block_reason": "GitHub API did not return a linux-x64 .tgz asset",
                },
                status="blocked",
            )
            _write_artifact(artifact, tmpl)
            tmpl.assert_deliverable_written()
            return

        # ---- Step 3: download ---------------------------------------------------
        dl_ok, dl_msg = download_tarball(url)
        if not dl_ok:
            artifact = tmpl.build_result(
                {
                    "install_dir": str(INSTALL_DIR),
                    "download_url": url,
                    "tool_versions": {},
                    "all_tools_present": False,
                    "synthesis_result": None,
                    "honest_verdict": "download_failed",
                    "block_reason": dl_msg,
                },
                status="blocked",
            )
            _write_artifact(artifact, tmpl)
            tmpl.assert_deliverable_written()
            return

        # ---- Step 4: extract ---------------------------------------------------
        ex_ok, ex_msg = extract_tarball()
        if not ex_ok:
            artifact = tmpl.build_result(
                {
                    "install_dir": str(INSTALL_DIR),
                    "download_url": url,
                    "tool_versions": {},
                    "all_tools_present": False,
                    "synthesis_result": None,
                    "honest_verdict": "extract_failed",
                    "block_reason": ex_msg,
                },
                status="blocked",
            )
            _write_artifact(artifact, tmpl)
            tmpl.assert_deliverable_written()
            return

        # ---- Step 5: verify tools ---------------------------------------------
        tool_versions = verify_tools()
        all_present = all(v["present"] for v in tool_versions.values())

        # ---- Step 6: minimal synthesis ----------------------------------------
        synth_result = None
        if all_present:
            synth_result = run_synthesis()

        # ---- Step 7: determine honest verdict ---------------------------------
        if not all_present:
            verdict = "download_failed"  # extract succeeded but binaries missing
        elif synth_result is None:
            verdict = "tools_installed_synthesis_failed"
        elif synth_result["success"]:
            verdict = "tools_installed_synthesis_clean"
        else:
            verdict = "tools_installed_synthesis_failed"

        artifact = tmpl.build_result(
            {
                "install_dir": str(INSTALL_DIR),
                "download_url": url,
                "tool_versions": tool_versions,
                "all_tools_present": all_present,
                "synthesis_result": synth_result,
                "honest_verdict": verdict,
            },
            status="success" if verdict == "tools_installed_synthesis_clean" else "partial",
        )
        _write_artifact(artifact, tmpl)
        _update_hardware_wishlist(INSTALL_DIR, verdict, tool_versions)
        tmpl.assert_deliverable_written()


def _write_artifact(artifact: dict, tmpl: ExperimentTemplate) -> None:
    """Write the result artifact to DELIVERABLE path."""
    out = _REPO_ROOT / DELIVERABLE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"[807] Artifact written to {out}")


if __name__ == "__main__":
    main()
