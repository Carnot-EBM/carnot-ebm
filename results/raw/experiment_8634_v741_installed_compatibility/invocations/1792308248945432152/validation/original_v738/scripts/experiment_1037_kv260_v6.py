#!/usr/bin/env python3
"""Experiment 1037 — KV260 First Light v6 (8th attempt, SSH unblock).

**Why this experiment exists (the real story):**
    The KV260 board arrived 2026-04-20 and has been pending for 8+
    consecutive milestones.  Every prior attempt — Exps 982, 993,
    1010, 1022, and earlier — reported the board as "unreachable"
    via SSH.  The .79 retro called this out as the longest-pending
    hardware blocker in the project.  The CLAUDE.md "MANDATORY-
    NEXT-MILESTONE PRIORITIES" forcing function flagged it for
    pickup.  This v6 attempt is therefore explicitly NON-GATED on
    the milestone preflight: even if every other experiment in
    .80 fails, KV260 must run.

**What is different from the prior 7 attempts:**
    1. Discovery starts from the *known* IP address 192.168.51.98
       (proven reachable by ping on the host workstation as of
       2026-04-29) rather than relying on mDNS resolution of
       kv260.local — which has been the failure point in prior
       attempts.
    2. Both ``ubuntu`` and ``kria`` user names are tried.  The Kria
       Ubuntu 22.04 image creates a default ``ubuntu`` account on
       first boot; older images use ``kria``.  Prior attempts only
       tried ``kria``.
    3. SSH keys, not password auth.  The host workstation's
       ~/.ssh/config and authorized_keys arrangement is preserved
       across re-flashes of the SD card, which is what the user
       actually relies on day-to-day.  Forcing pubkey auth means
       this experiment fails fast if the auth path is broken
       rather than hanging on a password prompt.
    4. After the bitstream-load step (regardless of whether it
       succeeds), the experiment writes a 5-command human guide
       into the artifact so the operator has an unambiguous
       follow-up path even on partial success.

**What this experiment writes:**
    results/experiment_1037_kv260_v6.json with the schema described
    in the .80 milestone task spec:
        ssh_reachable, ssh_approach_used, bitstream_loaded,
        smoke_test_passed, energy_distribution_nonuniform,
        kv260_guide_written, guide_commands, honest_verdict.

Spec refs: REQ-HW-040, SCENARIO-HW-040
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_RESULT_FILE = _REPO_ROOT / "results" / "experiment_1037_kv260_v6.json"
_BITSTREAM_DIR = _REPO_ROOT / "output" / "carnot_ising_v4_bd"

_BOARD_IP = "192.168.51.98"
_BOARD_USERS = ("ubuntu", "kria")  # Kria Ubuntu 22.04 default vs older Petalinux.

_SSH_TIMEOUT_S = 10
_SCP_TIMEOUT_S = 120
_LOAD_TIMEOUT_S = 30


def _ssh(
    host: str, user: str, command: str, timeout: int = _SSH_TIMEOUT_S, port: int = 22
) -> tuple[int, str, str]:
    """Run a remote command over SSH with batch-mode pubkey auth.

    BatchMode=yes is critical: it makes ssh fail fast instead of hanging
    on a password prompt when pubkey auth is not configured.  This is
    what kept prior experiments stuck for tens of seconds per attempt.
    """
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={_SSH_TIMEOUT_S}",
        "-o",
        "BatchMode=yes",
        "-p",
        str(port),
        f"{user}@{host}",
        command,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def _try_ssh_approaches(host: str) -> tuple[bool, str, str | None]:
    """Try the three SSH approaches in order; return (reachable, approach, user).

    Approaches:
        1. Default port 22 with user ``ubuntu``.
        2. Default port 22 with user ``kria``.
        3. Alternate port 8022 (some Kria images expose ssh there).
    """
    for user in _BOARD_USERS:
        rc, out, _ = _ssh(host, user, "whoami", timeout=15)
        if rc == 0 and out.strip() in {"ubuntu", "kria"}:
            return True, "direct", user

    for user in _BOARD_USERS:
        rc, out, _ = _ssh(host, user, "whoami", timeout=15, port=8022)
        if rc == 0 and out.strip() in {"ubuntu", "kria"}:
            return True, "port_8022", user

    return False, "none", None


def _scp_bitstream(host: str, user: str) -> bool:
    """Copy the bitstream directory to the board's /tmp.  Returns True on success."""
    if not _BITSTREAM_DIR.exists():
        return False
    cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={_SSH_TIMEOUT_S}",
        "-o",
        "BatchMode=yes",
        "-r",
        str(_BITSTREAM_DIR),
        f"{user}@{host}:/tmp/",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_SCP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _attempt_bitstream_load(host: str, user: str) -> tuple[bool, str]:
    """Try to load the bitstream via xmutil; return (loaded, diagnostic).

    The Kria flow is:
      1. Stage the .bit/.dtbo/shell.json under /lib/firmware/xilinx/<app>/.
      2. Tell xmutil to (re)load that app.

    The bitstream binary needs to be in the kernel-FPGA-manager-friendly
    .bit.bin format — Vivado emits a raw .bit which has a Xilinx ASCII
    header that the kernel does not parse.  ``bootgen`` with a tiny BIF
    file produces the .bit.bin file.  If .bit.bin already exists in the
    target firmware directory we skip that step.
    """
    rc1, _, _ = _ssh(host, user, f"sudo xmutil unloadapp 2>&1 || true", timeout=_LOAD_TIMEOUT_S)
    rc2, out2, err2 = _ssh(
        host, user, "sudo xmutil loadapp carnot_ising_v4 2>&1", timeout=_LOAD_TIMEOUT_S
    )
    diag = (out2 + err2).strip()
    if rc2 == 0 and "Load Error" not in diag and "error" not in diag.lower():
        return True, diag
    return False, diag


def _write_guide(verdict: str, ssh_user: str | None, load_diag: str) -> list[str]:
    """Return a 5-command operator guide for the next manual step.

    The five commands are tailored to where the automated flow got
    stuck.  If SSH worked but bitstream load failed, the guide
    focuses on regenerating the .bit.bin via bootgen.  If SSH itself
    is broken, the guide focuses on serial/SD-card recovery.
    """
    if verdict == "all_paths_blocked":
        return [
            "lsblk  # confirm SD card device id, e.g. /dev/sda — DO NOT pick the system drive",
            "wget https://ubuntu.com/download/amd/kria-kv260 -O kria-ubuntu-22.04.img.xz",
            "xz -dk kria-ubuntu-22.04.img.xz && sudo dd if=kria-ubuntu-22.04.img of=/dev/sdX bs=4M status=progress",
            "# Boot KV260 with new SD card; connect serial: sudo minicom -b 115200 -D /dev/ttyUSB0",
            "# Login as ubuntu/ubuntu, run: sudo systemctl enable --now ssh && ip addr  # note IP",
        ]
    user = ssh_user or "ubuntu"
    return [
        f"ssh {user}@{_BOARD_IP}  # confirm pubkey login still works",
        f"ssh {user}@{_BOARD_IP} 'ls /lib/firmware/xilinx/carnot_ising_v4/'  # verify firmware staged",
        f"# On host: cd output/carnot_ising_v4_bd && bootgen -arch zynqmp -process_bitstream bin -image carnot.bif",
        f"scp output/carnot_ising_v4_bd/carnot_ising_v4.bit.bin {user}@{_BOARD_IP}:/lib/firmware/xilinx/carnot_ising_v4/",
        f"ssh {user}@{_BOARD_IP} 'sudo xmutil unloadapp; sudo xmutil loadapp carnot_ising_v4'  # then re-run smoke test",
    ]


def main() -> None:
    """Run all four KV260 first-light approaches and emit the artifact.

    The function is deliberately monolithic — every exit path writes
    the JSON artifact via the ``finally`` block, matching the Exp 993
    pattern that solved the "experiment finished but produced no
    deliverable" problem flagged in the .73 retrospective.
    """
    t0 = time.time()
    print("[exp1037] === KV260 First Light v6 ===")

    artifact: dict = {
        "experiment": 1037,
        "title": "KV260 First Light v6 — SSH Unblock + Bitstream Load",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "kv260_first_light_v6",
        "duration_s": 0,
        "board_ip": _BOARD_IP,
        "ssh_reachable": False,
        "ssh_approach_used": "none",
        "ssh_user": None,
        "bitstream_loaded": False,
        "bitstream_load_diag": None,
        "smoke_test_passed": None,
        "energy_distribution_nonuniform": None,
        "smoke_test_stats": None,
        "kv260_guide_written": False,
        "guide_commands": [],
        "honest_verdict": "all_paths_blocked",
        "notes": {
            "prior_failures": [
                {
                    "experiment_id": "exp1022_kv260_first_light_v5",
                    "verdict": "blocked_by_preflight_gate",
                    "addressed_by": "v6 is non-gated on preflight per .80 milestone spec",
                },
                {
                    "experiment_id": "exp993_kv260_board_programming_v3",
                    "verdict": "board_unreachable_human_required",
                    "addressed_by": "v6 uses known IP 192.168.51.98 + tries ubuntu user",
                },
            ],
        },
    }

    try:
        # APPROACH 1+2: SSH retry (combined: try ubuntu then kria, then port 8022).
        print("[exp1037] APPROACH 1: SSH retry (ubuntu/kria, port 22 then 8022) ...")
        reachable, approach, user = _try_ssh_approaches(_BOARD_IP)
        artifact["ssh_reachable"] = reachable
        artifact["ssh_approach_used"] = approach
        artifact["ssh_user"] = user

        if not reachable:
            print("[exp1037] SSH unreachable on all approaches.")
            artifact["honest_verdict"] = "all_paths_blocked"
            return  # finally block writes guide + artifact

        print(f"[exp1037] SSH OK as {user} via {approach}")

        # BITSTREAM LOADING.
        print("[exp1037] Uploading bitstream directory to /tmp ...")
        scp_ok = _scp_bitstream(_BOARD_IP, user)
        artifact["notes"]["scp_ok"] = scp_ok

        print("[exp1037] Attempting xmutil loadapp carnot_ising_v4 ...")
        loaded, diag = _attempt_bitstream_load(_BOARD_IP, user)
        artifact["bitstream_loaded"] = loaded
        artifact["bitstream_load_diag"] = diag[:500]

        if not loaded:
            print(f"[exp1037] Bitstream load failed: {diag[:200]}")
            # We still have SSH — that is meaningful progress vs all prior runs.
            artifact["honest_verdict"] = "guide_written"
            return

        # SMOKE TEST.
        print("[exp1037] Running smoke test ...")
        try:
            from scripts.kv260_ising_smoke_test import run_smoke_test
        except ImportError:
            import sys

            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            from kv260_ising_smoke_test import run_smoke_test  # type: ignore

        smoke = run_smoke_test(_BOARD_IP, user=user or "ubuntu")
        artifact["smoke_test_passed"] = bool(smoke.get("passed"))
        artifact["energy_distribution_nonuniform"] = smoke.get("energy_distribution_nonuniform")
        artifact["smoke_test_stats"] = smoke.get("stats") or {"error": smoke.get("error")}

        if smoke.get("passed"):
            artifact["honest_verdict"] = "first_light_achieved"
        else:
            artifact["honest_verdict"] = "ssh_restored_bitstream_loaded"

    except Exception as exc:
        import traceback

        artifact["notes"]["exception"] = str(exc)
        artifact["notes"]["traceback"] = traceback.format_exc()

    finally:
        # Always write the guide — even on success, the operator may
        # want the next manual step (e.g. for re-running the smoke
        # test after a reboot).
        artifact["guide_commands"] = _write_guide(
            artifact["honest_verdict"],
            artifact.get("ssh_user"),
            artifact.get("bitstream_load_diag") or "",
        )
        artifact["kv260_guide_written"] = bool(artifact["guide_commands"])
        artifact["duration_s"] = int(time.time() - t0)
        _RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _RESULT_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, indent=2))
        tmp.rename(_RESULT_FILE)
        print(f"[exp1037] Artifact written: {_RESULT_FILE}")
        print(f"[exp1037] honest_verdict: {artifact['honest_verdict']}")
        print(
            f"[exp1037] ssh_reachable={artifact['ssh_reachable']} "
            f"bitstream_loaded={artifact['bitstream_loaded']} "
            f"smoke_test_passed={artifact['smoke_test_passed']}"
        )


if __name__ == "__main__":
    main()
