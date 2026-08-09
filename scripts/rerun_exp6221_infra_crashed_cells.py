#!/usr/bin/env python3
"""Correction re-run for REQ-ARC-WMTE-6242/6243: three (game, "think") cells in
`results/experiment_6221_gemma_think_mode_ab_expanded_roster.json` recorded a
`RemoteDisconnected('Remote end closed connection without response')` failure with
`max_raw_completion_len=0, n_generate_calls=0` -- zero tokens generated. That is the project's own
long-standing, unsolved llama-server "reaper" crash signature (see ops/known-issues.md), NOT a
think-mode capability signal. The checkpoint marks a (game, arm) cell done on any result including
a hard failure, so these three were never retried by the original run.

Reruns ONLY sk48/ls20/cd82's "think" arm, on an otherwise-idle GPU. Reuses
`experiment_6199_gemma_think_mode_ab`'s own `build_levelup_window` / `run_arm` /
`_configure_arm` unmodified -- this is a targeted recollection of specific cells, not a new
measurement methodology.

RETRY-ON-REAPER (added after the first attempt hit the SAME crash mid-run, 2026-08-09): every
crash observed this session landed on a THINK-mode call specifically (200-900s duration; 0
crashes on the much-shorter no_think calls) -- consistent with, but not proof of, a
connection/watchdog timeout tied to request duration rather than pure randomness. Each game gets
up to MAX_ATTEMPTS tries; a RemoteDisconnected result calls `prop._ensure_server()` again before
retrying (it health-checks and relaunches a dead server; a live, reusable server is a no-op) --
matches `_ensure_server`'s own documented reuse-or-relaunch contract. Per-game checkpointing
(write the output file after EVERY game, not just at the end) so a process kill mid-run does not
lose already-completed cells, mirroring the parent experiment's own checkpoint discipline.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

TARGET_GAMES = ("sk48", "ls20", "cd82")
ARTIFACT_PATH = REPO_ROOT / "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json"
OUT_PATH = REPO_ROOT / "results/rerun_exp6221_infra_crashed_cells_20260809.json"
MAX_ATTEMPTS = 4


def _load_done() -> dict:
    if not OUT_PATH.exists():
        return {}
    data = json.loads(OUT_PATH.read_text())
    return {row["game"]: row for row in data.get("new_rows", []) if row.get("game")}


def _save(done: dict) -> None:
    OUT_PATH.write_text(json.dumps({"new_rows": list(done.values())}, indent=2, default=str) + "\n")


def main() -> int:
    if os.environ.get("CARNOT_ARC_INDUCE_N_CTX") != "32768":
        raise SystemExit("set CARNOT_ARC_INDUCE_N_CTX=32768 before running this script")
    if not ARTIFACT_PATH.exists():
        raise SystemExit(f"missing {ARTIFACT_PATH}")

    from carnot import experiment_6199_gemma_think_mode_ab as exp6199

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = exp6199.CUDA_GPU_INDEX
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"

    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    prop = LocalGGUFProposer(
        repo_substr=exp6199.GGUF_REPO_SUBSTR,
        port=exp6199.SERVER_PORT,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=exp6199.SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=exp6199.INDUCE_TIMEOUT_S,
    )

    def _ensure(label: str) -> None:
        try:
            up = prop._ensure_server()
        except GeneratorCudaRequiredError as exc:
            raise SystemExit(f"blocked_cuda_unavailable ({label}): {exc}") from exc
        if not up:
            raise SystemExit(f"blocked_cuda_server_failed_to_start ({label})")

    _ensure("initial")

    done = _load_done()
    for game in TARGET_GAMES:
        if game in done and not done[game].get("induction_failure_detail", "").startswith(
            "split induce: engine failed: local gguf (GPU server) failed: RemoteDisconnected"
        ):
            print(f"[rerun] {game} already done (checkpoint), skipping", flush=True)
            continue
        built = exp6199.build_levelup_window(game)
        if built is None:
            done[game] = {"game": game, "arm": "think", "rerun_window_error": "no_levelup_window"}
            _save(done)
            continue
        window, cell = built
        print(f"[rerun] {game}: window n={len(window)} cell={cell}", flush=True)

        row = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            row = exp6199.run_arm(prop, game, "think", window, cell)
            detail = str(row.get("induction_failure_detail", ""))
            is_reaper_crash = "RemoteDisconnected" in detail
            print(
                f"[rerun] {game} think attempt {attempt}/{MAX_ATTEMPTS}: "
                f"ok={row.get('induction_ok')} reaper_crash={is_reaper_crash} "
                f"detail={detail[:120]}",
                flush=True,
            )
            if not is_reaper_crash:
                break  # success, or a genuine (non-infra) failure -- either way, stop retrying
            if attempt < MAX_ATTEMPTS:
                print(f"[rerun] {game}: reaper crash, re-ensuring server before retry", flush=True)
                _ensure(f"{game} retry {attempt + 1}")

        row["rerun_of_infra_crash"] = True
        row["rerun_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row["rerun_attempts_used"] = attempt
        done[game] = row
        _save(done)
        print(f"[rerun] {game} think FINAL: {row}", flush=True)

    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
