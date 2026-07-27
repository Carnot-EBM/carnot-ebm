#!/usr/bin/env python3
"""MUTATION PROOF for the 2026-07-27 generator-liveness guard.

A guard is not real until (a) it fires on the recorded failure it was written for, and
(b) DISABLING it makes a test go red. (b) is the half this project has shipped without
before: the determination-preservation lint printed OK on a faithful replay of its own
origin incident, because nothing ever checked that its checks were load-bearing.

Each mutation below neuters exactly ONE branch of the new code by text substitution,
runs the two liveness suites, and records which tests die. A mutation that kills ZERO
tests means that branch is decorative. Originals are restored in a finally block.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
LINT = REPO / "scripts" / "arc_llm_on_liveness_lint.py"
WM = REPO / "python" / "carnot" / "agentic" / "arc_executable_world_model.py"
AG = REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
SUITES = [
    "tests/python/test_arc_scored_path_liveness_witness.py",
    "tests/python/test_arc_llm_on_liveness_lint.py",
]

MUTATIONS = [
    # (name, file, find, replace, what_it_neuters)
    (
        "M1_emitter_stops_counting_server_failures",
        WM,
        "        self.n_server_failures += 1\n",
        "        self.n_server_failures += 0  # MUTANT\n",
        "the dead channel goes dead again: real swallowed failures stop being counted",
    ),
    (
        "M2_emitter_stops_counting_calls",
        WM,
        '        self.n_completion_calls += 1\n        if not self._ensure_server():\n            msg = (\n                f"GPU llama-server failed for {self.repo_substr}; SOTA models "\n                "must run on GPU (no CPU fallback)"\n            )\n            self._note_server_failure(msg)\n            return False, msg\n        # L2 induction truncation fix',  # noqa: E501
        '        if not self._ensure_server():\n            msg = (\n                f"GPU llama-server failed for {self.repo_substr}; SOTA models "\n                "must run on GPU (no CPU fallback)"\n            )\n            return False, msg\n        # L2 induction truncation fix',  # noqa: E501
        "generate() reverts to the pre-change swallow with no counters at all",
    ),
    (
        "M3_lint_drops_DEAD_GENERATOR",
        LINT,
        "    if healthy_after is False:\n",
        "    if False:  # MUTANT\n",
        "the primary liveness check",
    ),
    (
        "M4_lint_treats_absent_calls_as_never_engaged",
        LINT,
        "    never_engaged = calls == 0  # explicit zero only",
        "    never_engaged = calls is None or calls == 0  # MUTANT: absent counts as zero",
        "the plausible-but-wrong None-is-zero reading, which would EXEMPT every "
        "pre-existing row (including the 8 origin cells) from NO_COMPLETIONS",
    ),
    (
        "M5_lint_drops_the_never_engaged_exemption",
        LINT,
        "    if responses == 0 and not never_engaged:",
        "    if responses == 0:  # MUTANT",
        "the over-fire guard: a game that never stalled would hard-FAIL",
    ),
    (
        "M6_lint_drops_WITNESS_SELF_CONTRADICTORY",
        LINT,
        "    if never_engaged and isinstance(errors, int) and errors > 0:",
        "    if False and isinstance(errors, int) and errors > 0:  # MUTANT",
        "the anti-gaming branch: zeroing calls would silently earn the WARN",
    ),
    (
        "M7_cleanup_stops_emitting",
        AG,
        "            try:\n                self._emit_generator_liveness_witness()",
        "            try:\n                pass  # MUTANT: witness not emitted",
        "the scored-path witness emission entirely",
    ),
    (
        "M8_pool_truncation_message_collapses_into_the_budget_message",
        WM,
        "        got = self.last_generated_tokens\n        if isinstance(got, int) and 0 <= got < self.max_tokens - 8:",  # noqa: E501
        "        got = self.last_generated_tokens\n        if False and isinstance(got, int) and 0 <= got < self.max_tokens - 8:",  # noqa: E501
        "the mode-C/mode-budget distinction (both would read as 'HIT n_predict')",
    ),
    (
        "M9_describer_drops_the_response_body",
        WM,
        '        reader = getattr(exc, "read", None)',
        "        reader = None  # MUTANT",
        "the one informative string in the failure (the 500/400 response body)",
    ),
]


def run_suites() -> tuple[int, list[str]]:
    proc = subprocess.run(
        [PY, "-m", "pytest", *SUITES, "-q", "--no-cov", "-n", "4"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    dead = sorted(
        {
            line.split(" ")[1].split("::")[-1]
            for line in proc.stdout.splitlines()
            if line.startswith("FAILED ") or line.startswith("ERROR ")
        }
    )
    return proc.returncode, dead


def main() -> int:
    baseline_rc, baseline_dead = run_suites()
    out = {"baseline_returncode": baseline_rc, "baseline_failures": baseline_dead, "mutations": []}
    if baseline_rc != 0:
        print("BASELINE IS NOT GREEN -- mutation results would be meaningless")
        print(json.dumps(out, indent=1))
        return 2

    for name, path, find, repl, neuters in MUTATIONS:
        original = path.read_text()
        if original.count(find) != 1:
            out["mutations"].append(
                {
                    "mutation": name,
                    "status": "PATCH_DID_NOT_APPLY",
                    "occurrences": original.count(find),
                    "note": "the anchor text is not uniquely present -- this mutation proves nothing",  # noqa: E501
                }
            )
            print(f"{name}: PATCH DID NOT APPLY ({original.count(find)} matches)")
            continue
        try:
            path.write_text(original.replace(find, repl))
            rc, dead = run_suites()
        finally:
            path.write_text(original)
        killed = sorted(set(dead) - set(baseline_dead))
        out["mutations"].append(
            {
                "mutation": name,
                "file": str(path.relative_to(REPO)),
                "neuters": neuters,
                "returncode": rc,
                "tests_killed": killed,
                "n_killed": len(killed),
                "status": "LOAD_BEARING" if killed else "DECORATIVE_BRANCH_NO_TEST_DIED",
            }
        )
        print(
            f"{name}: {'LOAD_BEARING' if killed else '*** DECORATIVE ***'} "
            f"killed={len(killed)} {killed}"
        )

    # restore-integrity check: after all mutations the suite must be green again
    rc, dead = run_suites()
    out["post_restore_returncode"] = rc
    out["post_restore_failures"] = dead
    out["all_branches_load_bearing"] = all(
        m.get("status") == "LOAD_BEARING" for m in out["mutations"]
    )
    Path(__file__).with_name("mutate.json").write_text(json.dumps(out, indent=1))
    print(f"\npost-restore rc={rc} failures={dead}")
    print(f"all_branches_load_bearing={out['all_branches_load_bearing']}")
    return 0 if out["all_branches_load_bearing"] and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
