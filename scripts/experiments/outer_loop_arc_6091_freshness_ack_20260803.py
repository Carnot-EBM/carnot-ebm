"""Record a VERIFIED-INERT freshness acknowledgement for the REQ-ARC-WMTE-6091 source change.

WHAT DRIFTED AND WHY IT IS INERT. REQ-ARC-WMTE-6091 added, to
`python/carnot/agentic/arc_executable_world_model.py`: `refactor_show_engine_enabled()`,
`_current_engine_source()`, `_REFACTOR_ENGINE_BLOCK_HEADER`, two module constants, and a
CONDITIONAL splice inside `refactor_prompt`. The flag `CARNOT_ARC_REFACTOR_SHOW_ENGINE` defaults
OFF, and with it off `refactor_prompt` renders BYTE-IDENTICAL output to the pre-6091 version --
which is asserted directly, not argued:

  tests/python/test_arc_refactor_show_engine_20260803.py
    ::test_off_arm_is_byte_identical_to_the_shipped_prompt   (unset == "0" == shipped, and an
                                                              explicit engine_source= changes
                                                              nothing while the flag is off)
    ::test_flag_defaults_off

and the splice is MUTATION-PROVEN live: deleting `{engine_block}` from the prompt f-string turns
3 of the 7 tests red, so the ON path is genuinely exercised and the OFF path is genuinely the old
one. No other function in the module was touched, so no artifact that never sets the flag can
have moved.

A REBUILD IS PREFERRED WHERE POSSIBLE and is NOT possible here: every drifted artifact is a live
GPU measurement (world-model trust/change gates, inducer head-to-head, LLM-on concurrency
envelopes), and re-running them needs a working local generator -- which is exactly what this
session established the host currently cannot keep alive (see
results/experiment_6091_refine_engine_visible_ab.json: server lifetimes 369/17/10/53 s, plus a
no-harness control at 22 s). Rebuilding under a generator that dies mid-run would REPLACE sound
historical numbers with worse ones, which is the opposite of what the freshness gate protects.

APPEND-ONLY. Each artifact gains one entry in `provenance.freshness_acknowledgements`. Nothing
is removed, no recorded sha256 is edited, and no number is touched -- editing the recorded hash
is the exact laundering `_acknowledged_inert_drift` exists to prevent. The acknowledgement pins
the EXACT new hash, so any further edit to the module re-arms the lint automatically.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEP = REPO / "python" / "carnot" / "agentic" / "arc_executable_world_model.py"

TARGETS = [
    "results/experiment_6011_world_model_change_gate_four_arm.json",
    "results/experiment_6012_hidden_state_trust_gate_hole.json",
    "results/experiment_6013_hidden_state_change_gate_closure.json",
    "results/experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b.json",
    "results/outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json",
    "results/outer_loop_arc_generator_concurrency_fix_20260727.json",
    "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json",
]

REASON = (
    "REQ-ARC-WMTE-6091 added a DEFAULT-OFF flag (CARNOT_ARC_REFACTOR_SHOW_ENGINE) and a "
    "conditional engine-source splice to refactor_prompt. With the flag off the rendered prompt "
    "is byte-identical to the pre-6091 version, and no other function in the module changed, so "
    "this artifact -- which never sets the flag -- cannot have moved."
)
EVIDENCE = (
    "tests/python/test_arc_refactor_show_engine_20260803.py::"
    "test_off_arm_is_byte_identical_to_the_shipped_prompt asserts unset == '0' == the shipped "
    "prompt (and that an explicit engine_source= is ignored while off); "
    "::test_flag_defaults_off pins the default. MUTATION-PROVEN: deleting the {engine_block} "
    "splice from the prompt f-string turns 3 of 7 tests red, source restored and re-verified "
    "green. Rebuild not attempted because every target is a live-GPU measurement and this host "
    "cannot currently keep a generator alive (results/experiment_6091_refine_engine_visible_ab"
    ".json: server lifetimes 369/17/10/53 s; no-harness control 22 s)."
)


def main() -> int:
    now = hashlib.sha256(DEP.read_bytes()).hexdigest()
    changed = []
    for rel in TARGETS:
        p = REPO / rel
        if not p.exists():
            print(f"SKIP (absent): {rel}")
            continue
        d = json.loads(p.read_text())
        prov = d.get("provenance")
        if not isinstance(prov, dict):
            print(f"SKIP (no provenance): {rel}")
            continue
        # find what the artifact recorded for this dependency, so `sha256_was` is READ, not typed
        was = None
        entries = list(prov.get("code") or [])
        for src_key in ("rows_source", "rows_sources", "sources"):
            v = prov.get(src_key)
            if isinstance(v, list):
                entries.extend(x for x in v if isinstance(x, dict))
        for e in entries:
            if str(e.get("path", "")).endswith("arc_executable_world_model.py"):
                was = e.get("sha256")
                path_as_recorded = str(e.get("path"))
                break
        if was is None:
            print(f"SKIP (dependency not recorded): {rel}")
            continue
        if was == now:
            print(f"SKIP (already fresh): {rel}")
            continue
        acks = prov.setdefault("freshness_acknowledgements", [])
        if any(
            isinstance(a, dict)
            and str(a.get("path", "")).endswith("arc_executable_world_model.py")
            and a.get("sha256_now") == now
            for a in acks
        ):
            print(f"SKIP (already acknowledged): {rel}")
            continue
        acks.append(
            {
                "path": path_as_recorded,
                "sha256_was": was,
                "sha256_now": now,
                "reason": REASON,
                "evidence": EVIDENCE,
                "acknowledged_by": "outer-loop 2026-08-03 (REQ-ARC-WMTE-6091)",
            }
        )
        # PRESERVE THE FILE'S OWN INDENT. Writing indent=1 over an indent=2 artifact reflows
        # the WHOLE file: measured at 94,822 deletions across these seven artifacts on the
        # first attempt, which is a rewrite of the research record wearing the costume of a
        # one-field append. Detected from the file's second line, not assumed.
        first_lines = p.read_text().splitlines()[:2]
        indent = 2
        if len(first_lines) > 1:
            lead = len(first_lines[1]) - len(first_lines[1].lstrip(" "))
            indent = lead or 2
        p.write_text(json.dumps(d, indent=indent) + "\n")
        changed.append(rel)
        print(f"ACK: {rel}")
    print(json.dumps({"dependency_sha256_now": now, "artifacts_acknowledged": changed}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
