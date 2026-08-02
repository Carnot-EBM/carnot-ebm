"""Append a `provenance.freshness_acknowledgements` entry to each artifact this change stales.

WHY NOT REBUILD. `artifact_freshness_lint.py` offers two remedies: rebuild the artifact,
or acknowledge the drift as verified-inert. Rebuilding is the stronger proof and is the
right default -- but these 7 artifacts carry MEASURED wall-clock timings and live-server
witnesses that a rebuild would overwrite with today's numbers. This project has already
paid for that once: a previous commit rebuilt four artifacts and 132 timing values had to
be restored from git history. Since the drift here is provably inert (`prove_inertness.py`
establishes it mechanically, and REFUSES to pass if it ever stops being true), a rebuild
would trade recorded facts for a formality.

THE INDENT HAZARD, which has bitten this repo before. `json.dump(..., indent=2)` over a
file written with `indent=1` reformats EVERY line: a 6-leaf append became +52,572/-52,562
once. So the indent is DETECTED per file from its own second line and reused, and the
result is verified: the byte-diff must contain no deletions beyond the ones the append
structurally requires, and the parsed object must differ from the original ONLY at
`provenance.freshness_acknowledgements`. Both are asserted, per file, before anything is
written to disk.

ORDER OF OPERATIONS MATTERS. Run `ruff format` on the source file BEFORE this script:
`sha256_now` is computed from the file on disk, and a later reformat would silently
invalidate the acknowledgement that was just written. There is no cascade to worry about
here -- these 7 artifacts pin SOURCE files, and no artifact in the set pins another
member of the set, which the script checks rather than assumes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET = "python/carnot/agentic/arc_executable_world_model.py"

ARTIFACTS = [
    "results/experiment_6011_world_model_change_gate_four_arm.json",
    "results/experiment_6012_hidden_state_trust_gate_hole.json",
    "results/experiment_6013_hidden_state_change_gate_closure.json",
    "results/experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b.json",
    "results/outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json",
    "results/outer_loop_arc_generator_concurrency_fix_20260727.json",
    "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json",
]

REASON = (
    "Adds an OPT-IN goal-predicate dedup to the split-induce path (env "
    "CARNOT_ARC_GOAL_DEDUP, DEFAULT OFF). Two pre-existing functions changed -- "
    "`LocalGGUFProposer._combine_world_model` and `LocalGGUFProposer.induce` -- and in both "
    "every added executable statement is inside an `if _goal_dedup_on():` body or behind "
    "`_goal_dedup_on() and ...` short-circuit. Four new module-level helpers are added and "
    "are unreachable with the flag unset. With the variable unset the emitted world model is "
    "byte-identical to the shipped path, so nothing this artifact measured can move."
)

EVIDENCE = (
    "Established mechanically, not by reading the diff: "
    "`results/arc_goal_predicate_shadowing_20260802/prove_inertness.py` diffs the module "
    "against HEAD at AST level and reports `inert_with_flag_unset: true` "
    "(`inertness_proof.json`) via three independent checks -- (1) no new executable statement "
    "in either changed function lies outside a `_goal_dedup_on()` guard; (2) a least-fixed-point "
    "over the call graph shows all four new helpers unreachable with the flag off, modelling "
    "`and` short-circuit explicitly; (3) the flag is False when unset and resolves exactly as "
    "`value.strip() == '1'`, matching every sibling flag in the module. Corroborated by "
    "`tests/python/test_arc_goal_predicate_shadowing.py::test_dedup_off_leaves_the_combined_"
    "output_byte_identical`, which pins the flag-off output byte-for-byte, and by 128 passing "
    "tests across the induce/world-model/goal suites. `git diff` on the module is 182 "
    "insertions / 1 deletion, the single deletion being a replaced docstring line."
)

WHY_NOT_REBUILT = (
    "These artifacts record MEASURED wall-clock timings and live-server witnesses that a "
    "rebuild would overwrite with today's numbers; a previous commit did exactly that to four "
    "artifacts and 132 timing values had to be restored from git history. Inertness here is "
    "established directly and mechanically (see evidence), and the proof script exits non-zero "
    "if the change ever stops being inert, so a rebuild would trade recorded facts for a "
    "formality. The lint's `rebuild_command` remains recorded for anyone wanting "
    "rebuild-strength proof."
)


def _detect_indent(text: str) -> int:
    """The file's own indent width, read from its second line.

    Writing `indent=2` over a file written with `indent=1` reformats every line in it. The
    only safe source of truth is the file itself.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return 2
    m = re.match(r"^( +)", lines[1])
    return len(m.group(1)) if m else 2


def _head_sha() -> str:
    src = subprocess.run(  # noqa: S603
        ["git", "show", f"HEAD:{TARGET}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return hashlib.sha256(src.encode()).hexdigest()


def _commit() -> str:
    return subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _strip_acks(obj: Any) -> Any:
    """A deep copy with `provenance.freshness_acknowledgements` removed, for the diff assertion."""
    clone = json.loads(json.dumps(obj))
    prov = clone.get("provenance")
    if isinstance(prov, dict):
        prov.pop("freshness_acknowledgements", None)
    return clone


def main() -> int:
    proof_path = HERE / "inertness_proof.json"
    if not proof_path.exists():
        print("inertness_proof.json missing -- run prove_inertness.py first")
        return 1
    proof = json.loads(proof_path.read_text())
    if not proof.get("inert_with_flag_unset"):
        print("REFUSING: prove_inertness.py did not certify the change as inert")
        return 1

    sha_now = hashlib.sha256((REPO / TARGET).read_bytes()).hexdigest()
    if sha_now != proof["sha256_now"]:
        print(
            "REFUSING: the module changed since the proof was generated"
            " -- re-run prove_inertness.py"
        )
        return 1
    sha_was = _head_sha()

    # CASCADE CHECK. No member of this set may pin another member as a DEPENDENCY: writing an
    # acknowledgement into A changes A's bytes, which would stale any B whose dependency list
    # records A's sha256, and B's freshly-written acknowledgement would be wrong the moment it
    # landed. Chasing that by hand is where this task's warning about +52,572/-52,562 diffs
    # comes from.
    #
    # The check is STRUCTURAL -- it reads the `{path, sha256}` entries the lint itself uses
    # (`provenance.code` and `provenance.rows_sources`) rather than grepping the file text. A
    # substring scan was tried first and refused the whole run because one artifact NAMES
    # another in the English prose of a pre-existing `evidence` field. That is a mention, not a
    # dependency: prose does not carry a sha256 and cannot go stale. A guard that cannot tell
    # those apart is a guard that gets switched off, which is worse than one that is precise.
    dep_keys = ("code", "rows_sources", "inputs", "dependencies")
    names = {Path(a).name for a in ARTIFACTS}
    for rel in ARTIFACTS:
        prov = json.loads((REPO / rel).read_text()).get("provenance") or {}
        pinned = {
            Path(str(dep.get("path", ""))).name
            for key in dep_keys
            for dep in (prov.get(key) or [])
            if isinstance(dep, dict) and dep.get("sha256")
        }
        clash = (pinned & names) - {Path(rel).name}
        if clash:
            print(
                f"REFUSING: {rel} pins {sorted(clash)} as a dependency -- cascade, handle by hand"
            )
            return 1

    entry = {
        "path": TARGET,
        "sha256_was": sha_was,
        "sha256_now": sha_now,
        "reason": REASON,
        "evidence": EVIDENCE,
        "acknowledged_at_commit": _commit(),
        "acknowledged_date": "2026-08-02",
        "why_not_rebuilt": WHY_NOT_REBUILT,
    }

    for rel in ARTIFACTS:
        path = REPO / rel
        original_text = path.read_text()
        indent = _detect_indent(original_text)
        obj = json.loads(original_text)
        prov = obj.setdefault("provenance", {})
        if not isinstance(prov, dict):
            print(f"REFUSING: {rel} has a non-dict `provenance` -- handle by hand")
            return 1
        acks = prov.setdefault("freshness_acknowledgements", [])
        if any(
            a.get("path") == TARGET and a.get("sha256_now") == sha_now
            for a in acks
            if isinstance(a, dict)
        ):
            print(f"  [skip ] {rel} -- already acknowledged at this sha")
            continue
        acks.append(dict(entry))

        new_text = json.dumps(obj, indent=indent) + "\n"
        # ASSERTION 1: nothing but the acknowledgement changed, semantically.
        if _strip_acks(json.loads(new_text)) != _strip_acks(json.loads(original_text)):
            print(f"REFUSING: {rel} would change outside freshness_acknowledgements")
            return 1
        # ASSERTION 2: the rewrite is an APPEND, not a reformat. Every original line must
        # survive verbatim; if the indent guess were wrong this is what catches it.
        old_lines, new_lines = original_text.splitlines(), new_text.splitlines()
        removed = len(old_lines) - sum(1 for line in old_lines if line in set(new_lines))
        if removed > 2:  # the trailing `}` and its predecessor's comma may legitimately shift
            print(f"REFUSING: {rel} rewrite drops {removed} original lines (indent mismatch?)")
            return 1
        path.write_text(new_text)
        print(f"  [ack  ] {rel} (indent={indent}, +{len(new_lines) - len(old_lines)} lines)")

    print(f"acknowledged {TARGET} {sha_was[:12]} -> {sha_now[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
