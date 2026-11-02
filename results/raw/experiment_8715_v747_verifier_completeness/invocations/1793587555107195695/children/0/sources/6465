#!/usr/bin/env python3
"""Merge-preserve for analyzer rebuilds: carry hand-authored keys forward.

WHY THIS EXISTS (REQ-OPS-REBUILD-PRESERVE-1). The `scripts/analyze_*.py`
rebuilders regenerate result artifacts wholesale from saved rows, and
`scripts/artifact_freshness_lint.py` MANDATES a rebuild when a declared
code dependency changes. So every freshness-mandated rebuild silently
deleted the record left by the previous one. Three independent hits on
2026-08-21: `rebuild_note_20260731_split_enable` lost from the early-stop
sweep artifact; 26 freshness acknowledgements lost across artifacts (the
blocking finding of that change's own adversarial review); 3-4
`rebuild_note_*` keys per artifact lost twice in one evening. The keys
destroyed are the project's own never-prune apparatus — the notes that
explain a correction and the attestations that a drift was checked and
found inert.

THE RULE. A key the analyzer generated this run wins. A key present in
the old artifact and absent from the new payload is carried forward
verbatim. The failure being fixed is silent deletion, so carry is the
default direction (same choice as dae_mutmap's `_carry_triage`: present
in new wins; absent from new but present in old is carried, not dropped).

THE OWNED SET IS DERIVED, NOT HAND-LISTED. The keys a rebuild may replace
are exactly the keys the generated payload contains, plus an explicit
`retired_keys` argument for deliberate drops. A hand-maintained
protected-name list is the pattern-narrower-than-concept bug this project
keeps hitting (`inference_substrate_correction_note` is the field a
literal `corrigendum` pattern missed; this repo's freshness hook's own
hand-maintained `files:` regex missed 3 of its 5 dependencies). A
hand-maintained OWNED list would drift the same way, so ownership is
stated by the code's own output, mechanically true every run.

THE ONE NAMED NESTED FIELD, AND WHY IT IS NOT A PATTERN LIST.
`provenance` is regenerated wholesale on every rebuild, and correctly so:
its `sha256` / `git_head` / `rebuild_command` fields describe the CURRENT
build. `provenance.freshness_acknowledgements` is different in kind — an
append-only audit log of past human judgment calls, defined by
`artifact_freshness_lint.py` — so it is carried into the regenerated
provenance. This is one structurally-known field of the freshness system
itself, not an open class; the open class (hand keys anywhere at top
level, whatever their names) is covered by the generic carry above. Any
OTHER old sub-key that a regenerated dict key drops is reported on
stderr: deletion there may be correct, silent deletion is not.

FAIL CLOSED, deliberately inverting the older ack-only helper
(`analyze_scored_path_lever_ab.preserve_freshness_acknowledgements`,
which swallowed every error): when the existing artifact exists but
cannot be read or parsed, `merge_preserve_with_file` RAISES instead of
letting the caller overwrite. A rebuild is re-runnable at zero cost after
a human looks; the keys under an unreadable file are not recoverable
after an overwrite. A caller who really means "replace whatever is
there" moves the old file aside first — that act states the intent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The one nested append-only audit field of the freshness system (see
# docstring). Name matches artifact_freshness_lint.py's reader.
_ACK_KEY = "freshness_acknowledgements"


class MergeRefusedError(RuntimeError):
    """The existing artifact exists but cannot be read; refusing to overwrite."""


def merge_preserve(existing: dict, generated: dict, retired_keys: tuple = ()) -> dict:
    """Merge a regenerated payload over an existing artifact, carry-first.

    * generated keys win;
    * old-only top-level keys carry, unless named in `retired_keys`;
    * `provenance.freshness_acknowledgements` carries into a regenerated
      provenance when the new payload does not itself provide one;
    * old-only sub-keys inside regenerated dict keys are dropped WITH a
      stderr notice naming them (never silently).
    """
    merged: dict = {}
    for key, value in existing.items():
        if key in generated or key in retired_keys:
            continue
        merged[key] = value  # carry-forward: the fix for silent deletion
    merged.update(generated)

    # Nested audit-log carry (REQ-OPS-REBUILD-PRESERVE-1 rule 3).
    old_prov = existing.get("provenance")
    new_prov = merged.get("provenance")
    if isinstance(old_prov, dict) and isinstance(new_prov, dict):
        acks = old_prov.get(_ACK_KEY)
        if acks and _ACK_KEY not in new_prov:
            new_prov[_ACK_KEY] = acks

    # Stated (never silent) deletion inside regenerated dict keys
    # (rule 4). Deleting a stale current-build fact is correct; the
    # notice exists so a future hand-authored NESTED key cannot vanish
    # without a trace the way the top-level ones did.
    for key, new_value in generated.items():
        old_value = existing.get(key)
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            dropped = [
                sub
                for sub in old_value
                if sub not in merged[key] and not (key == "provenance" and sub == _ACK_KEY)
            ]
            if dropped:
                print(
                    f"[merge-preserve] NOTE: rebuild drops {key}.{{{', '.join(sorted(dropped))}}} "
                    "(old sub-keys the new payload does not regenerate). If any of these was "
                    "hand-authored, restore it and file it top-level or as an acknowledgement.",
                    file=sys.stderr,
                )
    return merged


def merge_preserve_with_file(out_path: Path, generated: dict, retired_keys: tuple = ()) -> dict:
    """merge_preserve against whatever is on disk at `out_path`.

    Missing file -> first build, generated returned as-is. Unreadable or
    non-dict file -> MergeRefusedError (fail closed; see module
    docstring for why this inverts the older helper's fail-open).
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return dict(generated)
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MergeRefusedError(
            f"REFUSING rebuild over {out_path}: the existing artifact cannot be parsed "
            f"({type(exc).__name__}: {exc}). Overwriting would destroy any hand-authored "
            "keys it holds with no trace. Inspect it; move it aside to state the intent "
            "to replace it."
        ) from exc
    if not isinstance(existing, dict):
        raise MergeRefusedError(
            f"REFUSING rebuild over {out_path}: existing artifact is "
            f"{type(existing).__name__}, not an object; cannot merge-preserve."
        )
    return merge_preserve(existing, generated, retired_keys=retired_keys)
