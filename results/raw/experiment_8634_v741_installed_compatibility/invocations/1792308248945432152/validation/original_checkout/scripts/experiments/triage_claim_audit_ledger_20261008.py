"""One-off: record 2026-10-08 outer-loop triage dispositions in ops/audit-findings-ledger.md.

Edits only the Disposition and Note cells of rows listed below. Never removes or reorders a row
(the ledger is append-only). Run with --apply to write; without it, prints what would change.
"""

from __future__ import annotations

import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "ops" / "audit-findings-ledger.md"

CIRCULAR = {
    "6596",
    "6888",
    "6919",
    "6913",
    "6914",
    "6926",
    "7203",
    "7215",
    "7267",
    "7284",
    "7298",
    "7324",
    "7325",
    "7361",
    "7377",
    "7403",
    "7468",
    "7482",
    "7534",
    "7580",
    "7666",
    "7672",
    "7680",
    "7708",
    "7817",
    "7825",
    "8085",
    "8213",
}
LABEL_CONTRADICTS = {"7029", "7098", "7130"}
BLOCKED_NO_CLAIM = {"6633"}
NEEDS_REVIEW = {"6587", "6654", "6798", "6813", "6854", "7028"}

NOTE_CIRCULAR = (
    "2026-10-08 outer-loop triage, artifact-level evidence only. The audit's original complaint text "
    "is no longer recoverable: the report is regenerated at every milestone close. Checked live: the "
    "artifact declares `verdict_class: circular_positive` and `verifier_is_oracle: true`, so it is "
    "already labelled a circular result. Disposition: ACCEPTED as a framing concern. Treat it as "
    "execution-grounded: not headline-eligible and not evidence of a benefit. Artifact left unedited "
    "per never-prune. NOT checked: whether the verdict wording goes further than the declared class."
)
NOTE_CONTRADICTS = (
    "2026-10-08 outer-loop triage. Finding stands on the artifact's own text: `verdict_class` is "
    "`positive` but the verdict string itself says the scientific delta is zero, empty, or possibly "
    "null. The label records that the mechanics completed, not that a benefit was found. Do not cite "
    "as a positive result. Artifact left unedited per never-prune."
)
NOTE_BLOCKED = (
    "2026-10-08 outer-loop triage. The verdict is `blocked_...: infrastructure checks failed; no "
    "model-quality claim` with `verdict_class: blocked`, so the artifact asserts no result that its "
    "own data could refute. WONTFIX. The original complaint text is not recoverable (report "
    "regenerated), so this rests on the verdict wording alone."
)
NOTE_REVIEW = (
    "2026-10-08 outer-loop triage: NOT dispositioned. The artifact is not self-labelled circular, or "
    "its claim is a real positive or a contract result, and the original complaint text is not "
    "recoverable. Needs a per-artifact review against its own data. Left OPEN on purpose."
)


def number(artifact: str) -> str:
    return artifact.split("_")[1]


def main() -> int:
    apply = "--apply" in sys.argv
    out, changed, counts = [], 0, {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("| 20") and "| experiment_claim_audit |" in line:
            cells = line.rstrip("\n").split("|")
            # cells: ['', first, audit, artifact, verdict, disposition, note, '']
            if (
                len(cells) >= 8
                and cells[5].strip() == "OPEN"
                and cells[3].strip().endswith(".json")
            ):
                n = number(cells[3].strip())
                new = None
                if n in CIRCULAR:
                    new = ("ACCEPTED", NOTE_CIRCULAR, "circular_accepted")
                elif n in LABEL_CONTRADICTS:
                    new = ("ACCEPTED", NOTE_CONTRADICTS, "label_contradicts_accepted")
                elif n in BLOCKED_NO_CLAIM:
                    new = ("WONTFIX", NOTE_BLOCKED, "blocked_wontfix")
                elif n in NEEDS_REVIEW:
                    new = ("OPEN", NOTE_REVIEW, "needs_review_open_with_note")
                if new:
                    cells[5] = f" {new[0]} "
                    cells[6] = f" {new[1]} "
                    line = "|".join(cells) + "\n"
                    changed += 1
                    counts[new[2]] = counts.get(new[2], 0) + 1
        out.append(line)
    print("rows changed:", changed, counts)
    if apply:
        LEDGER.write_text("".join(out), encoding="utf-8")
        print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
