"""In-process doc reconciliation — flag-gated alternative to the Haiku call.

The Haiku doc-reconciliation step in the conductor reads the experiment
artifact, maps `honest_verdict` to a status label, and appends to:

  - ops/changelog.md       (always)
  - ops/status.md          (only when a new capability is added)
  - _bmad/traceability.md  (only when new REQ-*/SCENARIO-* appear in the
                            most recent commit's diff)

That step costs ~1-2 minutes of wall time per iteration and burns API
quota on what is mostly mechanical work — the artifact JSON already
contains every field needed.

This module reproduces the mechanical parts in pure Python.  It is wired
into research_conductor.py behind a `--in-process-docs` flag so it can
be A/B-compared against the Haiku path without committing to the new
behaviour.

WHAT IT REPRODUCES (faithfully):
  - The honest-verdict to status-label mapping table from the Haiku
    prompt, verbatim.  No interpretation; the artifact is ground truth.
  - The "always append a 1-line changelog entry" rule.
  - The "only update traceability.md when new REQ-*/SCENARIO-* appear
    in the most recent commit's diff" rule.
  - The "do not delete existing content" invariant — every write is an
    append.

WHAT IT INTENTIONALLY OMITS:
  - The freeform "research finding" / "research priority" prose that
    Haiku writes into changelog entries.  That prose is non-mechanical
    (it requires reading and summarizing the artifact's notes/notes-like
    fields).  The in-process path produces a shorter changelog entry
    that contains the verdict, the mapped status label, key metrics, and
    a pointer to the artifact JSON — enough for honest record-keeping;
    a reader who wants the full narrative reads the artifact.
  - The status.md "is this a new capability?" decision.  Haiku makes a
    nuanced call here.  In-process uses a simple heuristic: append to
    status.md only when the most recent commit added new REQ-* lines
    (i.e., when the spec actually changed).  That is conservative; it
    will under-update status.md compared to Haiku, never over-update it.

WHAT IT DOES NOT TOUCH:
  - scripts/research_conductor.py and research-roadmap.yaml — these are
    explicitly guarded against accidental modification (matching the
    Haiku-path behaviour at conductor.py:2333-2336).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Honest-verdict to status-label mapping. Tokens are matched as substrings
# in the lower-cased honest_verdict string. Order is important: more
# specific categories (failed) are checked before more general ones
# (research finding). The mapping mirrors the Haiku prompt at
# scripts/research_conductor.py:2280-2295.
_FAILED_TOKENS = (
    "timed_out",
    "exception",
    "failed",
    "crash",
)
_BLOCKED_TOKENS = (
    "blocked",
    "gpu_required",
    "synthesis_required",
    "tools_unavailable",
    "tools_not_installed",
    # Note: the bare token "required" is intentionally NOT in this list.
    # It produced false positives on verdicts like "fr11_required_pass".
)
_PARTIAL_TOKENS = (
    "partial",
    "inverted",
    "insufficient",
    "neutral",
    "not_viable",
    "no_improvement",
    "tolerance_exceeded",
    "marginal",
    "still_wrong",
    "no_delta",
    "below",
    "regression",
    "incorrect",
    "negative",
    "flat",
    "plateau",
)
_WIN_TOKENS = (
    "complete",
    "confirmed",
    "viable",
    "closed",
    "resolved",
    "done",
    "fixed",
    "effective",
    "ships",
    "improved",
    "lossless",
    "positive",
    "ready",
    "adequate",
    "operational",
    "published",
    "exceeds_target",
    # Added 2026-04-25 — verdicts observed in .65/.66/.67/.68 that
    # were genuine wins but defaulted to ⚠️ Research Finding because
    # the relevant token wasn't in this list:
    #   .65 Exp 848  "tier1_relay_works_live"      → "works"
    #   .65 Exp 849  "gguf_cache_implemented"      → "implemented"
    #   .66 Exp 856  "deployed"                     → "deployed"
    #   .66 Exp 875  "fr11_tier2_loop_closed"      (already had "closed")
    #   .67 Exp 874  "streaming_cot_wired"          → "wired"
    "wired",
    "implemented",
    "works",
    "deployed",
    "viable_tier",
    # Added 2026-07-01 -- CLAUDE.md "Verdict Terminal-Prefix Discipline" mandates
    # honest_verdict start with complete:/complete_/success:/success_/passed:/passed_/
    # shipped:/shipped_. This list already had "complete" and "ships" (present tense)
    # but never "success" or "shipped" (past tense) or "passed" themselves -- so a
    # clean, unambiguous win like "success: RTL structural logic validated
    # theoretically..." (exp1791) or "success_verified_symbolic_gating" (exp2071) fell
    # through every category to the ⚠️ Research Finding default, DESPITE correctly
    # following the terminal-prefix discipline. Traced live: this is what let a
    # genuinely clean v470 success (exp5128 KAN certificates) get treated as a
    # "prior failure" by FailureLedger, HARD-blocking .471's exp5140 as a doomed
    # rerun it wasn't. Corpus-wide impact measured at fix time: 352 "success:"-
    # prefixed and 13 "shipped:"-prefixed artifacts (of 4160 scanned) were pure
    # oversight misclassifications (zero genuine blocked/failed/partial content) --
    # fixed by this addition. Verdicts that ALSO contain a real negative token (e.g.
    # "complete_..._weak_fit..._blocked_...") are UNAFFECTED: blocked/failed/partial
    # are checked before win-tokens in the cascade above, so a genuinely mixed result
    # correctly stays non-Complete either way -- this fix only rescues the CLEAN wins.
    "success",
    "succeeded",
    "shipped",
    "passed",
)

# Artifact fields that, if present, are pulled into the changelog line as
# the leading "key metric". First match wins; the order reflects which
# metric is most informative for the reader of a recent changelog entry.
_KEY_METRIC_FIELDS = (
    "auc",
    "ood_auc",
    "id_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp_count",
    "fp_count",
    "delta_overall",
    "discrimination_rate",
    "skip_rate",
    "lut_count",
    "lut_count_n32",
    "n_repaired",
    "repair_delta",
    "ising_calls_saved",
    "augmentation_ratio",
    "n_published",
    "models_published",
)


@dataclass
class ReconResult:
    """Outcome of an in-process reconciliation pass."""

    artifact_path: Path | None
    verdict: str
    status_label: str
    changelog_appended: bool
    status_appended: bool
    traceability_rows_added: int
    skipped_reason: str | None = None


def map_status_label(verdict: str) -> str:
    """Map a honest_verdict string to a status-label emoji-prefix.

    The mapping is verbatim from the Haiku prompt (research_conductor.py:2280).
    Tokens are checked as case-insensitive substrings; the first matching
    category wins.

    Returns one of the four status labels used in ops/changelog.md and
    ops/status.md across the project's history:

      - "✅ Complete"          — unambiguous win
      - "⚠️ Research Finding" — partial / inverted / no improvement
      - "⚠️ Blocked"          — prerequisite missing
      - "❌ Failed"            — exception or timeout

    The default for an unrecognised verdict is "⚠️ Research Finding"
    rather than "✅ Complete". This bias is intentional: the
    research_conductor.py docstring at 2241-2265 traces a series of past
    incidents where the Haiku rubber-stamped "✅ Complete" on partial
    results, polluting downstream traceability. Defaulting to research
    finding keeps the project's honest-record discipline intact.
    """
    v = (verdict or "").lower()

    # Blocked must beat Failed — a verdict like "blocked_model_load_failed"
    # contains both "blocked" and "failed" but is structurally a block.
    # The model failed to load *because* a prerequisite was missing; the
    # block is the determining state, not the failure mode that surfaced
    # it. Haiku's behaviour on Exp 811 (.62) confirmed this ordering.
    if any(tok in v for tok in _BLOCKED_TOKENS):
        return "⚠️ Blocked"
    if any(tok in v for tok in _FAILED_TOKENS):
        return "❌ Failed"
    if any(tok in v for tok in _PARTIAL_TOKENS):
        return "⚠️ Research Finding"
    if any(tok in v for tok in _WIN_TOKENS):
        return "✅ Complete"
    return "⚠️ Research Finding"


def classify_artifact(artifact: dict) -> str:
    """Map a full artifact dict to a status label.

    Wraps `map_status_label(verdict)` with two artifact-level upgrades:

    1. **`retro_*_closed` upgrade**: any artifact field whose key starts
       with `retro_` and ends with `_closed` and whose value is a
       truthy string (non-empty, not "false") promotes the verdict to
       "✅ Complete" *unless* the verdict already maps to ❌ Failed
       (a closed retro on top of a failure is contradictory and the
       failure verdict wins to keep the discipline strict).

       Example: .68 Exp 880 `retro_closed: RETRO-HALLUSAE-AUC-BELOW-THRESHOLD`
       — the experiment retired HalluSAE; the verdict text was
       "hallusae_retired_governance_ready" which already mapped to ✅,
       but the rule is general: if `retro_*_closed` is populated, the
       experiment closed something concrete and deserves the win label
       even when the verdict text doesn't carry a win-token.

    2. **Backwards-compat**: when the artifact has no special fields,
       behaviour is identical to `map_status_label(artifact.get('honest_verdict',''))`.

    Use this from reconcile(); leave `map_status_label` unchanged for
    callers that only have the verdict string (tests, external tools).
    """
    verdict = artifact.get("honest_verdict", "")
    base = map_status_label(verdict)
    # Strict-failure-wins: a closed retro on top of a failure is a
    # contradictory shape, prefer the failure for honest-record
    # discipline. Same reasoning as the in-process reconciler's
    # default-to-Research-Finding bias on unrecognised verdicts.
    if base == "❌ Failed":
        return base
    # Promote on retro_*_closed = truthy
    for key, value in artifact.items():
        if not key.startswith("retro_"):
            continue
        if not key.endswith("_closed"):
            continue
        if value in (True, 1):
            return "✅ Complete"
        if isinstance(value, str) and value.strip() and value.strip().lower() != "false":
            return "✅ Complete"
    return base


def find_artifact(task_id: str, results_dir: Path) -> Path | None:
    """Locate the experiment artifact JSON matching a task ID.

    Task IDs in research-roadmap.yaml have the shape
    `expNNN-some-slug` (e.g., `exp819-injection-field-fix`). The
    artifact filename has the shape `experiment_NNN_some_slug.json` in
    the results/ directory. This function pulls the numeric portion out
    of the task ID and returns the matching artifact, preferring the
    most recently modified file when multiple match (which can happen
    on retries).
    """
    match = re.match(r"exp(\d+)", task_id.lower())
    if not match:
        return None
    exp_num = match.group(1)
    candidates = sorted(results_dir.glob(f"experiment_{exp_num}_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_key_metric(artifact: dict) -> str:
    """Pick a single representative metric from the artifact.

    Returns a `"key=value"` string suitable for inclusion in a
    one-line changelog entry, or "" if no recognised metric is present.
    Only scalar (non-collection) values are considered; nested dicts
    and lists are ignored because they don't fit cleanly on one line.
    """
    for key in _KEY_METRIC_FIELDS:
        if key not in artifact:
            continue
        value = artifact[key]
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            continue
        return f"{key}={value}"
    return ""


def _commit_added_lines_in_specs(repo_root: Path) -> list[str]:
    """Return added lines from spec.md files in the most recent commit.

    Used to detect new REQ-* and SCENARIO-* additions, which trigger
    optional traceability.md and status.md updates. Empty list if the
    most recent commit didn't touch spec.md or if git isn't available.
    """
    try:
        result = subprocess.run(
            ["git", "show", "--unified=0", "--no-color", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    added: list[str] = []
    in_spec_file = False
    for raw_line in result.stdout.splitlines():
        if raw_line.startswith("diff --git"):
            in_spec_file = "spec.md" in raw_line
            continue
        if in_spec_file and raw_line.startswith("+") and not raw_line.startswith("+++"):
            added.append(raw_line[1:])
    return added


_REQ_PATTERN = re.compile(r"\*\*(REQ-[A-Z0-9-]+)\*\*")
_SCENARIO_PATTERN = re.compile(r"\*\*(SCENARIO-[A-Z0-9-]+)\*\*")


def extract_new_req_scenario_ids(repo_root: Path) -> tuple[list[str], list[str]]:
    """Return (new REQ ids, new SCENARIO ids) added in the most recent commit.

    Each id is a string like "REQ-VERIFY-095" or "SCENARIO-LEARN-148".
    Duplicates within the same commit are deduplicated. Order is the
    order of first appearance, which keeps the appended traceability
    rows in spec-file declaration order.
    """
    added_lines = _commit_added_lines_in_specs(repo_root)
    req_ids: list[str] = []
    scenario_ids: list[str] = []
    for line in added_lines:
        for match in _REQ_PATTERN.finditer(line):
            rid = match.group(1)
            if rid not in req_ids:
                req_ids.append(rid)
        for match in _SCENARIO_PATTERN.finditer(line):
            sid = match.group(1)
            if sid not in scenario_ids:
                scenario_ids.append(sid)
    return req_ids, scenario_ids


def build_changelog_entry(
    artifact: dict,
    task_title: str,
    status_label: str,
    artifact_relpath: str,
    today_iso: str,
) -> str:
    """Build a single changelog line.

    Format matches the project convention used since milestone .50ish:

      - YYYY-MM-DD: <task title> (<status>) — honest_verdict=<v>; <metric>; <artifact>

    The entry intentionally fits on one line. Anyone wanting more
    detail reads the linked artifact JSON.
    """
    verdict = artifact.get("honest_verdict", "unspecified")
    pieces = [f"honest_verdict={verdict}"]
    metric = extract_key_metric(artifact)
    if metric:
        pieces.append(metric)
    pieces.append(artifact_relpath)
    return f"- {today_iso}: {task_title} ({status_label}) — " + "; ".join(pieces) + "\n"


def append_line(target: Path, line: str) -> bool:
    """Append a single line to a file. Returns True on success.

    Refuses to create the file if it doesn't exist — this matches the
    Haiku rule that doc reconciliation only adds to existing docs.
    Ensures a trailing newline before the appended line so successive
    runs don't run together.
    """
    if not target.exists():
        return False
    existing = target.read_bytes()
    needs_newline = bool(existing) and not existing.endswith(b"\n")
    with target.open("ab") as fh:
        if needs_newline:
            fh.write(b"\n")
        fh.write(line.encode("utf-8"))
    return True


def reconcile(
    task: dict,
    repo_root: Path = PROJECT_ROOT,
    *,
    today: str | None = None,
) -> ReconResult:
    """Run a full in-process doc reconciliation pass for one experiment.

    The conductor calls this in place of (or alongside) the Haiku
    `run_agent` call when the `--in-process-docs` flag is set. Returns
    a `ReconResult` describing what was appended; the conductor uses
    that to log the outcome and to decide whether to commit.
    """
    today_iso = today or datetime.now(UTC).strftime("%Y-%m-%d")
    task_id = task.get("id", "")
    task_title = task.get("title", "(untitled task)")

    artifact_path = find_artifact(task_id, repo_root / "results")
    if not artifact_path:
        return ReconResult(
            artifact_path=None,
            verdict="",
            status_label="",
            changelog_appended=False,
            status_appended=False,
            traceability_rows_added=0,
            skipped_reason=f"no artifact found for task id {task_id!r}",
        )

    try:
        artifact = json.loads(artifact_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return ReconResult(
            artifact_path=artifact_path,
            verdict="",
            status_label="",
            changelog_appended=False,
            status_appended=False,
            traceability_rows_added=0,
            skipped_reason=f"artifact unreadable: {exc}",
        )

    verdict = artifact.get("honest_verdict", "")
    # Use classify_artifact (not just map_status_label) so retro_*_closed
    # fields can promote a verdict that didn't carry a win-token in its
    # text. Without this, .65 Exp 849 (gguf_cache_implemented +
    # retro_closed=RETRO-GGUF-CACHE-IMPORT) would still be undercounted.
    status_label = classify_artifact(artifact)
    artifact_rel = str(artifact_path.relative_to(repo_root))

    # 1. Always append to ops/changelog.md.
    changelog_line = build_changelog_entry(
        artifact,
        task_title,
        status_label,
        artifact_rel,
        today_iso,
    )
    changelog_appended = append_line(
        repo_root / "ops" / "changelog.md",
        changelog_line,
    )

    # 2. Conditionally update _bmad/traceability.md and ops/status.md.
    # Both depend on whether the most recent commit added new REQ or
    # SCENARIO ids in any spec.md file.
    new_reqs, new_scenarios = extract_new_req_scenario_ids(repo_root)
    traceability_rows_added = 0
    status_appended = False

    if new_reqs or new_scenarios:
        # Pick a traceability status string consistent with the verdict.
        # Wins → Implemented; partials/blocked → Implemented-Partial.
        # The Haiku prompt at scripts/research_conductor.py:2306 spells
        # this rule out — never mark Implemented when the source
        # experiment is partial or blocked.
        if status_label == "✅ Complete":
            trace_status = "Implemented"
        elif status_label == "⚠️ Blocked":
            trace_status = "Scaffolding"
        else:
            trace_status = "Implemented-Partial"
        trace_lines: list[str] = []
        for rid in new_reqs:
            trace_lines.append(f"| {rid} | {task_title} | {trace_status} | {artifact_rel} |\n")
        for sid in new_scenarios:
            trace_lines.append(f"| {sid} | {task_title} | {trace_status} | {artifact_rel} |\n")
        trace_path = repo_root / "_bmad" / "traceability.md"
        if trace_path.exists():
            for line in trace_lines:
                append_line(trace_path, line)
            traceability_rows_added = len(trace_lines)

        # Touch ops/status.md only when the experiment is a clear win.
        # Conservative: never mark a partial / blocked experiment as a
        # delivered capability in status.md.
        if status_label == "✅ Complete":
            status_line = (
                f"| {today_iso} | {task_title} | {status_label} | {verdict} | {artifact_rel} |\n"
            )
            status_appended = append_line(
                repo_root / "ops" / "status.md",
                status_line,
            )

    return ReconResult(
        artifact_path=artifact_path,
        verdict=verdict,
        status_label=status_label,
        changelog_appended=changelog_appended,
        status_appended=status_appended,
        traceability_rows_added=traceability_rows_added,
        skipped_reason=None,
    )
