"""Failure-Ledger v2 — Issue 1 (id-based fail counting), Issue 5 (≥2 keyword
threshold), and dispatch-time exclusion-manifest enforcement.

This module exists because three milestone-86 regressions slipped past the
v1 ledger:

Issue 1 — Title-prefix failure-count inheritance.
  ``pick_next_task`` parses the conductor log for failure counts using the
  task's title (truncated to 50 chars) as the key.  When .85's planner
  re-proposed a previously-retired .84 experiment with the same title
  (exp1080 SemEnergy → exp1096 SemEnergy, exp1086 N-Queens → exp1097
  N-Queens), the new attempt inherited the prior milestone's fail count
  and was retired BEFORE the planner's prior_failures discipline could
  even fire.  The fix here counts failures by experiment_id (extracted
  from the log entry's `id=expNNN-...` marker) when available, and falls
  back to title-prefix only for legacy log entries that pre-date this
  schema change.

Issue 5 — Coarse keyword matcher false-positives.
  The ledger v1 keyword overlap requires only 1 substantive token shared
  between two task titles to declare them a scope match.  The .85
  preflight blocked exp1106 Phase-1a Adversarial Verifier on 18 prior
  failures, all of which only shared the single token "verifier" or
  "adversarial".  That is below the noise floor — every future verifier
  experiment would block the planner.  The fix here requires ≥2 distinct
  scope-vocabulary keywords from the task title to overlap with a prior
  failure title before flagging the rerun.  Word tokenization is exact
  (split on whitespace, lowercase, strip ASCII punctuation), and
  scaffolding tokens (live, phase, audit, v1, ...) are filtered out so
  that scaffolding alone does not satisfy the threshold.

Manifest dispatch enforcement.
  ``ops/exclusion_manifest.yaml`` is the human-authored retirement
  ledger.  exp906 was retired in milestone .80 yet appeared as the
  slowest experiment for three consecutive milestones (.83/.84/.85)
  because the conductor only consulted the JSON manifest at planning
  time, not at dispatch time, and the YAML manifest was never consulted
  at dispatch time at all.  The fix here adds a dispatch-time check
  that loads BOTH the JSON manifest (legacy, integer ids) and the YAML
  manifest (newer, mixed integer + string-scope ids) and skips any task
  whose extracted experiment_id appears in either.

Why a separate module: the v1 ``failure_ledger.py`` still owns the
scope-signature + prior_failures discipline, which is a different
mechanism (it gates Sonnet spawn AFTER pick_next_task already chose the
task).  This v2 module owns the dispatch-time signals — fail counting,
exclusion check — that run BEFORE Sonnet is even considered.  Keeping
them split means each pass of conductor logic is independently testable
and the existing failure_ledger.py tests do not have to change.

Spec: openspec/change-proposals/failure-ledger-v2-and-planner-discipline.md
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Regex for the id marker the conductor's `log_step` writes into the details
# column once Issue 1 is wired in: `[id=expNNN-slug]` somewhere in details.
# Tolerant of surrounding whitespace, quoting, and other prose.
_ID_MARKER_RE = re.compile(r"\[id=([a-zA-Z0-9_\-]+)\]")

# Statuses that count as a failure for the per-task fail budget.  Lifted
# verbatim from pick_next_task so the v2 counter agrees with the v1 logic
# on the failure side and only differs in the bucketing key.
_FAIL_STATUSES = frozenset(
    {
        "FAIL",
        "REVERT",
        "SKIP",
        "NOOP",
        "GATE_BLOCK",
        "DOOMED_RERUN_BLOCK",
    }
)
_SUCCESS_STATUSES = frozenset({"OK"})

# Scaffolding words that should not count toward the keyword-overlap
# threshold.  Same spirit as failure_ledger.py's _SCAFFOLDING_TOKENS but
# scoped to title-word vocabulary (no version markers — those almost
# never appear as standalone words in titles, they appear in slugs).
_TITLE_SCAFFOLDING = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "live",
        "fix",
        "fixes",
        "test",
        "tests",
        "audit",
        "phase",
        "tier",
        "exp",
        "experiment",
        "milestone",
        "retro",
        "retrospective",
        "complete",
        "implementation",
        "deliverable",
    }
)


def _tokenize_title(title: str) -> set[str]:
    """Split a title into substantive lower-case word tokens.

    Strategy: lowercase, strip ASCII punctuation, split on whitespace,
    drop tokens shorter than 3 characters and any in
    ``_TITLE_SCAFFOLDING``.  Returns a set so that duplicate words and
    word order do not affect the overlap count.

    Why ≥3 chars: 1- and 2-char tokens (a, of, v1, v2) are noise.  The
    substantive vocabulary in our titles starts at 3 chars (kan, sat,
    fpga, kv260, jepa).
    """
    if not title:
        return set()
    cleaned = title.lower()
    # Replace punctuation with spaces so "verifier," and "verifier" tokenize the same
    cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = {t for t in cleaned.split() if len(t) >= 3 and t not in _TITLE_SCAFFOLDING}
    return tokens


def keywords_overlap(title_a: str, title_b: str, min_count: int = 2) -> bool:
    """True iff the two titles share at least ``min_count`` substantive tokens.

    This is the Issue 5 fix.  The v1 ledger declared a doomed rerun
    on a single shared token, which over-fired badly on broad
    vocabulary like "verifier" or "adversarial".  Requiring ≥2
    distinct substantive tokens raises the bar to a meaningful
    cross-cut without losing genuine scope reruns (a real
    code-repair rerun shares "code" + "repair", a real ising-sampler
    rerun shares "ising" + "sampler", etc.).
    """
    a = _tokenize_title(title_a)
    b = _tokenize_title(title_b)
    return len(a & b) >= min_count


def parse_log_entry(line: str) -> tuple[str | None, str, str]:
    """Parse one conductor-log line into (experiment_id, title_prefix, status).

    Conductor log format (since 2026-04 schema):
        | <timestamp> | <title[:50]> | <status> | <details> |

    Issue 1 schema extension: ``log_step`` now appends ``[id=<exp_id>]``
    to the details column when the caller knows the task id.  Returns
    ``experiment_id=None`` for legacy entries that lack the marker, in
    which case downstream code falls back to title-prefix bucketing.
    """
    parts = line.split("|")
    if len(parts) < 5:
        return None, "", ""
    title_prefix = parts[2].strip()
    status = parts[3].strip()
    details = parts[4].strip() if len(parts) > 4 else ""
    m = _ID_MARKER_RE.search(details)
    exp_id = m.group(1) if m else None
    return exp_id, title_prefix, status


def count_failures_for_task(
    task: dict,
    log_lines: list[str],
    success_resets: bool = True,
) -> int:
    """Return the consecutive-failure count for ``task`` in the given log lines.

    Counting rule (preserves v1 semantics):
      * ``OK`` resets the counter to 0 (assumes the task ran cleanly).
      * Any status in ``_FAIL_STATUSES`` increments.
      * Other statuses (e.g., ``ESCALATE_OPUS``) are ignored.

    Bucketing rule (the Issue 1 fix):
      * Each log line is parsed once via ``parse_log_entry``.
      * If the line carries an ``[id=...]`` marker, it is bucketed to
        that experiment_id.  A line bucketed to the task's own id
        increments / resets the task's counter.
      * If the line lacks the marker (LEGACY), it is bucketed to its
        title prefix.  A legacy line whose title prefix matches the
        task's title prefix counts toward the task — preserving the
        pre-Issue-1 behaviour for older log entries.
      * Modern lines for OTHER tasks (different id) DO NOT count
        toward this task even if their title prefix happens to
        collide.  This is the entire point of the fix.

    The caller is responsible for scoping ``log_lines`` to the current
    milestone (e.g., everything after the most recent ``Milestone X
    activated`` marker) — that policy lives in ``pick_next_task`` and
    is preserved across this refactor.
    """
    task_id = (task.get("id") or "").strip()
    title_prefix = (task.get("title") or "")[:50].strip()
    count = 0
    for line in log_lines:
        exp_id, line_title, status = parse_log_entry(line)
        if exp_id is not None:
            # Modern entry — bucket strictly by id.
            if exp_id != task_id:
                continue
        else:
            # Legacy entry — fall back to title-prefix match.
            if line_title != title_prefix:
                continue
        if status in _SUCCESS_STATUSES and success_resets:
            count = 0
        elif status in _FAIL_STATUSES:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Manifest dispatch enforcement
# ---------------------------------------------------------------------------

# Task id formats we accept: "exp906-foo-bar", "exp906_foo_bar", or just
# the integer "906" (rare).  We also accept a plain integer id field.
_EXP_ID_RE = re.compile(r"^exp(\d+)[-_]")
_TITLE_EXP_RE = re.compile(r"\bExp(?:eriment)?\s+(\d+)\b", re.IGNORECASE)


def extract_experiment_id(task: dict) -> int | None:
    """Pull an integer experiment id out of a task dict.

    Tries the structured ``id`` field first (``exp906-...`` → 906),
    then the title (``Experiment 906 — ...`` → 906) as a fallback.
    Returns None if no integer id can be extracted; callers treat
    None as "do not exclude" so that malformed task dicts cannot be
    accidentally retired by manifest enforcement.
    """
    raw_id = task.get("id")
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        m = _EXP_ID_RE.match(raw_id.strip())
        if m:
            return int(m.group(1))
    title = task.get("title") or ""
    m = _TITLE_EXP_RE.search(title)
    if m:
        return int(m.group(1))
    return None


def _yaml_manifest_excluded_ids(manifest_path: Path) -> set[int]:
    """Walk the YAML exclusion manifest and return all integer experiment ids.

    The YAML file has multiple top-level shapes accumulated over time:
      * ``retired:`` — list of dicts with ``experiment_id`` ints.
      * ``retired_experiments:`` — list of dicts with ``experiment_id``
        ints OR scope-name strings (we drop the strings — they're
        handled elsewhere).
      * Top-level loose ``- experiment_id: ...`` entries (the file
        accumulated some malformed entries over .68–.79).

    We tolerate the malformed entries — anything that fails to parse
    as an integer experiment_id is silently skipped.  The function
    never raises; manifest enforcement should fail-open, not fail-
    closed, because a malformed manifest line should not stall the
    whole conductor.
    """
    if not manifest_path.exists():
        return set()
    try:
        text = manifest_path.read_text()
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return set()
    ids: set[int] = set()

    def _ingest_list(entries) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            exp_id = entry.get("experiment_id")
            if isinstance(exp_id, int):
                ids.add(exp_id)
            elif isinstance(exp_id, str) and exp_id.isdigit():
                ids.add(int(exp_id))
            # entries with experiment_scope-only or string ids are not
            # numeric retirements; they're handled by other code paths.

    if isinstance(data, dict):
        _ingest_list(data.get("retired"))
        _ingest_list(data.get("retired_experiments"))
        # Some YAML files have additional sections — ingest any list value
        # whose entries look like exclusion entries.
        for key, value in data.items():
            if key in ("retired", "retired_experiments"):
                continue
            if isinstance(value, list):
                _ingest_list(value)
    elif isinstance(data, list):
        _ingest_list(data)
    return ids


def is_excluded_by_manifest(
    task: dict,
    yaml_manifest_path: Path | None = None,
    extra_excluded_ids: set[int] | None = None,
) -> tuple[bool, str]:
    """Dispatch-time exclusion check against the YAML manifest.

    Returns ``(True, reason)`` if the task's extracted experiment id is
    listed in the YAML manifest at ``yaml_manifest_path`` (default:
    ``ops/exclusion_manifest.yaml``).  Returns ``(False, reason)``
    otherwise.

    The optional ``extra_excluded_ids`` set lets callers union in ids
    from a JSON manifest or a runtime cache without re-reading the
    YAML on every dispatch decision.  Exclusion is the union of both
    sources — if either lists the id, the task is excluded.

    Fail-open: if the manifest is missing or unreadable, returns
    ``(False, "manifest unavailable")``.  Exclusion is a performance
    optimisation, not a safety gate, so a manifest error must not
    stall research.
    """
    if yaml_manifest_path is None:
        yaml_manifest_path = PROJECT_ROOT / "ops" / "exclusion_manifest.yaml"
    exp_id = extract_experiment_id(task)
    if exp_id is None:
        return False, "no id parsed"
    yaml_ids = _yaml_manifest_excluded_ids(yaml_manifest_path)
    extras = extra_excluded_ids or set()
    if exp_id in yaml_ids or exp_id in extras:
        source = "yaml" if exp_id in yaml_ids else "extra"
        return True, f"exp_id={exp_id} in {source} exclusion manifest"
    return False, "not excluded"
