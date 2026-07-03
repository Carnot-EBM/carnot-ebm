"""Standalone poison-test-cascade pretest triage.

WHY THIS EXISTS
---------------
The conductor runs a *smart-subset* pytest gate BEFORE launching each task in a
milestone (``scripts/research_conductor.py::run_tests``). That gate is SHARED
across every subsequent task, not scoped per-task. So a SINGLE red test poisons
the gate for the whole milestone tail: every later task reads "Pre-tests
failing, self-heal failed: 1 failed, N passed" and is SKIPped / GATE_BLOCKed.

One recurring way a red test appears: a task ships its own
``tests/python/test_experiment_NNNN*.py`` whose assertion READS a
``results/experiment_NNNN*.json`` deliverable that a SIBLING module in the SAME
task's ``main()`` would have written -- but the task hit a wall-clock timeout (or
otherwise died) BEFORE ``main()`` ran, so the file never landed and the test
fails with ``FileNotFoundError``. Milestone ``2026.07.475`` lost 10 of 12 tasks
to exactly this: ``test_experiment_5182_...::test_ondisk_deliverable_is_valid``
read a not-yet-produced deliverable and cascade-blocked the rest of the run
(see ``ops/conductor-log.md`` 2026-07-03 08:39-09:42 UTC).

WHAT THIS DOES
--------------
This module is a NARROW detector for that ONE signature. Given the pytest failure
output plus the current ``research-roadmap.yaml`` it returns which failing test
node ids are "waiting on an as-yet-unproduced declared deliverable" (and which are
NOT -- those keep blocking, because a genuine regression must never be masked).
For a matched node it recommends / applies a self-expiring ``xfail`` marker: the
test still RUNS and still ASSERTS (per the project's Tests-Must-Run-And-Assert
rule), but its failure is *expected* WHILE the deliverable is absent. The instant
the deliverable lands, the ``xfail`` condition flips to inactive and the test runs
live again -- so the marker cannot silently outlive the problem it triages.

This is the same pattern as ``scripts/retro_timing_fallback.py``: a new standalone
module the conductor imports, so no task (including the one that built this) has to
edit ``scripts/research_conductor.py`` directly.

HOW TO WIRE THIS IN
-------------------
In ``scripts/research_conductor.py::run_tests`` (the smart-subset pretest gate),
the failure branch currently reads (around lines 1504-1519)::

    elif failed_names:
        logger.warning("Pre-test failures (showing %d of %d): %s", ...)
        if not full:
            _handle_pretest_poison(failed_names)

Wrap that branch so the deliverable-read poison is triaged BEFORE the blunt
consecutive-fail quarantine, and re-run the subset once if every failure is an
explained poison-cascade wait::

    elif failed_names:
        logger.warning("Pre-test failures (showing %d of %d): %s", ...)
        if not full:
            from scripts.pretest_triage import detect_poison_cascade, apply_xfail
            triage = detect_poison_cascade(stdout or stderr, PROJECT_ROOT)
            if triage.matched and triage.all_failures_explained:
                apply_xfail(triage, PROJECT_ROOT)          # self-expiring xfail
                return run_tests(full=full)                # re-run once; expect green
            _handle_pretest_poison(failed_names)           # fall back to the old guard

``detect_poison_cascade`` is a pure function (no side effects); ``apply_xfail`` is
the only function that edits test files, and it only ever ADDS a conditional
``xfail`` marker -- it never removes a test or its assertions. Both are optional:
skipping the wiring leaves the pre-existing consecutive-fail quarantine guard as
the (slower, coverage-losing) backstop.

This module NEVER edits ``scripts/research_conductor.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re


MODULE_REL_PATH = "scripts/pretest_triage.py"

# A ``results/...json`` deliverable path, matched wherever it appears in the
# pytest output. It is a substring of the absolute path pytest prints in a
# ``FileNotFoundError`` (``/repo/results/experiment_5182_....json`` contains
# ``results/experiment_5182_....json``), so anchoring on ``results/`` captures the
# repo-relative tail from both absolute and relative renderings. The character
# class is exactly the set that appears in these deliverable names.
_RESULT_PATH_RE = re.compile(r"results/[A-Za-z0-9_./-]+\.json")

# Tokens that mean "the failure was caused by the deliverable being MISSING /
# UNREADABLE", not by a real assertion on already-present data. Requiring one of
# these is a precision guard: a test that merely mentions a results path in an
# unrelated repr, or that fails a numeric assertion on a file that DOES exist,
# will not carry one of these markers and therefore will NOT be triaged.
_FILE_ABSENCE_MARKERS = (
    "FileNotFoundError",
    "No such file or directory",
    "[Errno 2]",
    "JSONDecodeError",  # a partially-written / empty deliverable read mid-flight
)

# The pytest FAILURES / ERRORS per-test header, e.g.
# ``______________________ test_ondisk_deliverable_is_valid ______________________``
# The captured group is the test's display name (function, ``Class.method``, or a
# parametrized ``test_x[case]``).
_FAILURE_HEADER_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}\s*$")

# A short-test-summary line: ``FAILED tests/python/test_x.py::test_y`` (optionally
# ``ERROR ...``; optionally a ``- reason`` suffix when ``-ra`` is active).
_SUMMARY_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(tests/[^\s:]+\.py)(::[^\s]+)?(?:\s+-\s+.*)?$")

# A traceback location line naming the failing test file: ``tests/python/test_x.py:12: in ...``
_TRACEBACK_FILE_RE = re.compile(r"(tests/[^\s:]+\.py):\d+:")


@dataclass(frozen=True)
class TriageMatch:
    """One failing test confirmed to be waiting on a not-yet-produced deliverable."""

    test_file: str
    nodeid: str
    deliverable_path: str
    producing_task_id: str
    reason: str


@dataclass(frozen=True)
class TriageResult:
    """The outcome of triaging one pretest failure blob against the roadmap."""

    matched: bool
    matches: tuple[TriageMatch, ...] = ()
    unmatched_failures: tuple[str, ...] = ()
    roadmap_milestone: str | None = None

    @property
    def all_failures_explained(self) -> bool:
        """True iff EVERY failing test is an explained poison-cascade wait.

        The conductor should only treat the gate as recoverable-by-xfail when this
        is True. If any unmatched failure remains, there is a genuine regression
        mixed in and the gate must still block.
        """
        return self.matched and not self.unmatched_failures


@dataclass
class _FailureEvidence:
    """Per-failing-test evidence gathered from the pytest output blob."""

    nodeid: str
    test_file: str
    result_paths: list[str] = field(default_factory=list)
    has_absence_marker: bool = False


def _normalize_rel_path(path_value: object, repo_root: Path) -> str | None:
    """Return a repo-relative POSIX path string, or None if unusable.

    Absolute paths under ``repo_root`` are made relative; a stray ``./`` prefix is
    stripped. This lets a roadmap ``deliverable`` and a path scraped from a
    traceback compare equal regardless of how each was written.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    raw = Path(path_value.strip())
    if raw.is_absolute():
        try:
            return raw.relative_to(repo_root).as_posix()
        except ValueError:
            return raw.as_posix()
    return raw.as_posix().lstrip("./") or None


def _load_yaml(path: Path) -> dict:
    """Load a YAML mapping, returning ``{}`` on any error (missing file, parse)."""
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_roadmap_deliverables(
    repo_root: Path | str = Path.cwd(),  # noqa: B008 - matches retro_timing_fallback.py idiom
) -> tuple[dict[str, str], str | None]:
    """Map each declared ``results/*.json`` deliverable to the task id that owns it.

    Reads ``research-roadmap.yaml`` (the CURRENT active milestone). Returns a
    ``(deliverable_path -> task_id)`` dict plus the milestone string. A task with
    no deliverable, or a non-results deliverable, is skipped -- only ``results/``
    JSON artifacts can be the subject of the deliverable-read poison.
    """
    root = Path(repo_root)
    roadmap = _load_yaml(root / "research-roadmap.yaml")
    milestone = roadmap.get("milestone")
    milestone_str = str(milestone) if milestone is not None else None
    mapping: dict[str, str] = {}
    tasks = roadmap.get("tasks", [])
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            rel = _normalize_rel_path(task.get("deliverable"), root)
            if rel is None or not _RESULT_PATH_RE.fullmatch(rel):
                continue
            mapping.setdefault(rel, str(task.get("id") or Path(rel).stem))
    return mapping, milestone_str


def is_experiment_specific_test(test_file: str) -> bool:
    """Whether a test file is safe to auto-xfail (experiment-specific only).

    Mirrors the conductor's ``_is_auto_quarantinable`` SAFETY rule: only
    ``tests/python/test_experiment_*.py`` / ``test_exp*.py`` may be triaged. A
    core/shared test (pipeline, docs, cli) failing the gate is a REAL regression
    and must keep blocking -- it is never a poison-cascade candidate.
    """
    if not test_file.startswith("tests/python/test_"):
        return False
    if not test_file.endswith(".py"):
        return False
    if "/quarantine/" in test_file:
        return False
    base = test_file[len("tests/python/") :]
    return base.startswith("test_experiment_") or base.startswith("test_exp")


def _iter_failure_sections(output: str) -> Iterable[tuple[str, str]]:
    """Yield ``(display_name, section_text)`` for each per-test FAILURES/ERRORS block.

    pytest groups tracebacks under ``=== FAILURES ===`` / ``=== ERRORS ===`` and
    separates each with an ``___ name ___`` header. We segment on those headers so
    each test's evidence (its referenced paths + error markers) stays associated
    with that test and does not bleed across unrelated failures.
    """
    lines = output.splitlines()
    in_section = False
    current_name: str | None = None
    buffer: list[str] = []

    def _flush() -> Iterable[tuple[str, str]]:
        if current_name is not None:
            yield current_name, "\n".join(buffer)

    for line in lines:
        stripped = line.strip()
        # A ``=== FAILURES ===`` / ``=== ERRORS ===`` banner opens the section;
        # ``=== short test summary`` / ``=== warnings`` / a bare ``===`` closes it.
        if stripped.startswith("=") and stripped.endswith("="):
            banner = stripped.strip("= ").lower()
            if banner in ("failures", "errors"):
                yield from _flush()
                current_name, buffer = None, []
                in_section = True
                continue
            if in_section:
                yield from _flush()
                current_name, buffer = None, []
                in_section = False
                continue
        if not in_section:
            continue
        header = _FAILURE_HEADER_RE.match(line)
        if header:
            yield from _flush()
            current_name = header.group(1).strip()
            buffer = []
            continue
        if current_name is not None:
            buffer.append(line)
    yield from _flush()


def _parse_summary_failures(output: str) -> list[tuple[str, str | None]]:
    """Return ``(test_file, nodeid_or_None)`` for each short-summary FAILED/ERROR line.

    This is the authoritative source of WHICH tests failed and their exact node
    ids (present even under ``-q``, which omits the ``- reason`` suffix).
    """
    found: list[tuple[str, str | None]] = []
    for line in output.splitlines():
        m = _SUMMARY_LINE_RE.match(line.strip())
        if not m:
            continue
        test_file = m.group(1)
        node_suffix = m.group(2)
        nodeid = f"{test_file}{node_suffix}" if node_suffix else None
        found.append((test_file, nodeid))
    return found


def _display_name_matches_nodeid(display_name: str, nodeid: str) -> bool:
    """Whether a FAILURES-header display name refers to the same test as a node id.

    pytest headers render ``Class.method`` / ``test_x[case]`` while node ids render
    ``file.py::Class::method`` / ``file.py::test_x[case]``. Compare on the node
    id's function part with ``::`` normalized to ``.`` and any ``[param]`` stripped,
    so both spellings of the same test agree.
    """
    node_func = nodeid.split("::", 1)[1] if "::" in nodeid else nodeid
    node_func = node_func.replace("::", ".").split("[", 1)[0]
    disp = display_name.replace("::", ".").split("[", 1)[0]
    return disp == node_func


def _collect_failure_evidence(output: str) -> list[_FailureEvidence]:
    """Build one evidence record per failing node id from the pytest blob.

    Combines the authoritative summary node ids with the traceback section that
    carries the paths + error markers. When a node id has no matching section
    header (rare -- e.g. a differently-rendered header), its section falls back to
    the traceback block that names the same test file, so the path/marker evidence
    is still associated correctly.
    """
    sections = list(_iter_failure_sections(output))
    summary = _parse_summary_failures(output)
    evidence: list[_FailureEvidence] = []
    used_sections: set[int] = set()

    def _section_for(nodeid: str, test_file: str) -> str:
        # Prefer the section whose header name matches this node id.
        for idx, (name, text) in enumerate(sections):
            if _display_name_matches_nodeid(name, nodeid):
                used_sections.add(idx)
                return text
        # Fallback: a section whose traceback names this same test file.
        for idx, (_name, text) in enumerate(sections):
            if any(
                _TRACEBACK_FILE_RE.search(l) and _TRACEBACK_FILE_RE.search(l).group(1) == test_file  # type: ignore[union-attr]
                for l in text.splitlines()
            ):
                used_sections.add(idx)
                return text
        return ""

    seen_nodeids: set[str] = set()
    for test_file, nodeid in summary:
        node = nodeid or f"{test_file}::<unknown>"
        if node in seen_nodeids:
            continue
        seen_nodeids.add(node)
        text = _section_for(node, test_file)
        evidence.append(
            _FailureEvidence(
                nodeid=node,
                test_file=test_file,
                result_paths=_RESULT_PATH_RE.findall(text),
                has_absence_marker=any(m in text for m in _FILE_ABSENCE_MARKERS),
            )
        )

    # A traceback-only failure with no summary line (e.g. summary suppressed):
    # recover it from any section not already consumed above.
    for idx, (name, text) in enumerate(sections):
        if idx in used_sections:
            continue
        file_match = _TRACEBACK_FILE_RE.search(text)
        if not file_match:
            continue
        test_file = file_match.group(1)
        node = f"{test_file}::{name.replace('.', '::')}"
        if node in seen_nodeids:
            continue
        seen_nodeids.add(node)
        evidence.append(
            _FailureEvidence(
                nodeid=node,
                test_file=test_file,
                result_paths=_RESULT_PATH_RE.findall(text),
                has_absence_marker=any(m in text for m in _FILE_ABSENCE_MARKERS),
            )
        )
    return evidence


def detect_poison_cascade(
    pytest_output: str,
    repo_root: Path | str = Path.cwd(),  # noqa: B008 - matches retro_timing_fallback.py idiom
    roadmap_deliverables: Mapping[str, str] | None = None,
    roadmap_milestone: str | None = None,
) -> TriageResult:
    """Detect the deliverable-read poison-test-cascade signature. Pure; no side effects.

    A failing test MATCHES (is a poison-cascade wait, safe to xfail) ONLY when ALL:

    1. its test file is experiment-specific (``test_experiment_*`` / ``test_exp*``);
    2. its failure block references a ``results/*.json`` path P;
    3. P is a declared ``deliverable`` of a task in ``research-roadmap.yaml``;
    4. P does NOT exist on disk at ``repo_root/P`` (the producer has not delivered);
    5. its failure block carries a file-absence marker (FileNotFoundError / No such
       file or directory / JSONDecodeError).

    Anything failing to meet all five is left in ``unmatched_failures`` and keeps
    blocking the gate -- an unrelated assertion, an import error, a stale/typo path
    absent from the roadmap, a deliverable already on disk (a genuine bug), or a
    core test are all correctly NOT masked.
    """
    root = Path(repo_root)
    if roadmap_deliverables is None:
        deliverables, milestone = load_roadmap_deliverables(root)
    else:
        deliverables = dict(roadmap_deliverables)
        milestone = roadmap_milestone

    matches: list[TriageMatch] = []
    unmatched: list[str] = []

    for ev in _collect_failure_evidence(pytest_output):
        match = _match_one(ev, deliverables, root)
        if match is not None:
            matches.append(match)
        else:
            unmatched.append(ev.nodeid)

    return TriageResult(
        matched=bool(matches),
        matches=tuple(matches),
        unmatched_failures=tuple(unmatched),
        roadmap_milestone=milestone,
    )


def _match_one(
    ev: _FailureEvidence,
    deliverables: Mapping[str, str],
    repo_root: Path,
) -> TriageMatch | None:
    """Apply the five-part signature to one failure. Returns a match or None."""
    if not is_experiment_specific_test(ev.test_file):
        return None
    if not ev.has_absence_marker:
        return None
    for raw_path in ev.result_paths:
        rel = _normalize_rel_path(raw_path, repo_root)
        if rel is None or rel not in deliverables:
            continue  # not a declared deliverable -> a genuinely broken test
        if (repo_root / rel).exists():
            continue  # deliverable IS present -> failure is a real assertion bug
        task_id = deliverables[rel]
        reason = (
            f"poison-test-cascade: {ev.nodeid} reads deliverable {rel}, which task "
            f"{task_id} main() has not yet produced; xfail self-expires when the "
            f"file lands"
        )
        return TriageMatch(
            test_file=ev.test_file,
            nodeid=ev.nodeid,
            deliverable_path=rel,
            producing_task_id=task_id,
            reason=reason,
        )
    return None


# ---------------------------------------------------------------------------
# xfail remediation -- the ONLY functions that touch test files. They never
# remove a test or an assertion; they only ADD a conditional, self-expiring mark.
# ---------------------------------------------------------------------------

_RUNTIME_SENTINEL = "# >>> pretest-triage runtime (auto-inserted) >>>"
_RUNTIME_BLOCK = """\
# >>> pretest-triage runtime (auto-inserted) >>>
import os as _pretest_triage_os  # noqa: E402
import pytest as _pretest_triage_pytest  # noqa: E402

_PRETEST_TRIAGE_REPO_ROOT = _pretest_triage_os.path.abspath(
    _pretest_triage_os.path.join(_pretest_triage_os.path.dirname(__file__), "..", "..")
)
# <<< pretest-triage runtime <<<
"""


def _xfail_sentinel(match: TriageMatch) -> str:
    """The per-node idempotency marker + human "remove me when delivered" hint."""
    return f"# pretest-triage:xfail {match.producing_task_id} -> {match.deliverable_path}"


def render_xfail_decorator(match: TriageMatch) -> str:
    """Return the exact source of the self-expiring xfail decorator for a match.

    The decorator is scoped to a SINGLE test (never the module) and its
    ``condition`` re-checks the deliverable's on-disk absence on every collection,
    so the moment the producing task delivers, the mark deactivates and the test
    runs live again. ``strict=False`` keeps an unexpected pass (should not happen
    while the condition holds) a non-failing XPASS signal, never a re-block.
    """
    reason = (
        f"pretest-triage poison-cascade guard: waits on not-yet-produced deliverable "
        f"{match.deliverable_path} from task {match.producing_task_id}; self-expires "
        f"when the file lands. See scripts/pretest_triage.py."
    )
    return (
        f"@_pretest_triage_pytest.mark.xfail(\n"
        f"    condition=not _pretest_triage_os.path.exists(\n"
        f"        _pretest_triage_os.path.join(_PRETEST_TRIAGE_REPO_ROOT, "
        f"{match.deliverable_path!r})\n"
        f"    ),\n"
        f"    reason={reason!r},\n"
        f"    strict=False,\n"
        f")  {_xfail_sentinel(match)}\n"
    )


def _nodeid_function(nodeid: str) -> str:
    """The bare ``def`` name to locate in source (``Class::m`` -> ``m``, strip params)."""
    func = nodeid.split("::", 1)[1] if "::" in nodeid else nodeid
    func = func.split("::")[-1]  # inner-most def for a class method
    return func.split("[", 1)[0]


def _ensure_runtime_block(lines: list[str]) -> list[str]:
    """Insert the aliased-import runtime block once, after the file's import head."""
    if any(_RUNTIME_SENTINEL in ln for ln in lines):
        return lines
    # Place it after the last top-level import (or __future__) near the top so the
    # aliased names exist before any injected decorator references them.
    insert_at = 0
    for idx, ln in enumerate(lines[:80]):
        stripped = ln.strip()
        if stripped.startswith(("import ", "from ")):
            insert_at = idx + 1
    block_lines = _RUNTIME_BLOCK.splitlines()
    return lines[:insert_at] + [""] + block_lines + lines[insert_at:]


def apply_xfail(result: TriageResult, repo_root: Path | str = Path.cwd()) -> list[str]:  # noqa: B008
    """Insert a self-expiring xfail decorator above each matched test's ``def``.

    Idempotent (skips a match whose sentinel is already above its def) and
    conservative: a node whose ``def`` cannot be confidently located at column 0 is
    left UNTOUCHED rather than risk corrupting the file. Returns the sorted list of
    test files actually modified. This is the only function here with side effects.
    """
    root = Path(repo_root)
    modified: set[str] = set()
    # Group matches by file so each file is read/written once.
    by_file: dict[str, list[TriageMatch]] = {}
    for m in result.matches:
        by_file.setdefault(m.test_file, []).append(m)

    for test_file, file_matches in by_file.items():
        path = root / test_file
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        for match in file_matches:
            func = _nodeid_function(match.nodeid)
            def_prefix = f"def {func}("
            def_idx = _find_module_level_def(lines, def_prefix)
            if def_idx is None:
                continue  # cannot locate safely -> leave it to the fallback guard
            if _sentinel_already_above(lines, def_idx, _xfail_sentinel(match)):
                continue  # already triaged
            lines = _ensure_runtime_block(lines)
            def_idx = _find_module_level_def(lines, def_prefix)  # re-find post-insert
            if def_idx is None:  # pragma: no cover - defensive, block never hides a def
                continue
            deco_lines = render_xfail_decorator(match).splitlines()
            lines = lines[:def_idx] + deco_lines + lines[def_idx:]
            changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            modified.add(test_file)
    return sorted(modified)


def _find_module_level_def(lines: list[str], def_prefix: str) -> int | None:
    """Index of a top-level (column-0) ``def <name>(`` line, or None if not found."""
    for idx, ln in enumerate(lines):
        if ln.startswith(def_prefix):
            return idx
    return None


def _sentinel_already_above(lines: list[str], def_idx: int, sentinel: str) -> bool:
    """Whether this match's xfail sentinel is already in the def's decorator head."""
    scan = max(0, def_idx - 12)
    return any(sentinel in ln for ln in lines[scan:def_idx])


# ---------------------------------------------------------------------------
# Historical retrospective validation (honest -- see the incident audit).
# ---------------------------------------------------------------------------

# The four cited incidents. Only exp5182/.475 is the deliverable-read signature
# this detector targets; the earlier three assert on the script's in-memory
# verdict / return value (verified: 0 ``results/experiment_*.json`` references in
# their poison versions) and are the SIBLING sub-class already handled by the
# conductor's consecutive-fail auto-quarantine guard. Reporting a 4/4 narrow-
# signature match would be false; this table reports the truth.
_HISTORICAL_INCIDENTS = (
    {
        "experiment_id": "exp5182",
        "milestone": "2026.07.475",
        "poison_class": "deliverable_read",
        "detector_matches": True,
        "note": (
            "test_experiment_5182_...::test_ondisk_deliverable_is_valid reads "
            "results/experiment_5182_...json before main() wrote it -> "
            "FileNotFoundError. Exact match; primary regression fixture."
        ),
    },
    {
        "experiment_id": "exp3521",
        "milestone": "2026.05.325",
        "poison_class": "verdict_assertion",
        "detector_matches": False,
        "note": (
            "Poison test asserts on in-memory arm results (0 results/experiment_*.json "
            "refs). Sibling sub-class; handled by the consecutive-fail quarantine guard."
        ),
    },
    {
        "experiment_id": "exp3544",
        "milestone": "2026.05.326",
        "poison_class": "verdict_assertion",
        "detector_matches": False,
        "note": (
            "Poison test asserts on the script's honest verdict, not a deliverable read "
            "(0 results/experiment_*.json refs). Sibling sub-class."
        ),
    },
    {
        "experiment_id": "exp3612",
        "milestone": "2026.05.332",
        "poison_class": "verdict_assertion",
        "detector_matches": False,
        "note": (
            "Poison test patches assert_deliverable_written and asserts honest_verdict "
            "strings on synthetic input (0 results/experiment_*.json refs). Sibling sub-class."
        ),
    },
)


def validate_historical_incidents() -> list[dict]:
    """Return the honest per-incident retrospective classification.

    For each of the four cited incidents, records whether THIS narrow detector
    would have matched it. Deliberately does NOT over-claim: only the deliverable-
    read variant (.475) matches; the verdict-assertion variants (.325/.326/.332)
    are correctly left to the pre-existing consecutive-fail quarantine guard.
    """
    return [dict(row) for row in _HISTORICAL_INCIDENTS]


def _synthetic_output_for(experiment_id: str, deliverable: str) -> str:
    """A minimal pytest FAILURES blob for the deliverable-read variant of an incident.

    Used to demonstrate that, HAD any incident read its real deliverable (the .475
    shape), the detector would catch it -- while keeping the honest note that the
    three earlier incidents did not actually take that shape.
    """
    test_file = f"tests/python/test_{experiment_id.replace('exp', 'experiment_')}_demo.py"
    return (
        "=================================== FAILURES ===================================\n"
        f"_______________________ test_ondisk_deliverable_is_valid _______________________\n"
        "    def test_ondisk_deliverable_is_valid() -> None:\n"
        f">       art = json.loads((REPO / '{deliverable}').read_text())\n"
        f"{test_file}:9: \n"
        f"E       FileNotFoundError: [Errno 2] No such file or directory: '/repo/{deliverable}'\n"
        "=========================== short test summary info ============================\n"
        f"FAILED {test_file}::test_ondisk_deliverable_is_valid\n"
        "1 failed in 0.10s\n"
    )


def _cli(argv: list[str] | None = None) -> int:
    """Console entry point: ``--validate-historical`` or triage a pytest-output file."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Path to a file holding captured pytest output to triage.",
    )
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument(
        "--validate-historical",
        action="store_true",
        help="Print the honest retrospective classification of the 4 cited incidents.",
    )
    args = parser.parse_args(argv)

    if args.validate_historical:
        print(json.dumps(validate_historical_incidents(), indent=2))
        return 0

    if not args.output_file:
        parser.error("provide a pytest-output file, or use --validate-historical")

    output = Path(args.output_file).read_text(encoding="utf-8")
    result = detect_poison_cascade(output, repo_root=Path(args.repo_root))
    payload = {
        "matched": result.matched,
        "all_failures_explained": result.all_failures_explained,
        "roadmap_milestone": result.roadmap_milestone,
        "matches": [
            {
                "nodeid": m.nodeid,
                "deliverable_path": m.deliverable_path,
                "producing_task_id": m.producing_task_id,
            }
            for m in result.matches
        ],
        "unmatched_failures": list(result.unmatched_failures),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via _cli in tests
    raise SystemExit(_cli())
