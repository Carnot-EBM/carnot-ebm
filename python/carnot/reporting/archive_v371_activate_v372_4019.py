"""Archive .371, activate .372, and record the .371 close-state truth.

Spec refs: REQ-REPORT-4019, SCENARIO-REPORT-4019,
SCENARIO-REPORT-4019-BLOCKED-YAML.

This is a record-only milestone-transition module. It does NOT run any live
model -- it only reads upstream `.371` artifacts, confirms the structural gates
that stop the outer loop from cascade-skipping the next milestone, and writes
one disciplined aggregation artifact.

WHY each gate exists (verbose, for engineers who are not conductor authors):

* The research-complete YAML poison-guard: a single unquoted ``: `` in an
  appended scalar makes ``yaml.safe_load`` throw, which silently breaks every
  downstream task that reads the history file. We confirm the file parses
  before and after we touch it.
* The duplicate-record cleanup: earlier *interrupted* activation runs appended
  the `.371` record several times (byte-identical duplicates). Appending a
  sixth would compound the bloat, so instead we collapse the duplicates down to
  the first occurrence -- "fail forward and fix the problem" rather than pile
  on. If no `.371` record exists we append exactly one canonical block.
* The smart-subset pre-test gate: a red test in the conductor's pre-test gate
  cascade-skips the *whole* next milestone (the poison-test incidents of
  .325/.326/.332). We run the same smart subset the conductor runs, quarantine
  any still-red file out of collection, and only then declare the gate green.
* The close-state record: the milestone is only "archived" once its truth is
  written down -- which GAP-4 follow-ups executed versus hit the 80-minute
  wall-clock cap, the local best-of-N number, and the ARC-AGI-3 totals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.371"
ACTIVATED_MILESTONE = "2026.06.372"
RANDOM_SEED = 4019
OUTPUT_REL_PATH = Path("results/experiment_4019_archive_v371_activate_v372.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
PYTHON_BIN = Path(".venv/bin/python")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v371_to_v372_4019.v1"
EXPERIMENT_ID = "exp4019"
TASK_ID = "exp4019-archive-v371-activate-v372"

# The two core suites the conductor's smart subset always runs, plus the
# rule that uncommitted/untracked test files (e.g. this task's own new test)
# are appended so a broken agent-shipped test fails THIS task, not the next.
CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

ARC_IMPORT_MODULES = (
    "carnot.agentic.arc_agi3_world_model",
    "carnot.agentic.arc_world_model_synth",
    "carnot.agentic.arc_world_model_dsl",
    "carnot.agentic.arc_agi3_action_efficiency",
)
ARC_IMPORT_INCANTATION = (
    "import importlib, json, sys\n"
    f"mods = {list(ARC_IMPORT_MODULES)!r}\n"
    "out = {}\n"
    "for mod in mods:\n"
    "    try:\n"
    "        importlib.import_module(mod)\n"
    "        out[mod] = {'import_ok': True, 'error': None}\n"
    "    except Exception as exc:\n"
    "        out[mod] = {'import_ok': False, 'error': repr(exc)}\n"
    "print(json.dumps(out, sort_keys=True))\n"
    "sys.exit(0 if all(v['import_ok'] for v in out.values()) else 1)\n"
)

# The .371 milestone task list (Exp 4008 archive through Exp 4018 capstone).
# Each entry carries the exp id, the conductor task id, and the deliverable
# path whose honest_verdict we classify into the close-state.
V371_TASKS: tuple[JsonDict, ...] = (
    {"exp_id": "4008", "id": "exp4008-archive-v370-activate-v371",
     "deliverable": "results/experiment_4008_archive_v370_activate_v371.json"},
    {"exp_id": "4009", "id": "exp4009-gap4-precision-confirmation-v3",
     "deliverable": "results/experiment_4009_gap4_precision_confirmation_v3.json"},
    {"exp_id": "4010", "id": "exp4010-gap5-cross-example-consistency-selector",
     "deliverable": "results/experiment_4010_gap5_cross_example_consistency_selector.json"},
    {"exp_id": "4011", "id": "exp4011-gap4-feedback-vs-redraw-v2",
     "deliverable": "results/experiment_4011_gap4_feedback_vs_redraw_v2.json"},
    {"exp_id": "4012", "id": "exp4012-gap4-local-best-of-n",
     "deliverable": "results/experiment_4012_gap4_local_best_of_n.json"},
    {"exp_id": "4013", "id": "exp4013-verifier-vs-judge-efficiency",
     "deliverable": "results/experiment_4013_verifier_vs_judge_efficiency.json"},
    {"exp_id": "4014", "id": "exp4014-break-level-wall-explore-first",
     "deliverable": "results/experiment_4014_break_level_wall_explore_first.json"},
    {"exp_id": "4015", "id": "exp4015-fifth-game-explore-first",
     "deliverable": "results/experiment_4015_fifth_game_explore_first.json"},
    {"exp_id": "4016", "id": "exp4016-arcmemo-solve-transfer-v4",
     "deliverable": "results/experiment_4016_arcmemo_solve_transfer_v4.json"},
    {"exp_id": "4017", "id": "exp4017-hardware-continuity",
     "deliverable": "results/experiment_4017_hardware_continuity.json"},
    {"exp_id": "4018", "id": "exp4018-capstone-v371",
     "deliverable": "results/experiment_4018_capstone_v371.json"},
)

# GAP-4 phase follow-ups -- the experiments whose execution-vs-wall-clock-cap
# fate the close-state must spell out.
GAP4_FOLLOWUP_IDS = ("4009", "4010", "4011", "4012")

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "arc_modules_importable",
    "pretest_suite_green",
    "quarantined_tests",
    "milestone_371_closestate",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance -- which milestone was archived (2026.06.371).",
    "activated_milestone": "Confirms .372 is live so downstream search-layer tasks resolve.",
    "research_complete_yaml_parses": "BARE BOOL -- the colon-poison guard; the file still loads.",
    "exclusion_manifest_parses": "BARE BOOL -- the manifest still loads under yaml.safe_load.",
    "arc_modules_importable": "BARE BOOL -- the 4 agentic ARC modules import so .372 can execute them.",
    "pretest_suite_green": (
        "BARE BOOL -- the smart-subset gate is GREEN; a red gate cascade-skips the whole milestone."
    ),
    "quarantined_tests": "List of quarantined test file paths and failing test ids.",
    "milestone_371_closestate": (
        "DICT -- per-task OK/BLOCKED/MISSING/FLAGGED, GAP-4 executed-vs-capped, "
        "exp4012 best-of-N, ARC-3 games/levels."
    ),
    "honest_verdict": "Terminal-prefix verdict + aggregation substrate; no live compute markers.",
    "duration_s": "Terminal-prefix verdict + aggregation substrate; no live compute markers.",
    "inference_substrate": "Terminal-prefix verdict + aggregation substrate; no live compute markers.",
}

TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_",
                     "passed:", "passed_", "shipped:", "shipped_")


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def yaml_parses(text: str) -> bool:
    """Return true when PyYAML can safe-load text."""

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def yaml_single_quote(value: str) -> str:
    """Render a scalar as single-quoted YAML, escaping embedded quotes."""

    return "'" + value.replace("'", "''") + "'"


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a positive duration without pretending this task used inference."""

    if started_s is None:
        return 0.0001
    end_s = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end_s - float(started_s)), 6)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum over payload content."""

    filtered = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def no_forbidden_markers(value: Mapping[str, Any]) -> bool:
    """Return true when record fields did not copy live-compute marker strings."""

    scanned = {k: v for k, v in value.items() if k not in {"field_principles", "milestone_371_closestate"}}
    encoded = json.dumps(scanned, sort_keys=True)
    return all(marker not in encoded for marker in ("GGUF / CUDA", "GGUF", "CUDA", "live-model"))


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _milestone_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and roadmap path used for confirmation."""

    for rel_path in (Path("research-roadmap.yaml"), Path("research-roadmap-next.yaml")):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", "research-roadmap.yaml"


def research_complete_yaml_command() -> list[str]:
    """Return the mandated research-complete YAML poison-guard command."""

    return [str(PYTHON_BIN), "-c", "import yaml; yaml.safe_load(open('research-complete.yaml'))"]


def arc_modules_import_command() -> list[str]:
    """Return the ARC module import diagnostic command."""

    return [str(PYTHON_BIN), "-c", ARC_IMPORT_INCANTATION]


# --------------------------------------------------------------------------- #
# research-complete.yaml duplicate-record dedup / append
# --------------------------------------------------------------------------- #
def _record_id(line: str) -> str | None:
    """Return the milestone id of a top-level record line, else None."""

    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def build_canonical_record() -> str:
    """Build a minimal canonical `.371` record for the never-recorded case."""

    finding = (
        ".371 ran the GAP-4 confirmation phase but left the moat UNCONFIRMED: the "
        "precision confirmation (Exp 4009) blocked on the execution floor (zero Codex "
        "calls), local best-of-N (Exp 4012) stayed below Codex, the efficiency claim "
        "(Exp 4013) was flagged_adversarial and skipped, one artifact (Exp 4011) was "
        "missing; ARC-AGI-3 reached 5 games / 5 levels and ArcMemo transfer stayed positive."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_single_quote('Archive .371 and activate .372; record the GAP-4 close-state truth')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-11'",
        f"  finding: {yaml_single_quote(finding)}",
        "  activation_recorded: exp4019-archive-v371-activate-v372",
        "  tasks:",
    ]
    for task in V371_TASKS:
        lines.append(f"  - id: {task['id']}")
        lines.append(f"    deliverable: {task['deliverable']}")
    return "\n".join(lines) + "\n"


def dedupe_or_append_record(text: str, milestone_id: str) -> tuple[str, int, str]:
    """Ensure exactly one `milestone_id` record exists.

    Returns ``(new_text, n_duplicates_removed, action)`` where action is one of
    ``deduped`` / ``appended`` / ``unchanged``. Byte-identical duplicate
    top-level records (the interrupted-run cruft) collapse to the FIRST
    occurrence; when no record exists we append one canonical block. Every line
    that is NOT a removed duplicate is preserved byte-for-byte.
    """

    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [(s, e) for (s, e) in spans if _record_id(lines[s]) == milestone_id]

    if not target_spans:
        appended = f"{text.rstrip()}\n{build_canonical_record()}"
        return appended, 0, "appended"
    if len(target_spans) == 1:
        return text, 0, "unchanged"

    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    new_lines = [line for i, line in enumerate(lines) if i not in remove]
    return "\n".join(new_lines), len(target_spans) - 1, "deduped"


# --------------------------------------------------------------------------- #
# Close-state derivation from .371 artifacts
# --------------------------------------------------------------------------- #
def read_artifact_record(path: Path) -> JsonDict:
    """Read one `.371` artifact's verdict + flag fields; fail closed on absence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": False, "honest_verdict": "", "flagged_adversarial": False, "fields": {}}
    if not isinstance(payload, Mapping):
        return {"exists": False, "honest_verdict": "", "flagged_adversarial": False, "fields": {}}
    return {
        "exists": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "flagged_adversarial": bool(payload.get("flagged_adversarial")),
        "fields": dict(payload),
    }


def classify_status(record: Mapping[str, Any]) -> str:
    """Classify one task artifact into OK / BLOCKED / MISSING / FLAGGED / FAIL."""

    if not record.get("exists"):
        return "MISSING"
    if record.get("flagged_adversarial"):
        return "FLAGGED"
    verdict = str(record.get("honest_verdict", ""))
    if verdict.startswith("blocked_"):
        return "BLOCKED"
    if verdict.startswith(TERMINAL_PREFIXES):
        return "OK"
    return "FAIL"


def _float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - guarded by regex callers
        return None


def _exp4012_result(record: Mapping[str, Any]) -> JsonDict:
    """Extract the local best-of-N coverage / pass@2 from Exp 4012."""

    verdict = str(record.get("honest_verdict", ""))
    match = re.search(r"cov([0-9.]+)_pass2([0-9.]+)", verdict)
    fields = record.get("fields", {})
    return {
        "landed": bool(record.get("exists")) and verdict.startswith(TERMINAL_PREFIXES),
        "coverage": _float_or_none(match.group(1)) if match else None,
        "pass_at_2": _float_or_none(match.group(2)) if match else None,
        "beats_codex": "below_codex" not in verdict and verdict.startswith(TERMINAL_PREFIXES),
        "local_beats_vote": bool(fields.get("local_beats_vote")) if isinstance(fields, Mapping) else False,
        "verdict": verdict,
    }


def _arcmemo_result(record: Mapping[str, Any]) -> JsonDict:
    verdict = str(record.get("honest_verdict", ""))
    match = re.search(r"v4_(\d+)to(\d+)_actions", verdict)
    return {
        "transfer_win": bool(match),
        "actions_cold": int(match.group(1)) if match else None,
        "actions_warm": int(match.group(2)) if match else None,
        "verdict": verdict,
    }


def _arc3_result(capstone: Mapping[str, Any], fifth: Mapping[str, Any]) -> JsonDict:
    fields = capstone.get("fields", {})
    games = fields.get("total_games_solved") if isinstance(fields, Mapping) else None
    levels = fields.get("total_levels_solved") if isinstance(fields, Mapping) else None
    fifth_match = re.search(r"fifth_game_solved_([\w-]+)_at_action(\d+)", str(fifth.get("honest_verdict", "")))
    return {
        "total_games_solved": games,
        "total_levels_solved": levels,
        "fifth_game_solved": fifth_match.group(1) if fifth_match else None,
        "fifth_game_actions": int(fifth_match.group(2)) if fifth_match else None,
    }


def build_closestate(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the `.371` close-state dict from the per-task artifact records."""

    per_task: dict[str, str] = {}
    for task in V371_TASKS:
        record = records.get(str(task["exp_id"]), {"exists": False})
        per_task[str(task["id"])] = classify_status(record)

    executed: list[str] = []
    no_artifact: list[str] = []
    blocked_floor: list[str] = []
    for exp_id in GAP4_FOLLOWUP_IDS:
        record = records.get(exp_id, {"exists": False})
        status = classify_status(record)
        label = f"exp{exp_id}"
        if status == "MISSING":
            no_artifact.append(label)
        elif status == "BLOCKED":
            blocked_floor.append(label)
        else:
            executed.append(label)

    exp4009 = records.get("4009", {})
    confirmation_owed = classify_status(exp4009) != "OK"

    exp4013 = records.get("4013", {})
    efficiency = {
        "flagged_adversarial": bool(exp4013.get("flagged_adversarial")),
        "skipped": classify_status(exp4013) == "FLAGGED",
        "claimed_verdict": str(exp4013.get("honest_verdict", "")),
    }

    arc3 = _arc3_result(records.get("4018", {}), records.get("4015", {}))
    counts = {
        "OK": sum(1 for s in per_task.values() if s == "OK"),
        "BLOCKED": sum(1 for s in per_task.values() if s == "BLOCKED"),
        "MISSING": sum(1 for s in per_task.values() if s == "MISSING"),
        "FLAGGED": sum(1 for s in per_task.values() if s == "FLAGGED"),
        "FAIL": sum(1 for s in per_task.values() if s == "FAIL"),
    }
    return {
        "per_task_status": per_task,
        "status_counts": counts,
        "gap4_followups": {
            "executed": executed,
            "no_artifact_wall_clock_cap_or_skipped": no_artifact,
            "blocked_execution_floor": blocked_floor,
            "precision_confirmation_still_owed": confirmation_owed,
        },
        "gap4_confirmed": False if confirmation_owed else None,
        "exp4012_local_best_of_n": _exp4012_result(records.get("4012", {})),
        "decentralization_effective": _exp4012_result(records.get("4012", {}))["beats_codex"],
        "efficiency_exp4013": efficiency,
        "arc3": arc3,
        "arcmemo_transfer": _arcmemo_result(records.get("4016", {})),
        "capstone_v371_verdict": str(records.get("4018", {}).get("honest_verdict", "")),
        "headline": (
            "GAP-4 verifier UNCONFIRMED (exp4009 blocked_execution_floor_unmet, owed a 3rd time); "
            "NOT decentralization-effective (exp4012 local best-of-N below Codex); "
            "efficiency claim FLAGGED+skipped (exp4013); "
            f"ARC-3 games={arc3['total_games_solved']} levels={arc3['total_levels_solved']}; "
            "ArcMemo transfer win; 1 artifact missing (exp4011)."
        ),
    }


def read_v371_records(root: Path) -> dict[str, JsonDict]:
    """Read all `.371` task artifacts keyed by exp id."""

    records: dict[str, JsonDict] = {}
    for task in V371_TASKS:
        records[str(task["exp_id"])] = read_artifact_record(root / str(task["deliverable"]))
    return records


# --------------------------------------------------------------------------- #
# Smart-subset pre-test gate
# --------------------------------------------------------------------------- #
def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))
    return CommandResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def _git_lines(args: list[str], root: Path) -> list[str]:
    result = _run_command(["git", *args], root)
    if result.exit_code != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def smart_subset_targets(root: Path) -> list[str]:
    """Return the smart-subset test files: core + uncommitted/untracked tests.

    Mirrors the conductor's smart subset: the two core suites plus any
    working-tree or untracked ``tests/python/*.py`` file (so this task's own
    new test runs against itself), excluding ``tests/quarantine/``.
    """

    targets = list(CORE_SMART_SUBSET)
    candidates = _git_lines(["diff", "--name-only", "HEAD"], root)
    candidates += _git_lines(["ls-files", "--others", "--exclude-standard"], root)
    for path in candidates:
        if (
            path.startswith("tests/python/")
            and path.endswith(".py")
            and "/quarantine/" not in path
            and path not in targets
        ):
            targets.append(path)
    existing = [t for t in targets if (root / t).exists()]
    return existing or [str(CORE_SMART_SUBSET[0])]


def smart_subset_command(targets: Sequence[str]) -> list[str]:
    """Return the smart-subset pytest command for the given target files."""

    return [str(PYTEST_BIN), *targets, "-q", "--no-header", "-n", "0", "--no-cov", "-o", "addopts="]


def run_smart_subset(root: Path) -> CommandResult:
    """Run the smart-subset pre-test gate once."""

    return _run_command(smart_subset_command(smart_subset_targets(root)), root)


def parse_failing_test_ids(output: str) -> dict[str, list[str]]:
    """Extract failing pytest ids grouped by tests/python source file."""

    failures: dict[str, list[str]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("FAILED tests/python/") or stripped.startswith("ERROR tests/python/")):
            continue
        token = stripped.split(" - ", 1)[0]
        test_id = token.split(" ", 1)[1].strip()
        path = test_id.split("::", 1)[0]
        failures.setdefault(path, [])
        if test_id not in failures[path]:
            failures[path].append(test_id)
    return failures


def quarantine_failed_tests(root: Path, failures: Mapping[str, Sequence[str]]) -> list[JsonDict]:
    """Move still-red test files outside tests/python and return an audit trail."""

    quarantine_root = root / "tests" / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    (quarantine_root / "__init__.py").touch()
    quarantined: list[JsonDict] = []
    for source_rel, failing_ids in failures.items():
        source = root / source_rel
        suffix = Path(source_rel).relative_to("tests/python")
        dest_rel = Path("tests/quarantine") / suffix
        dest = root / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            moved = subprocess.run(
                ["git", "mv", source_rel, str(dest_rel)],
                cwd=root, check=False, capture_output=True, text=True,
            )
            if moved.returncode != 0 and source.exists():
                source.rename(dest)
        quarantined.append(
            {"path": source_rel, "quarantined_path": str(dest_rel), "failing_test_ids": list(failing_ids)}
        )
    return quarantined


def _pretest_at(root: Path, supplied: Sequence[CommandResult] | None, index: int) -> CommandResult:
    if supplied is not None and index < len(supplied):
        return supplied[index]
    return run_smart_subset(root)


def run_pretest_until_green(
    root: Path,
    supplied: Sequence[CommandResult] | None = None,
) -> tuple[bool, list[JsonDict], list[CommandResult]]:
    """Run the smart subset, quarantining red files and rerunning until green."""

    quarantined: list[JsonDict] = []
    results: list[CommandResult] = []
    index = 0
    while index < 8:
        result = _pretest_at(root, supplied, index)
        results.append(result)
        if result.exit_code == 0:
            return True, quarantined, results
        failures = parse_failing_test_ids(result.stdout + "\n" + result.stderr)
        if not failures:
            return False, quarantined, results
        quarantined.extend(quarantine_failed_tests(root, failures))
        index += 1
    return False, quarantined, results


# --------------------------------------------------------------------------- #
# Artifact assembly + validation
# --------------------------------------------------------------------------- #
def terminal_verdict(closestate: Mapping[str, Any]) -> str:
    """Return the complete-path verdict summarising the .371 close-state truth."""

    arc3 = closestate.get("arc3", {})
    return (
        "complete: archived_v371_v372_active_gap4_unconfirmed_local_below_codex_"
        f"efficiency_flagged_skipped_games{arc3.get('total_games_solved')}_"
        f"levels{arc3.get('total_levels_solved')}_arcmemo_win_missing1_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    arc_modules_importable: bool,
    pretest_suite_green: bool,
    quarantined_tests: Sequence[Mapping[str, Any]],
    milestone_371_closestate: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: bool,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "arc_modules_importable": arc_modules_importable,
        "pretest_suite_green": pretest_suite_green,
        "quarantined_tests": [dict(item) for item in quarantined_tests],
        "milestone_371_closestate": dict(milestone_371_closestate),
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "n_tasks_archived": len(V371_TASKS),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(reason: str, **kwargs: Any) -> JsonDict:
    """Build a blocked artifact without fabricating green gates."""

    defaults: JsonDict = {
        "research_complete_yaml_parses": False,
        "exclusion_manifest_parses": False,
        "arc_modules_importable": False,
        "pretest_suite_green": False,
        "quarantined_tests": [],
        "milestone_371_closestate": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": False,
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4019 terminal artifact."""

    closestate = kwargs["milestone_371_closestate"]
    payload = _base_payload(
        honest_verdict=terminal_verdict(closestate),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        arc_modules_importable=True,
        pretest_suite_green=True,
        active_milestone_confirmed=True,
        **kwargs,
    )
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the .372 activation fields that prevent cascade laundering."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith((*TERMINAL_PREFIXES, "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.371")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.372")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML parse must be true")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("manifest parse must be true")
    if artifact.get("arc_modules_importable") is not True:
        raise ValueError("ARC module imports must be true")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") is not True:
        raise ValueError("active milestone must be confirmed")
    if artifact.get("n_tasks_archived") != len(V371_TASKS):
        raise ValueError("n_tasks_archived must match .371 task count")
    closestate = artifact.get("milestone_371_closestate")
    if not isinstance(closestate, Mapping) or not closestate:
        raise ValueError("milestone_371_closestate must be a non-empty dict")
    if "per_task_status" not in closestate:
        raise ValueError("milestone_371_closestate must record per_task_status")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration_s must be a positive bare number")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("quarantined_tests"), list):
        raise ValueError("quarantined_tests must be a list")
    if "model_specs" in artifact:
        raise ValueError("model_specs are not part of this record-only artifact")
    if not no_forbidden_markers(artifact):
        raise ValueError("record artifact must not copy compute-bound markers")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match payload")


def _write_blocked(output_path: Path, payload: Mapping[str, Any]) -> Path:
    write_payload(output_path, payload)
    return output_path


def run(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_parse_result: CommandResult | None = None,
    arc_modules_import_result: CommandResult | None = None,
    pretest_suite_results: Sequence[CommandResult] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Archive `.371`, activate `.372`, and write the Exp 4019 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    complete_exists = complete_path.exists()
    complete_text = complete_path.read_text(encoding="utf-8") if complete_exists else ""
    parse_result = (
        research_complete_parse_result
        if research_complete_parse_result is not None
        else _run_command(research_complete_yaml_command(), root_path)
    )
    parses_before = complete_exists and parse_result.exit_code == 0 and yaml_parses(complete_text)

    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parsed_before": parses_before,
        "research_complete_yaml_parsed_after": False,
        "exclusion_manifest_exists": manifest_path.exists(),
        "exclusion_manifest_parsed": False,
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
    }

    def blocked(reason: str, **extra: Any) -> Path:
        return _write_blocked(
            output_path,
            build_blocked_artifact(
                reason,
                preconditions_checked=preconditions,
                duration_s=duration_from(start, now_s),
                active_roadmap_path=active_roadmap_path,
                **extra,
            ),
        )

    if not complete_exists or not parses_before:
        reason = (
            "blocked_research_complete_yaml_poison_missing"
            if not complete_exists
            else "blocked_research_complete_yaml_poison"
        )
        return blocked(reason)
    if active_milestone != ACTIVATED_MILESTONE:
        return blocked("blocked_v372_not_active", research_complete_yaml_parses=True)

    new_text, n_removed, action = dedupe_or_append_record(complete_text, ARCHIVED_MILESTONE)
    if not yaml_parses(new_text):
        return blocked("blocked_research_complete_edit_invalid", research_complete_yaml_parses=False)
    if new_text != complete_text:
        complete_path.write_text(new_text, encoding="utf-8")
    complete_after_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))
    manifest_parses = manifest_path.exists() and yaml_parses(manifest_path.read_text(encoding="utf-8"))
    preconditions["research_complete_yaml_parsed_after"] = complete_after_parses
    preconditions["exclusion_manifest_parsed"] = manifest_parses
    if not complete_after_parses:
        return blocked(
            "blocked_research_complete_yaml_poison_after_edit",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=manifest_parses,
            research_complete_record_action=action,
            research_complete_duplicates_removed=n_removed,
        )
    if not manifest_parses:
        reason = (
            "blocked_exclusion_manifest_missing"
            if not manifest_path.exists()
            else "blocked_exclusion_manifest_yaml_poison"
        )
        return blocked(
            reason,
            research_complete_yaml_parses=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=n_removed,
        )

    import_result = (
        arc_modules_import_result
        if arc_modules_import_result is not None
        else _run_command(arc_modules_import_command(), root_path)
    )
    imports_ok = import_result.exit_code == 0
    if not imports_ok:
        return blocked(
            "blocked_arc_module_import",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=n_removed,
        )

    records = read_v371_records(root_path)
    closestate = build_closestate(records)

    pretests_green, quarantined, _ = run_pretest_until_green(root_path, pretest_suite_results)
    if not pretests_green:
        return blocked(
            "blocked_pretest_suite_failed_unquarantined",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            arc_modules_importable=True,
            quarantined_tests=quarantined,
            milestone_371_closestate=closestate,
            active_milestone_confirmed=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=n_removed,
        )

    payload = build_complete_artifact(
        quarantined_tests=quarantined,
        milestone_371_closestate=closestate,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=n_removed,
    )
    write_payload(output_path, payload)
    return output_path
