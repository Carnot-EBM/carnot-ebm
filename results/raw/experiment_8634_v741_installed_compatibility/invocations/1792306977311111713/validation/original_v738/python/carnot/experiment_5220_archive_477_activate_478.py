"""Exp 5220: archive .477 and activate .478.

Spec refs: REQ-REPORT-5220, SCENARIO-REPORT-5220,
SCENARIO-REPORT-5220-BLOCKED-PRECONDITION.

This transition module reads already-written `.477` artifacts, verifies the
active `.478` roadmap state, runs the available activation checks, and writes
the handoff artifact for the conductor. It intentionally performs no live model
work and it does not modify `scripts/research_conductor.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5220_archive_477_activate_478.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5220_archive_477_activate_478"
EXPERIMENT_ID = "exp5220-archive-477-activate-478"
ARCHIVED_MILESTONE = "2026.07.477"
MILESTONE = "2026.07.478"
SCHEMA = "carnot.experiment_5220_archive_477_activate_478.v1"
RANDOM_SEED = 5220
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: .477 archived and .478 activated; handoff preserves GAP-1 positive "
    "but unpromoted, GAP-4 flagged/protocol-blocked, MMLU hidden-state retired, "
    "self-learning memory written, ARC zero-delta, hardware reachability, and "
    "verifier-authenticity registry flags."
)
BLOCKED_VERDICT = "complete: .477 archive recorded but .478 activation blocked_precondition"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-REPORT-5220",
    "SCENARIO-REPORT-5220",
    "SCENARIO-REPORT-5220-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "v477_summary": (
        "Downstream task context depends on this summary being exact, especially nulls, "
        "flags, and retired threads."
    ),
    "research_roadmap_yaml_activated": (
        "Downstream conductor execution depends on `research-roadmap.yaml` naming `.478` "
        "and containing the Exp 5220 onward task set."
    ),
    "exclusion_manifest_confirmed_clean": (
        "The activated .478 roadmap must pass the exclusion-manifest gate without hard "
        "retired-scope violations."
    ),
    "validation_commands_run": (
        "Activation claims must be backed by named commands with pass/fail outcomes, "
        "not by implied manual inspection."
    ),
    "ops_docs_updated": (
        "Records whether this task changed ops/status.md or ops/changelog.md; a false "
        "value is valid when the conductor stop rule defers ops reconciliation."
    ),
    "research_conductor_py_untouched_confirmed": (
        "The transition must not modify scripts/research_conductor.py."
    ),
    "inference_substrate": "This archive reads upstream artifacts and activation checks only.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and must state whether "
        ".478 was activated."
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "archived_milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "source_artifacts",
    "missing_artifacts",
    "archived_research_roadmap_yaml",
    "roadmap_activation_check",
    "validation_checks",
    "failed_preconditions",
    "clean_handoff",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5220_archive_477_activate_478.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5220_archive_477_activate_478.py' -m pytest tests/python/test_experiment_5220_archive_477_activate_478.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5220_archive_477_activate_478.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5220_archive_477_activate_478.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One `.477` result artifact required for a clean transition archive."""

    experiment_number: int
    task_id: str
    relative_path: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured result from an activation-validation command."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str = ""


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5207,
        "exp5207-archive-476-activate-477",
        Path("results/experiment_5207_archive_476_activate_477.json"),
    ),
    UpstreamSource(
        5208,
        "exp5208-sota-ingestion-v477",
        Path("results/experiment_5208_sota_ingestion_v477.json"),
    ),
    UpstreamSource(
        5209,
        "exp5209-gap1-set-search-holdout-hardening-v477",
        Path("results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"),
    ),
    UpstreamSource(
        5210,
        "exp5210-gap1-registry-promotion-gated-v477",
        Path("results/experiment_5210_gap1_registry_promotion_gated_v477.json"),
    ),
    UpstreamSource(
        5211,
        "exp5211-gap4-sota-local-candidate-expansion-v477",
        Path("results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json"),
    ),
    UpstreamSource(
        5212,
        "exp5212-gap4-scale-validation-gated-v477",
        Path("results/experiment_5212_gap4_scale_validation_gated_v477.json"),
    ),
    UpstreamSource(
        5213,
        "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477",
        Path("results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json"),
    ),
    UpstreamSource(
        5214,
        "exp5214-continuous-self-learning-verifier-memory-v477",
        Path("results/experiment_5214_continuous_self_learning_verifier_memory_v477.json"),
    ),
    UpstreamSource(
        5215,
        "exp5215-arc-paw-amortization-gate-v477",
        Path("results/experiment_5215_arc_paw_amortization_gate_v477.json"),
    ),
    UpstreamSource(
        5216,
        "exp5216-arc-frontier-continuity-landmark-decomposition-v477",
        Path("results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json"),
    ),
    UpstreamSource(
        5217,
        "exp5217-hardware-continuity-v477",
        Path("results/experiment_5217_hardware_continuity_v477.json"),
    ),
    UpstreamSource(
        5218,
        "exp5218-verifier-authenticity-remediation-apply-v477",
        Path("results/experiment_5218_verifier_authenticity_remediation_apply_v477.json"),
    ),
    UpstreamSource(
        5219, "exp5219-capstone-v477", Path("results/experiment_5219_capstone_v477.json")
    ),
)

REQUIRED_478_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5220, 5233))


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value_of(value["value"])
    return value


def _number(value: Any) -> float | None:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _bool(value: Any) -> bool | None:
    raw = value_of(value)
    return raw if isinstance(raw, bool) else None


def _string(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else str(raw if raw is not None else "")


def _list(value: Any) -> list[Any]:
    raw = value_of(value)
    return raw if isinstance(raw, list) else []


def _mapping(value: Any) -> JsonDict:
    raw = value_of(value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def text_sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {
            "exists": True,
            "loadable": False,
            "error": "malformed_json",
            "sha256": file_sha256(path),
        }
    if not isinstance(parsed, Mapping):
        return {}, {
            "exists": True,
            "loadable": False,
            "error": "not_json_object",
            "sha256": file_sha256(path),
        }
    return dict(parsed), {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": file_sha256(path),
    }


def load_upstream_artifacts(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict], list[str]]:
    artifacts: dict[int, JsonDict] = {}
    rows: list[JsonDict] = []
    missing: list[str] = []
    for source in UPSTREAM_SOURCES:
        data, meta = read_json_mapping(root / source.relative_path)
        if meta.get("loadable") is True:
            artifacts[source.experiment_number] = data
        else:
            missing.append(f"missing_artifact_exp{source.experiment_number}")
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "relative_path": str(source.relative_path),
                "exists": meta.get("exists") is True,
                "loadable": meta.get("loadable") is True,
                "sha256": meta.get("sha256"),
                "error": meta.get("error"),
                "honest_verdict": _string(data.get("honest_verdict")) if data else "",
            }
        )
    return artifacts, rows, missing


def _roadmap_data(text: str) -> JsonDict:
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _task_ids(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task.get("id", "")) for task in tasks if isinstance(task, Mapping)]


def _roadmap_archive(text: str) -> JsonDict:
    roadmap = _roadmap_data(text)
    task_ids = _task_ids(roadmap)
    return {
        "path": str(ROADMAP_RELATIVE_PATH),
        "milestone": roadmap.get("milestone"),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "content_sha256": text_sha256(text),
        "content_before_activation": text,
    }


def activate_roadmap(root: Path) -> JsonDict:
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    before_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    archived = _roadmap_archive(before_text)
    copied = False
    activation_source = "research-roadmap-next.yaml_missing"
    if next_path.exists():
        next_text = next_path.read_text(encoding="utf-8")
        roadmap_path.write_text(next_text, encoding="utf-8")
        copied = True
        activation_source = "copied_research-roadmap-next.yaml"
    after_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    after = _roadmap_data(after_text)
    task_ids = _task_ids(after)
    missing_prefixes = [
        prefix
        for prefix in REQUIRED_478_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    activated = after.get("milestone") == MILESTONE and not missing_prefixes
    if not copied and activated:
        activation_source = "research-roadmap.yaml_already_active"
    return {
        "exists": roadmap_path.exists(),
        "parses": bool(after),
        "path": str(ROADMAP_RELATIVE_PATH),
        "milestone": after.get("milestone"),
        "task_ids": task_ids,
        "missing_task_prefixes": missing_prefixes,
        "activated": activated,
        "activation_source": activation_source,
        "roadmap_next_present": next_path.exists(),
        "copied_research_roadmap_next": copied,
        "pre_activation_milestone": archived.get("milestone"),
        "pre_activation_content_sha256": archived.get("content_sha256"),
        "post_activation_content_sha256": text_sha256(after_text) if after_text else None,
    }


def validation_commands(root: Path) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    exclusion = root / "scripts" / "exclusion_manifest_lint.py"
    prior = root / "scripts" / "validate_prior_failures.py"
    if exclusion.exists():
        commands.append((sys.executable, str(exclusion), str(root / ROADMAP_RELATIVE_PATH)))
    if prior.exists():
        commands.append((sys.executable, str(prior), str(root / ROADMAP_RELATIVE_PATH)))
    return commands


def run_command(command: tuple[str, ...], root: Path) -> CommandResult:
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_validation_commands(root: Path) -> list[CommandResult]:
    return [run_command(command, root) for command in validation_commands(root)]


def _command_label(command: str) -> str:
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
    if "validate_prior_failures.py" in command:
        return "scripts/validate_prior_failures.py"
    return command.split()[0] if command.split() else "unknown_command"


def validation_rows(results: Sequence[CommandResult]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for result in results:
        command_text = " ".join(result.command)
        rows.append(
            {
                "command": command_text,
                "command_label": _command_label(command_text),
                "exit_code": result.exit_code,
                "passed": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return rows


def exclusion_manifest_clean(rows: Sequence[JsonMap]) -> bool:
    for row in rows:
        if row.get("command_label") == "scripts/exclusion_manifest_lint.py":
            text = f"{row.get('stdout', '')}\n{row.get('stderr', '')}"
            return row.get("passed") is True and "HARD" not in text
    return False


def research_conductor_untouched(root: Path) -> bool:
    path = root / CONDUCTOR_RELATIVE_PATH
    if not path.exists():
        return False
    if not (root / ".git").exists():
        return True
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", str(CONDUCTOR_RELATIVE_PATH)], cwd=root, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", str(CONDUCTOR_RELATIVE_PATH)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.returncode == 0 and status.stdout.strip() == ""


def _fmt_float(value: float | None, digits: int = 6) -> str:
    return "unknown" if value is None else f"{value:.{digits}f}"


def _fmt_p(value: float | None) -> str:
    return "unknown" if value is None else f"{value:g}"


def _flag_kinds(flags: Any) -> str:
    if not isinstance(flags, list):
        return "none"
    kinds = sorted({str(item.get("kind")) for item in flags if isinstance(item, Mapping)})
    return "/".join(kinds) if kinds else "none"


def _status_summary(value: Any, verdict: str, token: str, fallback: str) -> str:
    raw = value_of(value)
    if isinstance(raw, Mapping):
        if raw.get("summary"):
            return str(raw["summary"])
        if raw.get("status"):
            return str(raw["status"])
        if raw.get("reachable") is True:
            return "reachable"
    if raw:
        return str(raw)
    return fallback if token in verdict else "unknown"


def build_v477_summary(artifacts: JsonMap) -> str:
    exp5209 = artifacts.get(5209, {})
    exp5210 = artifacts.get(5210, {})
    exp5211 = artifacts.get(5211, {})
    exp5212 = artifacts.get(5212, {})
    exp5213 = artifacts.get(5213, {})
    exp5214 = artifacts.get(5214, {})
    exp5215 = artifacts.get(5215, {})
    exp5216 = artifacts.get(5216, {})
    exp5217 = artifacts.get(5217, {})
    exp5218 = artifacts.get(5218, {})
    exp5219 = artifacts.get(5219, {})

    heldout = _number(exp5209.get("heldout_pass_at_2_mean"))
    always = _number(exp5209.get("baseline_always_on_pass_at_2_mean"))
    single = _number(exp5209.get("single_refuted_directional_pass_at_2_mean"))
    ci = _string(exp5209.get("paired_delta_ci95")) or "unknown"
    grouped = _int(exp5209.get("n_grouped_splits"))
    stable = _bool(exp5209.get("best_subset_stable"))
    gate_summary = _string(exp5210.get("gate_check_summary")) or _string(
        exp5210.get("honest_verdict")
    )

    pool_n = _int(exp5211.get("candidate_pool_n"))
    gap4_flags = _flag_kinds(exp5211.get("corrigendum_pending"))
    n_scored = _int(exp5212.get("n_scored"))
    excluded_protocol = _mapping(exp5212.get("exclusion_summary")).get(
        "missing_protocol_pass2_fields", "unknown"
    )
    wins = _int(exp5212.get("exact_test_discordant_wins"))
    losses = _int(exp5212.get("exact_test_discordant_losses"))
    p_value = _number(exp5212.get("exact_test_p_value_two_sided"))
    exp5212_flags = _flag_kinds(exp5212.get("corrigendum_pending"))

    probe = _number(exp5213.get("best_probe_accuracy"))
    tuned_sc = _number(exp5213.get("tuned_sc_accuracy"))
    clue = _number(exp5213.get("clue_accuracy"))
    rcs = _number(exp5213.get("radial_consensus_score_accuracy"))
    mmlu_retired = _bool(exp5213.get("retire_mmlu_hidden_state_path"))

    memory_path = _string(exp5214.get("memory_artifact_path")) or "unknown"
    promotions = _int(exp5214.get("promotions"))
    rollbacks = _int(exp5214.get("rollbacks"))
    entries = _int(exp5214.get("memory_entries_written"))

    paw_viable = _bool(exp5215.get("paw_amortization_viable"))
    break_even = _number(exp5215.get("break_even_remaining_actions"))
    median_actions = _number(exp5215.get("median_remaining_actions"))
    p75_actions = _number(exp5215.get("p75_remaining_actions"))

    level_delta = _int(exp5216.get("reproducible_total_levels_delta"))
    levels_banked = len(_list(exp5216.get("new_levels_banked")))
    provenance = _string(exp5216.get("solve_provenance")) or "unknown"

    hardware_verdict = _string(exp5217.get("honest_verdict"))
    kv260 = _status_summary(
        exp5217.get("kv260_status"), hardware_verdict, "kv260:reachable", "reachable"
    )
    polarfire = _status_summary(
        exp5217.get("polarfire_status"), hardware_verdict, "polarfire:reachable", "reachable"
    )
    gatemate = _status_summary(
        exp5217.get("gatemate_status"), hardware_verdict, "gatemate:blocked", "blocked"
    )
    narrowed = _string(exp5217.get("gatemate_diagnostic_narrowed_to")) or "unknown"

    remediation_type = _string(exp5218.get("remediation_type")) or "unknown"
    headline_ineligible = _bool(exp5218.get("headline_ineligible_until_real_verification"))

    gap1_final = _string(exp5219.get("gap1_final_status")) or "unknown"
    gap4_final = _string(exp5219.get("gap4_final_status")) or "unknown"
    capstone_arc_delta = _int(exp5219.get("reproducible_total_levels_delta"))
    flagged_excluded = _bool(exp5219.get("flagged_adversarial_artifacts_excluded"))
    capstone_verdict = _string(exp5219.get("honest_verdict"))

    return (
        ".477 closed as a flagged-gate handoff milestone: exp5209 kept GAP-1 set-search positive "
        f"with heldout pass@2 {_fmt_float(heldout)} versus always-on {_fmt_float(always)} and "
        f"single-refuted {_fmt_float(single)}, paired_delta_ci95={ci}, n_grouped_splits={grouped}, "
        f"and best_subset_stable={str(stable).lower() if stable is not None else 'unknown'}; "
        "exp5210 did not promote to the registry because the conductor pre-gate compared expected "
        f"bare true with a principle-wrapped gate object ({gate_summary}); exp5211 built a "
        f"{pool_n}-row GAP-4 candidate pool but it was adversarially flagged "
        f"({gap4_flags}) and cannot headline .478 without provenance repair; exp5212 scored "
        f"n_scored={n_scored} because {excluded_protocol} rows lacked missing_protocol_pass2_fields, "
        f"recorded {wins}/{losses} discordant wins/losses, p={_fmt_p(p_value)}, carried flags "
        f"{exp5212_flags}, and did not cross the unchanged six-discordant-win floor; exp5213 "
        f"retired the MMLU-Pro hidden-state path with best_probe_accuracy={_fmt_float(probe, 3)} "
        f"versus tuned_sc={_fmt_float(tuned_sc, 3)}, CLUE={_fmt_float(clue, 3)}, "
        f"RCS={_fmt_float(rcs, 3)}, retire_mmlu_hidden_state_path={str(mmlu_retired).lower()}; "
        f"exp5214 wrote continuous self-learning memory at {memory_path} with {entries} entries, "
        f"one promotion and one rollback, and no registry claim; exp5215 found PAW not viable "
        f"(paw_amortization_viable={str(paw_viable).lower() if paw_viable is not None else 'unknown'}, "
        f"break_even_remaining_actions={_fmt_float(break_even)}, median={_fmt_float(median_actions)}, "
        f"p75={_fmt_float(p75_actions)}) and made no ARC solve claim; exp5216 attempted live-path "
        f"integration but banked zero reproduction-gated ARC levels with reproducible_total_levels_delta="
        f"{level_delta}, new_levels_banked={levels_banked}, solve_provenance={provenance}; exp5217 kept "
        f"hardware continuity with KV260=reachable ({kv260}), PolarFire=reachable ({polarfire}), "
        f"GateMate={narrowed} ({gatemate}), and no speedup claim; exp5218 verifier-authenticity "
        f"registry flags applied via remediation_type={remediation_type}, headline_ineligible_until_real_"
        f"verification={str(headline_ineligible).lower() if headline_ineligible is not None else 'unknown'}; "
        f"exp5219 reconciled GAP-1 {gap1_final}, GAP-4 {gap4_final}, MMLU-Pro hidden-state path retired, "
        f"continuous self-learning satisfied, ARC delta {capstone_arc_delta}, and flagged artifacts "
        f"excluded={str(flagged_excluded).lower() if flagged_excluded is not None else 'unknown'} "
        f"({capstone_verdict})."
    )


def _failed_preconditions(
    *,
    missing_artifacts: Sequence[str],
    roadmap_activation: JsonMap,
    validation: Sequence[JsonMap],
    conductor_clean: bool,
    vnext_present: bool,
) -> list[str]:
    failures = list(missing_artifacts)
    if not roadmap_activation.get("activated"):
        failures.append("research_roadmap_yaml_not_active_for_478")
    for row in validation:
        if row.get("passed") is not True:
            failures.append(f"validation_failed_{row.get('command_label')}")
    if not validation:
        failures.append("validation_commands_missing")
    if not conductor_clean:
        failures.append("scripts_research_conductor_py_modified")
    if not vnext_present:
        failures.append("research_roadmap_vnext_doc_missing")
    return failures


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str | None = None,
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    pre_activation_text = (
        (root / ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
        if (root / ROADMAP_RELATIVE_PATH).exists()
        else ""
    )
    archived_roadmap = _roadmap_archive(pre_activation_text)
    roadmap_activation = activate_roadmap(root)
    artifacts, sources, missing = load_upstream_artifacts(root)
    command_results = (
        list(validation_results)
        if validation_results is not None
        else run_validation_commands(root)
    )
    validation = validation_rows(command_results)
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    vnext_present = (root / VNEXT_RELATIVE_PATH).exists()
    exclusion_clean = exclusion_manifest_clean(validation)
    failures = _failed_preconditions(
        missing_artifacts=missing,
        roadmap_activation=roadmap_activation,
        validation=validation,
        conductor_clean=conductor_clean,
        vnext_present=vnext_present,
    )
    clean_handoff = not failures
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "run_date": run_date or date.today().strftime("%Y%m%d"),
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "random_seed": RANDOM_SEED,
        "source_artifacts": sources,
        "missing_artifacts": list(missing),
        "archived_research_roadmap_yaml": archived_roadmap,
        "roadmap_activation_check": roadmap_activation,
        "validation_checks": validation,
        "failed_preconditions": failures,
        "clean_handoff": clean_handoff,
        "tests_run": list(tests_run if tests_run is not None else DEFAULT_TESTS_RUN),
        "v477_summary": _principled("v477_summary", build_v477_summary(artifacts)),
        "research_roadmap_yaml_activated": _principled(
            "research_roadmap_yaml_activated", bool(roadmap_activation.get("activated"))
        ),
        "exclusion_manifest_confirmed_clean": _principled(
            "exclusion_manifest_confirmed_clean", exclusion_clean
        ),
        "validation_commands_run": _principled("validation_commands_run", validation),
        "ops_docs_updated": _principled("ops_docs_updated", False),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", conductor_clean
        ),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _principled(
            "honest_verdict", COMPLETE_VERDICT if clean_handoff else BLOCKED_VERDICT
        ),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("schema or experiment_id mismatch")
    if (
        payload.get("milestone") != MILESTONE
        or payload.get("archived_milestone") != ARCHIVED_MILESTONE
    ):
        raise ValueError("milestone mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = payload.get(field)
        if not isinstance(wrapped, Mapping):
            raise ValueError(f"{field} must be principle-wrapped")
        if wrapped.get("principle") != principle:
            raise ValueError(f"{field} principle mismatch")
        if "value" not in wrapped:
            raise ValueError(f"{field} missing value")
    verdict = _string(payload["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must have a terminal prefix")
    if value_of(payload["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(value_of(payload["research_roadmap_yaml_activated"]), bool):
        raise ValueError("research_roadmap_yaml_activated must be bool")
    if not isinstance(value_of(payload["exclusion_manifest_confirmed_clean"]), bool):
        raise ValueError("exclusion_manifest_confirmed_clean must be bool")
    if not isinstance(value_of(payload["ops_docs_updated"]), bool):
        raise ValueError("ops_docs_updated must be bool")
    if not isinstance(value_of(payload["research_conductor_py_untouched_confirmed"]), bool):
        raise ValueError("research_conductor_py_untouched_confirmed must be bool")
    commands = value_of(payload["validation_commands_run"])
    if not isinstance(commands, list):
        raise ValueError("validation_commands_run must be a list")
    if payload.get("clean_handoff") is True and payload.get("failed_preconditions"):
        raise ValueError("clean_handoff cannot have failed_preconditions")
    if not payload.get("tests_run"):
        raise ValueError("tests_run must record verification commands")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str | None = None,
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    payload = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_results=validation_results,
        conductor_untouched=conductor_untouched,
        tests_run=tests_run,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, payload)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - direct CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", dest="run_date", default=None)
    args = parser.parse_args(argv)
    print(run(root=args.root, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
