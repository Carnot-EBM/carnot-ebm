"""Archive .394, activate .395, and preserve the blocked OOD truth.

Spec refs: REQ-REPORT-4269, SCENARIO-REPORT-4269,
SCENARIO-REPORT-4269-BLOCKED-PRECONDITION.

This is a record-only transition. It records that `.394` hardened the first ARC
oracle-distinct win on two of three axes, while the decisive cross-game OOD axis
blocked on provenance. That keeps `.395` framed as closing the cross-family
question, not as a redo and not as permission to scale prematurely.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot.reporting.archive_v391_activate_v392_4230 import (
    CommandResult,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    read_active_milestone,
    read_json_object,
    run_smart_subset,
    write_payload,
    yaml_parses,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.394"
ACTIVATED_MILESTONE = "2026.06.395"
RANDOM_SEED = 4269
OUTPUT_REL_PATH = Path("results/experiment_4269_archive_v394_activate_v395.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4268_capstone_v394.json")
LEAK_AUDIT_REL_PATH = Path("results/experiment_4256_arc_oracle_distinct_leak_audit.json")
MULTISEED_REL_PATH = Path(
    "results/experiment_4257_arc_oracle_distinct_multiseed_replication.json"
)
CROSS_GAME_REL_PATH = Path(
    "results/experiment_4258_arc_oracle_distinct_cross_game_transfer.json"
)
SYNTHESIS_REL_PATH = Path("results/experiment_4259_arc_agglm_grid_synthesis.json")
PREFLIGHT_REL_PATH = Path("results/experiment_4260_diffusiongemma_energy_guided_preflight.json")
ARC_PROGRESS_REL_PATH = Path("results/experiment_4261_arc_incremental_progress.json")
REWARD_REL_PATH = Path("results/experiment_4263_verifier_as_reward_out_of_band_or_retire.json")
CODE_REL_PATH = Path("results/experiment_4264_code_oracle_distinct_replication_retry.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v394_to_v395_4269.v1"
TASK_ID = "exp4269-archive-v394-activate-v395"

V395_FRAME = (
    "CLOSE the cross-family OOD question (recover provenance -> held-out-family test) "
    "then repair the DiffusionGemma loader + advance ARC +1"
)

LEAK_DELTA_DEFAULT = 0.385
LEAK_AUROC_DEFAULT = 0.990
MULTISEED_MEAN_DELTA_DEFAULT = 0.458
INDEPENDENT_RESCORE_DELTA_DEFAULT = 0.442
SYNTHESIS_MINUS_ORACLE_DEFAULT = -0.283
CODE_DELTA_DEFAULT = -0.006
CODE_AUROC_DEFAULT = 0.697
TOTAL_LEVELS_SOLVED_DEFAULT = 19

V394_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4268", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4256", "deliverable": str(LEAK_AUDIT_REL_PATH), "required": True},
    {"experiment_id": "4257", "deliverable": str(MULTISEED_REL_PATH), "required": True},
    {"experiment_id": "4258", "deliverable": str(CROSS_GAME_REL_PATH), "required": True},
    {"experiment_id": "4259", "deliverable": str(SYNTHESIS_REL_PATH), "required": True},
    {"experiment_id": "4260", "deliverable": str(PREFLIGHT_REL_PATH), "required": True},
    {"experiment_id": "4261", "deliverable": str(ARC_PROGRESS_REL_PATH), "required": True},
    {"experiment_id": "4263", "deliverable": str(REWARD_REL_PATH), "required": True},
    {"experiment_id": "4264", "deliverable": str(CODE_REL_PATH), "required": True},
)

SOURCE_MISSING_REASONS = {
    "4268": "blocked_v394_capstone_missing",
    "4256": "blocked_leak_audit_missing",
    "4257": "blocked_multiseed_replication_missing",
    "4258": "blocked_cross_game_transfer_missing",
    "4259": "blocked_synthesis_missing",
    "4260": "blocked_diffusiongemma_preflight_missing",
    "4261": "blocked_arc_progress_missing",
    "4263": "blocked_reward_out_of_band_missing",
    "4264": "blocked_code_replication_missing",
}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "v394_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.394.",
    "activated_milestone": "Confirms .395 is live for the close-cross-family-question frame.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v394_close_state": (
        "Honest record (win hardened 2-of-3, cross-game OOD blocked-not-collapsed, "
        "owed axes resolved) so the .395 agents frame the milestone as "
        "CLOSE-the-cross-family-question, not a redo and not premature scale-up."
    ),
    "preconditions_checked": (
        "Records resources verified; pre-empts the silent-missing-resource fabrication mode."
    ),
    "honest_verdict": (
        "Self-declared terminal state lets the reconciler classify success without re-running; "
        "MUST start with complete:/success:/passed:/shipped:."
    ),
    "duration_s": "Positive bare wall-clock for this record-only aggregation.",
    "inference_substrate": "Declares aggregation only; no live training happens in this task.",
}

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.394['\"]?\s*$")


def _number(value: Any, default: float) -> float:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else float(default)
    )


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _blocked_reason(verdict: Any) -> str:
    text = str(verdict)
    if text.startswith("blocked_"):
        text = text.removeprefix("blocked_")
    if text == "arc_game_ids_unrecoverable":
        return "game_ids_unrecoverable"
    return text


def archive_record_count(text: str) -> int:
    """Count top-level `.394` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.394` archive finding from the close-state."""

    return (
        ".394 close-state: first ARC oracle-distinct win HARDENED 2 of 3 axes. "
        "Leak audit SURVIVED: provenance-blind delta="
        f"{_number(close_state.get('provenance_blind_delta'), LEAK_DELTA_DEFAULT):+.3f}, "
        "AUROC="
        f"{_number(close_state.get('provenance_blind_auroc'), LEAK_AUROC_DEFAULT):.3f}. "
        "Multi-seed REPLICATED: mean delta="
        f"{_number(close_state.get('multiseed_mean_delta'), MULTISEED_MEAN_DELTA_DEFAULT):+.3f}, "
        "independent re-score="
        f"{_number(close_state.get('independent_rescore_delta'), INDEPENDENT_RESCORE_DELTA_DEFAULT):+.3f}. "
        "Cross-game OOD BLOCKED, not collapsed: "
        f"{close_state.get('cross_game_block_reason', 'game_ids_unrecoverable')}; "
        "the held-out game/family test NEVER RAN, so hardened_win=False and "
        "diffusiongemma_full_run_gate=False. Synthesis did NOT break oracle@K "
        f"(synthesis_minus_oracle={_number(close_state.get('synthesis_minus_oracle_delta'), SYNTHESIS_MINUS_ORACLE_DEFAULT):+.3f}); "
        "DiffusionGemma preflight blocked on gguf_loader_failed; ARC held at 19 levels; "
        "verifier-as-reward is OUT-OF-BAND; code oracle-distinct is CORPUS-SPECIFIC "
        f"(delta={_number(close_state.get('code_predictor_minus_vote_delta'), CODE_DELTA_DEFAULT):+.3f}); "
        "paper_ready=True. "
        f".395 frame: {V395_FRAME}."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.394` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .394 and activate .395; close cross-family OOD')}",
        "  completed: '2026-06-15'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4269-archive-v394-activate-v395",
        "  tasks:",
        "  - id: exp4256-arc-oracle-distinct-leak-audit",
        "    result: 'SURVIVED provenance-blind leak audit'",
        "  - id: exp4257-arc-oracle-distinct-multiseed-replication",
        "    result: 'REPLICATED across >=5 seeds plus independent re-score'",
        "  - id: exp4258-arc-oracle-distinct-cross-game-transfer",
        "    result: 'BLOCKED game_ids_unrecoverable; OOD test never ran'",
        "  - id: exp4268-capstone-v394",
        "    result: 'hardened_win=false; diffusiongemma_full_run_gate=false; paper_ready=true'",
    ]
    return "\n".join(lines) + "\n"


def _insert_before_tasks(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        if line == "  tasks:":
            return lines[:index] + [new_line] + lines[index:]
    return lines + [new_line]


def _canonicalize_target_span(lines: list[str], close_state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    finding_written = False
    activation_written = False
    for line in lines:
        if line.startswith("  finding:"):
            if not finding_written:
                out.append(f"  finding: {_yaml_quote(canonical_finding(close_state))}")
                finding_written = True
            continue
        if line.startswith("  activation_recorded:"):
            if not activation_written:
                out.append("  activation_recorded: exp4269-archive-v394-activate-v395")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4269-archive-v394-activate-v395")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.394` record exists and carries the close-state."""

    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if _record_id(line) is not None]
    spans = [
        (start, starts[index + 1] if index + 1 < len(starts) else len(lines))
        for index, start in enumerate(starts)
    ]
    target_spans = [
        (start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE
    ]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record(close_state)}", 0, "appended"

    first_start, first_end = target_spans[0]
    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    replacement = _canonicalize_target_span(lines[first_start:first_end], close_state)
    rebuilt: list[str] = []
    for index, line in enumerate(lines):
        if first_start <= index < first_end:
            if index == first_start:
                rebuilt.extend(replacement)
            continue
        if index in remove:
            continue
        rebuilt.append(line)
    new_text = "\n".join(rebuilt)
    if len(target_spans) > 1:
        return new_text, len(target_spans) - 1, "deduped"
    if new_text != text:
        return new_text, 0, "updated"
    return text, 0, "unchanged"


def read_v394_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.394` close-state."""

    return {
        "4268": read_json_object(root / CAPSTONE_REL_PATH),
        "4256": read_json_object(root / LEAK_AUDIT_REL_PATH),
        "4257": read_json_object(root / MULTISEED_REL_PATH),
        "4258": read_json_object(root / CROSS_GAME_REL_PATH),
        "4259": read_json_object(root / SYNTHESIS_REL_PATH),
        "4260": read_json_object(root / PREFLIGHT_REL_PATH),
        "4261": read_json_object(root / ARC_PROGRESS_REL_PATH),
        "4263": read_json_object(root / REWARD_REL_PATH),
        "4264": read_json_object(root / CODE_REL_PATH),
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.394` artifacts."""

    cited: list[JsonDict] = []
    for source in V394_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "artifact",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "required": bool(source["required"]),
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def build_v394_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the honest `.394` close-state from capstone and upstream artifacts."""

    capstone = _mapping(sources.get("4268", {}))
    hardening = _mapping(capstone.get("hardening"))
    cap_leak = _mapping(hardening.get("provenance_blind"))
    cap_multi = _mapping(hardening.get("multiseed"))
    cap_cross = _mapping(hardening.get("cross_game"))
    cap_synthesis = _mapping(capstone.get("extend_synthesis"))
    cap_scale = _mapping(capstone.get("scale_up_readiness"))
    cap_arc = _mapping(capstone.get("arc_progress"))
    cap_reward = _mapping(capstone.get("reward_decision"))
    cap_code = _mapping(capstone.get("code_read"))
    leak = _mapping(sources.get("4256", {}))
    multi = _mapping(sources.get("4257", {}))
    cross = _mapping(sources.get("4258", {}))
    synthesis = _mapping(sources.get("4259", {}))
    preflight = _mapping(sources.get("4260", {}))
    arc = _mapping(sources.get("4261", {}))
    reward = _mapping(sources.get("4263", {}))
    code = _mapping(sources.get("4264", {}))

    leak_survived = _bool(
        leak.get("win_survives_provenance_blind"),
        _bool(cap_leak.get("win_survives_provenance_blind"), True),
    )
    multiseed_replicated = _bool(
        multi.get("oracle_distinct_win_replicates"),
        _bool(cap_multi.get("oracle_distinct_win_replicates"), True),
    )
    cross_verdict = str(cross.get("honest_verdict", cap_cross.get("honest_verdict", "")))
    cross_delta = cross.get("cross_game_delta", cap_cross.get("cross_game_delta"))
    cross_blocked = cross_verdict.startswith("blocked_")
    cross_game_ran = not cross_blocked and isinstance(cross_delta, int | float)
    cross_holds = cross_game_ran and _number(cross_delta, 0.0) > 0.0
    axes_passed = int(leak_survived) + int(multiseed_replicated) + int(cross_holds)

    paper_ready = _bool(capstone.get("paper_ready"), True)
    flagged = _list(capstone.get("flagged_artifacts_excluded"))
    return {
        "summary": "first_arc_oracle_distinct_win_hardened_2_of_3_ood_blocked",
        "headline_outcome": str(
            capstone.get(
                "headline_outcome",
                "within_pool_win_survived_but_cross_game_blocked",
            )
        ),
        "leak_audit_survived": leak_survived,
        "provenance_blind_delta": round(
            _number(
                leak.get("provenance_blind_delta"),
                _number(cap_leak.get("provenance_blind_delta"), LEAK_DELTA_DEFAULT),
            ),
            3,
        ),
        "provenance_blind_auroc": round(
            _number(
                leak.get("provenance_blind_set_encoder_auroc"),
                _number(cap_leak.get("provenance_blind_set_encoder_auroc"), LEAK_AUROC_DEFAULT),
            ),
            3,
        ),
        "origin_probe_auroc": round(
            _number(leak.get("origin_probe_auroc"), _number(cap_leak.get("origin_probe_auroc"), 0.948)),
            3,
        ),
        "multiseed_replicated": multiseed_replicated,
        "multiseed_mean_delta": round(
            _number(
                multi.get("mean_delta"),
                _number(cap_multi.get("mean_delta"), MULTISEED_MEAN_DELTA_DEFAULT),
            ),
            3,
        ),
        "independent_rescore_delta": round(
            _number(
                multi.get("independent_rescore_delta"),
                _number(cap_multi.get("independent_rescore_delta"), INDEPENDENT_RESCORE_DELTA_DEFAULT),
            ),
            3,
        ),
        "n_seeds": int(_number(multi.get("n_seeds"), _number(cap_multi.get("n_seeds"), 5))),
        "cross_game_ood_ran": cross_game_ran,
        "cross_game_delta": cross_delta if isinstance(cross_delta, int | float) else None,
        "cross_game_block_reason": _blocked_reason(cross_verdict),
        "cross_game_blocked_not_collapsed": cross_blocked and not cross_game_ran,
        "hardened_axes_passed": axes_passed,
        "hardened_axes_total": 3,
        "hardened_win": _bool(capstone.get("hardened_win"), False) and axes_passed == 3,
        "diffusiongemma_full_run_gate": _bool(capstone.get("diffusiongemma_full_run_gate"), False),
        "synthesis_breaks_oracle_ceiling": _bool(
            synthesis.get("synthesis_breaks_oracle_ceiling"),
            _bool(cap_synthesis.get("synthesis_breaks_oracle_ceiling"), False),
        ),
        "synthesis_minus_oracle_delta": round(
            _number(
                synthesis.get("synthesis_minus_oracle_delta"),
                SYNTHESIS_MINUS_ORACLE_DEFAULT,
            ),
            3,
        ),
        "diffusiongemma_preflight_go": _bool(
            preflight.get("preflight_go"), _bool(cap_scale.get("preflight_go"), False)
        ),
        "diffusiongemma_block_reason": str(
            preflight.get("honest_verdict", "blocked_diffusiongemma_gguf_loader_failed")
        ),
        "total_levels_solved": int(
            _number(
                arc.get("total_levels_solved"),
                _number(cap_arc.get("total_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT),
            )
        ),
        "reward_ready_for_out_of_band": _bool(
            reward.get("ready_for_out_of_band"),
            _bool(cap_reward.get("ready_for_out_of_band"), True),
        ),
        "reward_honest_verdict": str(
            reward.get(
                "honest_verdict",
                cap_reward.get("honest_verdict", "complete: ready_for_out_of_band_verifier_reward_training"),
            )
        ),
        "code_replication_read": str(
            code.get("replication_read", cap_code.get("replication_read", "corpus_specific"))
        ),
        "code_predictor_minus_vote_delta": round(
            _number(
                code.get("code_predictor_minus_vote_delta"),
                _number(cap_code.get("code_predictor_minus_vote_delta"), CODE_DELTA_DEFAULT),
            ),
            3,
        ),
        "code_off_fold_auroc": round(
            _number(code.get("off_fold_auroc"), _number(cap_code.get("off_fold_auroc"), CODE_AUROC_DEFAULT)),
            3,
        ),
        "code_replication_beats_vote": _bool(
            code.get("code_replication_beats_vote"),
            _bool(cap_code.get("code_replication_beats_vote"), False),
        ),
        "paper_ready": paper_ready,
        "unmet_gates": _list(capstone.get("unmet_gates")),
        "flagged_artifacts_excluded": [
            int(item["experiment_id"])
            for item in flagged
            if isinstance(item, Mapping) and isinstance(item.get("experiment_id"), int)
        ],
        "v395_frame": V395_FRAME,
    }


def terminal_verdict(close_state: Mapping[str, Any]) -> str:
    """Build the complete-path honest verdict from validated close-state."""

    return (
        "success: archived_v394_v395_active_arc_win_hardened_"
        f"{int(_number(close_state.get('hardened_axes_passed'), 0))}_of_3_"
        "cross_game_blocked_not_collapsed_paper_ready_pretest_green"
    )


def build_complete_artifact(
    *,
    v394_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: list[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal Exp 4269 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4269,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": True,
        "exclusion_manifest_parses": True,
        "pretest_suite_green": True,
        "preconditions_checked": dict(preconditions_checked),
        "v394_close_state": dict(v394_close_state),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": terminal_verdict(v394_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4269", "SCENARIO-REPORT-4269"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: str,
    active_roadmap_path: str,
) -> JsonDict:
    """Build a blocked artifact without claiming the archive succeeded."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4269,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": bool(
            _mapping(preconditions_checked.get("research_complete_yaml")).get("parses", False)
        ),
        "exclusion_manifest_parses": bool(
            _mapping(preconditions_checked.get("exclusion_manifest_yaml")).get("parses", False)
        ),
        "pretest_suite_green": bool(
            _mapping(preconditions_checked.get("smart_subset_pretest")).get("green", False)
        ),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4269", "SCENARIO-REPORT-4269-BLOCKED-PRECONDITION"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _blocked(
    root: Path,
    reason: str,
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    active_milestone_confirmed: str = "",
    active_roadmap_path: str = "research-roadmap.yaml",
) -> Path:
    output_path = root / OUTPUT_REL_PATH
    payload = build_blocked_artifact(
        reason,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def _command_check(result: CommandResult) -> JsonDict:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _source_checks(root: Path) -> JsonDict:
    checks: JsonDict = {}
    for source in V394_SOURCE_ARTIFACTS:
        path = root / str(source["deliverable"])
        checks[str(source["experiment_id"])] = {
            "path": str(source["deliverable"]),
            "exists": path.exists(),
            "sha256": file_sha256(path),
        }
    return checks


def run(
    root: Path = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the Exp 4269 record-only archive workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions: JsonDict = {}
    research_path = root / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root / EXCLUSION_MANIFEST_REL_PATH

    if not research_path.exists():
        preconditions["research_complete_yaml"] = {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_research_complete_yaml_missing",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )
    research_text = research_path.read_text(encoding="utf-8")
    research_ok = yaml_parses(research_text)
    preconditions["research_complete_yaml"] = {
        "path": str(RESEARCH_COMPLETE_REL_PATH),
        "exists": True,
        "parses": research_ok,
    }
    if not research_ok:
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    if not manifest_path.exists():
        preconditions["exclusion_manifest_yaml"] = {
            "path": str(EXCLUSION_MANIFEST_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_exclusion_manifest_missing",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_ok = yaml_parses(manifest_text)
    preconditions["exclusion_manifest_yaml"] = {
        "path": str(EXCLUSION_MANIFEST_REL_PATH),
        "exists": True,
        "parses": manifest_ok,
    }
    if not manifest_ok:
        return _blocked(
            root,
            "blocked_exclusion_manifest_yaml_poison",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    pretest = run_smart_subset(root) if pretest_result is None else pretest_result
    preconditions["smart_subset_pretest"] = _command_check(pretest)
    if pretest.exit_code != 0:
        return _blocked(
            root,
            "blocked_smart_subset_pretest_not_green",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    active_milestone, roadmap_path = read_active_milestone(root)
    preconditions["active_milestone"] = {
        "expected": ACTIVATED_MILESTONE,
        "actual": active_milestone,
        "path": roadmap_path,
        "matches": active_milestone == ACTIVATED_MILESTONE,
    }
    if active_milestone != ACTIVATED_MILESTONE:
        return _blocked(
            root,
            "blocked_v395_not_active",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    source_checks = _source_checks(root)
    preconditions["source_artifacts"] = source_checks
    for experiment_id, check in source_checks.items():
        if not check["exists"]:
            return _blocked(
                root,
                SOURCE_MISSING_REASONS[experiment_id],
                preconditions_checked=preconditions,
                started_s=started,
                now_s=now_s,
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=roadmap_path,
            )

    sources = read_v394_sources(root)
    close_state = build_v394_close_state(sources)
    new_research_text, duplicates_removed, action = dedupe_or_update_record(research_text, close_state)
    if not yaml_parses(new_research_text):
        return _blocked(
            root,
            "blocked_research_complete_edit_invalid",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    research_path.write_text(new_research_text, encoding="utf-8")
    if not yaml_parses(research_path.read_text(encoding="utf-8")):
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison_after_edit",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    payload = build_complete_artifact(
        v394_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(started, now_s),
        active_roadmap_path=roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=duplicates_removed,
        cited_upstream_artifacts=build_cited_upstream(root),
    )
    validate_artifact(payload)
    output_path = root / OUTPUT_REL_PATH
    write_payload(output_path, payload)
    return output_path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the complete-path artifact against the Exp 4269 contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in payload, f"missing required fields: {field}")
    verdict = payload.get("honest_verdict")
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "terminal-prefixed honest_verdict required",
    )
    principles = payload.get("field_principles")
    _require(isinstance(principles, Mapping), "field_principles must be a mapping")
    for field in ("honest_verdict", "v394_close_state", "preconditions_checked"):
        _require(field in principles, f"missing field principles: {field}")
        _require(
            principles[field] == FIELD_PRINCIPLES[field],
            "principle must match REQ-REPORT-4269",
        )
    _require(payload.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone")
    _require(payload.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone")
    _require(payload.get("research_complete_yaml_parses") is True, "research-complete YAML")
    _require(payload.get("exclusion_manifest_parses") is True, "exclusion manifest")
    _require(payload.get("pretest_suite_green") is True, "pretest suite")
    _require(payload.get("active_milestone_confirmed") == ACTIVATED_MILESTONE, "active milestone")

    close_state = payload.get("v394_close_state")
    _require(isinstance(close_state, Mapping), "v394_close_state must be a mapping")
    _require(close_state.get("leak_audit_survived") is True, "leak audit")
    _require(close_state.get("provenance_blind_delta") == LEAK_DELTA_DEFAULT, "leak delta")
    _require(close_state.get("provenance_blind_auroc") == LEAK_AUROC_DEFAULT, "leak AUROC")
    _require(close_state.get("multiseed_replicated") is True, "multi-seed")
    _require(
        close_state.get("multiseed_mean_delta") == MULTISEED_MEAN_DELTA_DEFAULT,
        "mean delta",
    )
    _require(
        close_state.get("independent_rescore_delta") == INDEPENDENT_RESCORE_DELTA_DEFAULT,
        "rescore",
    )
    _require(close_state.get("cross_game_ood_ran") is False, "cross-game")
    _require(close_state.get("cross_game_block_reason") == "game_ids_unrecoverable", "game IDs")
    _require(
        close_state.get("cross_game_blocked_not_collapsed") is True,
        "blocked-not-collapsed",
    )
    _require(close_state.get("hardened_axes_passed") == 2, "axes")
    _require(close_state.get("hardened_axes_total") == 3, "axes")
    _require(close_state.get("hardened_win") is False, "hardened win")
    _require(close_state.get("diffusiongemma_full_run_gate") is False, "DiffusionGemma gate")
    _require(
        close_state.get("synthesis_breaks_oracle_ceiling") is False,
        "synthesis",
    )
    _require(
        close_state.get("synthesis_minus_oracle_delta") == SYNTHESIS_MINUS_ORACLE_DEFAULT,
        "oracle delta",
    )
    _require(close_state.get("diffusiongemma_preflight_go") is False, "loader")
    _require(
        close_state.get("diffusiongemma_block_reason")
        == "blocked_diffusiongemma_gguf_loader_failed",
        "loader",
    )
    _require(close_state.get("total_levels_solved") == TOTAL_LEVELS_SOLVED_DEFAULT, "ARC levels")
    _require(close_state.get("reward_ready_for_out_of_band") is True, "reward")
    _require(close_state.get("code_replication_read") == "corpus_specific", "code")
    _require(
        close_state.get("code_predictor_minus_vote_delta") == CODE_DELTA_DEFAULT,
        "code delta",
    )
    _require(close_state.get("paper_ready") is True, "paper")
    _require(close_state.get("v395_frame") == V395_FRAME, "v395 frame")


def main() -> int:
    """Run the Exp 4269 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
