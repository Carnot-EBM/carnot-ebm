"""Exp 5181: archive .474 and activate the .475 frame.

Spec refs: REQ-REPORT-5181, SCENARIO-REPORT-5181,
SCENARIO-REPORT-5181-BLOCKED-PRECONDITION.

This is a record-only aggregation module. It reads the completed `.474`
artifacts, confirms the scoped Phase D exclusion-manifest retirement, checks
that the `.475` roadmap is active, and writes the `.475` handoff artifact. It
does not modify `scripts/research_conductor.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5181_archive_474_activate_475.json")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5181_archive_474_activate_475"
EXPERIMENT_ID = "exp5181-archive-474-activate-475"
MILESTONE = "2026.07.475"
ARCHIVED_MILESTONE = "2026.07.474"
SCHEMA = "carnot.experiment_5181_archive_474_activate_475.v1"
RANDOM_SEED = 5181
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_474_closed_475_active_precise_handoff_clean"
BLOCKED_VERDICT = "complete_archive_474_activation_blocked_precondition"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
PHASE_D_ENTRY_ID = "phase_d_external_text_scorer_retired_exp5163_v474"
REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5181, 5193))

PHASE_D_RETIRED_EXP_IDS = (
    "exp4940",
    "exp5003",
    "exp5004",
    "exp5005",
    "exp5007",
    "exp5015",
    "exp5017",
    "exp5018",
    "exp5022",
    "exp5029",
    "exp5031",
    "exp5032",
    "exp5033",
    "exp5036",
    "exp5045",
    "exp5046",
    "exp5047",
    "exp5050",
    "exp5059",
    "exp5060",
    "exp5063",
    "exp5072",
    "exp5086",
    "exp5087",
    "exp5088",
    "exp5126",
    "exp5163",
)

SPEC_REFS = [
    "REQ-REPORT-5181",
    "SCENARIO-REPORT-5181",
    "SCENARIO-REPORT-5181-BLOCKED-PRECONDITION",
]

V474_RESULT_PATHS: dict[int, Path] = {
    5168: Path("results/experiment_5168_archive_473_activate_474.json"),
    5169: Path("results/experiment_5169_adversarial_verify_qd_citation_scope_fix_v474.json"),
    5170: Path("results/experiment_5170_retire_phase_d_external_text_scorer_v474.json"),
    5171: Path("results/experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json"),
    5172: Path("results/experiment_5172_sota_ingestion_diffusion_hierarchical_search_v474.json"),
    5173: Path("results/experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474.json"),
    5174: Path("results/experiment_5174_gap_live_integration_reconciliation_v474.json"),
    5175: Path("results/experiment_5175_gap4891_relational_mask_pruner_ab_v474.json"),
    5176: Path("results/experiment_5176_deepen_live_levelup_attempt_v474.json"),
    5177: Path("results/experiment_5177_gap4_scaleup_decentralization_tier_v474.json"),
    5178: Path("results/experiment_5178_hidden_state_verifier_pilot_v474.json"),
    5179: Path("results/experiment_5179_hardware_continuity_board_timing_v474.json"),
    5180: Path("results/experiment_5180_capstone_v474.json"),
}

TASK_IDS: dict[int, str] = {
    5168: "exp5168-archive-473-activate-474",
    5169: "exp5169-adversarial-verify-qd-citation-scope-fix-v474",
    5170: "exp5170-retire-phase-d-external-text-scorer-v474",
    5171: "exp5171-harden-set-encoder-cross-corpus-n30-v474",
    5172: "exp5172-sota-ingestion-diffusion-hierarchical-search-v474",
    5173: "exp5173-diffusiongemma-energy-guided-diffusion-pilot-v474",
    5174: "exp5174-gap-live-integration-reconciliation-v474",
    5175: "exp5175-gap4891-relational-mask-pruner-ab-v474",
    5176: "exp5176-deepen-live-levelup-attempt-v474",
    5177: "exp5177-gap4-scaleup-decentralization-tier-v474",
    5178: "exp5178-hidden-state-verifier-pilot-v474",
    5179: "exp5179-hardware-continuity-board-timing-v474",
    5180: "exp5180-capstone-v474",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "v474_summary": (
        "An inaccurate handoff summary propagates errors into every downstream .475 task's "
        "CONTEXT section; precision here is load-bearing for the whole milestone."
    ),
    "exclusion_manifest_confirmed_clean": (
        "The Phase D retirement must block only the retired external-text-scorer mechanism and "
        "must not false-positive against .475 hidden-state verifier or ARC-deepening tasks."
    ),
    "research_roadmap_yaml_activated": (
        "Downstream conductor work depends on `research-roadmap.yaml` naming the `.475` "
        "milestone and containing the Exp 5181-5192 task set."
    ),
    "architecture_md_staleness_days": (
        "Mechanical input to the Architecture Freshness Check; feeds exp5189's priority."
    ),
    "inference_substrate": "This archive reads upstream artifacts and lint outputs only.",
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "archived_milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "duration_s",
    "random_seed",
    "source_artifact_audit",
    "source_artifacts_read",
    "v474_task_rows",
    "phase_d_manifest_audit",
    "exclusion_manifest_lint",
    "publication_gate",
    "research_conductor_modified",
    "failed_preconditions",
    "clean_handoff",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5181_archive_474_activate_475.py -q -o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5181_archive_474_activate_475.py' -m pytest tests/python/test_experiment_5181_archive_474_activate_475.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5181_archive_474_activate_475.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5181_archive_474_activate_475.py",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/publication_gate.py --json",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return _unwrap(value.get("value"))
    return value


def _mapping(value: Any) -> JsonDict:
    raw = _unwrap(value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _raw_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    raw = _unwrap(value)
    return list(raw) if isinstance(raw, list) else []


def _bool(value: Any) -> bool:
    return _unwrap(value) is True


def _int(value: Any, default: int = 0) -> int:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str:
    raw = _unwrap(value)
    return str(raw if raw is not None else "")


def _principle(value: Any, field: str) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    status: JsonDict = {"path": str(path), "exists": path.exists(), "loadable": False}
    if not path.exists():
        return {}, status
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {}, {**status, "error": str(exc)}
    if not isinstance(loaded, Mapping):
        return {}, {**status, "error": "top-level JSON is not an object"}
    return dict(loaded), {**status, "loadable": True}


def _source_artifact_rows(root: Path, statuses: Mapping[int, JsonMap]) -> list[JsonDict]:
    rows = []
    for exp_id, relative_path in V474_RESULT_PATHS.items():
        status = _mapping(statuses.get(exp_id))
        path = root / relative_path
        rows.append(
            {
                "exp_id": exp_id,
                "path": str(relative_path),
                "exists": bool(status.get("exists")),
                "loadable": bool(status.get("loadable")),
                "sha256": file_sha256(path),
            }
        )
    return rows


def load_v474_results(root: Path) -> tuple[dict[int, JsonDict], dict[int, JsonDict], JsonDict]:
    payloads: dict[int, JsonDict] = {}
    statuses: dict[int, JsonDict] = {}
    for exp_id, relative_path in V474_RESULT_PATHS.items():
        payload, status = read_json_mapping(root / relative_path)
        statuses[exp_id] = status
        if status.get("loadable") is True:
            payloads[exp_id] = payload
    rows = _source_artifact_rows(root, statuses)
    return payloads, statuses, {
        "all_present": all(row["exists"] for row in rows),
        "all_loadable": all(row["loadable"] for row in rows),
        "missing_exp_ids": [row["exp_id"] for row in rows if not row["exists"]],
        "unloadable_exp_ids": [row["exp_id"] for row in rows if row["exists"] and not row["loadable"]],
        "rows": rows,
    }


def _honest_verdict(payload: JsonMap) -> str:
    return _str(payload.get("honest_verdict"))


def _task_row(root: Path, exp_id: int, payload: JsonMap) -> JsonDict:
    facts: JsonDict = {}
    if exp_id == 5168:
        facts = {
            "v473_runtime_clean": _bool(payload.get("v473_runtime_clean")),
            "exp5161_unquarantine_noted": _bool(payload.get("exp5161_unquarantine_noted")),
        }
    elif exp_id == 5169:
        backfill = _mapping(payload.get("backfill_dry_run_summary"))
        facts = {
            "exp5156_resolved": _bool(payload.get("exp5156_resolved")),
            "artifacts_newly_unflagged_count": _int(backfill.get("artifacts_newly_unflagged_count")),
            "artifacts_newly_flagged_count": _int(backfill.get("artifacts_newly_flagged_count")),
            "any_unexpected_unflag": _bool(backfill.get("any_unexpected_unflag")),
        }
    elif exp_id == 5170:
        lineage = _mapping(payload.get("lineage_stage_summary"))
        facts = {
            "phase_d_source_artifact_count": len(_list(payload.get("phase_d_artifacts_enumerated"))),
            "phase_d_artifacts_enumerated": _list(payload.get("phase_d_artifacts_enumerated")),
            "false_positive_check_against_exp5178": _bool(payload.get("false_positive_check_against_exp5178")),
            "synthetic_match_check_passed": _bool(payload.get("synthetic_match_check_passed")),
            "cleanest_point_estimate": _mapping(lineage.get("cleanest_point_estimate")),
            "terminal_continuation": _mapping(lineage.get("terminal_continuation")),
        }
    elif exp_id == 5171:
        facts = {
            "gate_passed": _bool(payload.get("gate_passed")),
            "held_out_task_n": _int(payload.get("held_out_task_n")),
            "delta": _float(payload.get("cross_corpus_delta_n30")),
            "ci95": _list(payload.get("cross_corpus_delta_ci95_n30")),
            "per_seed_deltas": _list(payload.get("per_seed_deltas")),
            "random_seeds_used": _list(payload.get("random_seeds_used")),
            "verifier_is_oracle": _bool(payload.get("verifier_is_oracle")),
        }
    elif exp_id == 5172:
        facts = {
            "bottom_line_recommendation_for_475": _str(payload.get("bottom_line_recommendation_for_475")),
            "map_gate": _str(_mapping(payload.get("map_paper_deep_read")).get("comparison_vs_relational_mask_pruner")),
        }
    elif exp_id == 5173:
        smoke = _mapping(_mapping(payload.get("preconditions")).get("smoke"))
        facts = {
            "arm_rows": _list(payload.get("arm_rows")),
            "smoke_success": _bool(smoke.get("success")),
            "smoke_error": _str(smoke.get("error")),
            "tried_count": len(_list(smoke.get("tried"))),
            "verifier_is_oracle": _unwrap(payload.get("verifier_is_oracle")),
            "corrigendum_pending": _list(payload.get("corrigendum_pending")),
            "flagged_adversarial": _bool(payload.get("flagged_adversarial")),
        }
    elif exp_id == 5174:
        audit = _mapping(payload.get("solve_provenance_audit"))
        facts = {
            "router_dsl_unimported_claim": _bool(payload.get("claim_router_dsl_unimported")),
            "target_levels_1_claim": _bool(payload.get("claim_target_levels_1")),
            "value_weight_0_claim": _bool(payload.get("claim_value_weight_0")),
            "live_agent_self_discovery_count": _int(audit.get("live_agent_self_discovery_count")),
            "development_proxy_count": _int(audit.get("development_proxy_count")),
            "out_of_registry_declared_games": _int(audit.get("out_of_registry_declared_games")),
        }
    elif exp_id == 5175:
        facts = {
            "games_tested": _list(payload.get("games_tested")),
            "states_expanded_unpruned": _mapping(payload.get("states_expanded_unpruned")),
            "states_expanded_pruned": _mapping(payload.get("states_expanded_pruned")),
            "move_pruned_edges": _mapping(payload.get("move_pruned_edges")),
            "levels_banked": _list(payload.get("levels_banked")),
            "next_specific_lever": _str(payload.get("next_specific_lever")),
        }
    elif exp_id == 5176:
        facts = {
            "lever_used": _str(payload.get("lever_used")),
            "levels_banked": _list(payload.get("levels_banked")),
            "reproducible_levels_delta": _int(payload.get("reproducible_levels_delta")),
        }
    elif exp_id == 5177:
        checkpoint_path = _str(payload.get("checkpoint_path"))
        facts = {
            "target_n": _int(payload.get("target_n")),
            "achieved_n": _int(payload.get("achieved_n")),
            "checkpoint_resume_used": _bool(payload.get("checkpoint_resume_used")),
            "checkpoint_path": checkpoint_path,
            "checkpoint_exists": bool(checkpoint_path and (root / checkpoint_path).exists()),
            "exact_test_discordant_wins": _int(payload.get("exact_test_discordant_wins")),
            "exact_test_discordant_losses": _int(payload.get("exact_test_discordant_losses")),
            "exact_test_p_value_two_sided": _float(payload.get("exact_test_p_value_two_sided")),
            "exact_test_passes_min6_rule": _bool(payload.get("exact_test_passes_min6_rule")),
            "local_generator_arm_result": _unwrap(payload.get("local_generator_arm_result")),
            "gap4_status_recommendation": _str(payload.get("gap4_status_recommendation")),
        }
    elif exp_id == 5178:
        facts = {
            "hidden_state_access_feasible": _bool(payload.get("hidden_state_access_feasible")),
            "design_path_taken": _str(payload.get("design_path_taken")),
            "tuned_sc_baseline_accuracy": _float(payload.get("tuned_sc_baseline_accuracy")),
            "hidden_state_verifier_accuracy": _float(payload.get("hidden_state_verifier_accuracy")),
            "accuracy_delta_ci95": _list(payload.get("accuracy_delta_ci95")),
            "pilot_n_questions": _int(payload.get("pilot_n_questions")),
            "pilot_n_candidates": _int(payload.get("pilot_n_candidates")),
            "oracle_at_k_accuracy": _float(payload.get("oracle_at_k_accuracy")),
            "verifier_is_oracle": _bool(payload.get("verifier_is_oracle")),
            "headroom_present": _bool(payload.get("headroom_present")),
            "flagged_adversarial": _bool(payload.get("flagged_adversarial")),
        }
    elif exp_id == 5179:
        gatemate = _mapping(payload.get("gatemate_result"))
        facts = {
            "boards_reachable_count": _int(payload.get("boards_reachable_count")),
            "kv260_reachable": _bool(_mapping(payload.get("kv260_result")).get("reachable")),
            "polarfire_reachable": _bool(_mapping(payload.get("polarfire_result")).get("reachable")),
            "gatemate_reachable": _bool(gatemate.get("reachable")),
            "gatemate_blocked_reason": _str(gatemate.get("blocked_reason")),
            "expected_idcode": _str(_mapping(gatemate.get("timing_output")).get("expected_idcode")),
            "conductor_modified": _bool(payload.get("conductor_modified")),
        }
    elif exp_id == 5180:
        registry = _mapping(payload.get("registry_reconciliation"))
        facts = {
            "flagged_adversarial": _bool(payload.get("flagged_adversarial")),
            "flagged_adversarial_artifacts_excluded": _list(payload.get("flagged_adversarial_artifacts_excluded")),
            "paper_ready": _bool(_mapping(payload.get("publication_gate")).get("paper_ready")),
            "unmet_gates": _list(_mapping(payload.get("publication_gate")).get("unmet_gates")),
            "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
            "reproducible_total_games": _int(registry.get("reproducible_total_games")),
            "delta_from_exp5175_exp5176": _int(registry.get("delta_from_exp5175_exp5176")),
            "phase_d_retirement_confirmed_clean": _bool(payload.get("phase_d_retirement_confirmed_clean")),
        }
    return {
        "exp_id": exp_id,
        "task_id": TASK_IDS[exp_id],
        "path": str(V474_RESULT_PATHS[exp_id]),
        "honest_verdict": _honest_verdict(payload),
        "key_facts": facts,
    }


def build_v474_task_rows(root: Path, payloads: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [_task_row(root, exp_id, payloads[exp_id]) for exp_id in sorted(payloads)]


def _summary_value(rows: Sequence[JsonMap]) -> str:
    by_exp = {_int(row.get("exp_id")): row for row in rows}

    def verdict(exp_id: int) -> str:
        return _str(_mapping(by_exp.get(exp_id)).get("honest_verdict")) or f"missing_exp{exp_id}"

    facts5170 = _mapping(_mapping(by_exp.get(5170)).get("key_facts"))
    facts5171 = _mapping(_mapping(by_exp.get(5171)).get("key_facts"))
    facts5175 = _mapping(_mapping(by_exp.get(5175)).get("key_facts"))
    facts5177 = _mapping(_mapping(by_exp.get(5177)).get("key_facts"))
    facts5178 = _mapping(_mapping(by_exp.get(5178)).get("key_facts"))
    facts5180 = _mapping(_mapping(by_exp.get(5180)).get("key_facts"))
    checkpoint_note = "checkpoint missing on disk" if facts5177.get("checkpoint_exists") is False else "checkpoint present"
    return (
        f".474 task verdicts: exp5168 `{verdict(5168)}`; exp5169 `{verdict(5169)}`; "
        f"exp5170 `{verdict(5170)}` retired Phase D external-text scoring with "
        f"{_int(facts5170.get('phase_d_source_artifact_count'))} source artifacts and 27 retired exp* IDs "
        f"while preserving hidden-state/internal-representation verifiers, ARC oracle-distinct work, and FoVer; "
        f"exp5171 `{verdict(5171)}` is the genuine win with gate_passed={facts5171.get('gate_passed')}, "
        f"n={facts5171.get('held_out_task_n')}, delta={facts5171.get('delta')}, CI95={facts5171.get('ci95')}, "
        f"per-seed deltas={facts5171.get('per_seed_deltas')}; exp5172 `{verdict(5172)}` specified the "
        f"CD82/SK48/SP80 MAP pruner-only vs map-only vs map-plus-pruner 4000-expansion gate; exp5173 "
        f"`{verdict(5173)}` was blocked before measurement with arm_rows=[] and the DiffusionGemma "
        f"CPU/disk-dispatch/meta-tensor loader failure; exp5174 `{verdict(5174)}` re-scoped GAP-LIVE-INTEGRATION "
        f"to 4/24 live-agent-self-discovery vs 20/24 development-proxy; exp5175 `{verdict(5175)}` pruned "
        f"edges {facts5175.get('move_pruned_edges')} but states_expanded stayed unchanged and zero levels banked; "
        f"exp5176 `{verdict(5176)}` banked zero levels with no validated lever; exp5177 `{verdict(5177)}` "
        f"reached n={facts5177.get('achieved_n')}/{facts5177.get('target_n')}, wins={facts5177.get('exact_test_discordant_wins')}, "
        f"losses={facts5177.get('exact_test_discordant_losses')}, p={facts5177.get('exact_test_p_value_two_sided')}, "
        f"min6={facts5177.get('exact_test_passes_min6_rule')}, {checkpoint_note}; exp5178 `{verdict(5178)}` "
        f"was a small hidden-state negative with n={facts5178.get('pilot_n_questions')} questions/"
        f"{facts5178.get('pilot_n_candidates')} candidates, hidden={facts5178.get('hidden_state_verifier_accuracy')} "
        f"vs tuned_sc={facts5178.get('tuned_sc_baseline_accuracy')}, CI={facts5178.get('accuracy_delta_ci95')}; "
        f"exp5179 `{verdict(5179)}` found KV260 and PolarFire reachable but GateMate IDCODE-blocked; "
        f"exp5180 `{verdict(5180)}` closed with no flagged headline artifacts, publication gate ready, and "
        f"reproducible_total_levels/games={facts5180.get('reproducible_total_levels')}/{facts5180.get('reproducible_total_games')}."
    )


def _roadmap_activation_check(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(ROADMAP_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "milestone": "missing",
        "task_ids": [],
        "missing_task_prefixes": list(REQUIRED_TASK_PREFIXES),
        "activated": False,
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "error": str(exc)}
    mapping = _mapping(loaded)
    tasks = _list(mapping.get("tasks"))
    task_ids = [_str(_mapping(task).get("id")) for task in tasks]
    missing = [prefix for prefix in REQUIRED_TASK_PREFIXES if not any(task_id.startswith(prefix) for task_id in task_ids)]
    milestone = _str(mapping.get("milestone"))
    return {
        **base,
        "exists": True,
        "parses": True,
        "milestone": milestone,
        "task_ids": task_ids,
        "missing_task_prefixes": missing,
        "activated": milestone == MILESTONE and not missing,
    }


def _architecture_staleness_days(path: Path, run_date: str) -> int:
    if not path.exists():
        return -1
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", path.read_text(encoding="utf-8"))
    if not match:
        return -1
    reconciled = date.fromisoformat(match.group(1))
    today = datetime.strptime(run_date, "%Y%m%d").date()
    return (today - reconciled).days


def _manifest_audit(path: Path, phase_d_sources: Sequence[Any]) -> JsonDict:
    base: JsonDict = {
        "path": str(MANIFEST_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "entry_found": False,
        "retired_exp_id_count": 0,
        "source_artifact_count": len(phase_d_sources),
        "expected_retired_exp_ids_match": False,
        "exceptions_preserved": False,
        "scope_is_external_text": False,
        "clean": False,
        "errors": [],
    }
    if not path.exists():
        return {**base, "errors": ["manifest_missing"]}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "errors": [str(exc)]}
    entries = _list(_mapping(loaded).get("retired_extras"))
    entry = next((_mapping(item) for item in entries if _mapping(item).get("id") == PHASE_D_ENTRY_ID), {})
    if not entry:
        return {**base, "exists": True, "parses": True, "errors": ["phase_d_entry_missing"]}
    ids = [_str(item) for item in _list(entry.get("experiment_ids"))]
    reason = _str(entry.get("reason")).lower()
    scope = _str(entry.get("experiment_scope")).lower()
    exceptions = all(token in reason for token in ("hidden-state/internal-representation", "arc oracle-distinct", "fover production ensemble"))
    scope_ok = "external-text" in scope and "off-arc" in scope
    ids_match = ids == list(PHASE_D_RETIRED_EXP_IDS)
    errors = []
    if not ids_match:
        errors.append("retired_exp_ids_mismatch")
    if not exceptions:
        errors.append("sanctioned_exceptions_missing")
    if not scope_ok:
        errors.append("scope_not_external_text_off_arc")
    return {
        **base,
        "exists": True,
        "parses": True,
        "entry_found": True,
        "entry": entry,
        "retired_exp_id_count": len(ids),
        "expected_retired_exp_ids_match": ids_match,
        "exceptions_preserved": exceptions,
        "scope_is_external_text": scope_ok,
        "clean": not errors,
        "errors": errors,
    }


def _command_clean(result: CommandResult) -> bool:
    return result.exit_code == 0 and "HARD" not in (result.stdout + result.stderr)


def _lint_audit(result: CommandResult) -> JsonDict:
    combined = result.stdout + result.stderr
    hard_lines = [line for line in combined.splitlines() if "HARD" in line]
    return {
        "command": list(result.command),
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "hard_lines": hard_lines,
        "hidden_state_or_arc_deepening_hard_risk": any(("exp5185" in line or "exp5187" in line) for line in hard_lines),
        "clean": _command_clean(result),
    }


def _publication_gate_clean(publication_gate: JsonMap) -> bool:
    return publication_gate.get("paper_ready") is True and publication_gate.get("unmet_gates") == []


def _conductor_modified(root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", str(CONDUCTOR_RELATIVE_PATH)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 1


def _python_executable(root: Path) -> str:
    venv_python = root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def run_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - thin subprocess wrapper
    command = [_python_executable(root), "scripts/publication_gate.py", "--json"]
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        return {"paper_ready": False, "unmet_gates": ["publication_gate_command_failed"], "stderr": result.stderr}
    try:
        loaded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"paper_ready": False, "unmet_gates": ["publication_gate_json_invalid"], "error": str(exc)}
    return dict(loaded) if isinstance(loaded, Mapping) else {"paper_ready": False, "unmet_gates": ["publication_gate_not_object"]}


def run_exclusion_manifest_lint(root: Path) -> CommandResult:  # pragma: no cover - thin subprocess wrapper
    command = (_python_executable(root), "scripts/exclusion_manifest_lint.py", str(ROADMAP_RELATIVE_PATH))
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return CommandResult(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _failed_preconditions(
    *,
    source_audit: JsonMap,
    manifest_audit: JsonMap,
    lint_audit: JsonMap,
    roadmap_check: JsonMap,
    architecture_days: int,
    publication_gate: JsonMap,
    conductor_modified: bool,
) -> list[str]:
    failures = []
    if not source_audit.get("all_present") or not source_audit.get("all_loadable"):
        failures.append("v474_artifacts_missing_or_unloadable")
    if not manifest_audit.get("clean"):
        failures.append("phase_d_manifest_entry_absent_or_overbroad")
    if not lint_audit.get("clean"):
        failures.append("exclusion_manifest_lint_not_clean")
    if not roadmap_check.get("activated"):
        failures.append("research_roadmap_yaml_not_activated_to_475")
    if architecture_days < 0:
        failures.append("architecture_last_reconciled_unreadable")
    if not _publication_gate_clean(publication_gate):
        failures.append("publication_gate_not_ready")
    if conductor_modified:
        failures.append("scripts_research_conductor_py_modified")
    return failures


def build_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    publication_gate: JsonMap,
    exclusion_lint: CommandResult,
    tests_run: Sequence[str],
) -> JsonDict:
    payloads, _statuses, source_audit = load_v474_results(root)
    rows = build_v474_task_rows(root, payloads)
    phase_d_sources = _mapping(next((row for row in rows if row["exp_id"] == 5170), {})).get("key_facts", {}).get(
        "phase_d_artifacts_enumerated",
        [],
    )
    manifest_audit = _manifest_audit(root / MANIFEST_RELATIVE_PATH, _list(phase_d_sources))
    lint_audit = _lint_audit(exclusion_lint)
    roadmap_check = _roadmap_activation_check(root / ROADMAP_RELATIVE_PATH)
    architecture_days = _architecture_staleness_days(root / ARCHITECTURE_RELATIVE_PATH, run_date)
    conductor_modified = _conductor_modified(root)
    manifest_clean = bool(manifest_audit.get("clean")) and bool(lint_audit.get("clean"))
    failures = _failed_preconditions(
        source_audit=source_audit,
        manifest_audit=manifest_audit,
        lint_audit=lint_audit,
        roadmap_check=roadmap_check,
        architecture_days=architecture_days,
        publication_gate=publication_gate,
        conductor_modified=conductor_modified,
    )
    clean_handoff = not failures
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_artifact_audit": source_audit,
        "source_artifacts_read": source_audit["rows"],
        "v474_task_rows": rows,
        "phase_d_manifest_audit": manifest_audit,
        "exclusion_manifest_lint": lint_audit,
        "publication_gate": dict(publication_gate),
        "research_conductor_modified": conductor_modified,
        "failed_preconditions": failures,
        "clean_handoff": clean_handoff,
        "tests_run": list(tests_run),
        "v474_summary": _principle(_summary_value(rows), "v474_summary"),
        "exclusion_manifest_confirmed_clean": _principle(manifest_clean, "exclusion_manifest_confirmed_clean"),
        "research_roadmap_yaml_activated": _principle(bool(roadmap_check.get("activated")), "research_roadmap_yaml_activated"),
        "architecture_md_staleness_days": _principle(architecture_days, "architecture_md_staleness_days"),
        "inference_substrate": _principle(INFERENCE_SUBSTRATE, "inference_substrate"),
        "honest_verdict": _principle(COMPLETE_VERDICT if clean_handoff else BLOCKED_VERDICT, "honest_verdict"),
        "roadmap_activation_check": roadmap_check,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum({**artifact, "reproducibility_checksum": ""})
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = _raw_mapping(artifact.get(field))
        if wrapped.get("principle") != principle:
            errors.append(f"{field} principle mismatch")
        if "value" not in wrapped:
            errors.append(f"{field} missing value")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        errors.append("archived_milestone mismatch")
    if _mapping(artifact.get("field_principles")) != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if _raw_mapping(artifact.get("inference_substrate")).get("value") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    verdict = _str(_raw_mapping(artifact.get("honest_verdict")).get("value"))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict terminal prefix missing")
    if not _str(_raw_mapping(artifact.get("v474_summary")).get("value")):
        errors.append("v474_summary empty")
    if not isinstance(_raw_mapping(artifact.get("exclusion_manifest_confirmed_clean")).get("value"), bool):
        errors.append("exclusion_manifest_confirmed_clean not bool")
    if not isinstance(_raw_mapping(artifact.get("research_roadmap_yaml_activated")).get("value"), bool):
        errors.append("research_roadmap_yaml_activated not bool")
    if not isinstance(_raw_mapping(artifact.get("architecture_md_staleness_days")).get("value"), int):
        errors.append("architecture_md_staleness_days not int")
    if not _list(artifact.get("v474_task_rows")):
        errors.append("v474_task_rows empty")
    if not _mapping(artifact.get("source_artifact_audit")):
        errors.append("source_artifact_audit missing")
    if not _publication_gate_clean(_mapping(artifact.get("publication_gate"))):
        errors.append("publication_gate not clean")
    checksum = _str(artifact.get("reproducibility_checksum"))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum invalid")
    if errors:
        raise ValueError("invalid Exp 5181 archive artifact: " + "; ".join(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    run_date: str | None = None,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    clock: Any = time.perf_counter,
) -> Path:
    started = float(clock())
    publication_gate = run_publication_gate(root)
    exclusion_lint = run_exclusion_manifest_lint(root)
    finished = float(clock())
    artifact = build_artifact(
        root=root,
        duration_s=max(finished - started, 0.000001),
        run_date=run_date or date.today().strftime("%Y%m%d"),
        publication_gate=publication_gate,
        exclusion_lint=exclusion_lint,
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    destination = output or (root / RESULT_RELATIVE_PATH)
    write_json(destination, artifact)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", dest="run_date", default=None)
    args = parser.parse_args(argv)
    path = run(root=args.root, output=args.output, run_date=args.run_date)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
