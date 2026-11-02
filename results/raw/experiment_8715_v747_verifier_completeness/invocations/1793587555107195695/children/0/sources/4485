"""Build the Exp 4323 v399 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4323, SCENARIO-CAPSTONE-4323.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import capstone_aggregate_available as aggregate


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_4323_capstone_v399.json")
EXPERIMENT_ID = 4323
RANDOM_SEED = 4323
SCHEMA = "carnot.capstone_v399_4323.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4323", "SCENARIO-CAPSTONE-4323"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_no_v399_artifacts").hexdigest()

THESIS_STATES = {
    "cross_domain_moat_holds",
    "in_generation_moat_holds",
    "efficiency_cascade_deployed",
    "selection_moat_arc_only",
    "in_generation_still_open",
    "two_moats_still_open",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4314_cross_domain": Upstream(
        4314, Path("results/experiment_4314_cross_domain_selector_ir3de_cascal.json")
    ),
    "4315_in_generation": Upstream(
        4315, Path("results/experiment_4315_diffusiongemma_reward_guided_stitching.json")
    ),
    "4316_efficiency": Upstream(
        4316, Path("results/experiment_4316_efficiency_cascade_router_deploy.json")
    ),
    "4317_arc": Upstream(
        4317, Path("results/experiment_4317_arc_incremental_progress_adapter_free.json")
    ),
    "4318_self_learning": Upstream(
        4318, Path("results/experiment_4318_arc_cross_game_learned_verifier_transfer.json")
    ),
    "4319_off_arc": Upstream(
        4319, Path("results/experiment_4319_off_arc_execution_verifier_transfer_accumulate.json")
    ),
    "4321_registry": Upstream(
        4321, Path("results/experiment_4321_verifier_registry_gaps_hygiene.json")
    ),
    "4322_hardware": Upstream(4322, Path("results/experiment_4322_hardware_continuity.json")),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "cross_domain_moat_holds",
    "in_generation_moat_holds",
    "efficiency_cascade_dominates",
    "verifier_thesis_state",
    "flagged_artifacts_excluded",
    "per_axis_gaps",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .399 close-state -- whether the cross-domain moat closed, "
        "the in-generation moat closed, the efficiency cascade deployed."
    ),
    "headline_outcome": (
        "One honest string aggregating the cross-domain + in-generation + efficiency + "
        "ARC + self-learning + off-ARC reads; the single line the .400 planner frames from."
    ),
    "cross_domain_moat_holds": (
        "BARE bool: did the IR3DE+CASCAL rebuild close the cross-domain moat "
        "(held-out-DOMAIN CI95-excl-0, label_ablation_robust) -- escaping the "
        "verifier's math/ARC domain bound."
    ),
    "in_generation_moat_holds": (
        "BARE bool: did reward-guided stitching close the in-generation moat (beats the "
        "engaged control AND self-reward SMC, controls_differentiated, CI95-excl-0) -- "
        "the external verifier steers generation better than intrinsic confidence."
    ),
    "efficiency_cascade_dominates": (
        "BARE bool: did the budget-aware cascade router reach near-judge accuracy at a "
        "fraction of cost -- the deployed §5 efficiency operating point."
    ),
    "verifier_thesis_state": (
        "One honest string for the verifier thesis state -- the framing the .400 planner "
        "inherits."
    ),
    "flagged_artifacts_excluded": (
        "List of .399 artifacts excluded for flagged_adversarial -- the fabrication gate "
        "(their numbers are NOT aggregated)."
    ),
    "per_axis_gaps": (
        "List of .399 axes whose artifact was MISSING (reported as a gap, NOT defaulted "
        "False) -- the robust-aggregator fix that prevents the .397 spurious-all-False "
        "capstone bug."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline stays the "
        "publication target; a closed cross-domain or in-generation moat, or a deployed "
        "cascade router, would be a new headline-grade supporting result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited moat/headline result carried "
        "verifier_is_oracle=false (no circular/execution-grounded result -- e.g. exp4319 "
        "off-ARC -- headlines a moat)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party re-derive the "
        "capstone."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4314_cross_domain": [
        "cross_domain_selection_holds",
        "cross_domain_delta",
        "cross_domain_delta_ci95",
        "label_ablation_robust",
        "held_out_task_n",
        "primary_held_out_domain",
        "verifier_is_oracle",
    ],
    "4315_in_generation": [
        "diffusiongemma_guidance_moat",
        "controls_differentiated",
        "carnot_minus_best_control_delta",
        "carnot_minus_self_reward_smc_delta",
        "guidance_moat_ci95",
        "verifier_is_oracle",
    ],
    "4316_efficiency": [
        "cascade_dominates_controls",
        "accuracy_cascade",
        "accuracy_always_energy",
        "accuracy_always_judge",
        "cost_ratio_cascade",
        "escalation_rate",
        "verifier_is_oracle",
    ],
    "4317_arc": [
        "acceptance_gate_passed",
        "total_levels",
        "total_levels_solved",
        "offline_reproduced",
        "levels_completed",
        "new_levels_solved_this_task",
    ],
    "4318_self_learning": [
        "acceptance_gate_passed",
        "cross_game_transfer_helps",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "n_held_out_levels",
        "verifier_is_oracle",
    ],
    "4319_off_arc": [
        "off_arc_demofit_beats_vote",
        "off_arc_demofit_minus_vote_delta",
        "off_arc_delta_ci95",
        "accumulated_n",
        "verifier_is_oracle",
    ],
    "4321_registry": [
        "regression_guard_passed",
        "registry_reconciled",
        "manifest_reconciled",
    ],
    "4322_hardware": [
        "kv260",
        "polarfire",
        "gatemate",
    ],
}


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, bool) else None


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_metric(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_from_payload_checksum(payload: Mapping[str, Any]) -> str:
    value = payload.get("reproducibility_checksum")
    if not isinstance(value, str):
        return ""
    normalized = value.removeprefix("sha256:")
    return normalized if is_sha256(normalized) else ""


def live_has_critical(flags: list[dict[str, Any]]) -> bool:
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def run_live_flags(path: Path) -> list[dict[str, Any]]:  # pragma: no cover
    return list(av.verify_artifact(path).get("flags", []))


def run_summarize_artifact(path: Path, root: Path) -> int:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def run_publication_gate(root: Path) -> JsonDict:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json returned a non-object payload")
    return payload


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("non-object")
    return payload


def clean_payload(payload: JsonDict | None, skipped: bool) -> JsonDict | None:
    return None if skipped or payload is None else payload


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _safe_summarize(path: Path, root: Path, runner: SummarizeRunner) -> tuple[int | None, str]:
    try:
        return runner(path, root), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _safe_live_flags(path: Path, runner: LiveFlagRunner) -> list[dict[str, Any]]:
    try:
        return runner(path)
    except Exception as exc:
        return [{"kind": "VERIFY_ERROR", "severity": "warn", "detail": str(exc)}]


def _skipped_payload(payload: JsonDict) -> JsonDict:
    skipped = dict(payload)
    skipped["flagged_adversarial"] = True
    return skipped


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict], int]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    present_count = 0

    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            raw_artifacts[key] = None
            continue
        present_count += 1
        sha = sha256_file(path)
        summarize_exit_code, summarize_error = _safe_summarize(path, root, summarize_runner)
        live_flags = _safe_live_flags(path, live_flag_runner)
        critical = live_has_critical(live_flags)
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = _skipped_payload(payload) if payload is not None and skipped else payload
        provenance.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "sha256": sha,
                "payload_reproducibility_checksum": sha_from_payload_checksum(payload or {}),
                "summarize_exit_code": summarize_exit_code,
                "summarize_error": summarize_error,
                "live_adversarial_flags": live_flags,
                "stamped_flagged_adversarial": stamped,
                "live_critical": critical,
                "parse_error": parse_error,
                "skipped": skipped,
                "fields_imported": _fields_for_payload(key, skipped),
            }
        )
        if skipped:
            exclusions.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "parse_error": parse_error,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": _exclusion_reason(stamped, critical, parse_error),
                }
            )
    return raw_artifacts, provenance, exclusions, present_count


def _exclusion_reason(stamped: bool, critical: bool, parse_error: str) -> str:
    if stamped:
        return "flagged_adversarial"
    if critical:
        return "live_critical_adversarial"
    if parse_error:
        return "unparsable_or_non_object"
    return "excluded"


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="cross_domain",
            required_keys=("4314_cross_domain",),
            verdict_fn=lambda present: cross_domain_read(
                present.get("4314_cross_domain"), False
            )["cross_domain_moat_holds"]
            is True,
        ),
        aggregate.AxisSpec(
            name="in_generation",
            required_keys=("4315_in_generation",),
            verdict_fn=lambda present: in_generation_read(
                present.get("4315_in_generation"), False
            )["in_generation_moat_holds"]
            is True,
        ),
        aggregate.AxisSpec(
            name="efficiency",
            required_keys=("4316_efficiency",),
            verdict_fn=lambda present: efficiency_read(
                present.get("4316_efficiency"), False
            )["efficiency_cascade_dominates"]
            is True,
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4317_arc",),
            verdict_fn=lambda present: int_metric(present.get("4317_arc"), "total_levels") > 0
            and bool_metric(present.get("4317_arc"), "offline_reproduced") is True,
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4318_self_learning",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4318_self_learning"), False
            )["cross_game_transfer_helps"]
            is True,
        ),
        aggregate.AxisSpec(
            name="off_arc",
            required_keys=("4319_off_arc",),
            verdict_fn=lambda present: bool_metric(
                present.get("4319_off_arc"), "off_arc_demofit_beats_vote"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="registry",
            required_keys=("4321_registry",),
            verdict_fn=lambda present: registry_read(present.get("4321_registry"), False)[
                "registry_reconciled"
            ]
            is True,
        ),
        aggregate.AxisSpec(
            name="hardware",
            required_keys=("4322_hardware",),
            verdict_fn=lambda present: hardware_read(present.get("4322_hardware"), False)[
                "hardware_continuity_recorded"
            ]
            is True,
        ),
    ]


def cross_domain_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    moat = (
        bool_metric(payload, "cross_domain_selection_holds") is True
        and bool_metric(payload, "label_ablation_robust") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "moat_holds" if moat else "open",
        "cross_domain_moat_holds": moat,
        "reported_cross_domain_selection_holds": bool_metric(
            payload, "cross_domain_selection_holds"
        ),
        "cross_domain_delta": float_metric(payload, "cross_domain_delta"),
        "cross_domain_delta_ci95": list_metric(payload, "cross_domain_delta_ci95"),
        "label_ablation_robust": bool_metric(payload, "label_ablation_robust"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "primary_held_out_domain": str_metric(payload, "primary_held_out_domain"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def in_generation_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    moat = (
        bool_metric(payload, "diffusiongemma_guidance_moat") is True
        and bool_metric(payload, "controls_differentiated") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "moat_holds" if moat else "open",
        "in_generation_moat_holds": moat,
        "reported_diffusiongemma_guidance_moat": bool_metric(
            payload, "diffusiongemma_guidance_moat"
        ),
        "controls_differentiated": bool_metric(payload, "controls_differentiated"),
        "carnot_minus_best_control_delta": float_metric(
            payload, "carnot_minus_best_control_delta"
        ),
        "carnot_minus_self_reward_smc_delta": float_metric(
            payload, "carnot_minus_self_reward_smc_delta"
        ),
        "guidance_moat_ci95": list_metric(payload, "guidance_moat_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def efficiency_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    dominates = (
        bool_metric(payload, "cascade_dominates_controls") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "dominates" if dominates else "open",
        "efficiency_cascade_dominates": dominates,
        "reported_cascade_dominates_controls": bool_metric(
            payload, "cascade_dominates_controls"
        ),
        "accuracy_cascade": float_metric(payload, "accuracy_cascade"),
        "accuracy_always_energy": float_metric(payload, "accuracy_always_energy"),
        "accuracy_always_judge": float_metric(payload, "accuracy_always_judge"),
        "cost_ratio_cascade": float_metric(payload, "cost_ratio_cascade"),
        "escalation_rate": float_metric(payload, "escalation_rate"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def arc_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    total = int_metric(payload, "total_levels") or int_metric(payload, "total_levels_solved")
    return {
        "status": "included",
        "acceptance_gate_passed": bool_metric(payload, "acceptance_gate_passed"),
        "total_levels": total,
        "total_levels_solved": total,
        "levels_completed": int_metric(payload, "levels_completed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "offline_reproduced": bool_metric(payload, "offline_reproduced"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    helps = (
        bool_metric(payload, "cross_game_transfer_helps") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "helps" if helps else "open",
        "cross_game_transfer_helps": helps,
        "reported_cross_game_transfer_helps": bool_metric(payload, "cross_game_transfer_helps"),
        "acceptance_gate_passed": bool_metric(payload, "acceptance_gate_passed"),
        "cross_game_state_reduction": float_metric(payload, "cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": list_metric(
            payload, "cross_game_state_reduction_ci95"
        ),
        "n_held_out_levels": int_metric(payload, "n_held_out_levels"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def off_arc_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    won = bool_metric(payload, "off_arc_demofit_beats_vote") is True
    verifier_is_oracle = bool_metric(payload, "verifier_is_oracle")
    if won and verifier_is_oracle is True:
        status = "execution_grounded_win"
    elif won and verifier_is_oracle is False:
        status = "oracle_distinct_win"
    else:
        status = "open"
    return {
        "status": status,
        "off_arc_demofit_beats_vote": won,
        "off_arc_demofit_minus_vote_delta": float_metric(
            payload, "off_arc_demofit_minus_vote_delta"
        ),
        "off_arc_delta_ci95": list_metric(payload, "off_arc_delta_ci95"),
        "accumulated_n": int_metric(payload, "accumulated_n"),
        "verifier_is_oracle": verifier_is_oracle,
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def registry_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    reconciled = bool_metric(payload, "registry_reconciled") is True
    return {
        "status": "reconciled" if reconciled else "recorded",
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed"),
        "registry_reconciled": reconciled,
        "manifest_reconciled": bool_metric(payload, "manifest_reconciled"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def hardware_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    recorded = bool(str_metric(payload, "honest_verdict"))
    return {
        "status": "recorded" if recorded else "unrecorded",
        "hardware_continuity_recorded": recorded,
        "kv260": payload.get("kv260") if isinstance(payload.get("kv260"), Mapping) else {},
        "polarfire": payload.get("polarfire") if isinstance(payload.get("polarfire"), Mapping) else {},
        "gatemate": payload.get("gatemate") if isinstance(payload.get("gatemate"), Mapping) else {},
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def verifier_thesis_state(
    cross_domain_holds: bool,
    in_generation_holds: bool,
    efficiency_dominates: bool,
    selection_axis_available: bool,
    in_generation_axis_available: bool,
    efficiency_axis_available: bool,
) -> str:
    if cross_domain_holds:
        return "cross_domain_moat_holds"
    if in_generation_holds:
        return "in_generation_moat_holds"
    if efficiency_dominates:
        return "efficiency_cascade_deployed"
    if selection_axis_available and not in_generation_axis_available:
        return "selection_moat_arc_only"
    if not selection_axis_available and in_generation_axis_available:
        return "in_generation_still_open"
    if selection_axis_available and in_generation_axis_available:
        return "two_moats_still_open"
    if efficiency_axis_available:
        return "in_generation_still_open"
    return "two_moats_still_open"


def _part(read: Mapping[str, Any], true_key: str, true_part: str, false_part: str) -> str:
    if read.get(true_key) is True:
        return true_part
    status = str(read.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return false_part


def _arc_part(arc: Mapping[str, Any]) -> str:
    status = str(arc.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return f"levels_{int(arc.get('total_levels') or 0)}"


def _off_arc_part(off_arc: Mapping[str, Any]) -> str:
    status = str(off_arc.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return status


def _headline_outcome(
    cross_domain: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    arc: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    off_arc: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return (
        "cross_domain_"
        f"{_part(cross_domain, 'cross_domain_moat_holds', 'moat', 'open')}__"
        "in_generation_"
        f"{_part(in_generation, 'in_generation_moat_holds', 'moat', 'open')}__"
        f"efficiency_{_part(efficiency, 'efficiency_cascade_dominates', 'deployed', 'open')}__"
        f"arc_{_arc_part(arc)}__"
        f"self_learning_{_part(self_learning, 'cross_game_transfer_helps', 'helps', 'open')}__"
        f"off_arc_{_off_arc_part(off_arc)}__{paper}"
    )


def _honest_verdict(
    cross_domain: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    arc: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    off_arc: Mapping[str, Any],
) -> str:
    return (
        "complete: v399_cross_domain_"
        f"{_part(cross_domain, 'cross_domain_moat_holds', 'moat', 'open')}_"
        "in_generation_"
        f"{_part(in_generation, 'in_generation_moat_holds', 'moat', 'open')}_"
        f"efficiency_{_part(efficiency, 'efficiency_cascade_dominates', 'deployed', 'open')}_"
        f"arc_{_arc_part(arc)}_"
        f"self_learning_{_part(self_learning, 'cross_game_transfer_helps', 'helps', 'open')}_"
        f"off_arc_{_off_arc_part(off_arc)}"
    )


def _oracle_violations(
    cross_domain: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        cross_domain.get("reported_cross_domain_selection_holds") is True
        and cross_domain.get("verifier_is_oracle") is not False
    ):
        violations.append("4314_cross_domain:cross_domain")
    if (
        in_generation.get("reported_diffusiongemma_guidance_moat") is True
        and in_generation.get("verifier_is_oracle") is not False
    ):
        violations.append("4315_in_generation:in_generation")
    if (
        efficiency.get("reported_cascade_dominates_controls") is True
        and efficiency.get("verifier_is_oracle") is not False
    ):
        violations.append("4316_efficiency:efficiency_cascade")
    return violations


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return BLOCKED_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def _field_provenance(satisfied_by: str) -> dict[str, JsonDict]:
    return {
        field: {"principle": principle, "satisfied_by": satisfied_by}
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _available(read: Mapping[str, Any]) -> bool:
    return read.get("status") not in {"missing_or_excluded", "excluded_flagged_adversarial"}


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    raw_artifacts, provenance, exclusions, present_count = _read_inputs(
        root, live_flag_runner, summarize_runner
    )
    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }

    cross_domain = cross_domain_read(
        clean["4314_cross_domain"], skipped.get("4314_cross_domain", False)
    )
    in_generation = in_generation_read(
        clean["4315_in_generation"], skipped.get("4315_in_generation", False)
    )
    efficiency = efficiency_read(clean["4316_efficiency"], skipped.get("4316_efficiency", False))
    arc = arc_read(clean["4317_arc"], skipped.get("4317_arc", False))
    self_learning = self_learning_read(
        clean["4318_self_learning"], skipped.get("4318_self_learning", False)
    )
    off_arc = off_arc_read(clean["4319_off_arc"], skipped.get("4319_off_arc", False))
    registry = registry_read(clean["4321_registry"], skipped.get("4321_registry", False))
    hardware = hardware_read(clean["4322_hardware"], skipped.get("4322_hardware", False))

    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    violations = _oracle_violations(cross_domain, in_generation, efficiency)
    thesis = verifier_thesis_state(
        cross_domain.get("cross_domain_moat_holds") is True,
        in_generation.get("in_generation_moat_holds") is True,
        efficiency.get("efficiency_cascade_dominates") is True,
        _available(cross_domain),
        _available(in_generation),
        _available(efficiency),
    )
    end = time.time() if now_s is None else now_s
    blocked = present_count == 0

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "blocked_no_v399_artifacts"
            if blocked
            else _honest_verdict(cross_domain, in_generation, efficiency, arc, self_learning, off_arc)
        ),
        "headline_outcome": (
            "blocked_no_v399_artifacts"
            if blocked
            else _headline_outcome(
                cross_domain,
                in_generation,
                efficiency,
                arc,
                self_learning,
                off_arc,
                paper_ready,
            )
        ),
        "cross_domain_moat_holds": cross_domain.get("cross_domain_moat_holds") is True,
        "in_generation_moat_holds": in_generation.get("in_generation_moat_holds") is True,
        "efficiency_cascade_dominates": efficiency.get("efficiency_cascade_dominates") is True,
        "verifier_thesis_state": "two_moats_still_open" if blocked else thesis,
        "flagged_artifacts_excluded": exclusions,
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "paper_ready": paper_ready,
        "unmet_gates": list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "cross_domain": cross_domain,
        "in_generation": in_generation,
        "efficiency": efficiency,
        "arc": arc,
        "self_learning": self_learning,
        "off_arc": off_arc,
        "registry": registry,
        "hardware": hardware,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "reproducibility_checksum": checksum_from_provenance([] if blocked else provenance),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(
            "blocked precondition" if blocked else "aggregation logic"
        ),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if verdict != "blocked_no_v399_artifacts":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    headline = artifact.get("headline_outcome")
    if not isinstance(headline, str) or not headline:
        raise ValueError("headline_outcome must be a non-empty string")
    for field in (
        "cross_domain_moat_holds",
        "in_generation_moat_holds",
        "efficiency_cascade_dominates",
        "paper_ready",
        "verifier_is_oracle_honored",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("per_axis_gaps"), list):
        raise ValueError("per_axis_gaps must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if not isinstance(artifact.get("upstream_provenance"), list):
        raise ValueError("upstream_provenance must be a list")
    principles = artifact.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
    for row in artifact["upstream_provenance"]:
        if not is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected = checksum_from_provenance(artifact["upstream_provenance"])
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
        publication_gate_runner=publication_gate_runner,
    )
    validate_artifact(artifact)
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
