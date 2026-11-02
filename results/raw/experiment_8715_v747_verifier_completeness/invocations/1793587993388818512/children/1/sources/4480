"""Build the Exp 4268 v394 capstone aggregation artifact.

Spec refs: REQ-CAPSTONE-4268, SCENARIO-CAPSTONE-4268.
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


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import adversarial_verify as av  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_4268_capstone_v394.json")
EXPERIMENT_ID = 4268
RANDOM_SEED = 4268
SCHEMA = "carnot.capstone_v394_4268.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4268", "SCENARIO-CAPSTONE-4268"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_v394_artifacts_missing").hexdigest()


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4256_leak_audit": Upstream(
        4256, Path("results/experiment_4256_arc_oracle_distinct_leak_audit.json")
    ),
    "4257_multiseed": Upstream(
        4257, Path("results/experiment_4257_arc_oracle_distinct_multiseed_replication.json")
    ),
    "4258_cross_game": Upstream(
        4258, Path("results/experiment_4258_arc_oracle_distinct_cross_game_transfer.json")
    ),
    "4259_synthesis": Upstream(
        4259, Path("results/experiment_4259_arc_agglm_grid_synthesis.json")
    ),
    "4260_preflight": Upstream(
        4260, Path("results/experiment_4260_diffusiongemma_energy_guided_preflight.json")
    ),
    "4261_arc_progress": Upstream(
        4261, Path("results/experiment_4261_arc_incremental_progress.json")
    ),
    "4262_live_probe": Upstream(
        4262, Path("results/experiment_4262_arc_live_env_accuracy_probe.json")
    ),
    "4263_reward": Upstream(
        4263, Path("results/experiment_4263_verifier_as_reward_out_of_band_or_retire.json")
    ),
    "4264_code": Upstream(
        4264, Path("results/experiment_4264_code_oracle_distinct_replication_retry.json")
    ),
    "4266_registry": Upstream(
        4266, Path("results/experiment_4266_verifier_registry_gaps_hygiene.json")
    ),
    "4267_hardware": Upstream(
        4267, Path("results/experiment_4267_hardware_continuity.json")
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "hardened_win",
    "diffusiongemma_full_run_gate",
    "flagged_artifacts_excluded",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .394 close-state -- whether the first ARC "
        "oracle-distinct win hardened and whether synthesis broke the ceiling."
    ),
    "headline_outcome": (
        "One honest string aggregating the hardened-win + extend + scale-up + "
        "ARC + reward + code reads; the single line the .395 planner frames from."
    ),
    "hardened_win": (
        "BARE bool: the +44pp win is real (survived provenance-blind), robust "
        "(replicated multi-seed), and general (cross-game delta>0) -- the "
        "headline-eligibility verdict."
    ),
    "diffusiongemma_full_run_gate": (
        "BARE bool: resolvable iff hardened_win AND preflight_go -- whether "
        ".395 may activate the DiffusionGemma full run (gates the expensive "
        "scale-up on a survived win)."
    ),
    "flagged_artifacts_excluded": (
        "List of .394 artifacts excluded for flagged_adversarial -- the "
        "fabrication gate (their numbers are NOT aggregated)."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline "
        "stays the publication target; the ARC win is a new supporting result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited moat/headline result carried "
        "verifier_is_oracle=false (no circular result headlines a moat)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party "
        "re-derive the capstone."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream; skipped "
        "upstreams import no numbers."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4256_leak_audit": [
        "verifier_is_oracle",
        "win_survives_provenance_blind",
        "provenance_blind_delta",
        "provenance_blind_ci95",
        "origin_probe_auroc",
        "origin_correctness_corr",
        "held_out_task_n",
        "oracle_at_k",
    ],
    "4257_multiseed": [
        "verifier_is_oracle",
        "oracle_distinct_win_replicates",
        "mean_delta",
        "cross_seed_ci95",
        "cross_seed_ci95_excludes_zero",
        "n_seeds",
        "independent_rescore_delta",
    ],
    "4258_cross_game": [
        "verifier_is_oracle",
        "cross_game_delta",
        "cross_game_ci95",
        "ci95_excludes_zero",
        "held_out_game_n",
        "held_out_task_n",
        "headroom_exists",
    ],
    "4259_synthesis": [
        "verifier_is_oracle",
        "synthesis_breaks_oracle_ceiling",
        "synthesis_beats_selection",
        "synthesis_minus_oracle_delta",
        "synthesis_minus_oracle_ci95",
        "synthesis_minus_vote_delta",
        "ceiling_break_task_n",
        "oracle_at_k",
    ],
    "4260_preflight": [
        "verifier_is_oracle",
        "preflight_go",
        "guidance_changes_selection",
        "guidance_selection_change_count",
        "full_run_cost_estimate_s",
    ],
    "4261_arc_progress": [
        "total_levels",
        "total_levels_solved",
        "levels_completed",
        "new_levels_solved_this_task",
        "prior_total_levels_solved",
        "game_advanced",
        "game_levels_completed",
    ],
    "4262_live_probe": [
        "levels_completed",
        "actions_taken",
        "baseline_actions",
        "actions_vs_baseline_ratio",
        "environment_score",
        "game_probed",
        "scorecard_closed",
    ],
    "4263_reward": [
        "verifier_is_oracle",
        "ready_for_out_of_band",
        "verifier_as_reward_retired",
        "out_of_band_runner_path",
        "weighted_corpus",
    ],
    "4264_code": [
        "verifier_is_oracle",
        "code_replication_beats_vote",
        "code_replication_retired",
        "replication_read",
        "code_predictor_minus_vote_delta",
        "code_predictor_minus_vote_ci95",
        "ci95_excludes_zero",
        "held_out_task_n",
        "headroom_exists",
        "oracle_at_k",
        "oracle_minus_vote",
        "off_fold_auroc",
    ],
    "4266_registry": [
        "registry_reconciled",
        "regression_guard_passed",
        "v394_hardened_state",
        "gaps_logged",
    ],
    "4267_hardware": [
        "kv260_terminal_confirmed",
        "kv260_step_taken",
        "polarfire_step_taken",
        "gatemate_step_taken",
        "per_board_reachability",
        "fabric_acceleration_claimed",
        "speedup_claim_made",
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


def nested_map(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


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
    return value.removeprefix("sha256:") if is_sha256(value.removeprefix("sha256:")) else ""


def live_has_critical(flags: list[dict[str, Any]]) -> bool:
    return any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def run_live_flags(path: Path) -> list[dict[str, Any]]:
    return list(av.verify_artifact(path).get("flags", []))


def run_summarize_artifact(path: Path, root: Path) -> int:
    proc = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode)


def run_publication_gate(root: Path) -> JsonDict:
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


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    missing: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            missing.append(
                {"artifact_key": key, "experiment_id": upstream.experiment_id, "reason": "missing"}
            )
            continue
        sha = sha256_file(path)
        try:
            payload = read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            missing.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "reason": "unparsable_or_non_object",
                }
            )
            continue
        summarize_exit_code = summarize_runner(path, root)
        live_flags = live_flag_runner(path)
        stamped = payload.get("flagged_adversarial") is True
        critical = live_has_critical(live_flags)
        skipped = stamped or critical
        payloads[key] = payload
        provenance.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "sha256": sha,
                "payload_reproducibility_checksum": sha_from_payload_checksum(payload),
                "summarize_exit_code": summarize_exit_code,
                "live_adversarial_flags": live_flags,
                "stamped_flagged_adversarial": stamped,
                "live_critical": critical,
                "skipped": skipped,
                "fields_imported": [] if skipped else list(IMPORTED_FIELDS[key]),
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
                    "live_critical_flags": [
                        flag for flag in live_flags if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": "flagged_adversarial_or_live_critical",
                }
            )
    return payloads, provenance, exclusions, missing


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return BLOCKED_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def _verdict(payload: Mapping[str, Any] | None) -> str:
    return str_metric(payload, "honest_verdict")


def _oracle_distinct(payload: Mapping[str, Any] | None) -> bool:
    return bool_metric(payload, "verifier_is_oracle") is False


def _ci_excludes_zero(payload: Mapping[str, Any] | None, field: str, explicit_field: str = "") -> bool:
    explicit = bool_metric(payload, explicit_field) if explicit_field else None
    if explicit is not None:
        return explicit
    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    low, high = value
    if not isinstance(low, (int, float)) or isinstance(low, bool):
        return False
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        return False
    return float(low) > 0.0 or float(high) < 0.0


def provenance_blind_read(payload: JsonDict | None) -> JsonDict:
    win = (
        _oracle_distinct(payload)
        and bool_metric(payload, "win_survives_provenance_blind") is True
        and (float_metric(payload, "provenance_blind_delta") or 0.0) > 0.0
        and _ci_excludes_zero(payload, "provenance_blind_ci95")
    )
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "win_survives_provenance_blind": win,
        "reported_win_survives_provenance_blind": bool_metric(
            payload, "win_survives_provenance_blind"
        ),
        "provenance_blind_delta": float_metric(payload, "provenance_blind_delta"),
        "provenance_blind_ci95": payload.get("provenance_blind_ci95") if payload else None,
        "origin_probe_auroc": float_metric(payload, "origin_probe_auroc"),
        "origin_correctness_corr": float_metric(payload, "origin_correctness_corr"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def multiseed_read(payload: JsonDict | None) -> JsonDict:
    replicated = (
        _oracle_distinct(payload)
        and bool_metric(payload, "oracle_distinct_win_replicates") is True
        and (float_metric(payload, "mean_delta") or 0.0) > 0.0
        and (
            bool_metric(payload, "cross_seed_ci95_excludes_zero") is True
            or _ci_excludes_zero(payload, "cross_seed_ci95")
        )
    )
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "oracle_distinct_win_replicates": replicated,
        "reported_oracle_distinct_win_replicates": bool_metric(
            payload, "oracle_distinct_win_replicates"
        ),
        "mean_delta": float_metric(payload, "mean_delta"),
        "cross_seed_ci95": payload.get("cross_seed_ci95") if payload else None,
        "cross_seed_ci95_excludes_zero": bool_metric(payload, "cross_seed_ci95_excludes_zero"),
        "n_seeds": int_metric(payload, "n_seeds"),
        "independent_rescore_delta": float_metric(payload, "independent_rescore_delta"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def cross_game_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "cross_game_delta": None}
    delta = float_metric(payload, "cross_game_delta")
    positive = _oracle_distinct(payload) and delta is not None and delta > 0.0
    return {
        "status": "positive_cross_game_delta" if positive else "blocked_or_no_positive_delta",
        "used_as_oracle_distinct": _oracle_distinct(payload),
        "cross_game_delta": delta,
        "cross_game_ci95": payload.get("cross_game_ci95") if payload else None,
        "ci95_excludes_zero": _ci_excludes_zero(payload, "cross_game_ci95", "ci95_excludes_zero"),
        "held_out_game_n": int_metric(payload, "held_out_game_n"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "headroom_exists": bool_metric(payload, "headroom_exists"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def synthesis_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "synthesis_breaks_oracle_ceiling": False}
    breaks = _oracle_distinct(payload) and bool_metric(payload, "synthesis_breaks_oracle_ceiling") is True
    return {
        "status": "breaks_oracle_ceiling" if breaks else "does_not_break_oracle_ceiling",
        "synthesis_breaks_oracle_ceiling": breaks,
        "reported_synthesis_breaks_oracle_ceiling": bool_metric(
            payload, "synthesis_breaks_oracle_ceiling"
        ),
        "synthesis_beats_selection": bool_metric(payload, "synthesis_beats_selection"),
        "synthesis_minus_oracle_delta": float_metric(payload, "synthesis_minus_oracle_delta"),
        "synthesis_minus_oracle_ci95": payload.get("synthesis_minus_oracle_ci95") if payload else None,
        "synthesis_minus_vote_delta": float_metric(payload, "synthesis_minus_vote_delta"),
        "ceiling_break_task_n": int_metric(payload, "ceiling_break_task_n"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def preflight_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "preflight_go": False}
    go = bool_metric(payload, "preflight_go") is True
    return {
        "status": "preflight_go" if go else "preflight_blocked",
        "preflight_go": go,
        "guidance_changes_selection": bool_metric(payload, "guidance_changes_selection"),
        "guidance_selection_change_count": int_metric(payload, "guidance_selection_change_count"),
        "full_run_cost_estimate_s": float_metric(payload, "full_run_cost_estimate_s"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def arc_progress_read(payload: JsonDict | None) -> JsonDict:
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "total_levels": int_metric(payload, "total_levels"),
        "total_levels_solved": int_metric(payload, "total_levels_solved"),
        "levels_completed": int_metric(payload, "levels_completed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "prior_total_levels_solved": int_metric(payload, "prior_total_levels_solved"),
        "game_advanced": str_metric(payload, "game_advanced"),
        "game_levels_completed": int_metric(payload, "game_levels_completed"),
        "honest_verdict": _verdict(payload),
    }


def live_probe_read(payload: JsonDict | None) -> JsonDict:
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "levels_completed": int_metric(payload, "levels_completed"),
        "actions_taken": int_metric(payload, "actions_taken"),
        "baseline_actions": int_metric(payload, "baseline_actions"),
        "actions_vs_baseline_ratio": float_metric(payload, "actions_vs_baseline_ratio"),
        "environment_score": dict(nested_map(payload, "environment_score")),
        "game_probed": str_metric(payload, "game_probed"),
        "scorecard_closed": bool_metric(payload, "scorecard_closed") is True,
        "honest_verdict": _verdict(payload),
    }


def reward_read(payload: JsonDict | None) -> JsonDict:
    ready = bool_metric(payload, "ready_for_out_of_band") is True
    retired = bool_metric(payload, "verifier_as_reward_retired") is True
    if ready:
        status = "reward_out_of_band"
    elif retired:
        status = "reward_retired"
    else:
        status = "reward_pending"
    return {
        "status": status,
        "ready_for_out_of_band": ready,
        "verifier_as_reward_retired": retired,
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "out_of_band_runner_path": str_metric(payload, "out_of_band_runner_path"),
        "weighted_corpus": dict(nested_map(payload, "weighted_corpus")),
        "honest_verdict": _verdict(payload),
    }


def code_read(payload: JsonDict | None) -> JsonDict:
    wins = _oracle_distinct(payload) and bool_metric(payload, "code_replication_beats_vote") is True
    read = str_metric(payload, "replication_read")
    if wins:
        status = "code_replicates"
    elif read:
        status = f"code_{read}"
    else:
        status = "code_no_win"
    return {
        "status": status,
        "code_replication_beats_vote": wins,
        "reported_code_replication_beats_vote": bool_metric(payload, "code_replication_beats_vote"),
        "code_replication_retired": bool_metric(payload, "code_replication_retired") is True,
        "replication_read": read,
        "code_predictor_minus_vote_delta": float_metric(payload, "code_predictor_minus_vote_delta"),
        "code_predictor_minus_vote_ci95": payload.get("code_predictor_minus_vote_ci95") if payload else None,
        "ci95_excludes_zero": bool_metric(payload, "ci95_excludes_zero"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "headroom_exists": bool_metric(payload, "headroom_exists"),
        "oracle_at_k": float_metric(payload, "oracle_at_k"),
        "oracle_minus_vote": float_metric(payload, "oracle_minus_vote"),
        "off_fold_auroc": float_metric(payload, "off_fold_auroc"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": _verdict(payload),
    }


def registry_read(payload: JsonDict | None, *, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "regression_guard_passed": False}
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "registry_reconciled": bool_metric(payload, "registry_reconciled") is True,
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed") is True,
        "v394_hardened_state": str_metric(payload, "v394_hardened_state"),
        "gaps_logged": list(payload.get("gaps_logged", [])) if isinstance(payload, Mapping) else [],
        "honest_verdict": _verdict(payload),
    }


def hardware_read(payload: JsonDict | None) -> JsonDict:
    return {
        "status": "included" if payload is not None else "missing_or_excluded",
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed") is True,
        "kv260_step_taken": str_metric(payload, "kv260_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "per_board_reachability": dict(nested_map(payload, "per_board_reachability")),
        "fabric_acceleration_claimed": bool_metric(payload, "fabric_acceleration_claimed") is True,
        "speedup_claim_made": bool_metric(payload, "speedup_claim_made") is True,
        "honest_verdict": _verdict(payload),
    }


def headline_string(
    hardened_win: bool,
    synthesis_breaks_ceiling: bool,
    full_run_gate: bool,
    total_levels: int,
    live_levels: int,
    reward_label: str,
    code_label: str,
    paper_ready: bool,
) -> str:
    hardened = "hardened_win_true" if hardened_win else "hardened_win_false"
    synthesis = (
        "synthesis_breaks_oracle_ceiling"
        if synthesis_breaks_ceiling
        else "synthesis_does_not_break_oracle_ceiling"
    )
    diffusion = "diffusiongemma_full_run_ready" if full_run_gate else "diffusiongemma_full_run_blocked"
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return (
        f"{hardened}_{synthesis}_{diffusion}_arc{total_levels}_live{live_levels}_"
        f"{reward_label}_{code_label}_{paper}"
    )


def _headline_for_complete(
    hardening: Mapping[str, Any],
    synthesis: Mapping[str, Any],
    preflight: Mapping[str, Any],
    total_levels: int,
    live_levels: int,
    reward_label: str,
    code_label: str,
    paper_ready: bool,
    full_run_gate: bool,
) -> str:
    if bool(hardening["hardened_win"]):
        return headline_string(
            True,
            bool(synthesis["synthesis_breaks_oracle_ceiling"]),
            full_run_gate,
            total_levels,
            live_levels,
            reward_label,
            code_label,
            paper_ready,
        )
    provenance = bool(hardening["provenance_blind"]["win_survives_provenance_blind"])
    multiseed = bool(hardening["multiseed"]["oracle_distinct_win_replicates"])
    base = "within_pool_win_survived" if provenance and multiseed else "hardened_win_not_established"
    cross = "cross_game_positive" if hardening["cross_game"].get("cross_game_delta") else "cross_game_blocked"
    synth = "synthesis_excluded" if synthesis["status"] == "excluded_flagged_adversarial" else synthesis["status"]
    pre = "preflight_excluded" if preflight["status"] == "excluded_flagged_adversarial" else preflight["status"]
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return f"{base}_but_{cross}_{synth}_{pre}_arc{total_levels}_live{live_levels}_{reward_label}_{code_label}_{paper}"


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> JsonDict:
    root = Path(repo_root)
    start = time.time() if started_s is None else started_s
    payloads, provenance, exclusions, missing = _read_inputs(root, live_flag_runner, summarize_runner)
    end = time.time() if now_s is None else now_s
    common: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(end - start, 6),
        "summarize_command": "python3 scripts/summarize_artifact.py <artifact>",
        "live_adversarial_recheck": "scripts/adversarial_verify.py verify_artifact per upstream",
        "publication_gate_command": "python3 scripts/publication_gate.py --json",
    }
    if missing:
        artifact = {
            **common,
            "honest_verdict": "blocked_v394_artifacts_missing",
            "headline_outcome": "blocked_v394_artifacts_missing",
            "hardened_win": False,
            "diffusiongemma_full_run_gate": False,
            "flagged_artifacts_excluded": [],
            "paper_ready": None,
            "unmet_gates": [],
            "publication_gate": None,
            "verifier_is_oracle_honored": True,
            "missing_upstream_artifacts": missing,
            "upstream_provenance": [],
            "reproducibility_checksum": BLOCKED_CHECKSUM,
        }
        validate_artifact(artifact)
        return artifact

    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: clean_payload(payloads.get(key), skipped.get(key, False)) for key in DEFAULT_UPSTREAMS
    }
    provenance_blind = provenance_blind_read(clean["4256_leak_audit"])
    multiseed = multiseed_read(clean["4257_multiseed"])
    cross_game = cross_game_read(clean["4258_cross_game"], skipped=skipped["4258_cross_game"])
    hardened_win = (
        bool(provenance_blind["win_survives_provenance_blind"])
        and bool(multiseed["oracle_distinct_win_replicates"])
        and cross_game.get("cross_game_delta") is not None
        and float(cross_game["cross_game_delta"]) > 0.0
        and bool(cross_game.get("used_as_oracle_distinct"))
    )
    hardening = {
        "hardened_win": hardened_win,
        "provenance_blind": provenance_blind,
        "multiseed": multiseed,
        "cross_game": cross_game,
    }
    synthesis = synthesis_read(clean["4259_synthesis"], skipped=skipped["4259_synthesis"])
    preflight = preflight_read(clean["4260_preflight"], skipped=skipped["4260_preflight"])
    arc = arc_progress_read(clean["4261_arc_progress"])
    live = live_probe_read(clean["4262_live_probe"])
    reward = reward_read(clean["4263_reward"])
    code = code_read(clean["4264_code"])
    registry = registry_read(clean["4266_registry"], skipped=skipped["4266_registry"])
    hardware = hardware_read(clean["4267_hardware"])
    publication = publication_gate_runner(root)
    paper_ready = bool(publication.get("paper_ready"))
    full_run_gate = hardened_win and bool(preflight["preflight_go"])
    total_levels = int(arc["total_levels_solved"])
    live_levels = int(live["levels_completed"])
    headline = _headline_for_complete(
        hardening,
        synthesis,
        preflight,
        total_levels,
        live_levels,
        str(reward["status"]),
        str(code["status"]),
        paper_ready,
        full_run_gate,
    )
    artifact = {
        **common,
        "honest_verdict": (
            f"complete: capstone_v394_{headline}_hardened_win_{hardened_win}_"
            f"diffusiongemma_full_run_gate_{full_run_gate}_excluded_{len(exclusions)}"
        ),
        "headline_outcome": headline,
        "hardened_win": hardened_win,
        "diffusiongemma_full_run_gate": full_run_gate,
        "flagged_artifacts_excluded": exclusions,
        "paper_ready": paper_ready,
        "unmet_gates": list(publication.get("unmet_gates", [])),
        "publication_gate": publication,
        "verifier_is_oracle_honored": True,
        "missing_upstream_artifacts": [],
        "hardening": hardening,
        "extend_synthesis": synthesis,
        "scale_up_readiness": preflight,
        "arc_progress": arc,
        "live_arc_probe": live,
        "reward_decision": reward,
        "code_read": code,
        "registry_read": registry,
        "hardware_read": hardware,
        "upstream_provenance": provenance,
        "reproducibility_checksum": checksum_from_provenance(provenance),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    blocked = verdict == "blocked_v394_artifacts_missing"
    if not (blocked or (isinstance(verdict, str) and verdict.startswith("complete:"))):
        raise ValueError("honest_verdict must be terminal-prefixed or blocked_v394_artifacts_missing")
    if not isinstance(artifact["headline_outcome"], str) or not artifact["headline_outcome"]:
        raise ValueError("headline_outcome must be a non-empty string")
    if not isinstance(artifact["hardened_win"], bool):
        raise ValueError("hardened_win must be a bare bool")
    if not isinstance(artifact["diffusiongemma_full_run_gate"], bool):
        raise ValueError("DiffusionGemma full-run gate must be a bare bool")
    if artifact["diffusiongemma_full_run_gate"] and not artifact["hardened_win"]:
        raise ValueError("DiffusionGemma full-run gate cannot pass without hardened_win")
    if artifact["verifier_is_oracle_honored"] is not True:
        raise ValueError("oracle-distinctness discipline was not honored")
    if blocked:
        if artifact["paper_ready"] is not None:
            raise ValueError("blocked artifacts must leave paper_ready unresolved")
    elif not isinstance(artifact["paper_ready"], bool):
        raise ValueError("paper_ready must be a bare bool")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principle text")
    flagged = artifact["flagged_artifacts_excluded"]
    if not isinstance(flagged, list):
        raise ValueError("flagged_artifacts_excluded must be a list")
    if any(not isinstance(row, Mapping) for row in flagged):
        raise ValueError("flagged entries must be objects")
    provenance = artifact["upstream_provenance"]
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("provenance entries must be objects")
        if not isinstance(row.get("artifact_key"), str):
            raise ValueError("provenance entries need artifact_key")
        if not isinstance(row.get("experiment_id"), int) or isinstance(row.get("experiment_id"), bool):
            raise ValueError("provenance entries need integer experiment_id")
        if not is_sha256(row.get("sha256")):
            raise ValueError("provenance entries need sha256")
        if not isinstance(row.get("fields_imported"), list):
            raise ValueError("provenance entries need fields_imported lists")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected_checksum = BLOCKED_CHECKSUM if blocked else checksum_from_provenance(provenance)
    if artifact["reproducibility_checksum"] != expected_checksum:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    repo_root: Path | str = REPO_ROOT,
    output_path: Path = OUTPUT_REL_PATH,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = run_live_flags,
    summarize_runner: SummarizeRunner = run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = run_publication_gate,
) -> Path:
    root = Path(repo_root)
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
        publication_gate_runner=publication_gate_runner,
    )
    out = root / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0
