"""Build the Exp 4499 .415 ARC imitation/replay capstone.

Spec refs: REQ-CAPSTONE-4499, SCENARIO-CAPSTONE-4499.

This is an aggregation artifact. It reads the .415 upstream JSON files through
the disciplined summary/live-adversarial path, skips flagged inputs before
importing headline metrics, and reports the requested A1-A5/variant-transfer
signals without promoting reproducible_total_levels as the headline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import capstone_v400_4335 as base


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4499_capstone_v415.json")
EXPERIMENT_ID = 4499
RANDOM_SEED = 4499
SCHEMA = "carnot.capstone_v415_4499.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4499", "SCENARIO-CAPSTONE-4499"]
BASELINE_LOO_AUROC = 0.503
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4490_a1_human_replay": Upstream(
        4490,
        Path("results/experiment_4490_human_replay_frame_change_predictor.json"),
    ),
    "4491_a2_trust_energy": Upstream(
        4491,
        Path("results/experiment_4491_world_model_trust_energy.json"),
    ),
    "4492_a3_energy_loo": Upstream(
        4492,
        Path("results/experiment_4492_energy_augmentation_loo_gate.json"),
    ),
    "4493_a4_hud_register": Upstream(
        4493,
        Path("results/experiment_4493_hud_register_deepen.json"),
    ),
    "4494_a5_adapter_l2": Upstream(
        4494,
        Path("results/experiment_4494_adapter_deepen_l2.json"),
    ),
    "4495_replay_corpus": Upstream(
        4495,
        Path("results/experiment_4495_human_replay_corpus_staging.json"),
    ),
    "4496_scoreboard": Upstream(
        4496,
        Path("results/experiment_4496_submitted_agent_scoreboard.json"),
    ),
    "4497_hardware": Upstream(
        4497,
        Path("results/experiment_4497_hardware_continuity_audit.json"),
    ),
    "4498_sota": Upstream(
        4498,
        Path("results/experiment_4498_arc_imitation_sota_415.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "a1_actions_to_first_levelup_reduction",
    "a2_trust_energy_oracle_distinct_verdict",
    "a3_energy_augmentation_loo_auroc",
    "a4_a5_l2_banked",
    "variant_transfer_rate",
    "verifier_is_oracle",
    "verifier_claims",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right "
        "duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource "
        "fabrication."
    ),
    "a1_actions_to_first_levelup_reduction": (
        "the real A1 signal is held-out actions-to-first-level-up reduction, not "
        "reproducible_total_levels."
    ),
    "a2_trust_energy_oracle_distinct_verdict": (
        "reports trust-energy oracle-distinct status and declares verifier_is_oracle "
        "for the claim."
    ),
    "a3_energy_augmentation_loo_auroc": (
        "reports cross-game LOO-AUROC versus the 0.503 baseline."
    ),
    "a4_a5_l2_banked": (
        "reports whether any A4/A5 L2 banked only from clean offline reproduction gates."
    ),
    "variant_transfer_rate": (
        "bare float from the submitted-agent scoreboard headline, separate from "
        "reproducible_total_levels context."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; verifier-derived upstream claims "
        "declare their own circularity."
    ),
    "verifier_claims": (
        "each verifier claim declares verifier_is_oracle so circularity is explicit."
    ),
    "flagged_artifacts_skipped": "stamped or live-critical artifacts import no fields.",
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256}; skipped flagged artifacts "
        "import no fields."
    ),
    "reproducibility_checksum": "content hash for reproducibility",
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4490_a1_human_replay": [
        "honest_verdict",
        "heldout_median_actions_before",
        "heldout_median_actions_after",
        "implied_efficiency_delta",
        "solve_rate_before",
        "solve_rate_after",
        "solve_rate_dropped",
        "trained_on_human_corpus",
        "positive_control",
        "preconditions_checked",
    ],
    "4491_a2_trust_energy": [
        "honest_verdict",
        "baseline_pick_rate",
        "trust_energy_pick_rate",
        "positive_control_passed",
        "hidden_state_games_n",
        "selected_candidates",
        "verifier_is_oracle",
        "preconditions_checked",
    ],
    "4492_a3_energy_loo": [
        "honest_verdict",
        "baseline_loo_auroc",
        "v2_baseline_loo_auroc",
        "v3_loo_auroc",
        "target_loo_auroc",
        "loo_gate_passed",
        "feature_class_deltas",
        "feature_class_loo_auroc",
        "preconditions_checked",
    ],
    "4493_a4_hud_register": [
        "honest_verdict",
        "offline_reproduced",
        "reproduced_levels",
        "candidate_reproduction_attempts",
        "residual_blockers",
        "preconditions_checked",
    ],
    "4494_a5_adapter_l2": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "residual_blockers",
        "preconditions_checked",
    ],
    "4495_replay_corpus": [
        "honest_verdict",
        "training_example_count",
        "training_shard_count",
        "weights_committed",
        "official_license_verified",
        "preconditions_checked",
    ],
    "4496_scoreboard": [
        "honest_verdict",
        "headline_metrics",
        "leaderboard_submission",
        "preconditions_checked",
    ],
    "4497_hardware": [
        "honest_verdict",
        "per_board_reachability",
        "per_board_status",
        "preconditions_checked",
    ],
    "4498_sota": [
        "honest_verdict",
        "strongest_for_v416",
        "source_ids",
        "preconditions_checked",
    ],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _flagged_payload(payload: JsonDict) -> JsonDict:
    flagged = dict(payload)
    flagged["flagged_adversarial"] = True
    return flagged


def _number_or_none(payload: Mapping[str, Any] | None, field: str) -> float | None:
    return base.float_metric(payload, field)


def _metric_object(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    metrics = payload.get("headline_metrics") if isinstance(payload, Mapping) else None
    return metrics if isinstance(metrics, Mapping) else {}


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict]]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []

    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            raw_artifacts[key] = None
            continue

        sha = base.sha256_file(path)
        summarize_exit_code, summarize_error = base._safe_summarize(  # noqa: SLF001
            path,
            root,
            summarize_runner,
        )
        live_flags = base._safe_live_flags(path, live_flag_runner)  # noqa: SLF001
        critical = base.live_has_critical(live_flags)
        parse_error = ""
        payload: JsonDict | None = None
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = _flagged_payload(payload) if payload is not None and skipped else payload
        row = {
            "artifact_key": key,
            "experiment_id": upstream.experiment_id,
            "path": str(upstream.path),
            "sha256": sha,
            "payload_reproducibility_checksum": base.sha_from_payload_checksum(payload or {}),
            "summarize_exit_code": summarize_exit_code,
            "summarize_error": summarize_error,
            "live_adversarial_flags": live_flags,
            "stamped_flagged_adversarial": stamped,
            "live_critical": critical,
            "parse_error": parse_error,
            "skipped": skipped,
            "fields_imported": _fields_for_payload(key, skipped),
        }
        provenance.append(row)
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
                    "reason": base._exclusion_reason(stamped, critical, parse_error),  # noqa: SLF001
                }
            )
    return raw_artifacts, provenance, exclusions


def a1_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "actions_reduction": None,
            "efficiency_win": False,
        }
    if payload is None:
        return {"state": "missing_or_excluded", "actions_reduction": None, "efficiency_win": False}

    before = _number_or_none(payload, "heldout_median_actions_before")
    after = _number_or_none(payload, "heldout_median_actions_after")
    reduction = before - after if before is not None and after is not None else None
    solve_rate_dropped = base.bool_metric(payload, "solve_rate_dropped") is True
    efficiency_win = reduction is not None and reduction > 0.0 and not solve_rate_dropped
    positive_control = payload.get("positive_control")
    return {
        "state": "heldout_actions_reduced" if efficiency_win else "heldout_not_measured",
        "heldout_median_actions_before": before,
        "heldout_median_actions_after": after,
        "actions_reduction": reduction,
        "implied_efficiency_delta": _number_or_none(payload, "implied_efficiency_delta"),
        "solve_rate_before": _number_or_none(payload, "solve_rate_before"),
        "solve_rate_after": _number_or_none(payload, "solve_rate_after"),
        "solve_rate_dropped": solve_rate_dropped,
        "efficiency_win": efficiency_win,
        "trained_on_human_corpus": base.bool_metric(payload, "trained_on_human_corpus") is True,
        "positive_control": dict(positive_control) if isinstance(positive_control, Mapping) else {},
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def a2_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "oracle_distinct": False,
            "trust_energy_pick_rate": 0.0,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "oracle_distinct": False,
            "trust_energy_pick_rate": 0.0,
        }

    trust_rate = base.float_metric(payload, "trust_energy_pick_rate") or 0.0
    baseline_rate = base.float_metric(payload, "baseline_pick_rate") or 0.0
    verifier_is_oracle = base.bool_metric(payload, "verifier_is_oracle") is True
    positive_control_passed = base.bool_metric(payload, "positive_control_passed") is True
    oracle_distinct = trust_rate > baseline_rate and positive_control_passed and not verifier_is_oracle
    return {
        "state": "trust_energy_oracle_distinct_pass" if oracle_distinct else "trust_energy_open",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "baseline_pick_rate": baseline_rate,
        "trust_energy_pick_rate": trust_rate,
        "positive_control_passed": positive_control_passed,
        "hidden_state_games_n": base.int_metric(payload, "hidden_state_games_n"),
        "selected_candidate_count": len(base.list_metric(payload, "selected_candidates")),
        "oracle_distinct": oracle_distinct,
        "verifier_is_oracle": verifier_is_oracle,
    }


def a3_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "baseline_loo_auroc": BASELINE_LOO_AUROC,
            "v3_loo_auroc": 0.0,
            "loo_auroc_delta": 0.0,
            "beats_0503_baseline": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "baseline_loo_auroc": BASELINE_LOO_AUROC,
            "v3_loo_auroc": 0.0,
            "loo_auroc_delta": 0.0,
            "beats_0503_baseline": False,
        }

    baseline = base.float_metric(payload, "baseline_loo_auroc") or BASELINE_LOO_AUROC
    v3 = base.float_metric(payload, "v3_loo_auroc") or 0.0
    delta = v3 - baseline
    return {
        "state": "energy_aug_beats_0503" if delta > 0.0 else "energy_aug_null",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "baseline_loo_auroc": baseline,
        "v2_baseline_loo_auroc": base.float_metric(payload, "v2_baseline_loo_auroc"),
        "v3_loo_auroc": v3,
        "target_loo_auroc": base.float_metric(payload, "target_loo_auroc") or 0.0,
        "loo_auroc_delta": delta,
        "beats_0503_baseline": delta > 0.0,
        "loo_gate_passed": base.bool_metric(payload, "loo_gate_passed") is True,
        "feature_class_deltas": dict(payload.get("feature_class_deltas", {}))
        if isinstance(payload.get("feature_class_deltas"), Mapping)
        else {},
        "feature_class_loo_auroc": dict(payload.get("feature_class_loo_auroc", {}))
        if isinstance(payload.get("feature_class_loo_auroc"), Mapping)
        else {},
    }


def _attempt_l2_banked(attempt: Mapping[str, Any]) -> bool:
    claimed = base.int_metric(attempt, "claimed_level")
    reached = base.int_metric(attempt, "reached_level")
    return base.bool_metric(attempt, "reproduced") is True and claimed >= 2 and reached >= 2


def _hud_l2_attempts(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    attempts = base.list_metric(payload, "candidate_reproduction_attempts")
    rows: list[JsonDict] = []
    for raw in attempts:
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "game": base.str_metric(raw, "game"),
                "claimed_level": base.int_metric(raw, "claimed_level"),
                "reached_level": base.int_metric(raw, "reached_level"),
                "reproduced": base.bool_metric(raw, "reproduced") is True,
                "l2_banked": _attempt_l2_banked(raw),
            }
        )
    return rows


def _adapter_l2_attempt(payload: Mapping[str, Any] | None) -> JsonDict:
    gate = payload.get("reproduction_gate") if isinstance(payload, Mapping) else None
    gate_map = gate if isinstance(gate, Mapping) else {}
    row = {
        "game": base.str_metric(gate_map, "game") or base.str_metric(payload, "target_game"),
        "claimed_level": base.int_metric(gate_map, "claimed_level"),
        "reached_level": base.int_metric(gate_map, "reached_level"),
        "reproduced": base.bool_metric(gate_map, "reproduced") is True,
    }
    row["l2_banked"] = _attempt_l2_banked(row)
    return row


def a4_a5_l2_read(
    hud_payload: JsonDict | None,
    adapter_payload: JsonDict | None,
    hud_skipped: bool,
    adapter_skipped: bool,
) -> JsonDict:
    if hud_skipped and adapter_skipped:
        return {"state": "excluded_flagged_adversarial", "any_l2_banked": False}
    if hud_payload is None and adapter_payload is None:
        return {"state": "missing_or_excluded", "any_l2_banked": False}

    hud_attempts = [] if hud_skipped else _hud_l2_attempts(hud_payload)
    adapter_attempt = {} if adapter_skipped else _adapter_l2_attempt(adapter_payload)
    attempts = hud_attempts + ([adapter_attempt] if adapter_attempt else [])
    any_l2 = any(bool(row.get("l2_banked")) for row in attempts)
    return {
        "state": "l2_banked" if any_l2 else "l2_not_banked",
        "any_l2_banked": any_l2,
        "hud_register": {
            "skipped": hud_skipped,
            "honest_verdict": base.str_metric(hud_payload, "honest_verdict"),
            "offline_reproduced": base.bool_metric(hud_payload, "offline_reproduced") is True,
            "reproduced_levels": base.int_metric(hud_payload, "reproduced_levels"),
            "attempts": hud_attempts,
            "residual_blockers": base.list_metric(hud_payload, "residual_blockers"),
        },
        "adapter_l2": {
            "skipped": adapter_skipped,
            "honest_verdict": base.str_metric(adapter_payload, "honest_verdict"),
            "offline_reproduced": base.bool_metric(adapter_payload, "offline_reproduced") is True,
            "reproduced_levels": base.int_metric(adapter_payload, "reproduced_levels"),
            "attempt": adapter_attempt,
            "residual_blockers": base.list_metric(adapter_payload, "residual_blockers"),
        },
    }


def variant_transfer_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "variant_transfer_rate": 0.0,
            "variant_transfer_solved": 0,
            "variant_transfer_attempted": 0,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "variant_transfer_rate": 0.0,
            "variant_transfer_solved": 0,
            "variant_transfer_attempted": 0,
        }
    metrics = _metric_object(payload)
    return {
        "state": "variant_transfer_measured",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "variant_transfer_rate": base.float_metric(metrics, "variant_transfer_rate") or 0.0,
        "variant_transfer_solved": base.int_metric(metrics, "variant_transfer_solved"),
        "variant_transfer_attempted": base.int_metric(metrics, "variant_transfer_attempted"),
        "heldout_generic_solve_rate": base.float_metric(
            metrics,
            "submitted_default_heldout_generic_solve_rate",
        )
        or 0.0,
        "heldout_generic_solved": base.int_metric(
            metrics,
            "submitted_default_heldout_generic_solved",
        ),
        "heldout_generic_attempted": base.int_metric(
            metrics,
            "submitted_default_heldout_generic_attempted",
        ),
        "leaderboard_submission": base.bool_metric(payload, "leaderboard_submission") is True,
    }


def operational_context_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"state": "excluded_flagged_adversarial"}
    if payload is None:
        return {"state": "missing_or_excluded"}
    return {
        "state": "available",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "training_example_count": base.int_metric(payload, "training_example_count"),
        "training_shard_count": base.int_metric(payload, "training_shard_count"),
        "weights_committed": base.bool_metric(payload, "weights_committed"),
        "official_license_verified": base.bool_metric(payload, "official_license_verified"),
        "per_board_reachability": dict(payload.get("per_board_reachability", {}))
        if isinstance(payload.get("per_board_reachability"), Mapping)
        else {},
        "strongest_for_v416": base.str_metric(payload, "strongest_for_v416"),
        "source_ids": base.list_metric(payload, "source_ids"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="a1_actions_to_first_levelup_reduction",
            required_keys=("4490_a1_human_replay",),
            verdict_fn=lambda present: a1_read(
                present.get("4490_a1_human_replay"),
                False,
            )["efficiency_win"],
        ),
        aggregate.AxisSpec(
            name="a2_trust_energy_oracle_distinct_verdict",
            required_keys=("4491_a2_trust_energy",),
            verdict_fn=lambda present: a2_read(
                present.get("4491_a2_trust_energy"),
                False,
            )["oracle_distinct"],
        ),
        aggregate.AxisSpec(
            name="a3_energy_augmentation_loo_auroc",
            required_keys=("4492_a3_energy_loo",),
            verdict_fn=lambda present: a3_read(
                present.get("4492_a3_energy_loo"),
                False,
            )["beats_0503_baseline"],
        ),
        aggregate.AxisSpec(
            name="a4_a5_l2_banked",
            required_keys=("4493_a4_hud_register", "4494_a5_adapter_l2"),
            verdict_fn=lambda present: a4_a5_l2_read(
                present.get("4493_a4_hud_register"),
                present.get("4494_a5_adapter_l2"),
                False,
                False,
            )["any_l2_banked"],
        ),
        aggregate.AxisSpec(
            name="variant_transfer_rate",
            required_keys=("4496_scoreboard",),
            verdict_fn=lambda present: variant_transfer_read(
                present.get("4496_scoreboard"),
                False,
            )["variant_transfer_rate"],
        ),
        aggregate.AxisSpec(
            name="operational_context",
            required_keys=("4495_replay_corpus", "4497_hardware", "4498_sota"),
            verdict_fn=lambda present: sorted(present),
        ),
    ]


def _cited_upstream_artifacts(provenance: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "artifact_key": row["artifact_key"],
            "experiment_id": row["experiment_id"],
            "path": row["path"],
            "sha256": row["sha256"],
            "fields_imported": row["fields_imported"],
        }
        for row in provenance
    ]


def _claim(name: str, verifier_is_oracle: bool, source: str, skipped: bool = False) -> JsonDict:
    return {
        "claim": name,
        "source": source,
        "verifier_is_oracle": bool(verifier_is_oracle),
        "skipped": bool(skipped),
    }


def _verifier_claims(
    *,
    skipped: Mapping[str, bool],
    a2: Mapping[str, Any],
    a4_a5: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _claim(
            "a1_actions_to_first_levelup_reduction",
            False,
            "exp4490",
            skipped.get("4490_a1_human_replay", False),
        ),
        _claim(
            "a2_trust_energy_oracle_distinct_verdict",
            bool(a2.get("verifier_is_oracle")),
            "exp4491",
            skipped.get("4491_a2_trust_energy", False),
        ),
        _claim(
            "a3_energy_augmentation_loo_auroc",
            False,
            "exp4492",
            skipped.get("4492_a3_energy_loo", False),
        ),
        _claim(
            "a4_a5_l2_banked",
            False,
            "exp4493+exp4494",
            skipped.get("4493_a4_hud_register", False)
            or skipped.get("4494_a5_adapter_l2", False),
        ),
        _claim(
            "a4_a5_l2_gate_is_execution_grounded",
            False,
            str(a4_a5.get("state", "")),
            skipped.get("4493_a4_hud_register", False)
            and skipped.get("4494_a5_adapter_l2", False),
        ),
    ]


def _preconditions_checked(root: Path, provenance: list[JsonDict], clean: Mapping[str, Any]) -> JsonDict:
    provenance_by_key = {row["artifact_key"]: row for row in provenance}
    upstreams: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        row = provenance_by_key.get(key)
        payload = clean.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
                "upstream_preconditions": payload.get("preconditions_checked")
                if isinstance(payload, Mapping)
                else None,
            }
        )
    return {
        "upstream_artifacts": upstreams,
        "summarize_artifact_required": "scripts/summarize_artifact.py",
        "reading_results_discipline": True,
        "leaderboard_submission": False,
    }


def _capstone_recheck_status(flags: list[dict[str, Any]]) -> JsonDict:
    circular = any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    critical = base.live_has_critical(flags)
    return {
        "status": "critical_flags" if critical else "clean",
        "flags": flags,
        "circular_moat_overclaim": circular,
    }


def _honest_verdict(
    *,
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    l2_banked: bool,
    variant_rate: float,
) -> str:
    a1_state = "a1_efficiency_win" if a1.get("efficiency_win") is True else "a1_no_heldout_win"
    a2_state = "a2_oracle_distinct" if a2.get("oracle_distinct") is True else "a2_open"
    a3_state = "a3_beats_0503" if a3.get("beats_0503_baseline") is True else "a3_no_win"
    l2_state = "l2_banked" if l2_banked else "l2_not_banked"
    return (
        "complete: v415_"
        f"{a1_state}_{a2_state}_{a3_state}_{l2_state}_"
        f"variant_transfer_{variant_rate:.2f}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    *,
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4_a5: Mapping[str, Any],
    variant_rate: float,
) -> str:
    payload = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4_a5": a4_a5,
        "variant_transfer_rate": variant_rate,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    raw_artifacts, provenance, exclusions = _read_inputs(root, live_flag_runner, summarize_runner)
    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: base.clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }

    a1 = a1_read(clean["4490_a1_human_replay"], skipped.get("4490_a1_human_replay", False))
    a2 = a2_read(clean["4491_a2_trust_energy"], skipped.get("4491_a2_trust_energy", False))
    a3 = a3_read(clean["4492_a3_energy_loo"], skipped.get("4492_a3_energy_loo", False))
    a4_a5 = a4_a5_l2_read(
        clean["4493_a4_hud_register"],
        clean["4494_a5_adapter_l2"],
        skipped.get("4493_a4_hud_register", False),
        skipped.get("4494_a5_adapter_l2", False),
    )
    variant = variant_transfer_read(
        clean["4496_scoreboard"],
        skipped.get("4496_scoreboard", False),
    )
    corpus_context = operational_context_read(
        clean["4495_replay_corpus"],
        skipped.get("4495_replay_corpus", False),
    )
    hardware_context = operational_context_read(
        clean["4497_hardware"],
        skipped.get("4497_hardware", False),
    )
    sota_context = operational_context_read(clean["4498_sota"], skipped.get("4498_sota", False))
    variant_rate = float(variant.get("variant_transfer_rate") or 0.0)
    end = time.time() if now_s is None else now_s

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "honest_verdict": _honest_verdict(
            a1=a1,
            a2=a2,
            a3=a3,
            l2_banked=bool(a4_a5.get("any_l2_banked")),
            variant_rate=variant_rate,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _preconditions_checked(root, provenance, clean),
        "a1_actions_to_first_levelup_reduction": a1,
        "a2_trust_energy_oracle_distinct_verdict": a2,
        "a3_energy_augmentation_loo_auroc": a3,
        "a4_a5_l2_banked": bool(a4_a5.get("any_l2_banked")),
        "a4_a5_l2_details": a4_a5,
        "variant_transfer_rate": variant_rate,
        "variant_transfer_scoreboard": variant,
        "operational_context": {
            "exp4495_replay_corpus": corpus_context,
            "exp4497_hardware": hardware_context,
            "exp4498_sota": sota_context,
        },
        "verifier_is_oracle": False,
        "verifier_claims": _verifier_claims(skipped=skipped, a2=a2, a4_a5=a4_a5),
        "flagged_artifacts_skipped": exclusions,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", []))
        + list(availability_report.get("flagged_artifacts_excluded", [])),
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "submitted_to_leaderboard": False,
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        a1=a1,
        a2=a2,
        a3=a3,
        a4_a5=a4_a5,
        variant_rate=variant_rate,
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for field in (
        "preconditions_checked",
        "a1_actions_to_first_levelup_reduction",
        "a2_trust_energy_oracle_distinct_verdict",
        "a3_energy_augmentation_loo_auroc",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be an object")
    if not isinstance(artifact.get("a4_a5_l2_banked"), bool):
        raise ValueError("a4_a5_l2_banked must be a bare bool")
    rate = artifact.get("variant_transfer_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise ValueError("variant_transfer_rate must be a bare float")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    claims = artifact.get("verifier_claims")
    if not isinstance(claims, list):
        raise ValueError("verifier_claims must be a list")
    for claim in claims:
        if not isinstance(claim, Mapping) or not isinstance(claim.get("verifier_is_oracle"), bool):
            raise ValueError("verifier_claims must declare verifier_is_oracle as bool")
    for field in ("flagged_artifacts_skipped", "cited_upstream_artifacts"):
        if not isinstance(artifact.get(field), list):
            raise ValueError(f"{field} must be a list")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream provenance row must be an object")
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    expected = checksum_from_inputs(
        provenance,
        a1=artifact["a1_actions_to_first_levelup_reduction"],
        a2=artifact["a2_trust_energy_oracle_distinct_verdict"],
        a3=artifact["a3_energy_augmentation_loo_auroc"],
        a4_a5=artifact["a4_a5_l2_details"],
        variant_rate=float(artifact["variant_transfer_rate"]),
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match inputs")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    capstone_live_flag_runner: LiveFlagRunner = base.run_live_flags,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )
    validate_artifact(artifact)
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact["capstone_live_adversarial_recheck"] = _capstone_recheck_status(
        capstone_live_flag_runner(path)
    )
    validate_artifact(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args() -> JsonDict:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_REL_PATH)
    args = parser.parse_args()
    return {"output": args.output}


def main() -> int:  # pragma: no cover
    args = _parse_args()
    output = write_artifact(REPO_ROOT, output_path=args["output"])
    print(output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
