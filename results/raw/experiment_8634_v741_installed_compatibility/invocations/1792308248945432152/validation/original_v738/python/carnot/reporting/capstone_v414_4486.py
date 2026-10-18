"""Build the Exp 4486 .414 capstone scorecard.

Spec refs: REQ-CAPSTONE-4486, SCENARIO-CAPSTONE-4486.

This module is intentionally an aggregation pass. It reads the upstream JSON
artifacts through the same summary/adversarial-check discipline used by recent
capstones, skips quarantined inputs before importing metrics, and reports the
.414 score signals without turning execution-grounded verifier claims into
oracle-distinct moat claims.
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
OUTPUT_REL_PATH = Path("results/experiment_4486_capstone_v414.json")
EXPERIMENT_ID = 4486
RANDOM_SEED = 4486
SCHEMA = "carnot.capstone_v414_4486.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4486", "SCENARIO-CAPSTONE-4486"]
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
    "4475_a1_stack": Upstream(
        4475,
        Path("results/experiment_4475_wire_stronger_generic_stack.json"),
    ),
    "4476_a2_features": Upstream(
        4476,
        Path("results/experiment_4476_verifier_features_v3_loo_gate.json"),
    ),
    "4477_a3_routing": Upstream(
        4477,
        Path("results/experiment_4477_per_game_online_discriminative.json"),
    ),
    "4479_a4_re86": Upstream(4479, Path("results/experiment_4479_solve_re86.json")),
    "4480_a4_bp35": Upstream(
        4480,
        Path("results/experiment_4480_solve_bp35_goal_directed.json"),
    ),
    "4481_closeout": Upstream(
        4481,
        Path("results/experiment_4481_variant_transfer_benchmark.json"),
    ),
    "4482_lint": Upstream(4482, Path("results/experiment_4482_nocov_default_lint.json")),
    "4483_registry": Upstream(
        4483,
        Path("results/experiment_4483_gate_decouple_registry_reconcile.json"),
    ),
    "4484_hardware": Upstream(
        4484,
        Path("results/experiment_4484_hardware_continuity_audit.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "offline_reproduced",
    "reproduced_levels",
    "preconditions_checked",
    "a1_generic_solve_rate",
    "a2_cross_game_loo_auroc_v3",
    "a3_per_game_discriminative_delta",
    "a4_goal_state_deepen",
    "twenty_five_game_closeout",
    "verifier_is_oracle",
    "verifier_claims",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with a terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_ so the reconciler classifies it as "
        "terminal (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit declaration (live_llm_inference | "
        "verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) "
        "so adversarial_verify applies the right floor."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort -- only reproduced levels count "
        "(ARC Solve Reproducibility)."
    ),
    "reproduced_levels": (
        "headline metric reproducible_total_levels grows monotonically; report the count "
        "banked, real-env-confirmed."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified before launching; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "a1_generic_solve_rate": (
        "the real A1 score signal is held-out generic-solve-rate BEFORE->AFTER, "
        "not reproducible_total_levels."
    ),
    "a2_cross_game_loo_auroc_v3": (
        "reports whether richer features beat the 0.503 LOO-AUROC v2 baseline."
    ),
    "a3_per_game_discriminative_delta": (
        "reports the per-game online discriminative routing delta without converting "
        "a null delta into a success."
    ),
    "a4_goal_state_deepen": (
        "reports clean re86/bp35 goal-state deepening only when offline reproduced."
    ),
    "twenty_five_game_closeout": (
        "reports the clean 25-game closeout from registry/variant-transfer evidence "
        "while recording stale registry mismatches."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; verifier-derived upstream claims "
        "declare their own circular/execution-grounded status."
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
    "4475_a1_stack": [
        "honest_verdict",
        "before_generic_solve_rate",
        "after_generic_solve_rate",
        "generic_solve_rate_delta",
        "before_solved",
        "after_solved",
        "attempted_games",
        "benchmark",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
    "4476_a2_features": [
        "honest_verdict",
        "v2_baseline_loo_auroc",
        "v3_loo_auroc",
        "v3_in_sample_auroc",
        "target_loo_auroc",
        "loo_gate_passed",
        "feature_class_loo_auroc",
        "feature_class_deltas",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
    "4477_a3_routing": [
        "honest_verdict",
        "baseline_solve_rate",
        "online_solve_rate",
        "solve_rate_delta",
        "baseline_actions_to_first_levelup",
        "online_actions_to_first_levelup",
        "actions_to_first_levelup_delta",
        "per_game_results",
        "online_verifier",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
    "4479_a4_re86": [
        "honest_verdict",
        "target_game",
        "registered_verifier_operator",
        "sprite_overlay_verifier_built",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels",
        "verifier_is_oracle",
        "preconditions_checked",
    ],
    "4480_a4_bp35": [
        "honest_verdict",
        "target_game",
        "goal_directed_solver_built",
        "goal_region_identified",
        "shape_aware_state_key",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels",
        "verifier_is_oracle",
        "preconditions_checked",
    ],
    "4481_closeout": [
        "honest_verdict",
        "solved_games",
        "variants_attempted",
        "variants_solved",
        "transfer_solve_rate",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels",
        "verifier_is_oracle",
        "preconditions_checked",
    ],
    "4482_lint": [
        "honest_verdict",
        "roadmap_lint_shipped",
        "precommit_hook_wired",
        "activation_guard_wired",
        "coverage_new_code_100",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
    "4483_registry": [
        "honest_verdict",
        "registry_reconciliation",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
    "4484_hardware": [
        "honest_verdict",
        "per_board_reachability",
        "per_board_status",
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
    ],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _number_from(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = base.float_metric(payload, field)
    if value is not None:
        return value
    nested = payload.get("benchmark") if isinstance(payload, Mapping) else None
    return base.float_metric(nested, field) if isinstance(nested, Mapping) else None


def _int_from(payload: Mapping[str, Any] | None, field: str) -> int:
    value = base.int_metric(payload, field)
    if value:
        return value
    nested = payload.get("benchmark") if isinstance(payload, Mapping) else None
    return base.int_metric(nested, field) if isinstance(nested, Mapping) else 0


def _flagged_payload(payload: JsonDict) -> JsonDict:
    flagged = dict(payload)
    flagged["flagged_adversarial"] = True
    return flagged


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
                    "reason": base._exclusion_reason(stamped, critical, parse_error),  # noqa: SLF001
                }
            )
    return raw_artifacts, provenance, exclusions


def _signal_from_delta(delta: float) -> str:
    if delta > 0.0:
        return "improved"
    if delta < 0.0:
        return "regressed"
    return "flat"


def a1_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "before_generic_solve_rate": 0.0,
            "after_generic_solve_rate": 0.0,
            "generic_solve_rate_delta": 0.0,
            "signal": "flat",
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "before_generic_solve_rate": 0.0,
            "after_generic_solve_rate": 0.0,
            "generic_solve_rate_delta": 0.0,
            "signal": "flat",
        }
    before = _number_from(payload, "before_generic_solve_rate") or 0.0
    after = _number_from(payload, "after_generic_solve_rate") or 0.0
    delta = _number_from(payload, "generic_solve_rate_delta")
    delta = after - before if delta is None else delta
    return {
        "state": f"heldout_generic_solve_rate_{_signal_from_delta(delta)}",
        "before_generic_solve_rate": before,
        "after_generic_solve_rate": after,
        "generic_solve_rate_delta": delta,
        "signal": _signal_from_delta(delta),
        "before_solved": _int_from(payload, "before_solved"),
        "after_solved": _int_from(payload, "after_solved"),
        "attempted_games": _int_from(payload, "attempted_games"),
        "measurement": str(
            payload.get("benchmark", {}).get("measurement", "")
            if isinstance(payload.get("benchmark"), Mapping)
            else ""
        ),
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def a2_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "v2_baseline_loo_auroc": 0.503096152732577,
            "v3_loo_auroc": 0.0,
            "loo_auroc_delta": 0.0,
            "richer_features_beat_baseline": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "v2_baseline_loo_auroc": 0.503096152732577,
            "v3_loo_auroc": 0.0,
            "loo_auroc_delta": 0.0,
            "richer_features_beat_baseline": False,
        }
    baseline = base.float_metric(payload, "v2_baseline_loo_auroc") or 0.503096152732577
    v3 = base.float_metric(payload, "v3_loo_auroc") or 0.0
    delta = v3 - baseline
    return {
        "state": "richer_features_beat_baseline" if delta > 0.0 else "richer_features_null",
        "v2_baseline_loo_auroc": baseline,
        "v3_loo_auroc": v3,
        "v3_in_sample_auroc": base.float_metric(payload, "v3_in_sample_auroc") or 0.0,
        "target_loo_auroc": base.float_metric(payload, "target_loo_auroc") or 0.0,
        "loo_auroc_delta": delta,
        "richer_features_beat_baseline": delta > 0.0,
        "loo_gate_passed": base.bool_metric(payload, "loo_gate_passed") is True,
        "feature_class_deltas": dict(payload.get("feature_class_deltas", {}))
        if isinstance(payload.get("feature_class_deltas"), Mapping)
        else {},
        "feature_class_loo_auroc": dict(payload.get("feature_class_loo_auroc", {}))
        if isinstance(payload.get("feature_class_loo_auroc"), Mapping)
        else {},
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def a3_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "solve_rate_delta": 0.0,
            "routing_helped": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "solve_rate_delta": 0.0,
            "routing_helped": False,
        }
    delta = base.float_metric(payload, "solve_rate_delta") or 0.0
    return {
        "state": f"per_game_routing_{_signal_from_delta(delta)}",
        "baseline_solve_rate": base.float_metric(payload, "baseline_solve_rate") or 0.0,
        "online_solve_rate": base.float_metric(payload, "online_solve_rate") or 0.0,
        "solve_rate_delta": delta,
        "routing_helped": delta > 0.0,
        "baseline_actions_to_first_levelup": base.int_metric(
            payload,
            "baseline_actions_to_first_levelup",
        ),
        "online_actions_to_first_levelup": base.int_metric(
            payload,
            "online_actions_to_first_levelup",
        ),
        "actions_to_first_levelup_delta": base.int_metric(
            payload,
            "actions_to_first_levelup_delta",
        ),
        "per_game_count": len(base.list_metric(payload, "per_game_results")),
        "online_verifier": dict(payload.get("online_verifier", {}))
        if isinstance(payload.get("online_verifier"), Mapping)
        else {},
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def solve_read(
    payload: JsonDict | None,
    skipped: bool,
    target_game: str,
    operator_key: str,
) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "target_game": target_game,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "verifier_is_oracle": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "target_game": target_game,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "verifier_is_oracle": False,
        }
    offline = base.bool_metric(payload, "offline_reproduced") is True
    reproduced = base.int_metric(payload, "reproduced_levels")
    operator_built = base.bool_metric(payload, operator_key) is True
    banked = (
        base.str_metric(payload, "target_game") == target_game
        and offline
        and reproduced >= 1
        and operator_built
    )
    return {
        "state": f"{target_game}_offline_reproduced" if banked else f"{target_game}_open",
        "target_game": target_game,
        "operator_key": operator_key,
        "operator_built": operator_built,
        "offline_reproduced": offline,
        "reproduced_levels": reproduced,
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def a4_read(re86: Mapping[str, Any], bp35: Mapping[str, Any]) -> JsonDict:
    reproduced = int(re86.get("reproduced_levels") or 0) + int(bp35.get("reproduced_levels") or 0)
    return {
        "state": "goal_state_deepen_banked" if reproduced else "goal_state_deepen_open",
        "new_reproduced_levels": reproduced,
        "offline_reproduced": reproduced > 0,
        "re86": dict(re86),
        "bp35": dict(bp35),
    }


def variant_closeout_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "solved_games_count": 0,
            "variants_attempted": 0,
            "variants_solved": 0,
            "transfer_solve_rate": 0.0,
            "verifier_is_oracle": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "solved_games_count": 0,
            "variants_attempted": 0,
            "variants_solved": 0,
            "transfer_solve_rate": 0.0,
            "verifier_is_oracle": False,
        }
    solved_games = [str(game) for game in base.list_metric(payload, "solved_games")]
    return {
        "state": "variant_closeout_25_games" if len(solved_games) >= 25 else "variant_closeout_partial",
        "solved_games": solved_games,
        "solved_games_count": len(solved_games),
        "variants_attempted": base.int_metric(payload, "variants_attempted"),
        "variants_solved": base.int_metric(payload, "variants_solved"),
        "transfer_solve_rate": base.float_metric(payload, "transfer_solve_rate") or 0.0,
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def registry_closeout_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "registry_authoritative_total_levels": 0,
            "registry_authoritative_total_games": 0,
            "registry_stale_mismatch": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "registry_authoritative_total_levels": 0,
            "registry_authoritative_total_games": 0,
            "registry_stale_mismatch": False,
        }
    reconciliation = payload.get("registry_reconciliation")
    header: Mapping[str, Any] = {}
    computed: Mapping[str, Any] = {}
    if isinstance(reconciliation, Mapping):
        raw_header = reconciliation.get("authoritative_header")
        raw_computed = reconciliation.get("computed_from_game_rows")
        header = raw_header if isinstance(raw_header, Mapping) else {}
        computed = raw_computed if isinstance(raw_computed, Mapping) else {}
    authoritative_levels = base.int_metric(header, "reproducible_total_levels")
    authoritative_games = base.int_metric(header, "reproducible_total_games")
    computed_levels = base.int_metric(computed, "reproducible_total_levels")
    computed_games = base.int_metric(computed, "reproducible_total_games")
    match = base.bool_metric(reconciliation, "reproduced_counts_match_header") is True
    return {
        "state": "registry_reconciled" if authoritative_levels else "registry_empty",
        "registry_authoritative_total_levels": authoritative_levels,
        "registry_authoritative_total_games": authoritative_games,
        "registry_computed_total_levels": computed_levels,
        "registry_computed_total_games": computed_games,
        "registry_stale_mismatch": not match,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def operational_context_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"state": "excluded_flagged_adversarial"}
    if payload is None:
        return {"state": "missing_or_excluded"}
    return {
        "state": "available",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "roadmap_lint_shipped": base.bool_metric(payload, "roadmap_lint_shipped") is True,
        "coverage_new_code_100": base.bool_metric(payload, "coverage_new_code_100") is True,
        "per_board_reachability": dict(payload.get("per_board_reachability", {}))
        if isinstance(payload.get("per_board_reachability"), Mapping)
        else {},
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="a1_generic_solve_rate",
            required_keys=("4475_a1_stack",),
            verdict_fn=lambda present: a1_read(present.get("4475_a1_stack"), False)["signal"],
        ),
        aggregate.AxisSpec(
            name="a2_cross_game_loo_auroc_v3",
            required_keys=("4476_a2_features",),
            verdict_fn=lambda present: a2_read(present.get("4476_a2_features"), False)[
                "richer_features_beat_baseline"
            ],
        ),
        aggregate.AxisSpec(
            name="a3_per_game_discriminative_delta",
            required_keys=("4477_a3_routing",),
            verdict_fn=lambda present: a3_read(present.get("4477_a3_routing"), False)[
                "routing_helped"
            ],
        ),
        aggregate.AxisSpec(
            name="a4_goal_state_deepen",
            required_keys=("4479_a4_re86", "4480_a4_bp35"),
            verdict_fn=lambda present: a4_read(
                solve_read(present.get("4479_a4_re86"), False, "re86", "sprite_overlay_verifier_built"),
                solve_read(present.get("4480_a4_bp35"), False, "bp35", "goal_directed_solver_built"),
            )["new_reproduced_levels"],
        ),
        aggregate.AxisSpec(
            name="twenty_five_game_closeout",
            required_keys=("4481_closeout", "4483_registry"),
            verdict_fn=lambda present: {
                "solved_games_count": variant_closeout_read(present.get("4481_closeout"), False)[
                    "solved_games_count"
                ],
                "registry_authoritative_total_levels": registry_closeout_read(
                    present.get("4483_registry"),
                    False,
                )["registry_authoritative_total_levels"],
            },
        ),
        aggregate.AxisSpec(
            name="operational_context",
            required_keys=("4482_lint", "4484_hardware"),
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
    re86: Mapping[str, Any],
    bp35: Mapping[str, Any],
    closeout: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _claim("a1_generic_solve_rate", False, "exp4475", skipped.get("4475_a1_stack", False)),
        _claim(
            "a2_cross_game_loo_auroc_v3",
            False,
            "exp4476",
            skipped.get("4476_a2_features", False),
        ),
        _claim(
            "a3_per_game_discriminative_delta",
            False,
            "exp4477",
            skipped.get("4477_a3_routing", False),
        ),
        _claim(
            "a4_re86_sprite_overlay_resize",
            bool(re86.get("verifier_is_oracle")),
            "exp4479",
            skipped.get("4479_a4_re86", False),
        ),
        _claim(
            "a4_bp35_goal_state_deepen",
            bool(bp35.get("verifier_is_oracle")),
            "exp4480",
            skipped.get("4480_a4_bp35", False),
        ),
        _claim(
            "twenty_five_game_variant_transfer_closeout",
            bool(closeout.get("verifier_is_oracle")),
            "exp4481",
            skipped.get("4481_closeout", False),
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
        "robust_aggregate_available_helper": (
            "capstone_aggregate_available.aggregate_available_report_gaps"
        ),
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
    a4: Mapping[str, Any],
    closeout: Mapping[str, Any],
) -> str:
    a2_state = "a2_beats_0503" if a2.get("richer_features_beat_baseline") is True else "a2_no_win"
    a3_state = "a3_gain" if a3.get("routing_helped") is True else "a3_no_delta"
    return (
        "complete: v414_"
        f"a1_{a1.get('signal', 'missing')}_"
        f"{a2_state}_{a3_state}_"
        f"a4_{a4.get('new_reproduced_levels', 0)}_levels_"
        f"closeout_{closeout.get('solved_games_count', 0)}_games_"
        f"total_{closeout.get('registry_authoritative_total_levels', 0)}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    *,
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    closeout: Mapping[str, Any],
    reproduced_levels: int,
) -> str:
    payload = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "closeout": closeout,
        "reproduced_levels": reproduced_levels,
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

    a1 = a1_read(clean["4475_a1_stack"], skipped.get("4475_a1_stack", False))
    a2 = a2_read(clean["4476_a2_features"], skipped.get("4476_a2_features", False))
    a3 = a3_read(clean["4477_a3_routing"], skipped.get("4477_a3_routing", False))
    re86 = solve_read(
        clean["4479_a4_re86"],
        skipped.get("4479_a4_re86", False),
        "re86",
        "sprite_overlay_verifier_built",
    )
    bp35 = solve_read(
        clean["4480_a4_bp35"],
        skipped.get("4480_a4_bp35", False),
        "bp35",
        "goal_directed_solver_built",
    )
    a4 = a4_read(re86, bp35)
    variant = variant_closeout_read(
        clean["4481_closeout"],
        skipped.get("4481_closeout", False),
    )
    registry = registry_closeout_read(
        clean["4483_registry"],
        skipped.get("4483_registry", False),
    )
    closeout = {**variant, **registry}
    lint_context = operational_context_read(clean["4482_lint"], skipped.get("4482_lint", False))
    hardware_context = operational_context_read(
        clean["4484_hardware"],
        skipped.get("4484_hardware", False),
    )
    reproduced_levels = int(closeout.get("registry_authoritative_total_levels") or 0)
    end = time.time() if now_s is None else now_s

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "honest_verdict": _honest_verdict(a1=a1, a2=a2, a3=a3, a4=a4, closeout=closeout),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": reproduced_levels > 0,
        "reproduced_levels": reproduced_levels,
        "a1_generic_solve_rate": a1,
        "a2_cross_game_loo_auroc_v3": a2,
        "a3_per_game_discriminative_delta": a3,
        "a4_goal_state_deepen": a4,
        "twenty_five_game_closeout": closeout,
        "operational_context": {
            "exp4482_nocov_default_lint": lint_context,
            "exp4484_hardware_continuity": hardware_context,
        },
        "verifier_is_oracle": False,
        "verifier_claims": _verifier_claims(
            skipped=skipped,
            re86=re86,
            bp35=bp35,
            closeout=variant,
        ),
        "flagged_artifacts_skipped": exclusions,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(root, provenance, clean),
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
        a4=a4,
        closeout=closeout,
        reproduced_levels=reproduced_levels,
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
    if not isinstance(artifact.get("offline_reproduced"), bool):
        raise ValueError("offline_reproduced must be a bare bool")
    if not isinstance(artifact.get("reproduced_levels"), int) or isinstance(
        artifact.get("reproduced_levels"),
        bool,
    ):
        raise ValueError("reproduced_levels must be a bare int")
    for field in (
        "preconditions_checked",
        "a1_generic_solve_rate",
        "a2_cross_game_loo_auroc_v3",
        "a3_per_game_discriminative_delta",
        "a4_goal_state_deepen",
        "twenty_five_game_closeout",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be an object")
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
        a1=artifact["a1_generic_solve_rate"],
        a2=artifact["a2_cross_game_loo_auroc_v3"],
        a3=artifact["a3_per_game_discriminative_delta"],
        a4=artifact["a4_goal_state_deepen"],
        closeout=artifact["twenty_five_game_closeout"],
        reproduced_levels=int(artifact["reproduced_levels"]),
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
