"""Build the Exp 4478 .413 generic-solver capstone.

Spec refs: REQ-CAPSTONE-4478, SCENARIO-CAPSTONE-4478.

This aggregation reads the .413 upstream artifacts, skips flagged or
live-critical inputs before importing metrics, preserves the frozen publication
gate, and keeps the capstone's own oracle stamp false. Execution-grounded ARC
solves remain progress measurements, not verifier-moat headlines.
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
from carnot.reporting import capstone_v405_4390 as v405


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]
PublicationGateRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4478_capstone_v413.json")
EXPERIMENT_ID = 4478
RANDOM_SEED = 4478
SCHEMA = "carnot.capstone_v413_4478.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4478", "SCENARIO-CAPSTONE-4478"]
V412_REPRODUCIBLE_TOTAL_LEVELS = 39
GENERIC_LOO_V3_BASELINE = 6
GENERIC_SOLVER_GAP_STATES = {"closing", "partial", "total-gap"}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4467_dc22": Upstream(4467, Path("results/experiment_4467_solve_dc22_cegis_nocov.json")),
    "4468_sc25_deep": Upstream(
        4468, Path("results/experiment_4468_bank_sc25_provisional_levels.json")
    ),
    "4469_sc25_operator": Upstream(
        4469, Path("results/experiment_4469_generic_cast_grid_fsm_operator.json")
    ),
    "4470_sb26": Upstream(
        4470, Path("results/experiment_4470_color_match_slot_operator_solve_sb26.json")
    ),
    "4471_first_contact": Upstream(
        4471, Path("results/experiment_4471_first_contact_rotated_new_game.json")
    ),
    "4472_variant_loo_v4": Upstream(
        4472, Path("results/experiment_4472_variant_generic_transfer_benchmark_v4.json")
    ),
    "4473_submission": Upstream(
        4473, Path("results/experiment_4473_submission_package_prep_refresh.json")
    ),
    "4474_hygiene": Upstream(4474, Path("results/experiment_4474_registry_gaps_hygiene.json")),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels_grew",
    "generic_solver_gap_state",
    "generic_loo_solve_count_v4",
    "generic_transfer_rate_over_variants",
    "reproducible_total_levels",
    "reproducible_total_games",
    "submission_package_ready",
    "next_backlog",
    "publication_gate",
    "verifier_is_oracle",
    "inference_substrate",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed",
    "reproducible_total_levels_grew": (
        "BARE bool: reproducible_total_levels > 39 -- the .413 headline answer to "
        "'did we finally move the .412-flat metric?'"
    ),
    "generic_solver_gap_state": (
        "one honest string (closing/partial/total-gap) -- did dc22+sc25+sb26 close?"
    ),
    "generic_loo_solve_count_v4": (
        "bare int from exp4472 -- did the falsifiable generic metric rise above 6?"
    ),
    "generic_transfer_rate_over_variants": (
        "bare float from exp4472 -- the operator-mandated OOD-proxy generalization metric"
    ),
    "reproducible_total_levels": (
        "the authoritative monotonic sprint metric (target > 39 after dc22 + sc25 deepening)"
    ),
    "submission_package_ready": (
        "bare bool from exp4473 -- is the operator package ready to beat the 13-level "
        "prior baseline (now > 39)?"
    ),
    "next_backlog": (
        "the still-open residual_deltas + missing primitives that become the .414 "
        "generic-solver build backlog"
    ),
    "publication_gate": (
        "the G1-G4 publication_gate.py --json output; paper_ready stays True "
        "(FoVer 0.9131, FROZEN)"
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; upstream execution-grounded ARC "
        "solves carried separately so CIRCULAR_MOAT_OVERCLAIM does not fire"
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- reads upstream JSON + publication_gate.py; "
        "100us floor"
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256}; skipped flagged artifacts "
        "import no fields"
    ),
    "reproducibility_checksum": "content hash for reproducibility",
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4467_dc22": [
        "honest_verdict",
        "target_game",
        "dc22_grounded",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels",
        "reproducible_total_games",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4468_sc25_deep": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "new_sc25_levels_reproduced",
        "prior_sc25_levels_reproduced",
        "sc25_levels_reproduced_total",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4469_sc25_operator": [
        "honest_verdict",
        "target_game",
        "sc25_resolved_generically",
        "sc25_generic_level_reproduced",
        "offline_reproduced",
        "missing_verifier_gaps",
        "no_regression",
        "verifier_is_oracle",
    ],
    "4470_sb26": [
        "honest_verdict",
        "target_game",
        "color_match_operator_built",
        "selected_operator",
        "operator_result",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4471_first_contact": [
        "honest_verdict",
        "target_game",
        "target_selection",
        "routed_to",
        "selected_operator",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "submitted_to_leaderboard",
        "verifier_is_oracle",
    ],
    "4472_variant_loo_v4": [
        "honest_verdict",
        "generic_loo_solve_count_v3_baseline",
        "generic_loo_solve_count_v4",
        "generic_transfer_rate_over_variants",
        "variants_attempted",
        "variants_solved",
        "closed_residuals_by_413_operator",
        "preconditions_checked",
        "verifier_is_oracle",
    ],
    "4473_submission": [
        "honest_verdict",
        "submission_package_ready",
        "total_reproduced_levels_in_package",
        "prior_submitted_baseline_levels",
        "prior_package_412_levels",
        "beats_prior_baseline",
        "grew_vs_412",
        "submitted_to_leaderboard",
        "quarantined_games",
        "verifier_is_oracle",
    ],
    "4474_hygiene": [
        "honest_verdict",
        "reproducible_total_games",
        "reproducible_total_levels",
        "regression_guard_passed",
        "availability_report",
        "open_gap_ids",
        "registry_reconciliation",
    ],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _skipped_payload(payload: JsonDict) -> JsonDict:
    skipped = dict(payload)
    skipped["flagged_adversarial"] = True
    return skipped


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
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = (
            _skipped_payload(payload) if payload is not None and skipped else payload
        )
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


def _residual_rows(
    payload: Mapping[str, Any],
    source_artifact: str = "",
    default_game: str = "",
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    sources: list[Any] = []
    gaps = payload.get("missing_verifier_gaps")
    if isinstance(gaps, list):
        sources.extend(gaps)
    per_game = payload.get("per_game")
    if isinstance(per_game, list):
        sources.extend(per_game)
    for raw in sources:
        if isinstance(raw, str):
            game = default_game
            residual = raw
            status = "open"
            extras: Mapping[str, Any] = {}
        elif isinstance(raw, Mapping):
            game = str(raw.get("game") or default_game)
            residual = str(raw.get("residual_delta") or "")
            status = str(raw.get("status") or "open")
            extras = raw
        else:
            continue
        if not residual or residual == "none":
            continue
        key = (game, residual)
        if key in seen:
            continue
        seen.add(key)
        row = {"game": game, "residual_delta": residual, "status": status}
        if source_artifact:
            row["source_artifact"] = source_artifact
        for optional in (
            "gap_id",
            "retrieved_operator",
            "attempt_mode",
            "operator",
            "routed_to",
            "candidate_design",
        ):
            if extras.get(optional):
                row[optional] = extras[optional]
        rows.append(row)
    return rows


def dc22_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "dc22_l1_cleanly_banked": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "dc22_l1_cleanly_banked": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    banked = (
        base.str_metric(payload, "target_game") == "dc22"
        and base.bool_metric(payload, "dc22_grounded") is True
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    return {
        "state": "dc22_grounded_l1_banked" if banked else "dc22_open",
        "dc22_grounded": base.bool_metric(payload, "dc22_grounded") is True,
        "dc22_l1_cleanly_banked": banked,
        "gap_closed": banked,
        "target_game": base.str_metric(payload, "target_game"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4467_dc22"].path),
            "dc22",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def sc25_deep_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "new_sc25_level_banked": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "new_sc25_level_banked": False,
            "residual_deltas": [],
        }
    new_levels = base.int_metric(payload, "new_sc25_levels_reproduced")
    banked = (
        base.str_metric(payload, "target_game") == "sc25"
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
        and new_levels >= 1
    )
    return {
        "state": "sc25_deepened_banked" if banked else "sc25_deepening_open",
        "new_sc25_level_banked": banked,
        "new_sc25_levels_reproduced": new_levels,
        "prior_sc25_levels_reproduced": base.int_metric(payload, "prior_sc25_levels_reproduced"),
        "sc25_levels_reproduced_total": base.int_metric(payload, "sc25_levels_reproduced_total"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4468_sc25_deep"].path),
            "sc25",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def sc25_operator_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "sc25_resolved_generically": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "sc25_resolved_generically": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    resolved = (
        base.str_metric(payload, "target_game") == "sc25"
        and base.bool_metric(payload, "sc25_resolved_generically") is True
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "sc25_generic_level_reproduced") >= 1
    )
    return {
        "state": "sc25_generic_gap_closed" if resolved else "sc25_generic_gap_open",
        "sc25_resolved_generically": resolved,
        "gap_closed": resolved,
        "sc25_generic_level_reproduced": base.int_metric(
            payload,
            "sc25_generic_level_reproduced",
        ),
        "no_regression": base.bool_metric(payload, "no_regression") is True,
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4469_sc25_operator"].path),
            "sc25",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _color_match_operator_present(payload: Mapping[str, Any]) -> bool:
    if base.bool_metric(payload, "color_match_operator_built") is True:
        return True
    selected = payload.get("selected_operator")
    if isinstance(selected, Mapping) and selected.get("operator") == "color_match_slot_sequence_verifier":
        return True
    operator_result = payload.get("operator_result")
    return (
        isinstance(operator_result, Mapping)
        and operator_result.get("operator") == "color_match_slot_sequence_verifier"
    )


def sb26_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "color_match_operator_built": False,
            "sb26_banked": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "color_match_operator_built": False,
            "sb26_banked": False,
            "residual_deltas": [],
        }
    built = _color_match_operator_present(payload)
    banked = (
        base.str_metric(payload, "target_game") == "sb26"
        and built
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    return {
        "state": "sb26_banked" if banked else "sb26_open",
        "color_match_operator_built": built,
        "sb26_banked": banked,
        "target_game": base.str_metric(payload, "target_game"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4470_sb26"].path),
            "sb26",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def rotated_first_contact_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "banked_new_rotated_game": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "banked_new_rotated_game": False,
            "residual_deltas": [],
        }
    target = base.str_metric(payload, "target_game")
    banked = (
        bool(target)
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    return {
        "state": "rotated_game_banked" if banked else "rotated_game_open",
        "target_game": target,
        "banked_new_rotated_game": banked,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "routed_to": base.str_metric(payload, "routed_to"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4471_first_contact"].path),
            target,
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "submitted_to_leaderboard": base.bool_metric(payload, "submitted_to_leaderboard") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def variant_transfer_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "generic_loo_solve_count_v4": 0,
            "generic_transfer_rate_over_variants": 0.0,
            "v4_rose_above_baseline": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "generic_loo_solve_count_v4": 0,
            "generic_transfer_rate_over_variants": 0.0,
            "v4_rose_above_baseline": False,
            "residual_deltas": [],
        }
    baseline = base.int_metric(payload, "generic_loo_solve_count_v3_baseline")
    count = base.int_metric(payload, "generic_loo_solve_count_v4")
    rate = base.float_metric(payload, "generic_transfer_rate_over_variants")
    rate = 0.0 if rate is None else rate
    rose = count > baseline
    verdict = base.str_metric(payload, "honest_verdict")
    state = (
        "v4_rises_above_baseline"
        if rose
        else "blocked_baseline_smoke"
        if "blocked_baseline_smoke" in verdict
        else "v4_not_above_baseline"
    )
    return {
        "state": state,
        "generic_loo_solve_count_v3_baseline": baseline,
        "generic_loo_solve_count_v4": count,
        "generic_transfer_rate_over_variants": rate,
        "v4_rose_above_baseline": rose,
        "variants_attempted": base.int_metric(payload, "variants_attempted"),
        "variants_solved": base.int_metric(payload, "variants_solved"),
        "closed_residuals_by_413_operator": base.list_metric(
            payload,
            "closed_residuals_by_413_operator",
        ),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4472_variant_loo_v4"].path),
        ),
        "preconditions_checked": dict(payload.get("preconditions_checked", {}))
        if isinstance(payload.get("preconditions_checked"), Mapping)
        else {},
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": verdict,
    }


def submission_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "submission_package_ready": False,
            "total_reproduced_levels_in_package": 0,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "submission_package_ready": False,
            "total_reproduced_levels_in_package": 0,
        }
    levels = base.int_metric(payload, "total_reproduced_levels_in_package")
    prior_submitted = base.int_metric(payload, "prior_submitted_baseline_levels")
    prior_package = base.int_metric(payload, "prior_package_412_levels") or V412_REPRODUCIBLE_TOTAL_LEVELS
    submitted = base.bool_metric(payload, "submitted_to_leaderboard") is True
    ready = (
        base.bool_metric(payload, "submission_package_ready") is True
        and levels > V412_REPRODUCIBLE_TOTAL_LEVELS
        and levels > prior_submitted
        and not submitted
    )
    return {
        "state": "ready_beats_412_flat_metric" if ready else "not_ready",
        "submission_package_ready": ready,
        "total_reproduced_levels_in_package": levels,
        "prior_submitted_baseline_levels": prior_submitted,
        "prior_package_412_levels": prior_package,
        "beats_prior_baseline": base.bool_metric(payload, "beats_prior_baseline") is True,
        "grew_vs_412": base.bool_metric(payload, "grew_vs_412") is True,
        "submitted_to_leaderboard": submitted,
        "quarantined_games": base.list_metric(payload, "quarantined_games"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def hygiene_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
            "open_gap_ids": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
            "open_gap_ids": [],
        }
    reconciliation = payload.get("registry_reconciliation")
    open_gap_ids = [str(gap_id) for gap_id in base.list_metric(payload, "open_gap_ids")]
    if not open_gap_ids and isinstance(reconciliation, Mapping):
        open_gap_ids = [
            str(gap_id)
            for gap_id in reconciliation.get("open_gap_ids", [])
            if isinstance(gap_id, str)
        ]
    filled_gap_ids: list[str] = []
    if isinstance(reconciliation, Mapping):
        filled_gap_ids = [
            str(gap_id)
            for gap_id in reconciliation.get("filled_gap_ids", [])
            if isinstance(gap_id, str)
        ]
    return {
        "state": "reconciled" if base.int_metric(payload, "reproducible_total_levels") else "empty",
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "regression_guard_passed": base.bool_metric(payload, "regression_guard_passed") is True,
        "open_gap_ids": open_gap_ids,
        "filled_gap_ids": filled_gap_ids,
        "registry_reconciliation": dict(reconciliation) if isinstance(reconciliation, Mapping) else {},
        "availability_report": dict(payload.get("availability_report", {}))
        if isinstance(payload.get("availability_report"), Mapping)
        else {},
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="dc22_bank",
            required_keys=("4467_dc22",),
            verdict_fn=lambda present: dc22_read(present.get("4467_dc22"), False)[
                "dc22_l1_cleanly_banked"
            ],
        ),
        aggregate.AxisSpec(
            name="sc25_deeper_bank",
            required_keys=("4468_sc25_deep",),
            verdict_fn=lambda present: {
                "new_levels": sc25_deep_read(present.get("4468_sc25_deep"), False)[
                    "new_sc25_levels_reproduced"
                ],
                "moved_from_provisional": sc25_deep_read(
                    present.get("4468_sc25_deep"),
                    False,
                )["new_sc25_level_banked"],
                "total_levels": sc25_deep_read(present.get("4468_sc25_deep"), False)[
                    "sc25_levels_reproduced_total"
                ],
            },
        ),
        aggregate.AxisSpec(
            name="sc25_generic_operator",
            required_keys=("4469_sc25_operator",),
            verdict_fn=lambda present: sc25_operator_read(
                present.get("4469_sc25_operator"),
                False,
            )["gap_closed"],
        ),
        aggregate.AxisSpec(
            name="sb26_bank",
            required_keys=("4470_sb26",),
            verdict_fn=lambda present: sb26_read(present.get("4470_sb26"), False)[
                "sb26_banked"
            ],
        ),
        aggregate.AxisSpec(
            name="first_contact_new_game",
            required_keys=("4471_first_contact",),
            verdict_fn=lambda present: {
                "banked": rotated_first_contact_read(
                    present.get("4471_first_contact"),
                    False,
                )["banked_new_rotated_game"],
                "target_game": rotated_first_contact_read(
                    present.get("4471_first_contact"),
                    False,
                ).get("target_game", ""),
                "gap_id": (
                    rotated_first_contact_read(
                        present.get("4471_first_contact"),
                        False,
                    )
                    .get("residual_deltas", [{}])[0]
                    .get("gap_id", "")
                ),
            },
        ),
        aggregate.AxisSpec(
            name="variant_transfer_loo_v4",
            required_keys=("4472_variant_loo_v4",),
            verdict_fn=lambda present: {
                "generic_loo_solve_count_v4": variant_transfer_read(
                    present.get("4472_variant_loo_v4"),
                    False,
                )["generic_loo_solve_count_v4"],
                "variants_attempted": variant_transfer_read(
                    present.get("4472_variant_loo_v4"),
                    False,
                )["variants_attempted"],
                "variants_solved": variant_transfer_read(
                    present.get("4472_variant_loo_v4"),
                    False,
                )["variants_solved"],
            },
        ),
        aggregate.AxisSpec(
            name="submission_package",
            required_keys=("4473_submission",),
            verdict_fn=lambda present: {
                "ready": submission_read(present.get("4473_submission"), False)[
                    "submission_package_ready"
                ],
                "levels": submission_read(present.get("4473_submission"), False)[
                    "total_reproduced_levels_in_package"
                ],
                "submitted_to_leaderboard": submission_read(
                    present.get("4473_submission"),
                    False,
                )["submitted_to_leaderboard"],
            },
        ),
        aggregate.AxisSpec(
            name="registry_hygiene",
            required_keys=("4474_hygiene",),
            verdict_fn=lambda present: hygiene_read(present.get("4474_hygiene"), False)[
                "reproducible_total_levels"
            ],
        ),
    ]


def _publication_gate_or_gap(
    root: Path,
    runner: PublicationGateRunner,
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    publication_gate, check = v405._publication_gate_check(root, runner)  # noqa: SLF001
    if publication_gate is not None:
        return publication_gate, check, []
    return (
        {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(check.get("error", "unrunnable")),
        },
        check,
        [{"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}],
    )


def _missing_primitive_name(residual_delta: str) -> str:
    return residual_delta.removeprefix("missing_")


def build_next_backlog(
    *,
    dc22: Mapping[str, Any],
    sc25_deep: Mapping[str, Any],
    sc25_operator: Mapping[str, Any],
    sb26: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    variant_transfer: Mapping[str, Any],
    hygiene: Mapping[str, Any],
) -> JsonDict:
    residuals: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    for source in (dc22, sc25_deep, sc25_operator, sb26, first_contact, variant_transfer):
        for raw in source.get("residual_deltas", []):
            if not isinstance(raw, Mapping):
                continue
            residual = str(raw.get("residual_delta") or "")
            game = str(raw.get("game") or "")
            key = (game, residual)
            if not residual or key in seen:
                continue
            seen.add(key)
            residuals.append(dict(raw))

    missing_primitives = {
        _missing_primitive_name(str(row["residual_delta"]))
        for row in residuals
        if isinstance(row.get("residual_delta"), str)
        and str(row.get("residual_delta")).startswith("missing_")
    }
    operational_residuals: list[str] = []
    if variant_transfer.get("state") == "blocked_baseline_smoke":
        operational_residuals.append("variant_transfer_baseline_smoke_failed")
    return {
        "residual_deltas": residuals,
        "missing_primitives": sorted(missing_primitives),
        "open_gap_ids": [
            str(gap_id) for gap_id in hygiene.get("open_gap_ids", []) if isinstance(gap_id, str)
        ],
        "filled_gap_ids": [
            str(gap_id) for gap_id in hygiene.get("filled_gap_ids", []) if isinstance(gap_id, str)
        ],
        "closed_residuals_by_413_operator": base.list_metric(
            variant_transfer,
            "closed_residuals_by_413_operator",
        ),
        "operational_residuals": operational_residuals,
    }


def decide_generic_solver_gap_state(
    *,
    dc22: Mapping[str, Any],
    sc25_deep: Mapping[str, Any],
    sc25_operator: Mapping[str, Any],
    sb26: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    variant_transfer: Mapping[str, Any],
    next_backlog: Mapping[str, Any],
) -> str:
    residuals = next_backlog.get("residual_deltas")
    has_residuals = isinstance(residuals, list) and bool(residuals)
    closing = (
        dc22.get("dc22_l1_cleanly_banked") is True
        and sc25_deep.get("new_sc25_level_banked") is True
        and sc25_operator.get("gap_closed") is True
        and sb26.get("sb26_banked") is True
        and first_contact.get("banked_new_rotated_game") is True
        and variant_transfer.get("v4_rose_above_baseline") is True
        and not has_residuals
    )
    if closing:
        return "closing"
    if (
        dc22.get("dc22_l1_cleanly_banked") is True
        or sc25_deep.get("new_sc25_level_banked") is True
        or sc25_operator.get("gap_closed") is True
        or sb26.get("sb26_banked") is True
        or first_contact.get("banked_new_rotated_game") is True
        or variant_transfer.get("v4_rose_above_baseline") is True
    ):
        return "partial"
    return "total-gap"


def _honest_verdict(
    *,
    gap_state: str,
    grew: bool,
    loo_count: int,
    transfer_rate: float,
    total_levels: int,
    total_games: int,
    submission_ready: bool,
    publication_available: bool,
    paper_ready: bool,
) -> str:
    growth = "levels_grew" if grew else "levels_flat"
    publication = (
        "publication_ready"
        if publication_available and paper_ready
        else "publication_not_ready"
        if publication_available
        else "publication_gate_gap"
    )
    submission = "submission_ready" if submission_ready else "submission_not_ready"
    rate_bp = int(round(transfer_rate * 10_000))
    return (
        f"complete: v413_generic_solver_{gap_state}_{growth}_loo_v4_{loo_count}_"
        f"transfer_bp_{rate_bp}_levels_{total_levels}_games_{total_games}_"
        f"{submission}_{publication}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    *,
    grew: bool,
    gap_state: str,
    loo_count: int,
    transfer_rate: float,
    total_levels: int,
    total_games: int,
    submission_ready: bool,
    next_backlog: Mapping[str, Any],
) -> str:
    payload = {
        "generic_loo_solve_count_v4": loo_count,
        "generic_solver_gap_state": gap_state,
        "generic_transfer_rate_over_variants": transfer_rate,
        "next_backlog": next_backlog,
        "publication_gate": publication_gate,
        "reproducible_total_games": total_games,
        "reproducible_total_levels": total_levels,
        "reproducible_total_levels_grew": grew,
        "submission_package_ready": submission_ready,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _field_provenance(satisfied_by: str) -> dict[str, JsonDict]:
    return {
        field: {"principle": principle, "satisfied_by": satisfied_by}
        for field, principle in FIELD_PRINCIPLES.items()
    }


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


def _oracle_declarations(
    provenance: list[JsonDict],
    clean: Mapping[str, JsonDict | None],
) -> list[JsonDict]:
    declarations: list[JsonDict] = []
    for row in provenance:
        key = str(row["artifact_key"])
        payload = clean.get(key)
        declarations.append(
            {
                "artifact_key": key,
                "experiment_id": row["experiment_id"],
                "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
                "skipped": row["skipped"],
            }
        )
    return declarations


def _capstone_recheck_status(flags: list[dict[str, Any]]) -> JsonDict:
    circular = any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    critical = base.live_has_critical(flags)
    return {
        "status": "critical_flags" if critical else "clean",
        "flags": flags,
        "circular_moat_overclaim": circular,
    }


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    hygiene: Mapping[str, Any],
) -> JsonDict:
    provenance_by_key = {row["artifact_key"]: row for row in provenance}
    upstreams: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        row = provenance_by_key.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
            }
        )
    return {
        "publication_gate": dict(publication_gate_check),
        "upstream_artifacts": upstreams,
        "registry_hygiene": dict(hygiene),
        "robust_aggregate_available_helper": (
            "capstone_aggregate_available.aggregate_available_report_gaps"
        ),
        "leaderboard_submission": False,
    }


def _headline_answers(
    dc22: Mapping[str, Any],
    sc25_deep: Mapping[str, Any],
    sc25_operator: Mapping[str, Any],
    sb26: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    variant_transfer: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> JsonDict:
    return {
        "exp4467": {
            "state": dc22.get("state"),
            "dc22_l1_cleanly_banked": dc22.get("dc22_l1_cleanly_banked") is True,
            "gap_closed": dc22.get("gap_closed") is True,
            "residual_deltas": dc22.get("residual_deltas", []),
        },
        "exp4468": {
            "state": sc25_deep.get("state"),
            "new_sc25_level_banked": sc25_deep.get("new_sc25_level_banked") is True,
            "new_sc25_levels_reproduced": sc25_deep.get("new_sc25_levels_reproduced", 0),
            "sc25_levels_reproduced_total": sc25_deep.get("sc25_levels_reproduced_total", 0),
        },
        "exp4469": {
            "state": sc25_operator.get("state"),
            "sc25_resolved_generically": sc25_operator.get("sc25_resolved_generically") is True,
            "gap_closed": sc25_operator.get("gap_closed") is True,
            "residual_deltas": sc25_operator.get("residual_deltas", []),
        },
        "exp4470": {
            "state": sb26.get("state"),
            "color_match_operator_built": sb26.get("color_match_operator_built") is True,
            "sb26_banked": sb26.get("sb26_banked") is True,
            "reproduced_levels": sb26.get("reproduced_levels", 0),
        },
        "exp4471": {
            "state": first_contact.get("state"),
            "target_game": first_contact.get("target_game", ""),
            "banked_new_rotated_game": first_contact.get("banked_new_rotated_game") is True,
            "residual_deltas": first_contact.get("residual_deltas", []),
        },
        "exp4472": {
            "state": variant_transfer.get("state"),
            "generic_loo_solve_count_v3_baseline": variant_transfer.get(
                "generic_loo_solve_count_v3_baseline",
                GENERIC_LOO_V3_BASELINE,
            ),
            "generic_loo_solve_count_v4": variant_transfer.get(
                "generic_loo_solve_count_v4",
                0,
            ),
            "generic_transfer_rate_over_variants": variant_transfer.get(
                "generic_transfer_rate_over_variants",
                0.0,
            ),
            "v4_rose_above_baseline": variant_transfer.get("v4_rose_above_baseline") is True,
        },
        "exp4473": {
            "state": submission.get("state"),
            "submission_package_ready": submission.get("submission_package_ready") is True,
            "total_reproduced_levels_in_package": submission.get(
                "total_reproduced_levels_in_package",
                0,
            ),
            "prior_package_412_levels": submission.get(
                "prior_package_412_levels",
                V412_REPRODUCIBLE_TOTAL_LEVELS,
            ),
        },
    }


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    publication_gate, publication_gate_check, publication_gate_gaps = _publication_gate_or_gap(
        root,
        publication_gate_runner,
    )
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

    dc22 = dc22_read(clean["4467_dc22"], skipped.get("4467_dc22", False))
    sc25_deep = sc25_deep_read(
        clean["4468_sc25_deep"],
        skipped.get("4468_sc25_deep", False),
    )
    sc25_operator = sc25_operator_read(
        clean["4469_sc25_operator"],
        skipped.get("4469_sc25_operator", False),
    )
    sb26 = sb26_read(clean["4470_sb26"], skipped.get("4470_sb26", False))
    first_contact = rotated_first_contact_read(
        clean["4471_first_contact"],
        skipped.get("4471_first_contact", False),
    )
    variant_transfer = variant_transfer_read(
        clean["4472_variant_loo_v4"],
        skipped.get("4472_variant_loo_v4", False),
    )
    submission = submission_read(
        clean["4473_submission"],
        skipped.get("4473_submission", False),
    )
    hygiene = hygiene_read(clean["4474_hygiene"], skipped.get("4474_hygiene", False))

    next_backlog = build_next_backlog(
        dc22=dc22,
        sc25_deep=sc25_deep,
        sc25_operator=sc25_operator,
        sb26=sb26,
        first_contact=first_contact,
        variant_transfer=variant_transfer,
        hygiene=hygiene,
    )
    gap_state = decide_generic_solver_gap_state(
        dc22=dc22,
        sc25_deep=sc25_deep,
        sc25_operator=sc25_operator,
        sb26=sb26,
        first_contact=first_contact,
        variant_transfer=variant_transfer,
        next_backlog=next_backlog,
    )
    total_levels = int(hygiene.get("reproducible_total_levels") or 0)
    total_games = int(hygiene.get("reproducible_total_games") or 0)
    grew = total_levels > V412_REPRODUCIBLE_TOTAL_LEVELS
    loo_count = int(variant_transfer.get("generic_loo_solve_count_v4") or 0)
    transfer_rate = float(variant_transfer.get("generic_transfer_rate_over_variants") or 0.0)
    submission_ready = bool(submission.get("submission_package_ready"))
    paper_ready = bool(publication_gate.get("paper_ready"))
    publication_available = bool(publication_gate_check.get("runnable"))
    end = time.time() if now_s is None else now_s
    per_axis_gaps = list(availability_report.get("missing_upstream_artifacts", []))
    per_axis_gaps.extend(publication_gate_gaps)

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            gap_state=gap_state,
            grew=grew,
            loo_count=loo_count,
            transfer_rate=transfer_rate,
            total_levels=total_levels,
            total_games=total_games,
            submission_ready=submission_ready,
            publication_available=publication_available,
            paper_ready=paper_ready,
        ),
        "reproducible_total_levels_grew": grew,
        "generic_solver_gap_state": gap_state,
        "generic_loo_solve_count_v4": loo_count,
        "generic_transfer_rate_over_variants": transfer_rate,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "submission_package_ready": submission_ready,
        "submission_readiness_decision": submission.get("state", "missing_or_excluded"),
        "total_reproduced_levels_in_package": submission.get(
            "total_reproduced_levels_in_package",
            0,
        ),
        "next_backlog": next_backlog,
        "dc22_bank": dc22,
        "sc25_deeper_bank": sc25_deep,
        "sc25_generic_operator": sc25_operator,
        "sb26_bank": sb26,
        "first_contact_rotated_game": first_contact,
        "generic_loo_v4": variant_transfer,
        "submission_package": submission,
        "registry_hygiene": hygiene,
        "headline_question_answers": _headline_answers(
            dc22,
            sc25_deep,
            sc25_operator,
            sb26,
            first_contact,
            variant_transfer,
            submission,
        ),
        "publication_gate": publication_gate,
        "paper_ready": paper_ready,
        "unmet_gates": base.list_metric(publication_gate, "unmet_gates"),
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "upstream_oracle_declarations": _oracle_declarations(provenance, clean),
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root,
            publication_gate_check,
            provenance,
            hygiene,
        ),
        "per_axis_gaps": per_axis_gaps,
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "publication_gate_checksum": hashlib.sha256(
            json.dumps(publication_gate, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "submitted_to_leaderboard": False,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        publication_gate,
        grew=grew,
        gap_state=gap_state,
        loo_count=loo_count,
        transfer_rate=transfer_rate,
        total_levels=total_levels,
        total_games=total_games,
        submission_ready=submission_ready,
        next_backlog=next_backlog,
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(
        ("complete:", "success:", "passed:", "shipped:"),
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("reproducible_total_levels_grew"), bool):
        raise ValueError("reproducible_total_levels_grew must be a bare bool")
    if artifact.get("generic_solver_gap_state") not in GENERIC_SOLVER_GAP_STATES:
        raise ValueError("generic_solver_gap_state is not recognized")
    for field in (
        "generic_loo_solve_count_v4",
        "reproducible_total_levels",
        "reproducible_total_games",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare int")
    rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(rate, float):
        raise ValueError("generic_transfer_rate_over_variants must be a bare float")
    if not isinstance(artifact.get("submission_package_ready"), bool):
        raise ValueError("submission_package_ready must be a bare bool")
    if not isinstance(artifact.get("next_backlog"), Mapping):
        raise ValueError("next_backlog must be an object")
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
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
        artifact["publication_gate"],
        grew=bool(artifact["reproducible_total_levels_grew"]),
        gap_state=str(artifact["generic_solver_gap_state"]),
        loo_count=int(artifact["generic_loo_solve_count_v4"]),
        transfer_rate=float(artifact["generic_transfer_rate_over_variants"]),
        total_levels=int(artifact["reproducible_total_levels"]),
        total_games=int(artifact["reproducible_total_games"]),
        submission_ready=bool(artifact["submission_package_ready"]),
        next_backlog=artifact["next_backlog"],
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
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
    capstone_live_flag_runner: LiveFlagRunner = base.run_live_flags,
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
