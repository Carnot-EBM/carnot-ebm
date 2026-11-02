"""Build the Exp 4465 .412 generic-solver capstone.

Spec refs: REQ-CAPSTONE-4465, SCENARIO-CAPSTONE-4465.

This is a pure aggregation artifact. It reads upstream JSON results, uses the
stable publication gate, reports missing upstreams as per-axis gaps, and keeps
the capstone's own `verifier_is_oracle` stamp false. Upstream execution-grounded
ARC solves remain progress measurements, not verifier-moat headlines.
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
OUTPUT_REL_PATH = Path("results/experiment_4465_capstone_v412.json")
EXPERIMENT_ID = 4465
RANDOM_SEED = 4465
SCHEMA = "carnot.capstone_v412_4465.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4465", "SCENARIO-CAPSTONE-4465"]
GENERIC_SOLVER_GAP_STATES = {"closing", "partial", "total-gap"}
DC22_OPEN_GAP_ID = "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"
SC25_OPEN_GAP_ID = "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER"
SB26_OPEN_GAP_ID = "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"
TR87_CLOSED_GAP_ID = (
    "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path
    glob_pattern: str = ""


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4455_dc22": Upstream(
        4455,
        Path("results/experiment_4455_solve_dc22_cegis_config_rule.json"),
    ),
    "4456_glyph_rewrite": Upstream(
        4456,
        Path("results/experiment_4456_generic_glyph_rewrite_operator.json"),
    ),
    "4457_cast_grid": Upstream(
        4457,
        Path("results/experiment_4457_cast_grid_spell_shrink_tank_exit.json"),
        "results/experiment_4457_*.json",
    ),
    "4458_first_contact": Upstream(
        4458,
        Path("results/experiment_4458_first_contact_new_game.json"),
    ),
    "4459_loo_v3": Upstream(
        4459,
        Path("results/experiment_4459_loo_generic_solve_benchmark_v3.json"),
    ),
    "4460_submission": Upstream(
        4460,
        Path("results/experiment_4460_submission_package_prep.json"),
    ),
    "4461_hygiene": Upstream(
        4461,
        Path("results/experiment_4461_registry_gaps_hygiene.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "generic_solver_gap_state",
    "generic_loo_solve_count_v3",
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
    "generic_solver_gap_state": (
        "one honest string (closing/partial/total-gap) -- the .412 headline answer "
        "to 'is the example corpus + new operators closing the per-game-RE gap?'"
    ),
    "generic_loo_solve_count_v3": (
        "bare int from exp4459 -- did the falsifiable generic metric rise above 5?"
    ),
    "reproducible_total_levels": (
        "the authoritative monotonic sprint metric (target >= 40 after dc22 + sc25 deepening)"
    ),
    "submission_package_ready": (
        "bare bool from exp4460 -- is the operator package ready to beat the 13-level prior baseline?"
    ),
    "next_backlog": (
        "the still-open residual_deltas + missing primitives that become the .413 "
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
    "4455_dc22": [
        "honest_verdict",
        "target_game",
        "dc22_grounded",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "reproducible_total_levels",
        "verifier_is_oracle",
    ],
    "4456_glyph_rewrite": [
        "honest_verdict",
        "target_game",
        "tr87_resolved_generically",
        "tr87_generic_level_reproduced",
        "offline_reproduced",
        "closed_gap_ids",
        "missing_verifier_gaps",
        "no_regression",
        "verifier_is_oracle",
    ],
    "4457_cast_grid": [
        "honest_verdict",
        "target_game",
        "sc25_resolved_generically",
        "provisional_promoted_to_reproduced",
        "offline_reproduced",
        "reproduced_levels",
        "closed_gap_ids",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4458_first_contact": [
        "honest_verdict",
        "target_game",
        "never_attempted_game",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "reproducible_total_levels",
        "submitted_to_leaderboard",
        "verifier_is_oracle",
    ],
    "4459_loo_v3": [
        "honest_verdict",
        "generic_loo_solve_count_v2_baseline",
        "generic_loo_solve_count_v3",
        "loo_gate_passed",
        "missing_verifier_gaps",
        "per_game",
        "closed_residuals_by_412_operator",
        "offline_reproduced",
        "verifier_is_oracle",
    ],
    "4460_submission": [
        "honest_verdict",
        "submission_package_ready",
        "total_reproduced_levels_in_package",
        "prior_submitted_baseline_levels",
        "beats_prior_baseline",
        "submitted_to_leaderboard",
        "quarantined_games",
        "verifier_is_oracle",
    ],
    "4461_hygiene": [
        "honest_verdict",
        "reproducible_total_games",
        "reproducible_total_levels",
        "regression_guard_passed",
        "availability_report",
        "open_gap_ids",
        "registry_reconciliation",
    ],
}


def _selected_path(root: Path, upstream: Upstream) -> Path:
    fixed = root / upstream.path
    if fixed.exists() or not upstream.glob_pattern:
        return fixed
    matches = sorted(root.glob(upstream.glob_pattern))
    return matches[0] if matches else fixed


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: _selected_path(root, upstream) for key, upstream in DEFAULT_UPSTREAMS.items()}


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
        except (OSError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = (
            _skipped_payload(payload) if payload is not None and skipped else payload
        )
        row = {
            "artifact_key": key,
            "experiment_id": upstream.experiment_id,
            "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
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
                    "path": row["path"],
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
        for optional in ("gap_id", "retrieved_operator", "attempt_mode", "operator", "routed_to"):
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
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4455_dc22"].path),
            "dc22",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def glyph_rewrite_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "tr87_resolved_generically": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "tr87_resolved_generically": False,
            "gap_closed": False,
            "residual_deltas": [],
        }
    closed_ids = [str(gap_id) for gap_id in base.list_metric(payload, "closed_gap_ids")]
    resolved = (
        base.str_metric(payload, "target_game") == "tr87"
        and base.bool_metric(payload, "tr87_resolved_generically") is True
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "tr87_generic_level_reproduced") >= 1
    )
    gap_closed = resolved and TR87_CLOSED_GAP_ID in closed_ids
    return {
        "state": "tr87_closed_generically" if gap_closed else "tr87_open",
        "tr87_resolved_generically": resolved,
        "gap_closed": gap_closed,
        "closed_gap_ids": closed_ids,
        "tr87_generic_level_reproduced": base.int_metric(payload, "tr87_generic_level_reproduced"),
        "no_regression": base.bool_metric(payload, "no_regression") is True,
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4456_glyph_rewrite"].path),
            "tr87",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def cast_grid_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "sc25_gap_closed": False,
            "sc25_level_banked": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "sc25_gap_closed": False,
            "sc25_level_banked": False,
            "residual_deltas": [],
        }
    banked = (
        base.str_metric(payload, "target_game") == "sc25"
        and base.bool_metric(payload, "sc25_resolved_generically") is True
        and base.bool_metric(payload, "provisional_promoted_to_reproduced") is True
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    closed_ids = [str(gap_id) for gap_id in base.list_metric(payload, "closed_gap_ids")]
    return {
        "state": "sc25_closed_and_banked" if banked else "sc25_open",
        "sc25_gap_closed": banked,
        "sc25_level_banked": banked,
        "closed_gap_ids": closed_ids,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4457_cast_grid"].path),
            "sc25",
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def first_contact_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "banked_new_game": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "banked_new_game": False,
            "residual_deltas": [],
        }
    target = base.str_metric(payload, "target_game")
    banked = base.bool_metric(payload, "offline_reproduced") is True and (
        base.int_metric(payload, "reproduced_levels") >= 1
    )
    never_attempted = base.bool_metric(payload, "never_attempted_game")
    banked_new_game = banked and bool(target) and never_attempted is not False
    return {
        "state": "new_game_banked" if banked_new_game else "new_game_open",
        "target_game": target,
        "banked_new_game": banked_new_game,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4458_first_contact"].path),
            target,
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "submitted_to_leaderboard": base.bool_metric(payload, "submitted_to_leaderboard") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def loo_v3_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "generic_loo_solve_count_v3": 0,
            "v3_rose_above_baseline": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "generic_loo_solve_count_v3": 0,
            "v3_rose_above_baseline": False,
            "residual_deltas": [],
        }
    baseline = base.int_metric(payload, "generic_loo_solve_count_v2_baseline")
    count = base.int_metric(payload, "generic_loo_solve_count_v3")
    rose = count > baseline
    return {
        "state": "v3_rises_above_baseline" if rose else "v3_not_above_baseline",
        "generic_loo_solve_count_v2_baseline": baseline,
        "generic_loo_solve_count_v3": count,
        "v3_rose_above_baseline": rose,
        "loo_gate_passed": base.bool_metric(payload, "loo_gate_passed") is True,
        "per_game": base.list_metric(payload, "per_game"),
        "closed_residuals_by_412_operator": base.list_metric(
            payload,
            "closed_residuals_by_412_operator",
        ),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4459_loo_v3"].path),
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
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
    baseline = base.int_metric(payload, "prior_submitted_baseline_levels")
    submitted = base.bool_metric(payload, "submitted_to_leaderboard") is True
    ready = (
        base.bool_metric(payload, "submission_package_ready") is True
        and levels > baseline
        and not submitted
    )
    return {
        "state": "ready_beats_prior_baseline" if ready else "not_ready",
        "submission_package_ready": ready,
        "total_reproduced_levels_in_package": levels,
        "prior_submitted_baseline_levels": baseline,
        "beats_prior_baseline": base.bool_metric(payload, "beats_prior_baseline") is True,
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
    return {
        "state": "reconciled" if base.int_metric(payload, "reproducible_total_levels") else "empty",
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "regression_guard_passed": base.bool_metric(payload, "regression_guard_passed") is True,
        "open_gap_ids": open_gap_ids,
        "registry_reconciliation": dict(reconciliation) if isinstance(reconciliation, Mapping) else {},
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="dc22_bank",
            required_keys=("4455_dc22",),
            verdict_fn=lambda present: dc22_read(present.get("4455_dc22"), False)[
                "dc22_l1_cleanly_banked"
            ],
        ),
        aggregate.AxisSpec(
            name="glyph_rewrite",
            required_keys=("4456_glyph_rewrite",),
            verdict_fn=lambda present: glyph_rewrite_read(
                present.get("4456_glyph_rewrite"),
                False,
            )["gap_closed"],
        ),
        aggregate.AxisSpec(
            name="cast_grid",
            required_keys=("4457_cast_grid",),
            verdict_fn=lambda present: {
                "sc25_closed": cast_grid_read(present.get("4457_cast_grid"), False)[
                    "sc25_gap_closed"
                ],
                "sc25_levels": cast_grid_read(present.get("4457_cast_grid"), False).get(
                    "reproduced_levels",
                    0,
                ),
            },
        ),
        aggregate.AxisSpec(
            name="first_contact_new_game",
            required_keys=("4458_first_contact",),
            verdict_fn=lambda present: {
                "banked": first_contact_read(present.get("4458_first_contact"), False)[
                    "banked_new_game"
                ],
                "target_game": first_contact_read(
                    present.get("4458_first_contact"),
                    False,
                ).get("target_game", ""),
            },
        ),
        aggregate.AxisSpec(
            name="loo_v3",
            required_keys=("4459_loo_v3",),
            verdict_fn=lambda present: loo_v3_read(present.get("4459_loo_v3"), False)[
                "generic_loo_solve_count_v3"
            ],
        ),
        aggregate.AxisSpec(
            name="submission_package",
            required_keys=("4460_submission",),
            verdict_fn=lambda present: {
                "ready": submission_read(present.get("4460_submission"), False)[
                    "submission_package_ready"
                ],
                "levels": submission_read(present.get("4460_submission"), False)[
                    "total_reproduced_levels_in_package"
                ],
                "submitted_to_leaderboard": submission_read(
                    present.get("4460_submission"),
                    False,
                )["submitted_to_leaderboard"],
            },
        ),
        aggregate.AxisSpec(
            name="registry_hygiene",
            required_keys=("4461_hygiene",),
            verdict_fn=lambda present: hygiene_read(present.get("4461_hygiene"), False)[
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
    cast_grid: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    loo_v3: Mapping[str, Any],
    hygiene: Mapping[str, Any],
) -> JsonDict:
    residuals: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    for source in (dc22, cast_grid, first_contact, loo_v3):
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

    open_gap_ids = [
        str(gap_id) for gap_id in hygiene.get("open_gap_ids", []) if isinstance(gap_id, str)
    ]
    missing_primitives = {
        _missing_primitive_name(str(row["residual_delta"]))
        for row in residuals
        if isinstance(row.get("residual_delta"), str)
        and str(row.get("residual_delta")).startswith("missing_")
    }
    if DC22_OPEN_GAP_ID in open_gap_ids:
        missing_primitives.add("config_rule_verifier_grounding")
    return {
        "residual_deltas": residuals,
        "missing_primitives": sorted(missing_primitives),
        "open_gap_ids": open_gap_ids,
        "closed_residuals_by_412_operator": base.list_metric(
            loo_v3,
            "closed_residuals_by_412_operator",
        ),
    }


def decide_generic_solver_gap_state(
    *,
    dc22: Mapping[str, Any],
    glyph_rewrite: Mapping[str, Any],
    cast_grid: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    loo_v3: Mapping[str, Any],
    next_backlog: Mapping[str, Any],
) -> str:
    residuals = next_backlog.get("residual_deltas")
    has_residuals = isinstance(residuals, list) and bool(residuals)
    closing = (
        dc22.get("dc22_l1_cleanly_banked") is True
        and glyph_rewrite.get("gap_closed") is True
        and cast_grid.get("sc25_gap_closed") is True
        and first_contact.get("banked_new_game") is True
        and loo_v3.get("v3_rose_above_baseline") is True
        and not has_residuals
    )
    if closing:
        return "closing"
    if (
        glyph_rewrite.get("gap_closed") is True
        or loo_v3.get("v3_rose_above_baseline") is True
        or dc22.get("dc22_l1_cleanly_banked") is True
        or cast_grid.get("sc25_gap_closed") is True
        or first_contact.get("banked_new_game") is True
    ):
        return "partial"
    return "total-gap"


def _honest_verdict(
    *,
    gap_state: str,
    loo_count: int,
    total_levels: int,
    total_games: int,
    submission_ready: bool,
    publication_available: bool,
    paper_ready: bool,
) -> str:
    publication = (
        "publication_ready"
        if publication_available and paper_ready
        else "publication_not_ready"
        if publication_available
        else "publication_gate_gap"
    )
    submission = "submission_ready" if submission_ready else "submission_not_ready"
    return (
        f"complete: v412_generic_solver_{gap_state}_loo_v3_{loo_count}_"
        f"levels_{total_levels}_games_{total_games}_{submission}_{publication}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    *,
    gap_state: str,
    loo_count: int,
    total_levels: int,
    total_games: int,
    submission_ready: bool,
    next_backlog: Mapping[str, Any],
) -> str:
    payload = {
        "generic_loo_solve_count_v3": loo_count,
        "generic_solver_gap_state": gap_state,
        "next_backlog": next_backlog,
        "publication_gate": publication_gate,
        "reproducible_total_games": total_games,
        "reproducible_total_levels": total_levels,
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
    glyph_rewrite: Mapping[str, Any],
    cast_grid: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    loo_v3: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> JsonDict:
    return {
        "exp4455": {
            "state": dc22.get("state"),
            "dc22_l1_cleanly_banked": dc22.get("dc22_l1_cleanly_banked") is True,
            "gap_closed": dc22.get("gap_closed") is True,
            "residual_deltas": dc22.get("residual_deltas", []),
        },
        "exp4456": {
            "state": glyph_rewrite.get("state"),
            "tr87_resolved_generically": glyph_rewrite.get("tr87_resolved_generically") is True,
            "gap_closed": glyph_rewrite.get("gap_closed") is True,
            "closed_gap_ids": glyph_rewrite.get("closed_gap_ids", []),
        },
        "exp4457": {
            "state": cast_grid.get("state"),
            "sc25_gap_closed": cast_grid.get("sc25_gap_closed") is True,
            "sc25_level_banked": cast_grid.get("sc25_level_banked") is True,
            "residual_deltas": cast_grid.get("residual_deltas", []),
        },
        "exp4458": {
            "state": first_contact.get("state"),
            "target_game": first_contact.get("target_game", ""),
            "banked_new_game": first_contact.get("banked_new_game") is True,
            "residual_deltas": first_contact.get("residual_deltas", []),
        },
        "exp4459": {
            "state": loo_v3.get("state"),
            "generic_loo_solve_count_v2_baseline": loo_v3.get(
                "generic_loo_solve_count_v2_baseline",
                0,
            ),
            "generic_loo_solve_count_v3": loo_v3.get("generic_loo_solve_count_v3", 0),
            "v3_rose_above_baseline": loo_v3.get("v3_rose_above_baseline") is True,
            "residual_deltas": loo_v3.get("residual_deltas", []),
        },
        "exp4460": {
            "state": submission.get("state"),
            "submission_package_ready": submission.get("submission_package_ready") is True,
            "total_reproduced_levels_in_package": submission.get(
                "total_reproduced_levels_in_package",
                0,
            ),
            "prior_submitted_baseline_levels": submission.get(
                "prior_submitted_baseline_levels",
                0,
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

    dc22 = dc22_read(clean["4455_dc22"], skipped.get("4455_dc22", False))
    glyph_rewrite = glyph_rewrite_read(
        clean["4456_glyph_rewrite"],
        skipped.get("4456_glyph_rewrite", False),
    )
    cast_grid = cast_grid_read(clean["4457_cast_grid"], skipped.get("4457_cast_grid", False))
    first_contact = first_contact_read(
        clean["4458_first_contact"],
        skipped.get("4458_first_contact", False),
    )
    loo_v3 = loo_v3_read(clean["4459_loo_v3"], skipped.get("4459_loo_v3", False))
    submission = submission_read(
        clean["4460_submission"],
        skipped.get("4460_submission", False),
    )
    hygiene = hygiene_read(clean["4461_hygiene"], skipped.get("4461_hygiene", False))

    next_backlog = build_next_backlog(
        dc22=dc22,
        cast_grid=cast_grid,
        first_contact=first_contact,
        loo_v3=loo_v3,
        hygiene=hygiene,
    )
    gap_state = decide_generic_solver_gap_state(
        dc22=dc22,
        glyph_rewrite=glyph_rewrite,
        cast_grid=cast_grid,
        first_contact=first_contact,
        loo_v3=loo_v3,
        next_backlog=next_backlog,
    )
    total_levels = int(hygiene.get("reproducible_total_levels") or 0)
    total_games = int(hygiene.get("reproducible_total_games") or 0)
    loo_count = int(loo_v3.get("generic_loo_solve_count_v3") or 0)
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
            loo_count=loo_count,
            total_levels=total_levels,
            total_games=total_games,
            submission_ready=submission_ready,
            publication_available=publication_available,
            paper_ready=paper_ready,
        ),
        "generic_solver_gap_state": gap_state,
        "generic_loo_solve_count_v3": loo_count,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "submission_package_ready": submission_ready,
        "submission_readiness_decision": submission.get("state", "missing_or_excluded"),
        "next_backlog": next_backlog,
        "dc22_bank": dc22,
        "glyph_rewrite": glyph_rewrite,
        "cast_grid": cast_grid,
        "first_contact_new_game": first_contact,
        "loo_v3": loo_v3,
        "submission_package": submission,
        "registry_hygiene": hygiene,
        "headline_question_answers": _headline_answers(
            dc22,
            glyph_rewrite,
            cast_grid,
            first_contact,
            loo_v3,
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
        gap_state=gap_state,
        loo_count=loo_count,
        total_levels=total_levels,
        total_games=total_games,
        submission_ready=submission_ready,
        next_backlog=next_backlog,
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(
        ("complete:", "success:", "passed:", "shipped:"),
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("generic_solver_gap_state") not in GENERIC_SOLVER_GAP_STATES:
        raise ValueError("generic_solver_gap_state is not recognized")
    for field in (
        "generic_loo_solve_count_v3",
        "reproducible_total_levels",
        "reproducible_total_games",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare int")
    if not isinstance(artifact.get("submission_package_ready"), bool):
        raise ValueError("submission_package_ready must be a bare bool")
    if not isinstance(artifact.get("next_backlog"), Mapping):
        raise ValueError("next_backlog must be an object")
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")  # pragma: no cover
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")  # pragma: no cover
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")  # pragma: no cover
    for row in provenance:
        if not isinstance(row, Mapping):  # pragma: no cover
            raise ValueError("upstream provenance row must be an object")
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")  # pragma: no cover
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")  # pragma: no cover
    expected = checksum_from_inputs(
        provenance,
        artifact["publication_gate"],
        gap_state=str(artifact["generic_solver_gap_state"]),
        loo_count=int(artifact["generic_loo_solve_count_v3"]),
        total_levels=int(artifact["reproducible_total_levels"]),
        total_games=int(artifact["reproducible_total_games"]),
        submission_ready=bool(artifact["submission_package_ready"]),
        next_backlog=artifact["next_backlog"],
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match inputs")  # pragma: no cover


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
