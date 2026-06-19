"""Build the Exp 4453 .411 generic-solver capstone.

Spec refs: REQ-CAPSTONE-4453, SCENARIO-CAPSTONE-4453.

This capstone is a pure aggregation artifact. It reads upstream JSON results,
uses the publication gate script as the stable paper-readiness source, and
keeps the capstone's own `verifier_is_oracle` stamp false. Upstream ARC solves
that are execution-grounded are reported as ARC progress, not as verifier-moat
headlines.
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
OUTPUT_REL_PATH = Path("results/experiment_4453_capstone_v411.json")
EXPERIMENT_ID = 4453
RANDOM_SEED = 4453
SCHEMA = "carnot.capstone_v411_4453.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4453", "SCENARIO-CAPSTONE-4453"]

GENERIC_SOLVER_GAP_STATES = {"closing", "partial", "total-gap"}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4443_g50t_bank": Upstream(
        4443,
        Path("results/experiment_4443_bank_g50t_example_conditioned_win.json"),
    ),
    "4444_config_rule": Upstream(
        4444,
        Path("results/experiment_4444_generic_config_rule_verifier_operator.json"),
    ),
    "4445_object_motion": Upstream(
        4445,
        Path("results/experiment_4445_generic_object_motion_world_model_operator.json"),
    ),
    "4446_first_contact": Upstream(
        4446,
        Path("results/experiment_4446_drive_generic_first_contact_bank.json"),
    ),
    "4447_library": Upstream(
        4447,
        Path("results/experiment_4447_lilo_documented_primitive_library.json"),
    ),
    "4448_loo_v2": Upstream(
        4448,
        Path("results/experiment_4448_loo_generic_solve_benchmark_v2.json"),
    ),
    "4449_hygiene": Upstream(
        4449,
        Path("results/experiment_4449_registry_gaps_hygiene.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "generic_solver_gap_state",
    "generic_loo_solve_count_v2",
    "reproducible_total_levels",
    "reproducible_total_games",
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
        "one honest string (closing/partial/total-gap) -- the .411 headline answer "
        "to 'is the example corpus closing the per-game-RE gap?'"
    ),
    "generic_loo_solve_count_v2": (
        "bare int from exp4448 -- did the falsifiable generic metric rise above 2?"
    ),
    "reproducible_total_levels": "the authoritative monotonic sprint metric (target >= 38)",
    "next_backlog": (
        "the still-open residual_deltas + missing primitives that become the .412 "
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
    "4443_g50t_bank": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_games",
        "reproducible_total_levels",
        "verifier_is_oracle",
    ],
    "4444_config_rule": [
        "honest_verdict",
        "ft09_resolved_generically",
        "dc22_state",
        "dc22_reproduction_result",
        "missing_verifier_gaps",
        "offline_reproduced",
        "reproduced_levels",
        "verifier_is_oracle",
    ],
    "4445_object_motion": [
        "honest_verdict",
        "residuals_closed_generically",
        "world_model_accuracy_cold",
        "world_model_accuracy_with_examples",
        "accuracy_margin",
        "missing_verifier_gaps",
        "offline_reproduced",
        "reproduced_levels",
        "verifier_is_oracle",
    ],
    "4446_first_contact": [
        "honest_verdict",
        "target_game",
        "routed_to",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4447_library": [
        "honest_verdict",
        "library_coverage",
        "retrieval_precision_at_1",
        "primitives_documented",
        "no_regression",
        "verifier_is_oracle",
    ],
    "4448_loo_v2": [
        "honest_verdict",
        "generic_loo_solve_count_v1_baseline",
        "generic_loo_solve_count_v2",
        "loo_gate_passed",
        "missing_verifier_gaps",
        "closed_residuals_by_new_operator",
        "offline_reproduced",
        "verifier_is_oracle",
    ],
    "4449_hygiene": [
        "honest_verdict",
        "reproducible_total_games",
        "reproducible_total_levels",
        "regression_guard_passed",
        "availability_report",
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


def _residual_rows(payload: Mapping[str, Any], source_artifact: str = "") -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    sources = []
    gaps = payload.get("missing_verifier_gaps")
    if isinstance(gaps, list):
        sources.extend(gaps)
    per_game = payload.get("per_game")
    if isinstance(per_game, list):
        sources.extend(per_game)
    for raw in sources:
        if not isinstance(raw, Mapping):
            continue
        residual = str(raw.get("residual_delta") or "")
        if not residual or residual == "none":
            continue
        game = str(raw.get("game") or "")
        key = (game, residual)
        if key in seen:
            continue
        seen.add(key)
        row = {
            "game": game,
            "residual_delta": residual,
            "status": str(raw.get("status") or "open"),
        }
        if source_artifact:
            row["source_artifact"] = source_artifact
        for optional in ("gap_id", "retrieved_operator", "v1_routed_to"):
            if raw.get(optional):
                row[optional] = raw[optional]
        rows.append(row)
    return rows


def g50t_bank_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "g50t_l1_cleanly_banked": False,
            "plus_one_game_level": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "g50t_l1_cleanly_banked": False,
            "plus_one_game_level": False,
        }
    banked = (
        base.str_metric(payload, "target_game") == "g50t"
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    return {
        "state": "g50t_l1_banked" if banked else "not_banked",
        "g50t_l1_cleanly_banked": banked,
        "plus_one_game_level": banked,
        "target_game": base.str_metric(payload, "target_game"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def config_rule_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "ft09_resolved_generically": False,
            "dc22_banked": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "ft09_resolved_generically": False,
            "dc22_banked": False,
            "residual_deltas": [],
        }
    dc22_result = payload.get("dc22_reproduction_result")
    dc22_banked = (
        isinstance(dc22_result, Mapping)
        and dc22_result.get("reproduced") is True
        and isinstance(dc22_result.get("reached_level"), int)
        and not isinstance(dc22_result.get("reached_level"), bool)
        and int(dc22_result.get("reached_level") or 0) >= 1
    )
    ft09_resolved = (
        base.bool_metric(payload, "ft09_resolved_generically") is True
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    if ft09_resolved and dc22_banked:
        state = "ft09_and_dc22_closed"
    elif ft09_resolved:
        state = "ft09_closed_dc22_open"
    else:
        state = "config_rule_open"
    return {
        "state": state,
        "ft09_resolved_generically": ft09_resolved,
        "dc22_banked": dc22_banked,
        "dc22_state": base.str_metric(payload, "dc22_state"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4444_config_rule"].path),
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def object_motion_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "closed_ar25_ka59": False,
            "accuracy_lift": 0.0,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "closed_ar25_ka59": False,
            "accuracy_lift": 0.0,
            "residual_deltas": [],
        }
    closed = set(str(game) for game in base.list_metric(payload, "residuals_closed_generically"))
    cold = base.float_metric(payload, "world_model_accuracy_cold")
    with_examples = base.float_metric(payload, "world_model_accuracy_with_examples")
    lift = round(float(with_examples or 0.0) - float(cold or 0.0), 6)
    helped = cold is not None and with_examples is not None and with_examples > cold
    closed_ar25_ka59 = (
        {"ar25", "ka59"}.issubset(closed)
        and base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 2
    )
    return {
        "state": (
            "ar25_ka59_closed_accuracy_lift"
            if closed_ar25_ka59 and helped
            else "object_motion_partial"
        ),
        "closed_ar25_ka59": closed_ar25_ka59,
        "residuals_closed_generically": sorted(closed),
        "world_model_accuracy_cold": cold,
        "world_model_accuracy_with_examples": with_examples,
        "accuracy_lift": lift,
        "helped_vs_cold_control": helped,
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4445_object_motion"].path),
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def first_contact_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "routed_generic_first_contact_banked": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "routed_generic_first_contact_banked": False,
            "residual_deltas": [],
        }
    banked = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") >= 1
    )
    return {
        "state": "routed_generic_first_contact_banked" if banked else "first_contact_open",
        "routed_generic_first_contact_banked": banked,
        "target_game": base.str_metric(payload, "target_game"),
        "routed_to": base.str_metric(payload, "routed_to"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4446_first_contact"].path),
        ),
        "missing_verifier_gaps": base.list_metric(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def library_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "library_coverage": 0.0,
            "library_generalizes": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "library_coverage": 0.0,
            "library_generalizes": False,
        }
    coverage = base.float_metric(payload, "library_coverage") or 0.0
    precision = base.float_metric(payload, "retrieval_precision_at_1") or 0.0
    documented = base.list_metric(payload, "primitives_documented")
    generalizes = coverage >= 1.0 and precision >= 1.0 and base.bool_metric(payload, "no_regression") is True
    return {
        "state": "documented_library_generalizes" if generalizes else "library_partial",
        "library_coverage": coverage,
        "retrieval_precision_at_1": precision,
        "library_generalizes": generalizes,
        "primitives_documented_count": len(documented),
        "no_regression": base.bool_metric(payload, "no_regression") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def loo_v2_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "generic_loo_solve_count_v2": 0,
            "v2_rose_above_baseline": False,
            "residual_deltas": [],
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "generic_loo_solve_count_v2": 0,
            "v2_rose_above_baseline": False,
            "residual_deltas": [],
        }
    baseline = base.int_metric(payload, "generic_loo_solve_count_v1_baseline")
    count = base.int_metric(payload, "generic_loo_solve_count_v2")
    rose = count > baseline
    return {
        "state": "v2_rises_above_baseline" if rose else "v2_not_above_baseline",
        "generic_loo_solve_count_v1_baseline": baseline,
        "generic_loo_solve_count_v2": count,
        "v2_rose_above_baseline": rose,
        "loo_gate_passed": base.bool_metric(payload, "loo_gate_passed") is True,
        "residual_deltas": _residual_rows(
            payload,
            str(DEFAULT_UPSTREAMS["4448_loo_v2"].path),
        ),
        "closed_residuals_by_new_operator": base.list_metric(
            payload,
            "closed_residuals_by_new_operator",
        ),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def hygiene_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
        }
    reconciliation = payload.get("registry_reconciliation")
    return {
        "state": "reconciled" if base.int_metric(payload, "reproducible_total_levels") else "empty",
        "reproducible_total_levels": base.int_metric(payload, "reproducible_total_levels"),
        "reproducible_total_games": base.int_metric(payload, "reproducible_total_games"),
        "regression_guard_passed": base.bool_metric(payload, "regression_guard_passed") is True,
        "registry_reconciliation": dict(reconciliation) if isinstance(reconciliation, Mapping) else {},
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="g50t_bank",
            required_keys=("4443_g50t_bank",),
            verdict_fn=lambda present: g50t_bank_read(present.get("4443_g50t_bank"), False)[
                "state"
            ],
        ),
        aggregate.AxisSpec(
            name="config_rule",
            required_keys=("4444_config_rule",),
            verdict_fn=lambda present: config_rule_read(present.get("4444_config_rule"), False)[
                "state"
            ],
        ),
        aggregate.AxisSpec(
            name="object_motion",
            required_keys=("4445_object_motion",),
            verdict_fn=lambda present: object_motion_read(
                present.get("4445_object_motion"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="first_contact",
            required_keys=("4446_first_contact",),
            verdict_fn=lambda present: first_contact_read(
                present.get("4446_first_contact"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="library",
            required_keys=("4447_library",),
            verdict_fn=lambda present: library_read(present.get("4447_library"), False)["state"],
        ),
        aggregate.AxisSpec(
            name="loo_v2",
            required_keys=("4448_loo_v2",),
            verdict_fn=lambda present: loo_v2_read(present.get("4448_loo_v2"), False)[
                "generic_loo_solve_count_v2"
            ],
        ),
        aggregate.AxisSpec(
            name="registry_hygiene",
            required_keys=("4449_hygiene",),
            verdict_fn=lambda present: hygiene_read(present.get("4449_hygiene"), False)[
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


def decide_generic_solver_gap_state(
    loo_v2: Mapping[str, Any],
    backlog: Mapping[str, Any],
    g50t: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    object_motion: Mapping[str, Any],
) -> str:
    count = int(loo_v2.get("generic_loo_solve_count_v2") or 0)
    residuals = backlog.get("residual_deltas")
    has_residuals = isinstance(residuals, list) and bool(residuals)
    closing = (
        loo_v2.get("v2_rose_above_baseline") is True
        and count >= 5
        and not has_residuals
        and g50t.get("g50t_l1_cleanly_banked") is True
        and first_contact.get("routed_generic_first_contact_banked") is True
        and object_motion.get("closed_ar25_ka59") is True
    )
    if closing:
        return "closing"
    if (
        count > 0
        or g50t.get("g50t_l1_cleanly_banked") is True
        or first_contact.get("routed_generic_first_contact_banked") is True
        or object_motion.get("closed_ar25_ka59") is True
    ):
        return "partial"
    return "total-gap"


def _missing_primitive_name(residual_delta: str) -> str:
    return residual_delta.removeprefix("missing_")


def build_next_backlog(
    *,
    config_rule: Mapping[str, Any],
    loo_v2: Mapping[str, Any],
    hygiene: Mapping[str, Any],
) -> JsonDict:
    residuals: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    for source in (config_rule, loo_v2):
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

    missing_primitives = sorted(
        {
            _missing_primitive_name(str(row["residual_delta"]))
            for row in residuals
            if isinstance(row.get("residual_delta"), str)
        }
    )
    registry = hygiene.get("registry_reconciliation")
    open_gap_ids = []
    if isinstance(registry, Mapping):
        open_gap_ids = [
            str(gap_id)
            for gap_id in registry.get("open_gap_ids", [])
            if isinstance(gap_id, str)
        ]
    return {
        "residual_deltas": residuals,
        "missing_primitives": missing_primitives,
        "open_gap_ids": open_gap_ids,
        "closed_residuals_by_new_operator": base.list_metric(
            loo_v2,
            "closed_residuals_by_new_operator",
        ),
    }


def _honest_verdict(
    *,
    gap_state: str,
    loo_count: int,
    total_levels: int,
    total_games: int,
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
    return (
        f"complete: v411_generic_solver_{gap_state}_loo_v2_{loo_count}_"
        f"levels_{total_levels}_games_{total_games}_{publication}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    *,
    gap_state: str,
    loo_count: int,
    total_levels: int,
    total_games: int,
    next_backlog: Mapping[str, Any],
) -> str:
    payload = {
        "generic_loo_solve_count_v2": loo_count,
        "generic_solver_gap_state": gap_state,
        "next_backlog": next_backlog,
        "publication_gate": publication_gate,
        "reproducible_total_games": total_games,
        "reproducible_total_levels": total_levels,
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
    g50t: Mapping[str, Any],
    config_rule: Mapping[str, Any],
    object_motion: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    library: Mapping[str, Any],
    loo_v2: Mapping[str, Any],
) -> JsonDict:
    return {
        "exp4443": {
            "state": g50t.get("state"),
            "g50t_l1_cleanly_banked": g50t.get("g50t_l1_cleanly_banked") is True,
            "plus_one_game_level": g50t.get("plus_one_game_level") is True,
        },
        "exp4444": {
            "state": config_rule.get("state"),
            "ft09_resolved_generically": config_rule.get("ft09_resolved_generically") is True,
            "dc22_banked": config_rule.get("dc22_banked") is True,
            "residual_deltas": config_rule.get("residual_deltas", []),
        },
        "exp4445": {
            "state": object_motion.get("state"),
            "closed_ar25_ka59": object_motion.get("closed_ar25_ka59") is True,
            "world_model_accuracy_cold": object_motion.get("world_model_accuracy_cold"),
            "world_model_accuracy_with_examples": object_motion.get(
                "world_model_accuracy_with_examples"
            ),
            "accuracy_lift": object_motion.get("accuracy_lift", 0.0),
        },
        "exp4446": {
            "state": first_contact.get("state"),
            "routed_generic_first_contact_banked": (
                first_contact.get("routed_generic_first_contact_banked") is True
            ),
            "target_game": first_contact.get("target_game", ""),
            "routed_to": first_contact.get("routed_to", ""),
        },
        "exp4447": {
            "state": library.get("state"),
            "library_coverage": library.get("library_coverage", 0.0),
            "library_generalizes": library.get("library_generalizes") is True,
            "retrieval_precision_at_1": library.get("retrieval_precision_at_1", 0.0),
        },
        "exp4448": {
            "state": loo_v2.get("state"),
            "generic_loo_solve_count_v1_baseline": loo_v2.get(
                "generic_loo_solve_count_v1_baseline",
                0,
            ),
            "generic_loo_solve_count_v2": loo_v2.get("generic_loo_solve_count_v2", 0),
            "v2_rose_above_baseline": loo_v2.get("v2_rose_above_baseline") is True,
            "residual_deltas": loo_v2.get("residual_deltas", []),
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

    g50t = g50t_bank_read(clean["4443_g50t_bank"], skipped.get("4443_g50t_bank", False))
    config_rule = config_rule_read(
        clean["4444_config_rule"],
        skipped.get("4444_config_rule", False),
    )
    object_motion = object_motion_read(
        clean["4445_object_motion"],
        skipped.get("4445_object_motion", False),
    )
    first_contact = first_contact_read(
        clean["4446_first_contact"],
        skipped.get("4446_first_contact", False),
    )
    library = library_read(clean["4447_library"], skipped.get("4447_library", False))
    loo_v2 = loo_v2_read(clean["4448_loo_v2"], skipped.get("4448_loo_v2", False))
    hygiene = hygiene_read(clean["4449_hygiene"], skipped.get("4449_hygiene", False))

    next_backlog = build_next_backlog(
        config_rule=config_rule,
        loo_v2=loo_v2,
        hygiene=hygiene,
    )
    gap_state = decide_generic_solver_gap_state(
        loo_v2,
        next_backlog,
        g50t,
        first_contact,
        object_motion,
    )
    total_levels = int(hygiene.get("reproducible_total_levels") or 0)
    total_games = int(hygiene.get("reproducible_total_games") or 0)
    loo_count = int(loo_v2.get("generic_loo_solve_count_v2") or 0)
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
            publication_available=publication_available,
            paper_ready=paper_ready,
        ),
        "generic_solver_gap_state": gap_state,
        "generic_loo_solve_count_v2": loo_count,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "next_backlog": next_backlog,
        "g50t_bank": g50t,
        "config_rule": config_rule,
        "object_motion": object_motion,
        "first_contact": first_contact,
        "library": library,
        "loo_v2": loo_v2,
        "registry_hygiene": hygiene,
        "headline_question_answers": _headline_answers(
            g50t,
            config_rule,
            object_motion,
            first_contact,
            library,
            loo_v2,
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
        "generic_loo_solve_count_v2",
        "reproducible_total_levels",
        "reproducible_total_games",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare int")
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
        loo_count=int(artifact["generic_loo_solve_count_v2"]),
        total_levels=int(artifact["reproducible_total_levels"]),
        total_games=int(artifact["reproducible_total_games"]),
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
