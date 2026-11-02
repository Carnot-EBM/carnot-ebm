"""Build the Exp 4412 v407 verifier localizer scorecard capstone.

Spec refs: REQ-CAPSTONE-4412, SCENARIO-CAPSTONE-4412.
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
OUTPUT_REL_PATH = Path("results/experiment_4412_capstone_v407.json")
EXPERIMENT_ID = 4412
RANDOM_SEED = 4412
SCHEMA = "carnot.capstone_v407_4412.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4412", "SCENARIO-CAPSTONE-4412"]
REGISTRY_REL_PATH = v405.REGISTRY_REL_PATH
PUBLICATION_GATE_REL_PATH = v405.PUBLICATION_GATE_REL_PATH
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34

LOCALIZER_STATES = {
    "localizes_genuine_cross_domain",
    "localizes_fover_bound",
    "position_bound_retired",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4403_localizer": Upstream(
        4403,
        Path("results/experiment_4403_real_intervention_localizer_deconfound.json"),
    ),
    "4404_typed_generalization": Upstream(
        4404,
        Path("results/experiment_4404_localizer_typed_taxonomy_cross_domain.json"),
    ),
    "4405_e3_deeper": Upstream(
        4405,
        Path("results/experiment_4405_e3_deeper_mechanic_unit_tests.json"),
    ),
    "4406_e3_blocked": Upstream(
        4406,
        Path("results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json"),
    ),
    "4407_compounds": Upstream(
        4407,
        Path("results/experiment_4407_active_learning_self_learning_compounds.json"),
    ),
    "4408_calibration": Upstream(
        4408,
        Path("results/experiment_4408_cross_domain_detection_calibration_repair.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "localizer_state",
    "localizer_compounds",
    "detection_calibrated_multi_domain",
    "reproducible_total_levels",
    "verifier_thesis_state",
    "publication_gate",
    "verifier_is_oracle",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .407 scorecard string (localizer genuine "
        "cross-domain? compounds? calibrated multi-domain? the ARC "
        "reproducible-total)."
    ),
    "localizer_state": (
        "One of localizes_genuine_cross_domain / localizes_fover_bound / "
        "position_bound_retired -- the headline decision: did the oracle-distinct "
        "first-error localizer graduate from the .406 position-bias quarantine to "
        "a genuine cross-domain capability, stay FoVer-bound, or retire as "
        "position-bound?"
    ),
    "localizer_compounds": (
        "BARE bool: did the localizer self-improve via ACTIVE selection where "
        "size-only growth saturated (exp4407) -- the mandated continuous-self-"
        "learning reading on the real-headroom axis?"
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: did cross-domain detection become a calibrated multi-domain "
        "contract on de-confounded proper pools (exp4408), or are domains at "
        "chance (logged gaps)?"
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .407 (>= the prior 34) "
        "-- the monotonic north-star accuracy signal (from the authoritative "
        "registry)."
    ),
    "verifier_thesis_state": (
        "One honest string summarizing where the verifier-as-the-value-add stands "
        "after .407 (localizer genuine/bound/retired; compounding; multi-domain "
        "calibration; ARC progress)."
    ),
    "publication_gate": (
        "The G1-G4 booleans + paper_ready + unmet_gates from publication_gate.py "
        "-- the FROZEN FoVer headline (0.9131) gate, carried not re-litigated."
    ),
    "verifier_is_oracle": (
        "BARE bool: the capstone's own aggregation is oracle-distinct=false; it "
        "HONORS each upstream's verifier_is_oracle (ARC E3 solves are "
        "execution-grounded=true=NOT a moat) so it does not trip "
        "CIRCULAR_MOAT_OVERCLAIM."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} -- the audit trail so "
        "the capstone numbers trace to real measurements."
    ),
    "preconditions_checked": (
        "Records the upstream artifacts + registry + publication_gate loaded; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the aggregation ordering.",
    "reproducibility_checksum": (
        "Hash of the aggregated upstream set + the gate computation; lets a third "
        "party re-run the scorecard."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, "
        "the ARC registry, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4403_localizer": [
        "localizer_genuinely_beats_position_only",
        "beats_position_only_baseline",
        "position_only_baseline_f1",
        "localization_f1_by_domain",
        "template_family_holdout_drop",
        "verifier_is_oracle",
    ],
    "4404_typed_generalization": [
        "localizer_generalizes_typed",
        "typed_taxonomy_agreement_above_chance",
        "non_fover_domain_delta_ci95",
        "status",
        "gate_check_summary",
        "gates_evaluated",
        "verifier_is_oracle",
    ],
    "4405_e3_deeper": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "verifier_is_oracle",
    ],
    "4406_e3_blocked": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4407_compounds": [
        "localizer_compounds",
        "active_vs_random_learning_curve",
        "compounding_delta_ci95",
        "gate_summary",
        "positive_control_passed",
        "verifier_is_oracle",
    ],
    "4408_calibration": [
        "detection_calibrated_multi_domain",
        "detection_by_domain",
        "domains_at_chance",
        "unavailable_domains",
        "verifier_is_oracle",
    ],
}

read_registry_progress = v405.read_registry_progress
arc_progress_read = v405.arc_progress_read
arc_e3_summary = v405.arc_e3_summary


def _thesis_states() -> set[str]:
    localizer_parts = {
        "localizes_genuine_cross_domain": "localizer_genuine_cross_domain",
        "localizes_fover_bound": "localizer_fover_bound",
        "position_bound_retired": "localizer_position_bound_retired",
    }
    compound_parts = {
        True: "localizer_compounds",
        False: "localizer_compounding_open",
    }
    calibration_parts = {
        True: "detection_calibrated_multi_domain",
        False: "detection_not_calibrated_multi_domain",
    }
    states = set()
    for localizer in localizer_parts.values():
        for compound in compound_parts.values():
            for calibration in calibration_parts.values():
                for total in (0, PRIOR_REPRODUCIBLE_TOTAL_LEVELS):
                    states.add(f"{localizer}_{compound}_{calibration}_arc_progress_{total}")
    return states


THESIS_STATES = _thesis_states()


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
        provenance_row = {
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
        provenance.append(provenance_row)
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


def real_intervention_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    genuine = (
        base.bool_metric(payload, "localizer_genuinely_beats_position_only") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "genuine" if genuine else "position_only_tied",
        "localizer_genuinely_beats_position_only": genuine,
        "reported_localizer_genuinely_beats_position_only": base.bool_metric(
            payload,
            "localizer_genuinely_beats_position_only",
        ),
        "beats_position_only_baseline": base.bool_metric(
            payload,
            "beats_position_only_baseline",
        ),
        "position_only_baseline_f1": base.float_metric(payload, "position_only_baseline_f1"),
        "localization_f1_by_domain": dict(payload.get("localization_f1_by_domain", {})),
        "template_family_holdout_drop": base.float_metric(
            payload,
            "template_family_holdout_drop",
        ),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def typed_generalization_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    generalizes = (
        base.bool_metric(payload, "localizer_generalizes_typed") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "generalizes" if generalizes else "not_generalized",
        "localizer_generalizes_typed": generalizes,
        "reported_localizer_generalizes_typed": base.bool_metric(
            payload,
            "localizer_generalizes_typed",
        ),
        "typed_taxonomy_agreement_above_chance": base.bool_metric(
            payload,
            "typed_taxonomy_agreement_above_chance",
        ),
        "non_fover_domain_delta_ci95": base.list_metric(
            payload,
            "non_fover_domain_delta_ci95",
        ),
        "artifact_status": base.str_metric(payload, "status"),
        "gate_check_summary": base.str_metric(payload, "gate_check_summary"),
        "gates_evaluated": base.list_metric(payload, "gates_evaluated"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def decide_localizer_state(
    real_intervention: Mapping[str, Any],
    typed_generalization: Mapping[str, Any],
) -> str:
    if (
        real_intervention.get("localizer_genuinely_beats_position_only") is True
        and typed_generalization.get("localizer_generalizes_typed") is True
    ):
        return "localizes_genuine_cross_domain"
    if real_intervention.get("localizer_genuinely_beats_position_only") is True:
        return "localizes_fover_bound"
    return "position_bound_retired"


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    compounds = (
        base.bool_metric(payload, "localizer_compounds") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "compounds" if compounds else "does_not_compound",
        "localizer_compounds": compounds,
        "reported_localizer_compounds": base.bool_metric(payload, "localizer_compounds"),
        "active_vs_random_learning_curve": base.list_metric(
            payload,
            "active_vs_random_learning_curve",
        ),
        "compounding_delta_ci95": base.list_metric(payload, "compounding_delta_ci95"),
        "gate_summary": dict(payload.get("gate_summary", {}))
        if isinstance(payload.get("gate_summary"), Mapping)
        else {},
        "positive_control_passed": base.bool_metric(payload, "positive_control_passed"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def calibration_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    calibrated = (
        base.bool_metric(payload, "detection_calibrated_multi_domain") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "calibrated_multi_domain" if calibrated else "not_calibrated_multi_domain",
        "detection_calibrated_multi_domain": calibrated,
        "reported_detection_calibrated_multi_domain": base.bool_metric(
            payload,
            "detection_calibrated_multi_domain",
        ),
        "detection_by_domain": base.list_metric(payload, "detection_by_domain"),
        "domains_at_chance": base.list_metric(payload, "domains_at_chance"),
        "unavailable_domains": base.list_metric(payload, "unavailable_domains"),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="localizer",
            required_keys=("4403_localizer", "4404_typed_generalization"),
            verdict_fn=lambda present: decide_localizer_state(
                real_intervention_read(present.get("4403_localizer"), False),
                typed_generalization_read(
                    present.get("4404_typed_generalization"),
                    False,
                ),
            ),
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4407_compounds",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4407_compounds"),
                False,
            )["localizer_compounds"],
        ),
        aggregate.AxisSpec(
            name="calibration",
            required_keys=("4408_calibration",),
            verdict_fn=lambda present: calibration_read(
                present.get("4408_calibration"),
                False,
            )["detection_calibrated_multi_domain"],
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4405_e3_deeper", "4406_e3_blocked"),
            verdict_fn=lambda present: arc_e3_summary(
                arc_progress_read(
                    present.get("4405_e3_deeper"),
                    False,
                    "per_target_scorecard",
                ),
                arc_progress_read(
                    present.get("4406_e3_blocked"),
                    False,
                    "per_game_scorecard",
                ),
            )["new_levels_reproduced_from_artifacts"],
        ),
    ]


def verifier_thesis_state(
    localizer_state: str,
    localizer_compounds: bool,
    detection_calibrated_multi_domain: bool,
    reproducible_total_levels: int,
) -> str:
    localizer_parts = {
        "localizes_genuine_cross_domain": "localizer_genuine_cross_domain",
        "localizes_fover_bound": "localizer_fover_bound",
        "position_bound_retired": "localizer_position_bound_retired",
    }
    compound = "localizer_compounds" if localizer_compounds else "localizer_compounding_open"
    calibration = (
        "detection_calibrated_multi_domain"
        if detection_calibrated_multi_domain
        else "detection_not_calibrated_multi_domain"
    )
    localizer = localizer_parts.get(localizer_state, "localizer_position_bound_retired")
    return f"{localizer}_{compound}_{calibration}_arc_progress_{reproducible_total_levels}"


def _honest_verdict(
    localizer_state: str,
    localizer_compounds: bool,
    detection_calibrated_multi_domain: bool,
    total_levels: int,
    publication_gate_available: bool,
    paper_ready: bool,
) -> str:
    if publication_gate_available:
        paper = "publication_ready" if paper_ready else "publication_not_ready"
    else:
        paper = "publication_gate_gap"
    compounds = "true" if localizer_compounds else "false"
    calibrated = "true" if detection_calibrated_multi_domain else "false"
    return (
        f"complete: v407_localizer_{localizer_state}_compounds_{compounds}_"
        f"calibrated_{calibrated}_arc_levels_{total_levels}_{paper}"
    )


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
        [
            {
                "axis": "publication_gate",
                "artifact_key": "publication_gate",
                "reason": "unrunnable",
            }
        ],
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
) -> str:
    payload = {
        "publication_gate": publication_gate,
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
    cited: list[JsonDict] = []
    for row in provenance:
        if row.get("skipped") is True:
            continue
        fields = row.get("fields_imported")
        if not isinstance(fields, list) or not fields:
            continue
        cited.append(
            {
                "artifact_key": row["artifact_key"],
                "experiment_id": row["experiment_id"],
                "path": row["path"],
                "sha256": row["sha256"],
                "fields_imported": fields,
            }
        )
    return cited


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
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
        "arc_registry": dict(registry),
    }


def _oracle_declarations(provenance: list[JsonDict], clean: Mapping[str, JsonDict | None]) -> list[JsonDict]:
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

    real_intervention = real_intervention_read(
        clean["4403_localizer"],
        skipped.get("4403_localizer", False),
    )
    typed_generalization = typed_generalization_read(
        clean["4404_typed_generalization"],
        skipped.get("4404_typed_generalization", False),
    )
    localizer_state = decide_localizer_state(real_intervention, typed_generalization)
    self_learning = self_learning_read(
        clean["4407_compounds"],
        skipped.get("4407_compounds", False),
    )
    calibration = calibration_read(
        clean["4408_calibration"],
        skipped.get("4408_calibration", False),
    )
    deeper = arc_progress_read(
        clean["4405_e3_deeper"],
        skipped.get("4405_e3_deeper", False),
        "per_target_scorecard",
    )
    blocked = arc_progress_read(
        clean["4406_e3_blocked"],
        skipped.get("4406_e3_blocked", False),
        "per_game_scorecard",
    )
    arc_e3 = arc_e3_summary(deeper, blocked)
    registry = read_registry_progress(root)
    compounds = self_learning.get("localizer_compounds") is True
    calibrated = calibration.get("detection_calibrated_multi_domain") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    paper_ready = bool(publication_gate.get("paper_ready"))
    publication_available = bool(publication_gate_check.get("runnable"))
    thesis = verifier_thesis_state(localizer_state, compounds, calibrated, total_levels)
    end = time.time() if now_s is None else now_s
    per_axis_gaps = list(availability_report.get("missing_upstream_artifacts", []))
    per_axis_gaps.extend(publication_gate_gaps)

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            localizer_state,
            compounds,
            calibrated,
            total_levels,
            publication_available,
            paper_ready,
        ),
        "localizer_state": localizer_state,
        "localizer": {
            "status": localizer_state,
            "real_intervention": real_intervention,
            "typed_generalization": typed_generalization,
        },
        "localizer_compounds": compounds,
        "self_learning": self_learning,
        "detection_calibrated_multi_domain": calibrated,
        "calibration": calibration,
        "reproducible_total_levels": total_levels,
        "arc_reproducible_progress": registry,
        "arc_e3_outcomes": arc_e3,
        "verifier_thesis_state": thesis,
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
            registry,
        ),
        "per_axis_gaps": per_axis_gaps,
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "publication_gate_checksum": hashlib.sha256(
            json.dumps(publication_gate, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "reproducibility_checksum": checksum_from_inputs(provenance, publication_gate),
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:  # pragma: no cover
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must be terminal-prefixed")  # pragma: no cover
    if artifact.get("localizer_state") not in LOCALIZER_STATES:
        raise ValueError("localizer_state is not recognized")  # pragma: no cover
    if not isinstance(artifact.get("localizer_compounds"), bool):
        raise ValueError("localizer_compounds must be a bare bool")  # pragma: no cover
    if not isinstance(artifact.get("detection_calibrated_multi_domain"), bool):
        raise ValueError("detection_calibrated_multi_domain must be a bare bool")  # pragma: no cover
    total = artifact.get("reproducible_total_levels")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("reproducible_total_levels must be a bare int")  # pragma: no cover
    if not isinstance(artifact.get("verifier_thesis_state"), str):
        raise ValueError("verifier_thesis_state must be a string")  # pragma: no cover
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")  # pragma: no cover
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")  # pragma: no cover
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")  # pragma: no cover
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")  # pragma: no cover
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")  # pragma: no cover
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
    expected = checksum_from_inputs(provenance, artifact["publication_gate"])
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
