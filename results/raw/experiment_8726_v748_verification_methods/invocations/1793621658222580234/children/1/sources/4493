"""Build the Exp 4401 v406 verifier localizer scorecard capstone.

Spec refs: REQ-CAPSTONE-4401, SCENARIO-CAPSTONE-4401.
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
OUTPUT_REL_PATH = Path("results/experiment_4401_capstone_v406.json")
EXPERIMENT_ID = 4401
RANDOM_SEED = 4401
SCHEMA = "carnot.capstone_v406_4401.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4401", "SCENARIO-CAPSTONE-4401"]
REGISTRY_REL_PATH = v405.REGISTRY_REL_PATH
PUBLICATION_GATE_REL_PATH = v405.PUBLICATION_GATE_REL_PATH
BLOCKED_PUBLICATION_GATE_CHECKSUM = hashlib.sha256(
    b"blocked_publication_gate_unrunnable_v406"
).hexdigest()
EMPTY_UPSTREAM_CHECKSUM = hashlib.sha256(b"no_v406_upstream_artifacts").hexdigest()

LOCALIZER_STATES = {
    "localizes_cross_domain_genuine",
    "localizes_but_not_genuine",
    "detects_but_not_localizes",
}


def _thesis_states() -> set[str]:
    localizer_parts = {
        "localizes_cross_domain_genuine": "localizer_genuine",
        "localizes_but_not_genuine": "localizer_not_genuine",
        "detects_but_not_localizes": "detects_but_not_localizes",
    }
    compound_parts = {
        True: "localizer_compounds",
        False: "localizer_compounding_open",
    }
    calibration_parts = {
        True: "detection_calibrated_multi_domain",
        False: "detection_not_calibrated_multi_domain",
    }
    states = {"blocked_publication_gate_unrunnable"}
    for localizer in localizer_parts.values():
        for compound in compound_parts.values():
            for calibration in calibration_parts.values():
                states.add(f"{localizer}_{compound}_{calibration}")
    return states


THESIS_STATES = _thesis_states()


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4392_localizer": Upstream(
        4392,
        Path("results/experiment_4392_verifiable_process_data_localizer.json"),
    ),
    "4393_skeptic": Upstream(
        4393,
        Path("results/experiment_4393_localizer_skeptic_proof.json"),
    ),
    "4394_e3_deeper": Upstream(
        4394,
        Path("results/experiment_4394_e3_deeper_fidelity_gate.json"),
    ),
    "4395_e3_blocked": Upstream(
        4395,
        Path("results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"),
    ),
    "4396_compounds": Upstream(
        4396,
        Path("results/experiment_4396_localizer_self_learning_compounds.json"),
    ),
    "4397_calibration": Upstream(
        4397,
        Path("results/experiment_4397_cross_domain_detection_calibration.json"),
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
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .406 scorecard string (whether the detector "
        "became a genuine cross-domain localizer, whether it compounds, whether "
        "detection is calibrated multi-domain, the ARC reproducible-total)."
    ),
    "localizer_state": (
        "One of localizes_cross_domain_genuine / localizes_but_not_genuine / "
        "detects_but_not_localizes -- the headline decision: did the verifier "
        "graduate from 'detects but cannot localize' (.405 F1 0.096) to an "
        "actionable cross-domain first-error localizer, genuinely (not "
        "template-leak/position/overfit)?"
    ),
    "localizer_compounds": (
        "BARE bool: did the localizer self-improve as first-error labels "
        "accumulate (exp4396) -- the mandated continuous-self-learning reading "
        "on the live vehicle where headroom is real?"
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: did cross-domain detection become a calibrated multi-domain "
        "contract (exp4397) -- the verifier-domain-expansion reading -- or are "
        "domains at chance (logged gaps)?"
    ),
    "reproducible_total_levels": (
        "BARE int: the ARC reproducible-level count after .406 (>= the prior 34) "
        "-- the monotonic north-star accuracy signal."
    ),
    "verifier_thesis_state": (
        "One honest string summarizing where the verifier-moat thesis stands "
        "after .406 (localizer-genuine / localizer-not-genuine / "
        "detects-but-not-localizes / localizer-compounds / "
        "detection-calibrated-multi-domain / etc.)."
    ),
    "publication_gate": (
        "G1-G4 via publication_gate.py (paper_ready + unmet_gates) -- the stable "
        "finish line (north-star \u00a72)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the oracle-distinct localizer/detection reads (the "
        "durable exp4355 stamp fix) -- so this capstone does NOT trip "
        "CIRCULAR_MOAT_OVERCLAIM."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported} -- the audit trail so the "
        "capstone numbers trace to real measurements."
    ),
    "preconditions_checked": (
        "Records the upstream-artifact + publication_gate availability; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, "
        "the ARC registry, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4392_localizer": [
        "localizer_beats_ensemble_baseline",
        "localization_f1_by_domain",
        "verifier_is_oracle",
    ],
    "4393_skeptic": [
        "localizer_win_is_genuine",
        "held_out_real_localization_delta_ci95",
        "gate_check_summary",
        "gates_evaluated",
        "verifier_is_oracle",
    ],
    "4394_e3_deeper": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_target_scorecard",
        "verifier_is_oracle",
    ],
    "4395_e3_blocked": [
        "new_levels_reproduced",
        "reproducible_total_levels",
        "per_game_scorecard",
        "verifier_is_oracle",
    ],
    "4396_compounds": [
        "localizer_compounds",
        "learning_curve",
        "compounding_delta_ci95",
        "verifier_is_oracle",
    ],
    "4397_calibration": [
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
            path, root, summarize_runner
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


def localizer_measurement_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    beats = (
        base.bool_metric(payload, "localizer_beats_ensemble_baseline") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "beats_baseline" if beats else "clean_null",
        "localizer_beats_ensemble_baseline": beats,
        "reported_localizer_beats_ensemble_baseline": base.bool_metric(
            payload,
            "localizer_beats_ensemble_baseline",
        ),
        "localization_f1_by_domain": dict(payload.get("localization_f1_by_domain", {})),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def skeptic_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    gates = base.list_metric(payload, "gates_evaluated")
    all_gates_passed = bool(gates) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gates
    )
    reported_genuine = base.bool_metric(payload, "localizer_win_is_genuine") is True
    genuine = (
        (reported_genuine or all_gates_passed)
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "genuine" if genuine else "not_genuine",
        "localizer_win_is_genuine": genuine,
        "reported_localizer_win_is_genuine": base.bool_metric(
            payload,
            "localizer_win_is_genuine",
        ),
        "held_out_real_localization_delta_ci95": base.list_metric(
            payload,
            "held_out_real_localization_delta_ci95",
        ),
        "gate_check_summary": base.str_metric(payload, "gate_check_summary"),
        "gates_evaluated": gates,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def decide_localizer_state(
    measurement: Mapping[str, Any],
    skeptic: Mapping[str, Any],
) -> str:
    if (
        measurement.get("localizer_beats_ensemble_baseline") is True
        and skeptic.get("localizer_win_is_genuine") is True
    ):
        return "localizes_cross_domain_genuine"
    if measurement.get("localizer_beats_ensemble_baseline") is True:
        return "localizes_but_not_genuine"
    return "detects_but_not_localizes"


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
        "learning_curve": base.list_metric(payload, "learning_curve"),
        "compounding_delta_ci95": base.list_metric(payload, "compounding_delta_ci95"),
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
            required_keys=("4392_localizer", "4393_skeptic"),
            verdict_fn=lambda present: decide_localizer_state(
                localizer_measurement_read(present.get("4392_localizer"), False),
                skeptic_read(present.get("4393_skeptic"), False),
            ),
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4396_compounds",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4396_compounds"),
                False,
            )["localizer_compounds"],
        ),
        aggregate.AxisSpec(
            name="calibration",
            required_keys=("4397_calibration",),
            verdict_fn=lambda present: calibration_read(
                present.get("4397_calibration"),
                False,
            )["detection_calibrated_multi_domain"],
        ),
        aggregate.AxisSpec(
            name="arc",
            required_keys=("4394_e3_deeper", "4395_e3_blocked"),
            verdict_fn=lambda present: arc_e3_summary(
                arc_progress_read(
                    present.get("4394_e3_deeper"),
                    False,
                    "per_target_scorecard",
                ),
                arc_progress_read(
                    present.get("4395_e3_blocked"),
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
) -> str:
    localizer_parts = {
        "localizes_cross_domain_genuine": "localizer_genuine",
        "localizes_but_not_genuine": "localizer_not_genuine",
        "detects_but_not_localizes": "detects_but_not_localizes",
    }
    compound = "localizer_compounds" if localizer_compounds else "localizer_compounding_open"
    calibration = (
        "detection_calibrated_multi_domain"
        if detection_calibrated_multi_domain
        else "detection_not_calibrated_multi_domain"
    )
    return f"{localizer_parts.get(localizer_state, 'detects_but_not_localizes')}_{compound}_{calibration}"


def _honest_verdict(
    localizer_state: str,
    localizer_compounds: bool,
    detection_calibrated_multi_domain: bool,
    total_levels: int,
    paper_ready: bool,
) -> str:
    paper = "publication_ready" if paper_ready else "publication_not_ready"
    compounds = "true" if localizer_compounds else "false"
    calibrated = "true" if detection_calibrated_multi_domain else "false"
    return (
        f"complete: v406_localizer_{localizer_state}_compounds_{compounds}_"
        f"calibrated_{calibrated}_arc_levels_{total_levels}_{paper}"
    )


def checksum_from_provenance(provenance: list[Mapping[str, Any]]) -> str:
    if not provenance:
        return EMPTY_UPSTREAM_CHECKSUM
    shas = sorted(str(row["sha256"]) for row in provenance)
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


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


def _blocked_publication_gate_artifact(
    started_s: float,
    now_s: float | None,
    publication_gate_check: Mapping[str, Any],
) -> JsonDict:
    end = time.time() if now_s is None else now_s
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - started_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked_publication_gate_unrunnable",
        "localizer_state": "detects_but_not_localizes",
        "localizer": {"status": "blocked_publication_gate_unrunnable"},
        "localizer_compounds": False,
        "self_learning": {"status": "blocked_publication_gate_unrunnable"},
        "detection_calibrated_multi_domain": False,
        "calibration": {"status": "blocked_publication_gate_unrunnable"},
        "reproducible_total_levels": 0,
        "arc_reproducible_progress": {"status": "not_checked", "path": str(REGISTRY_REL_PATH)},
        "arc_e3_outcomes": {"status": "not_checked"},
        "verifier_thesis_state": "blocked_publication_gate_unrunnable",
        "publication_gate": {
            "paper_ready": False,
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(publication_gate_check.get("error", "unrunnable")),
        },
        "paper_ready": False,
        "unmet_gates": ["publication_gate_unrunnable"],
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "cited_upstream_artifacts": [],
        "preconditions_checked": {
            "publication_gate": dict(publication_gate_check),
            "upstream_artifacts": [],
            "arc_registry": {"status": "not_checked", "path": str(REGISTRY_REL_PATH)},
        },
        "per_axis_gaps": [],
        "flagged_artifacts_excluded": [],
        "availability_report": {},
        "upstream_provenance": [],
        "upstream_sha256_set": [],
        "reproducibility_checksum": BLOCKED_PUBLICATION_GATE_CHECKSUM,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("blocked precondition"),
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
    publication_gate, publication_gate_check = v405._publication_gate_check(  # noqa: SLF001
        root,
        publication_gate_runner,
    )
    if publication_gate is None:
        return _blocked_publication_gate_artifact(start, now_s, publication_gate_check)

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

    measurement = localizer_measurement_read(
        clean["4392_localizer"],
        skipped.get("4392_localizer", False),
    )
    skeptic = skeptic_read(clean["4393_skeptic"], skipped.get("4393_skeptic", False))
    localizer_state = decide_localizer_state(measurement, skeptic)
    localizer = {
        "status": localizer_state,
        "measurement": measurement,
        "skeptic_validation": skeptic,
    }
    self_learning = self_learning_read(
        clean["4396_compounds"],
        skipped.get("4396_compounds", False),
    )
    calibration = calibration_read(
        clean["4397_calibration"],
        skipped.get("4397_calibration", False),
    )
    deeper = arc_progress_read(
        clean["4394_e3_deeper"],
        skipped.get("4394_e3_deeper", False),
        "per_target_scorecard",
    )
    blocked = arc_progress_read(
        clean["4395_e3_blocked"],
        skipped.get("4395_e3_blocked", False),
        "per_game_scorecard",
    )
    arc_e3 = arc_e3_summary(deeper, blocked)
    registry = read_registry_progress(root)
    compounds = self_learning.get("localizer_compounds") is True
    calibrated = calibration.get("detection_calibrated_multi_domain") is True
    total_levels = int(registry.get("reproducible_total_levels") or 0)
    paper_ready = bool(publication_gate.get("paper_ready"))
    thesis = verifier_thesis_state(localizer_state, compounds, calibrated)
    end = time.time() if now_s is None else now_s

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
            paper_ready,
        ),
        "localizer_state": localizer_state,
        "localizer": localizer,
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
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "preconditions_checked": _preconditions_checked(
            root,
            publication_gate_check,
            provenance,
            registry,
        ),
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "flagged_artifacts_excluded": exclusions,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "reproducibility_checksum": checksum_from_provenance(provenance),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance("aggregation logic"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if verdict != "blocked_publication_gate_unrunnable":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("localizer_state") not in LOCALIZER_STATES:
        raise ValueError("localizer_state is not recognized")
    if not isinstance(artifact.get("localizer_compounds"), bool):
        raise ValueError("localizer_compounds must be a bare bool")
    if not isinstance(artifact.get("detection_calibrated_multi_domain"), bool):
        raise ValueError("detection_calibrated_multi_domain must be a bare bool")
    if not isinstance(artifact.get("reproducible_total_levels"), int) or isinstance(
        artifact.get("reproducible_total_levels"),
        bool,
    ):
        raise ValueError("reproducible_total_levels must be a bare int")
    if artifact.get("verifier_thesis_state") not in THESIS_STATES:
        raise ValueError("verifier_thesis_state is not recognized")
    if not isinstance(artifact.get("publication_gate"), Mapping):
        raise ValueError("publication_gate must be an object")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be an object")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not base.is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required principles")
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
    expected = (
        BLOCKED_PUBLICATION_GATE_CHECKSUM
        if artifact.get("honest_verdict") == "blocked_publication_gate_unrunnable"
        else checksum_from_provenance(provenance)
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match upstream sha256 set")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    publication_gate_runner: PublicationGateRunner = base.run_publication_gate,
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
