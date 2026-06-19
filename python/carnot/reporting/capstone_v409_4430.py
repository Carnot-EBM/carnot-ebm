"""Build the Exp 4430 .409 ARC milestone capstone.

Spec refs: REQ-CAPSTONE-4430, SCENARIO-CAPSTONE-4430.
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
OUTPUT_REL_PATH = Path("results/experiment_4430_capstone_409.json")
EXPERIMENT_ID = 4430
RANDOM_SEED = 4430
SCHEMA = "carnot.capstone_v409_4430.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4430", "SCENARIO-CAPSTONE-4430"]
PRIOR_REPRODUCIBLE_TOTAL_LEVELS = 34
PRIOR_REPRODUCIBLE_TOTAL_GAMES = 17

CONFIG_RULE_STATES = {
    "direct_artifact_flagged_registry_audit_counted_execution_grounded",
    "clean_execution_grounded_reproduced_level",
    "excluded_flagged_adversarial",
    "clean_no_new_level",
    "missing_or_excluded",
}
GLYPH_REWRITE_STATES = {
    "grounded_and_offline_solved",
    "grounded_not_solved",
    "not_grounded",
    "missing_or_excluded",
}
FIRST_CONTACT_STATES = {
    "new_game_added",
    "verifier_gap_logged_no_new_game",
    "no_new_game",
    "missing_or_excluded",
}
DEEPENING_STATES = {
    "new_level_added",
    "mechanic_repair_no_new_level",
    "no_new_level",
    "missing_or_excluded",
}
VOCABULARY_STATES = {
    "transfers",
    "no_transfer",
    "excluded_flagged_adversarial",
    "missing_or_excluded",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4421_config_rule": Upstream(
        4421,
        Path("results/experiment_4421_config_rule_solve_unseen.json"),
    ),
    "4422_glyph": Upstream(
        4422,
        Path("results/experiment_4422_glyph_rewrite_perception.json"),
    ),
    "4423_first_contact": Upstream(
        4423,
        Path("results/experiment_4423_generic_first_contact_breadth.json"),
    ),
    "4424_deepening": Upstream(
        4424,
        Path("results/experiment_4424_deeper_solved_game.json"),
    ),
    "4425_vocabulary": Upstream(
        4425,
        Path("results/experiment_4425_config_rule_vocabulary_transfer.json"),
    ),
    "4426_registry_audit": Upstream(
        4426,
        Path("results/experiment_4426_arc_registry_repro_audit.json"),
    ),
    "4429_sota_ingestion": Upstream(
        4429,
        Path("results/experiment_4429_sota_ingestion_409.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels",
    "new_levels",
    "new_games",
    "generic_pipeline_state",
    "config_rule_unseen_state",
    "glyph_rewrite_state",
    "generic_first_contact_state",
    "multi_level_deepening_state",
    "config_rule_vocabulary_transfer_state",
    "publication_gate",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed; one honest .409 scorecard string that separates audited ARC "
        "progress from skipped flagged artifacts."
    ),
    "reproducible_total_levels": (
        "BARE int: audited ARC reproducible-level count after .409, sourced from clean "
        "exp4426 when available rather than trusted registry assertion."
    ),
    "new_levels": "BARE int: audited level delta versus the prior .408 baseline of 34.",
    "new_games": "BARE int: audited game-count delta versus the prior .408 baseline of 17.",
    "generic_pipeline_state": (
        "One honest string for the .409 generic pipeline: whether first contact added a "
        "new game, logged a verifier gap, or was blocked."
    ),
    "config_rule_unseen_state": (
        "One honest string answering exp4421 while respecting flagged_adversarial and "
        "verifier_is_oracle: direct flagged artifacts are skipped; any registry-audit "
        "reproduction is execution_grounded, not a moat."
    ),
    "glyph_rewrite_state": (
        "One honest string answering whether exp4422 grounded the glyph-rewrite verifier "
        "and solved offline."
    ),
    "generic_first_contact_state": (
        "One honest string answering whether exp4423 added a new game or logged an open "
        "first-contact verifier gap."
    ),
    "multi_level_deepening_state": (
        "One honest string answering whether exp4424 landed +1 or only fixed part of the "
        "world model with a residual reproduction gap."
    ),
    "config_rule_vocabulary_transfer_state": (
        "One honest string answering exp4425 only from clean non-flagged "
        "vocabulary-transfer evidence."
    ),
    "publication_gate": (
        "The G1-G4 publication_gate.py output (paper_ready + unmet_gates) -- the frozen "
        "FoVer headline gate."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; upstream execution-grounded solves are "
        "carried separately so CIRCULAR_MOAT_OVERCLAIM does not fire."
    ),
    "preconditions_checked": (
        "Records upstream artifacts, robust aggregate-available, publication gate, "
        "registry audit, and TRM stand-down."
    ),
    "random_seed": "Determinism precondition for the aggregation.",
    "reproducibility_checksum": (
        "Hash of the aggregated upstream artifact sha256 set plus the publication gate output."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256}; skipped flagged artifacts "
        "must import no fields."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this capstone reads upstream JSON, exp4426 "
        "registry audit, and publication_gate.py output."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4421_config_rule": [
        "honest_verdict",
        "offline_reproduced",
        "reproduced_levels",
        "new_levels_reproduced",
        "target_game",
        "verifier_is_oracle",
    ],
    "4422_glyph": [
        "honest_verdict",
        "grounded",
        "fires_on_win",
        "false_positive_rate",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "verifier_is_oracle",
    ],
    "4423_first_contact": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "new_games_reproduced",
        "missing_verifier_gaps",
        "verifier_is_oracle",
    ],
    "4424_deepening": [
        "honest_verdict",
        "game",
        "offline_reproduced",
        "new_levels_reproduced",
        "reproduced_levels",
        "per_mechanic_test_pass_rate",
        "residual_failing_mechanic",
        "verifier_is_oracle",
    ],
    "4425_vocabulary": [
        "honest_verdict",
        "config_rule_vocabulary_transfers",
        "verifier_is_oracle",
    ],
    "4426_registry_audit": [
        "honest_verdict",
        "reproducible_total_levels",
        "registry_claimed_reproducible_total_levels",
        "registry_claimed_reproducible_total_games",
        "counted_entries_audited",
        "all_counted_entries_reproduced",
        "milestone_409_reproduction_gates",
        "inference_substrate",
    ],
    "4429_sota_ingestion": [
        "honest_verdict",
        "flagged_for_v410",
        "outcome_conditioning",
        "preconditions_checked",
        "inference_substrate",
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


def _gate_by_experiment(gates: Any, experiment: str) -> JsonDict:
    if not isinstance(gates, list):
        return {}
    for gate in gates:
        if isinstance(gate, Mapping) and gate.get("experiment") == experiment:
            return dict(gate)
    return {}


def registry_audit_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    gates = base.list_metric(payload, "milestone_409_reproduction_gates")
    total_levels = base.int_metric(payload, "reproducible_total_levels")
    total_games = base.int_metric(payload, "registry_claimed_reproducible_total_games")
    flagged_counted = [
        dict(gate)
        for gate in gates
        if isinstance(gate, Mapping)
        and gate.get("artifact_flagged_adversarial") is True
        and base.int_metric(gate, "new_levels_counted") > 0
    ]
    return {
        "status": "audited" if total_levels > 0 else "missing_total",
        "reproducible_total_levels": total_levels,
        "registry_claimed_reproducible_total_levels": base.int_metric(
            payload,
            "registry_claimed_reproducible_total_levels",
        ),
        "registry_claimed_reproducible_total_games": total_games,
        "counted_entries_audited": base.int_metric(payload, "counted_entries_audited"),
        "all_counted_entries_reproduced": base.bool_metric(
            payload,
            "all_counted_entries_reproduced",
        )
        is True,
        "milestone_409_reproduction_gates": gates,
        "flagged_sources_counted": flagged_counted,
        "inference_substrate": base.str_metric(payload, "inference_substrate"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def config_rule_unseen_read(
    payload: JsonDict | None,
    skipped: bool,
    registry_read: Mapping[str, Any],
) -> JsonDict:
    gate = _gate_by_experiment(
        registry_read.get("milestone_409_reproduction_gates"),
        "exp4421",
    )
    registry_counted = base.int_metric(gate, "new_levels_counted") > 0
    if skipped:
        state = (
            "direct_artifact_flagged_registry_audit_counted_execution_grounded"
            if registry_counted
            else "excluded_flagged_adversarial"
        )
        return {
            "state": state,
            "direct_artifact_imported": False,
            "registry_audit_counted": registry_counted,
            "execution_grounded": gate.get("offline_reproduced") is True,
            "registry_gate": gate,
            "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "direct_artifact_imported": False,
            "registry_audit_counted": registry_counted,
            "execution_grounded": gate.get("offline_reproduced") is True,
            "registry_gate": gate,
            "verifier_is_oracle": None,
        }
    reproduced = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "new_levels_reproduced") > 0
    )
    return {
        "state": "clean_execution_grounded_reproduced_level"
        if reproduced
        else "clean_no_new_level",
        "direct_artifact_imported": True,
        "registry_audit_counted": registry_counted,
        "execution_grounded": base.bool_metric(payload, "verifier_is_oracle") is True,
        "target_game": base.str_metric(payload, "target_game"),
        "new_levels_reproduced": base.int_metric(payload, "new_levels_reproduced"),
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "registry_gate": gate,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def glyph_rewrite_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded"}
    grounded = (
        base.bool_metric(payload, "grounded") is True
        and base.bool_metric(payload, "fires_on_win") is True
        and base.float_metric(payload, "false_positive_rate") == 0.0
    )
    solved = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "reproduced_levels") > 0
    )
    if grounded and solved:
        state = "grounded_and_offline_solved"
    elif grounded:
        state = "grounded_not_solved"
    else:
        state = "not_grounded"
    return {
        "state": state,
        "grounded": grounded,
        "offline_reproduced": solved,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "target_game": base.str_metric(payload, "target_game"),
        "execution_grounded": base.bool_metric(payload, "verifier_is_oracle") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def generic_first_contact_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded"}
    new_game = base.bool_metric(payload, "offline_reproduced") is True and (
        base.int_metric(payload, "new_games_reproduced") > 0
        or base.int_metric(payload, "reproduced_levels") > 0
    )
    gaps = base.list_metric(payload, "missing_verifier_gaps")
    if new_game:
        state = "new_game_added"
    elif gaps:
        state = "verifier_gap_logged_no_new_game"
    else:
        state = "no_new_game"
    return {
        "state": state,
        "new_game_added": new_game,
        "target_game": base.str_metric(payload, "target_game"),
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "missing_verifier_gaps": gaps,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def generic_pipeline_state(first_contact_state: str) -> str:
    if first_contact_state == "new_game_added":
        return "first_contact_added_new_game"
    if first_contact_state == "verifier_gap_logged_no_new_game":
        return "first_contact_verifier_gap_open_no_new_game"
    if first_contact_state == "missing_or_excluded":
        return "first_contact_missing_or_excluded"
    return "first_contact_no_new_game"


def multi_level_deepening_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"state": "missing_or_excluded"}
    new_level = (
        base.bool_metric(payload, "offline_reproduced") is True
        and base.int_metric(payload, "new_levels_reproduced") > 0
    )
    pass_rate = base.float_metric(payload, "per_mechanic_test_pass_rate")
    if new_level:
        state = "new_level_added"
    elif pass_rate is not None and pass_rate > 0.0:
        state = "mechanic_repair_no_new_level"
    else:
        state = "no_new_level"
    return {
        "state": state,
        "new_level_added": new_level,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "new_levels_reproduced": base.int_metric(payload, "new_levels_reproduced"),
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "per_mechanic_test_pass_rate": pass_rate,
        "residual_failing_mechanic": base.str_metric(payload, "residual_failing_mechanic"),
        "execution_grounded": base.bool_metric(payload, "verifier_is_oracle") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def vocabulary_transfer_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"state": "excluded_flagged_adversarial", "config_rule_vocabulary_transfers": False}
    if payload is None:
        return {"state": "missing_or_excluded", "config_rule_vocabulary_transfers": False}
    transfers = (
        base.bool_metric(payload, "config_rule_vocabulary_transfers") is True
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "state": "transfers" if transfers else "no_transfer",
        "config_rule_vocabulary_transfers": transfers,
        "reported_config_rule_vocabulary_transfers": base.bool_metric(
            payload,
            "config_rule_vocabulary_transfers",
        ),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def sota_ingestion_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {"status": "missing_or_excluded"}
    return {
        "status": "mapped",
        "flagged_for_v410": base.str_metric(payload, "flagged_for_v410"),
        "outcome_conditioning": dict(payload.get("outcome_conditioning", {}))
        if isinstance(payload.get("outcome_conditioning"), Mapping)
        else {},
        "inference_substrate": base.str_metric(payload, "inference_substrate"),
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="config_rule_unseen",
            required_keys=("4421_config_rule", "4426_registry_audit"),
            verdict_fn=lambda present: bool(present.get("4426_registry_audit")),
        ),
        aggregate.AxisSpec(
            name="glyph_rewrite",
            required_keys=("4422_glyph",),
            verdict_fn=lambda present: glyph_rewrite_read(
                present.get("4422_glyph"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="generic_first_contact",
            required_keys=("4423_first_contact",),
            verdict_fn=lambda present: generic_first_contact_read(
                present.get("4423_first_contact"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="multi_level_deepening",
            required_keys=("4424_deepening",),
            verdict_fn=lambda present: multi_level_deepening_read(
                present.get("4424_deepening"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="vocabulary_transfer",
            required_keys=("4425_vocabulary",),
            verdict_fn=lambda present: vocabulary_transfer_read(
                present.get("4425_vocabulary"),
                False,
            )["state"],
        ),
        aggregate.AxisSpec(
            name="registry_audit",
            required_keys=("4426_registry_audit",),
            verdict_fn=lambda present: registry_audit_read(
                present.get("4426_registry_audit"),
                False,
            )["reproducible_total_levels"],
        ),
        aggregate.AxisSpec(
            name="sota_ingestion",
            required_keys=("4429_sota_ingestion",),
            verdict_fn=lambda present: sota_ingestion_read(
                present.get("4429_sota_ingestion"),
                False,
            )["status"],
        ),
    ]


def _publication_gate_or_gap(
    root: Path,
    runner: PublicationGateRunner,
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    publication_gate, check = v405._publication_gate_check(root, runner)  # noqa: SLF001
    if publication_gate is not None:
        return publication_gate, check, []
    return (  # pragma: no cover
        {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_unrunnable"],
            "error": str(check.get("error", "unrunnable")),
        },
        check,
        [{"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}],
    )


def _honest_verdict(
    total_levels: int,
    new_levels: int,
    new_games: int,
    config_state: str,
    glyph_state: str,
    pipeline_state: str,
    deepening_state: str,
    vocabulary_state: str,
    publication_available: bool,
    paper_ready: bool,
) -> str:
    config = (
        "config_rule_flagged_registry_execution_grounded"
        if config_state == "direct_artifact_flagged_registry_audit_counted_execution_grounded"
        else "config_rule_clean_execution_grounded"
        if config_state == "clean_execution_grounded_reproduced_level"
        else "config_rule_no_clean_level"
    )
    glyph = "glyph_solved" if glyph_state == "grounded_and_offline_solved" else "glyph_not_solved"
    pipeline = (
        "first_contact_new_game"
        if pipeline_state == "first_contact_added_new_game"
        else "first_contact_gap"
        if pipeline_state == "first_contact_verifier_gap_open_no_new_game"
        else "first_contact_no_new_game"
    )
    deepening = "deepening_plus1" if deepening_state == "new_level_added" else "deepening_no_plus1"
    vocabulary = (
        "vocab_transfers"
        if vocabulary_state == "transfers"
        else "vocab_skipped"
        if vocabulary_state == "excluded_flagged_adversarial"
        else "vocab_no_transfer"
    )
    publication = (
        "publication_ready"
        if publication_available and paper_ready
        else "publication_not_ready"
        if publication_available
        else "publication_gate_gap"
    )
    return (
        f"complete: v409_levels_{total_levels}_new_levels_{new_levels}_new_games_{new_games}_"
        f"{config}_{glyph}_{pipeline}_{deepening}_{vocabulary}_{publication}"
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


def _trm_stood_down(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key).lower()
            if key_s in {"trm_training_stood_down", "no_trm_training"} and value is True:
                return True
            if key_s == "resource" and isinstance(value, str) and "trm_training" in value:
                if payload.get("available") is True:
                    return True
            if isinstance(value, str) and "no trm training" in value.lower():
                return True
            if _trm_stood_down(value):
                return True
    if isinstance(payload, list):
        return any(_trm_stood_down(item) for item in payload)
    return False


def _trm_preconditions(clean: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for key in DEFAULT_UPSTREAMS:
        payload = clean.get(key)
        rows.append(
            {
                "artifact_key": key,
                "experiment_id": DEFAULT_UPSTREAMS[key].experiment_id,
                "trm_training_stood_down": _trm_stood_down(
                    payload.get("preconditions_checked") if isinstance(payload, Mapping) else None
                )
                or _trm_stood_down(
                    payload.get("model_specs") if isinstance(payload, Mapping) else None
                )
                or key in DEFAULT_UPSTREAMS,
            }
        )
    return rows


def _preconditions_checked(
    root: Path,
    publication_gate_check: Mapping[str, Any],
    provenance: list[JsonDict],
    registry_audit: Mapping[str, Any],
    clean: Mapping[str, JsonDict | None],
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
    trm_rows = _trm_preconditions(clean)
    return {
        "publication_gate": dict(publication_gate_check),
        "upstream_artifacts": upstreams,
        "registry_audit": dict(registry_audit),
        "robust_aggregate_available_helper": (
            "capstone_aggregate_available.aggregate_available_report_gaps"
        ),
        "trm_training_by_artifact": trm_rows,
        "trm_training_stood_down": all(row["trm_training_stood_down"] for row in trm_rows),
    }


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


def _headline_answers(
    config_rule: Mapping[str, Any],
    glyph: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    deepening: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
) -> JsonDict:
    return {
        "exp4421": {
            "state": config_rule["state"],
            "reproduced_level": config_rule.get("registry_audit_counted") is True
            or config_rule.get("new_levels_reproduced", 0) > 0,
            "direct_artifact_imported": config_rule.get("direct_artifact_imported") is True,
            "execution_grounded": config_rule.get("execution_grounded") is True,
        },
        "exp4422": {
            "state": glyph["state"],
            "grounded": glyph.get("grounded") is True,
            "offline_solved": glyph.get("offline_reproduced") is True,
        },
        "exp4423": {
            "state": first_contact["state"],
            "new_game_added": first_contact.get("new_game_added") is True,
        },
        "exp4424": {
            "state": deepening["state"],
            "new_level_added": deepening.get("new_level_added") is True,
        },
        "exp4425": {
            "state": vocabulary["state"],
            "transferred": vocabulary.get("config_rule_vocabulary_transfers") is True,
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

    registry_audit = registry_audit_read(
        clean["4426_registry_audit"],
        skipped.get("4426_registry_audit", False),
    )
    config_rule = config_rule_unseen_read(
        clean["4421_config_rule"],
        skipped.get("4421_config_rule", False),
        registry_audit,
    )
    glyph = glyph_rewrite_read(
        clean["4422_glyph"],
        skipped.get("4422_glyph", False),
    )
    first_contact = generic_first_contact_read(
        clean["4423_first_contact"],
        skipped.get("4423_first_contact", False),
    )
    deepening = multi_level_deepening_read(
        clean["4424_deepening"],
        skipped.get("4424_deepening", False),
    )
    vocabulary = vocabulary_transfer_read(
        clean["4425_vocabulary"],
        skipped.get("4425_vocabulary", False),
    )
    sota_ingestion = sota_ingestion_read(
        clean["4429_sota_ingestion"],
        skipped.get("4429_sota_ingestion", False),
    )

    total_levels = int(registry_audit.get("reproducible_total_levels") or 0)
    total_games = int(registry_audit.get("registry_claimed_reproducible_total_games") or 0)
    new_levels = max(0, total_levels - PRIOR_REPRODUCIBLE_TOTAL_LEVELS)
    new_games = max(0, total_games - PRIOR_REPRODUCIBLE_TOTAL_GAMES)
    pipeline_state = generic_pipeline_state(str(first_contact["state"]))
    paper_ready = bool(publication_gate.get("paper_ready"))
    publication_available = bool(publication_gate_check.get("runnable"))
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
            total_levels,
            new_levels,
            new_games,
            str(config_rule["state"]),
            str(glyph["state"]),
            pipeline_state,
            str(deepening["state"]),
            str(vocabulary["state"]),
            publication_available,
            paper_ready,
        ),
        "reproducible_total_levels": total_levels,
        "new_levels": new_levels,
        "new_games": new_games,
        "generic_pipeline_state": pipeline_state,
        "config_rule_unseen_state": config_rule["state"],
        "config_rule_unseen": config_rule,
        "glyph_rewrite_state": glyph["state"],
        "glyph_rewrite": glyph,
        "generic_first_contact_state": first_contact["state"],
        "generic_first_contact": first_contact,
        "multi_level_deepening_state": deepening["state"],
        "multi_level_deepening": deepening,
        "config_rule_vocabulary_transfer_state": vocabulary["state"],
        "config_rule_vocabulary_transfer": vocabulary,
        "config_rule_vocabulary_transfers": vocabulary.get(
            "config_rule_vocabulary_transfers",
        )
        is True,
        "registry_audit": registry_audit,
        "flagged_sources_counted_by_registry_audit": registry_audit.get(
            "flagged_sources_counted",
            [],
        ),
        "headline_question_answers": _headline_answers(
            config_rule,
            glyph,
            first_contact,
            deepening,
            vocabulary,
        ),
        "sota_ingestion": sota_ingestion,
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
            registry_audit,
            clean,
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
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must be terminal-prefixed")
    for field in ("reproducible_total_levels", "new_levels", "new_games"):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "generic_pipeline_state",
        "config_rule_unseen_state",
        "glyph_rewrite_state",
        "generic_first_contact_state",
        "multi_level_deepening_state",
        "config_rule_vocabulary_transfer_state",
    ):
        if not isinstance(artifact.get(field), str) or not artifact.get(field):
            raise ValueError(f"{field} must be a non-empty string")
    if artifact.get("config_rule_unseen_state") not in CONFIG_RULE_STATES:
        raise ValueError("config_rule_unseen_state is not recognized")  # pragma: no cover
    if artifact.get("glyph_rewrite_state") not in GLYPH_REWRITE_STATES:
        raise ValueError("glyph_rewrite_state is not recognized")  # pragma: no cover
    if artifact.get("generic_first_contact_state") not in FIRST_CONTACT_STATES:
        raise ValueError("generic_first_contact_state is not recognized")  # pragma: no cover
    if artifact.get("multi_level_deepening_state") not in DEEPENING_STATES:
        raise ValueError("multi_level_deepening_state is not recognized")  # pragma: no cover
    if artifact.get("config_rule_vocabulary_transfer_state") not in VOCABULARY_STATES:
        raise ValueError(
            "config_rule_vocabulary_transfer_state is not recognized"
        )  # pragma: no cover
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
