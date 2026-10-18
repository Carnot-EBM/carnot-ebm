"""Build the Exp 4312 v398 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4312, SCENARIO-CAPSTONE-4312.
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


OUTPUT_REL_PATH = Path("results/experiment_4312_capstone_v398.json")
EXPERIMENT_ID = 4312
RANDOM_SEED = 4312
SCHEMA = "carnot.capstone_v398_4312.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4312", "SCENARIO-CAPSTONE-4312"]
BLOCKED_CHECKSUM = hashlib.sha256(b"blocked_no_v398_artifacts").hexdigest()

THESIS_STATES = {
    "efficiency_parity_hardened",
    "in_generation_moat_holds",
    "cross_domain_moat_holds",
    "selection_moat_arc_only",
    "in_generation_still_open",
}


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4303_efficiency": Upstream(
        4303, Path("results/experiment_4303_verifier_efficiency_parity_isoflops.json")
    ),
    "4304_in_generation": Upstream(
        4304, Path("results/experiment_4304_diffusiongemma_in_generation_engaged_controls.json")
    ),
    "4305_cross_domain": Upstream(
        4305, Path("results/experiment_4305_cross_domain_selector_generalization.json")
    ),
    "4306_self_learning": Upstream(
        4306, Path("results/experiment_4306_self_learning_powered_ci_cross_domain.json")
    ),
    "4307_arc_progress": Upstream(
        4307, Path("results/experiment_4307_arc_incremental_progress_new_game.json")
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "efficiency_pareto_hardened",
    "in_generation_moat_holds",
    "cross_domain_moat_holds",
    "verifier_thesis_state",
    "flagged_artifacts_excluded",
    "per_axis_gaps",
    "paper_ready",
    "verifier_is_oracle_honored",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The .398 close-state -- whether efficiency-parity hardened, "
        "the in-generation moat held with engaged controls, the selection moat broadened "
        "to cross-domain."
    ),
    "headline_outcome": (
        "One honest string aggregating the efficiency + in-generation + cross-domain + "
        "self-learning + ARC reads; the single line the .399 planner frames from."
    ),
    "efficiency_pareto_hardened": (
        "BARE bool: did the energy verifier match/beat a WELL-PROMPTED judge at <=0.1x "
        "cost on an iso-FLOPs curve -- the operator's §5 win condition, finally hardened."
    ),
    "in_generation_moat_holds": (
        "BARE bool: did the LEARNED partial-state-guided DiffusionGemma beat a "
        "GENUINELY-ENGAGED control (controls_differentiated) with CI95-excl-0 -- "
        "the §5 in-generation moat (false if gated off / no-op controls / ties / "
        "scorer leaked)."
    ),
    "cross_domain_moat_holds": (
        "BARE bool: did the selection moat transfer to a HELD-OUT DOMAIN with "
        "label_ablation_robust -- escaping the verifier's math/ARC domain bound."
    ),
    "verifier_thesis_state": (
        "One honest string for the verifier thesis state (efficiency_parity_hardened / "
        "in_generation_moat_holds / cross_domain_moat_holds / selection_moat_arc_only / "
        "in_generation_still_open) -- the framing the .399 planner inherits."
    ),
    "flagged_artifacts_excluded": (
        "List of .398 artifacts excluded for flagged_adversarial -- the fabrication "
        "gate (their numbers are NOT aggregated)."
    ),
    "per_axis_gaps": (
        "List of .398 axes whose artifact was MISSING (reported as a gap, NOT defaulted "
        "False) -- the robust-aggregator fix that prevents the .397 spurious-all-False "
        "capstone bug."
    ),
    "paper_ready": (
        "From publication_gate.py --json -- the G1-G4 status (FoVer headline stays the "
        "publication target; a hardened efficiency-parity, an established in-generation "
        "moat, or a cross-domain win would be a new headline-grade supporting result)."
    ),
    "verifier_is_oracle_honored": (
        "BARE bool=true -- confirms every cited moat/headline result carried "
        "verifier_is_oracle=false (no circular/execution-grounded result headlines a moat)."
    ),
    "reproducibility_checksum": (
        "Hash of the aggregated upstream sha256 set; lets a third party re-derive the "
        "capstone."
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4303_efficiency": [
        "efficiency_pareto_holds",
        "cost_ratio",
        "accuracy_delta_ci95",
        "accuracy_energy_verifier",
        "accuracy_best_judge",
        "verifier_is_oracle",
    ],
    "4304_in_generation": [
        "diffusiongemma_guidance_moat",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "carnot_minus_best_control_delta",
        "guidance_moat_ci95",
        "verifier_is_oracle",
    ],
    "4305_cross_domain": [
        "cross_domain_selection_holds",
        "cross_domain_delta",
        "cross_domain_ci95",
        "label_ablation_robust",
        "held_out_task_n",
        "verifier_is_oracle",
    ],
    "4306_self_learning": [
        "online_adaptation_helps",
        "best_adaptive_minus_static_delta",
        "best_adaptive_minus_static_ci95",
        "held_out_task_n",
        "verifier_is_oracle",
    ],
    "4307_arc_progress": [
        "total_levels",
        "total_levels_solved",
        "levels_completed",
        "new_levels_solved_this_task",
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
            name="efficiency",
            required_keys=("4303_efficiency",),
            verdict_fn=lambda present: efficiency_read(
                present.get("4303_efficiency"), False
            )["efficiency_pareto_hardened"]
            is True,
        ),
        aggregate.AxisSpec(
            name="in_generation",
            required_keys=("4304_in_generation",),
            verdict_fn=lambda present: in_generation_read(
                present.get("4304_in_generation"), False
            )["in_generation_moat_holds"]
            is True,
        ),
        aggregate.AxisSpec(
            name="cross_domain",
            required_keys=("4305_cross_domain",),
            verdict_fn=lambda present: cross_domain_read(
                present.get("4305_cross_domain"), False
            )["cross_domain_moat_holds"]
            is True,
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4306_self_learning",),
            verdict_fn=lambda present: self_learning_read(
                present.get("4306_self_learning"), False
            )["online_adaptation_helps"]
            is True,
        ),
        aggregate.AxisSpec(
            name="arc_progress",
            required_keys=("4307_arc_progress",),
            verdict_fn=lambda present: int_metric(present.get("4307_arc_progress"), "total_levels")
            > 0,
        ),
    ]


def efficiency_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    hardened = (
        bool_metric(payload, "efficiency_pareto_holds") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "hardened" if hardened else "not_hardened",
        "efficiency_pareto_hardened": hardened,
        "reported_efficiency_pareto_holds": bool_metric(payload, "efficiency_pareto_holds"),
        "cost_ratio": float_metric(payload, "cost_ratio"),
        "accuracy_delta_ci95": list_metric(payload, "accuracy_delta_ci95"),
        "accuracy_energy_verifier": float_metric(payload, "accuracy_energy_verifier"),
        "accuracy_best_judge": float_metric(payload, "accuracy_best_judge"),
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
        and bool_metric(payload, "scorer_leak_recheck_passed") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "moat_holds" if moat else "open",
        "in_generation_moat_holds": moat,
        "reported_diffusiongemma_guidance_moat": bool_metric(
            payload, "diffusiongemma_guidance_moat"
        ),
        "controls_differentiated": bool_metric(payload, "controls_differentiated"),
        "scorer_leak_recheck_passed": bool_metric(payload, "scorer_leak_recheck_passed"),
        "carnot_minus_best_control_delta": float_metric(payload, "carnot_minus_best_control_delta"),
        "guidance_moat_ci95": list_metric(payload, "guidance_moat_ci95"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


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
        "cross_domain_ci95": list_metric(payload, "cross_domain_ci95"),
        "label_ablation_robust": bool_metric(payload, "label_ablation_robust"),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def self_learning_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    helps = (
        bool_metric(payload, "online_adaptation_helps") is True
        and bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "helps" if helps else "not_helped",
        "online_adaptation_helps": helps,
        "reported_online_adaptation_helps": bool_metric(payload, "online_adaptation_helps"),
        "best_adaptive_minus_static_delta": float_metric(
            payload, "best_adaptive_minus_static_delta"
        ),
        "best_adaptive_minus_static_ci95": list_metric(
            payload, "best_adaptive_minus_static_ci95"
        ),
        "held_out_task_n": int_metric(payload, "held_out_task_n"),
        "verifier_is_oracle": bool_metric(payload, "verifier_is_oracle"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def arc_progress_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial"}
    if payload is None:
        return {"status": "missing_or_excluded"}
    total = int_metric(payload, "total_levels") or int_metric(payload, "total_levels_solved")
    return {
        "status": "included",
        "total_levels": total,
        "total_levels_solved": total,
        "levels_completed": int_metric(payload, "levels_completed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task"),
        "honest_verdict": str_metric(payload, "honest_verdict"),
    }


def verifier_thesis_state(
    efficiency_hardened: bool,
    in_generation_holds: bool,
    cross_domain_holds: bool,
    selection_axis_available: bool,
) -> str:
    if cross_domain_holds:
        return "cross_domain_moat_holds"
    if in_generation_holds:
        return "in_generation_moat_holds"
    if efficiency_hardened:
        return "efficiency_parity_hardened"
    if selection_axis_available:
        return "selection_moat_arc_only"
    return "in_generation_still_open"


def _part(read: Mapping[str, Any], true_key: str, true_part: str, false_part: str) -> str:
    if read.get(true_key) is True:
        return true_part
    status = str(read.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return false_part


def _headline_outcome(
    efficiency: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    cross_domain: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
    paper_ready: bool,
) -> str:
    paper = "paper_ready" if paper_ready else "paper_not_ready"
    return (
        f"efficiency_{_part(efficiency, 'efficiency_pareto_hardened', 'hardened', 'open')}__"
        f"in_generation_{_part(in_generation, 'in_generation_moat_holds', 'moat', 'open')}__"
        f"cross_domain_{_part(cross_domain, 'cross_domain_moat_holds', 'moat', 'open')}__"
        f"self_learning_{_part(self_learning, 'online_adaptation_helps', 'helps', 'not_helped')}__"
        f"arc_{_arc_part(arc_progress)}__{paper}"
    )


def _arc_part(arc_progress: Mapping[str, Any]) -> str:
    status = str(arc_progress.get("status"))
    if status == "excluded_flagged_adversarial":
        return "excluded"
    if status == "missing_or_excluded":
        return "missing"
    return f"levels_{int(arc_progress.get('total_levels') or 0)}"


def _honest_verdict(
    efficiency: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    cross_domain: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    arc_progress: Mapping[str, Any],
) -> str:
    return (
        "complete: v398_efficiency_"
        f"{_part(efficiency, 'efficiency_pareto_hardened', 'hardened', 'open')}_"
        "in_generation_"
        f"{_part(in_generation, 'in_generation_moat_holds', 'moat', 'open')}_"
        "cross_domain_"
        f"{_part(cross_domain, 'cross_domain_moat_holds', 'moat', 'open')}_"
        "self_learning_"
        f"{_part(self_learning, 'online_adaptation_helps', 'helps', 'not_helped')}_"
        f"arc_{_arc_part(arc_progress)}"
    )


def _oracle_violations(
    efficiency: Mapping[str, Any],
    in_generation: Mapping[str, Any],
    cross_domain: Mapping[str, Any],
    self_learning: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if (
        efficiency.get("reported_efficiency_pareto_holds") is True
        and efficiency.get("verifier_is_oracle") is not False
    ):
        violations.append("4303_efficiency:efficiency_pareto")
    if (
        in_generation.get("reported_diffusiongemma_guidance_moat") is True
        and in_generation.get("verifier_is_oracle") is not False
    ):
        violations.append("4304_in_generation:in_generation")
    if (
        cross_domain.get("reported_cross_domain_selection_holds") is True
        and cross_domain.get("verifier_is_oracle") is not False
    ):
        violations.append("4305_cross_domain:cross_domain")
    if (
        self_learning.get("reported_online_adaptation_helps") is True
        and self_learning.get("verifier_is_oracle") is not False
    ):
        violations.append("4306_self_learning:self_learning")
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
    efficiency = efficiency_read(clean["4303_efficiency"], skipped.get("4303_efficiency", False))
    in_generation = in_generation_read(
        clean["4304_in_generation"], skipped.get("4304_in_generation", False)
    )
    cross_domain = cross_domain_read(
        clean["4305_cross_domain"], skipped.get("4305_cross_domain", False)
    )
    self_learning = self_learning_read(
        clean["4306_self_learning"], skipped.get("4306_self_learning", False)
    )
    arc_progress = arc_progress_read(
        clean["4307_arc_progress"], skipped.get("4307_arc_progress", False)
    )
    publication_gate = publication_gate_runner(root)
    paper_ready = bool(publication_gate.get("paper_ready"))
    violations = _oracle_violations(efficiency, in_generation, cross_domain, self_learning)
    selection_axis_available = cross_domain.get("status") not in {
        "missing_or_excluded",
        "excluded_flagged_adversarial",
    }
    thesis = verifier_thesis_state(
        efficiency.get("efficiency_pareto_hardened") is True,
        in_generation.get("in_generation_moat_holds") is True,
        cross_domain.get("cross_domain_moat_holds") is True,
        selection_axis_available,
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
            "blocked_no_v398_artifacts"
            if blocked
            else _honest_verdict(efficiency, in_generation, cross_domain, self_learning, arc_progress)
        ),
        "headline_outcome": (
            "blocked_no_v398_artifacts"
            if blocked
            else _headline_outcome(
                efficiency, in_generation, cross_domain, self_learning, arc_progress, paper_ready
            )
        ),
        "efficiency_pareto_hardened": efficiency.get("efficiency_pareto_hardened") is True,
        "in_generation_moat_holds": in_generation.get("in_generation_moat_holds") is True,
        "cross_domain_moat_holds": cross_domain.get("cross_domain_moat_holds") is True,
        "verifier_thesis_state": "in_generation_still_open" if blocked else thesis,
        "flagged_artifacts_excluded": exclusions,
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", [])),
        "paper_ready": paper_ready,
        "unmet_gates": list_metric(publication_gate, "unmet_gates"),
        "publication_gate": publication_gate,
        "verifier_is_oracle_honored": not violations,
        "oracle_distinct_violations": violations,
        "efficiency": efficiency,
        "in_generation": in_generation,
        "cross_domain": cross_domain,
        "self_learning": self_learning,
        "arc_progress": arc_progress,
        "availability_report": availability_report,
        "upstream_provenance": provenance,
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
    if verdict != "blocked_no_v398_artifacts":
        if not isinstance(verdict, str) or not verdict.startswith(
            ("complete:", "success:", "passed:", "shipped:", "blocked:")
        ):
            raise ValueError("honest_verdict must be terminal-prefixed")
    headline = artifact.get("headline_outcome")
    if not isinstance(headline, str) or not headline:
        raise ValueError("headline_outcome must be a non-empty string")
    for field in (
        "efficiency_pareto_hardened",
        "in_generation_moat_holds",
        "cross_domain_moat_holds",
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
