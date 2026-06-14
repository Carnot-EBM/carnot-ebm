"""Exp 4181 registry/gaps hygiene for .387 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4181, SCENARIO-VERIFY-4181.

This runner is a read-only reconciler for model state. It replays the frozen
GAP-4 ARC-1 guard from cached artifacts and records the .387 moat and GAP-3
Stage-1 outcomes from already-written result JSON. It does not run Codex, load
GGUF models, use a GPU, launch TRM training, stop TRM training, or write the
stable TRM checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4153 as exp4153


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4181_ARTIFACT_PATH = "results/experiment_4181_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4175_PATH = "results/experiment_4175_headroom_gate_executable_census.json"
EXP4177_PATH = "results/experiment_4177_decisive_headroom_controlled_moat_test.json"
EXP4178_PATH = "results/experiment_4178_gap3_stage1_model_native_arc_energy.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
MOAT_GAP_ID = "GAP-MOAT-HEADROOM-CONTROLLED-4177"
GAP3_STAGE1_GAP_ID = "GAP-3-STAGE1-MODEL-NATIVE-LATENT-4178"
V387_ROLE_ID = "verifier_moat_gap3_hygiene_4181"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "moat_verdict",
    "gap3_stage1_result",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .387 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": "Lists the verifier_gaps entries touched (moat verdict + GAP-3 Stage-1).",
}


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _check_json_resource(repo_root: Path, resource: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"resource": resource, "available": False, "detail": f"missing: {rel_path}"}
    try:
        loaded = base._load_json(path)
    except Exception as exc:  # pragma: no cover - parse exception type varies by Python version.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):  # pragma: no cover - guards corrupted upstream artifacts.
        return {"resource": resource, "available": False, "detail": "not_json_object"}
    return {"resource": resource, "available": True, "detail": rel_path}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4181: verify cached fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4175_headroom_census", EXP4175_PATH),
        _check_json_resource(repo_root, "exp4177_headroom_controlled_moat", EXP4177_PATH),
        _check_json_resource(repo_root, "exp4178_gap3_stage1", EXP4178_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4181: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _moat_status(outcome: dict[str, Any]) -> str:
    if bool(outcome.get("verifier_value_added")):
        return "filled_headroom_controlled_verifier_value_added"
    return "open_headroom_controlled_no_value_added"


def classify_moat_verdict(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4181: summarize the Exp 4177 headroom-controlled moat verdict."""
    census = base._load_json(repo_root / EXP4175_PATH)
    artifact = base._load_json(repo_root / EXP4177_PATH)
    headroom_domain = str(census.get("headroom_present_domain") or artifact.get("domain") or "")
    per_domain = census.get("per_domain_headroom", {})
    domain_headroom = per_domain.get(headroom_domain, {}) if isinstance(per_domain, dict) else {}
    max_headroom = _numeric_or_none(census.get("max_selectable_headroom"))
    outcome = {
        "gap_id": MOAT_GAP_ID,
        "status": _moat_status(artifact),
        "artifact_path": EXP4177_PATH,
        "headroom_census_path": EXP4175_PATH,
        "source_artifacts": [EXP4175_PATH, EXP4177_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "headroom_present_domain": headroom_domain,
        "domain": str(artifact.get("domain", headroom_domain)),
        "verifier_value_added": artifact.get("verifier_value_added") is True,
        "positive_control_confirmed": artifact.get("positive_control_confirmed") is True,
        "moat_delta_vs_vote": dict(artifact.get("moat_delta_vs_vote", {})),
        "moat_vs_matched_control": dict(artifact.get("moat_vs_matched_control", {})),
        "accuracy_cost_pareto": dict(artifact.get("accuracy_cost_pareto", {})),
        "positive_control": dict(artifact.get("positive_control", {})),
        "max_selectable_headroom": max_headroom,
        "max_selectable_headroom_rounded": _round4(max_headroom),
        "domain_headroom": dict(domain_headroom) if isinstance(domain_headroom, dict) else {},
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "candidate_pool_source": str(artifact.get("candidate_pool_source", "")),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "missing_discriminator": "headroom_controlled_verifier_value_added_moat",
    }
    outcome["status"] = _moat_status(outcome)
    return outcome


def _gap3_stage1_status(outcome: dict[str, Any]) -> str:
    if bool(outcome.get("advances_toward_filled")):
        return "building_stage1_advances_toward_filled"
    return "open_stage1_honest_negative_does_not_advance"


def classify_gap3_stage1_result(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4181: summarize Exp 4178 without promoting an honest negative."""
    artifact = base._load_json(repo_root / EXP4178_PATH)
    pass2_delta = _numeric_or_none(artifact.get("pass2_energy_vs_vote"))
    headroom_capture = _numeric_or_none(artifact.get("headroom_capture_fraction"))
    advances = (
        artifact.get("all_four_gates_pass") is True
        and pass2_delta is not None
        and pass2_delta > 0.0
        and headroom_capture is not None
        and headroom_capture > 0.0
    )
    outcome = {
        "gap_id": GAP3_STAGE1_GAP_ID,
        "status": "building_stage1_advances_toward_filled" if advances else "open_stage1_honest_negative_does_not_advance",
        "artifact_path": EXP4178_PATH,
        "source_artifacts": [EXP4178_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "selected_energy": str(artifact.get("selected_energy", "")),
        "pass2_energy_vs_vote": pass2_delta,
        "pass2_energy_vs_vote_detail": dict(artifact.get("pass2_energy_vs_vote_detail", {})),
        "headroom_capture_fraction": headroom_capture,
        "headroom_capture_detail": dict(artifact.get("headroom_capture_detail", {})),
        "candidate_auroc": _numeric_or_none(artifact.get("candidate_auroc")),
        "candidate_auroc_detail": dict(artifact.get("candidate_auroc_detail", {})),
        "coverage_fraction": _numeric_or_none(artifact.get("coverage_fraction")),
        "all_four_gates_pass": artifact.get("all_four_gates_pass") is True,
        "gates": dict(artifact.get("gates", {})),
        "adversarial_checks": dict(artifact.get("adversarial_checks", {})),
        "advances_toward_filled": advances,
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "n_tasks": artifact.get("n_tasks"),
        "n_candidates": artifact.get("n_candidates"),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "missing_discriminator": "gap3_model_native_energy_beyond_vote",
    }
    outcome["status"] = _gap3_stage1_status(outcome)
    return outcome


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    moat_verdict: dict[str, Any],
    gap3_stage1_result: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .387 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_v387_role(updated_registry, moat_verdict, gap3_stage1_result)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4181-headroom-controlled-moat",
        _moat_gap_block(moat_verdict),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4181-gap3-stage1",
        _gap3_stage1_gap_block(gap3_stage1_result),
    )
    touched = [gap_id for gap_id in (MOAT_GAP_ID, GAP3_STAGE1_GAP_ID) if gap_id in updated_gaps]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "moat_recorded": MOAT_GAP_ID in touched,
            "gap3_stage1_recorded": GAP3_STAGE1_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - real/minimal registries include the GAP-4 entry.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4181": EXP4181_ARTIFACT_PATH,
            "exp4181_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4181_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4181_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4181_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4181_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_v387_role(
    registry: dict[str, Any],
    moat_verdict: dict[str, Any],
    gap3_stage1_result: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    role = {
        "role_id": V387_ROLE_ID,
        "experiment": EXP4181_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v387",
        "status": "moat_positive_gap3_stage1_nonadvancing",
        "moat_gap_id": moat_verdict.get("gap_id"),
        "moat_status": moat_verdict.get("status"),
        "moat_artifact": EXP4177_PATH,
        "moat_verifier_value_added": bool(moat_verdict.get("verifier_value_added")),
        "headroom_present_domain": moat_verdict.get("headroom_present_domain"),
        "moat_delta_vs_vote": moat_verdict.get("moat_delta_vs_vote", {}),
        "gap3_gap_id": gap3_stage1_result.get("gap_id"),
        "gap3_stage1_status": gap3_stage1_result.get("status"),
        "gap3_stage1_artifact": EXP4178_PATH,
        "gap3_stage1_pass2_energy_vs_vote": gap3_stage1_result.get(
            "pass2_energy_vs_vote"
        ),
        "gap3_stage1_headroom_capture_fraction": gap3_stage1_result.get(
            "headroom_capture_fraction"
        ),
        "gap3_stage1_advances_toward_filled": bool(
            gap3_stage1_result.get("advances_toward_filled")
        ),
        "eval_exp_4181": EXP4181_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V387_ROLE_ID
    ] + [role]


def _moat_gap_block(outcome: dict[str, Any]) -> str:
    delta = outcome.get("moat_delta_vs_vote", {})
    control = outcome.get("moat_vs_matched_control", {})
    return (
        f"### {MOAT_GAP_ID}: Exp 4181 .387 headroom-controlled moat verdict\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4177_PATH}` with headroom census `{EXP4175_PATH}`; "
        f"headroom_present_domain={outcome.get('headroom_present_domain')}; "
        f"verifier_value_added={str(bool(outcome.get('verifier_value_added'))).lower()}; "
        f"positive_control_confirmed={str(bool(outcome.get('positive_control_confirmed'))).lower()}; "
        f"moat_delta_vs_vote_delta={delta.get('delta')}; "
        f"moat_delta_vs_vote_ci95={delta.get('ci95')}; "
        f"moat_vs_matched_control_delta={control.get('delta')}; "
        f"max_selectable_headroom={outcome.get('max_selectable_headroom_rounded')}; "
        f"inference_substrate={outcome.get('inference_substrate')}.\n"
        "- failure mode: closed for this headroom-controlled code-domain moat test; "
        "the verifier-plus-selector arm beats self-consistency vote with a positive CI.\n"
        "- missing discriminator: none for the measured .387 code-domain moat verdict; "
        "continue to require headroom-positive domains before interpreting moat nulls.\n"
        "- candidate design: preserve the objective headroom gate and matched-control arm "
        "for future verifier-value tests.\n"
        "- priority: medium\n"
    )


def _gap3_stage1_gap_block(outcome: dict[str, Any]) -> str:
    pass2 = outcome.get("pass2_energy_vs_vote_detail", {})
    headroom = outcome.get("headroom_capture_detail", {})
    return (
        f"### {GAP3_STAGE1_GAP_ID}: Exp 4181 .387 GAP-3 Stage-1 latent-energy result\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4178_PATH}`; "
        f"selected_energy={outcome.get('selected_energy')}; "
        f"pass2_energy_vs_vote={outcome.get('pass2_energy_vs_vote')}; "
        f"energy_pass2={pass2.get('energy_pass2')}; "
        f"vote_pass2={pass2.get('vote_pass2')}; "
        f"oracle_pass2={pass2.get('oracle_pass2')}; "
        f"bootstrap_ci95={pass2.get('bootstrap_ci95')}; "
        f"headroom_capture_fraction={outcome.get('headroom_capture_fraction')}; "
        f"oracle_minus_vote={headroom.get('oracle_minus_vote')}; "
        f"candidate_auroc={outcome.get('candidate_auroc')}; "
        f"coverage_fraction={outcome.get('coverage_fraction')}; "
        f"all_four_gates_pass={str(bool(outcome.get('all_four_gates_pass'))).lower()}; "
        f"advances_toward_filled={str(bool(outcome.get('advances_toward_filled'))).lower()}.\n"
        "- failure mode: the model-native latent energy ties vote at pass@2 and captures "
        "none of the oracle-minus-vote headroom, so Stage 1 is an honest negative rather "
        "than a filled GAP-3 discriminator.\n"
        "- missing discriminator: a model-native ARC energy that improves pass@2 over vote "
        "and captures real headroom without oracle leakage.\n"
        "- candidate design: keep GAP-3 open for a stronger generator-independent content "
        "energy; do not promote Stage 1 toward filled from this result.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4181") == EXP4181_ARTIFACT_PATH
        and any(role.get("role_id") == V387_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    moat_verdict: dict[str, Any],
    gap3_stage1_result: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4181 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {MOAT_GAP_ID, GAP3_STAGE1_GAP_ID}
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4181_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4181_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v387_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"moat_{moat_verdict['status']}_"
            f"gap3_{gap3_stage1_result['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "moat_verdict": moat_verdict,
        "gap3_stage1_result": gap3_stage1_result,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4175_PATH,
            EXP4177_PATH,
            EXP4178_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4181", "SCENARIO-VERIFY-4181"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:  # pragma: no cover
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4181_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4181_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "moat_verdict": {"status": "blocked_precondition", "gap_id": MOAT_GAP_ID},
        "gap3_stage1_result": {
            "status": "blocked_precondition",
            "gap_id": GAP3_STAGE1_GAP_ID,
            "advances_toward_filled": False,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4175_PATH,
            EXP4177_PATH,
            EXP4178_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4181", "SCENARIO-VERIFY-4181"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4181 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a bare bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4181 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4181 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4181_ARTIFACT_PATH
    if not preflight["ok"]:  # pragma: no cover - success path is the required artifact path.
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    moat_verdict = classify_moat_verdict(repo_root)
    gap3_stage1_result = classify_gap3_stage1_result(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        moat_verdict,
        gap3_stage1_result,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        moat_verdict=moat_verdict,
        gap3_stage1_result=gap3_stage1_result,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4181_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
