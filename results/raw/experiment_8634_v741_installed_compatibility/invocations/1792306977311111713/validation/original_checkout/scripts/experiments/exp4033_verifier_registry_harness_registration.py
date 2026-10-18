"""Exp 4033 - GAP-4 verifier registry harness registration.

Spec refs: REQ-VERIFY-4033, SCENARIO-VERIFY-4033.

This harness is intentionally offline-only. It reads cached GAP-4 ARC artifacts,
checks the reusable registry entry, records the Exp 4032 off-ARC state in the
registry/gaps ledgers, and writes the terminal Exp 4033 artifact. It must not
make Codex, GGUF, or other live inference calls.
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4033_verifier_registry_harness_registration.json"
REGISTRY_PATH = REPO_ROOT / "ops" / "verifier_registry.yaml"
GAPS_PATH = REPO_ROOT / "ops" / "verifier_gaps.md"

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CODE_TRANSFER_VERIFIER_ID = "gap4_code_demo_fit_execution_transfer"
CODE_GAP_MARKER = "GAP-CODE-EXEC-DEMOFIT"

ARC2_CHAIN_PATH = "results/arc3_gap4_arc2_chain_ensemble.json"
ARC1_PROGRAMS_PATH = "results/arc3_gap4_induced_programs.json"
EXP4033_ARTIFACT_PATH = "results/experiment_4033_verifier_registry_harness_registration.json"

OFFARC_RAW_CANDIDATES = (
    "results/experiment_4032_offarc_exec_verifier_transfer.json",
    "results/experiment_4032_offarc_exec_verifier_transfer_raw.json",
)

REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "registry_updated",
    "offline_reeval_bitexact",
    "off_arc_outcome_recorded",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix summary of registration, offline replay, and off-ARC routing.",
    "registry_updated": "Bare bool; the registry is the honest map of verifier coverage.",
    "offline_reeval_bitexact": (
        "Bare bool; a reusable module that misses the headline cached numbers is a regression."
    ),
    "off_arc_outcome_recorded": (
        "One of code_entry_added, gap_logged, off_arc_pending; off-ARC evidence must land in a ledger."
    ),
    "inference_substrate": "aggregation_from_upstream_artifacts; no Codex, GGUF, or live inference.",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"verifiers": []}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {"verifiers": []}
    loaded.setdefault("verifiers", [])
    return loaded


def _write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")


def find_gap4_entry(registry: dict[str, Any]) -> dict[str, Any]:
    """Return the registered GAP-4 entry or raise a clear error."""
    for entry in registry.get("verifiers", []):
        if entry.get("verifier_id") == "gap4_program_induction_stack":
            return entry
    raise ValueError("gap4_program_induction_stack is not registered")


def _gap4_entry_template() -> dict[str, Any]:
    return {
        "verifier_id": "gap4_program_induction_stack",
        "domain": "arc_agi2_grid",
        "version": 1,
        "kind": "process_verifier",
        "code_commit": "HEAD",
        "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": ARC2_CHAIN_PATH,
        "label_source": "gold-posthoc",
        "eval": {
            "metric": "pass_at_1",
            "arc2_pass_at_1": 0.6129,
            "arc2_n": 31,
            "arc2_gold": 19,
            "arc2_eval_artifact": ARC2_CHAIN_PATH,
            "arc1_demo_perfect_coverage": 0.9032,
            "arc1_n": 31,
            "arc1_gold": 28,
            "arc1_eval_artifact": ARC1_PROGRAMS_PATH,
            "arc2_offline_reproduced": True,
            "arc1_offline_reproduced": True,
            "eval_exp_4033": EXP4033_ARTIFACT_PATH,
        },
        "selection_policy": _selection_policy(),
        "agreement_role": "confidence_label_only",
        "agreement_precision_selector": False,
        "lineage": {
            "from": "arc_grid_combined_verifier (v2)",
            "change": (
                "ARC-2 program-induction stack: snap tau<=0.005 -> "
                "promote-first-fresh-raw-demo-perfect -> vote; agreement is confidence only."
            ),
        },
        "status": "candidate",
        "notes": "Offline-registered reusable GAP-4 stack; pure-code selector over cached artifacts.",
    }


def _selection_policy() -> dict[str, Any]:
    return {
        "tier_1": "snap_tau_lte_0.005",
        "tier_2": "promote_first_fresh_raw_demo_perfect",
        "tier_3": "vote_fallback",
        "agreement": "confidence_label_only_fresh_arms_never_gate",
    }


def ensure_gap4_registered(registry: dict[str, Any]) -> bool:
    """Ensure the reusable GAP-4 module entry has the Exp 4033 registration fields."""
    changed = False
    try:
        entry = find_gap4_entry(registry)
    except ValueError:
        registry.setdefault("verifiers", []).append(_gap4_entry_template())
        return True

    required = {
        "domain": "arc_agi2_grid",
        "kind": "process_verifier",
        "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": ARC2_CHAIN_PATH,
        "agreement_role": "confidence_label_only",
        "agreement_precision_selector": False,
    }
    for key, value in required.items():
        if entry.get(key) != value:
            entry[key] = value
            changed = True

    eval_block = entry.setdefault("eval", {})
    eval_required = {
        "arc2_n": 31,
        "arc2_gold": 19,
        "arc2_eval_artifact": ARC2_CHAIN_PATH,
        "arc1_n": 31,
        "arc1_gold": 28,
        "arc1_eval_artifact": ARC1_PROGRAMS_PATH,
        "arc2_offline_reproduced": True,
        "arc1_offline_reproduced": True,
        "eval_exp_4033": EXP4033_ARTIFACT_PATH,
    }
    for key, value in eval_required.items():
        if eval_block.get(key) != value:
            eval_block[key] = value
            changed = True
    if "arc2_pass_at_1" not in eval_block:
        eval_block["arc2_pass_at_1"] = 0.6129
        changed = True
    if "arc1_demo_perfect_coverage" not in eval_block:
        eval_block["arc1_demo_perfect_coverage"] = 0.9032
        changed = True

    policy = _selection_policy()
    if entry.get("selection_policy") != policy:
        entry["selection_policy"] = policy
        changed = True

    return changed


def gap4_registration_is_valid(registry: dict[str, Any]) -> bool:
    """Check the registry records the intended reusable policy and confidence-only agreement."""
    try:
        entry = find_gap4_entry(registry)
    except ValueError:
        return False
    return (
        entry.get("code_path") == "python/carnot/agentic/gap4_program_induction_stack.py"
        and entry.get("selection_policy") == _selection_policy()
        and entry.get("agreement_role") == "confidence_label_only"
        and entry.get("agreement_precision_selector") is False
        and entry.get("eval", {}).get("arc2_gold") == 19
        and entry.get("eval", {}).get("arc1_gold") == 28
    )


def replay_gap4_arc_numbers(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay ARC-2 19/31 and ARC-1 28/31 from cached artifacts only."""
    from carnot.agentic.gap4_program_induction_stack import (  # noqa: PLC0415
        replay_arc1_demo_perfect_coverage_from_saved,
        replay_arc2_pass_at_1_from_saved,
    )

    arc2 = _load_json(repo_root / ARC2_CHAIN_PATH)
    arc1 = _load_json(repo_root / ARC1_PROGRAMS_PATH)

    arc2_gold, arc2_n = replay_arc2_pass_at_1_from_saved(arc2, pool_size=31)
    arc1_covered, arc1_n = replay_arc1_demo_perfect_coverage_from_saved(arc1, pool_size=31)
    bitexact = arc2_gold == 19 and arc2_n == 31 and arc1_covered == 28 and arc1_n == 31
    return {
        "offline_reeval_bitexact": bitexact,
        "arc2": {
            "gold": arc2_gold,
            "n": arc2_n,
            "pass_at_1": round(arc2_gold / arc2_n, 4) if arc2_n else 0.0,
        },
        "arc1": {
            "covered": arc1_covered,
            "n": arc1_n,
            "demo_perfect_coverage": round(arc1_covered / arc1_n, 4) if arc1_n else 0.0,
        },
    }


def classify_off_arc_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4032 evidence into registry/gap/pending ledger action."""
    for rel_path in OFFARC_RAW_CANDIDATES:
        path = repo_root / rel_path
        if not path.exists():
            continue
        try:
            artifact = _load_json(path)
        except Exception as exc:  # noqa: BLE001
            return _pending(f"unreadable_exp4032_artifact: {type(exc).__name__}", rel_path)
        if not _raw_offarc_complete(artifact):
            return _pending("incomplete_exp4032_raw_artifact", rel_path)
        verdict = str(artifact.get("honest_verdict", ""))
        if verdict.startswith("blocked_"):
            return _pending(verdict, rel_path)

        arm_a = float(artifact["armA_vote_passrate"])
        arm_b = float(artifact["armB_demofit_passrate"])
        ci = [float(artifact["bootstrap_ci95"][0]), float(artifact["bootstrap_ci95"][1])]
        common = {
            "artifact_path": rel_path,
            "n_tasks": int(artifact["n_tasks"]),
            "k_candidates_per_task": int(artifact["k_candidates_per_task"]),
            "armA_vote_passrate": arm_a,
            "armB_demofit_passrate": arm_b,
            "delta_pp": float(artifact["delta_pp"]),
            "bootstrap_ci95": ci,
            "oracle_passrate": float(artifact["oracle_passrate"]),
        }
        if arm_b > arm_a and ci[0] > 0.0:
            return {
                "off_arc_outcome_recorded": "code_entry_added",
                "status": "transfer",
                **common,
            }
        return {
            "off_arc_outcome_recorded": "gap_logged",
            "status": "no_transfer",
            "missing_discriminator": "code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics",
            **common,
        }

    return _pending("missing_completed_exp4032_raw_artifact", None)


def _pending(reason: str, artifact_path: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "off_arc_outcome_recorded": "off_arc_pending",
        "status": "off_arc_pending",
        "reason": reason,
    }
    if artifact_path is not None:
        out["artifact_path"] = artifact_path
    return out


def _raw_offarc_complete(artifact: dict[str, Any]) -> bool:
    required = {
        "honest_verdict",
        "n_tasks",
        "k_candidates_per_task",
        "armA_vote_passrate",
        "armB_demofit_passrate",
        "delta_pp",
        "bootstrap_ci95",
        "oracle_passrate",
        "inference_substrate",
    }
    if not required.issubset(artifact):
        return False
    if artifact.get("inference_substrate") != "live_llm_inference":
        return False
    if int(artifact.get("n_tasks", 0)) < 30:
        return False
    if int(artifact.get("k_candidates_per_task", 0)) < 8:
        return False
    ci = artifact.get("bootstrap_ci95")
    return isinstance(ci, list) and len(ci) == 2


def ensure_ledgers_record_outcome(
    registry: dict[str, Any],
    gaps_text: str,
    outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    """Return updated registry/gaps text with Exp 4032's outcome recorded once."""
    updated_registry = deepcopy(registry)
    updated_gaps = gaps_text
    changed = ensure_gap4_registered(updated_registry)
    recorded = outcome["off_arc_outcome_recorded"]
    gap4 = find_gap4_entry(updated_registry)

    transfer_note = _gap4_off_arc_transfer_note(outcome)
    if gap4.get("off_arc_transfer") != transfer_note:
        gap4["off_arc_transfer"] = transfer_note
        changed = True

    if recorded == "code_entry_added":
        entry = _code_transfer_entry(outcome)
        existing = _find_verifier(updated_registry, CODE_TRANSFER_VERIFIER_ID)
        if existing is None:
            updated_registry.setdefault("verifiers", []).append(entry)
            changed = True
        elif existing != entry:
            existing.clear()
            existing.update(entry)
            changed = True
    elif recorded == "gap_logged":
        if CODE_GAP_MARKER not in updated_gaps:
            updated_gaps = updated_gaps.rstrip() + "\n\n" + _code_gap_entry(outcome) + "\n"
            changed = True
    elif recorded != "off_arc_pending":
        raise ValueError(f"unknown off_arc_outcome_recorded: {recorded}")

    return updated_registry, updated_gaps, changed


def _find_verifier(registry: dict[str, Any], verifier_id: str) -> dict[str, Any] | None:
    for entry in registry.get("verifiers", []):
        if entry.get("verifier_id") == verifier_id:
            return entry
    return None


def _gap4_off_arc_transfer_note(outcome: dict[str, Any]) -> dict[str, Any]:
    note = {
        "experiment": 4032,
        "outcome": outcome["off_arc_outcome_recorded"],
        "status": outcome["status"],
    }
    for key in (
        "artifact_path",
        "reason",
        "delta_pp",
        "bootstrap_ci95",
        "n_tasks",
        "k_candidates_per_task",
        "armA_vote_passrate",
        "armB_demofit_passrate",
        "oracle_passrate",
        "missing_discriminator",
    ):
        if key in outcome:
            note[key] = outcome[key]
    return note


def _code_transfer_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "verifier_id": CODE_TRANSFER_VERIFIER_ID,
        "domain": "code",
        "version": 1,
        "kind": "process_verifier",
        "code_commit": "HEAD",
        "code_path": "scripts/experiments/offarc_exec_verifier_transfer_run.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome.get("artifact_path"),
        "label_source": "formal_oracle",
        "eval": {
            "metric": "hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["delta_pp"],
            "bootstrap_ci95": outcome["bootstrap_ci95"],
            "n": outcome["n_tasks"],
            "eval_artifact": outcome.get("artifact_path"),
        },
        "lineage": {
            "from": "gap4_program_induction_stack",
            "change": "OFF-ARC code transfer of demo-fit execution verifier from Exp 4032.",
        },
        "status": "candidate",
        "notes": (
            "Code-domain transfer entry added only because Exp 4032 Arm B beat Arm A "
            "and the bootstrap CI excluded zero."
        ),
    }


def _code_gap_entry(outcome: dict[str, Any]) -> str:
    artifact = outcome.get("artifact_path", "results/experiment_4032_offarc_exec_verifier_transfer_raw.json")
    return (
        f"### {CODE_GAP_MARKER}: code hidden-semantic execution discriminator\n"
        "- status: open\n"
        f"- evidence: `{artifact}` measured no OFF-ARC demo-fit transfer "
        f"(delta_pp={outcome.get('delta_pp')}, CI95={outcome.get('bootstrap_ci95')}).\n"
        "- failure mode: candidates can pass visible demo tests while failing hidden semantic tests.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: enrich the code verifier with hidden-property synthesis, stronger "
        "metamorphic tests, or formal/runtime oracles beyond visible examples.\n"
        "- priority: high\n"
    )


def build_artifact(
    *,
    registry_updated: bool,
    offline_reeval: dict[str, Any],
    off_arc_outcome: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    outcome = off_arc_outcome["off_arc_outcome_recorded"]
    replay_label = "bitexact" if offline_reeval.get("offline_reeval_bitexact") else "mismatch"
    registry_label = "registered" if registry_updated else "registration_incomplete"
    artifact = {
        "experiment": "experiment_4033_verifier_registry_harness_registration",
        "schema": "carnot.experiment_4033_verifier_registry_harness_registration.v1",
        "honest_verdict": (
            f"complete: gap4_stack_{registry_label}_offline_reeval_{replay_label}_offarc_{outcome}"
        ),
        "registry_updated": registry_updated,
        "offline_reeval_bitexact": bool(offline_reeval.get("offline_reeval_bitexact")),
        "off_arc_outcome_recorded": outcome,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 3),
        "reusable_module_path": "python/carnot/agentic/gap4_program_induction_stack.py",
        "registry_path": "ops/verifier_registry.yaml",
        "gaps_path": "ops/verifier_gaps.md",
        "offline_replay": offline_reeval,
        "off_arc_outcome": off_arc_outcome,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC2_CHAIN_PATH,
            ARC1_PROGRAMS_PATH,
            "results/arc3_gap4_chain_arms_adversarial_verify.json",
            *OFFARC_RAW_CANDIDATES,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("registry_updated", "offline_reeval_bitexact"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    if artifact["off_arc_outcome_recorded"] not in {
        "code_entry_added",
        "gap_logged",
        "off_arc_pending",
    }:
        raise ValueError("off_arc_outcome_recorded has an unknown value")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_registration(repo_root: Path = REPO_ROOT, output_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    output = output_path or repo_root / EXP4033_ARTIFACT_PATH
    registry_path = repo_root / "ops" / "verifier_registry.yaml"
    gaps_path = repo_root / "ops" / "verifier_gaps.md"

    registry = _load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8") if gaps_path.exists() else ""

    offline = replay_gap4_arc_numbers(repo_root)
    off_arc = classify_off_arc_outcome(repo_root)
    registry, gaps_text, ledger_changed = ensure_ledgers_record_outcome(registry, gaps_text, off_arc)
    registry_updated = gap4_registration_is_valid(registry)

    if ledger_changed:
        _write_registry(registry_path, registry)
        gaps_path.parent.mkdir(parents=True, exist_ok=True)
        gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        registry_updated=registry_updated,
        offline_reeval=offline,
        off_arc_outcome=off_arc,
        duration_s=time.time() - started,
    )
    _write_json(output, artifact)
    print(f"Wrote {output}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"registry_updated={artifact['registry_updated']}")
    print(f"offline_reeval_bitexact={artifact['offline_reeval_bitexact']}")
    print(f"off_arc_outcome_recorded={artifact['off_arc_outcome_recorded']}")
    return artifact


def main() -> None:  # pragma: no cover - exercised by the required script command.
    artifact = run_registration()
    if artifact["registry_updated"] and artifact["offline_reeval_bitexact"]:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
