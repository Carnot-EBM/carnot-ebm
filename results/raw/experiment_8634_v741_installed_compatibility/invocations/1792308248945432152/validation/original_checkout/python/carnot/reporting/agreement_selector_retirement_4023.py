"""Build the Exp 4023 GAP-4 agreement-selector retirement artifact.

Spec refs: REQ-VERIFY-4023, SCENARIO-VERIFY-4023.

This is an accounting module, not a new ARC run. The point is to preserve the
useful shipped product while closing the failed research question honestly:
demo-fit execution remains a trust gate, but independent-program agreement is
only a confidence label. The code reads the source artifacts and ops notes so
the closure cannot silently turn a non-executed confirmation into a positive
selector claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4023_retire_agreement_selector.json")
EXPERIMENT_ID = 4023
RANDOM_SEED = 4023
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

CHAIN_ARMS_REL_PATH = Path("results/arc3_gap4_chain_arms_adversarial_verify.json")
EXP3999_REL_PATH = Path("results/experiment_3999_gap4_precision_confirmation_v2.json")
EXP4009_REL_PATH = Path("results/experiment_4009_gap4_precision_confirmation_v3.json")
ROADMAP370_REL_PATH = Path("openspec/change-proposals/research-roadmap-v370.md")
KNOWN_ISSUES_REL_PATH = Path("ops/known-issues.md")
REGISTRY_REL_PATH = Path("ops/verifier_registry.yaml")
GAPS_REL_PATH = Path("ops/verifier_gaps.md")

HONEST_VERDICT = "complete: agreement_selector_retired_confidence_label_only"
AGREEMENT_ROLE = "confidence_label_only"
RETIRED_LINE = "smart_selector_agreement_precision_confirmation"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "registry_updated",
    "safety_gate_kept",
    "inference_substrate",
    "agreement_is_precision_selector",
    "agreement_role_after_retirement",
    "retired_r_and_d_line",
    "no_precision_confirmation_v4_proposed",
    "retire_if_same_verdict_triggered",
    "evidence_chain",
    "registry_entry",
    "gaps_entry",
    "field_principles",
    "duration_s",
    "random_seed",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix verdict prevents the conductor from retrying a completed negative "
        "scientific closure."
    ),
    "registry_updated": (
        "The registry is the durable verifier source of truth; the selector retirement must "
        "be recorded there, not only in a transient result."
    ),
    "safety_gate_kept": (
        "Retiring selector R&D must not delete the shipped trust product: demo-fit execution "
        "continues to gate generated programs."
    ),
    "inference_substrate": (
        "This task aggregates upstream artifacts and ops notes; it does not run fresh LLM or "
        "ARC inference."
    ),
    "agreement_is_precision_selector": (
        "The retired claim is specifically selector precision; keeping this false prevents "
        "agreement from being reintroduced as a gate."
    ),
    "agreement_role_after_retirement": (
        "Agreement can still label confidence for routing, which is weaker than selecting or "
        "promoting candidates."
    ),
    "retired_r_and_d_line": (
        "Names the research line being closed so the shipped demo-fit gate is not confused "
        "with the failed selector experiment."
    ),
    "no_precision_confirmation_v4_proposed": (
        "The retire_if_same_verdict trigger fired; proposing another confirmation would evade "
        "the retirement rule."
    ),
    "retire_if_same_verdict_triggered": (
        "Records the failed-experiment discipline trigger that converts repeated non-execution "
        "into retirement instead of another rerun."
    ),
    "evidence_chain": (
        "Lists the source artifacts and ops notes so the closure can be audited without "
        "trusting the summary text."
    ),
    "registry_entry": (
        "Captures the exact registry semantics expected for gap4_program_induction_stack."
    ),
    "gaps_entry": "Captures the verifier-gap closure semantics expected in ops/verifier_gaps.md.",
    "field_principles": "Echoes why each required field matters for future audit.",
    "duration_s": "Measured wall time for this aggregation pass.",
    "random_seed": "Stable experiment id seed for reproducible artifact identity.",
}


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - defensive malformed artifact guard
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def registry_has_closure(registry_text: str) -> bool:
    required_tokens = (
        "verifier_id: gap4_program_induction_stack",
        "agreement_role: confidence_label_only",
        "agreement_precision_selector: false",
        "selector_retirement:",
        "retire_if_same_verdict_triggered: true",
        "safety_gate_kept: true",
        "no_precision_confirmation_v4_proposed: true",
        "not a precision selector",
    )
    return all(token in registry_text for token in required_tokens)


def gaps_have_closure(gaps_text: str) -> bool:
    required_tokens = (
        "GAP-4 Agreement Selector Closure (Exp 4023)",
        "CONFIDENCE LABEL ONLY",
        "not a precision selector",
        "shipped demo-fit execution safety-gate is KEPT",
        "no precision-confirmation v4 is proposed",
    )
    return all(token in gaps_text for token in required_tokens)


def _int_value(payload: JsonDict, field: str) -> int:
    value = payload.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _collect_evidence(root: Path) -> list[JsonDict]:
    chain = read_json_object(root / CHAIN_ARMS_REL_PATH)
    exp3999 = read_json_object(root / EXP3999_REL_PATH)
    exp4009 = read_json_object(root / EXP4009_REL_PATH)
    roadmap370 = read_text(root / ROADMAP370_REL_PATH)
    known_issues = read_text(root / KNOWN_ISSUES_REL_PATH)

    chain_verdict = str(chain.get("honest_verdict", ""))
    if "precision_uplift_not_established" not in chain_verdict:
        raise ValueError("chain-arms report must record precision uplift not established")
    if "exp3988" not in roadmap370 or "poison-skipped" not in roadmap370:
        raise ValueError("roadmap v370 must record the exp3988 poison-skipped non-execution")

    calls3999 = _int_value(exp3999, "total_codex_calls")
    events3999 = _int_value(exp3999, "n_agreement_events")
    verdict3999 = str(exp3999.get("honest_verdict", ""))
    if calls3999 != 0 or events3999 != 0 or "pending_execution" not in verdict3999:
        raise ValueError("Exp 3999 must be the zero-call pending-execution non-execution")

    calls4009 = _int_value(exp4009, "total_codex_calls")
    events4009 = _int_value(exp4009, "n_agreement_events")
    if calls4009 != 0 or events4009 != 0 or exp4009.get("execution_floor_met") is not False:
        raise ValueError("Exp 4009 must be the zero-call execution-floor non-execution")
    if "POWERED MULTI-CODEX-CALL EXPERIMENTS MUST BE TASK-SPLIT" not in known_issues:
        raise ValueError("known-issues must record the unfeedable powered-task finding")

    return [
        {
            "source": str(CHAIN_ARMS_REL_PATH),
            "evidence_type": "adversarial_report",
            "finding": "precision_uplift_not_established_agreement_confidence_label_only",
            "honest_verdict": chain_verdict,
        },
        {
            "source": str(ROADMAP370_REL_PATH),
            "evidence_type": "non_execution",
            "experiment_id": 3988,
            "finding": "poison_skipped_precision_confirmation_never_executed",
        },
        {
            "source": str(EXP3999_REL_PATH),
            "evidence_type": "non_execution",
            "experiment_id": 3999,
            "finding": "protocol_preregistered_pending_execution_zero_calls_zero_agreement_events",
            "total_codex_calls": calls3999,
            "n_agreement_events": events3999,
        },
        {
            "source": str(EXP4009_REL_PATH),
            "evidence_type": "non_execution",
            "experiment_id": 4009,
            "finding": "execution_floor_unmet_zero_calls_zero_agreement_events",
            "total_codex_calls": calls4009,
            "n_agreement_events": events4009,
            "execution_floor_met": False,
        },
        {
            "source": str(KNOWN_ISSUES_REL_PATH),
            "evidence_type": "unfeedable_power_finding",
            "finding": "monolithic_powered_multi_call_confirmation_unfeedable_without_task_split",
        },
    ]


def _registry_entry() -> JsonDict:
    return {
        "path": str(REGISTRY_REL_PATH),
        "verifier_id": "gap4_program_induction_stack",
        "agreement_role": AGREEMENT_ROLE,
        "agreement_precision_selector": False,
        "retirement_rationale": (
            "Chain-arms narrowed agreement to a confidence label, then exp3988, exp3999, "
            "and exp4009 failed to produce the powered confirmation required to promote it."
        ),
        "retire_if_same_verdict_triggered": True,
        "safety_gate_kept": True,
        "no_precision_confirmation_v4_proposed": True,
    }


def _gaps_entry() -> JsonDict:
    return {
        "path": str(GAPS_REL_PATH),
        "gap": "GAP-4 Agreement Selector Closure (Exp 4023)",
        "status": "retired_as_selector_r_and_d",
        "agreement_role": AGREEMENT_ROLE,
        "safety_gate_kept": True,
    }


def build_artifact(
    root: str | Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    evidence_chain = _collect_evidence(root_path)
    registry_text = read_text(root_path / REGISTRY_REL_PATH)
    gaps_text = read_text(root_path / GAPS_REL_PATH)
    end = time.perf_counter() if now_s is None else now_s

    artifact: JsonDict = {
        "honest_verdict": HONEST_VERDICT,
        "registry_updated": registry_has_closure(registry_text),
        "safety_gate_kept": gaps_have_closure(gaps_text),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "agreement_is_precision_selector": False,
        "agreement_role_after_retirement": AGREEMENT_ROLE,
        "retired_r_and_d_line": RETIRED_LINE,
        "no_precision_confirmation_v4_proposed": True,
        "retire_if_same_verdict_triggered": True,
        "evidence_chain": evidence_chain,
        "registry_entry": _registry_entry(),
        "gaps_entry": _gaps_entry(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(end - start, 6),
        "random_seed": RANDOM_SEED,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:  # pragma: no cover - defensive schema guard
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact.get("honest_verdict") != HONEST_VERDICT:  # pragma: no cover - defensive schema guard
        raise ValueError("honest_verdict must close agreement selector with terminal prefix")
    if artifact.get("registry_updated") is not True:  # pragma: no cover - defensive schema guard
        raise ValueError("registry_updated must be true")
    if artifact.get("safety_gate_kept") is not True:  # pragma: no cover - defensive schema guard
        raise ValueError("safety_gate_kept must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:  # pragma: no cover
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("agreement_is_precision_selector") is not False:  # pragma: no cover
        raise ValueError("agreement must not remain a precision selector")
    if artifact.get("agreement_role_after_retirement") != AGREEMENT_ROLE:  # pragma: no cover
        raise ValueError("agreement role must be confidence_label_only")
    if artifact.get("retired_r_and_d_line") != RETIRED_LINE:  # pragma: no cover
        raise ValueError("retired_r_and_d_line names the wrong line")
    if artifact.get("no_precision_confirmation_v4_proposed") is not True:  # pragma: no cover
        raise ValueError("precision-confirmation v4 must not be proposed")
    if artifact.get("retire_if_same_verdict_triggered") is not True:  # pragma: no cover
        raise ValueError("retire_if_same_verdict trigger must be recorded")
    if artifact.get("random_seed") != RANDOM_SEED:  # pragma: no cover - defensive schema guard
        raise ValueError("random_seed must be 4023")

    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration < 0:
        # pragma: no cover - defensive schema guard
        raise ValueError("duration_s must be a non-negative number")

    evidence = artifact.get("evidence_chain")
    if not isinstance(evidence, list) or len(evidence) != 5:  # pragma: no cover
        raise ValueError("evidence_chain must contain the five closure evidence records")
    if [row.get("experiment_id") for row in evidence if "experiment_id" in row] != [3988, 3999, 4009]:
        # pragma: no cover - defensive schema guard
        raise ValueError("evidence_chain must include the three non-executions in order")

    registry_entry = artifact.get("registry_entry")
    if not isinstance(registry_entry, dict) or registry_entry != _registry_entry():
        # pragma: no cover - defensive schema guard
        raise ValueError("registry_entry must match the Exp 4023 registry contract")
    gaps_entry = artifact.get("gaps_entry")
    if not isinstance(gaps_entry, dict) or gaps_entry != _gaps_entry():
        # pragma: no cover - defensive schema guard
        raise ValueError("gaps_entry must match the Exp 4023 gaps contract")

    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):  # pragma: no cover - defensive schema guard
        raise ValueError("field_principles must be a dict")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:  # pragma: no cover - defensive schema guard
        raise ValueError(f"field_principles missing fields: {sorted(missing_principles)}")


def write_artifact(root: str | Path, artifact: JsonDict) -> Path:
    validate_artifact(artifact)
    output_path = Path(root) / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_experiment(root: str | Path = REPO_ROOT) -> Path:
    artifact = build_artifact(root)
    return write_artifact(root, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper
    output_path = run_experiment()
    artifact = read_json_object(output_path)
    print(f"artifact: {output_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"registry_updated: {artifact['registry_updated']}")
    print(f"safety_gate_kept: {artifact['safety_gate_kept']}")


if __name__ == "__main__":  # pragma: no cover
    main()
