"""Apply low-risk verifier-authenticity remediation for Exp 5218.

Spec refs: REQ-VERIFY-5218, SCENARIO-VERIFY-5218.
"""

from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260704"
SCHEMA = "carnot.experiment_5218_verifier_authenticity_remediation_apply.v477"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5218_verifier_authenticity_remediation_apply_v477.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
REMEDIATED_MODULES = (
    "python/carnot/verify/and_composition_verifier.py",
    "python/carnot/verify/claim_isolation_uncertainty_router.py",
)
SPEC_REFS = ("REQ-VERIFY-5218", "SCENARIO-VERIFY-5218")
ALLOWED_REMEDIATION_TYPES = {
    "rename",
    "warning",
    "registry_flag",
    "behavior_fix",
    "docs_only",
    "blocked",
}
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")

FIELD_PRINCIPLES = {
    "remediation_applied": (
        "True only when both flagged modules expose truthful non-headline authenticity metadata."
    ),
    "remediated_modules": "Exact flagged module paths remediated by this apply step.",
    "remediation_type": "One of rename | warning | registry_flag | behavior_fix | docs_only | blocked.",
    "headline_ineligible_until_real_verification": (
        "Both remediated surfaces must remain headline-ineligible until a real trained/live "
        "verification substrate is implemented."
    ),
    "tests_run": "Commands run for this apply step, with pass/fail status.",
    "specs_updated": (
        "True when REQ-VERIFY-5218 and SCENARIO-VERIFY-5218 are present before implementation."
    ),
    "no_research_conductor_change": (
        "Must remain true; scripts/research_conductor.py is outside this remediation."
    ),
    "inference_substrate": "Must be code_and_doc_remediation.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_ and state whether "
        "the dishonest-naming risk is actually reduced."
    ),
}


def inspect_remediation(
    *,
    and_module: Any | None = None,
    router_module: Any | None = None,
) -> JsonDict:
    """Inspect the two flagged modules for the REQ-VERIFY-5218 quarantine flags."""

    if and_module is None:
        and_module = import_module("carnot.verify.and_composition_verifier")
    if router_module is None:
        router_module = import_module("carnot.verify.claim_isolation_uncertainty_router")

    and_reason = str(getattr(and_module, "HEADLINE_INELIGIBLE_REASON", ""))
    router_reason = str(getattr(router_module, "HEADLINE_INELIGIBLE_REASON", ""))
    and_ok = bool(
        getattr(and_module, "AUTHENTICITY_REMEDIATION_TYPE", None) == "registry_flag"
        and getattr(and_module, "AUTHENTICITY_STATUS", None) == "advisory_adapter_harness"
        and getattr(and_module, "HEADLINE_ELIGIBLE", None) is False
        and "untrained" in and_reason.lower()
        and "advisory" in and_reason.lower()
    )
    router_ok = bool(
        getattr(router_module, "AUTHENTICITY_REMEDIATION_TYPE", None) == "registry_flag"
        and getattr(router_module, "AUTHENTICITY_STATUS", None) == "artifact_routing_ledger"
        and getattr(router_module, "HEADLINE_ELIGIBLE", None) is False
        and getattr(router_module, "LIVE_ISOLATED_CLAIM_VERIFICATION", None) is False
        and "artifact" in router_reason.lower()
        and "no live isolated-claim verifier call" in router_reason.lower()
    )

    return {
        "and_composition_verifier": {
            "remediated": and_ok,
            "authenticity_status": getattr(and_module, "AUTHENTICITY_STATUS", None),
            "headline_eligible": getattr(and_module, "HEADLINE_ELIGIBLE", None),
            "headline_ineligible_reason": and_reason,
        },
        "claim_isolation_uncertainty_router": {
            "remediated": router_ok,
            "authenticity_status": getattr(router_module, "AUTHENTICITY_STATUS", None),
            "headline_eligible": getattr(router_module, "HEADLINE_ELIGIBLE", None),
            "live_isolated_claim_verification": getattr(
                router_module,
                "LIVE_ISOLATED_CLAIM_VERIFICATION",
                None,
            ),
            "headline_ineligible_reason": router_reason,
        },
    }


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: list[str] | tuple[str, ...] | None = None,
    inspection: JsonDict | None = None,
    spec_path: Path | str = SPEC_RELATIVE_PATH,
) -> JsonDict:
    """Build the terminal Exp 5218 artifact from current source metadata."""

    current_inspection = inspect_remediation() if inspection is None else inspection
    applied = bool(
        current_inspection["and_composition_verifier"]["remediated"]
        and current_inspection["claim_isolation_uncertainty_router"]["remediated"]
    )
    specs_updated = _specs_updated(Path(spec_path))
    remediation_type = "registry_flag" if applied else "blocked"
    headline_ineligible = bool(
        applied
        and current_inspection["and_composition_verifier"]["headline_eligible"] is False
        and current_inspection["claim_isolation_uncertainty_router"]["headline_eligible"] is False
        and current_inspection["claim_isolation_uncertainty_router"][
            "live_isolated_claim_verification"
        ]
        is False
    )
    honest_verdict = (
        "complete: dishonest-naming risk reduced by registry flags; modules remain "
        "headline-ineligible until real verification"
        if applied and specs_updated
        else "blocked_verifier_authenticity_remediation_incomplete"
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": "experiment_5218_verifier_authenticity_remediation_apply_v477",
        "experiment_id": "exp5218-verifier-authenticity-remediation-apply-v477",
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "source_artifacts_read": [
            "results/experiment_5203_verifier_authenticity_remediation_options_v476.json",
            "ops/verifier_remediation_options_v476.md",
            "ops/verifier_authenticity_audit_report.md",
            *REMEDIATED_MODULES,
            str(SPEC_RELATIVE_PATH),
        ],
        "remediation_inspection": current_inspection,
        "remediation_applied": _wrap("remediation_applied", applied),
        "remediated_modules": _wrap("remediated_modules", list(REMEDIATED_MODULES)),
        "remediation_type": _wrap("remediation_type", remediation_type),
        "headline_ineligible_until_real_verification": _wrap(
            "headline_ineligible_until_real_verification",
            headline_ineligible,
        ),
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "specs_updated": _wrap("specs_updated", specs_updated),
        "no_research_conductor_change": _wrap("no_research_conductor_change", True),
        "inference_substrate": _wrap("inference_substrate", "code_and_doc_remediation"),
        "honest_verdict": _wrap("honest_verdict", honest_verdict),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the REQ-VERIFY-5218 terminal artifact shape and honesty fields."""

    missing = [field for field in FIELD_PRINCIPLES if field not in artifact]
    assert not missing, f"missing required fields: {missing}"
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = artifact[field]
        assert (
            isinstance(wrapped, dict)
            and wrapped.get("principle") == principle
            and "value" in wrapped
        ), f"{field} must be principle-wrapped"
    assert artifact["remediation_type"]["value"] in ALLOWED_REMEDIATION_TYPES
    assert artifact["inference_substrate"]["value"] == "code_and_doc_remediation", (
        "inference_substrate must be code_and_doc_remediation"
    )
    assert artifact["no_research_conductor_change"]["value"] is True
    assert str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES)
    assert (
        not artifact["remediation_applied"]["value"]
        or artifact["headline_ineligible_until_real_verification"]["value"] is True
    )


def run_experiment(
    *,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: list[str] | tuple[str, ...] | None = None,
) -> JsonDict:
    """Write the Exp 5218 JSON artifact and return it."""

    started = time.perf_counter()
    elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    artifact = build_artifact(run_date=run_date, duration_s=elapsed, tests_run=tests_run)
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _specs_updated(spec_path: Path) -> bool:
    text = spec_path.read_text(encoding="utf-8")
    return all(ref in text for ref in SPEC_REFS)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
