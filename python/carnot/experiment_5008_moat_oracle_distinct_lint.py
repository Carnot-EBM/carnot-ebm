#!/usr/bin/env python3
"""Exp 5008: moat oracle-distinct lint deliverable.

Spec refs: REQ-VERIFY-5008, SCENARIO-VERIFY-5008.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "experiment_5008_moat_oracle_distinct_lint.json"
RANDOM_SEED = 20260630
CHECK_FUNCTION_NAME = "check_moat_claim_rigor"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
D_ARM_PATHS = [
    REPO_ROOT / "results" / "experiment_5003_lora_ebm_scorer_musr.json",
    REPO_ROOT / "results" / "experiment_5004_uprm_replication.json",
    REPO_ROOT / "results" / "experiment_5005_ebrm_uncertainty_verifier.json",
    REPO_ROOT / "results" / "experiment_5006_moat_second_corpus.json",
]

RULES_IMPLEMENTED = [
    {
        "rule": "a",
        "severity": "critical",
        "contract": "verifier-moat / beats-SC / verifier_value_added claims require verifier_is_oracle=False.",
    },
    {
        "rule": "b",
        "severity": "critical",
        "contract": "positive delta wins require headroom_present=True with oracle@K - tuned_sc >= 0.10 and flips>0 when those fields are present.",
    },
    {
        "rule": "c",
        "severity": "warn",
        "contract": "naive self-consistency comparisons are not a tuned-SC headroom-control baseline.",
    },
    {
        "rule": "d",
        "severity": "critical",
        "contract": "positive beats-SC wins require paired_ci95 and mcnemar_p significance evidence.",
    },
    {
        "rule": "e",
        "severity": "warn",
        "contract": "no-headroom null or moat-retirement claims are uninformative and must not be mislabeled as a moat bound.",
    },
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success_moat_rigor_lint_shipped_fixtures_green."
    },
    "check_function_name": {
        "principle": "check_moat_claim_rigor (the added adversarial_verify check)."
    },
    "rules_implemented": {
        "principle": "the (a)-(e) rule list -- the mechanical contract enforcing oracle-distinctness + headroom-control + paired-CI on moat claims."
    },
    "fixtures_passed": {
        "principle": "each synthetic fixture (clean/circular/no-headroom/naive-SC/no-CI) fires the expected severity (the test evidence)."
    },
    "d_arms_lint_findings": {
        "principle": "advisory findings on the existing D1-D4 artifacts (does not block them; surfaces any residual over-claim)."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads artifacts + runs a lint; no LLM)."
    },
    "random_seed": {"principle": "determinism for the fixture generation."},
    "preconditions_checked": {
        "principle": "records adversarial_verify-importable check; a missing module emits blocked_."
    },
}


def _base_win_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "synthetic_moat_rigor_clean",
        "honest_verdict": "success_verifier_moat_beats_sc_musr_0p120",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "oracle_at_k": 0.82,
        "tuned_sc_accuracy": 0.60,
        "delta_vs_tuned_sc": 0.12,
        "n_flips_possible": 7,
        "paired_ci95": [0.03, 0.20],
        "mcnemar_p": 0.01,
        "random_seed": RANDOM_SEED,
    }
    payload.update(overrides)
    return payload


def _fixture_payloads() -> dict[str, tuple[dict[str, Any], str | None]]:
    naive = _base_win_fixture(
        honest_verdict="success_verifier_moat_beats_naive_sc_musr_0p120",
        naive_sc_accuracy=0.60,
        delta_vs_naive_sc=0.12,
    )
    naive.pop("tuned_sc_accuracy")
    naive.pop("delta_vs_tuned_sc")

    no_ci = _base_win_fixture()
    no_ci.pop("paired_ci95")
    no_ci.pop("mcnemar_p")

    no_headroom_null = {
        "experiment": "synthetic_moat_rigor_no_headroom_null",
        "honest_verdict": "complete_moat_retired_bounded_does_not_beat_sc",
        "verifier_is_oracle": False,
        "headroom_present": False,
        "oracle_at_k": 0.60,
        "tuned_sc_accuracy": 0.60,
        "delta_vs_tuned_sc": 0.0,
        "n_flips_possible": 0,
        "paired_ci95": [-0.03, 0.02],
        "mcnemar_p": 1.0,
        "random_seed": RANDOM_SEED,
    }

    return {
        "clean": (_base_win_fixture(), None),
        "circular": (_base_win_fixture(verifier_is_oracle=True), "critical"),
        "no_headroom_win": (
            _base_win_fixture(
                headroom_present=False,
                oracle_at_k=0.65,
                tuned_sc_accuracy=0.60,
                n_flips_possible=0,
            ),
            "critical",
        ),
        "naive_sc": (naive, "warn"),
        "no_ci": (no_ci, "critical"),
        "no_headroom_null": (no_headroom_null, "warn"),
    }


def _import_adversarial_verify() -> tuple[Any | None, str | None]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import scripts.adversarial_verify as adversarial_verify
    except Exception as exc:  # pragma: no cover - blocked artifact path.
        return None, repr(exc)
    return adversarial_verify, None


def _write_artifact(artifact: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flag_dicts(flags: list[Any]) -> list[dict[str, Any]]:
    return [flag.to_dict() for flag in flags if getattr(flag, "kind", None) == "MOAT_CLAIM_RIGOR"]


def _run_fixture_checks(adversarial_verify: Any) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, (payload, expected_severity) in _fixture_payloads().items():
        flags: list[Any] = []
        adversarial_verify.check_moat_claim_rigor(payload, flags)
        moat_flags = _flag_dicts(flags)
        severities = sorted({flag["severity"] for flag in moat_flags})
        if expected_severity is None:
            passed = not moat_flags
        else:
            passed = expected_severity in severities
        results[name] = {
            "expected": "pass" if expected_severity is None else expected_severity,
            "actual_severities": severities,
            "flags": moat_flags,
            "passed": passed,
        }
    return results


def _lint_d_arms(adversarial_verify: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in D_ARM_PATHS:
        if not path.exists():
            findings.append(
                {
                    "artifact": str(path.relative_to(REPO_ROOT)),
                    "loaded": False,
                    "moat_claim_rigor_flags": [],
                    "error": "missing",
                }
            )
            continue
        report = adversarial_verify.verify_artifact(path)
        moat_flags = [
            flag for flag in report.get("flags", []) if flag.get("kind") == "MOAT_CLAIM_RIGOR"
        ]
        findings.append(
            {
                "artifact": str(path.relative_to(REPO_ROOT)),
                "loaded": report.get("loaded", False),
                "moat_claim_rigor_flags": moat_flags,
                "moat_claim_rigor_flag_count": len(moat_flags),
            }
        )
    return findings


def main() -> int:
    started = time.perf_counter()
    module_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    adversarial_verify, import_error = _import_adversarial_verify()
    preconditions = {
        "adversarial_verify_present": module_path.exists(),
        "adversarial_verify_importable": adversarial_verify is not None,
        "import_error": import_error,
        "check_function_present": False,
    }

    if adversarial_verify is None or not hasattr(adversarial_verify, CHECK_FUNCTION_NAME):
        preconditions["check_function_present"] = False
        artifact = {
            "experiment": "experiment_5008_moat_oracle_distinct_lint",
            "honest_verdict": "blocked_adversarial_verify_importable_check_failed",
            "check_function_name": CHECK_FUNCTION_NAME,
            "rules_implemented": RULES_IMPLEMENTED,
            "fixtures_passed": {},
            "d_arms_lint_findings": [],
            "inference_substrate": INFERENCE_SUBSTRATE,
            "random_seed": RANDOM_SEED,
            "preconditions_checked": preconditions,
            "duration_s": time.perf_counter() - started,
            "field_principles": FIELD_PRINCIPLES,
            "spec_refs": ["REQ-VERIFY-5008", "SCENARIO-VERIFY-5008"],
        }
        _write_artifact(artifact)
        return 1

    preconditions["check_function_present"] = True
    fixtures_passed = _run_fixture_checks(adversarial_verify)
    d_arm_findings = _lint_d_arms(adversarial_verify)
    all_fixtures_passed = all(row["passed"] for row in fixtures_passed.values())

    artifact = {
        "experiment": "experiment_5008_moat_oracle_distinct_lint",
        "honest_verdict": (
            "success_moat_rigor_lint_shipped_fixtures_green."
            if all_fixtures_passed
            else "blocked_moat_rigor_lint_fixture_expectation_failed"
        ),
        "check_function_name": CHECK_FUNCTION_NAME,
        "rules_implemented": RULES_IMPLEMENTED,
        "fixtures_passed": fixtures_passed,
        "d_arms_lint_findings": d_arm_findings,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions,
        "duration_s": time.perf_counter() - started,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-5008", "SCENARIO-VERIFY-5008"],
    }
    _write_artifact(artifact)
    return 0 if all_fixtures_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
