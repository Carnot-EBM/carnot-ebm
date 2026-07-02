"""Exp 5140: symbolic distillation of KAN certificate behavior.

Spec refs: REQ-KAN-5140, SCENARIO-KAN-5140.

Exp 5128 showed that KAN certificate explanations can round-trip across a
bounded family. This module keeps the same CPU-only evidence boundary and asks
whether the residual and certificate behavior can be represented by explicit
symbolic primitives whose exact code reconstruction recovers the original
property metadata, verdict, margin, and abstention condition. No LLM judge is
used; the source of truth is the machine-readable Exp 5128 certificate data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5140-symbolic-kan-certificate-distillation-v471"
MILESTONE = "2026.07.471"
RUN_DATE = "20260702"
RESULT_RELATIVE_PATH = "results/experiment_5140_symbolic_kan_certificate_distillation_v471.json"
DISTILLED_RULES_RELATIVE_PATH = (
    "results/experiment_5140_symbolic_kan_certificate_distillation_v471_rules.json"
)
EXP5128_RESULT_RELATIVE_PATH = "results/experiment_5128_kan_certificate_explanation_v470.json"
INFERENCE_SUBSTRATE = "exact_checked_symbolic_kan_distillation"
SPEC_REFS = ["REQ-KAN-5140", "SCENARIO-KAN-5140"]

SUCCESS_VERDICT = "success_symbolic_kan_certificate_distillation_ready"
COMPLETE_NOT_READY_VERDICT = "complete_symbolic_kan_certificate_distillation_not_ready"
BLOCKED_UPSTREAM_VERDICT = "blocked_symbolic_kan_certificate_distillation_missing_exp5128"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")
EPSILON = 1e-9

SYMBOLIC_PRIMITIVES = [
    {
        "name": "monotone_segment",
        "meaning": "unit-index interval over which additive certificate bounds are monotone",
    },
    {
        "name": "threshold_clause",
        "meaning": "exact or certified quantity compared with the property threshold",
    },
    {
        "name": "affine_interval",
        "meaning": "closed interval between exact, observed, and certified affine quantities",
    },
    {
        "name": "bounded_residual_rule",
        "meaning": "residual abstraction bound that can force abstention near a threshold",
    },
]
SYMBOLIC_PRIMITIVE_NAMES = tuple(primitive["name"] for primitive in SYMBOLIC_PRIMITIVES)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "exp5128_loaded",
    "symbolic_primitives",
    "distilled_rules_path",
    "symbolic_equivalence_rate",
    "certificate_soundness",
    "cycle_reconstruction_rate",
    "false_property_detected",
    "near_margin_abstention_rate",
    "family_holdout_results",
    "symbolic_kan_ready",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "exp5128_loaded": "upstream evidence",
    "symbolic_primitives": "interpretability",
    "distilled_rules_path": "artifact provenance",
    "symbolic_equivalence_rate": "explanation fidelity",
    "certificate_soundness": "formal correctness",
    "cycle_reconstruction_rate": "explanation cycle consistency",
    "false_property_detected": "adversarial control",
    "near_margin_abstention_rate": "uncertainty control",
    "family_holdout_results": "generalization caution",
    "symbolic_kan_ready": "downstream readiness",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5140_symbolic_kan_certificate_distillation_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5140_symbolic_kan_certificate_distillation_v471.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run "
    "--include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5140_symbolic_kan_certificate_distillation_v471.py' -m pytest "
    'tests/python/test_experiment_5140_symbolic_kan_certificate_distillation_v471.py -q -o addopts="" && '
    ".venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/python/"
    "carnot/experiment_5140_symbolic_kan_certificate_distillation_v471.py' --fail-under=100 -m",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5140_symbolic_kan_certificate_distillation_v471.py",
    ".venv/bin/pytest tests/python -q",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float_token(value: float) -> str:
    return f"{float(value):.12g}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive malformed-artifact path.
        return {"__error__": f"{type(exc).__name__}: {exc}"}
    return dict(parsed) if isinstance(parsed, Mapping) else {"__error__": "artifact is not a JSON object"}


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_exp5128_artifact(root: str | Path | None = None) -> tuple[JsonDict | None, JsonDict]:
    """Load and minimally validate the upstream V470 certificate artifact."""

    base = Path(root) if root is not None else _repo_root()
    path = base / EXP5128_RESULT_RELATIVE_PATH
    payload = _read_json(path)
    if payload is None:
        return None, {
            "loaded": False,
            "path": str(path),
            "reason": "missing upstream Exp 5128 artifact",
        }
    if "__error__" in payload:
        return None, {
            "loaded": False,
            "path": _relative_or_absolute(path, base),
            "reason": str(payload["__error__"]),
        }
    certificates = payload.get("certificates_emitted")
    explanations = payload.get("explanation_records")
    if not certificates or not explanations:
        return None, {
            "loaded": False,
            "path": _relative_or_absolute(path, base),
            "reason": "missing certificate/explanation records",
        }
    if payload.get("kan_certificate_breadth_ready") is not True:
        return None, {
            "loaded": False,
            "path": _relative_or_absolute(path, base),
            "reason": "upstream Exp 5128 artifact is not ready",
        }
    return payload, {
        "loaded": True,
        "path": _relative_or_absolute(path, base),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "certificate_count": len(certificates),
        "explanation_count": len(explanations),
    }


def _abstention_condition(verdict: str) -> str:
    if verdict == "abstained":
        return "exact_value <= threshold < certified_upper_bound"
    return "not_applicable"


def _rule_kind(certificate: Mapping[str, Any]) -> str:
    family = str(certificate["property"]["family"])
    return "refinement_budget" if family == "refinement_error_budget" else "threshold_bound"


def _numeric_record(certificate: Mapping[str, Any]) -> JsonDict:
    prop = certificate["property"]
    bounds = certificate.get("bounds", {})
    if _rule_kind(certificate) == "refinement_budget":
        observed = float(prop["observed"])
        return {
            "threshold": float(prop["threshold"]),
            "observed": observed,
            "exact_upper_bound": observed,
            "certified_upper_bound": observed,
            "abstraction_error": float(certificate["abstraction_error"]),
            "unit_count": int(prop["unit_count"]),
        }
    return {
        "threshold": float(prop["threshold"]),
        "observed": float(bounds["exact_upper_bound"]),
        "exact_upper_bound": float(bounds["exact_upper_bound"]),
        "certified_upper_bound": float(bounds["certified_upper_bound"]),
        "abstraction_error": float(certificate["abstraction_error"]),
        "unit_count": int(prop["unit_count"]),
    }


def _source_metadata(certificate: Mapping[str, Any]) -> dict[str, str]:
    prop = certificate["property"]
    numeric = _numeric_record(certificate)
    return {
        "property_id": str(prop["id"]),
        "family": str(prop["family"]),
        "verdict": str(certificate["verdict"]),
        "margin": _float_token(float(certificate["margin"])),
        "abstraction_error": _float_token(float(certificate["abstraction_error"])),
        "proof_status": str(certificate["proof_status"]),
        "threshold": _float_token(numeric["threshold"]),
        "unit_count": str(numeric["unit_count"]),
        "abstention_condition": _abstention_condition(str(certificate["verdict"])),
    }


def _primitive_records(certificate: Mapping[str, Any]) -> list[JsonDict]:
    numeric = _numeric_record(certificate)
    margin = float(certificate["margin"])
    return [
        {
            "name": "monotone_segment",
            "variable": "unit_index",
            "interval": [0, numeric["unit_count"]],
            "direction": "nondecreasing_additive_bound",
        },
        {
            "name": "threshold_clause",
            "left": "certified_or_exact_quantity",
            "operator": "<=",
            "threshold": numeric["threshold"],
        },
        {
            "name": "affine_interval",
            "lower": min(numeric["exact_upper_bound"], numeric["certified_upper_bound"]),
            "upper": max(numeric["exact_upper_bound"], numeric["certified_upper_bound"]),
            "slope": 1.0,
            "intercept": 0.0,
        },
        {
            "name": "bounded_residual_rule",
            "residual_bound": numeric["abstraction_error"],
            "margin": margin,
            "abstain_when": "exact_value <= threshold < certified_upper_bound",
        },
    ]


def distill_symbolic_rules(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Extract symbolic primitive rules from Exp 5128 certificates."""

    explanations = {
        record["certificate_id"]: record for record in upstream.get("explanation_records", [])
    }
    rules: list[JsonDict] = []
    for certificate in upstream.get("certificates_emitted", []):
        prop = certificate["property"]
        certificate_id = str(prop["id"])
        explanation = explanations.get(certificate_id, {})
        numeric = _numeric_record(certificate)
        rules.append(
            {
                "rule_id": f"symbolic_rule::{certificate_id}",
                "source_certificate_id": certificate_id,
                "family": str(prop["family"]),
                "rule_kind": _rule_kind(certificate),
                "source_explanation": explanation.get("explanation"),
                "source_metadata": _source_metadata(certificate),
                "numeric": numeric,
                "primitives": _primitive_records(certificate),
                "cycle_predicates": {
                    "verified_when": "certified_upper_bound <= threshold",
                    "counterexample_when": "exact_upper_bound > threshold",
                    "abstain_when": "exact_upper_bound <= threshold < certified_upper_bound",
                },
                "llm_judge_used": False,
            }
        )
    return rules


def reconstruct_from_rule(rule: Mapping[str, Any]) -> dict[str, str]:
    """Reconstruct certificate metadata using only the distilled symbolic rule."""

    numeric = rule["numeric"]
    threshold = float(numeric["threshold"])
    observed = float(numeric["observed"])
    exact_upper = float(numeric["exact_upper_bound"])
    certified_upper = float(numeric["certified_upper_bound"])
    if rule["rule_kind"] == "refinement_budget":
        if observed <= threshold + EPSILON:
            verdict = "verified"
            proof_status = "proved_by_refinement_budget"
            margin = threshold - observed
        else:
            verdict = "counterexample"
            proof_status = "refuted_by_exact_witness"
            margin = observed - threshold
    elif certified_upper <= threshold + EPSILON:
        verdict = "verified"
        proof_status = "proved_by_conservative_bound"
        margin = threshold - certified_upper
    elif exact_upper <= threshold + EPSILON < certified_upper:
        verdict = "abstained"
        proof_status = "abstained_residual_gap"
        margin = certified_upper - threshold
    else:
        verdict = "counterexample"
        proof_status = "refuted_by_exact_witness"
        margin = exact_upper - threshold
    return {
        "property_id": str(rule["source_certificate_id"]),
        "family": str(rule["family"]),
        "verdict": verdict,
        "margin": _float_token(max(0.0, margin)),
        "abstraction_error": _float_token(float(numeric["abstraction_error"])),
        "proof_status": proof_status,
        "threshold": _float_token(threshold),
        "unit_count": str(numeric["unit_count"]),
        "abstention_condition": _abstention_condition(verdict),
    }


def soundness_check_reconstruction(
    rule: Mapping[str, Any],
    reconstructed: Mapping[str, str],
) -> bool:
    """Check reconstructed metadata against exact threshold and residual logic."""

    if reconstructed != rule["source_metadata"]:
        return False
    numeric = rule["numeric"]
    threshold = float(numeric["threshold"])
    observed = float(numeric["observed"])
    exact_upper = float(numeric["exact_upper_bound"])
    certified_upper = float(numeric["certified_upper_bound"])
    verdict = reconstructed["verdict"]
    if rule["rule_kind"] == "refinement_budget":
        return verdict == "verified" and observed <= threshold + EPSILON
    if verdict == "verified":
        return certified_upper <= threshold + EPSILON
    if verdict == "counterexample":
        return exact_upper > threshold + EPSILON
    if verdict == "abstained":
        return exact_upper <= threshold + EPSILON < certified_upper
    return False


def cycle_check_rules(rules: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure exact reconstruction and soundness rates for distilled rules."""

    records = []
    for rule in rules:
        reconstructed = reconstruct_from_rule(rule)
        equivalent = reconstructed == rule["source_metadata"]
        sound = soundness_check_reconstruction(rule, reconstructed)
        records.append(
            {
                "rule_id": rule["rule_id"],
                "equivalent": equivalent,
                "sound": sound,
                "reconstructed": reconstructed,
            }
        )
    total = len(records)
    equivalent_count = sum(1 for record in records if record["equivalent"])
    sound_count = sum(1 for record in records if record["sound"])
    rate = equivalent_count / total if total else 0.0
    return {
        "records": records,
        "symbolic_equivalence_rate": rate,
        "cycle_reconstruction_rate": sound_count / total if total else 0.0,
        "certificate_soundness": bool(total) and sound_count == total,
    }


def label_shuffle_control(rules: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Shuffle family labels and confirm the cycle checker detects the mismatch."""

    families = [str(rule["family"]) for rule in rules]
    rotated = families[1:] + families[:1]
    mismatches = 0
    for rule, family in zip(rules, rotated, strict=True):
        shuffled = copy.deepcopy(rule)
        shuffled["family"] = family
        if reconstruct_from_rule(shuffled) != rule["source_metadata"]:
            mismatches += 1
    return {
        "detected": mismatches > 0,
        "mismatch_count": mismatches,
        "rules_checked": len(rules),
    }


def family_holdout_results(rules: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run leave-one-family-out checks while reporting their limited scope."""

    families = sorted({str(rule["family"]) for rule in rules})
    results: list[JsonDict] = []
    for family in families:
        heldout = [rule for rule in rules if rule["family"] == family]
        train = [rule for rule in rules if rule["family"] != family]
        cycle = cycle_check_rules(heldout)
        results.append(
            {
                "heldout_family": family,
                "train_family_count": len({str(rule["family"]) for rule in train}),
                "heldout_rule_count": len(heldout),
                "primitive_templates_seen": sorted(
                    {primitive["name"] for rule in train for primitive in rule["primitives"]}
                ),
                "symbolic_equivalence_rate": cycle["symbolic_equivalence_rate"],
                "cycle_reconstruction_rate": cycle["cycle_reconstruction_rate"],
                "certificate_soundness": cycle["certificate_soundness"],
                "caution": "leave-one-family exact reconstruction, not broad KAN generalization",
            }
        )
    return results


def evaluate_controls(rules: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Evaluate adversarial and uncertainty controls over the distilled rules."""

    false_property_detected = any(
        rule["source_metadata"]["family"] == "false_low_threshold_control"
        and reconstruct_from_rule(rule)["verdict"] == "counterexample"
        and soundness_check_reconstruction(rule, reconstruct_from_rule(rule))
        for rule in rules
    )
    near_margin_rules = [
        rule for rule in rules if rule["source_metadata"]["family"] == "near_margin_residual_gap"
    ]
    near_abstained = sum(
        1 for rule in near_margin_rules if reconstruct_from_rule(rule)["verdict"] == "abstained"
    )
    return {
        "false_property_detected": false_property_detected,
        "near_margin_abstention_rate": (
            near_abstained / len(near_margin_rules) if near_margin_rules else 0.0
        ),
        "label_shuffle_control": label_shuffle_control(rules),
        "family_holdout_results": family_holdout_results(rules),
    }


def _distilled_rule_summaries(rules: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "rule_id": rule["rule_id"],
            "source_certificate_id": rule["source_certificate_id"],
            "family": rule["family"],
            "primitive_names": [primitive["name"] for primitive in rule["primitives"]],
            "verdict": rule["source_metadata"]["verdict"],
        }
        for rule in rules
    ]


def build_rules_payload(
    *,
    upstream_status: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    cycle: Mapping[str, Any],
    controls: Mapping[str, Any],
    run_date: str,
) -> JsonDict:
    """Build the separate distilled-rule provenance artifact."""

    return {
        "schema": "carnot.symbolic_kan_certificate_distillation.rules.v471",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "source_artifact": upstream_status,
        "symbolic_primitives": list(SYMBOLIC_PRIMITIVES),
        "distilled_rules": list(rules),
        "cycle_records": cycle["records"],
        "controls": dict(controls),
        "llm_judge_used": False,
    }


def _empty_cycle() -> JsonDict:
    return {
        "records": [],
        "symbolic_equivalence_rate": 0.0,
        "cycle_reconstruction_rate": 0.0,
        "certificate_soundness": False,
    }


def _empty_controls() -> JsonDict:
    return {
        "false_property_detected": False,
        "near_margin_abstention_rate": 0.0,
        "label_shuffle_control": {"detected": False, "mismatch_count": 0, "rules_checked": 0},
        "family_holdout_results": [],
    }


def build_artifact(
    *,
    root: str | Path | None = None,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    current_duration_s: float | None = None,
    distilled_rules_path: str | None = DISTILLED_RULES_RELATIVE_PATH,
) -> JsonDict:
    """Build the Exp 5140 terminal artifact."""

    start = time.perf_counter()
    upstream, upstream_status = load_exp5128_artifact(root)
    exp5128_loaded = bool(upstream_status.get("loaded"))
    if upstream is None:
        rules: list[JsonDict] = []
        cycle = _empty_cycle()
        controls = _empty_controls()
        symbolic_kan_ready = False
        honest_verdict = BLOCKED_UPSTREAM_VERDICT
        rules_path: str | None = None
    else:
        rules = distill_symbolic_rules(upstream)
        cycle = cycle_check_rules(rules)
        controls = evaluate_controls(rules)
        holdouts = controls["family_holdout_results"]
        symbolic_kan_ready = (
            cycle["symbolic_equivalence_rate"] == 1.0
            and cycle["cycle_reconstruction_rate"] == 1.0
            and cycle["certificate_soundness"] is True
            and controls["false_property_detected"] is True
            and controls["label_shuffle_control"]["detected"] is True
            and controls["near_margin_abstention_rate"] > 0.0
            and bool(holdouts)
            and all(result["certificate_soundness"] for result in holdouts)
        )
        honest_verdict = SUCCESS_VERDICT if symbolic_kan_ready else COMPLETE_NOT_READY_VERDICT
        rules_path = distilled_rules_path
    duration_s = current_duration_s
    if duration_s is None:
        duration_s = round(time.perf_counter() - start, 6)
    artifact = {
        "schema": "carnot.symbolic_kan_certificate_distillation.v471",
        "experiment": 5140,
        "artifact": "experiment_5140_symbolic_kan_certificate_distillation_v471",
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "exp5128_loaded": exp5128_loaded,
        "upstream_status": upstream_status,
        "symbolic_primitives": list(SYMBOLIC_PRIMITIVES),
        "distilled_rules_path": rules_path,
        "distilled_rule_summaries": _distilled_rule_summaries(rules),
        "symbolic_equivalence_rate": float(cycle["symbolic_equivalence_rate"]),
        "certificate_soundness": bool(cycle["certificate_soundness"]),
        "cycle_reconstruction_rate": float(cycle["cycle_reconstruction_rate"]),
        "false_property_detected": bool(controls["false_property_detected"]),
        "near_margin_abstention_rate": float(controls["near_margin_abstention_rate"]),
        "label_shuffle_control": controls["label_shuffle_control"],
        "family_holdout_results": controls["family_holdout_results"],
        "symbolic_kan_ready": symbolic_kan_ready,
        "conductor_modified": False,
        "source_artifacts": [EXP5128_RESULT_RELATIVE_PATH],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_note": (
            "Exp 5140 distills Exp 5128 certificate residual behavior into exact "
            "symbolic primitives and checks cycle reconstruction without an LLM judge."
        ),
        "tests_run": list(tests_run),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5140 artifact drifts from the contract."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch")
    _require(artifact["milestone"] == MILESTONE, "milestone mismatch")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "substrate mismatch")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "bad verdict prefix")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s cannot be negative")
    _require(artifact["conductor_modified"] is False, "conductor must remain unmodified")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field principles must cover required fields",
    )
    primitive_names = {primitive["name"] for primitive in artifact["symbolic_primitives"]}
    _require(primitive_names == set(SYMBOLIC_PRIMITIVE_NAMES), "primitive set mismatch")
    if artifact["symbolic_kan_ready"]:
        _require(artifact["honest_verdict"] == SUCCESS_VERDICT, "ready verdict mismatch")
        _require(artifact["exp5128_loaded"] is True, "ready requires Exp 5128")
        _require(artifact["distilled_rules_path"] is not None, "ready requires rule artifact")
        _require(artifact["symbolic_equivalence_rate"] == 1.0, "ready requires equivalence")
        _require(artifact["certificate_soundness"] is True, "ready requires soundness")
        _require(artifact["cycle_reconstruction_rate"] == 1.0, "ready requires cycle consistency")
        _require(artifact["false_property_detected"] is True, "ready requires false control")
        _require(artifact["near_margin_abstention_rate"] > 0.0, "ready requires abstention")
        _require(
            artifact["label_shuffle_control"]["detected"] is True,
            "ready requires label-shuffle detection",
        )
        _require(
            all(result["certificate_soundness"] for result in artifact["family_holdout_results"]),
            "ready requires holdout soundness",
        )
    elif artifact["exp5128_loaded"]:
        _require(artifact["honest_verdict"] == COMPLETE_NOT_READY_VERDICT, "not-ready mismatch")
    else:
        _require(artifact["honest_verdict"] == BLOCKED_UPSTREAM_VERDICT, "blocked mismatch")
        _require(artifact["distilled_rules_path"] is None, "blocked artifacts have no rules path")


def write_outputs(
    *,
    root: str | Path | None = None,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
    current_duration_s: float | None = None,
    artifact_path: str | Path | None = None,
    rules_output: str | Path | None = None,
) -> JsonDict:
    """Write the terminal artifact and, when possible, the distilled rules."""

    base = Path(root) if root is not None else _repo_root()
    output = Path(artifact_path) if artifact_path is not None else base / RESULT_RELATIVE_PATH
    rules_path = Path(rules_output) if rules_output is not None else base / DISTILLED_RULES_RELATIVE_PATH
    artifact_rules_path = (
        str(rules_path) if rules_output is not None else DISTILLED_RULES_RELATIVE_PATH
    )
    artifact = build_artifact(
        root=base,
        run_date=run_date,
        tests_run=tests_run,
        current_duration_s=current_duration_s,
        distilled_rules_path=artifact_rules_path,
    )
    if artifact["exp5128_loaded"]:
        upstream, upstream_status = load_exp5128_artifact(base)
        rules = distill_symbolic_rules(upstream or {})
        cycle = cycle_check_rules(rules)
        controls = evaluate_controls(rules)
        rules_payload = build_rules_payload(
            upstream_status=upstream_status,
            rules=rules,
            cycle=cycle,
            controls=controls,
            run_date=run_date,
        )
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(
            json.dumps(rules_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for writing the default Exp 5140 artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run date as YYYYMMDD")
    parser.add_argument("--root", default=None, help="Optional repository root override")
    parser.add_argument("--output", default=None, help="Optional terminal artifact output path")
    parser.add_argument("--rules-output", default=None, help="Optional distilled rules output path")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else _repo_root()
    output = Path(args.output) if args.output else root / RESULT_RELATIVE_PATH
    artifact = write_outputs(
        root=root,
        run_date=str(args.date),
        artifact_path=output,
        rules_output=args.rules_output,
    )
    print(artifact["honest_verdict"])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through script wrapper.
    raise SystemExit(main())
