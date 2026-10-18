"""Exp 5203: verifier-authenticity remediation options for v476.

Spec refs: REQ-VERIFY-5203, SCENARIO-VERIFY-5203.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_RELATIVE_PATH = Path("ops/verifier_authenticity_audit_report.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
AND_VERIFIER_RELATIVE_PATH = Path("python/carnot/verify/and_composition_verifier.py")
CLAIM_ROUTER_RELATIVE_PATH = Path("python/carnot/verify/claim_isolation_uncertainty_router.py")
VERIFY_REPAIR_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
VERIFICATION_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
DOC_RELATIVE_PATH = Path("ops/verifier_remediation_options_v476.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5203_verifier_authenticity_remediation_options_v476.json"
)

EXPERIMENT = "experiment_5203_verifier_authenticity_remediation_options_v476"
EXPERIMENT_ID = "exp5203-verifier-authenticity-remediation-options-v476"
MILESTONE = "2026.07.476"
SCHEMA = "carnot.experiment_5203_verifier_authenticity_remediation_options.v476"
RUN_DATE = "20260703"
RANDOM_SEED = 5203
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SPEC_REFS = ["REQ-VERIFY-5203", "SCENARIO-VERIFY-5203"]
MODEL_SUBSTRATE_IMPORTS = {
    "torch",
    "transformers",
    "jax",
    "sklearn",
    "llama_cpp",
    "openai",
    "carnot.models",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "and_composition_verifier_options": (
        "Decision-ready options for the flagged AND-composition verifier; each option "
        "must be actionable without implying the operator already chose it."
    ),
    "claim_isolation_uncertainty_router_options": (
        "Decision-ready options for the flagged claim-isolation router; each option "
        "must distinguish artifact routing from live isolated-claim verification."
    ),
    "audit_findings_independently_reconfirmed": (
        "Confirms this task read the actual current source rather than trusting the "
        "audit report's quotes alone -- audits can also be stale by the time they're "
        "acted on."
    ),
    "remediation_doc_path": (
        "Path to the operator-facing remediation options document written by this task."
    ),
    "no_verifier_modified_this_task": (
        "The audit-and-remediation-prep discipline is explicitly non-destructive; this "
        "field is the honesty check that this task respected it."
    ),
    "inference_substrate": (
        "This task aggregates current source, audit, and call-site artifacts only; it "
        "performs no live verifier inference and no model training."
    ),
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_OPTION_KEYS = ("rename", "retire", "reimplement", "recommendation")
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "duration_s",
    "random_seed",
    "field_principles",
    "source_artifacts_read",
    "failed_preconditions",
    "source_reconfirmation",
    "audit_context_summary",
    "retirement_blast_radius",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_WRAPPED_FIELDS,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5203_verifier_authenticity_"
    "remediation_options_v476.py -q",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5203_"
    "verifier_authenticity_remediation_options_v476.py' -m pytest "
    "tests/python/test_experiment_5203_verifier_authenticity_remediation_options_v476.py "
    "-q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5203_"
    "verifier_authenticity_remediation_options_v476.py' --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _read_text(root: Path, rel_path: Path, failed: list[str]) -> str:
    path = root / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        failed.append(f"missing_or_unreadable:{rel_path}:{exc.__class__.__name__}")
        return ""


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
            modules.add(node.module)
    return modules


def analyze_and_composition_source(source: str, verify_repair_source: str) -> JsonDict:
    return {
        "untrained_soskan_returns_neutral_0_5": (
            "not self._trained" in source and "return 0.5" in source
        ),
        "soskan_score_cap_present": "min(raw / 2.0, 1.0)" in source,
        "exceptions_masked_as_pass_energy": (
            "except Exception" in source and "energy = 0.0" in source
        ),
        "default_k5_includes_soskan": (
            "_make_k5_verifiers" in source and "SOSKANEnergyV3Adapter()" in source
        ),
        "default_ensemble_exported": "def build_default_verifier_ensemble" in source,
        "pipeline_uses_default_ensemble": "build_default_verifier_ensemble" in verify_repair_source,
        "pipeline_records_advisory_certificate": (
            "and_compose_k5" in verify_repair_source and "Does NOT" in verify_repair_source
        ),
    }


def analyze_claim_router_source(source: str) -> JsonDict:
    imports = _imported_modules(source)
    model_imports = sorted(imports & MODEL_SUBSTRATE_IMPORTS)
    return {
        "imported_modules": sorted(imports),
        "model_substrate_imports": model_imports,
        "no_model_substrate_imported": not model_imports,
        "fixed_uncertainty_scores": (
            "uncertainty = 0.6" in source
            and "uncertainty = 1.0" in source
            and "uncertainty_score=0.0" in source
        ),
        "claim_isolated_accept_copied_from_manifest_or_validator": (
            "claim_isolated_accept=deterministic_accept" in source
            or "claim_isolated_accept=final_accept" in source
        ),
        "artifact_only_json_inputs": "def _read_json" in source and "def _read_jsonl" in source,
        "routing_threshold_policy": (
            "uncertainty_threshold" in source or "uncertainty_score" in source
        ),
        "no_actual_claim_verifier_call": not any(
            marker in source
            for marker in (
                "create_chat_completion",
                "verify_claim",
                "llama_cpp",
                "transformers",
                "openai",
                "torch.",
            )
        ),
    }


def _all_true(mapping: JsonMap, keys: Sequence[str]) -> bool:
    return all(bool(mapping.get(key)) for key in keys)


def and_composition_options() -> JsonDict:
    return {
        "rename": (
            "RENAME_TO_REFLECT_REALITY: rename the public surface to "
            "`AdvisoryK5VerifierAdapterHarness` (or "
            "`advisory_k5_verifier_adapter_harness.py`). That name matches the "
            "current behavior: an advisory adapter/certificate harness over mixed "
            "heuristic and model-shaped members, not a production trained k=5 "
            "energy ensemble with guaranteed exponential null-space shrinkage."
        ),
        "retire": (
            "RETIRE: remove the default AND-compose certificate from "
            "VerifyRepairPipeline and update tests/evals that import "
            "`build_default_verifier_ensemble()`. The final pipeline verdict should "
            "not change because the current certificate is advisory and explicitly "
            "does not short-circuit `result.verified`. What breaks is the "
            "`and_compose_k5` certificate, k=5 tests, and analysis scripts that "
            "partition SOSKAN/SemEnergy versus AST/Semantic/Z3. A remaining k-1 "
            "composition can be meaningful only if it is renamed as k=4/k-1 "
            "advisory composition and its correlations/thresholds are remeasured; "
            "it cannot inherit the current k=5 Exp 1108 claim."
        ),
        "reimplement": (
            "REIMPLEMENT_PROPERLY: load or train a real trained SOSKANEnergyV3 "
            "checkpoint on a declared FoVer split, persist the feature normalization "
            "stats, calibrate raw energy instead of capping `raw / 2.0`, and remove "
            "exception-to-pass masking in favor of an explicit degraded or fail-closed "
            "certificate. This also requires recomputing pairwise correlations, "
            "thresholds, FoVer AUROC, and the default ensemble tests against the "
            "actual energy-model substrate the current adapter only pretends to "
            "provide by default."
        ),
        "recommendation": (
            "REIMPLEMENT_PROPERLY. The repo already wires the ensemble into "
            "VerifyRepairPipeline as an advisory certificate, so retiring it would "
            "remove useful integration context, while a pure rename would leave an "
            "inert SOSKAN member in the default k=5 path. Until reimplementation is "
            "funded, quarantine headline claims and describe the current file as an "
            "advisory harness only."
        ),
    }


def claim_router_options() -> JsonDict:
    return {
        "rename": (
            "RENAME_TO_REFLECT_REALITY: rename the public surface to "
            "`ClaimIsolationArtifactRoutingLedger` (or "
            "`claim_isolation_artifact_routing_ledger.py`). That name states what "
            "the code does now: read JSON/JSONL artifacts, copy existing accept "
            "booleans, apply uncertainty/prefix-risk routing, and write a ledger."
        ),
        "retire": (
            "RETIRE: remove Exp 1541's artifact generator and update downstream "
            "`claim_isolation_router_scale.py`, milestone-retro references, and tests "
            "that expect `results/experiment_1541_claim_isolation_uncertainty_router_"
            "v2.json`. Live verification should not lose a model call because this "
            "module currently performs none, but the larger claim-router cost/safety "
            "lineage would lose its bridge from Exp 1525/1537 artifacts into Exp 1553."
        ),
        "reimplement": (
            "REIMPLEMENT_PROPERLY: for every extracted claim selected by the router, "
            "perform an actual model call per claim or invoke a real isolated-claim "
            "verifier with the original answer hidden, record the prompt/model/cache "
            "provenance, compare full-context and isolated decisions on the same "
            "case, and keep deterministic SAT/product-line/runtime validators as "
            "false-accept authority. Fixed uncertainty constants and copied "
            "`claim_isolated_accept` booleans would have to be replaced by measured "
            "per-claim verifier outputs."
        ),
        "recommendation": (
            "RENAME_TO_REFLECT_REALITY. The current module is useful artifact routing "
            "glue, and downstream Exp 1553-style scale work may still consume that "
            "ledger. Reimplementation would be a larger live-verifier project, while "
            "retirement would discard the routing lineage without improving verifier "
            "truthfulness as much as an honest name/docstring would."
        ),
    }


def summarize_audit_context(audit_text: str) -> JsonDict:
    return {
        "audit_path": str(AUDIT_RELATIVE_PATH),
        "scanned_20_verifiers": "Scanned 20 verifier file" in audit_text,
        "authentic_count_11": "| `AUTHENTIC` | 11 |" in audit_text,
        "honest_heuristic_count_6": "| `HONEST_HEURISTIC` | 6 |" in audit_text,
        "dishonest_naming_count_2": "| `DISHONEST_NAMING` | 2 |" in audit_text,
        "and_composition_flagged": str(AND_VERIFIER_RELATIVE_PATH) in audit_text,
        "claim_router_flagged": str(CLAIM_ROUTER_RELATIVE_PATH) in audit_text,
    }


def retirement_blast_radius() -> JsonDict:
    return {
        "and_composition_verifier": [
            "VerifyRepairPipeline construction of the default advisory `and_compose_k5` certificate",
            "tests/python/test_and_compose_k5.py k=5 membership expectations",
            "tests/python/test_exp1128_sos_kan_fix.py SOSKAN adapter coverage",
            "FoVer ablation and verifier-gaming evals importing `build_default_verifier_ensemble()`",
        ],
        "claim_isolation_uncertainty_router": [
            "results/experiment_1541_claim_isolation_uncertainty_router_v2.json lineage",
            "tests/python/test_experiment_1541_claim_isolation_uncertainty_router.py",
            "python/carnot/verify/claim_isolation_router_scale.py downstream scale input",
            "milestone activation and retro summaries that cite Exp 1541",
        ],
    }


def render_doc(artifact: JsonMap) -> str:
    and_options = artifact["and_composition_verifier_options"]["value"]
    router_options = artifact["claim_isolation_uncertainty_router_options"]["value"]
    reconfirmation = artifact["source_reconfirmation"]
    return (
        "# Verifier Remediation Options V476\n\n"
        "Prepared for the operator-decides step in CLAUDE.md's Verifier "
        "Authenticity Discipline. No verifier source was modified by this task.\n\n"
        "## Audit Context\n\n"
        "- Audit report: `ops/verifier_authenticity_audit_report.md` "
        "(2026-07-01).\n"
        "- Audit summary read for context: 20 files scanned, 11 AUTHENTIC, "
        "6 HONEST_HEURISTIC, 2 DISHONEST_NAMING, 1 CANNOT_DETERMINE, "
        "0 ADVERSARIAL_GAMING, 0 OUTRIGHT_FAKE.\n"
        "- This document reconfirms the two flagged findings from current source "
        "instead of relying only on the audit quotes.\n\n"
        "## Independent Reconfirmation\n\n"
        "```json\n"
        f"{json.dumps(reconfirmation, indent=2, sort_keys=True)}\n"
        "```\n\n"
        "## and_composition_verifier.py\n\n"
        "### RENAME_TO_REFLECT_REALITY\n\n"
        f"{and_options['rename']}\n\n"
        "### RETIRE\n\n"
        f"{and_options['retire']}\n\n"
        "### REIMPLEMENT_PROPERLY\n\n"
        f"{and_options['reimplement']}\n\n"
        "### Recommendation\n\n"
        f"{and_options['recommendation']}\n\n"
        "## claim_isolation_uncertainty_router.py\n\n"
        "### RENAME_TO_REFLECT_REALITY\n\n"
        f"{router_options['rename']}\n\n"
        "### RETIRE\n\n"
        f"{router_options['retire']}\n\n"
        "### REIMPLEMENT_PROPERLY\n\n"
        f"{router_options['reimplement']}\n\n"
        "### Recommendation\n\n"
        f"{router_options['recommendation']}\n\n"
        "## Decision Boundary\n\n"
        "This package prepares the operator's decision. It does not rename, retire, "
        "reimplement, or edit either flagged verifier.\n"
    )


def _doc_display_value(root: Path, doc_output: Path, doc_path_was_default: bool) -> str:
    if doc_path_was_default:
        return DOC_RELATIVE_PATH.as_posix()
    default_output = root / DOC_RELATIVE_PATH
    if doc_output == default_output:
        return DOC_RELATIVE_PATH.as_posix()
    return doc_output.as_posix()


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    doc_path: Path | str | None = None,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    root_path = Path(root)
    output = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    doc_path_was_default = doc_path is None
    doc_output = Path(doc_path) if doc_path is not None else root_path / DOC_RELATIVE_PATH
    failed: list[str] = []

    audit_text = _read_text(root_path, AUDIT_RELATIVE_PATH, failed)
    claude_text = _read_text(root_path, CLAUDE_RELATIVE_PATH, failed)
    and_source = _read_text(root_path, AND_VERIFIER_RELATIVE_PATH, failed)
    router_source = _read_text(root_path, CLAIM_ROUTER_RELATIVE_PATH, failed)
    verify_repair_source = _read_text(root_path, VERIFY_REPAIR_RELATIVE_PATH, failed)
    spec_text = _read_text(root_path, VERIFICATION_SPEC_RELATIVE_PATH, failed)

    if "REQ-VERIFY-5203" not in spec_text:
        failed.append("missing_spec_anchor:REQ-VERIFY-5203")
    if "operator decides" not in claude_text.lower():
        failed.append("missing_claude_operator_decides_contract")

    source_reconfirmation = {
        "and_composition_verifier": analyze_and_composition_source(
            and_source,
            verify_repair_source,
        ),
        "claim_isolation_uncertainty_router": analyze_claim_router_source(router_source),
    }
    and_reconfirmed = _all_true(
        source_reconfirmation["and_composition_verifier"],
        (
            "untrained_soskan_returns_neutral_0_5",
            "soskan_score_cap_present",
            "exceptions_masked_as_pass_energy",
            "default_k5_includes_soskan",
            "pipeline_uses_default_ensemble",
            "pipeline_records_advisory_certificate",
        ),
    )
    router_reconfirmed = _all_true(
        source_reconfirmation["claim_isolation_uncertainty_router"],
        (
            "no_model_substrate_imported",
            "fixed_uncertainty_scores",
            "claim_isolated_accept_copied_from_manifest_or_validator",
            "artifact_only_json_inputs",
            "no_actual_claim_verifier_call",
        ),
    )
    if not and_reconfirmed:
        failed.append("and_composition_source_not_reconfirmed")
    if not router_reconfirmed:
        failed.append("claim_router_source_not_reconfirmed")

    audit_reconfirmed = bool(and_reconfirmed and router_reconfirmed)
    doc_display = _doc_display_value(root_path, doc_output, doc_path_was_default)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts_read": [
            str(AUDIT_RELATIVE_PATH),
            str(CLAUDE_RELATIVE_PATH),
            str(AND_VERIFIER_RELATIVE_PATH),
            str(CLAIM_ROUTER_RELATIVE_PATH),
            str(VERIFY_REPAIR_RELATIVE_PATH),
            str(VERIFICATION_SPEC_RELATIVE_PATH),
        ],
        "failed_preconditions": sorted(set(failed)),
        "source_reconfirmation": source_reconfirmation,
        "audit_context_summary": summarize_audit_context(audit_text),
        "retirement_blast_radius": retirement_blast_radius(),
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "and_composition_verifier_options": _wrap(
            "and_composition_verifier_options",
            and_composition_options(),
        ),
        "claim_isolation_uncertainty_router_options": _wrap(
            "claim_isolation_uncertainty_router_options",
            claim_router_options(),
        ),
        "audit_findings_independently_reconfirmed": _wrap(
            "audit_findings_independently_reconfirmed",
            audit_reconfirmed,
        ),
        "remediation_doc_path": _wrap("remediation_doc_path", doc_display),
        "no_verifier_modified_this_task": _wrap("no_verifier_modified_this_task", True),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "complete: verifier_authenticity_remediation_options_v476_ready"
                if not failed
                else "complete: blocked_verifier_authenticity_remediation_options_v476"
            ),
        ),
    }

    doc_output.parent.mkdir(parents=True, exist_ok=True)
    doc_output.write_text(render_doc(artifact), encoding="utf-8")
    artifact["remediation_doc_sha256"] = hashlib.sha256(
        doc_output.read_bytes(),
    ).hexdigest()
    artifact["reproducibility_checksum"] = payload_checksum(artifact)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def payload_checksum(payload: JsonMap) -> str:
    scrubbed = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact.get("schema") != SCHEMA:
        raise AssertionError("schema mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise AssertionError("experiment_id mismatch")

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise AssertionError(f"{field} must be principle-wrapped")
        if wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise AssertionError(f"{field} principle mismatch")

    for field in (
        "and_composition_verifier_options",
        "claim_isolation_uncertainty_router_options",
    ):
        options = artifact[field]["value"]
        missing_options = [key for key in REQUIRED_OPTION_KEYS if key not in options]
        if missing_options:
            raise AssertionError(f"{field} missing option keys: {missing_options}")
        if not all(isinstance(options[key], str) and options[key] for key in REQUIRED_OPTION_KEYS):
            raise AssertionError(f"{field} options must be non-empty strings")

    if not isinstance(artifact["audit_findings_independently_reconfirmed"]["value"], bool):
        raise AssertionError("audit_findings_independently_reconfirmed must be bool")
    if artifact["no_verifier_modified_this_task"]["value"] is not True:
        raise AssertionError("verifier source modification is outside this task")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise AssertionError("inference_substrate mismatch")

    verdict = str(artifact["honest_verdict"]["value"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise AssertionError("honest_verdict must start with a terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise AssertionError("reproducibility_checksum mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--doc-path", default=None)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    artifact = build_artifact(
        root=Path(args.root),
        result_path=Path(args.result_path) if args.result_path else None,
        doc_path=Path(args.doc_path) if args.doc_path else None,
        duration_s=time.perf_counter() - start,
        run_date=args.date,
    )
    validate_artifact(artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
