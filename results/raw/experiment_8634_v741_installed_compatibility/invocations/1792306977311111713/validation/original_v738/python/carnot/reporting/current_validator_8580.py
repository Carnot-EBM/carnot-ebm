"""REQ-REPORT-8580: bind a repaired validator without rewriting earlier authority.

Only new producers may use this revision. Original qualification hashes still
name the bytes that the old producer actually tested.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from carnot.reporting import count_validator_8567 as old
from carnot.reporting.count_semantics_evidence_8566 import read, reference
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8580_v737_current_validator_qualification"
TASK = "exp8580-current-validator-qualification"
MODEL_SPECS: list[Json] = []
AUTHORITY = ROOT / "results/raw" / NAME / "authority"
TESTS = ["tests/python/test_current_validator_8580.py"]
OWNED = [
    f"python/carnot/reporting/current_validator{s}_8580.py" for s in ("", "_runner", "_validation")
] + [f"scripts/experiments/{NAME}.py"]
REPAIRED = {
    "scripts/adversarial_verify.py": "sha256:18dfabaaed325c91bd03cd584c8f7b945a068529f00cbd68e45c0e0c013d49d2",
    "tests/python/test_count_validator_correction_8567.py": "sha256:9dddd0fab6df58395fc957c08a647163a4b2afcc10f5f300c3ec76642f66776d",
}
REPAIR_PINS = {
    "regression.xml": "sha256:e25c368ee5b538adf399b7887875a5b4fe383414a2180bf6307155ff07d845f2",
    "affected-e2e.xml": "sha256:0f02ee793cdf4ceb9dff7c24ab6d3bd31701e8d92679892ae55acf17d57f1052",
    "cli-controls.json": "sha256:c3c89ff43131300562acc7186c7dc493ddbd671540ac86679db517b1b8dac8ff",
}
REQUIRED = {
    "collection",
    "owned",
    "regression",
    "affected",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "cold",
    "missing",
    "owned_failure",
    "rehashed",
    "adversarial",
    "row_lint",
    "combine",
    "coverage",
    "consumer",
}
checksum = old.checksum


def xml_nodes(path: Path) -> list[str]:
    """Keep pytest class names separate from module paths so no node is lost."""
    result = []
    for row in ET.parse(path).iter("testcase"):
        parts = str(row.get("classname")).split(".")
        result.append("/".join(parts[:3]) + ".py::" + "::".join([*parts[3:], str(row.get("name"))]))
    return result


def reverse_repair(path: str) -> str:
    """Reverse only the documented import change and three additive CLI cases."""
    text = (ROOT / path).read_text()
    if path.endswith("adversarial_verify.py"):
        start = text.index("# Direct execution starts")
        end = text.index("from carnot.reporting.discrete_count_proof", start)
        text = text[:start] + text[end:]
        return text.replace("import proven_counts\n", "import proven_counts\n\n", 1)
    start = text.index('@pytest.mark.parametrize("proof_state"')
    end = text.index("def test_bounded_proof_rejections", start)
    return (text[:start] + text[end:]).replace("import os\n", "", 1)


def authenticate() -> Json:
    """Require exact old hashes, exact permitted repairs and sealed control bytes."""
    previous = read(
        dict(
            path=str(ROOT / "results" / (old.NAME + ".json")),
            sha256="sha256:ca6b5a53c8f4e0de8eb2f3dfd01ec1d08c462d50eac19dde904275c9c8b17a86",
        )
    )
    rejected = read(
        dict(
            path=str(ROOT / "results/experiment_8568_v736_causal_accounting_qualification.json"),
            sha256="sha256:cf4b256742a74783617ce692415bf8a4ae3a5abd966c22564eccaa5b73742e3a",
        )
    )
    upstream, original, corpus = old.authenticate()
    changed, deltas = [], []
    for ref in previous["code_config_hashes"]:
        relative = str(Path(ref["path"]).relative_to(ROOT))
        actual = reference(Path(ref["path"]))
        if actual != ref:
            if actual["sha256"] != REPAIRED.get(relative):
                raise ValueError("unexplained_revision_change")
            before = reverse_repair(relative)
            if "sha256:" + hashlib.sha256(before.encode()).hexdigest() != ref["sha256"]:
                raise ValueError("repair_original_hash")
            changed.append(relative)
            deltas.append(
                dict(
                    path=relative,
                    original=ref,
                    current=actual,
                    diff="".join(
                        difflib.unified_diff(
                            before.splitlines(True),
                            Path(ref["path"]).read_text().splitlines(True),
                            fromfile="original/" + relative,
                            tofile="current/" + relative,
                        )
                    ),
                )
            )
    if set(changed) != set(REPAIRED):
        raise ValueError("repair_missing")
    for name, pin in REPAIR_PINS.items():
        if reference(AUTHORITY / name)["sha256"] != pin:
            raise ValueError("repair_receipt_hash")
    return dict(
        previous=previous,
        rejected_revision=rejected,
        upstream=upstream,
        original=original,
        corpus=corpus,
        changed_paths=changed,
        deltas=deltas,
        regression_nodes=xml_nodes(AUTHORITY / "regression.xml"),
        affected_nodes=xml_nodes(AUTHORITY / "affected-e2e.xml"),
    )


def gates(work: Json) -> Json:
    """Readiness requires current execution as well as historical identity."""
    return dict(
        preconditions=work["preconditions"]["passed"],
        original_authority=not work["failures"],
        frozen_controls=len(work["controls"]) == 17
        and all(r["expectation_met"] for r in work["controls"]),
        preserved_detection=bool(work["preserved"])
        and all(r["unchanged"] for r in work["preserved"]),
        standalone_cli=len(work["cli_rows"]) == 17 and all(r["passed"] for r in work["cli_rows"]),
        exact_node_union=work.get("union_passed", False),
        required_commands=REQUIRED <= {r["name"] for r in work["receipts"] if r["passed"]},
        owned_children=all(r["passed"] for r in work["receipts"]),
        new_statement_coverage=work.get("coverage_complete", False),
    )


def revision(work: Json) -> Json:
    """Seal measured operands separately so the primary can reference immutable bytes."""
    return dict(
        ready=all(gates(work).values()),
        code=work["code"],
        controls=work["controls"],
        commands=work["receipts"],
        exact_node_union=work.get("node_union", {}),
        private_replay=work["cli_rows"],
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
    )


def accept_revision(current_validator_reference: Json) -> Json:
    """An explicit accounting argument must retain tested source and primitive evidence."""
    value: Json = read(current_validator_reference)
    if value.get("ready") is not True:
        raise ValueError("revision_not_ready")
    for ref in value["code"]:
        if reference(Path(ref["path"])) != ref:
            raise ValueError("revision_drift")
    work = read(value["primitive_reference"])
    if value != revision(work) or not valid_work(work):
        raise ValueError("revision_evidence")
    return value


def build(work: Json, output: Path) -> Json:
    """Keep qualification distinct from learning, with missing evidence left explicit."""
    checks = gates(work)
    ready = all(checks.values())
    kind = (
        "null"
        if ready
        else "blocked"
        if work["failures"] and checks["owned_children"]
        else "disqualified"
    )
    raw = Path(work["raw"])
    failures = [
        *work["failures"],
        *[
            dict(
                check=k,
                upstream=TASK,
                path=str(raw / "work.json"),
                sha256=reference(raw / "work.json")["sha256"],
                field=k,
                operator="==",
                expected=True,
                observed=v,
            )
            for k, v in checks.items()
            if not v
        ],
    ]
    value: Json = dict(
        experiment_id=8580,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        honest_verdict="complete_" + kind + "_v737_current_validator_qualification",
        verdict_class=kind,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=work.get("evidence", {}).get("historical_model_provenance", []),
        rows=work["controls"],
        intended_count=17,
        completed_count=len(work["controls"]),
        failed_count=sum(not r["expectation_met"] for r in work["controls"]),
        censored_count=17 - len(work["controls"]),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="frozen regression control", intended_units=17, independent_units=0
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed frozen count-validator regression controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=False,
        acceptance_gates=checks,
        validation_receipts=work["receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phases"],
        random_seed=7378580,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[reference(raw / "work.json"), reference(raw / "repair.diff.json")],
        cited_upstream_artifacts=work["sources"],
        current_validator_ready_score=int(ready),
        current_validator_reference=reference(raw / "current_validator_revision.json"),
        repair_diff_reference=reference(raw / "repair.diff.json"),
        frozen_control_rows=work["controls"],
        cli_import_classification_rows=work["cli_rows"],
        exact_node_union=work.get("node_union", {}),
        mutation_receipts=[
            r for r in work["receipts"] if r["name"] in {"missing", "owned_failure", "rehashed"}
        ],
        preserved_detection_rows=work["preserved"],
        newly_owned_statement_coverage_complete=work.get("coverage_complete", False),
        primitive_reference=reference(raw / "work.json"),
        output_path=str(output.absolute()),
        historical_quarantines=dict(
            exp8529="source_null_unchanged",
            exp8553="label_rejection_unchanged",
            exp8559="cold_cost_null_unchanged",
            exp8555="disqualified_unchanged",
            exp8568="old_revision_hash_rejection_unchanged",
            every_prior_quarantine="unchanged",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8581, 8591)
        ],
        methodology_note="Actual CPU control replay and current regression execution qualify this exact revision. No model load, learning benefit or independent generalization is measured. Earlier invalid hashes remain invalid.",
    )
    value["field_principles"] = {
        k: "Bind this invocation; exposed regression qualification grants no scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        current_validator_reference="Explicit immutable revision for new producers; old pins remain unchanged.",
        exact_node_union="Collected original identities equal the executed disjoint union.",
        repair_diff_reference="Reverse the documented repair to authenticate original source and assertions.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def valid_work(work: Json) -> bool:
    """Recompute controls and authenticate logs rather than trusting repaired summaries."""
    authority = authenticate()
    if (
        work["authority"] != authority
        or work["controls"] != old.classifications(authority["corpus"])
        or work["preserved"] != old.preserved(authority["upstream"])
    ):
        return False
    if work["controls"] != authority["previous"]["old_new_classification_rows"]:
        return False
    for ref in [*work["code"], *work["sources"]]:
        if reference(Path(ref["path"])) != ref:
            return False
    for receipt in work["receipts"]:
        if read(reference(Path(receipt["log_path"]).with_suffix(".receipt.json"))) != receipt:
            return False
        for stream in ("stdout", "stderr"):
            if reference(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
                return False
    for row in work["cli_rows"]:
        actual = old.av.verify_artifact(Path(row["candidate"]["path"]))["flags"]
        cli = json.loads(Path(row["receipt"]["stdout_path"]).read_bytes())["reports"][0]["flags"]
        if row["imported_findings"] != actual or row["cli_findings"] != cli or actual != cli:
            return False
    if work.get("union_passed"):
        from carnot.reporting.current_validator_validation_8580 import union_valid

        if not union_valid(work):
            return False
    return True


def replay(path: Path) -> bool:
    """Cold processes reject absent evidence and self-consistently rehashed inventions."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        sealed_revision = read(value["current_validator_reference"])
        return bool(
            value == build(work, Path(value["output_path"]))
            and sealed_revision == revision(work)
            and valid_work(work)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def accounting_revision(*, current_validator_reference: Json) -> Json:
    """Reuse the next accounting consumer's proof with an explicit qualified revision."""
    from carnot.reporting.causal_accounting_8568 import proof

    accept_revision(current_validator_reference)
    original = authenticate()["original"]
    metrics, count_proof = proof(read(original["evidence"]["events"]))
    findings = old.findings(dict(metrics, discrete_count_proof=count_proof), old.av.check_tautology)
    return dict(
        passed=not any(r["severity"] == "critical" for r in findings),
        metrics=metrics,
        findings=findings,
        current_validator_reference=current_validator_reference,
    )
