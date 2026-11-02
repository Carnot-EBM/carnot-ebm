"""REQ-REPORT-8581: qualify new evidence against an explicit current revision.

Old producer pins remain authoritative for old evidence. New accounting uses
an immutable qualified revision without changing numerical or count mechanics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import causal_accounting_8568 as previous
from carnot.reporting import current_validator_8580 as validator
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
old = previous.old
ROOT = previous.ROOT
NAME = "experiment_8581_v737_causal_accounting_qualification"
TASK = "exp8581-causal-accounting-qualification"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = ["tests/python/test_causal_accounting_8581.py"]
OWNED = [
    f"python/carnot/reporting/causal_accounting{s}_8581.py" for s in ("", "_runner", "_validation")
] + [CLI]
ORIGINAL_VALIDATOR_PIN = previous.VALIDATOR_PIN
UPSTREAM = {
    "experiment_8579_v737_contract_methods": "sha256:59aa488c9a37dd33cc5564bbe2f87b54ef6744cb63323dc6a291959985741a64",
    validator.NAME: "sha256:7656f6f49b69237528263cd6dc38ac9503e853b1f53d1b5b1037f6510d2892f4",
}


def accept_revision(reference: Json) -> Json:
    """A forged descriptor cannot replace the revision named by the qualified primary."""
    primary = old.read(
        dict(
            path=str(ROOT / "results" / (validator.NAME + ".json")), sha256=UPSTREAM[validator.NAME]
        )
    )
    if reference != primary["current_validator_reference"]:
        raise ValueError("unqualified_revision_reference")
    return validator.accept_revision(reference)


def authority(root: Path) -> tuple[Json, list[Json], list[Json]]:
    """Authenticate current prerequisites separately from original historical authority."""
    binding: Json = {}
    refs: list[Json] = []
    failures: list[Json] = []
    for name, pin in UPSTREAM.items():
        path = root / "results" / (name + ".json")
        value: Json = {}
        try:
            value = old.read(dict(path=str(path), sha256=pin))
            terminal = old.read(old.reference(Path(value["terminal_validation_sidecar_path"])))
            bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
            if not bound["report"]["passed"]:
                raise ValueError("upstream_publication")
            refs.append(old.reference(path))
            if name == validator.NAME:
                if value["current_validator_ready_score"] != 1:
                    raise ValueError("current_validator_ready_score")
                accept_revision(value["current_validator_reference"])
                binding["current_validator_reference"] = value["current_validator_reference"]
                refs.append(value["current_validator_reference"])
            else:
                binding["learning_inputs_ready_score"] = value["learning_inputs_ready_score"]
                if binding["learning_inputs_ready_score"] != 1:
                    raise ValueError("learning_inputs_ready_score")
        except (OSError, ValueError, KeyError, TypeError) as error:
            field = str(error)
            score_failure = field in {
                "current_validator_ready_score",
                "learning_inputs_ready_score",
            }
            failures.append(
                old.gate(
                    path,
                    field if score_failure else "current_upstream_authentication",
                    1 if score_failure else True,
                    value.get(field)
                    if score_failure
                    else None
                    if isinstance(error, FileNotFoundError)
                    else str(error),
                    upstream=value.get("task_id", name),
                )
            )
    return binding, refs, failures


def build(work: Json, output: Path) -> Json:
    """Reuse original reductions while giving only this invocation the new identity."""
    value = previous.build(work, output)
    value.update(
        experiment_id=8581,
        task_id=TASK,
        milestone="2026.10.737",
        random_seed=7378581,
        honest_verdict="complete_"
        + value["verdict_class"]
        + "_v737_causal_accounting_qualification",
        current_validator_reference=work["validator_reference"],
        learning_inputs_ready_score=work["binding"].get("learning_inputs_ready_score"),
        historical_revision_rejection=work["original_revision_failures"],
        historical_accounting_reference=work["historical_accounting_reference"],
        historical_quarantines=dict(
            exp8529="source_null_unchanged",
            exp8553="label_rejection_unchanged",
            exp8559="cold_cost_null_unchanged",
            exp8568="old_revision_hash_rejection_unchanged",
            every_prior_quarantine="unchanged",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8582, 8591)
        ],
    )
    value["field_principles"].update(
        {
            k: "Bind explicit current evidence; historical qualification and scientific limits remain unchanged."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = old.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild counts from original events and authenticate actual child receipts."""
    from carnot.reporting.causal_accounting_validation_8581 import ready

    try:
        value = json.loads(path.read_bytes())
        work = old.read(value["primitive_reference"])
        if value["reproducibility_checksum"] != old.checksum(value):
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if old.reference(Path(ref["path"])) != ref:
                return False
        for receipt in work["receipts"]:
            if (
                old.read(old.reference(Path(receipt["log_path"]).with_suffix(".receipt.json")))
                != receipt
            ):
                return False
            for stream in ("stdout", "stderr"):
                if (
                    old.reference(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        binding, _, failures = authority(ROOT)
        historical, _, original_failures = previous.authority(ROOT)
        original = old.controls() if work["private"] else old.history()[0]
        return bool(
            work["binding"] == binding
            and work["upstream_failures"] == failures
            and work["historical"] == historical
            and work["original_revision_failures"] == original_failures
            and previous.read(work["events"]) == original
            and work["controls"] == old.controls()
            and work["diagnostics"] == previous.diagnostics(ROOT)
            and work["owned_ready"] == ready(work)
            and value == build(work, Path(value["output_path"]))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
