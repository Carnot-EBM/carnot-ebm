"""REQ-REPORT-8756: qualify existing parity without changing historical producers.

Private module globals retain the original checks while giving this invocation
its own identity. Repository health remains evidence rather than a parity gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import single_extraction_8743 as prior
from carnot.reporting import single_extraction_runner_8743 as prior_runner
from carnot.reporting.v709_execution import child, progress
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(prior, "feature_reuse8756_report")
legacy = namespace(prior_runner, "feature_reuse8756_runner")
e.NAME, e.TASK = (
    "experiment_8756_v750_feature_reuse_qualification",
    "exp8756-feature-reuse-qualification",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.OWNED += ["python/carnot/reporting/feature_reuse_8756.py", e.CLI]
e.TESTS += ["tests/python/test_feature_reuse_8756.py"]
e.protocol_added_lines = prior.protocol_added_lines
legacy.e, legacy.runtime.e = e, e
BASE_INPUTS, BASE_BUILD = e.inputs, e.build
BASE_MEASURE = legacy.runtime.measurement.measure
PROTOCOL = dict(
    path=str(e.ROOT / "openspec/change-proposals/v750-feature-reuse-8756-protocol.json"),
    sha256="sha256:ca0b73760a21e8a90da8db6ad78f40f197172dc026458da8d43a8e9adc9c594a",
)


def inputs(root: Path) -> Json:
    """The archived node list and workload cannot silently change between milestones."""
    value: Json = BASE_INPUTS(root)
    if value["passed"]:
        protocol = e.load(PROTOCOL)
        if value["original_nodes"] != protocol["original_node_union"]:
            raise ValueError("original122_protocol")
        history = e.load(protocol["preserved_primary"])
        assert (
            history["verdict_class"] == "disqualified" and history["feature_reuse_ready_score"] == 0
        )
        value["references"] += [
            PROTOCOL,
            protocol["preserved_primary"],
            protocol["original_cost_protocol"]["source"],
        ]
    return value


def measure(raw: Path, scratch: Path, inputs: Json, *, deadline_ns: int = 2**63) -> Json:
    """Freeze every collected node before building or observing the installed providers."""
    raw.mkdir(parents=True, exist_ok=True)
    suites = list(
        dict.fromkeys(
            [n.split("::")[0] for n in inputs["original_nodes"]]
            + e.TESTS
            + [
                "tests/python/test_installed_lean_8633.py",
                "tests/python/test_primary_publication_7928.py",
            ]
        )
    )
    receipt = child(
        "freeze_test_nodes",
        [sys.executable, "-m", "pytest", *legacy.original.OPTIONS, "--collect-only", *suites],
        raw / "checks",
        deadline=120,
    )
    nodes = [
        n
        for n in Path(receipt["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    if (
        not receipt["passed"]
        or not set(inputs["original_nodes"]) <= set(nodes)
        or len(nodes) != len(set(nodes))
    ):
        raise ValueError("premeasurement_node_union")
    frozen = raw / "frozen_test_nodes.json"
    e.atomic_json(
        frozen,
        dict(
            node_ids=nodes,
            node_ids_sha256=e.canonical_hash(nodes),
            input_sha256=e.canonical_hash(inputs),
            sources=source_hashes(),
            receipt=receipt,
        ),
    )
    result: Json = BASE_MEASURE(
        raw / "installed_measurement", scratch, inputs, deadline_ns=deadline_ns
    )
    result["frozen_test_node_reference"] = e.reference(frozen)
    result["traces"].append(result["frozen_test_node_reference"])
    result["receipts"].append(receipt)
    return result


def source_hashes() -> list[Json]:
    """Freeze the wrapper, original assertions, numerical helpers and validator closure."""
    return list(legacy.source_hashes()) + [
        e.reference(e.ROOT / p)
        for p in [
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/v746_retained_evidence.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "python/carnot/reporting/installed_checkout_8660.py",
            "python/carnot/reporting/installed_validation_8660.py",
            "python/carnot/reporting/installed_batch_8702.py",
            "python/carnot/reporting/current_work_receipt.py",
            "scripts/conductor_gates.py",
            "scripts/in_process_doc_reconcile.py",
            "scripts/experiment_template.py",
            "pyproject.toml",
            "tests/python/conftest.py",
            "openspec/change-proposals/v750-feature-reuse-8756-protocol.json",
            "tests/python/test_installed_lean_8633.py",
            "tests/python/test_primary_publication_7928.py",
        ]
        + sorted({n.split("::")[0] for n in e.load(PROTOCOL)["original_node_union"]})
    ]


def build(work: Json, output: Path) -> Json:
    """Only the new producer receives scoped readiness; the old primary stays sealed."""
    value: Json = BASE_BUILD(work, output)
    validation = work["validation"]
    value.update(
        experiment_id=8756,
        task_id=e.TASK,
        milestone="2026.11.750",
        run_date="20261103",
        honest_verdict=f"complete_{value['verdict_class']}_feature_reuse_qualification",
        numerical_training_calls=0,
        random_seed=7508756,
        repository_health_reference=validation.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            original_node_ids=work["inputs"].get("original_nodes", []),
            required_suites=e.TESTS,
            repository_health_is_dependency=False,
            original_assertions_unchanged=True,
            e2e=["E2E-003", "E2E-004", "E2E-018", "E2E-021"],
        ),
        owned_failure_control=validation.get("owned_failure_control", {}),
        cost_protocol_reference=PROTOCOL,
    )
    value.pop("reproducibility_checksum")
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind this producer to complete scoped evidence; parity establishes no speed or generalization.",
    )
    value["reproducibility_checksum"] = e.canonical_hash(value)
    return value


def validate(raw: Path, scratch: Path, sources: list[Json], original: list[str]) -> Json:
    """Keep every original check, then prove a failed owned node still denies readiness."""
    result: Json = legacy.validate(raw, scratch, sources, original)
    health = next(r for r in result["receipts"] if r["name"] == "full_python_suite")
    result["repository_health_reference"] = dict(
        receipt=health,
        full_suite_pass=health["passed"],
        bounded_diagnostic=True,
        proves_complete_full_suite_pass=False,
    )
    result["checks"] = [r for r in result["checks"] if r["name"] != "full_python_suite"]
    result["original_node_union"].update(
        node_ids=original, node_ids_sha256=e.canonical_hash(original)
    )
    node = scratch / "test_owned_failure_8756.py"
    node.write_text(
        '"""REQ-VERIFY-8756: deliberate private negative control."""\ndef test_owned_failure():\n    assert False\n'
    )
    failed = child(
        "owned_failure_control",
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            str(node),
            "--basetemp=" + str(scratch / "owned_failure_tmp"),
            "--junitxml=" + str(raw / "owned_failure.xml"),
        ],
        raw / "checks",
        expected=1,
        deadline=60,
    )
    observed = int(
        result["coverage"]["complete"]
        and all(r["passed"] for r in result["checks"])
        and failed["actual_exit"] == 0
    )
    control = dict(
        passed=failed["passed"] and failed["actual_exit"] == 1 and observed == 0,
        receipt=failed,
        feature_reuse_ready_score=observed,
    )
    result["owned_failure_control"] = control
    result["receipts"].append(failed)
    result["checks"].append(dict(name="owned_failure_zeros_readiness", passed=control["passed"]))
    return result


def main(argv: list[str] | None = None) -> int:
    """Reject date and output aliases before any work, including real CLI children."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    progress("8756_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    if (
        args.date != "20261103"
        or args.private_output
        and (
            args.private_output.absolute().is_relative_to(e.ROOT)
            or args.private_output.resolve().is_relative_to(e.ROOT)
        )
    ):
        return 2
    if args.cold_replay:
        authority = dict(zip(["path", "sha256"], args.authority)) if args.authority else None
        return int(not replay(args.cold_replay, authority))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with patch.object(legacy.runtime.shared, "source_hashes", source_hashes):
            return int(
                legacy.runtime.run(
                    args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
                    root=args.root,
                    private=bool(args.private_output),
                )
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


e.inputs, e.build = inputs, build
legacy.runtime.measurement = SimpleNamespace(measure=measure)
legacy.runtime.validate = validate
replay = legacy.replay
legacy.runtime.replay = replay
