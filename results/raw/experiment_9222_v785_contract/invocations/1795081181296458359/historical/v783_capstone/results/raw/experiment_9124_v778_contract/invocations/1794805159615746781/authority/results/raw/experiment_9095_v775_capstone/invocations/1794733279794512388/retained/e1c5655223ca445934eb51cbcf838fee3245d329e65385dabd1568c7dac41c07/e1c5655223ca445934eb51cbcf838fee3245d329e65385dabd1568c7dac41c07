"""REQ-REPORT-9094-V775-PROVENANCE: unavailable costs do not block an operation map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import susceptibility_9094 as d
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
SOURCE_DIR = d.ROOT / "results/raw" / d.NAME / "sources"
SOURCE_FILES = {
    "abstract.html": "sha256:8f0b324649d57db9c9fe7dcbe5a49a78bf8b9bae84a9bdbd6844754b2d879251",
    "paper.html": "sha256:eb953e237b51112dc0db6f83df3e57a3d9195ed8d66f8008f2a2288d223ccbad",
}
COST = "results/experiment_9093_v775_service_cost.json"
HARDWARE = "results/experiment_9079_v774_hardware_boundary.json"
BOUNDARY = "GPU0 local inference hardware; GPU1 reserved; KV260 prior k_max<=5 only; PolarFire Fmax prior evidence only; Extropic and Kona vendor claims, no local measured access. No board acquisition, bring-up or dispatch."


def ref(path: Path) -> Json:
    """Missing evidence has no hash; it is never silently represented by a zero."""
    return dict(path=str(path), sha256=sha256_file(path) if path.is_file() else None)


def gate(
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    check: str,
    required: bool,
    upstream: str = d.TASK,
) -> Json:
    """Each gate names the original operand, its exact bytes and its required scope."""
    return dict(
        ref(path),
        check=check,
        upstream=upstream,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        required=required,
        passed=type(expected) is type(observed) and expected == observed,
    )


def retain(path: Path, raw: Path) -> Json:
    """Content-addressed read-only snapshots preserve producer bytes across later changes."""
    reference = ref(path)
    if reference["sha256"] is not None:
        target = raw / "objects" / reference["sha256"].split(":")[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
            target.chmod(0o400)
        reference.update(source_path=str(path), path=str(target))
    return reference


def authenticate(root: Path, raw: Path, source_dir: Path) -> Json:
    """Block dependent numerics on paper drift; optional service failures keep costs unknown."""
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    checks = [gate(raw, "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage", True)]
    references, unavailable, historical = [], [], []
    for name, digest in SOURCE_FILES.items():
        path = source_dir / name
        checks.append(
            gate(
                path,
                "sha256",
                digest,
                ref(path)["sha256"] or "missing_file",
                "primary_source",
                True,
                "arxiv:2611.14727v1",
            )
        )
        reference = retain(path, raw)
        reference.update(
            url="https://arxiv.org/"
            + ("abs/" if name == "abstract.html" else "html/")
            + "2611.14727v1",
            expected_sha256=digest,
        )
        references.append(reference)
    cost: Json = {}
    for name, identity, task in [
        (COST, 9093, "exp9093-service-cost"),
        (HARDWARE, 9079, "exp9079-hardware-boundary"),
    ]:
        path = root / name
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError):
            checks.append(
                gate(
                    path,
                    "readable_object",
                    True,
                    "missing_file" if not path.exists() else "invalid_json",
                    "optional_evidence",
                    False,
                    task,
                )
            )
            unavailable.append(
                dict(path=str(path), operand="complete_service_cost", reason=checks[-1]["observed"])
            )
            continue
        references.append(retain(path, raw))
        local = [
            gate(
                path,
                "experiment_id",
                identity,
                value.get("experiment_id", "missing_field"),
                "identity",
                False,
                task,
            ),
            gate(
                path,
                "task_id",
                task,
                value.get("task_id", "missing_field"),
                "identity",
                False,
                task,
            ),
        ]
        bindings = value.get("code_config_hashes", []) + value.get("raw_shard_hashes", [])
        bindings += [r for r in value.get("current_work_receipts", []) if "path" in r]
        local.append(
            gate(
                path,
                "producer_bindings_present",
                True,
                bool(bindings),
                "producer_presence",
                False,
                task,
            )
        )
        for binding in bindings:
            producer = Path(binding["path"])
            producer = producer if producer.is_absolute() else root / producer
            local.append(
                gate(
                    producer,
                    "sha256",
                    binding["sha256"],
                    ref(producer)["sha256"] or "missing_file",
                    "producer_hash",
                    False,
                    task,
                )
            )
            references.append(retain(producer, raw))
        historical.append(
            dict(
                experiment_id=identity,
                MODEL_SPECS=value.get("MODEL_SPECS", []),
                scope="historical producer only; never current invocations",
            )
        )
        if identity == 9093:
            if all(c["passed"] for c in local):
                cost = dict(
                    complete_service_qualified=False,
                    observed_latency_summary=value.get("observed_latency_summary", []),
                    scope="Authenticated administrative/qualification clocks only; no scientific request bottleneck or device crossover established",
                )
            local.append(
                gate(
                    path,
                    "complete_service_cost_ready_score",
                    1,
                    value.get("complete_service_cost_ready_score", "missing_field"),
                    "service_cost_readiness",
                    False,
                    task,
                )
            )
            if all(c["passed"] for c in local):
                cost = dict(
                    complete_service_qualified=True,
                    observed_latency_summary=value.get("observed_latency_summary", []),
                    scope="authenticated upstream only; no device crossover inferred",
                )
            else:
                unavailable.append(
                    dict(
                        path=str(path),
                        operand="complete_service_cost",
                        reason="unqualified or unavailable exact service gate; administrative clocks do not price scientific work",
                    )
                )
        checks.extend(local)
    return dict(
        gate_check_summary=checks,
        source_references=references,
        unavailable_cost_inputs=unavailable,
        cost=cost,
        historical_model_provenance=historical,
    )


def operation_map(evidence: Json) -> list[Json]:
    """Match operations to their mathematics before considering an unmeasured substrate."""
    definitions = [
        (
            "exact_graph_bfs",
            "Exact graph traversal and shortest paths",
            "CPU executable oracle",
            "Ising sampling cannot replace exact truth without proof and parity",
        ),
        (
            "small_head_spline_linear",
            "Local spline basis evaluation; linear/spline dot products and gradient arithmetic",
            "CPU deterministic small-head math",
            "No discrete spin-energy identity establishes an equivalent spline learner",
        ),
        (
            "durable_checkpointing",
            "Serialize state, flush/fsync, atomic replace and exact restart",
            "Host filesystem and checkpoint protocol",
            "Sampling cannot remove host I/O or durability",
        ),
        (
            "ising_sampling",
            "Quadratic binary energy and stochastic state transitions",
            "Hypothetical supported Ising device",
            "Needs sample-quality, precision, transfer and whole-service measurements",
        ),
        (
            "physical_learning",
            "Output forcing, equilibrium relaxation and cross-layer susceptibility",
            "CPU phase diagnostic; physical implementation hypothetical",
            "Equal binary energy does not certify force transmission or learning",
        ),
        (
            "fixed_point_local_basis_updates",
            "Quantized spline basis evaluation and local coefficient updates",
            "Hypothetical fixed-point kernel",
            "Needs rounding/overflow bounds, reference parity and host recovery; not Ising sampling",
        ),
    ]
    return [
        dict(
            d.row(775094, "operation_map", name, None),
            operation=name,
            mathematics=math,
            substrate=substrate,
            qualification_required=limit,
            cost_s=None,
            cost_status="unknown",
            upstream_cost_annotation=evidence["cost"],
            hardware_boundary=BOUNDARY,
            verifier_is_oracle=name == "exact_graph_bfs",
            graph_success_class="circular_positive" if name == "exact_graph_bfs" else None,
        )
        for name, math, substrate, limit in definitions
    ]
