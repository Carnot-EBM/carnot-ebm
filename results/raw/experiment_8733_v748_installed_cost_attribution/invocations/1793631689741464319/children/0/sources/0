"""REQ-REPORT-8733: isolated version bindings retain original installed authority."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import installed_batch_cost_8717 as original
from carnot.reporting import installed_batch_cost_measure_8717 as measure_original
from carnot.reporting import installed_batch_cost_runner_8717 as runner_original
from carnot.reporting import v748_collection_contract as contract
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
e = contract.namespace(original, "attribution_report")
m = contract.namespace(measure_original, "attribution_measure")
runtime = contract.namespace(runner_original, "attribution_runner")
e.NAME, e.TASK = (
    "experiment_8733_v748_installed_cost_attribution",
    "exp8733-installed-cost-attribution",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_installed_cost_attribution_8733.py"]
e.OWNED = [
    f"python/carnot/reporting/installed_cost_attribution{s}_8733.py"
    for s in ["", "_measure", "_report"]
]
e.OWNED += ["python/carnot/inference/installed_profile_8733.py", e.CLI]
e.contract = contract.m
COST = "results/experiment_8717_v747_installed_batch_cost.json"
COST_PIN = "sha256:938d2cd6a28f8b0ae1ee1a551199a8ab7c3b29de91dca12f2d83aa8d1379fef6"
original_inputs = e.inputs


def inputs(root: Path) -> Json:
    """Authenticate the current branch independently of learning, CUDA and board gates."""
    value: Json = original_inputs(root)
    checked = e.previous.authenticate(
        dict(path=str(root / COST), sha256=COST_PIN),
        dict(
            experiment_id=8717,
            required_checks_passed=True,
            flagged_adversarial=False,
            batch_cost_ready_score=1,
            same_batch_10x_score=0,
        ),
    )
    value["references"] += checked["references"]
    value["gate_check_summary"] += checked["failures"]
    if checked["passed"]:
        path = Path(checked["value"]["terminal_validation_sidecar_path"])
        try:
            wrapper = e.reference(path)
            path = Path(e.load(wrapper)["publication"]["sidecar_path"])
            if read_bound_sidecar(root / COST, path)["report"]["passed"] is not True:
                raise ValueError("cost_terminal_rejected")
            value["references"] += [wrapper, e.reference(path)]
            value["historical_paired_intervals"] = checked["value"]["paired_intervals"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            value["gate_check_summary"].append(
                contract.gate(path, "qualified_terminal_receipt", True, str(error), e.TASK)
            )
    value["references"] += [e.reference(Path(contract.__file__))]
    value["passed"] = not value["gate_check_summary"]
    return value


def roster(protocol: Json) -> list[Json]:
    """Rotate modes within each same-arm process pair; all forty cells remain visible."""
    return [
        dict(
            unit_id=f"B16-R{repeat}-{arm}-{int(mode)}",
            batch=16,
            repeat=repeat,
            arm=arm,
            instrumented=mode,
            panels=16,
            item_evaluations=1024,
            status="censored",
        )
        for repeat in range(10)
        for arm in (["python", "native"] if repeat % 2 == 0 else ["native", "python"])
        for mode in ([False, True] if repeat % 2 == 0 else [True, False])
    ]


def aggregate(rows: list[Json]) -> Json:
    """Process clusters, rather than repeated items, enter every bootstrap interval."""
    complete = len(rows) == 40 and all(r["status"] == "completed" and r["passed"] for r in rows)
    result: Json = dict(complete=complete, ready=False, overhead=None, fractions=[])
    if not complete:
        return result
    lookup = {(r["repeat"], r["arm"], r["instrumented"]): r for r in rows}
    indices = np.random.default_rng(7488733).integers(0, 10, size=(10000, 10))
    overhead, fractions = [], []
    for arm in ["python", "native"]:
        off = [lookup[i, arm, False] for i in range(10)]
        on = [lookup[i, arm, True] for i in range(10)]
        if any(
            a["request_digest"] != b["request_digest"] or a["output_digest"] != b["output_digest"]
            for a, b in zip(off, on, strict=True)
        ):
            raise ValueError("paired_exact_parity")
        a, b = (np.asarray([r["duration_s"] for r in group]) for group in [off, on])
        boot = b[indices].sum(axis=1) / a[indices].sum(axis=1) - 1
        overhead.append(
            dict(
                arm=arm,
                estimate=float(b.sum() / a.sum() - 1),
                lower95=float(np.quantile(boot, 0.025)),
                upper95=float(np.quantile(boot, 0.975)),
            )
        )
        for domain, field, duration in [
            ("child_serve", "components", "child_duration_s"),
            ("parent_process", "parent_components", "duration_s"),
        ]:
            for component in sorted(on[0][field]):
                cost = np.asarray([r[field].get(component, 0) for r in on])
                parent = np.asarray([r[duration] for r in on])
                fraction = cost[indices].sum(axis=1) / parent[indices].sum(axis=1)
                fractions.append(
                    dict(
                        arm=arm,
                        operation=component,
                        clock_domain=domain,
                        estimate=float(cost.sum() / parent.sum()),
                        lower95=float(np.quantile(fraction, 0.025)),
                        upper95=float(np.quantile(fraction, 0.975)),
                    )
                )
    high = max(r["upper95"] for r in overhead)
    result.update(
        ready=high <= 0.05,
        overhead=dict(
            upper95=high,
            arms=overhead,
            paired_repeats=10,
            bootstrap_draws=10000,
            bootstrap_seed=7488733,
        ),
        fractions=fractions,
    )
    return result


def replay(path: Path, authority: Json | None = None) -> bool:
    """An external primitive identity prevents a rewritten report from blessing itself."""
    try:
        value = json.loads(path.read_bytes())
        if authority is None or value["primitive_reference"] != authority:
            return False
        work = e.load(authority)
        if e.load(work["seal_reference"]) != {
            k: v for k, v in work.items() if k != "seal_reference"
        }:
            return False
        if inputs(Path(work["input_root"])) != work["inputs"]:
            return False
        if not all(
            e.reference(Path(ref["path"])) == ref
            for ref in work["sources"] + work["raw_references"]
        ):
            return False
        rebuilt: Json = dict(
            process_rows=[], request_rows=[], parity_rows=[], complete_cost_rows=[]
        )
        for row in work["measurement"]["process_rows"]:
            if row["status"] == "completed":
                primitive = e.load(
                    e.reference(Path(work["raw"]) / "measurement" / row["unit_id"] / "session.json")
                )
                reduced = reduce(primitive, work["inputs"])
                rebuilt["process_rows"].append(reduced["process_row"])
                for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                    rebuilt[key] += reduced[key]
            else:
                rebuilt["process_rows"].append(row)
        if any(rebuilt[k] != work["measurement"][k] for k in rebuilt):
            return False
        return bool(e.build(work, Path(work["output"])) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


from carnot.reporting.installed_cost_attribution_measure_8733 import prepare, session, reduce
from carnot.reporting.installed_cost_attribution_report_8733 import build

e.inputs, e.roster, e.aggregate, e.build = inputs, roster, aggregate, build
m.e, m.prepare, m.session, m.reduce = e, prepare, session, reduce
runtime.e, runtime.m, runtime.replay = e, m, replay
runtime.COVERAGE_INCLUDE = "include=" + ",".join(str(e.ROOT / p) for p in e.OWNED) + "\n"
runtime.shared = contract.namespace(runtime.shared, "attribution_validation")
runtime.shared.CONSUMERS += [
    "tests/python/test_installed_lean_8633.py::" + n
    for n in [
        "test_lean_import_and_no_path_mutation",
        "test_original_export_contract",
        "test_optional_jax_absence",
        "test_module_cli_help_and_rejection",
        "test_legacy_contract_snapshot",
    ]
]
main = runtime.main
