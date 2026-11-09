"""REQ-REPORT-8956: pin original producers so later results cannot rewrite this census."""

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import v765_branch_contract as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.v763_capstone import references, resolve
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain

Json = dict[str, Any]
ROOT = authority.ROOT
PINS = {
    8944: (
        "experiment_8944_v765_branch_contract.json",
        "sha256:80229f95dc2507f5cbd9a88b1bf8be69c0365d44841603b39c6dfee5a7906129",
    ),
    8945: (
        "experiment_8945_v765_depth_opportunity_audit.json",
        "sha256:9f5147a1abd285111b91f30aa0314a6e89c6aa009ba3ee19e525c8eed66f9b3d",
    ),
    8946: (
        "experiment_8946_v765_matched_source_methods.json",
        "sha256:7cd1769932adf184a01e75351d83a997b2f7d15797095b581ae54b6886777b19",
    ),
    8947: (
        "experiment_8947_v765_gradient_replay_methods.json",
        "sha256:65fffc8b8675bc066f25584d1fccf93257f9497b8d4e3ff2cad6a17a8f49a2ff",
    ),
    8948: (
        "experiment_8948_v765_matched_depth_control.json",
        "sha256:da6efe2bb0adcc3e48f6b04b2b387d4e653a0cb9bed9df92a66dbc7f9e1fd9dd",
    ),
    8949: (
        "experiment_8949_v765_matched_corpus_a.json",
        "sha256:84f4c2d4ab4d22aea9e6b61f10ebe07800b719de7c2530d941e48aaf25c9ee02",
    ),
    8950: (
        "experiment_8950_v765_matched_corpus_b.json",
        "sha256:fcd9822c71df4dfb85949c239e53839f316e67204a319257aa4f596ba1b6246d",
    ),
    8951: (
        "experiment_8951_v765_matched_corpus_admission.json",
        "sha256:21238c55f69ecf8f4bf8be563d96143e314a6cfaeba6f4c5dfa3fb15580e0fc7",
    ),
    8952: (
        "experiment_8952_v765_matched_calibrated_decisions.json",
        "sha256:e1a4067c3369c6720b08cf1a1820e36edefd7c8c7621da0b47539ab5f784652c",
    ),
    8953: (
        "experiment_8953_v765_gradient_causal_streams.json",
        "sha256:cff93f1c1f76d8eb6a0b9fe185a6d77d21a091879187d2b40e438a8ab87a2ff0",
    ),
    8954: (
        "experiment_8954_v765_gradient_benefit_audit.json",
        "sha256:90a5716c9c3f640d3060c5981fbb73509b9d0929a03b4cc26fa4cb9fe2c74740",
    ),
    8955: (
        "experiment_8955_v765_gradient_complete_cost.json",
        "sha256:00501d6e13a96fcd7d29fd99b3ef724f814a265349ecd2eb25e1dfccd44284ba",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase and census counts so an external supervisor can detect a stall."""
    print(f"[exp8956] phase={phase} completed={completed} pending={pending}", flush=True)


def diagnostic(
    task: Json, ref: Json | None, check: str, field: str, expected: Any, observed: Any
) -> Json:
    """Keep missing evidence distinct from an observed value, including one and zero."""
    return dict(
        upstream=task["id"],
        path=(ref or {}).get("path"),
        hash=(ref or {}).get("sha256"),
        check=check,
        artifact_field=field,
        expected=expected,
        observed=observed,
    )


def admission(value: Json, task: Json) -> str | None:
    """Scientific nulls are usable; failed controls and oracle positives cannot qualify."""
    fields = {
        "verdict_class",
        "acceptance_gates",
        "required_checks_passed",
        "flagged_adversarial",
        "rows",
        "independent_count",
        "owned_statement_coverage",
    }
    if not fields <= value.keys():
        return "missing_field"
    if value["verdict_class"] == "disqualified" or value["required_checks_passed"] is not True:
        return "disqualified_primary"
    if (
        value.get("task_id") != task["id"]
        or value.get("experiment_id") != int(task["id"][3:7])
        or value.get("milestone") != authority.MILESTONE
        or value["flagged_adversarial"] is not False
        or value["owned_statement_coverage"].get("complete") is not True
        or value["verdict_class"] == "positive"
        and (value.get("verifier_is_oracle") or not all(value["acceptance_gates"].values()))
    ):
        return "rejected_producer"
    return None


def collect(raw: Path, sources: Path = ROOT / "results") -> Json:
    """Authenticate immutable task identity, primary bytes and retained operand hashes first."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("authentication_before", 0, 12)
    original = load(dict(path=str(ROOT / "results" / PINS[8944][0]), sha256=PINS[8944][1]))
    tasks = original["full_task_objects"]
    require(canonical_hash(tasks) == authority.PIN and len(tasks) == 13, "ordered_task_contract")
    values, refs, errors = {}, {}, {}
    for n, (name, pin) in PINS.items():
        path = sources / name
        try:
            refs[n] = retain(path, raw, pin)
            values[n] = load(refs[n])
        except (OSError, ValueError):
            errors[n] = "rejected_producer" if path.is_file() else "missing_file"
    catalog = {
        r["sha256"]: r
        for v in values.values()
        for r in references(v)
        if r.get("sha256") and Path(r["path"]).is_file()
    }
    operands: dict[tuple[str, str], Json] = {}
    slots = []
    for index, task in enumerate(tasks[:-1]):
        n = 8944 + index
        value, ref = values.get(n, {}), refs.get(n)
        reason = errors.get(n)
        try:
            for r in references(value):
                key = (r["path"], r["sha256"])
                if key not in operands:
                    operands[key] = resolve(r, catalog)
                if len(operands) % 500 == 0:
                    progress("operands_authenticated", len(operands), 12 - index)
        except (OSError, ValueError, KeyError):
            reason = "rejected_producer"
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        if scheduler:
            skip_check = scheduler_check(value, task, values, refs)
        else:
            reason = reason or admission(value, task)
        check = []
        if reason:
            field = (
                "flagged_adversarial" if value.get("flagged_adversarial") else "producer_contract"
            )
            check.append(
                diagnostic(
                    task,
                    ref or dict(path=str(sources / PINS[n][0]), sha256=PINS[n][1]),
                    reason,
                    field,
                    False if field == "flagged_adversarial" else "qualified_original_producer",
                    value.get(field),
                )
            )
        if scheduler:
            check.append(skip_check)
        slots.append(
            dict(
                experiment_id=n,
                task_id=task["id"],
                reference=ref,
                accounted=ref is not None,
                qualified=reason is None,
                disposition=reason
                or ("authenticated_scheduler_skip" if scheduler else "qualified_primary"),
                gate_check_summary=check,
            )
        )
        progress("slot_authenticated", index + 1, 11 - index)
    work = dict(
        tasks=tasks,
        slots=slots,
        references=list(operands.values()),
        code=[],
        raw=str(raw),
        started_ns=start,
        ended_ns=time.monotonic_ns(),
    )
    commands = run_commands(
        ROOT,
        [
            CommandSpec(
                "publication_gate",
                (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                "owned",
            )
        ],
        log_dir=raw / "publication",
        heartbeat_s=30,
    )
    require(commands[0]["passed"], "publication_gate_execution")
    work["publication_gate"] = json.loads(Path(commands[0]["log_path"]).read_text())
    work["publication_receipts"] = commands
    work["references"].append(reference(Path(commands[0]["log_path"])))
    headline = work["publication_gate"]["gates"]["G1"].get("source")
    for name in [
        "scripts/publication_gate.py",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "ops/conductor-log.md",
        "research-complete.yaml",
        "ops/exclusion_manifest.yaml",
        *(["results/" + headline] if headline else []),
    ]:
        path = ROOT / name
        work["references"].append(retain(path, raw, sha256_file(path)))
    progress("scientific_reductions_before", 0, 5)
    work["scientific_reductions"] = reductions(
        {
            r["experiment_id"]: values[r["experiment_id"]]
            for r in slots
            if r["qualified"] and r["disposition"] == "qualified_primary"
        }
    )
    progress("scientific_reductions_after", 5, 0)
    work["inherited_global_test_debt"] = original["global_test_debt"]
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    return work


def reductions(evidence: dict[int, Json]) -> Json:
    """Reuse row reducers only; never call a generator, learner, scorer or benchmark."""
    from carnot.reporting import matched_depth_control_8948 as depth
    from carnot.reporting import matched_corpus_admission_8951 as corpus
    from carnot.reporting import matched_calibrated_decisions_8952 as decisions
    from carnot.reporting import gradient_benefit_audit_8954 as learning
    from carnot.reporting import gradient_complete_cost_8955 as cost
    from carnot.reporting.v763_capstone_reduction import compare

    result: Json = {}
    protocol = None
    if 8946 in evidence:
        protocol = load(evidence[8946]["raw_shard_hashes"][0])["protocol"]
    if 8948 in evidence and protocol:
        value = evidence[8948]
        measured = depth.reduce(depth.ordered_units(protocol), value["rows"])
        compare(value, measured)
        result["depth"] = measured
    if 8951 in evidence and protocol:
        value = evidence[8951]
        old = load(evidence[8946]["raw_shard_hashes"][0])["evidence"]["old_graphs"]
        units = corpus.source.scheduled(protocol, "A") + corpus.source.scheduled(protocol, "B")
        measured = corpus.reduce(units, value["rows"], old)
        compare(value, measured)
        result["collection"] = measured
    if 8952 in evidence:
        measured = decisions.b.summarize(evidence[8952]["rows"])
        require(measured == evidence[8952]["paired_intervals"], "calibration_comparison_drift")
        result["detector"] = measured
    if 8954 in evidence:
        measured = learning.summarize(evidence[8954]["rows"])
        compare(evidence[8954], measured)
        result["learning"] = measured
    if 8955 in evidence:
        measured = cost.reduce(Path(evidence[8955]["raw_directory"]), "")
        compare(evidence[8955], measured)
        result["cost"] = {k: v for k, v in measured.items() if k != "numerical_training_calls"}
    return result


def scheduler_check(
    value: Json, task: Json, values: dict[int, Json], refs: dict[int, Json]
) -> Json:
    """A skip needs exact task identity and a genuinely unsatisfied bound upstream gate."""
    edge = next(
        e
        for e in task["gated_on"]
        if e["upstream"] == value["failed_upstream"]
        and e["artifact_field"] == value["failed_field"]
    )
    upstream = int(edge["upstream"][3:7])
    observed = value["failed_observed"]
    require(
        value["task_id"] == task["id"]
        and value["experiment_id"] == int(task["id"][3:7])
        and value["failed_evidence_sha256"] == PINS[upstream][1]
        and values[upstream][edge["artifact_field"]] == observed
        and value["failed_expected"] == edge["value"]
        and value["failed_operator"] == edge["op"]
        and observed != edge["value"],
        "scheduler_gate_identity",
    )
    return diagnostic(
        dict(id=edge["upstream"]),
        refs.get(upstream),
        "observed_zero" if observed == 0 else "observed_value",
        edge["artifact_field"],
        edge["value"],
        observed,
    )
