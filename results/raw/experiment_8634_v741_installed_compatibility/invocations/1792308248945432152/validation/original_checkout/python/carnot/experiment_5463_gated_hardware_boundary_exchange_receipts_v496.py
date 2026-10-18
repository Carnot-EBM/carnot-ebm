#!/usr/bin/env python3
"""Exp5463: Exp5462-gated hardware boundary-exchange receipts.

Spec refs: REQ-VERIFY-5463, SCENARIO-VERIFY-5463.

This is a hardware-continuity receipt, not a speedup hunt. The runner first
checks that Exp5462 declared the minimal-core p-bit/p-dit bridge ready, then it
replays the exact bounded workload on CPU and any SSH-reachable boards. Timing
is compared only after result hashes match, and ``hardware_speedup_claim`` is
always false.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5371_pbit_boundary_exchange_schedule_v489 as exp5371
from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420
from carnot import experiment_5462_active_constraint_minimal_core_pdit_bridge_v496 as exp5462


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json"
)
UPSTREAM_EXP5462_RELATIVE_PATH = exp5462.RESULT_RELATIVE_PATH

EXPERIMENT = 5463
EXPERIMENT_ID = "exp5463-gated-hardware-boundary-exchange-receipts-v496"
MILESTONE = "2026.07.496"
RUN_DATE = "20260709"
RANDOM_SEED = 5463
SCHEMA = "carnot.experiment_5463.gated_hardware_boundary_exchange_receipts.v496"
SPEC_REFS = ("REQ-VERIFY-5463", "SCENARIO-VERIFY-5463")
INFERENCE_SUBSTRATE = "cpu_and_reachable_board_timing_receipts"
REPEAT_TARGET = 10
TERMINAL_PREFIXES = ("complete:", "blocked:")

SSH_TIMEOUT_S = 5.0
LOCAL_TIMEOUT_S = 10.0

KV260_SSH_COMMAND = exp5420.KV260_SSH_COMMAND
POLARFIRE_STATUS_COMMAND = exp5420.POLARFIRE_STATUS_COMMAND
HOST_STORAGE_MARKERS = exp5420.HOST_STORAGE_MARKERS
FORBIDDEN_COMMAND_TERMS = exp5420.FORBIDDEN_COMMAND_TERMS + ("rm -rf", "mkfs", "dd ")
EXECUTABLE_BOARDS = ("kv260", "polarfire")
ASSUMPTION_SOURCES = tuple(exp5462.ASSUMPTION_SOURCES)
EXPECTED_FIXTURE_COUNT = exp5462.EXPECTED_FIXTURE_COUNT

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "hardware task must fail fast",
    "gated_upstream_ready": "Exp5462 bridge gate provenance",
    "workload_hash": "matched p-bit/p-dit workload",
    "cpu_result_hash": "CPU correctness before timing",
    "board_result_hashes": "board correctness before timing",
    "board_reachability": "SSH-safe hardware provenance",
    "kv260_ssh_only_checked": "no host SD-card anti-pattern",
    "boundary_exchange_ratio_summary": "partitioned sampling metadata",
    "timing_repeat_counts": "variance support",
    "timing_summary": "measured timing distributions",
    "hashes_match_before_timing_compare": "no false timing comparison",
    "hardware_speedup_claim": "bounded claim",
    "hardware_receipts_ready": "receipt readiness",
    "inference_substrate": "explicit hardware substrate",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON deterministically so hashes survive host differences."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Hash command strings, compact receipts, and deterministic payloads."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping after deterministic serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render a command with shell quoting for compact command receipts."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and preserve unreachable-board failures."""

    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=int(result.returncode),
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )
    except FileNotFoundError as exc:
        return CommandProbe(
            command=tuple(command),
            exit_code=127,
            stderr=str(exc),
            duration_s=round(time.perf_counter() - started, 6),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(
            command=tuple(command),
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )


def load_upstream_gate(root: str | Path = REPO_ROOT) -> JsonDict:
    """Read Exp5462's bridge-ready gate before board probing."""

    path = Path(root) / UPSTREAM_EXP5462_RELATIVE_PATH
    record: JsonDict = {
        "artifact_path": str(UPSTREAM_EXP5462_RELATIVE_PATH),
        "gate_field": "minimal_core_pbit_bridge_ready",
        "gate_value": False,
        "source_status": "missing",
        "source_experiment_id": "",
        "source_reproducibility_checksum": "",
        "readiness_blockers": [],
    }
    if not path.exists():
        return record
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        record["source_status"] = "unreadable"
        return record
    record["gate_value"] = bool(source.get("minimal_core_pbit_bridge_ready"))
    record["source_status"] = str(source.get("status", "unknown"))
    record["source_experiment_id"] = str(source.get("experiment_id", ""))
    record["source_reproducibility_checksum"] = str(
        source.get("reproducibility_checksum", "")
    )
    record["readiness_blockers"] = list(source.get("readiness_blockers", []))
    return record


def _load_upstream_artifact(root: str | Path = REPO_ROOT) -> JsonDict:
    return json.loads((Path(root) / UPSTREAM_EXP5462_RELATIVE_PATH).read_text(encoding="utf-8"))


def _serialize_value(value: Any) -> bool | str:
    if isinstance(value, bool):
        return bool(value)
    return str(value)


def _serialize_fixture(fixture: exp5462.BridgeFixture) -> JsonDict:
    return {
        "fixture_id": fixture.fixture_id,
        "constraint_family": fixture.constraint_family,
        "source_module": fixture.source_module,
        "source_fixture_id": fixture.source_fixture_id,
        "variables": list(fixture.variables),
        "clauses": [list(clause) for clause in fixture.clauses],
        "precedence": [list(edge) for edge in fixture.precedence],
        "assignment_domain": list(fixture.assignment_domain),
        "assignment_costs": [list(row) for row in fixture.assignment_costs],
        "pairwise_costs": [list(row) for row in fixture.pairwise_costs],
        "expected_status": fixture.expected_status,
        "expected_solution": [_serialize_value(value) for value in fixture.expected_solution],
        "active_assumptions": list(fixture.active_assumptions),
        "pbit_control_names": list(fixture.pbit_control_names),
        "pbit_true_assumptions": list(fixture.pbit_true_assumptions),
        "pbit_false_assumptions": list(fixture.pbit_false_assumptions),
        "pbit_samples": [list(sample) for sample in fixture.pbit_samples],
        "pdit_control_names": list(fixture.pdit_control_names),
        "pdit_samples": [list(sample) for sample in fixture.pdit_samples],
        "pdit_domains": [[name, list(domain)] for name, domain in fixture.pdit_domains],
        "pdit_state_codes": [list(row) for row in fixture.pdit_state_codes],
    }


def extract_workload(root: str | Path = REPO_ROOT) -> JsonDict:
    """Extract Exp5462's exact fixture subset, seeds, and expected hashes."""

    upstream = _load_upstream_artifact(root)
    _require(
        upstream.get("minimal_core_pbit_bridge_ready") is True,
        "minimal_core_pbit_bridge_ready",
    )
    fixtures = [_serialize_fixture(fixture) for fixture in exp5462.build_bridge_fixtures()]
    workload: JsonDict = {
        "source_artifact": str(UPSTREAM_EXP5462_RELATIVE_PATH),
        "source_experiment_id": upstream["experiment_id"],
        "source_reproducibility_checksum": upstream["reproducibility_checksum"],
        "fixture_subset": fixtures,
        "fixture_subset_ids": [fixture["fixture_id"] for fixture in fixtures],
        "assumption_sources": list(ASSUMPTION_SOURCES),
        "seeds": {
            "upstream_random_seed": int(upstream["random_seed"]),
            "receipt_random_seed": RANDOM_SEED,
        },
        "claim_boundary": "Exp5462 exact solver replay; no sampler or speedup claim",
    }
    replay = replay_workload(workload)
    workload["expected_result_hashes"] = {
        "aggregate": workload_result_hash(replay),
        "rows": {
            f"{row['fixture_id']}::{row['assumption_source']}": sha256_json(row)
            for row in replay["rows"]
        },
        "derived_from": str(UPSTREAM_EXP5462_RELATIVE_PATH),
        "upstream_row_records_checksum": sha256_json(
            {"row_records": upstream.get("row_records", [])}
        ),
    }
    workload["workload_hash"] = sha256_json(
        {
            "source_artifact": workload["source_artifact"],
            "source_reproducibility_checksum": workload["source_reproducibility_checksum"],
            "fixture_subset": workload["fixture_subset"],
            "assumption_sources": workload["assumption_sources"],
            "seeds": workload["seeds"],
        }
    )
    return workload


def replay_workload(workload: Mapping[str, Any]) -> JsonDict:
    """Replay Exp5462's p-bit/p-dit rows independent of timing receipts."""

    rows = [
        _evaluate_fixture_source(fixture, str(source))
        for fixture in workload["fixture_subset"]
        for source in workload["assumption_sources"]
    ]
    return {
        "seed": int(workload["seeds"]["upstream_random_seed"]),
        "fixture_subset_ids": list(workload["fixture_subset_ids"]),
        "assumption_sources": list(workload["assumption_sources"]),
        "rows": rows,
    }


def workload_result_hash(replay: Mapping[str, Any]) -> str:
    """Hash exact replay output independently from timing receipts."""

    return sha256_json(dict(replay))


def _evaluate_fixture_source(fixture: Mapping[str, Any], source: str) -> JsonDict:
    assumptions = _assumptions_for_source(fixture, source)
    baseline = _solve_exact(fixture, ())
    attempt = _solve_exact(fixture, assumptions)
    final = attempt
    decision = "accepted"
    fallback_used = False
    if attempt["status"] != baseline["status"]:
        decision = "rejected"
        fallback_used = True
        final = baseline
    elif attempt["status"] == "sat" and attempt["solution"] != baseline["solution"]:
        decision = "overwritten"
        fallback_used = True
        final = baseline
    return {
        "fixture_id": fixture["fixture_id"],
        "constraint_family": fixture["constraint_family"],
        "assumption_source": source,
        "assumptions": list(assumptions),
        "assumption_decision": decision,
        "fallback_used": fallback_used,
        "baseline_status": baseline["status"],
        "assumption_attempt_status": attempt["status"],
        "final_status": final["status"],
        "baseline_solution": baseline["solution"],
        "assumption_solution": attempt["solution"],
        "final_solution": final["solution"],
        "baseline_objective_value": baseline["objective_value"],
        "final_objective_value": final["objective_value"],
        "final_matches_exact": final["status"] == baseline["status"]
        and final["solution"] == baseline["solution"],
        "solution_valid": _solution_valid(fixture, final),
        "objective_preserved": final["objective_value"] == baseline["objective_value"],
        "solver_authoritative": True,
    }


def _assumptions_for_source(fixture: Mapping[str, Any], source: str) -> tuple[str, ...]:
    if source == "active_constraint":
        return tuple(str(item) for item in fixture["active_assumptions"])
    if source == "pbit_binary":
        return _pbit_binary_assumptions(fixture)
    return _pdit_multistate_assumptions(fixture)


def _pbit_binary_assumptions(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    threshold = 0.75
    assumptions: list[str] = []
    samples = list(fixture["pbit_samples"])
    for index in range(len(fixture["pbit_control_names"])):
        true_count = sum(int(bool(sample[index])) for sample in samples)
        true_rate = true_count / len(samples)
        if true_rate >= threshold:
            assumptions.append(str(fixture["pbit_true_assumptions"][index]))
        elif true_rate <= 1.0 - threshold:
            assumptions.append(str(fixture["pbit_false_assumptions"][index]))
    return tuple(assumptions)


def _pdit_multistate_assumptions(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    threshold = 0.75
    assumptions: list[str] = []
    samples = list(fixture["pdit_samples"])
    for index, control in enumerate(fixture["pdit_control_names"]):
        counts: dict[str, int] = {}
        for sample in samples:
            value = str(sample[index])
            counts[value] = counts.get(value, 0) + 1
        value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if count / len(samples) >= threshold:
            assumption = _pdit_value_to_assumption(fixture, str(control), value)
            if assumption:
                assumptions.append(assumption)
    return tuple(assumptions)


def _pdit_value_to_assumption(
    fixture: Mapping[str, Any],
    control: str,
    value: str,
) -> str:
    if value in {"unknown", "abstain"}:
        return ""
    if fixture["constraint_family"] == "sat":
        return control if value == "true" else f"!{control}"
    if fixture["constraint_family"] == "assignment":
        return f"{control}={value}"
    position = int(control.removeprefix("pos"))
    return f"{value}@{position}"


def _solve_exact(fixture: Mapping[str, Any], assumptions: Sequence[str]) -> JsonDict:
    best: tuple[Any, ...] | None = None
    best_score: int | None = None
    for candidate in _candidate_space(fixture):
        if not _satisfies_assumptions(fixture, candidate, assumptions):
            continue
        if _constraint_violation_count(fixture, candidate) != 0:
            continue
        score = _objective_value(fixture, candidate)
        if best is None or score < int(best_score):
            best = candidate
            best_score = score
    if best is None:
        return {"status": "unsat", "solution": None, "objective_value": None}
    return {
        "status": "sat",
        "solution": _serialize_solution(fixture, best),
        "objective_value": int(best_score),
    }


def _candidate_space(fixture: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    if fixture["constraint_family"] == "sat":
        return tuple(itertools.product((False, True), repeat=len(fixture["variables"])))
    if fixture["constraint_family"] == "assignment":
        return tuple(itertools.permutations(tuple(fixture["assignment_domain"])))
    return tuple(itertools.permutations(tuple(fixture["variables"])))


def _satisfies_assumptions(
    fixture: Mapping[str, Any],
    candidate: Sequence[Any],
    assumptions: Sequence[str],
) -> bool:
    variables = list(fixture["variables"])
    for assumption in assumptions:
        if fixture["constraint_family"] == "sat":
            if assumption.startswith("!"):
                if bool(candidate[variables.index(assumption[1:])]) is not False:
                    return False
            elif bool(candidate[variables.index(assumption)]) is not True:
                return False
        elif fixture["constraint_family"] == "assignment":
            worker, job = assumption.split("=", 1)
            if str(candidate[variables.index(worker)]) != job:
                return False
        else:
            action, position_text = assumption.rsplit("@", 1)
            if str(candidate[int(position_text)]) != action:
                return False
    return True


def _constraint_violation_count(fixture: Mapping[str, Any], candidate: Sequence[Any]) -> int:
    if fixture["constraint_family"] == "sat":
        return sum(int(not _clause_satisfied(candidate, clause)) for clause in fixture["clauses"])
    if fixture["constraint_family"] == "assignment":
        return int(set(str(job) for job in candidate) != set(fixture["assignment_domain"]))
    positions = {str(action): index for index, action in enumerate(candidate)}
    return sum(int(positions[before] > positions[after]) for before, after in fixture["precedence"])


def _clause_satisfied(candidate: Sequence[Any], clause: Sequence[int]) -> bool:
    for literal in clause:
        value = bool(candidate[abs(int(literal)) - 1])
        if value == (int(literal) > 0):
            return True
    return False


def _objective_value(fixture: Mapping[str, Any], candidate: Sequence[Any]) -> int:
    expected = list(fixture["expected_solution"])
    if fixture["constraint_family"] == "sat":
        return sum(
            (index + 1) * int(bool(candidate[index]) != expected[index])
            for index in range(len(candidate))
        )
    if fixture["constraint_family"] == "assignment":
        return _assignment_objective(fixture, candidate)
    preferred = {str(value): index for index, value in enumerate(expected)}
    return sum(
        (index + 1) * abs(index - preferred[str(value)])
        for index, value in enumerate(candidate)
    )


def _assignment_objective(fixture: Mapping[str, Any], candidate: Sequence[Any]) -> int:
    assignment = {
        str(worker): str(candidate[index]) for index, worker in enumerate(fixture["variables"])
    }
    cost_lookup = {
        (str(worker), str(job)): int(cost)
        for worker, job, cost in fixture["assignment_costs"]
    }
    total = sum(cost_lookup[(str(worker), assignment[str(worker)])] for worker in fixture["variables"])
    for left_worker, left_job, right_worker, right_job, cost in fixture["pairwise_costs"]:
        if assignment[str(left_worker)] == str(left_job) and assignment[str(right_worker)] == str(right_job):
            total += int(cost)
    return total


def _serialize_solution(
    fixture: Mapping[str, Any],
    solution: Sequence[Any] | None,
) -> list[bool] | list[str] | None:
    if solution is None:
        return None
    if fixture["constraint_family"] == "sat":
        return [bool(value) for value in solution]
    return [str(value) for value in solution]


def _solution_valid(fixture: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    if metrics["status"] == "unsat":
        return fixture["expected_status"] == "unsat"
    solution = metrics["solution"]
    if solution is None:
        return False
    return _constraint_violation_count(fixture, tuple(solution)) == 0


def timing_distribution(values: Sequence[float]) -> JsonDict:
    """Summarize repeat timings with mean, median, variance, min, and max."""

    timings = [float(value) for value in values]
    if not timings:
        return {
            "count": 0,
            "min_s": 0.0,
            "max_s": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "variance_s2": 0.0,
        }
    variance = statistics.pvariance(timings) if len(timings) > 1 else 0.0
    return {
        "count": len(timings),
        "min_s": round(min(timings), 9),
        "max_s": round(max(timings), 9),
        "mean_s": round(sum(timings) / len(timings), 9),
        "median_s": round(float(statistics.median(timings)), 9),
        "variance_s2": round(float(variance), 12),
    }


def cpu_timing_receipt(
    workload: Mapping[str, Any],
    *,
    repeat_count: int = REPEAT_TARGET,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Repeat the Exp5462 replay workload on CPU for timing variance."""

    timings: list[float] = []
    repeats: list[JsonDict] = []
    for repeat_index in range(repeat_count):
        started = clock()
        replay = replay_workload(workload)
        elapsed = round(max(clock() - started, 0.0), 9)
        repeat_hash = workload_result_hash(replay)
        timings.append(elapsed)
        repeats.append(
            {
                "repeat_index": repeat_index + 1,
                "seed": int(workload["seeds"]["upstream_random_seed"]),
                "workload_hash": workload["workload_hash"],
                "result_hash": repeat_hash,
                "wall_time_s": elapsed,
                "result_matches_expected": repeat_hash
                == workload["expected_result_hashes"]["aggregate"],
            }
        )
    result_hashes = {str(item["result_hash"]) for item in repeats}
    result_hash_value = result_hashes.pop() if len(result_hashes) == 1 else ""
    return {
        "kind": "cpu_timing",
        "substrate": "cpu_exact_exp5462_pbit_pdit_replay",
        "seed": int(workload["seeds"]["upstream_random_seed"]),
        "workload_hash": workload["workload_hash"],
        "result_hash": result_hash_value,
        "repeat_count": len(repeats),
        "repeat_timings_s": timings,
        "timing_distribution": timing_distribution(timings),
        "repeats": repeats,
    }


def _remote_workload_source(board: str, workload: Mapping[str, Any]) -> str:
    payload = {
        "board_identity": board,
        "fixture_subset": workload["fixture_subset"],
        "fixture_subset_ids": workload["fixture_subset_ids"],
        "assumption_sources": workload["assumption_sources"],
        "seed": int(workload["seeds"]["upstream_random_seed"]),
        "workload_hash": workload["workload_hash"],
        "expected_result_hash": workload["expected_result_hashes"]["aggregate"],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return "\n".join(
        [
            "import hashlib,itertools,json,time",
            f"payload=json.loads({payload_json!r})",
            "started=time.perf_counter()",
            "def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'))",
            "def h(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()",
            "def pdit_assumption(f,control,value):",
            "    if value in ('unknown','abstain'): return ''",
            "    if f['constraint_family']=='sat': return control if value=='true' else '!'+control",
            "    if f['constraint_family']=='assignment': return control+'='+value",
            "    return value+'@'+control.removeprefix('pos')",
            "def assumptions(f,source):",
            "    if source=='active_constraint': return list(f['active_assumptions'])",
            "    out=[]",
            "    if source=='pbit_binary':",
            "        samples=f['pbit_samples']",
            "        for i in range(len(f['pbit_control_names'])):",
            "            rate=sum(int(bool(s[i])) for s in samples)/len(samples)",
            "            if rate>=0.75: out.append(f['pbit_true_assumptions'][i])",
            "            elif rate<=0.25: out.append(f['pbit_false_assumptions'][i])",
            "        return out",
            "    samples=f['pdit_samples']",
            "    for i,control in enumerate(f['pdit_control_names']):",
            "        counts={}",
            "        for sample in samples: counts[str(sample[i])]=counts.get(str(sample[i]),0)+1",
            "        value,count=sorted(counts.items(),key=lambda item:(-item[1],item[0]))[0]",
            "        if count/len(samples)>=0.75:",
            "            item=pdit_assumption(f,str(control),value)",
            "            if item: out.append(item)",
            "    return out",
            "def candidate_space(f):",
            "    if f['constraint_family']=='sat': return itertools.product((False,True), repeat=len(f['variables']))",
            "    if f['constraint_family']=='assignment': return itertools.permutations(f['assignment_domain'])",
            "    return itertools.permutations(f['variables'])",
            "def sat_assumptions(f,c,asm):",
            "    variables=f['variables']",
            "    for a in asm:",
            "        if f['constraint_family']=='sat':",
            "            if a.startswith('!'):",
            "                if bool(c[variables.index(a[1:])]) is not False: return False",
            "            elif bool(c[variables.index(a)]) is not True: return False",
            "        elif f['constraint_family']=='assignment':",
            "            worker,job=a.split('=',1)",
            "            if str(c[variables.index(worker)])!=job: return False",
            "        else:",
            "            name,pos=a.rsplit('@',1)",
            "            if str(c[int(pos)])!=name: return False",
            "    return True",
            "def clause_ok(c,clause):",
            "    for lit in clause:",
            "        val=bool(c[abs(int(lit))-1])",
            "        if val==(int(lit)>0): return True",
            "    return False",
            "def violations(f,c):",
            "    if f['constraint_family']=='sat': return sum(int(not clause_ok(c,cl)) for cl in f['clauses'])",
            "    if f['constraint_family']=='assignment': return int(set(str(job) for job in c)!=set(f['assignment_domain']))",
            "    pos={str(a):i for i,a in enumerate(c)}",
            "    return sum(int(pos[b]>pos[a]) for b,a in f['precedence'])",
            "def objective(f,c):",
            "    expected=f['expected_solution']",
            "    if f['constraint_family']=='sat': return sum((i+1)*int(bool(c[i])!=expected[i]) for i in range(len(c)))",
            "    if f['constraint_family']=='assignment':",
            "        assign={str(w):str(c[i]) for i,w in enumerate(f['variables'])}",
            "        costs={(str(w),str(j)):int(cost) for w,j,cost in f['assignment_costs']}",
            "        total=sum(costs[(str(w),assign[str(w)])] for w in f['variables'])",
            "        for lw,lj,rw,rj,cost in f['pairwise_costs']:",
            "            if assign[str(lw)]==str(lj) and assign[str(rw)]==str(rj): total+=int(cost)",
            "        return total",
            "    pref={str(v):i for i,v in enumerate(expected)}",
            "    return sum((i+1)*abs(i-pref[str(v)]) for i,v in enumerate(c))",
            "def sol(f,c):",
            "    if c is None: return None",
            "    if f['constraint_family']=='sat': return [bool(v) for v in c]",
            "    return [str(v) for v in c]",
            "def solve(f,asm):",
            "    best=None; best_score=None",
            "    for c in candidate_space(f):",
            "        if not sat_assumptions(f,c,asm): continue",
            "        if violations(f,c)!=0: continue",
            "        score=objective(f,c)",
            "        if best is None or score<int(best_score): best=c; best_score=score",
            "    if best is None: return {'status':'unsat','solution':None,'objective_value':None}",
            "    return {'status':'sat','solution':sol(f,best),'objective_value':int(best_score)}",
            "def valid(f,m):",
            "    if m['status']=='unsat': return f['expected_status']=='unsat'",
            "    return m['solution'] is not None and violations(f,tuple(m['solution']))==0",
            "rows=[]",
            "for f in payload['fixture_subset']:",
            "    for source in payload['assumption_sources']:",
            "        asm=assumptions(f,source); base=solve(f,()); attempt=solve(f,asm); final=attempt; decision='accepted'; fallback=False",
            "        if attempt['status']!=base['status']: decision='rejected'; fallback=True; final=base",
            "        elif attempt['status']=='sat' and attempt['solution']!=base['solution']: decision='overwritten'; fallback=True; final=base",
            "        rows.append({'fixture_id':f['fixture_id'],'constraint_family':f['constraint_family'],'assumption_source':source,'assumptions':asm,'assumption_decision':decision,'fallback_used':fallback,'baseline_status':base['status'],'assumption_attempt_status':attempt['status'],'final_status':final['status'],'baseline_solution':base['solution'],'assumption_solution':attempt['solution'],'final_solution':final['solution'],'baseline_objective_value':base['objective_value'],'final_objective_value':final['objective_value'],'final_matches_exact':final['status']==base['status'] and final['solution']==base['solution'],'solution_valid':valid(f,final),'objective_preserved':final['objective_value']==base['objective_value'],'solver_authoritative':True})",
            "replay={'seed':payload['seed'],'fixture_subset_ids':payload['fixture_subset_ids'],'assumption_sources':payload['assumption_sources'],'rows':rows}",
            "receipt={'board_identity':payload['board_identity'],'board_local':True,'fixture_subset':payload['fixture_subset_ids'],'seed':payload['seed'],'workload_hash':payload['workload_hash'],'result_hash':h(replay),'wall_time_s':round(time.perf_counter()-started,9),'boundary_exchange_supported':False}",
            "print(json.dumps(receipt,sort_keys=True))",
        ]
    )


def board_workload_command(board: str, workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the SSH command for a board-local Exp5462 replay."""

    target = {"kv260": "kria", "polarfire": "polarfire"}[board]
    remote = "python3 - <<'PY'\n" + _remote_workload_source(board, workload) + "\nPY"
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        target,
        remote,
    )


def parse_board_workload_stdout(
    stdout: str,
    workload: Mapping[str, Any],
    board: str,
) -> tuple[JsonDict | None, str | None]:
    """Parse and validate one board-local workload receipt."""

    parsed: Any | None = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            break
    if not isinstance(parsed, Mapping):
        return None, "workload stdout is not valid JSON"

    receipt = dict(parsed)
    errors: list[str] = []
    if receipt.get("board_identity") != board:
        errors.append("board_identity mismatch")
    if receipt.get("workload_hash") != workload["workload_hash"]:
        errors.append("workload_hash mismatch")
    if receipt.get("seed") != workload["seeds"]["upstream_random_seed"]:
        errors.append("seed mismatch")
    if receipt.get("fixture_subset") != workload["fixture_subset_ids"]:
        errors.append("fixture_subset mismatch")
    if receipt.get("result_hash") != workload["expected_result_hashes"]["aggregate"]:
        errors.append("result_hash mismatch")
    if receipt.get("board_local") is not True:
        errors.append("board_local missing")
    if not isinstance(receipt.get("wall_time_s"), int | float) or receipt["wall_time_s"] < 0:
        errors.append("wall_time_s invalid")
    return receipt, "; ".join(errors) if errors else None


def command_receipt(
    probe: CommandProbe,
    *,
    kind: str,
    timeout_s: float,
    outcome: str,
) -> JsonDict:
    """Build a compact command receipt without storing full board logs."""

    return exp5420.command_receipt(probe, kind=kind, timeout_s=timeout_s, outcome=outcome)


def _board_result_hash(valid_attempts: Sequence[Mapping[str, Any]], repeat_count: int) -> str:
    hashes = [
        str(attempt["result_hash"])
        for attempt in valid_attempts
        if isinstance(attempt.get("result_hash"), str)
    ]
    return hashes[0] if len(hashes) == repeat_count and len(set(hashes)) == 1 else ""


def _collect_one_board(
    board: str,
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    status_probe: CommandProbe,
    command_runner: CommandRunner,
) -> JsonDict:
    reachable = status_probe.exit_code == 0
    attempts: list[JsonDict] = []
    if reachable:
        command = board_workload_command(board, workload)
        for repeat_index in range(repeat_count):
            probe = command_runner(command, SSH_TIMEOUT_S)
            receipt, parse_error = parse_board_workload_stdout(probe.stdout, workload, board)
            valid = probe.exit_code == 0 and receipt is not None and parse_error is None
            attempts.append(
                {
                    "repeat_index": repeat_index + 1,
                    "valid": valid,
                    "parse_error": parse_error,
                    "workload_hash": receipt.get("workload_hash")
                    if isinstance(receipt, Mapping)
                    else None,
                    "result_hash": receipt.get("result_hash")
                    if isinstance(receipt, Mapping)
                    else None,
                    "wall_time_s": receipt.get("wall_time_s")
                    if isinstance(receipt, Mapping)
                    else None,
                    "receipt": receipt,
                    "command_receipt": command_receipt(
                        probe,
                        kind=f"{board}_exp5462_pbit_pdit_timing_repeat_{repeat_index + 1}",
                        timeout_s=SSH_TIMEOUT_S,
                        outcome="valid_repeat" if valid else "invalid_repeat",
                    ),
                }
            )
    valid_attempts = [attempt for attempt in attempts if attempt["valid"] is True]
    timings = [
        float(attempt["wall_time_s"])
        for attempt in valid_attempts
        if isinstance(attempt.get("wall_time_s"), int | float)
    ]
    result_hash_value = _board_result_hash(valid_attempts, repeat_count)
    return {
        "board": board,
        "reachable": reachable,
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "timings": timings,
        "result_hash": result_hash_value,
        "repeat_count": len(valid_attempts),
        "invalid_repeat_count": len(attempts) - len(valid_attempts),
        "timing_distribution": timing_distribution(timings),
        "workload_hash_match": bool(
            len(valid_attempts) == repeat_count
            and all(attempt.get("workload_hash") == workload["workload_hash"] for attempt in valid_attempts)
        ),
        "result_hash_match": result_hash_value
        == workload["expected_result_hashes"]["aggregate"],
    }


def collect_board_receipts(
    *,
    workload: Mapping[str, Any],
    repeat_count: int,
    command_runner: CommandRunner,
) -> JsonDict:
    """Collect SSH-safe KV260 and PolarFire receipts."""

    command_receipts: list[JsonDict] = []

    kv_probe = command_runner(KV260_SSH_COMMAND, SSH_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            kv_probe,
            kind="kv260_ssh_only_reachability",
            timeout_s=SSH_TIMEOUT_S,
            outcome="reachable" if kv_probe.exit_code == 0 else "blocked",
        )
    )
    kv = _collect_one_board(
        "kv260",
        workload=workload,
        repeat_count=repeat_count,
        status_probe=kv_probe,
        command_runner=command_runner,
    )
    command_receipts.extend(attempt["command_receipt"] for attempt in kv["attempts"])

    pf_probe = command_runner(POLARFIRE_STATUS_COMMAND, SSH_TIMEOUT_S)
    command_receipts.append(
        command_receipt(
            pf_probe,
            kind="polarfire_ssh_reachability",
            timeout_s=SSH_TIMEOUT_S,
            outcome="reachable" if pf_probe.exit_code == 0 else "blocked",
        )
    )
    pf = _collect_one_board(
        "polarfire",
        workload=workload,
        repeat_count=repeat_count,
        status_probe=pf_probe,
        command_runner=command_runner,
    )
    command_receipts.extend(attempt["command_receipt"] for attempt in pf["attempts"])

    board_results = {"kv260": kv, "polarfire": pf}
    board_result_hashes = {
        "kv260": str(kv["result_hash"]),
        "polarfire": str(pf["result_hash"]),
    }
    timing_repeat_counts = {
        "kv260": int(kv["repeat_count"]),
        "polarfire": int(pf["repeat_count"]),
    }
    timing_summary = {
        "kv260": kv["timing_distribution"],
        "polarfire": pf["timing_distribution"],
    }
    board_reachability = {
        "kv260": {
            "reachable": bool(kv["reachable"]),
            "check_method": "ssh_only",
            "identity": "kria",
            "command": command_to_string(KV260_SSH_COMMAND),
            "workload_execution_attempted": bool(kv["reachable"]),
            "blocked_reason": None if kv["reachable"] else "blocked_kv260_ssh",
        },
        "polarfire": {
            "reachable": bool(pf["reachable"]),
            "check_method": "ssh",
            "identity": "polarfire",
            "command": command_to_string(POLARFIRE_STATUS_COMMAND),
            "workload_execution_attempted": bool(pf["reachable"]),
            "blocked_reason": None if pf["reachable"] else "blocked_polarfire_ssh",
        },
    }
    return {
        "board_results": board_results,
        "board_result_hashes": board_result_hashes,
        "board_reachability": board_reachability,
        "timing_repeat_counts": timing_repeat_counts,
        "timing_summary": timing_summary,
        "command_receipts": command_receipts,
    }


def boundary_exchange_ratio_summary(*, enabled: bool = True) -> JsonDict:
    """Summarize simulated partitioned-sampling boundary-exchange metadata."""

    if not enabled:
        return {
            "source_artifact": str(exp5371.RESULT_RELATIVE_PATH),
            "source": "not_checked_upstream_gate",
            "simulation_only": True,
            "board_supported": False,
            "eta_values": [],
            "eta_threshold_estimate": None,
            "timing_ratios_present": False,
            "baseline_comparison_present": False,
            "repeat_counts": {"cpu_simulated": 0, "board_supported": 0},
            "ratio_summary_by_eta": {},
        }
    diagnostic = exp5371.run_boundary_diagnostic()
    return {
        "source_artifact": str(exp5371.RESULT_RELATIVE_PATH),
        "source": "exp5371_cpu_simulated_partitioned_sampling",
        "simulation_only": True,
        "board_supported": False,
        "eta_values": list(exp5371.ETA_VALUES),
        "eta_threshold_estimate": diagnostic["eta_threshold_estimate"],
        "timing_ratios_present": bool(diagnostic["timing_ratios_present"]),
        "baseline_comparison_present": bool(diagnostic["baseline_comparison_present"]),
        "repeat_counts": {
            "cpu_simulated": len(diagnostic["boundary_exchange_results"]),
            "board_supported": 0,
        },
        "ratio_summary_by_eta": diagnostic["eta_summaries"],
    }


def default_tests_run() -> list[JsonDict]:
    """Keep CLI artifacts schema-valid before external tests are attached."""

    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    return [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]


def readiness_blockers(
    *,
    gated_upstream_ready: bool,
    cpu_repeat_count: int,
    executable_board_count: int,
    hashes_match_before_timing_compare: bool,
    boundary_exchange_present: bool,
) -> list[str]:
    """Explain why Exp5463 hardware receipts are not complete."""

    blockers: list[str] = []
    if not gated_upstream_ready:
        blockers.append("minimal_core_pbit_bridge_not_ready")
    if cpu_repeat_count < REPEAT_TARGET:
        blockers.append("cpu_repeat_count_below_threshold")
    if not boundary_exchange_present:
        blockers.append("boundary_exchange_ratio_missing")
    if executable_board_count == 0:
        blockers.append("blocked_board_unreachable")
    elif not hashes_match_before_timing_compare:
        blockers.append("board_hash_or_repeat_mismatch")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """State terminal receipt status while refusing speedup claims."""

    if ready:
        return (
            "complete: Exp5462-gated CPU and reachable-board p-bit/p-dit timing "
            "receipts are hash-matched before timing comparison; "
            "hardware_speedup_claim=false"
        )
    joined = ",".join(blockers) if blockers else "hardware_receipts_not_ready"
    return f"blocked: {joined}; hardware_speedup_claim=false"


def _empty_board_reachability() -> JsonDict:
    return {
        "kv260": {
            "reachable": False,
            "check_method": "ssh_only",
            "identity": "kria",
            "command": command_to_string(KV260_SSH_COMMAND),
            "workload_execution_attempted": False,
            "blocked_reason": "not_checked_upstream_gate",
        },
        "polarfire": {
            "reachable": False,
            "check_method": "ssh",
            "identity": "polarfire",
            "command": command_to_string(POLARFIRE_STATUS_COMMAND),
            "workload_execution_attempted": False,
            "blocked_reason": "not_checked_upstream_gate",
        },
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp5463 artifact from CPU and reachable-board receipts."""

    started = clock()
    upstream_gate = load_upstream_gate(root)
    gated_upstream_ready = bool(upstream_gate["gate_value"])
    selected_workload: JsonDict = {}
    cpu_receipt: JsonDict = {
        "kind": "cpu_timing",
        "result_hash": "",
        "repeat_count": 0,
        "timing_distribution": timing_distribution([]),
    }
    command_receipts: list[JsonDict] = []
    board_result_hashes = {"kv260": "", "polarfire": ""}
    board_reachability = _empty_board_reachability()
    board_repeat_counts = {"kv260": 0, "polarfire": 0}
    board_timing_summary = {
        "kv260": timing_distribution([]),
        "polarfire": timing_distribution([]),
    }
    board_results: dict[str, Any] = {}
    boundary_summary = boundary_exchange_ratio_summary(enabled=False)

    if gated_upstream_ready:
        selected_workload = extract_workload(root)
        cpu_receipt = cpu_timing_receipt(selected_workload, repeat_count=REPEAT_TARGET, clock=clock)
        boundary_summary = boundary_exchange_ratio_summary(enabled=True)
        board_receipts = collect_board_receipts(
            workload=selected_workload,
            repeat_count=REPEAT_TARGET,
            command_runner=command_runner,
        )
        command_receipts = board_receipts["command_receipts"]
        board_result_hashes = board_receipts["board_result_hashes"]
        board_reachability = board_receipts["board_reachability"]
        board_repeat_counts = board_receipts["timing_repeat_counts"]
        board_timing_summary = board_receipts["timing_summary"]
        board_results = board_receipts["board_results"]

    cpu_repeat_count = int(cpu_receipt["repeat_count"])
    executable_boards = [
        board
        for board in EXECUTABLE_BOARDS
        if board_reachability[board]["workload_execution_attempted"] is True
    ]
    repeated_hash_matched_boards = [
        board
        for board in executable_boards
        if board_repeat_counts[board] >= REPEAT_TARGET
        and board_result_hashes[board] == cpu_receipt["result_hash"]
        and board_results.get(board, {}).get("workload_hash_match") is True
        and board_results.get(board, {}).get("result_hash_match") is True
    ]
    hashes_match_before_timing_compare = bool(repeated_hash_matched_boards) and len(
        repeated_hash_matched_boards
    ) == len(executable_boards)
    boundary_present = _boundary_summary_present(
        boundary_summary,
        gated_upstream_ready=gated_upstream_ready,
    )
    blockers = readiness_blockers(
        gated_upstream_ready=gated_upstream_ready,
        cpu_repeat_count=cpu_repeat_count,
        executable_board_count=len(executable_boards),
        hashes_match_before_timing_compare=hashes_match_before_timing_compare,
        boundary_exchange_present=boundary_present,
    )
    hardware_receipts_ready = bool(
        gated_upstream_ready
        and cpu_repeat_count >= REPEAT_TARGET
        and hashes_match_before_timing_compare
        and boundary_present
        and not blockers
    )
    timing_repeat_counts = {"cpu": cpu_repeat_count, **board_repeat_counts}
    timing_summary = {"cpu": cpu_receipt["timing_distribution"], **board_timing_summary}
    timing_comparison = {
        "comparison_performed": hardware_receipts_ready,
        "hashes_match_before_timing_compare": hashes_match_before_timing_compare,
        "boards_compared": repeated_hash_matched_boards if hardware_receipts_ready else [],
        "hardware_speedup_claim": False,
        "ratios": {
            board: round(
                float(timing_summary[board]["mean_s"]) / float(timing_summary["cpu"]["mean_s"]),
                9,
            )
            for board in repeated_hash_matched_boards
            if float(timing_summary["cpu"]["mean_s"]) > 0.0
        },
        "comparison_boundary": (
            "Ratios are matched timing facts only and are not hardware speedup claims."
        ),
    }

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(clock() - started, 0.0), 6),
        "commit": commit,
        "preconditions_checked": True,
        "upstream_gate": upstream_gate,
        "gated_upstream_ready": gated_upstream_ready,
        "selected_workload": selected_workload,
        "workload_hash": str(selected_workload.get("workload_hash", "")),
        "cpu_result_hash": str(cpu_receipt["result_hash"]),
        "board_result_hashes": board_result_hashes,
        "board_reachability": board_reachability,
        "kv260_ssh_only_checked": bool(gated_upstream_ready),
        "boundary_exchange_ratio_summary": boundary_summary,
        "timing_repeat_counts": timing_repeat_counts,
        "timing_summary": timing_summary,
        "hashes_match_before_timing_compare": hashes_match_before_timing_compare,
        "hardware_speedup_claim": False,
        "hardware_receipts_ready": hardware_receipts_ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(hardware_receipts_ready, blockers),
        "readiness_blockers": blockers,
        "timing_receipts": {
            "cpu": cpu_receipt,
            "boards": {
                board: {
                    "repeat_count": board_repeat_counts[board],
                    "result_hash": board_result_hashes[board],
                    "timing_distribution": board_timing_summary[board],
                    "attempts": board_results.get(board, {}).get("attempts", []),
                }
                for board in EXECUTABLE_BOARDS
            },
        },
        "timing_comparison": timing_comparison,
        "command_receipts": command_receipts,
        "claim_refusal": (
            "No hardware speedup is claimed; correctness hashes must match before timing "
            "facts are compared."
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalize_tests(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _boundary_summary_present(
    summary: Mapping[str, Any],
    *,
    gated_upstream_ready: bool,
) -> bool:
    if not gated_upstream_ready:
        return True
    return bool(
        summary.get("eta_values")
        and summary.get("timing_ratios_present") is True
        and summary.get("baseline_comparison_present") is True
        and int(summary.get("repeat_counts", {}).get("cpu_simulated", 0)) > 0
    )


def _artifact_mentions_host_storage(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return any(marker in encoded for marker in HOST_STORAGE_MARKERS)


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for index, item in enumerate(tests_run):
        _require(isinstance(item, Mapping), f"tests_run[{index}]")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run command")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run outcome")


def _validate_command_receipts(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("command_receipts")
    if artifact.get("gated_upstream_ready") is False:
        _require(receipts == [], "command_receipts")
        return
    _require(isinstance(receipts, list) and receipts, "command_receipts")
    kv260_count = 0
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipt")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command_receipt command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command hash")
        if receipt.get("kind") == "kv260_ssh_only_reachability":
            kv260_count += 1
            _require(
                command == command_to_string(KV260_SSH_COMMAND),
                "KV260 command must be exact SSH-only reachability precondition",
            )
    _require(kv260_count == 1, "exactly one KV260 SSH receipt required")


def _validate_boundary_summary(
    summary: Any,
    *,
    gated_upstream_ready: bool,
) -> None:
    _require(isinstance(summary, Mapping), "boundary_exchange_ratio_summary")
    if not gated_upstream_ready:
        _require(summary.get("eta_values") == [], "boundary_exchange_ratio_summary")
        return
    _require(summary.get("simulation_only") is True, "boundary_exchange_ratio_summary")
    _require(summary.get("board_supported") is False, "boundary_exchange_ratio_summary")
    _require(list(summary.get("eta_values", [])) == list(exp5371.ETA_VALUES), "boundary_exchange_ratio_summary")
    _require(summary.get("timing_ratios_present") is True, "boundary_exchange_ratio_summary")
    _require(summary.get("baseline_comparison_present") is True, "boundary_exchange_ratio_summary")
    _require(
        int(summary.get("repeat_counts", {}).get("cpu_simulated", 0)) > 0,
        "boundary_exchange_ratio_summary",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on mismatched hashes, unsafe probes, or speedup claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("preconditions_checked") is True, "preconditions_checked")
    _require(isinstance(artifact.get("gated_upstream_ready"), bool), "gated_upstream_ready")
    _require(isinstance(artifact.get("workload_hash"), str), "workload_hash")
    _require(isinstance(artifact.get("cpu_result_hash"), str), "cpu_result_hash")
    _require(isinstance(artifact.get("board_result_hashes"), Mapping), "board_result_hashes")
    _require(isinstance(artifact.get("board_reachability"), Mapping), "board_reachability")
    _require(
        isinstance(artifact.get("kv260_ssh_only_checked"), bool),
        "kv260_ssh_only_checked",
    )
    _validate_boundary_summary(
        artifact.get("boundary_exchange_ratio_summary"),
        gated_upstream_ready=bool(artifact.get("gated_upstream_ready")),
    )
    _require(isinstance(artifact.get("timing_repeat_counts"), Mapping), "timing_repeat_counts")
    _require(isinstance(artifact.get("timing_summary"), Mapping), "timing_summary")
    _require(
        isinstance(artifact.get("hashes_match_before_timing_compare"), bool),
        "hashes_match_before_timing_compare",
    )
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(isinstance(artifact.get("hardware_receipts_ready"), bool), "hardware_receipts_ready")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict speedup boundary")
    _require(not _artifact_mentions_host_storage(artifact), "host block-device evidence present")
    _validate_tests_run(artifact.get("tests_run"))
    _validate_command_receipts(artifact)
    _require(len(str(artifact.get("reproducibility_checksum", ""))) == 64, "checksum")
    if artifact.get("hardware_receipts_ready") is True:
        _require(artifact.get("gated_upstream_ready") is True, "gated_upstream_ready")
        _require(artifact.get("hashes_match_before_timing_compare") is True, "hash comparison")
        _require(str(verdict).startswith("complete:"), "honest_verdict")
        _require(not artifact.get("readiness_blockers"), "readiness_blockers")
    else:
        _require(str(verdict).startswith("blocked:"), "honest_verdict")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact under ``root`` and return the path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    workload_root: str | Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Build, validate, and write Exp5463's JSON deliverable."""

    artifact = build_artifact(
        root=workload_root if workload_root is not None else repo_root,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit,
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--workload-root", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    tests = [{"command": command, "outcome": "passed"} for command in args.test_run] or None
    run_experiment(
        repo_root=args.output_root,
        workload_root=args.workload_root,
        tests_run=tests,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main(sys.argv[1:]))
