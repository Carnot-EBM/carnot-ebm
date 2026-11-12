"""REQ-REPORT-8943: original byte identity prevents later results rewriting history."""

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import v764_branch_contract as authority
from carnot.reporting.v763_capstone import references, resolve
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference, load

Json = dict[str, Any]
ROOT = authority.ROOT
PINS = {
    8930: (
        "experiment_8930_v764_branch_contract.json",
        "05d43791590dbfde8f18d104faef650f70970fa74779d39f8f938d883430f553",
    ),
    8931: (
        "experiment_8931_v764_graph_source_methods.json",
        "810b976a4d7883a3a600c9501982eb882d1feb658b0a54cb86afea00320f5b57",
    ),
    8932: (
        "experiment_8932_v764_pilot_label_audit.json",
        "f98f862d034531080553eb8b6757e305b5dd81e3f21a30684fd6277f63cf1c5e",
    ),
    8933: (
        "experiment_8933_v764_compatible_replay_methods.json",
        "317b615ee1bc56f48b56161577ca77a716c40214c18ce15badba4aa2cf210169",
    ),
    8934: (
        "experiment_8934_v764_depth_control.json",
        "7e82e593722459bf4eb5705118925d22c7ce698869d235a80f4bd999baedd47d",
    ),
    8935: (
        "experiment_8935_graph_corpus_a.json",
        "db568eb1e31f838afd0b3bad2a898aba9f9507bd74af0b21e13fef8e4f543916",
    ),
    8937: (
        "experiment_8937_v764_calibration_methods.json",
        "3cd2616c429c3902e2e9ece4bfa57a52b5f59b7dd364161813b7fa94b9829d25",
    ),
    8940: (
        "experiment_8940_v764_compatible_replay_streams.json",
        "2a4a7abddbc91cc4cca180cc7e65799639adee6ef881b5966123e812d2287a8f",
    ),
    8941: (
        "experiment_8941_v764_compatible_replay_benefit.json",
        "99a9be53396e5fd3867ab0981296d6be37e8863686d1b278188c11ceb0b9b489",
    ),
    8942: (
        "experiment_8942_v764_replay_complete_cost.json",
        "9e23a1b5a120845a81656fff3297627a07b2becaeadd55fc94f519dd45a13405",
    ),
}
LOG_PIN = "sha256:184aa3b67ab6f202a276c2d1e70dddc593cb0861873a95a74092cd89b38b0e04"
SCHEDULER_PIN = "sha256:8af429226e559cce237973c151529b28ebc719ecc8d6ca1bbd33d37cca954ef3"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish slow authentication from a stall."""
    print(f"[exp8943] phase={phase} completed={completed} pending={pending}", flush=True)


def admission(value: Json, n: int) -> str | None:
    """Null scientific gates are valid outcomes; failed owned controls disqualify."""
    required = {
        "verdict_class",
        "acceptance_gates",
        "required_checks_passed",
        "flagged_adversarial",
        "rows",
        "independent_count",
    }
    if not required <= value.keys():
        return "missing_field"
    if value["verdict_class"] == "disqualified" or value["required_checks_passed"] is False:
        return "disqualified_primary"
    if (
        value.get("experiment_id") != n
        or value.get("milestone") != "2026.11.764"
        or value.get("verifier_is_oracle")
        and value["verdict_class"] == "positive"
    ):
        return "rejected_producer"
    return None


def retain(path: Path, raw: Path, pin: str) -> Json:
    """Copy exact authenticated bytes; never normalize or repair a historical producer."""
    require(sha256_file(path) == pin, "source_bytes:" + str(path))
    target = raw / "retained" / pin.split(":")[1] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return dict(reference(target), original_path=str(path))


def collect(raw: Path, sources: Path = ROOT / "results") -> Json:
    """Authenticate original primaries and real scheduler records before reduction."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("authentication_before", 0, 13)
    original = load(
        dict(path=str(ROOT / "results" / PINS[8930][0]), sha256="sha256:" + PINS[8930][1])
    )
    tasks = original["full_task_objects"]
    require(canonical_hash(tasks) == authority.PIN, "ordered_task_contract")
    values, refs, errors = {}, {}, {}
    for n, (name, pin) in PINS.items():
        path = sources / name
        try:
            refs[n] = retain(path, raw, "sha256:" + pin)
            values[n] = load(refs[n])
        except (OSError, ValueError):
            errors[n] = "rejected_producer" if path.is_file() else "missing_file"
    log = retain(ROOT / "ops/conductor-log.md", raw, LOG_PIN)
    scheduler = retain(ROOT / "scripts/research_conductor.py", raw, SCHEDULER_PIN)
    lines = Path(log["path"]).read_text().splitlines()
    catalog = {
        r["sha256"]: r
        for v in values.values()
        for r in references(v)
        if r.get("sha256") and Path(r["path"]).is_file()
    }
    operands: dict[tuple[str, str], Json] = {}
    # References can repeat across branches. Their repetitions are not new evidence.
    for n, value in values.items():
        try:
            for ref in references(value):
                key = (ref["path"], ref.get("sha256"))
                if ref.get("sha256") and key not in operands:
                    operands[key] = resolve(ref, catalog)
        except (OSError, ValueError):
            errors[n] = "rejected_producer"
        progress("producer_operands_authenticated", n - 8929, 8942 - n)
    slots = []
    for index, task in enumerate(tasks[:-1]):
        n = 8930 + index
        v = values.get(n, {})
        reason = errors.get(n)
        ref = refs.get(n)
        checks = []
        disposition = "qualified_primary"
        if n in (8936, 8938, 8939):
            matches = [
                line
                for line in lines
                if "| GATE_BLOCK |" in line and ("| " + task["title"][:50].strip() + " |") in line
            ]
            require(len(matches) == 1, "scheduler_log_identity")
            ref = log
            disposition = (
                "shared_title_scheduler_retirement" if n == 8936 else "authenticated_scheduler_skip"
            )
            v = dict(verdict_class="blocked", honest_verdict=None, scheduler_record=matches[0])
            reason = None
        elif n == 8935 and v:
            upstream = values.get(8934, {})
            require(
                v["failed_evidence_sha256"] == "sha256:" + PINS[8934][1]
                and v["failed_observed"] == upstream["depth_source_ready_score"] == 0
                and v["blocked_at_layer"] == "conductor_pre_gate",
                "scheduler_gate_evidence",
            )
            disposition = "authenticated_scheduler_skip"
        elif v:
            reason = reason or admission(v, n)
            disposition = reason or disposition
        if reason or n in (8935, 8936, 8938, 8939):
            edge = next(
                (e for e in task["gated_on"] if e["upstream"] != "exp8931-graph-source-methods"), {}
            )
            upstream_id = edge.get("upstream", task["id"]) if not reason else task["id"]
            upstream_n = int(upstream_id[3:7])
            field = (
                edge.get("artifact_field", "producer_contract")
                if not reason
                else "producer_contract"
            )
            upstream_value = values.get(upstream_n, {})
            upstream_path = sources / (
                PINS[upstream_n][0]
                if upstream_n in PINS
                else tasks[upstream_n - 8930]["deliverable"].split("/")[-1]
            )
            observed = upstream_value.get(field)
            checks.append(
                dict(
                    check=reason
                    or (
                        "observed_zero"
                        if observed == 0
                        else "missing_field"
                        if upstream_value
                        else "missing_file"
                    ),
                    upstream=upstream_id,
                    artifact_field=field,
                    expected=1,
                    observed=observed,
                    path=str(upstream_path),
                    hash=refs.get(upstream_n, {}).get("sha256"),
                )
            )
        slots.append(
            dict(
                task_id=task["id"],
                experiment_id=n,
                reference=ref,
                accounted=reason in (None, "disqualified_primary"),
                disposition=reason if reason else disposition,
                gate_check_summary=checks,
                scheduler_record=v.get("scheduler_record"),
            )
        )
        progress("slot_authenticated", index + 1, 12 - index)
    history = retain(ROOT / authority.HISTORY, raw, authority.PINS[authority.HISTORY])
    health = load(history)["repository_health"]
    for r in health:
        operands[(r["log_path"], r["log_sha256"])] = retain(
            ROOT / r["log_path"], raw, r["log_sha256"]
        )
    work = dict(
        tasks=tasks,
        slots=slots,
        references=[*operands.values(), log, scheduler, history],
        history_reference=history,
        repository_health=health,
        started_ns=start,
        ended_ns=time.monotonic_ns(),
        raw=str(raw),
        code=[],
    )
    atomic_json(raw / "work.json", work)
    return work
