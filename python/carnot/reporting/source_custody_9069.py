"""REQ-REPORT-9069-V774: missing custody authority cannot authorize fresh collection."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

from carnot.reporting import fresh_corpus_a_9058 as a
from carnot.reporting import fresh_seal_methods_9042 as methods
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import invocation_provenance_9000 as p
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
__all__ = ["a", "source", "p", "canonical_hash"]
ROOT = a.ROOT
NAME, TASK = "experiment_9069_v774_source_custody", "exp9069-source-custody"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_source_custody_9069.py"
OWNED = [
    "python/carnot/reporting/source_custody_9069.py",
    "python/carnot/reporting/source_custody_runner_9069.py",
    CLI,
]
ROSTER, A, CAPSTONE = (
    a.ROSTER,
    "results/experiment_9058_v773_fresh_corpus_a.json",
    "results/experiment_9067_v773_capstone.json",
)
PINS = {
    ROSTER: a.PINS[ROSTER],
    A: "sha256:88b49f41bf789b5e28e37e80d22c983a6cd36993c0f71f146241fffbe9414048",
    CAPSTONE: "sha256:eb76ada3c0777398b8ac9c6910a59678400ab3b7deb7f8eb92048412c14bcf75",
}
TOKENIZER = dict(
    model_sha256="sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
    tokenizer="embedded_GGUF",
    tokenizer_container_sha256="sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
    quantization="Q4_K_M",
)
ATTEMPT_TASKS = (9045, 9046, 9058, 9059, 9071)
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries make bounded metadata work visible to the supervisor."""
    print(f"[exp9069] phase={phase} completed={completed} pending={pending}", flush=True)


def contract_checks(
    manifest: Json, primary: Json, path: Path, *, expected_roles: Json | None = None
) -> list[Json]:
    """Original role counts, settings and tokenizer identity remain scientific authority."""
    values = [
        (
            "role_counts",
            expected_roles or dict(calibration=128, test=128, online=768),
            dict(Counter(g["role"] for g in manifest["graph_manifest"])),
        ),
        ("first_response_budget", 1, manifest.get("first_response_budget", "missing_field")),
        (
            "generator_settings",
            source.SETTINGS,
            manifest.get("generator_settings", "missing_field"),
        ),
        (
            "state_hashes",
            primary.get("state_hashes", "missing_field"),
            manifest.get("state_hashes", "missing_field"),
        ),
        *[
            (
                "model_tokenizer_hashes." + k,
                v,
                primary.get("model_tokenizer_hashes", {}).get(k, "missing_field"),
            )
            for k, v in TOKENIZER.items()
        ],
    ]
    return [
        p.gate(path, field, expected, observed, "immutable_contract")
        for field, expected, observed in values
    ]


def disclosures(value: Any, test_ids: set[str], pointer: str = "") -> list[Json]:
    """Report field locations only, so checking a disclosure never republishes its contents."""
    rows: list[Json] = []
    if isinstance(value, dict):
        is_test = (
            value.get("unit_id") in test_ids
            or value.get("role", value.get("split", value.get("condition"))) == "test"
        )
        sensitive = {
            "answer",
            "labels",
            "label",
            "correct",
            "prediction",
            "features",
            "raw_response",
            "parser_category",
            "response",
            "truth",
            "outcome",
            "content",
        }
        forbidden = sensitive.intersection(value) if is_test else set()
        if (
            is_test
            and value.get("metric") not in {"transport_completed", "graph_role_frozen"}
            and any(value.get(k) is not None for k in ("value", "numerator"))
        ):
            forbidden.add("value")
        rows.extend(
            dict(pointer=pointer + "/" + k, unit_id=value.get("unit_id"), kind="test_disclosure")
            for k in sorted(forbidden)
        )
        for k, child in value.items():
            rows.extend(disclosures(child, test_ids, pointer + "/" + k))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            rows.extend(disclosures(child, test_ids, pointer + "/" + str(i)))
    return rows


def attempt_inventory(root: Path, b_ids: set[str]) -> Json:
    """Inventory intent, orphan response and checkpoint starts across every known task root."""
    files: list[Json] = []
    attempts: list[Json] = []
    roots: list[Json] = []
    for number in ATTEMPT_TASKS:
        directories = sorted((root / "results/raw").glob(f"experiment_{number}_*"))
        roots.append(
            dict(
                experiment_id=number,
                paths=[str(d) for d in directories],
                coverage="observed_paths_only",
            )
        )
        for directory in directories:
            for name in (
                "intent.json",
                "wire_response.json",
                "response.json",
                "checkpoint.json",
                "invocation_ledger.json",
                "invocation_manifest.json",
            ):
                for path in sorted(directory.rglob(name)):
                    files.append(p.reference(path))
                    value = json.loads(path.read_bytes())
                    ids = set(path.parts).intersection(b_ids)
                    if isinstance(value, dict):
                        ids.update(set(value.get("starts", [])).intersection(b_ids))
                        ids.update(set(value.get("transport", {})).intersection(b_ids))
                        if value.get("unit_id") in b_ids:
                            ids.add(value["unit_id"])
                        ids.update(
                            uid
                            for uid, status in value.get("status", {}).items()
                            if uid in b_ids and status != "pending"
                        )
                    attempts.extend(
                        dict(unit_id=uid, authority=p.reference(path), kind=name)
                        for uid in sorted(ids)
                    )
    return dict(roots=roots, files=files, attempts=attempts, complete_no_attempt_authority=False)


def audit(root: Path, raw: Path) -> Json:
    """Authenticate sources before inspecting transport; missing authority stops readiness."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: Json = dict(
        checks=[],
        references=[],
        rows=[],
        role_counts={},
        state_hashes={},
        a_transport_count=0,
        a_ready=False,
        b_ready=False,
        retained_a_reference={},
        roster_reference={},
        exposure_inventory={},
        b_attempt_inventory={},
    )
    values = {}
    for name, pin in PINS.items():
        path = root / name
        observed = p.reference(path)["sha256"] or "missing_file"
        result["checks"].append(p.gate(path, "sha256", pin, observed, "declared_source_bytes"))
        if path.is_file():
            result["references"].append(p.retain(path, raw))
        if observed == pin:
            values[name] = json.loads(path.read_bytes())
    if len(values) != len(PINS):
        return result
    roster, primary = values[ROSTER], values[A]
    refs = [roster["fresh_role_manifest"], primary["custody_reference"]]
    for value in values.values():
        refs += (
            value.get("raw_shard_hashes", [])
            + value.get("code_config_hashes", [])
            + value.get("validation_scope_manifest", {}).get("code_test_hashes", [])
        )
    for index, ref in enumerate({r["path"]: r for r in refs}.values()):
        path = Path(ref["path"])
        observed = p.reference(path)["sha256"] or "missing_file"
        result["checks"].append(
            p.gate(path, "sha256", ref["sha256"], observed, "immutable_producer_and_primitive")
        )
        if path.is_file():
            result["references"].append(p.retain(path, raw))
        if index % 128 == 0:
            progress("authenticate_primitives", index, len(refs) - index)
    if any(not r["passed"] for r in result["checks"]):
        return result
    original = methods.authenticate(root, raw)
    result["checks"].extend(original["checks"])
    result["references"].extend(original["references"])
    if original["failures"]:
        return result
    manifest = json.loads(Path(roster["fresh_role_manifest"]["path"]).read_bytes())
    result["checks"].extend(contract_checks(manifest, primary, root / A))
    result["checks"].append(
        p.gate(
            root / ROSTER,
            "state_hashes",
            original["state_hashes"],
            manifest["state_hashes"],
            "original_states",
        )
    )
    result.update(
        state_hashes=original["state_hashes"],
        roster_reference=p.reference(root / ROSTER),
        retained_a_reference=p.reference(root / A),
    )
    units = source.scheduled(manifest, "A") + source.scheduled(manifest, "B")
    a_ids = {u["unit_id"] for u in units if u["shard"] == "A"}
    b_ids = {u["unit_id"] for u in units if u["shard"] == "B"}
    test_ids = {u["unit_id"] for u in units if u["role"] == "test"}
    checkpoint = json.loads(Path(primary["custody_reference"]["path"]).read_bytes())
    result["checks"].extend(
        p.gate(root / A, field, expected, observed, "a_transport_custody")
        for field, expected, observed in [
            ("unique_identities", 1024, len({u["unit_id"] for u in units})),
            ("A_starts", sorted(a_ids), sorted(checkpoint["starts"])),
            ("A_transport", sorted(a_ids), sorted(checkpoint["transport"])),
            ("checkpoint_manifest", canonical_hash(manifest), checkpoint["manifest_sha256"]),
        ]
    )
    public_rows = {r["unit_id"]: r for r in primary["rows"]}
    work = json.loads(Path(primary["raw_shard_hashes"][0]["path"]).read_bytes())
    ledger = {r["call_id"]: r for r in work["runtime"]["ledger"] if r["operation"] == "generation"}
    for index, unit in enumerate(source.scheduled(manifest, "A")):
        uid = unit["unit_id"]
        folder = a.runtime.folder(Path(work["raw"]) / "runtime/collection", index, unit)
        request = json.loads((folder / "intent.json").read_bytes())
        envelope = json.loads((folder / "response.json").read_bytes())
        response = envelope["raw_response"]
        transport = checkpoint["transport"][uid]
        matched = (
            request["unit_id"] == uid
            and request["request"]
            == dict(source.SETTINGS, messages=[dict(role="user", content=unit["prompt"])])
            and transport["response_sha256"] == p.reference(folder / "response.json")["sha256"]
            and transport["request_id"] == response["id"]
            and transport["token_count"] == response["usage"]["completion_tokens"]
            and transport["duration_s"] == envelope["duration_s"]
            and ledger[uid]["response_sha256"] == canonical_hash(response)
            and all(public_rows[uid].get(k) == v for k, v in transport.items())
        )
        result["checks"].append(
            p.gate(
                folder / "response.json",
                "transport_binding",
                True,
                matched,
                "a_request_response_receipt",
            )
        )
        result["a_transport_count"] += int(matched)
        if index % 64 == 0:
            progress("a_transport_receipts", index + 1, 511 - index)
    exposure_files = {root / name for name in PINS}
    exposure_files.update(
        path
        for number in range(9055, 9069)
        for path in (root / "results").glob(f"experiment_{number}_*.json")
    )
    exposure_files.update(
        Path(ref["path"])
        for ref in values[CAPSTONE].get("source_artifact_hashes", [])
        if Path(ref["path"]).suffix == ".json"
    )
    exposed: list[Json] = []
    inventory: list[Json] = []
    for path in sorted(exposure_files):
        ref = p.reference(path)
        inventory.append(ref)
        exposed.extend(
            dict(row, authority=ref) for row in disclosures(json.loads(path.read_bytes()), test_ids)
        )
    access_logs = [
        path
        for directory in (root / "results/raw").glob("experiment_9058_*")
        for path in directory.rglob("*access*.jsonl")
    ]
    for path in access_logs:
        inventory.append(p.reference(path))
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("role") == "test" and event.get("authorized") is True:
                exposed.append(dict(kind="test_access", authority=p.reference(path)))
    result["exposure_inventory"] = dict(
        files=inventory,
        access_logs=[p.reference(path) for path in access_logs],
        disclosures=exposed,
        complete_access_authority=False,
    )
    result["b_attempt_inventory"] = attempt_inventory(root, b_ids)
    result["checks"].extend(
        [
            p.gate(root / A, "TEST_disclosure_count", 0, len(exposed), "a_test_nonexposure"),
            p.gate(
                root / A,
                "custody_access_log_reference",
                "authenticated_complete_access_coverage",
                primary.get("custody_access_log_reference", "missing_field"),
                "a_access_coverage",
            ),
            p.gate(
                root / "results/raw",
                "B_attempt_count",
                0,
                len(result["b_attempt_inventory"]["attempts"]),
                "b_no_attempt",
            ),
            p.gate(
                root / A,
                "invocation_inventory_reference",
                "authenticated_complete_task_path_coverage",
                primary.get("invocation_inventory_reference", "missing_field"),
                "b_invocation_coverage",
            ),
        ]
    )
    attempted = {r["unit_id"] for r in result["b_attempt_inventory"]["attempts"]}
    result["rows"] = [
        dict(
            unit_id=u["unit_id"],
            group_id=u["graph_id"],
            arm=u["shard"],
            condition=u["role"],
            status="completed",
            metric="custody_authenticated",
            value=0,
            numerator=0,
            denominator=1,
            exclusion_reason="a_access_coverage_unproven"
            if u["shard"] == "A"
            else "prior_attempt"
            if u["unit_id"] in attempted
            else "b_invocation_coverage_unproven",
            disposition="retained_transport"
            if u["shard"] == "A"
            else "attempted"
            if u["unit_id"] in attempted
            else "pending_observed_only",
        )
        for u in units
    ]
    result["role_counts"] = dict(Counter(u["role"] for u in units))
    progress("custody_audit_after", len(units), 0)
    return result


def build(work: Json, validation: Json, accepted: bool) -> Json:
    """Readiness requires custody evidence and current checks; historical calls earn no credit."""
    evidence = work["evidence"]
    gates = evidence["checks"] + work["environment"]["checks"]
    failures = [r for r in gates if not r["passed"]]
    cls = "disqualified" if not accepted else "blocked" if failures else "null"
    rows = evidence["rows"] or [
        dict(
            unit_id=f"v772-graph-{i:04d}-k2",
            group_id=f"v772-graph-{i:04d}",
            arm="unavailable",
            condition="unavailable",
            status="censored",
            metric="custody_authenticated",
            value=None,
            numerator=0,
            denominator=1,
            exclusion_reason="source_preconditions_blocked",
        )
        for i in range(1024)
    ]
    value: Json = dict(
        experiment_id=9069,
        task_id=TASK,
        milestone="2026.11.774",
        run_date="20261112",
        honest_verdict=f"complete_{cls}_"
        + (failures[0]["check"] if cls == "blocked" else "source_custody"),
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Transport and source custody only; graph truth is an executable oracle. No detector, human correctness or generalization result.",
        exposure_scope="Private transport metadata inspection; no TEST response contents or derived labels published. Missing access coverage cannot certify nonexposure.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        model_invoked=False,
        numerical_training_calls=0,
        minimum_duration_s=0,
        rows=rows,
        intended_count=1024,
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=0,
        excluded_count=0,
        independent_count=0,
        a_custody_ready_score=int(accepted and evidence["a_ready"]),
        b_collection_ready_score=int(accepted and evidence["a_ready"] and evidence["b_ready"]),
        shard_ready_score=0,
        retained_a_reference=evidence["retained_a_reference"],
        roster_reference=evidence["roster_reference"],
        state_hashes=evidence["state_hashes"],
        role_counts=evidence["role_counts"],
        a_transport_count=evidence["a_transport_count"],
        exposure_inventory=evidence["exposure_inventory"],
        b_attempt_inventory=evidence["b_attempt_inventory"],
        custody_mutation_rows=work.get("controls", []),
        gate_check_summary=gates,
        required_checks_passed=accepted,
        flagged_adversarial=not accepted,
        duration_s=work["duration_s"],
        random_seed=7749069,
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            environment=work["environment"], input_checks=evidence["checks"]
        ),
        validation_scope_manifest=validation["scope"],
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        repository_health=work.get(
            "repository_health",
            dict(
                passed=False,
                full_suite_repeated=False,
                reason="retained_historical_failures",
                reference=p.reference(ROOT / "results/experiment_9068_v774_dispatch_contract.json"),
            ),
        ),
        raw_shard_hashes=work["raw_references"],
        source_artifact_hashes=evidence["references"],
        code_config_hashes=work["code_references"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(name).name.split("_")[1]),
                fields_imported=["original_custody_and_metadata_only"],
                sha256=pin,
            )
            for name, pin in PINS.items()
        ],
        methodology_note="Pin completed original sources and producer/test bytes. Authenticate every A first request/response without evaluating truth; inspect public projections and task-owned inventories. Pending B and no files prove observed absence only. Missing complete access and invocation authority terminally blocks reuse and collection. No retries or replacement cohorts.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Authenticated primitive bytes and current owned checks govern custody only; missing authority blocks readiness and historical calls stay upstream.",
    )
    value["reproducibility_checksum"] = p.checksum(value)
    return value
