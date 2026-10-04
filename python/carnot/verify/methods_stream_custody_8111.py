"""REQ-VERIFY-8111: authenticate methods without borrowing historical model work.

Public bytes determine the original slots. Human targets remain evaluator-only,
and missing stream observations cannot invalidate an independently sealed design.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting import v702_contract_custody as historical
from carnot.verify import development_methods_8098 as cohort
from carnot.verify import qwen_learning_stream_capture_8102 as stream

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8111_v702_methods_and_stream_custody"
TASK = "exp8111-methods-and-stream-custody"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/methods_stream_custody_8111.py"
RUNNER = "python/carnot/reporting/methods_stream_execution_8111.py"
TEST = "tests/python/test_methods_stream_custody_8111.py"
DESIGN = ROOT / "openspec/change-proposals/v702-methods-and-stream-protocol.md"
COHORT = "results/experiment_8098_v701_development_methods.json"
STREAM = "results/experiment_8102_v701_learning_stream_capture.json"
FROZEN_PROTOCOL_HASH = "sha256:898fe736d52e35ae1873aeeb70a5eea477ad4df255da0d9fd07cfd9f473f112e"
UPSTREAM_HASHES = {
    8098: "sha256:4d28524a53f7eed96c2ce814da5a798828594a7e540d70d0de2f6a25ca5d6f76",
    8102: "sha256:32ca6d4b55322e3b31c8cf69fbc95dbb5a7fa065be8b84c5e59ce07c899623e8",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real completed counts distinguish custody work from invented inference."""
    print(f"[exp8111] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol(path: Path = DESIGN) -> Json:
    """An exact contract hash prevents silent changes after outcomes become known."""
    try:
        value: Json = json.loads(path.read_text().split("```json\n")[1].split("```", 1)[0])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("protocol_contract") from error
    if canonical_hash(value) != FROZEN_PROTOCOL_HASH:
        raise ValueError("protocol_contract")
    return value


class Custody(historical.Binder):
    """Keep successful operands too, so readiness can be independently reduced."""

    def __init__(self, raw: Path):
        super().__init__(raw)
        self.checks: list[Json] = []
        self.upstream = TASK

    def require(self, path: Path, field: str, expected: Any, observed: Any) -> None:
        digest = next((r["sha256"] for r in self.refs if r["path"] == str(path)), None)
        check = dict(
            check=field,
            upstream=self.upstream,
            path=str(path),
            hash=digest,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=observed == expected,
        )
        self.checks.append(check)
        if not check["passed"]:
            self.failures.append(check)
            raise historical.InputFailure(field)


def upstream(path: Path, n: int, b: Custody, fixture: bool) -> Json:
    """Only byte-bound, normally validated historical inputs authorize reuse."""
    b.upstream = f"exp{n}"
    value = b.read(path, None if fixture else UPSTREAM_HASHES[n])
    evidence = historical.terminal(path, value, b)
    for key, expected in [
        ("experiment_id", n),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        b.require(path, key, expected, value.get(key))
    b.require(path, "terminal.report.passed", True, evidence["report"].get("passed"))
    for ref in value["raw_shard_hashes"]:
        b.bind(Path(ref["path"]), ref["sha256"])
    for label, digest in value.get("code_config_hashes", {}).items():
        if label.startswith(("python/", "scripts/", "tests/")):
            b.bind(ROOT / label, digest)
    return value


def read_ref(b: Custody, ref: Json) -> Json:
    """Read a snapshot only after its declared exact bytes are authenticated."""
    return b.read(Path(ref["path"]), ref["sha256"])


def reduce_rows(rows: list[Json]) -> Json:
    """Every slot remains in the denominator, including unknown human targets."""
    if len({r["source_cluster_id"] for r in rows}) != len(rows) or any(
        r["denominator"] != 1 for r in rows
    ):
        raise ValueError("source_denominator")
    counts = Counter(r["status"] for r in rows)
    return dict(
        intended_count=640,
        eligible_count=counts["completed"],
        independent_count=sum(r["source_cluster_id"].startswith("sha256:") for r in rows),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=counts["failed"],
    )


def authenticate_cohort(root: Path, b: Custody, fixture: bool, mutation: str) -> Json:
    """Rebuild public selection before opening evaluator records; never resplit."""
    path = root / COHORT
    original = upstream(path, 8098, b, fixture)
    b.require(path, "cohort_ready_score", 1, original.get("cohort_ready_score"))
    for name in ["source_info", "response"]:
        operand = root / f"data/ragtruth/{name}.jsonl"
        reference = next(
            r
            for r in original["source_artifact_hashes"]
            if r["path"].endswith(f"data/ragtruth/{name}.jsonl")
        )
        b.bind(operand, reference["sha256"])
    roster, public = [], []
    views = {role: read_ref(b, original["role_manifests"][role]) for role in cohort.ROLES}
    for role in cohort.ROLES:
        roster.extend(views[role]["roster"])
        public.extend(views[role]["request_rows"])
    if mutation == "labels":
        public[0]["y"] = 1
    if mutation == "roles":
        roster[0]["role"] = "tune"
    if mutation == "slots":
        roster.pop()
    if mutation == "source":
        public[0]["source_bytes"] = b"changed original".hex()
    cohort.separation(roster, public)
    progress("rebuild_original_public_selection", 0, 640)
    sources, responses = cohort.public_training(root)
    selected, rendering, _, _ = cohort.select(sources, responses)
    b.require(
        path,
        "original_selection",
        canonical_hash([selected, rendering]),
        canonical_hash([roster, public]),
    )
    for name in ["source_info", "response"]:
        operand = root / f"data/ragtruth/{name}.jsonl"
        b.require(
            operand,
            "unchanged_after_selection",
            next(r["sha256"] for r in b.refs if r["path"] == str(operand)),
            sha256_file(operand),
        )
    selection = read_ref(b, original["selection"])
    b.require(
        path,
        "public_selection_hash",
        canonical_hash(selection),
        canonical_hash(dict(roster=roster, public=public)),
    )
    progress("public_authenticated_evaluator_open", 640)
    wanted = {r["response_id"] for r in roster}
    annotated = {}
    for i, line in enumerate((root / "data/ragtruth/response.jsonl").open()):
        meta = cohort.public_fields(line)
        if meta["id"] in wanted:
            annotated[meta["id"]] = json.loads(line)
        if i % 1024 == 0:
            progress("original_annotation_scan", i + 1)
    rows = []
    for role in cohort.ROLES:
        ev = read_ref(b, original["evaluator_label_manifests"][role])
        pairs = [(r, p) for r, p in zip(roster, public, strict=True) if r["role"] == role]
        b.require(
            Path(original["evaluator_label_manifests"][role]["path"]),
            "evaluator_only",
            "evaluator_only",
            ev["access_policy"],
        )
        b.require(
            path, "evaluator.public_seal", original["selection"]["sha256"], ev["public_seal_sha256"]
        )
        records, targets, annotations = [], [], []
        for meta, feature in pairs:
            record = annotated[meta["response_id"]]
            records.append(record)
            target, spans = cohort.target(record, bytes.fromhex(feature["answer_bytes"]))
            targets.append(dict(meta, **target))
            annotations.extend(
                dict(
                    s,
                    family_id=meta["family_id"],
                    response_id=meta["response_id"],
                    response_sha256="sha256:" + cohort.sha(bytes.fromhex(feature["answer_bytes"])),
                )
                for s in spans
            )
            rows.append(
                dict(
                    unit_id=meta["unit_id"],
                    source_cluster_id=meta["source_cluster_id"],
                    source_id=meta["source_id"],
                    role=role,
                    slot=meta["slot"],
                    arm="source_custody",
                    condition="original_exposed_development",
                    metric="complete_human_target",
                    numerator=int(target["y"] is not None),
                    denominator=1,
                    status=target["status"],
                    exclusion_reason=target["exclusion_reason"],
                )
            )
        b.require(
            path,
            "annotation_custody_" + role,
            canonical_hash([records, targets, annotations]),
            canonical_hash([ev["original_response_records"], ev["rows"], ev["annotation_rows"]]),
        )
    return dict(
        rows=rows,
        role_manifests=original["role_manifests"],
        evaluator_label_manifests=original["evaluator_label_manifests"],
        views=views,
        response_selection_rule=original.get("response_selection_rule"),
        class_support=original.get("class_support", {}),
    )


def authenticate_stream(path: Path, b: Custody, views: Json, fixture: bool) -> Json:
    """Reparse original completions and recompute public values without targets."""
    original = upstream(path, 8102, b, fixture)
    b.require(path, "stream_capture_ready_score", 1, original.get("stream_capture_ready_score"))
    refs = {Path(r["path"]).name: r for r in original["raw_shard_hashes"]}
    primitives = read_ref(b, refs["primitive_rows.json"])["rows"]
    frozen = stream.freeze({r: views[r] for r in stream.ROLES})
    captured = read_ref(b, refs["capture_manifest.json"])["rows"]
    b.require(path, "label_free_manifest", canonical_hash(frozen), canonical_hash(captured))
    for i, (row, slot) in enumerate(zip(primitives, frozen, strict=True)):
        b.require(
            path,
            "public_primitive_" + str(i),
            canonical_hash({k: v for k, v in slot.items() if k != "exclusion_reason"}),
            canonical_hash({k: row[k] for k in slot if k != "exclusion_reason"}),
        )
        b.require(path, "human_target_absent_" + str(i), None, row.get("human_target"))
        if i % 64 == 0:
            progress("historical_completion_custody", i, 320 - i)
    reduced = stream.reduce(primitives)
    b.require(
        path,
        "raw_completion_hashes",
        original["raw_completion_hashes"],
        [canonical_hash(r["raw_response"]) for r in primitives],
    )
    for role in stream.ROLES:
        feature = read_ref(b, original[role + "_feature_manifest"])
        b.require(
            path,
            role + "_feature_rows",
            canonical_hash(reduced[role + "_features"]),
            canonical_hash(feature["rows"]),
        )
    b.require(
        path,
        "retained_original_mask",
        original["original_slot_mask"],
        reduced["original_slot_mask"],
    )
    b.require(path, "stream_floor", 1, reduced["stream_capture_ready_score"])
    return dict(
        original_slot_mask=reduced["original_slot_mask"],
        stream_feature_manifest=original["stream_feature_manifest"],
        retention_feature_manifest=original["retention_feature_manifest"],
        role_completion_counts=reduced["role_completion_counts"],
        historical_model_provenance=dict(
            primary_sha256=UPSTREAM_HASHES[8102],
            MODEL_SPECS=original.get("MODEL_SPECS", []),
            model_invocation_counts=original.get("model_invocation_counts", {}),
        ),
    )


METHOD_SOURCES = [
    dict(
        arxiv="2609.08267",
        use="source interventions and duplicate variation",
        difference="EAEV aligns entities; this protocol diagnoses whole-answer probability",
    ),
    dict(
        arxiv="2503.21076",
        use="bounded local radial classifier",
        difference="KAC channel-wise bases differ from multivariate Euclidean Gaussian centers",
    ),
    dict(
        arxiv="2511.12828",
        use="independent retention evaluation",
        difference="Local support does not guarantee resistance to high-dimensional forgetting",
    ),
    dict(
        arxiv="2606.11711",
        use="bounded delayed pending state",
        difference="Empirical admission guards inherit no scheduler or regret theorem",
    ),
    dict(
        arxiv="2602.02056",
        use="charge coefficient updates and memory movement",
        difference="Sparse spline FPGA gains do not transfer to CPU radial distances",
    ),
]


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Seal methods first and qualify historical stream in a separate branch."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    b = Custody(raw)
    work: Json = dict(
        methods_ready_score=0,
        stream_input_ready_score=0,
        owned_failure=bool(mutation),
        source_artifact_hashes=b.refs,
        gate_check_summary=b.checks,
        phase_spans=[],
        role_manifests={},
        evaluator_label_manifests={},
        stream_feature_manifest=None,
        retention_feature_manifest=None,
        original_slot_mask={},
        protocol=protocol(),
        method_source_map=METHOD_SOURCES,
        rows=[
            dict(
                unit_id=f"missing-{role}-{i}",
                source_cluster_id=f"missing-{role}-{i}",
                arm="source_custody",
                condition="original_slot_missing",
                metric="complete_human_target",
                numerator=0,
                denominator=1,
                role=role,
                slot=i + 1,
                status="excluded",
                exclusion_reason="original_source_custody_unavailable",
            )
            for role, total in cohort.ROLES.items()
            for i in range(total)
        ],
    )
    work["method_freeze"] = cohort.immutable(
        raw / "methods.json",
        dict(
            protocol=work["protocol"],
            design_text=DESIGN.read_text(),
            design_sha256=sha256_file(DESIGN),
        ),
    )
    for branch in ["methods", "stream_input"]:
        progress("before_" + branch)
        phase = time.monotonic()
        path = root / COHORT if branch == "methods" else (stream_path or root / STREAM)
        try:
            if branch == "methods":
                result = authenticate_cohort(root, b, fixture, mutation)
                work.update(result, methods_ready_score=1)
            else:
                b.require(path, "original_public_roles_available", True, "views" in work)
                work.update(
                    authenticate_stream(path, b, work["views"], fixture), stream_input_ready_score=1
                )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            if not b.failures or b.failures[-1]["check"] != str(error):
                b.checks.append(
                    dict(
                        check=branch + "_custody",
                        upstream=b.upstream,
                        path=str(path),
                        hash=None,
                        artifact_field=branch + "_custody",
                        op="==",
                        expected="authenticated",
                        observed=str(error),
                        passed=False,
                    )
                )
        work["phase_spans"].append(
            dict(phase=branch, start_s=phase - started, duration_s=time.monotonic() - phase)
        )
        progress("after_" + branch, int(work[branch + "_ready_score"]))
    for name in [
        MODULE,
        RUNNER,
        CLI,
        TEST,
        "python/carnot/verify/development_methods_8098.py",
        "python/carnot/verify/qwen_learning_stream_capture_8102.py",
        "python/carnot/verify/radial_memory_8085.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
    ]:
        b.bind(ROOT / name)
    work["code_config_hashes"] = {
        str(Path(r["path"]).relative_to(ROOT)): r["sha256"]
        for r in b.refs
        if Path(r["path"]).is_relative_to(ROOT) and r["path"].endswith(".py")
    }
    b.bind(DESIGN)
    if not fixture:
        for source in METHOD_SOURCES:
            b.bind(ROOT / "results/raw" / NAME / "literature" / (source["arxiv"] + ".html"))
    work["memory_genesis"] = dict(
        residual_coefficients=[0.0] * 17,
        pending_events=[],
        consumed_update_ids=[],
        used_admission_ids=[],
        optimizer_step=0,
        rng_state=random.Random(70211).getstate(),
        original_slot_mask=work["original_slot_mask"],
        baseline_hashes={r: work.get(r + "_feature_manifest") for r in stream.ROLES},
        state_scope="zero-residual genesis only; no trained center or feedback consumed",
    )
    work["memory_genesis"]["state_hash"] = canonical_hash(work["memory_genesis"])
    work.pop("views", None)
    work["duration_s"] = time.monotonic() - started
    work["raw_shard_hashes"] = [
        work["method_freeze"],
        cohort.immutable(raw / "primitive_rows.json", dict(rows=work["rows"])),
        cohort.immutable(raw / "memory_genesis.json", work["memory_genesis"]),
    ]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", 640)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Owned validation controls publication, while external branches stay separate."""
    value = dict(work)
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    if not owned:
        verdict, terminal = "disqualified", "complete_disqualified_owned_validation"
    elif not work["methods_ready_score"] or not work["stream_input_ready_score"]:
        failed = next(r for r in work["gate_check_summary"] if not r["passed"])
        verdict, terminal = "blocked", "complete_blocked_" + failed["check"]
    else:
        verdict = "circular_positive" if fixture else "null"
        terminal = "complete_" + verdict + "_methods_and_stream_custody_sealed"
    value.update(
        reduce_rows(work["rows"]),
        experiment_id=8111,
        task_id=TASK,
        honest_verdict=terminal,
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        methods_ready_score=int(owned and work["methods_ready_score"]),
        stream_input_ready_score=int(owned and work["stream_input_ready_score"]),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[],
        claim_scope="sealed methods and historical input custody; no benefit measurement",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        run_date="20261004",
        random_seed=70211,
        milestone="2026.10.702",
        method_config=work["protocol"],
        statistical_plan=work["protocol"]["statistical_plan"],
        fixture_protocol_only=fixture,
        sample_size_budget=dict(
            intended=640,
            unit="original_source_cluster",
            stream=256,
            warmup=64,
            later_evaluation=192,
            retention=64,
        ),
        acceptance_gates=dict(
            methods="exact protocol and original source custody",
            stream="independent raw completion replay; stream>=224 retention>=48",
        ),
        field_principles=dict(
            readiness="methods and stream are independent administrative gates",
            denominators="original slots, no replacement or seed inflation",
            genesis="state contract is not executed training",
            inference="historical Qwen provenance is never current invocation",
            science="no independent generalization or learning benefit measured",
        ),
        methodology_note="Public role selection precedes evaluator authentication; stream has no fitted-head prerequisite.",
        literature_limitations=[
            "Five primary arXiv sources ingested; prior inaccessible citation service is a bibliographic limitation, not a science gate."
        ],
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild reductions from saved measurement bytes and reject changed custody."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_text())
        for ref in [*work["source_artifact_hashes"], *work["raw_shard_hashes"]]:
            saved = Path(ref.get("snapshot_path", ref["path"]))
            if sha256_file(saved) != ref["sha256"]:
                return False
        receipts = json.loads((raw / "validation_receipts.json").read_text())["rows"]
        for receipt in receipts:
            if (
                "log_path" in receipt
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        rebuilt = build(work, raw, receipts, fixture=value["fixture_protocol_only"])
        return canonical_hash(rebuilt) == canonical_hash(
            dict(value, reproducibility_checksum=checksum)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
