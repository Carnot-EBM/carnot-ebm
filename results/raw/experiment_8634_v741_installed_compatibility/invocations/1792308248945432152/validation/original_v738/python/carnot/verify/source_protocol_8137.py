"""REQ-VERIFY-8137: qualify source transport before any future model call.

The original custody worker authenticates roles and targets. Only the three
source roles contribute here; stream features and learning remain separate.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.verify import evidence_protocol_8124 as previous

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8137_v704_source_protocol"
TASK = "exp8137-source-protocol"
MODULE = "python/carnot/verify/source_protocol_8137.py"
CLI = f"scripts/experiments/{NAME}.py"
RUNNER = MODULE
TEST = "tests/python/test_source_protocol_8137.py"
OWNED = [MODULE, CLI]
qualified = previous.qualified
ROLES = dict(fit=128, tune=64, evaluation=128)
METHODS = [
    dict(
        arxiv="2603.05828v1",
        sections="3.2-3.4",
        mapping="source/answer attribution and quote custody",
        limits="No HART retrieval, mechanism attribution, causal tracing or dataset reproduction.",
    ),
    dict(
        arxiv="2603.27752v2",
        sections="3.3-3.6",
        mapping="complete-source holistic versus source_span evidence",
        limits="No claim decomposition, chunk joins, enhanced labels or entailment certification.",
    ),
    dict(
        arxiv="2605.03534",
        sections="IV.A-IV.D",
        mapping="frozen probabilities; fit-only normalization and tune-only selective calibration",
        limits="No relation encoder, sufficiency labels, calibration fit or risk-coverage result is measured.",
    ),
]
progress = previous.progress


def permission_evidence() -> list[Json]:
    """Exercise both permissions on disposable private bytes before measurement."""
    with TemporaryDirectory(prefix="carnot-8137-permissions-") as directory:
        raw = Path(directory)
        production = raw / "production.json"
        previous.capture_seal(production, dict(rows=[]), fixture=False)
        denied = False
        try:
            production.write_bytes(b"changed")
        except PermissionError:
            denied = True
        private = raw / "private.json"
        ref = previous.capture_seal(private, dict(rows=[]), fixture=True)
        private.write_bytes(b"changed")
        return [
            dict(
                production_write_denied=denied,
                private_write_completed=True,
                mutation_rejected=sha256_file(private) != ref["sha256"],
                historical_failure="PermissionError at Exp8124 manifest write; 9 passed, 1 failed",
                scope="private permission probe; original tamper assertion retained",
            )
        ]


def reduce_rows(rows: list[Json]) -> Json:
    """Retain every original slot; repeated arms create no independent sources."""
    if (
        len(rows) != 320
        or len({r["source_cluster_id"] for r in rows}) != 320
        or any(r["denominator"] != 1 for r in rows)
    ):
        raise ValueError("source_denominator")
    counts = Counter(r["status"] for r in rows)
    return dict(
        intended_count=320,
        eligible_count=counts["completed"],
        independent_count=sum(r["source_cluster_id"].startswith("sha256:") for r in rows),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=counts["failed"],
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Seal unchanged source requests before opening original evaluator records."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    b = qualified.Custody(raw)
    config = previous.protocol()
    freeze = qualified.cohort.immutable(
        raw / "v703_methods.json",
        dict(
            protocol=config,
            design_text=previous.DESIGN.read_text(),
            design_sha256=sha256_file(previous.DESIGN),
        ),
    )
    progress("v704_source_protocol_frozen", 1)
    permissions = permission_evidence()
    plan = previous.preconditions(root, raw, fixture=fixture)
    rows = [
        dict(
            unit_id=f"missing-{role}-{i}",
            source_cluster_id=f"missing-{role}-{i}",
            role=role,
            slot=i + 1,
            arm="source_custody",
            condition="original_slot_missing",
            metric="complete_human_target",
            numerator=0,
            denominator=1,
            status="excluded",
            exclusion_reason="original_source_custody_unavailable",
        )
        for role, total in ROLES.items()
        for i in range(total)
    ]
    conformance: list[Json] = []
    roles: Json = {}
    ready = 0
    try:
        for role in ROLES:
            ref = plan["role_manifests"][role]
            view = qualified.read_ref(b, ref)
            roles[role] = ref
            for meta, public in zip(view["roster"], view["request_rows"], strict=True):
                source, answer = (
                    bytes.fromhex(public[k]).decode() for k in ("source_bytes", "answer_bytes")
                )
                arms = config["arms"][
                    :: 1 if int(meta["source_cluster_id"].split(":")[-1], 16) % 2 == 0 else -1
                ]
                for order, arm in enumerate(arms):
                    prompt = config["prompt_prefixes"][arm] + json.dumps(
                        dict(source=source, answer=answer), ensure_ascii=False
                    )
                    conformance.append(
                        dict(
                            unit_id=meta["unit_id"],
                            source_cluster_id=meta["source_cluster_id"],
                            role=role,
                            arm=arm,
                            order=order,
                            prompt=prompt,
                            prompt_sha256=canonical_hash(prompt),
                            source_bytes=public["source_bytes"],
                            answer_bytes=public["answer_bytes"],
                            max_tokens=128,
                            status="frozen_not_invoked",
                            entailment_label=None,
                        )
                    )
            progress("frozen_source_" + role, len(conformance), 640 - len(conformance))
    except (OSError, ValueError, KeyError) as error:
        b.checks.append(
            dict(
                check="source_roles",
                upstream=previous.METHODS,
                path=str(root / previous.METHODS),
                hash=None,
                artifact_field="role_manifests",
                op="==",
                expected=list(ROLES),
                observed=str(error),
                passed=False,
            )
        )
    manifest = previous.capture_seal(
        raw / "evidence_capture_manifest.json", dict(rows=conformance), fixture=fixture
    )
    progress("source_capture_manifest_sealed", len(conformance))
    try:
        authenticated = qualified.authenticate_cohort(root, b, fixture, mutation)
        rows = [r for r in authenticated["rows"] if r["role"] in ROLES]
        ready = int(len(conformance) == 640 and len(rows) == 320)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        b.checks.append(
            dict(
                check="original_source_custody",
                upstream=qualified.COHORT,
                path=str(root / qualified.COHORT),
                hash=None,
                artifact_field="original_selection",
                op="==",
                expected="authenticated original source roles",
                observed=str(error),
                passed=False,
            )
        )
    progress("source_custody_complete", len(rows))
    source_map = [dict(m, source_url="https://arxiv.org/html/" + m["arxiv"]) for m in METHODS]
    if not fixture:
        for method in METHODS:
            b.bind(ROOT / "results/raw" / NAME / "literature" / (method["arxiv"] + ".html"))
        b.bind(ROOT / "results/raw" / NAME / "historical_permission_failure.log")
    expected = dict(plan["expected_runtime_identity"])
    expected["runtime_identity"] = dict(model_revision=expected["model_revision"])
    work: Json = dict(
        rows=rows,
        protocol_conformance_rows=conformance,
        capture_manifest=manifest,
        source_custody_ready_score=ready,
        owned_failure=bool(mutation),
        expected_runtime_identity=expected,
        source_role_manifests=roles,
        source_role_masks={
            role: [r["status"] for r in rows if r["role"] == role] for role in ROLES
        },
        source_method_map=source_map,
        fixture_permission_rows=permissions,
        altered_source_diagnostics=[],
        evidence_protocol=config,
        method_freeze=freeze,
        cited_upstream_artifacts=plan["cited_upstream_artifacts"],
        gate_check_summary=[*plan["checks"], *b.checks],
        source_artifact_hashes=[
            *plan["references"],
            *b.refs,
            dict(path=str(previous.DESIGN), sha256=sha256_file(previous.DESIGN)),
        ],
        code_config_hashes={n: sha256_file(ROOT / n) for n in [*OWNED, TEST, previous.MODULE]},
        phase_spans=[
            dict(
                phase="source_freeze_and_custody", start_s=0, duration_s=time.monotonic() - started
            )
        ],
        duration_s=time.monotonic() - started,
    )
    work["raw_shard_hashes"] = [
        freeze,
        manifest,
        qualified.cohort.immutable(raw / "primitive_rows.json", dict(rows=rows)),
        qualified.cohort.immutable(raw / "runtime_preconditions.json", plan),
        qualified.cohort.immutable(raw / "fixture_permissions.json", dict(rows=permissions)),
        qualified.cohort.immutable(raw / "source_methods.json", dict(rows=source_map)),
    ]
    atomic_json(raw / "measurement.json", work)
    progress("source_measurement_normal_exit", len(rows))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Grant readiness only to source custody with normally passed owned checks."""
    value = dict(work)
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    owned = owned and all(
        r["production_write_denied"] and r["mutation_rejected"]
        for r in work["fixture_permission_rows"]
    )
    failed = [r for r in work["gate_check_summary"] if not r["passed"]]
    ready = int(owned and work["source_custody_ready_score"] and not failed)
    verdict = "circular_positive" if fixture else "null"
    terminal = "complete_" + verdict + "_source_protocol_sealed"
    if not owned:
        verdict, terminal = "disqualified", "complete_disqualified_owned_validation"
    elif failed:
        verdict, terminal = "blocked", "complete_blocked_" + failed[0]["check"]
    value.update(
        reduce_rows(work["rows"]),
        experiment_id=8137,
        task_id=TASK,
        milestone="2026.10.704",
        honest_verdict=terminal,
        verdict_class=verdict,
        source_protocol_ready_score=ready,
        required_checks_passed=owned,
        flagged_adversarial=False,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        MODEL_SPECS=[],
        call_ledger=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        claim_scope="source transport and byte custody only; no entailment, calibration or learning benefit measured",
        exposure_scope="exposed_development_within_run_disjoint",
        run_date="20261004",
        random_seed=70437,
        sample_size_budget=dict(
            intended=320,
            fit=128,
            tune=64,
            evaluation=128,
            independent_unit="original_source_cluster",
            frozen_arm_requests=640,
        ),
        acceptance_gates=dict(
            source="original roles/masks and expected identity",
            owned="normal checks, private tamper rejection and immutable production seals",
        ),
        field_principles=dict(
            readiness="Source transport readiness cannot veto downstream learning.",
            inference="Imported Qwen identity is cited upstream provenance; current calls remain zero.",
            quotes="Valid UTF-8 quotes prove custody only; invalid quotes retain probabilities.",
            diagnostics="Altered sources inherit no target; no altered-source judgment is measured.",
            science="Exposed development cannot demonstrate independent generalization.",
            denominators="320 source units; two frozen requests do not double independence.",
            verdict="External blocks are terminal; owned failures disqualify.",
            literature="Citation service availability is bibliographic, not a science gate.",
        ),
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reuse byte authentication while independently rebuilding source counts."""
    with patch.object(previous, "build", build):
        return previous.replay(path)


def main(argv: list[str] | None = None) -> int:
    """Use the qualified bounded supervisor and publication transaction."""
    import sys

    original = execution.manifest

    def manifest(private: Path, candidate: Path) -> Json:
        specs = original(private, candidate)
        config = private / "coverage.ini"
        config.write_text(config.read_text() + "    " + str(ROOT / previous.MODULE) + "\n")
        specs["commands"][3]["argv"] += ["--include=" + ",".join(str(ROOT / n) for n in OWNED)]
        specs["commands"][0]["argv"] += [previous.TEST]
        specs["commands"][1]["argv"] += [
            "tests/python/test_methods_stream_custody_8111.py",
            "tests/python/test_current_work_receipt.py",
        ]
        check = (
            "import ast,json,sys; from pathlib import Path; "
            "p=Path(sys.argv[2]); tree=ast.parse(p.read_text()); "
            "fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_seal'); "
            "lines={n.lineno for n in ast.walk(fn) if isinstance(n,ast.stmt)}; "
            "data=json.loads(Path(sys.argv[1]).read_text())['files'][sys.argv[2]]; "
            "assert lines and not lines.intersection(data['missing_lines']), data['missing_lines']"
        )
        specs["commands"].append(
            dict(
                name="changed_permission_coverage",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-c",
                    check,
                    str(private / "coverage.json"),
                    previous.MODULE,
                ],
                deadline_s=30,
                expected_exit=0,
                classification="required",
            )
        )
        return specs

    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
    ):
        return execution.main(argv)
