"""REQ-VERIFY-8620: bounded extraction and original-byte validation precede publication."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import triviaplus_extraction_8620 as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v729_contract_runner import collect
from carnot.reporting.v738_contract_runner import date_value, exact_union

Json = dict[str, Any]


def measure(root: Path, raw: Path, date: str) -> Json:
    """Authenticate CPU prerequisites and seal child results before opening human labels."""
    start = time.monotonic_ns()
    e.progress("preconditions")
    raw.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(prefix="exp8620-private-", dir="/var/tmp"))
    scratch.chmod(0o700)
    probe = scratch / "disk_probe"
    probe.write_bytes(b"disk-backed")
    probe.unlink()
    mount = max(
        (
            line.split()
            for line in Path("/proc/mounts").read_text().splitlines()
            if scratch.is_relative_to(Path(line.split()[1]))
        ),
        key=lambda row: len(row[1]),
    )
    preconditions = dict(
        memory=Path("/proc/meminfo").read_text().splitlines()[:3],
        scratch_path=str(scratch),
        scratch_free_bytes=shutil.disk_usage(scratch).free,
        writable_disk_backed_scratch=mount[2] not in {"tmpfs", "ramfs"},
        scratch_filesystem=mount[2],
        python=sys.executable,
        tools={n: e.reference(e.ROOT / ".venv/bin" / n) for n in ["pytest", "ruff", "mypy"]},
        optional_gpu_required=False,
        optional_board_required=False,
    )
    gates, sources, checks = [], [], []
    report, target, provenance = e.describe([]), {}, []
    features_ref, seal = None, None
    try:
        method, upstream, public, questions = e.content(root)
        target = dict(method["target_model"], evaluated_this_invocation=False)
        if e.sha256_file(Path(target["checkpoint"]["path"])) != target["checkpoint"]["sha256"]:
            raise ValueError("historical_head_checkpoint_hash")
        provenance = upstream["historical_model_provenance"]
        sources = [
            e.reference(root / p)
            for p in [
                e.release.METHOD,
                e.RELEASE,
                e.release.PROTOCOL,
                "ops/exclusion_manifest.yaml",
            ]
        ]
        source = scratch / "predictor.jsonl"
        source.write_text("".join(json.dumps(r) + "\n" for r in public))
        source.chmod(0o600)
        features = raw / "features.json"
        check = child(
            "predictor",
            [
                sys.executable,
                "-u",
                str(e.ROOT / e.CLI),
                "--extract",
                str(source),
                "--features",
                str(features),
            ],
            raw / "measurement_checks",
            deadline=600,
        )
        checks.append(check)
        if not check["passed"]:
            raise ValueError("predictor_owned_failure")
        sealed = json.loads(features.read_bytes())
        if [r["opaque_id"] for r in sealed["rows"]] != [r["opaque_id"] for r in public]:
            raise ValueError("feature_roster")
        features_ref, seal = e.reference(features), e.canonical_hash(sealed["rows"])
        e.atomic_json(
            raw / "score_seal.json",
            dict(
                feature_seal=seal,
                features_reference=features_ref,
                sealed_monotonic_ns=time.monotonic_ns(),
            ),
        )
        e.progress("labels_open_after_score_seal", len(public), 0)
        import pyarrow.parquet as pq

        original = pq.read_table(
            Path(upstream["private_cache_reference"]["path"]) / e.release.PARQUET
        ).to_pylist()
        report = e.describe(e.join_labels(sealed["rows"], original, questions))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if str(error) in {"feature_roster", "human_aggregation"}:
            checks.append(dict(name=str(error), passed=False, scope="owned"))
        gates.append(
            dict(
                check="original_extraction_inputs",
                upstream="exp8615-contract-methods/exp8617-triviaplus-release",
                path=str(root / e.RELEASE),
                sha256=e.sha256_file(root / e.RELEASE) if (root / e.RELEASE).is_file() else None,
                field="qualified_original_inputs",
                operator="==",
                expected="pinned qualified methods, release and complete descriptor",
                observed=str(error),
                passed=False,
            )
        )
    end = time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        report=report,
        target_model=target,
        historical_model_provenance=provenance,
        gate_check_summary=gates,
        code_refs=[e.reference(e.ROOT / p) for p in [*e.OWNED, e.TEST]],
        source_refs=sources,
        features_reference=features_ref,
        feature_seal=seal,
        measurement_receipts=checks,
        preconditions_checked=preconditions,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    e.atomic_json(raw / "measurement.json", work)
    e.progress("measurement_complete", len(report["extraction_rows"]), 0)
    return work


def build(work: Json, validation: Json, output: Path) -> Json:
    """A qualified complete census can be ready even when every score abstains."""
    raw = Path(work["raw"])
    owned = bool(validation["receipts"]) and all(
        r["passed"] for r in validation["receipts"] + work["measurement_receipts"]
    )
    gates = list(work["gate_check_summary"])
    if not owned:
        gates.append(e.release.gate(raw / "validation.json", "owned_validation", True, False))
    ready = int(owned and not gates)
    verdict = "disqualified" if not owned else "null" if ready else "blocked"
    rows = work["report"]["extraction_rows"]
    value = dict(
        work["report"],
        experiment_id=8620,
        task_id=e.TASK,
        milestone="2026.10.740",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v740_triviaplus_extraction",
        verdict_class=verdict,
        extraction_audit_ready_score=ready,
        branch_gate_values=dict(extraction_audit_ready_score=ready),
        gate_check_summary=gates,
        inference_substrate="no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=work["target_model"],
        historical_model_provenance=work["historical_model_provenance"],
        rows=[
            dict(
                opaque_id=r["opaque_id"],
                question_hash=r["question_hash"],
                status="completed",
                metric=int(r["abstention"] is None),
                missing_reason=r["abstention"],
            )
            for r in rows
        ],
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=len({r["question_hash"] for r in rows}),
        sample_size_budget=dict(question_clusters=128, all_siblings=True),
        verifier_is_oracle=False,
        exposure_scope="released historical generators; historical overlap unknown; lexical internal audit",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(
            validation.get("adversarial_findings", {}).get("flagged_count", 0)
        ),
        acceptance_gates=dict(owned_checks=owned, original_inputs=not work["gate_check_summary"]),
        validation_receipts=validation["receipts"] + work["measurement_receipts"],
        adversarial_findings=validation.get("adversarial_findings", {}),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=e.SEED,
        source_artifact_hashes=work["source_refs"],
        cited_upstream_artifacts=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[work["features_reference"]] if work["features_reference"] else [],
        primitive_reference=e.reference(raw / "measurement.json"),
        validation_reference=e.reference(raw / "validation.json"),
        publication_output=str(output.absolute()),
        missing_q_contract=dict(
            q_available=False,
            head_inference_authorized=False,
            reason="The nine-input109-coefficient energy head requires released p2 judge confidence, unavailable in Trivia+; no noisy labels or0.5 substitution.",
        ),
        followup_evidence_requirements=dict(
            long_context_coverage=bool(work["report"]["abstention_reason_counts"]),
            label_disagreement=any(r["annotation_disagreement"] for r in rows),
            missing_judge_scores=True,
            calibration_study_ready=False,
            evidence="Complete long-context support, annotation sensitivity and valid released p2 q are required before calibration; historical overlap remains unknown.",
        ),
        methodology_note="Complete-source lexical coverage; four exploratory scores; strict release majority with severity tie order; response target is any unsupported/contradicted sentence, absent if any sentence lacks a vote. No training, threshold, winner, factual-error certification, or current-Qwen claim.",
    )
    value["field_principles"] = {
        k: "Recorded operand or descriptive census value; does not establish detector benefit."
        for k in value
    }
    value["field_principles"].update(
        extraction_audit_ready_score="Numeric1 iff original inputs and owned checks qualify the complete honest census, including zero usable rows.",
        fixed_score_rows="Fixed lexical reductions, never calibrated probabilities; uncovered_fraction mean and1-overlap mean are complementary duplicates.",
        descriptive_auroc_intervals="Question bootstrap10000 draws seed7398607; intervals require32 usable clusters and8 clusters per class; no winner selection.",
        annotation_disagreement_rows="Raw votes remain separate from strict aggregation; unsupported and contradicted strata may overlap.",
        source_length_strata="Complete source windows1..32/33..128/>128 crossed with answer units<=16/>16; zero windows explicitly counted.",
    )
    value["reproducibility_checksum"] = e.canonical_hash(value)
    return value


def run_checks(raw: Path) -> Json:
    """Collect original nodes, then preserve their exact disjoint execution union."""
    scratch = Path(mkdtemp(prefix="exp8620-validation-", dir="/var/tmp"))
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "\n".join("    " + str(e.ROOT / p) for p in e.OWNED)
        + "\n[report]\nfail_under = 100\n"
    )
    receipts, plan = [], []
    previous = os.environ.get("COVERAGE_FILE")
    os.environ["COVERAGE_FILE"] = str(scratch / ".coverage")
    try:
        for suite in [
            e.TEST,
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]:
            nodes, receipt = collect(scratch, suite)
            receipts.append(receipt)
            plan.append(dict(name=receipt["name"], collected_node_ids=nodes))
            size = len(nodes) or 1 if suite == e.TEST else 1 if "8210" in suite else 8
            for i in range(0, len(nodes), size):
                group = nodes[i : i + size]
                name = Path(suite).stem + f"_{i}"
                junit = scratch / (name + ".xml")
                argv = (
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        "run",
                        "--rcfile=" + str(config),
                        "-m",
                        "pytest",
                    ]
                    if suite == e.TEST
                    else [str(e.ROOT / ".venv/bin/pytest")]
                )
                argv += [
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *group,
                    "--junitxml=" + str(junit),
                    "--basetemp=" + str(scratch / name),
                ]
                plan.append(
                    dict(name=name, argv=argv, collected_node_ids=group, junit_path=str(junit))
                )
                receipts.append(child(name, argv, raw / "checks", deadline=180))
        receipts.append(exact_union(plan))
        commands = [
            (
                "coverage_combine",
                [sys.executable, "-m", "coverage", "combine", "--keep", str(scratch)],
            ),
            (
                "coverage_report",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "report",
                    "--rcfile=" + str(config),
                    "--show-missing",
                ],
            ),
            (
                "coverage_json",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "json",
                    "--rcfile=" + str(config),
                    "-o",
                    str(raw / "coverage.json"),
                ],
            ),
            ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
            (
                "ruff_format",
                [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST],
            ),
            (
                "mypy",
                [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
            ),
            (
                "spec_coverage",
                [sys.executable, "scripts/check_spec_coverage.py", "--files", e.TEST],
            ),
        ]
        for name, argv in commands:
            receipts.append(child(name, argv, raw / "checks", deadline=120))
    finally:
        if previous is None:
            os.environ.pop("COVERAGE_FILE", None)
        else:
            os.environ["COVERAGE_FILE"] = previous
    e.atomic_json(raw / "execution_manifest.json", dict(plan=plan))
    return dict(receipts=receipts)


def validate_candidate(path: Path, raw: Path) -> Json:
    """Use fresh replay and unchanged validators on the exact candidate bytes."""
    logs = raw / "terminal_checks" / e.sha256_file(path).split(":")[1]
    commands = [
        ("cold_replay", [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)]),
        ("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]),
        (
            "row_lint",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        ),
    ]
    receipts = [child(name, argv, logs, deadline=180) for name, argv in commands]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(r["passed"] for r in receipts), receipts=receipts, adversarial_findings=findings
    )


def main(argv: list[str] | None = None) -> int:
    """Publish checked census bytes; preserve external blocks and owned failures separately."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args(argv)
    if args.extract:
        if args.features is None:
            parser.error("--features is required with --extract")
        e.extract_file(args.extract, args.features)
        return 0
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    previous = signal.setitimer(signal.ITIMER_REAL, 3600)
    started = time.monotonic()
    try:
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        work = measure(args.root, raw, args.date)
        signal.setitimer(signal.ITIMER_REAL, 4800 - (time.monotonic() - started))
        validation = run_checks(raw)
        e.atomic_json(raw / "validation.json", validation)
        candidate = raw / "private_candidate.json"
        value = build(work, validation, output)
        e.atomic_json(candidate, value)
        for mode in ["missing", "rehashed"]:
            changed = json.loads(candidate.read_bytes())
            if mode == "missing":
                changed["primitive_reference"]["path"] = str(raw / "absent.json")
            else:
                changed["extraction_audit_ready_score"] = (
                    1 - changed["extraction_audit_ready_score"]
                )
                changed["reproducibility_checksum"] = e.canonical_hash(
                    {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
                )
            control = raw / (mode + ".json")
            e.atomic_json(control, changed)
            validation["receipts"].append(
                child(
                    "reject_" + mode,
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(control)],
                    raw / "controls",
                    expected=1,
                    deadline=180,
                )
            )
        e.atomic_json(raw / "validation.json", validation)
        e.atomic_json(candidate, build(work, validation, output))
        report = validate_candidate(candidate, raw)
        validation["receipts"] += report["receipts"]
        validation["adversarial_findings"] = report["adversarial_findings"]
        e.atomic_json(raw / "validation.json", validation)
        value = build(work, validation, output)

        def final_validator(path: Path) -> Json:
            checked = validate_candidate(path, raw)
            safe = value["verdict_class"] == "disqualified" and all(
                r["passed"] for r in checked["receipts"] if r["name"] != "adversarial"
            )
            return dict(
                checked,
                validation_passed=checked["passed"],
                passed=checked["passed"] or safe,
                failed_result_publication=safe,
            )

        result = publish_primary(output, value, final_validator)
        readers = reader_receipt(
            e.TASK,
            output.parent,
            field="extraction_audit_ready_score",
            expected=value["extraction_audit_ready_score"],
        )
        e.atomic_json(raw / "terminal_validation.json", dict(result, reader_receipt=readers))
        e.progress("published", value["completed_count"], value["censored_count"])
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
