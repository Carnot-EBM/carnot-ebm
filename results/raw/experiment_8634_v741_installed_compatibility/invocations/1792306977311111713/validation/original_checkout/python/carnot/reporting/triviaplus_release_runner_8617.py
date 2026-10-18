"""REQ-VERIFY-8617: bounded validation qualifies the private release inventory."""

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

from carnot.reporting import triviaplus_release_8617 as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v738_contract_runner import date_value, exact_union
from carnot.reporting.v729_contract_runner import collect

Json = dict[str, Any]


def measure(root: Path, raw: Path, cache: Path, date: str) -> Json:
    """Check CPU prerequisites and preserve original inputs before reduction."""
    start = time.monotonic_ns()
    e.progress("preconditions")
    if cache.resolve().is_relative_to((root / "results").resolve()):
        raise ValueError("private_cache_outside_results_required")
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    cache.chmod(0o700)
    probe = cache / ".disk_probe"
    probe.write_bytes(b"disk-backed")
    probe.unlink()
    preconditions = dict(
        memory=Path("/proc/meminfo").read_text().splitlines()[:3],
        scratch_path=str(cache),
        scratch_free_bytes=shutil.disk_usage(cache).free,
        writable_disk_backed_scratch=True,
        python=sys.executable,
        tools={n: e.reference(e.ROOT / ".venv/bin" / n) for n in ["pytest", "ruff", "mypy"]},
        optional_gpu_required=False,
        optional_board_required=False,
    )
    refs, gates = e.acquire(cache)
    sources = []
    plan: Json = {}
    target: Json = {}
    for name, digest in [(e.METHOD, e.METHOD_HASH), (e.PROTOCOL, e.PROTOCOL_HASH)]:
        path = root / name
        try:
            value = e.read(dict(path=str(path), sha256=digest))
            sources.append(e.reference(path))
            if name == e.METHOD:
                plan = value["triviaplus_coverage_plan"]
                target = value["target_model"]
                if value["methods_ready_score"] != 1 or not value["required_checks_passed"]:
                    gates.append(
                        e.gate(path, "methods_ready_score", 1, value.get("methods_ready_score"))
                    )
        except (OSError, ValueError, KeyError) as error:
            gates.append(e.gate(path, "original_methods_authority", digest, str(error)))
    rows = e.release_rows(cache) if len(refs) == len(e.BLOBS) else []
    reduced = e.reduce_rows(rows)
    prior = [
        root / ("results/" + name + ".json")
        for name in [
            "experiment_8305_v717_cached_sentence_custody",
            "experiment_8403_v724_released_question_identity",
            "experiment_8430_v726_criterion_feature_panel",
            "experiment_8553_v735_hallmark_release_admissibility",
            "experiment_8585_v737_hallutruth_release_audit",
            "experiment_8597_v738_medexpert_evidence_audit",
        ]
    ]
    overlap = e.overlap_inventory(rows, prior)
    sources += [e.reference(p) for p in prior if p.is_file()]
    sources.append(e.reference(root / "ops/exclusion_manifest.yaml"))
    e.progress("measurement_complete", len(rows), 0)
    end = time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        cache=str(cache),
        run_date=date,
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        release_file_hashes=refs,
        gate_check_summary=gates,
        reduced=reduced,
        source_refs=sources,
        code_refs=[e.reference(e.ROOT / p) for p in [*e.OWNED, e.TEST]],
        overlap_rows=overlap,
        plan=plan,
        target_model=target,
        preconditions_checked=preconditions,
        private_predictor_reference=e.reference(cache / "predictor.jsonl") if rows else None,
        retrieval_receipts_reference=e.reference(cache / "retrieval_receipts.json")
        if (cache / "retrieval_receipts.json").is_file()
        else None,
    )
    e.atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """A qualified exposed coverage input is distinct from independent generalization."""
    reduced = work["reduced"]
    gates = list(work["gate_check_summary"])
    owned = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    available = int(len(work["release_file_hashes"]) == len(e.BLOBS))
    private = int(
        bool(available and reduced["complete_custody"] and work["plan"].get("internal_only"))
    )
    ready = int(owned and not gates and private and reduced["annotation_authority_score"])
    if not owned:
        gates.append(e.gate(raw / "validation.json", "owned_validation", True, False))
    if not reduced["annotation_authority_score"]:
        gates.append(e.gate(Path(work["cache"]) / e.PARQUET, "human_aggregation", 1, 0))
    if not private:
        gates.append(
            e.gate(Path(work["cache"]), "private_complete_source_answer_internal_use", 1, 0)
        )
    verdict = "disqualified" if not owned else "circular_positive" if ready else "blocked"
    count = len(reduced["rows"])
    value = dict(
        reduced,
        experiment_id=8617,
        task_id=e.TASK,
        milestone="2026.10.740",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v740_triviaplus_release",
        verdict_class=verdict,
        gate_check_summary=gates,
        inference_substrate="no_model_load",
        inference_substrate_details="CPU release parsing and human vote aggregation; no numerical head inference",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=dict(work["target_model"], evaluated_this_invocation=False),
        intended_count=3224,
        completed_count=count,
        failed_count=0,
        censored_count=3224 - count,
        excluded_count=0,
        independent_count=len(reduced["sibling_groups"]),
        sample_size_budget=dict(
            release_responses=3224,
            roster_question_cap=128,
            actual_roster_questions=len(reduced["label_blind_roster"]["question_hashes"]),
        ),
        verifier_is_oracle=True,
        exposure_scope="released_human_labels_internal_exposed_coverage_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            release_authenticated=bool(available),
            human_aggregation=bool(reduced["annotation_authority_score"]),
            private_internal_use=bool(private),
            methods_authority=not work["gate_check_summary"],
        ),
        validation_receipts=validation["receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=validation.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=[dict(phase="measurement", start_ns=work["start_ns"], end_ns=work["end_ns"])],
        random_seed="seed7398607",
        source_artifact_hashes=work["source_refs"]
        + [work["release_file_hashes"][name] for name in sorted(work["release_file_hashes"])],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[e.reference(raw / "measurement.json")],
        cited_upstream_artifacts=work["source_refs"],
        triviaplus_coverage_inputs_ready_score=ready,
        release_available_score=available,
        private_use_ready_score=private,
        release_revision=e.REVISION,
        release_file_hashes=work["release_file_hashes"],
        overlap_rows=work["overlap_rows"],
        private_cache_reference=dict(
            path=work["cache"], predictor=work["private_predictor_reference"]
        ),
        predictor_field_allowlist=e.ALLOWLIST,
        internal_use_decision="Internal noncommercial evaluation only; scope resolved for this task",
        redistribution_decision="No source or transformed text redistribution; no product-training permission",
        primitive_reference=e.reference(raw / "measurement.json"),
        validation_reference=e.reference(raw / "validation.json"),
        publication_output=str(output),
        methodology_note="Pinned author release; reconstruct raw human votes, freeze label-blind questions and retain complete private source. Descriptive coverage inputs only; no detector benefit, numerical inference, fitting or independent generalization.",
    )
    value["branch_gate_values"] = {
        k: value[k]
        for k in [
            "triviaplus_coverage_inputs_ready_score",
            "release_available_score",
            "annotation_authority_score",
            "private_use_ready_score",
            "question_disjoint_score",
        ]
    }
    value["field_principles"] = {
        k: "Bind this invocation to exact original operands; custody qualification alone cannot establish detector benefit."
        for k in value
    }
    value["field_principles"].update(
        triviaplus_coverage_inputs_ready_score="One only for authenticated release, human aggregation, usable private custody and passed owned checks; disjointness is separate.",
        release_available_score="One only when every release Git blob authenticates.",
        annotation_authority_score="One only when raw human votes reproduce every released target; automatic and noisy labels are excluded.",
        private_use_ready_score="One for complete private source/answer custody and registered internal-only use.",
        question_disjoint_score="Zero: historical exposure is incompletely known and this is an exposed coverage study.",
        independent_count="Normalized question clusters in the entire release; answers and siblings do not multiply questions.",
        label_aggregation_rows="Human majority counts, strict ties, n/a missing votes and surplus annotation positions; no automatic targets.",
        overlap_rows="Compare actual historical corpus manifests; unavailable identities remain unknown rather than zero overlap.",
        private_cache_reference="Complete author source, answers, annotations and label-blind predictor stay in a private local directory.",
    )
    value["reproducibility_checksum"] = e.canonical_hash(value)
    return value


def run_checks(raw: Path) -> Json:
    """Freeze original IDs before executing disjoint bounded test children."""
    scratch = Path(mkdtemp(prefix="exp8617-validation-", dir="/var/tmp"))
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "\n".join("    " + str(e.ROOT / p) for p in e.OWNED)
        + "\n[report]\nfail_under = 100\n"
    )
    receipts = []
    plan = []
    suites = [
        e.TEST,
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    prefix = [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config)]
    for suite in suites:
        nodes, receipt = collect(scratch, suite)
        receipts.append(receipt)
        plan.append(dict(name=receipt["name"], collected_node_ids=nodes))
        size = len(nodes) if suite == e.TEST else 1 if "8210" in suite else 8
        for i in range(0, len(nodes), size):
            group = nodes[i : i + size]
            name = Path(suite).stem + f"_{i}"
            junit = scratch / (name + ".xml")
            argv = (
                prefix + ["-m", "pytest"] if suite == e.TEST else [str(e.ROOT / ".venv/bin/pytest")]
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
                dict(
                    name=name,
                    argv=argv,
                    deadline=180,
                    expected=0,
                    collected_node_ids=group,
                    junit_path=str(junit),
                )
            )
    old_coverage = os.environ.get("COVERAGE_FILE")
    os.environ["COVERAGE_FILE"] = str(scratch / ".coverage")
    try:
        for spec in plan:
            if "argv" in spec:
                receipts.append(
                    child(spec["name"], spec["argv"], raw / "checks", deadline=spec["deadline"])
                )
        receipts.append(exact_union(plan))
        for name, argv in [
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
        ]:
            receipts.append(child(name, argv, raw / "checks", deadline=120))
    finally:
        if old_coverage is None:
            os.environ.pop("COVERAGE_FILE", None)
        else:
            os.environ["COVERAGE_FILE"] = old_coverage
    e.atomic_json(raw / "execution_manifest.json", dict(plan=plan))
    return dict(receipts=receipts)


def validate_candidate(path: Path, raw: Path) -> Json:
    """Read final candidate bytes through cold replay and unchanged shared validators."""
    commands = [
        ("cold_replay", [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)]),
        ("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]),
        (
            "row_lint",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        ),
    ]
    logs = raw / "terminal_checks" / e.sha256_file(path).split(":")[1]
    receipts = [child(name, argv, logs, deadline=120) for name, argv in commands]
    adversarial = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        adversarial_findings=adversarial,
    )


def main(argv: list[str] | None = None) -> int:
    """Publish only checked bytes while preserving external and owned failures separately."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/carnot-private-8617"))
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        invocation_start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, 3600)
        work = measure(args.root, raw, args.cache, args.date)
        signal.setitimer(signal.ITIMER_REAL, 4800 - (time.monotonic() - invocation_start))
        validation = run_checks(raw)
        e.atomic_json(raw / "validation.json", validation)
        value = build(work, validation, raw, output)
        candidate = raw / "private_candidate.json"
        e.atomic_json(candidate, value)
        controls = []
        for mode in ["missing", "rehashed"]:
            changed = json.loads(candidate.read_bytes())
            if mode == "missing":
                changed["primitive_reference"]["path"] = str(raw / "absent.json")
            else:
                changed["release_available_score"] = 1 - changed["release_available_score"]
                changed["reproducibility_checksum"] = e.canonical_hash(
                    {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
                )
            control = raw / (mode + ".json")
            e.atomic_json(control, changed)
            controls.append(
                child(
                    "reject_" + mode,
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(control)],
                    raw / "controls",
                    expected=1,
                    deadline=60,
                )
            )
        validation["receipts"] += controls
        e.atomic_json(raw / "validation.json", validation)
        value = build(work, validation, raw, output)
        e.atomic_json(candidate, value)
        report = validate_candidate(candidate, raw)
        validation["receipts"] += report["receipts"]
        validation["adversarial_findings"] = report["adversarial_findings"]
        e.atomic_json(raw / "validation.json", validation)
        value = build(work, validation, raw, output)

        def final_validator(path: Path) -> Json:
            checked = validate_candidate(path, raw)
            safe_failure = value["verdict_class"] == "disqualified" and all(
                r["passed"] for r in checked["receipts"] if r["name"] != "adversarial"
            )
            return dict(
                checked,
                validation_passed=checked["passed"],
                passed=checked["passed"] or safe_failure,
                failed_result_publication=safe_failure,
            )

        result = publish_primary(output, value, final_validator)
        readers = reader_receipt(
            e.TASK,
            output.parent,
            field="triviaplus_coverage_inputs_ready_score",
            expected=value["triviaplus_coverage_inputs_ready_score"],
        )
        e.atomic_json(raw / "terminal_validation.json", dict(result, reader_receipt=readers))
        e.progress("published", value["completed_count"], value["censored_count"])
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
