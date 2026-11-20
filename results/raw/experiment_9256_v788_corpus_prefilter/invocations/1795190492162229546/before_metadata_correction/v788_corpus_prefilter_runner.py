from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import logit_methods_runner_9126 as lifecycle
from carnot.reporting import v788_corpus_prefilter as p
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.validation_diagnostics_9181 import run
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]

CONSUMER_TESTS = (
    "tests/python/test_adversarial_verify_principle_wrapped_fields.py",
    "tests/python/test_qa_layer_widenings_20260901.py",
    "tests/python/test_adversarial_verify_substrate_class_20260905.py",
    "tests/python/test_adversarial_verify_substrate_classification_5933.py",
    "tests/python/test_adversarial_verify_exp5895_no_llm_substrate.py",
    "tests/python/test_arc_solve_artifact_discipline.py",
    "tests/python/test_arc_artifact_lint.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
)
V782_CONTROL_TESTS = (
    "tests/python/test_forward_evidence_9111.py",
    "tests/python/test_verdict_row_consistency_lint.py",
    "tests/python/test_v782_contract_9180.py",
    "tests/python/test_causal_validation_9167.py",
    "tests/python/test_causal_validation_9181.py",
    "tests/python/test_validation_schedule_9139.py",
)
E2E_TESTS = (
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
)
REQUIRED_CHECKS = (
    "target_module",
    "owned_tests",
    "v782_controls",
    "direct_consumers",
    "e2e018_021",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "full_python_suite",
)


def checksum(value: Json) -> str:
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build_scope(root: Path, raw: Path) -> Json:
    raw.mkdir(parents=True, exist_ok=True)
    cov_ini = raw / "coverage.ini"
    inc_lines = "\n".join(f"    {p.ROOT / rel}" for rel in (*p.OWNED, *p.MODIFIED))
    cov_ini.write_text(
        f"[run]\nbranch = False\nparallel = True\npatch = subprocess\ninclude =\n{inc_lines}\n[report]\nexclude_lines =\n",
        encoding="utf-8",
    )
    priv_root = Path(tempfile.mkdtemp(prefix=f"carnot-9256-{raw.name}-", dir="/var/tmp"))
    ruff_bin = str(p.ROOT / ".venv/bin/ruff")
    mypy_bin = str(p.ROOT / ".venv/bin/mypy")
    py_bin = str(p.ROOT / ".venv/bin/python")
    scan_out = str(raw / "corpus_scan_summary.json")

    def _priv_pytest(
        name: str,
        files: tuple[str, ...],
        extra_env: list[str],
        deadline: int,
        *,
        with_cov: bool = False,
    ) -> Json:
        pdir = priv_root / name
        pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cov_env = (
            [f"COVERAGE_RCFILE={cov_ini}", f"COVERAGE_FILE={raw / f'.coverage.{name}'}"]
            if with_cov
            else []
        )
        run_prefix = (
            [py_bin, "-m", "coverage", "run", f"--rcfile={cov_ini}", "-m", "pytest"]
            if with_cov
            else [py_bin, "-m", "pytest"]
        )
        return {
            "name": name,
            "deadline_s": deadline,
            "argv": [
                "env",
                f"--chdir={pdir}",
                "--unset=PYTHONPATH",
                "TMPDIR=/var/tmp",
                *cov_env,
                *extra_env,
                *run_prefix,
                "-c",
                str(p.ROOT / "pyproject.toml"),
                "--rootdir",
                str(p.ROOT),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *(str(p.ROOT / f) for f in files),
            ],
        }

    commands = [
        _priv_pytest(
            "target_module",
            ("tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py",),
            [f"CARNOT_9256_SCAN_OUTPUT={scan_out}"],
            3300,
            with_cov=True,
        ),
        _priv_pytest(
            "owned_tests",
            ("tests/python/test_v788_corpus_prefilter_9256.py",),
            [],
            600,
            with_cov=True,
        ),
        _priv_pytest("v782_controls", V782_CONTROL_TESTS, [], 600, with_cov=True),
        _priv_pytest("direct_consumers", CONSUMER_TESTS, [], 600),
        _priv_pytest("e2e018_021", E2E_TESTS, [], 600),
        {
            "name": "ruff_check",
            "deadline_s": 300,
            "argv": [ruff_bin, "check", *(str(p.ROOT / f) for f in (*p.OWNED, *p.OWNED_TESTS))],
        },
        {
            "name": "ruff_format",
            "deadline_s": 300,
            "argv": [
                ruff_bin,
                "format",
                "--check",
                *(str(p.ROOT / f) for f in (*p.OWNED, *p.OWNED_TESTS)),
            ],
        },
        {
            "name": "strict_mypy",
            "deadline_s": 300,
            "argv": [
                mypy_bin,
                "--config-file",
                str(p.ROOT / "pyproject.toml"),
                "--strict",
                *(str(p.ROOT / f) for f in p.OWNED),
            ],
        },
        {
            "name": "spec_coverage",
            "deadline_s": 300,
            "argv": [
                py_bin,
                str(p.ROOT / "scripts/check_spec_coverage.py"),
                *(str(p.ROOT / f) for f in p.OWNED_TESTS),
            ],
        },
        {
            "name": "full_python_suite",
            "deadline_s": 600,
            "argv": [str(p.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        },
    ]
    scope = {
        "owned_changed_paths": list(p.OWNED),
        "modified_paths": list(p.MODIFIED),
        "owned_test_paths": list(p.OWNED_TESTS),
        "required_check_names": list(REQUIRED_CHECKS),
        "coverage_config_reference": p.file_ref(cov_ini, root),
        "commands": commands,
    }
    atomic_json(raw / "frozen_scope.json", scope)
    return scope


def modified_coverage(data_reference: Json, paths: list[str]) -> list[Json]:
    """Measure added statements in shared files without charging old code to this change."""
    tracer = Coverage(data_file=data_reference["path"], config_file=False)
    tracer.set_option("report:exclude_lines", [])
    tracer.load()
    rows = []
    base = p.ROOT / p.RAW / "baseline"
    for rel in paths:
        source = p.ROOT / rel
        old = (base / source.name).read_text().splitlines()
        new = source.read_text().splitlines()
        changed: set[int] = set()
        for tag, _, _, start, end in difflib.SequenceMatcher(None, old, new).get_opcodes():
            if tag in {"insert", "replace"}:
                changed.update(range(start + 1, end + 1))
        _, statements, excluded, missing, _ = tracer.analysis2(str(source))
        added = changed.intersection(statements)
        rows.append(
            {
                "path": rel,
                "statements": sorted(added),
                "missing": sorted(added.intersection(missing)),
                "excluded": sorted(changed.intersection(excluded)),
                "baseline_reference": p.file_ref(base / source.name, p.ROOT),
            }
        )
    return rows


def execute_validation(
    raw: Path, scope: Json, stop: float | None = None, *, target_only: bool = False
) -> Json:
    checks: list[Json] = []
    stop = time.monotonic() + p.VERIFICATION_S if stop is None else stop
    for cmd in scope["commands"]:
        if (cmd["name"] == "target_module") != target_only:
            continue
        p.log(f"validation_cmd_start name={cmd['name']}")
        rec = run(
            cmd["name"],
            cmd["argv"],
            raw / "validation",
            min(stop, time.monotonic() + cmd["deadline_s"]),
        )
        rec["scope_sha256"] = canonical_hash(scope)
        checks.append(rec)
        p.log(
            f"validation_cmd_done name={cmd['name']} passed={rec.get('passed')} "
            f"duration_s={rec.get('duration_s')}"
        )
    if target_only:
        return {"checks": checks}
    cov = dict(
        lifecycle.measured(
            sorted(raw.glob(".coverage.*")),
            list(scope["owned_changed_paths"]),
            raw / "combined_coverage",
        )
    )
    res = {"checks": checks, "owned_statement_coverage": cov}
    res["modified_statement_coverage"] = modified_coverage(
        cov["data_reference"], scope["modified_paths"]
    )
    atomic_json(raw / "validation.json", res)
    return res


def receipts_pass(work: Json, validation: Json) -> bool:
    by_name = {item["name"]: item for item in validation.get("checks", []) if "name" in item}
    for req in work["scope"]["required_check_names"]:
        rec = by_name.get(req)
        if not rec or not rec.get("passed") or rec.get("exit_code") != 0:
            return False
        reference = rec.get("receipt_reference", {})
        receipt_path = Path(reference.get("path", "/missing-9256"))
        if not receipt_path.is_file() or p.file_ref(receipt_path, None)["sha256"] != reference.get(
            "sha256"
        ):
            return False
        actual = json.loads(receipt_path.read_bytes())
        if any(
            actual.get(key) != rec.get(key)
            for key in ("name", "argv", "exit_code", "passed", "stdout_sha256", "stderr_sha256")
        ):
            return False
        for stream in ("stdout", "stderr"):
            path = Path(rec[f"{stream}_path"])
            if not path.is_file() or p.file_ref(path, None)["sha256"] != rec[f"{stream}_sha256"]:
                return False
    cov = validation.get("owned_statement_coverage", {})
    if not cov.get("complete") or float(cov.get("percent", 0.0)) < 100.0:
        return False
    files = cov.get("files", [])
    obs_paths = [str(item.get("path")) for item in files]
    if obs_paths != list(work["scope"]["owned_changed_paths"]):
        return False
    modified = validation.get("modified_statement_coverage", [])
    return (
        verify_measured(cov, list(work["scope"]["owned_changed_paths"]))
        and [row["path"] for row in modified] == list(work["scope"].get("modified_paths", []))
        and all(
            row["statements"] and not row["missing"] and not row["excluded"] for row in modified
        )
    )


def measure(
    root: Path,
    raw: Path,
    date: str,
    *,
    scope: Json | None = None,
    inspection: Json | None = None,
    started_monotonic_ns: int | None = None,
) -> Json:
    start_ns = time.monotonic_ns() if started_monotonic_ns is None else started_monotonic_ns
    raw.mkdir(parents=True, exist_ok=True)
    active_scope = build_scope(root, raw) if scope is None else scope
    inspected = p.inspect(root, raw) if inspection is None else inspection
    ended_ns = time.monotonic_ns()
    cw_receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="corpus_byte_prefilter_streaming_qualification",
        inference_substrate_details={
            "work": (
                "Authenticate Exp9255 V788 contract, Exp9181 direct_consumers timeout receipt, "
                "Exp9241/9242 V787 history, qualify lossless five-sentinel byte prefilter over "
                "complete recursive results/**/*.json manifest, prove exact semantic parity and "
                "mutation detection; zero LLM loads, forwards or generations."
            )
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=ended_ns,
    )
    cw_path = raw / "current_work.json"
    atomic_json(cw_path, cw_receipt)
    work = {
        "root": str(root),
        "raw": str(raw),
        "run_date": date,
        "scope": active_scope,
        "inspection": inspected,
        "current_work_reference": p.file_ref(cw_path, None),
        "start_ns": start_ns,
        "duration_s": (ended_ns - start_ns) / 1e9,
    }
    atomic_json(raw / "work.json", work)
    return work


def _build_rows(inspected: Json, blocked: bool) -> list[Json]:
    rows: list[Json] = []
    ctrl = inspected["controls"]
    for idx, item in enumerate(ctrl["parity_rows"]):
        ok = int(bool(item["parity_match"]))
        qid = f"parity_{idx:02d}_{item['fixture']}"
        rows.append(
            {
                "unit_id": f"{qid}:candidate",
                "qid": qid,
                "group": qid,
                "sibling": "prefilter_vs_original",
                "arm": "five_sentinel_byte_prefilter",
                "condition": item["category"],
                "seed": p.SEED,
                "metric": "exact_semantic_parity",
                "absolute_metric": None if blocked else float(ok),
                "numerator": 0 if blocked else ok,
                "denominator": 1,
                "status": "censored" if blocked else "completed",
                "exclusion_reason": None,
                "censor_reason": "gate_check_failed" if blocked else None,
            }
        )
    for idx, m_item in enumerate(ctrl["mutation_controls"]):
        ok = int(bool(m_item["mutation_detected"]))
        qid = f"mutation_{idx:02d}_{m_item['control_id']}"
        rows.append(
            {
                "unit_id": f"{qid}:control",
                "qid": qid,
                "group": qid,
                "sibling": "adversarial_deletion_control",
                "arm": "five_sentinel_byte_prefilter",
                "condition": "mutation_control",
                "seed": p.SEED,
                "metric": "mutation_detected",
                "absolute_metric": None if blocked else float(ok),
                "numerator": 0 if blocked else ok,
                "denominator": 1,
                "status": "censored" if blocked else "completed",
                "exclusion_reason": None,
                "censor_reason": "gate_check_failed" if blocked else None,
            }
        )
    for idx, s_item in enumerate(inspected["stratum_costs"]["strata"]):
        ok = int(s_item["parity_discrepancies"] == 0)
        qid = f"stratum_{idx:02d}_{s_item['stratum']}"
        rows.append(
            {
                "unit_id": f"{qid}:clock_parity",
                "qid": qid,
                "group": qid,
                "sibling": "stratum_benchmark",
                "arm": "five_sentinel_byte_prefilter",
                "condition": s_item["stratum"],
                "seed": p.SEED,
                "metric": "stratum_parity_verified",
                "absolute_metric": None if blocked else float(ok),
                "numerator": 0 if blocked else ok,
                "denominator": 1,
                "status": "censored" if blocked else "completed",
                "exclusion_reason": None,
                "censor_reason": "gate_check_failed" if blocked else None,
            }
        )
    return rows


def build(root: Path, raw: Path, work: Json, validation: Json, date: str) -> Json:
    inspected = work["inspection"]
    scope = work["scope"]
    pre_ok = all(item["available"] for item in inspected["preconditions"])
    gates_ok = all(item["passed"] for item in inspected["gate_checks"])
    blocked = not (pre_ok and gates_ok)

    scan = inspected["corpus_scan"]
    ctrl = inspected["controls"]
    costs = inspected["stratum_costs"]
    checks_ok = receipts_pass({"scope": scope}, validation)
    scan_ok = (
        int(scan["total_files"]) > 0
        and int(scan["visited_files"]) == int(scan["total_files"])
        and int(scan["unfinished_files"]) == 0
        and int(scan["changed_files"]) == 0
        and bool(scan["candidate_bytes_stable"])
        and bool(scan["semantic_parity_exact"])
        and int(scan["checked"]) > 0
        and len(scan["offenders"]) == 0
        and bool(ctrl["semantic_parity_exact"])
        and bool(ctrl["all_mutations_detected"])
        and not bool(costs["full_corpus_speedup_claimed"])
    )
    ready = int(not blocked and checks_ok and scan_ok)
    verdict_cls = "blocked" if blocked else ("null" if ready == 1 else "disqualified")
    honest = f"complete_{verdict_cls}_v788_corpus_prefilter"
    rows = _build_rows(inspected, blocked)
    total_units = len(rows)
    comp_units = 0 if blocked else total_units
    cens_units = total_units if blocked else 0

    cw_path = raw / "current_work.json"
    work_path = raw / "work.json"
    art: Json = {
        "schema": "carnot.v788_corpus_prefilter.v1",
        "experiment_id": p.EXP_ID,
        "task_id": p.TASK,
        "milestone": p.MILESTONE,
        "run_date": date,
        "honest_verdict": honest,
        "verdict_class": verdict_cls,
        "corpus_scan_ready_score": ready,
        "required_checks_passed": bool(not blocked and checks_ok),
        "flagged_adversarial": False,
        "gate_check_summary": inspected["gate_checks"],
        "preconditions_checked": inspected["preconditions"],
        "rows": rows,
        "intended_count": total_units,
        "attempted_count": comp_units,
        "completed_count": comp_units,
        "failed_count": 0,
        "censored_count": cens_units,
        "excluded_count": 0,
        "effective_independent_groups": comp_units,
        "sample_size_budget": {
            "intended": total_units,
            "attempted": comp_units,
            "completed": comp_units,
            "failed": 0,
            "censored": cens_units,
            "effective_independent_groups": comp_units,
        },
        "verifier_is_oracle": True,
        "independent_generalization_score": 0,
        "claim_scope": {
            "exposure_scope": "exposed_development_only",
            "independent_generalization_score": 0,
            "allowed_claims": [
                "lossless five-sentinel plus JSON Unicode-escape byte prefilter over recursive results/**/*.json",
                "exact semantic parity against unchanged read_text/json.loads and _is_arc_live_agent_no_llm classifier",
                "predetermined size-stratum read/decode/classifier cost comparison without full-corpus speedup ratio claim from censored Exp9181 baseline",
            ],
        },
        "MODEL_SPECS": [],
        "model_invocation_status": "not_requested_no_model_load",
        "requested_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run" if blocked else "no_model_load",
        "inference_substrate": "corpus_byte_prefilter_streaming_qualification",
        "current_llm_invocations": 0,
        "generated_tokens": 0,
        "model_invocation_counts": {
            "current_llm_invocations": 0,
            "loads_attempted": 0,
            "loads_completed": 0,
            "loads_failed": 0,
            "forward_calls": 0,
            "generated_tokens": 0,
        },
        "duration_s": work["duration_s"],
        "random_seed": p.SEED,
        "source_references": inspected["source_references"],
        "code_config_hashes": inspected["code_config_hashes"],
        "current_work_receipts": [p.file_ref(cw_path, None)],
        "validation_receipts": validation.get("checks", []),
        "owned_statement_coverage": validation.get("owned_statement_coverage", {}),
        "modified_statement_coverage": validation.get("modified_statement_coverage", []),
        "acceptance_gates": {
            "complete_path_traversal": {
                "condition": "Visit every recursive results/**/*.json path to EOF in bounded chunks with overlap >= max_needle_len - 1 bytes and zero path/size/age/verdict filtering",
                "principle": "Every original file remains in scope, including raw and nested results.",
                "passed": int(scan["visited_files"]) == int(scan["total_files"])
                and int(scan["unfinished_files"]) == 0,
            },
            "exact_semantic_parity": {
                "condition": "Zero classification discrepancies across corpus candidates and all 31 parity fixtures, and 6/6 sentinel+unicode mutation controls detected",
                "principle": "Parsing less must not change classification outcomes.",
                "passed": bool(scan["semantic_parity_exact"])
                and bool(ctrl["semantic_parity_exact"])
                and bool(ctrl["all_mutations_detected"]),
            },
            "stable_candidate_bytes": {
                "condition": "Zero concurrent candidate file mutations during traversal (changed_files == 0)",
                "principle": "Concurrent corpus mutation cannot certify completeness.",
                "passed": bool(scan["candidate_bytes_stable"]),
            },
            "all_required_checks_and_coverage": {
                "condition": "Target test module, owned tests, V782 controls, affected consumers, E2E-018/021, Ruff, strict mypy, spec coverage and 100% owned statement coverage pass",
                "principle": "Self-reported passing checks do not establish readiness.",
                "passed": bool(not blocked and checks_ok),
            },
        },
        "protocol_references": inspected["protocol_references"],
        "sentinel_derivation": inspected["sentinel_info"],
        "exp9181_timeout_diagnostics": inspected["exp9181_timeout_diagnostics"],
        "corpus_manifest_reference": {
            "scope": "results/**/*.json",
            "total_files": scan["total_files"],
            "visited_files": scan["visited_files"],
            "unfinished_files": scan["unfinished_files"],
            "changed_files": scan["changed_files"],
            "candidate_bytes_stable": scan["candidate_bytes_stable"],
            "total_size_bytes": scan["total_size_bytes"],
            "total_bytes_read": scan["total_bytes_read"],
            "population_sha256": scan["population_sha256"],
            "decoded_count": scan["decoded_count"],
            "safely_negative_eof_count": scan["safely_negative_eof_count"],
            "malformed_count": scan["malformed_count"],
            "nondict_count": scan["nondict_count"],
            "checked": scan["checked"],
            "offenders": scan["offenders"],
            "stratum_summary": scan["stratum_summary"],
            "ordered_paths": [
                {
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "stratum": item["stratum"],
                    "sha256": item.get("sha256"),
                    "stat_identity": item.get("stat_identity"),
                }
                for item in scan["manifest_rows"]
            ],
        },
        "scan_rows": scan["scan_rows"],
        "candidate_counts": {
            "visited": scan["visited_files"],
            "decoded": scan["decoded_count"],
            "safely_negative": scan["safely_negative_eof_count"],
            "malformed": scan["malformed_count"],
            "changed": scan["changed_files"],
            "unfinished": scan["unfinished_files"],
        },
        "parity_controls": ctrl,
        "cost_model_rows": costs.get("unit_rows", []),
        "stratum_cost_comparison": costs,
        "parity_and_mutation_controls": ctrl,
        "work_reference": p.file_ref(work_path, None),
    }
    if not ready:
        art["claim_scope"]["allowed_claims"] = [
            "Recorded failed qualification operands and completed controls only."
        ]
    art["field_principles"] = {
        key: (
            "Bound by authenticated Exp9255/Exp9181/Exp9241/Exp9242 provenance, lossless "
            "five-sentinel plus JSON Unicode-escape byte prefilter traversal, and exact "
            "semantic parity controls; zero current LLM work."
        )
        for key in art
    }
    art["reproducibility_checksum"] = checksum(art)
    return art


def _resolve_ref_path(ref_path: str, base_root: Path) -> Path:
    cand = Path(ref_path)
    return cand if cand.is_absolute() else (base_root / cand)


def replay(path: Path, *, root: Path | None = None) -> Json:
    value = json.loads(path.read_text("utf-8"))
    if value.get("reproducibility_checksum") != checksum(value):
        raise ValueError("checksum_mismatch")
    wref = value["work_reference"]
    base_root = root or p.ROOT
    work_path = _resolve_ref_path(str(wref["path"]), base_root)
    work = json.loads(work_path.read_text("utf-8"))
    work_root = Path(work["root"])
    if p.file_ref(work_path, None) != wref:
        raise ValueError("work_reference_hash_mismatch")
    cw_ref = work["current_work_reference"]
    cw_path = _resolve_ref_path(str(cw_ref["path"]), work_root)
    if p.file_ref(cw_path, None) != cw_ref:
        raise ValueError("current_work_hash_mismatch")
    if validate_current_work_receipt(json.loads(cw_path.read_text("utf-8")), []):
        raise ValueError("current_work_receipt_invalid")
    validation = {
        "checks": value["validation_receipts"],
        "owned_statement_coverage": value["owned_statement_coverage"],
        "modified_statement_coverage": value["modified_statement_coverage"],
    }
    rebuilt = build(work_root, Path(work["raw"]), work, validation, str(value["run_date"]))
    if (
        rebuilt["reproducibility_checksum"] != value["reproducibility_checksum"]
        or rebuilt["corpus_scan_ready_score"] != value["corpus_scan_ready_score"]
        or rebuilt["honest_verdict"] != value["honest_verdict"]
    ):
        raise ValueError("replay_reduction_drift")
    return {"passed": True, "replay_passed": True, "path": str(path)}


def terminal(candidate: Path, logs: Path, *, stop: float | None = None) -> Json:
    logs.mkdir(parents=True, exist_ok=True)
    py_bin = str(p.ROOT / ".venv/bin/python")
    script = str(p.ROOT / "scripts/experiments/experiment_9256_v788_corpus_prefilter.py")
    deadline = time.monotonic() + p.PUBLICATION_S if stop is None else stop
    adv = run(
        "adversarial_verify",
        [py_bin, str(p.ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)],
        logs,
        deadline,
    )
    lint = run(
        "verdict_row_consistency",
        [
            py_bin,
            str(p.ROOT / "scripts/verdict_row_consistency_lint.py"),
            "--strict",
            str(candidate),
        ],
        logs,
        deadline,
    )
    rep = run(
        "cold_replay",
        [
            "env",
            "--chdir=/var/tmp",
            "--unset=PYTHONPATH",
            py_bin,
            script,
            "--date",
            p.RUN_DATE,
            "--cold-replay",
            str(candidate),
        ],
        logs,
        deadline,
    )
    return {
        "passed": bool(adv.get("passed") and lint.get("passed") and rep.get("passed")),
        "adversarial_verify": adv,
        "verdict_row_consistency": lint,
        "cold_replay": rep,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp9256 V788 lossless corpus byte prefilter")
    parser.add_argument("--date", default=p.RUN_DATE)
    parser.add_argument("--root", type=Path, default=p.ROOT)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--from-work", action="store_true")
    parser.add_argument("--cold-replay", nargs="?", const="USE_OUTPUT", default=None)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    p.log("start_no_model_load")
    try:
        if args.date != p.RUN_DATE:
            raise ValueError(f"invalid_run_date:{args.date}")
        root = args.root.resolve()
        out_path = (args.output or (root / p.OUT)).resolve()
        raw_dir = (args.raw_dir or (root / p.RAW)).resolve()

        if args.cold_replay is not None:
            target = (
                out_path if args.cold_replay == "USE_OUTPUT" else Path(args.cold_replay).resolve()
            )
            rep = replay(target, root=root)
            print(json.dumps(rep), flush=True)
            return 0

        if args.from_work and out_path.is_relative_to(p.ROOT.resolve()):
            raise ValueError("private_work_requires_external_output")

        if args.raw_dir is None:
            raw_dir = raw_dir / "invocations" / str(time.time_ns())
        scope = build_scope(root, raw_dir)
        stop = t0 + p.BUDGET_S
        measurement_stop = t0 + p.MEASUREMENT_S
        verification_stop = stop - p.PUBLICATION_S
        if args.from_work:
            work = json.loads((raw_dir / "work.json").read_bytes())
            validation_path = raw_dir / "validation.json"
            validation = (
                json.loads(validation_path.read_bytes()) if validation_path.exists() else {}
            )
        else:
            p.log("authenticate_inputs_before_measurement")
            inspected = p.inspect(root, raw_dir, authenticate_only=True)
            gates_ok = all(row["passed"] for row in inspected["gate_checks"]) and all(
                row["available"] for row in inspected["preconditions"]
            )
            if args.checks_only:
                print(json.dumps({"gates_passed": gates_ok}), flush=True)
                return 0 if gates_ok else 1
            validation = {}
            if gates_ok:
                os.environ["CARNOT_9256_DEADLINE"] = str(measurement_stop)
                target = execute_validation(raw_dir, scope, measurement_stop, target_only=True)
                scan_path = raw_dir / "corpus_scan_summary.json"
                scan = json.loads(scan_path.read_bytes()) if scan_path.exists() else None
                # A killed target has no complete scan receipt and cannot grant readiness.
                inspected = p.inspect(
                    root, raw_dir, authenticate_only=scan is None, scan_summary=scan
                )
                os.environ.pop("CARNOT_9256_DEADLINE", None)
                validation = execute_validation(raw_dir, scope, verification_stop)
                validation["checks"] = target["checks"] + validation["checks"]
                atomic_json(raw_dir / "validation.json", validation)
            work = measure(
                root,
                raw_dir,
                args.date,
                scope=scope,
                inspection=inspected,
                started_monotonic_ns=int(t0 * 1e9),
            )
        art = build(root, raw_dir, work, validation, args.date)
        cand = raw_dir / "candidate.json"
        atomic_json(cand, art)
        if time.monotonic() >= stop:
            raise TimeoutError("total_budget_exceeded")

        p.log("publish_primary_start")
        pub = publish_primary(
            out_path, art, lambda cpath: terminal(cpath, raw_dir / "terminal", stop=stop)
        )
        atomic_json(raw_dir / "publication.json", pub)
        p.log(f"publish_primary_done honest_verdict={art['honest_verdict']}")
        print(
            json.dumps({"published": str(out_path), "honest_verdict": art["honest_verdict"]}),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as exc:
        p.log(f"refused:{exc}")
        return 1
