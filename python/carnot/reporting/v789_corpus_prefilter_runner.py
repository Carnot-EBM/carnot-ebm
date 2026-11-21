from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import verify_measured
from carnot.reporting.validation_diagnostics_9181 import run

Json = dict[str, Any]
_DIGEST_CACHE: dict[tuple[str, int, int, int, int], str] = {}
_COMPACT_CACHE: dict[str, Json] = {}
_TRIM_PAIRS: dict[bytes, tuple[bytes, bytes]] = {
    b'  "scan_rows": [\n': (b"  ],\n", b'  "scan_rows": [],\n'),
    b'    "ordered_paths": [\n': (b"    ],\n", b'    "ordered_paths": [],\n'),
    b'  "historical_input_roots": [\n': (b"  ],\n", b'  "historical_input_roots": [],\n'),
    b'  "validation_obligations": {\n': (b"  },\n", b'  "validation_obligations": {},\n'),
    b'  "validation_scope_manifest": {\n': (b"  },\n", b'  "validation_scope_manifest": {},\n'),
}


def _digest(path: Path) -> str:
    resolved = path.resolve()
    st = resolved.stat()
    key = (str(resolved), st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    if key not in _DIGEST_CACHE:
        _DIGEST_CACHE[key] = sha256_file(resolved)
    return _DIGEST_CACHE[key]


def load_compact_primary(path: Path) -> Json:
    digest = _digest(path)
    if digest not in _COMPACT_CACHE:
        out = bytearray()
        skip_until: bytes | None = None
        with path.open("rb") as stream:
            for line in stream:
                if skip_until is not None:
                    if line == skip_until:
                        skip_until = None
                    continue
                hit = _TRIM_PAIRS.get(line)
                if hit is not None:
                    skip_until, replacement = hit
                    out.extend(replacement)
                else:
                    out.extend(line)
        _COMPACT_CACHE[digest] = json.loads(out)
    return dict(json.loads(json.dumps(_COMPACT_CACHE[digest])))


RUN_DATE = "20261121"
MILESTONE = "2026.11.789"
TASK = "exp9270-corpus-prefilter"
EXP_ID = 9270
OUT = "results/experiment_9270_v789_corpus_prefilter.json"
RAW = "results/raw/experiment_9270_v789_corpus_prefilter"
SEED = 92702026
MAX_PRIMARY_BYTES = 10_485_760

OWNED = (
    "python/carnot/reporting/v788_corpus_prefilter.py",
    "python/carnot/reporting/v789_corpus_prefilter_runner.py",
    "scripts/experiments/experiment_9270_v789_corpus_prefilter.py",
)
OWNED_TESTS = (
    "tests/python/test_v789_corpus_prefilter_9270.py",
    "tests/python/test_v788_corpus_prefilter_9256.py",
    "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py",
)
MODIFIED = (
    "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py",
    "python/carnot/reporting/causal_validation_9167.py",
    "python/carnot/reporting/v782_contract_evidence.py",
)
UPSTREAM_PINS = {
    "exp9269-contract": (
        "results/experiment_9269_v789_contract.json",
        "sha256:0d0aa11d171cf91085491cf59b6add32b221068d487e110e5b0c1a3009ca4e4e",
    ),
    "exp9256-v788-corpus-prefilter": (
        "results/experiment_9256_v788_corpus_prefilter.json",
        "sha256:d1ac40d315f6a740c5cb30d853045f6c92ed51d2cebeef14e09b7b945f17481a",
    ),
    "exp9255-v788-contract": (
        "results/experiment_9255_v788_contract.json",
        "sha256:7c669d5df1331eb5d22211d22bca58db4959b76af6abcaa814ca5de5a5ea24ef",
    ),
    "exp9181-causal-validation": (
        "results/experiment_9181_v782_causal_validation.json",
        "sha256:dbdb988f3ce372334dc1634309556987d26f05c73739eeae574624ea601d2934",
    ),
    "exp9241-v787-contract": (
        "results/experiment_9241_v787_contract.json",
        "sha256:32cdfef7df5fb0b829ef9eba8d68a44a0e6b7947afbd402d24129dae27a92f5b",
    ),
    "exp9242-v787-corpus-prefilter": (
        "results/experiment_9242_corpus_prefilter.json",
        "sha256:9e5cf741802b6c1202c6bbbb5ff3b008929d2fbd39b079ff64a00a2d50e63063",
    ),
}
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
)


def log(msg: str) -> None:
    print(f"[exp9270] {msg}", flush=True)


def checksum(value: Json) -> str:
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def compact_guard(value: Json) -> None:
    """Reject oversized primary bytes so primary JSON stays strictly below 10 MB."""
    if (
        len((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        >= MAX_PRIMARY_BYTES
    ):
        raise ValueError("primary_size_exceeded")


def _stream_file_ref(path: Path) -> Json:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_scan_sidecars(raw: Path, scan: Json) -> Json:
    """Write full ordered_paths and scan_rows to hashed sidecar files under raw."""
    raw.mkdir(parents=True, exist_ok=True)
    compact = dict(scan)
    if (
        "manifest_sidecar_reference" in compact
        and "scan_rows_sidecar_reference" in compact
        and "manifest_rows" not in compact
        and "scan_rows" not in compact
    ):
        for key in ("manifest_sidecar_reference", "scan_rows_sidecar_reference"):
            ref = compact[key]
            if _stream_file_ref(Path(ref["path"])) != {
                name: ref[name] for name in ("path", "sha256", "size_bytes")
            }:
                raise ValueError("scan_sidecar_changed")
        return compact

    manifest_rows = list(compact.pop("manifest_rows", []))
    scan_rows = list(compact.pop("scan_rows", []))
    manifest_path = (raw / "corpus_manifest.json").resolve()
    manifest_payload = {
        "scope": "results/**/*.json",
        "total_files": compact.get("total_files", 0),
        "visited_files": compact.get("visited_files", 0),
        "unfinished_files": compact.get("unfinished_files", 0),
        "changed_files": compact.get("changed_files", 0),
        "candidate_bytes_stable": compact.get("candidate_bytes_stable", False),
        "population_sha256": compact.get("population_sha256", p.digest_bytes(b"")),
        "ordered_paths": [
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "stratum": item["stratum"],
                "sha256": item.get("sha256"),
                "stat_identity": item.get("stat_identity"),
            }
            for item in manifest_rows
        ],
    }
    manifest_bytes = (
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with manifest_path.open("wb") as m_handle:
        m_handle.write(manifest_bytes)
        m_handle.flush()
        os.fsync(m_handle.fileno())
    manifest_ref = {
        "path": manifest_path.as_posix(),
        "sha256": p.digest_bytes(manifest_bytes),
        "size_bytes": len(manifest_bytes),
        "row_count": len(manifest_rows),
    }

    scan_rows_path = (raw / "scan_rows.jsonl").resolve()
    scan_hasher = hashlib.sha256()
    scan_size = 0
    with scan_rows_path.open("wb") as s_handle:
        for row in scan_rows:
            line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            s_handle.write(line)
            scan_hasher.update(line)
            scan_size += len(line)
        s_handle.flush()
        os.fsync(s_handle.fileno())
    scan_rows_ref = {
        "path": scan_rows_path.as_posix(),
        "sha256": f"sha256:{scan_hasher.hexdigest()}",
        "size_bytes": scan_size,
        "format": "jsonl",
        "row_count": len(scan_rows),
        "visited": compact.get("visited_files", 0),
        "decoded": compact.get("decoded_count", 0),
        "safely_negative": compact.get("safely_negative_eof_count", 0),
        "malformed": compact.get("malformed_count", 0),
        "nondict": compact.get("nondict_count", 0),
        "checked": compact.get("checked", 0),
        "offender_count": len(compact.get("offenders", [])),
        "changed": compact.get("changed_files", 0),
        "unfinished": compact.get("unfinished_files", 0),
        "parity_discrepancies": compact.get("parity_discrepancies", 0),
        "semantic_parity_exact": compact.get("semantic_parity_exact", False),
    }
    compact["manifest_sidecar_reference"] = manifest_ref
    compact["scan_rows_sidecar_reference"] = scan_rows_ref
    return compact


def inspect_v789(
    root: Path,
    raw: Path,
    *,
    authenticate_only: bool = False,
    scan_summary: Json | None = None,
) -> Json:
    raw.mkdir(parents=True, exist_ok=True)
    av_mod = p.load_av()
    mem_text = Path("/proc/meminfo").read_text("utf-8")
    mem_kb = next(
        int(line.split()[1]) for line in mem_text.splitlines() if line.startswith("MemAvailable:")
    )
    disk_free = shutil.disk_usage(raw).free
    preconditions = [
        {
            "resource": "cpu_cores",
            "available": (os.cpu_count() or 1) >= 1,
            "observed": os.cpu_count() or 1,
        },
        {
            "resource": "mem_available_mb",
            "available": mem_kb // 1024 >= 512,
            "observed": mem_kb // 1024,
        },
        {
            "resource": "disk_free_mb",
            "available": disk_free // (1024 * 1024) >= 512,
            "observed": disk_free // (1024 * 1024),
        },
        {
            "resource": "required_tools",
            "available": (root / "scripts/adversarial_verify.py").exists()
            and (root / "scripts/verdict_row_consistency_lint.py").exists()
            and all(
                os.access(p.ROOT / ".venv/bin" / name, os.X_OK)
                for name in ("python", "pytest", "ruff", "mypy")
            ),
            "observed": {
                name: str(p.ROOT / ".venv/bin" / name)
                for name in ("python", "pytest", "ruff", "mypy")
            },
        },
    ]

    loaded: dict[str, Json | None] = {}
    source_refs: list[Json] = []
    gate_checks: list[Json] = []
    for upstream, (rel, exp_sha) in UPSTREAM_PINS.items():
        p_file = root / rel
        if p_file.exists():
            sha_obs = _digest(p_file)
            loaded[upstream] = load_compact_primary(p_file)
            source_refs.append(
                {
                    "upstream": upstream,
                    "path": rel,
                    "sha256": sha_obs,
                    "expected_sha256": exp_sha,
                    "size_bytes": p_file.stat().st_size,
                    "imported_fields": [
                        "honest_verdict",
                        "verdict_class",
                        "contract_ready_score",
                        "replay_authority_ready_score",
                        "corpus_scan_ready_score",
                        "causal_fix_ready_score",
                        "candidate_counts",
                        "validation_receipts",
                        "protocol_references",
                    ],
                }
            )
        else:
            sha_obs = "missing_file"
            loaded[upstream] = None
        gate_checks.append(
            {
                "check": f"{upstream.split('-')[0]}_sha256",
                "upstream": upstream,
                "path": rel,
                "sha256": None if sha_obs == "missing_file" else sha_obs,
                "artifact_field": None,
                "op": "sha256==",
                "expected": exp_sha,
                "observed": sha_obs,
                "passed": sha_obs == exp_sha,
            }
        )

    c9269 = loaded["exp9269-contract"]
    sha9269 = gate_checks[0]["sha256"]
    for field_name, exp_val in (
        ("contract_ready_score", 1),
        ("replay_authority_ready_score", 1),
        ("required_checks_passed", True),
    ):
        gate_checks.append(
            p._gate_field(
                c9269,
                check=f"exp9269_{field_name}",
                upstream="exp9269-contract",
                rel_path=UPSTREAM_PINS["exp9269-contract"][0],
                sha256=sha9269,
                field=field_name,
                expected=exp_val,
            )
        )

    c9256 = loaded["exp9256-v788-corpus-prefilter"]
    sha9256 = gate_checks[1]["sha256"]
    for field_name, exp_val in (
        ("honest_verdict", "complete_disqualified_v788_corpus_prefilter"),
        ("corpus_scan_ready_score", 0),
    ):
        gate_checks.append(
            p._gate_field(
                c9256,
                check=f"exp9256_{field_name}",
                upstream="exp9256-v788-corpus-prefilter",
                rel_path=UPSTREAM_PINS["exp9256-v788-corpus-prefilter"][0],
                sha256=sha9256,
                field=field_name,
                expected=exp_val,
            )
        )
    c9256_counts = (c9256 or {}).get("candidate_counts") if c9256 is not None else None
    gate_checks.append(
        p._gate_field(
            c9256_counts,
            check="exp9256_visited_files",
            upstream="exp9256-v788-corpus-prefilter",
            rel_path=UPSTREAM_PINS["exp9256-v788-corpus-prefilter"][0],
            sha256=sha9256,
            field="visited",
            expected=764150,
        )
    )

    c9255 = loaded["exp9255-v788-contract"]
    sha9255 = gate_checks[2]["sha256"]
    for field_name, exp_val in (
        ("contract_ready_score", 1),
        ("replay_authority_ready_score", 1),
    ):
        gate_checks.append(
            p._gate_field(
                c9255,
                check=f"exp9255_{field_name}",
                upstream="exp9255-v788-contract",
                rel_path=UPSTREAM_PINS["exp9255-v788-contract"][0],
                sha256=sha9255,
                field=field_name,
                expected=exp_val,
            )
        )

    c9181 = loaded["exp9181-causal-validation"]
    sha9181 = gate_checks[3]["sha256"]
    gate_checks.append(
        p._gate_field(
            c9181,
            check="exp9181_causal_fix_ready_score",
            upstream="exp9181-causal-validation",
            rel_path=UPSTREAM_PINS["exp9181-causal-validation"][0],
            sha256=sha9181,
            field="causal_fix_ready_score",
            expected=0,
        )
    )
    gate_checks.append(
        p._gate_field(
            loaded["exp9241-v787-contract"],
            check="exp9241_honest_verdict",
            upstream="exp9241-v787-contract",
            rel_path=UPSTREAM_PINS["exp9241-v787-contract"][0],
            sha256=gate_checks[4]["sha256"],
            field="honest_verdict",
            expected="complete_blocked_v787_contract",
        )
    )
    gate_checks.append(
        p._gate_field(
            loaded["exp9242-v787-corpus-prefilter"],
            check="exp9242_honest_verdict",
            upstream="exp9242-v787-corpus-prefilter",
            rel_path=UPSTREAM_PINS["exp9242-v787-corpus-prefilter"][0],
            sha256=gate_checks[5]["sha256"],
            field="honest_verdict",
            expected="blocked_gate_check_failed",
        )
    )

    historical = next(
        (
            row
            for row in (c9181 or {}).get("validation_receipts", [])
            if row.get("name") == "direct_consumers"
        ),
        {},
    )
    for field, expected in (("exit_code", -15), ("duration_s", 1224.3159505779913)):
        gate_checks.append(
            p._gate_field(
                historical,
                check=f"exp9181_direct_consumers_{field}",
                upstream="exp9181-causal-validation",
                rel_path=UPSTREAM_PINS["exp9181-causal-validation"][0],
                sha256=sha9181,
                field=field,
                expected=expected,
            )
        )
    for key in ("receipt_reference", "events_reference"):
        reference = historical.get(key, {})
        path = Path(reference.get("path", "/missing-9270"))
        observed = p.file_ref(path, None)["sha256"] if path.is_file() else "missing_file"
        gate_checks.append(
            {
                "check": f"exp9181_{key}",
                "upstream": "exp9181-causal-validation",
                "path": str(path),
                "sha256": observed,
                "artifact_field": "sha256",
                "op": "==",
                "expected": reference.get("sha256", "missing_field"),
                "observed": observed,
                "passed": observed == reference.get("sha256"),
            }
        )

    results_dir = root / "results"
    gates_passed = all(row["passed"] for row in gate_checks) and all(
        row["available"] for row in preconditions
    )
    if gates_passed and not authenticate_only and results_dir.exists():
        raw_scan = scan_summary if scan_summary is not None else p.scan_corpus(results_dir, av_mod)
    else:
        raw_scan = {
            "results_dir": str(results_dir),
            "total_files": 0,
            "visited_files": 0,
            "unfinished_files": 0,
            "changed_files": 0,
            "candidate_bytes_stable": False,
            "added_paths": [],
            "removed_paths": [],
            "total_size_bytes": 0,
            "total_bytes_read": 0,
            "population_sha256": p.digest_bytes(b""),
            "decoded_count": 0,
            "safely_negative_eof_count": 0,
            "malformed_count": 0,
            "nondict_count": 0,
            "checked": 0,
            "offenders": [],
            "parity_discrepancies": 0,
            "semantic_parity_exact": False,
            "duration_s": 0.0,
            "stratum_summary": [],
            "manifest_rows": [],
            "scan_rows": [],
        }
    controls = (
        p.run_parity_and_mutation_controls(av_mod)
        if gates_passed and not authenticate_only
        else {
            "fixture_count": 0,
            "parity_discrepancies": 0,
            "semantic_parity_exact": False,
            "parity_rows": [],
            "mutation_controls": [],
            "all_mutations_detected": False,
        }
    )
    stratum_costs = (
        p.measure_stratum_costs(root, raw_scan.get("manifest_rows", []), av_mod)
        if gates_passed and not authenticate_only
        else {"full_corpus_speedup_claimed": False, "unit_rows": [], "strata": []}
    )
    corpus_scan = write_scan_sidecars(raw, raw_scan)
    proto_refs = c9269["protocol_references"] if c9269 and "protocol_references" in c9269 else {}
    code_hashes = [
        p.file_ref(p.ROOT / rel, p.ROOT)
        for rel in (
            *OWNED,
            *OWNED_TESTS,
            *MODIFIED,
            "scripts/adversarial_verify.py",
            "pyproject.toml",
        )
        if (p.ROOT / rel).exists()
    ]
    return {
        "preconditions": preconditions,
        "gate_checks": gate_checks,
        "source_references": source_refs,
        "protocol_references": proto_refs,
        "code_config_hashes": code_hashes,
        "sentinel_info": {
            "constants": [
                {"name": name, "value": text, "byte_length": len(raw_b)}
                for name, text, raw_b in p.derive_sentinels(av_mod)["sentinels"]
            ],
            "max_needle_len": 73,
            "min_overlap_bytes": 72,
        },
        "exp9181_timeout_diagnostics": {
            "upstream_path": UPSTREAM_PINS["exp9181-causal-validation"][0],
            "direct_consumers_exit_code": historical.get("exit_code"),
            "direct_consumers_duration_s": historical.get("duration_s"),
            "receipt_reference": historical.get("receipt_reference"),
            "events_reference": historical.get("events_reference"),
            "unfinished_nodeid": (
                "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py::"
                "test_no_artifact_on_disk_is_newly_flagged_by_closing_this_hole"
            ),
            "baseline_target_test_sha256": p.BASELINE_TARGET_SHA256,
        },
        "exp9256_historical_summary": {
            "upstream_path": UPSTREAM_PINS["exp9256-v788-corpus-prefilter"][0],
            "sha256": sha9256,
            "honest_verdict": (c9256 or {}).get("honest_verdict"),
            "corpus_scan_ready_score": (c9256 or {}).get("corpus_scan_ready_score"),
            "visited_files": (c9256_counts or {}).get("visited"),
            "disqualification_reason": "spurious_full_python_suite_600s_timeout_and_875mb_inline_json",
        },
        "corpus_scan": corpus_scan,
        "controls": controls,
        "stratum_costs": stratum_costs,
    }


def build_scope(root: Path, raw: Path) -> Json:
    raw.mkdir(parents=True, exist_ok=True)
    cov_ini = raw / "coverage.ini"
    inc_lines = "\n".join(f"    {p.ROOT / rel}" for rel in (*OWNED, *MODIFIED))
    cov_ini.write_text(
        f"[run]\nbranch = False\nparallel = True\npatch = subprocess\ninclude =\n{inc_lines}\n[report]\nexclude_lines =\n",
        encoding="utf-8",
    )
    priv_root = Path(tempfile.mkdtemp(prefix=f"carnot-9270-{raw.name}-", dir="/var/tmp"))
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
        chdir_root: bool = False,
    ) -> Json:
        pdir = p.ROOT if chdir_root else (priv_root / name)
        if not chdir_root:
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
                *(["--unset=COVERAGE_PROCESS_START"] if not with_cov else []),
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
            with_cov=False,
        ),
        _priv_pytest(
            "owned_tests",
            (
                "tests/python/test_v789_corpus_prefilter_9270.py",
                "tests/python/test_v788_corpus_prefilter_9256.py",
            ),
            [],
            600,
            with_cov=True,
        ),
        _priv_pytest(
            "v782_controls",
            V782_CONTROL_TESTS,
            [],
            600,
            with_cov=True,
            chdir_root=True,
        ),
        _priv_pytest("direct_consumers", CONSUMER_TESTS, [], 600),
        _priv_pytest("e2e018_021", E2E_TESTS, [], 600),
        {
            "name": "ruff_check",
            "deadline_s": 300,
            "argv": [ruff_bin, "check", *(str(p.ROOT / f) for f in (*OWNED, *OWNED_TESTS))],
        },
        {
            "name": "ruff_format",
            "deadline_s": 300,
            "argv": [
                ruff_bin,
                "format",
                "--check",
                *(str(p.ROOT / f) for f in (*OWNED, *OWNED_TESTS)),
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
                *(str(p.ROOT / f) for f in OWNED),
            ],
        },
        {
            "name": "spec_coverage",
            "deadline_s": 300,
            "argv": [
                py_bin,
                str(p.ROOT / "scripts/check_spec_coverage.py"),
                *(str(p.ROOT / f) for f in OWNED_TESTS),
            ],
        },
    ]
    scope = {
        "owned_changed_paths": list(OWNED),
        "modified_paths": list(MODIFIED),
        "owned_test_paths": list(OWNED_TESTS),
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
        old = (base / source.name).read_text("utf-8").splitlines()
        new = source.read_text("utf-8").splitlines()
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
        log(f"validation_cmd_start name={cmd['name']}")
        rec = run(
            cmd["name"],
            cmd["argv"],
            raw / "validation",
            min(stop, time.monotonic() + cmd["deadline_s"]),
        )
        rec["scope_sha256"] = canonical_hash(scope)
        checks.append(rec)
        log(
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
        receipt_path = Path(reference.get("path", "/missing-9270"))
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
    inspected = inspect_v789(root, raw) if inspection is None else inspection
    ended_ns = time.monotonic_ns()
    cw_receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="read_only_live_artifact_inventory",
        inference_substrate_details={
            "work": (
                "Authenticate Exp9269 V789 contract, Exp9256 V788 corpus prefilter, Exp9255 V788 "
                "contract, Exp9181 direct_consumers timeout receipt, and Exp9241/9242 V787 history; "
                "qualify scoped lossless five-sentinel byte prefilter over complete recursive "
                "results/**/*.json manifest with hashed sidecars and nine affected checks; "
                "zero LLM loads, forwards or generations."
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
                "seed": SEED,
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
                "seed": SEED,
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
                "seed": SEED,
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

    scan = write_scan_sidecars(raw, inspected["corpus_scan"])
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
    honest = f"complete_{verdict_cls}_v789_corpus_prefilter"
    rows = _build_rows(inspected, blocked)
    total_units = len(rows)
    comp_units = 0 if blocked else total_units
    cens_units = total_units if blocked else 0

    cw_path = raw / "current_work.json"
    work_path = raw / "work.json"
    manifest_ref = dict(scan["manifest_sidecar_reference"])
    scan_rows_ref = dict(scan["scan_rows_sidecar_reference"])

    art: Json = {
        "schema": "carnot.v789_corpus_prefilter.v1",
        "experiment_id": EXP_ID,
        "task_id": TASK,
        "milestone": MILESTONE,
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
                "scoped lossless five-sentinel plus JSON Unicode-escape byte prefilter over recursive results/**/*.json with hashed sidecars",
                "exact semantic parity against unchanged read_text/json.loads and _is_arc_live_agent_no_llm classifier",
                "predetermined size-stratum read/decode/classifier cost comparison without full-corpus speedup ratio claim from censored Exp9181 baseline",
            ],
        },
        "MODEL_SPECS": [],
        "model_invocation_status": "not_requested_no_model_load",
        "requested_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run" if blocked else "no_model_load",
        "inference_substrate": "read_only_live_artifact_inventory",
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
        "random_seed": SEED,
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
                "condition": "Zero classification discrepancies across corpus candidates and all frozen parity fixtures, and 6/6 sentinel+unicode mutation controls detected",
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
                "condition": "All nine affected checks (target_module, owned_tests, v782_controls, direct_consumers, e2e018_021, ruff_check, ruff_format, strict_mypy, spec_coverage) and 100% owned+modified statement coverage pass",
                "principle": "Self-reported passing checks do not establish readiness.",
                "passed": bool(not blocked and checks_ok),
            },
            "compact_hashed_sidecars": {
                "condition": "Primary JSON is strictly below 10 MB; full ordered_paths and scan_rows live in authenticated hashed sidecar files",
                "principle": "Every original file remains in scope and verifiable without 875 MB primary artifacts.",
                "passed": True,
            },
        },
        "protocol_references": inspected["protocol_references"],
        "sentinel_derivation": inspected["sentinel_info"],
        "exp9181_timeout_diagnostics": inspected["exp9181_timeout_diagnostics"],
        "exp9256_historical_summary": inspected["exp9256_historical_summary"],
        "corpus_manifest_reference": {
            **manifest_ref,
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
        },
        "scan_rows_sidecar_reference": scan_rows_ref,
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
            "Bound by authenticated Exp9269/Exp9256/Exp9255/Exp9181/Exp9241/Exp9242 provenance, "
            "lossless five-sentinel plus JSON Unicode-escape byte prefilter traversal, hashed "
            "sidecars, and exact semantic parity controls; zero current LLM work."
        )
        for key in art
    }
    art["reproducibility_checksum"] = checksum(art)
    compact_guard(art)
    return art


def _resolve_ref_path(ref_path: str, base_root: Path) -> Path:
    cand = Path(ref_path)
    return cand if cand.is_absolute() else (base_root / cand)


def replay(path: Path, *, root: Path | None = None) -> Json:
    if path.stat().st_size >= MAX_PRIMARY_BYTES:
        raise ValueError("primary_size_exceeded")
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

    m_ref = value["corpus_manifest_reference"]
    m_path = _resolve_ref_path(str(m_ref["path"]), work_root)
    if not m_path.is_file() or _stream_file_ref(m_path)["sha256"] != m_ref.get("sha256"):
        raise ValueError("manifest_sidecar_hash_mismatch")
    s_ref = value["scan_rows_sidecar_reference"]
    s_path = _resolve_ref_path(str(s_ref["path"]), work_root)
    if not s_path.is_file() or _stream_file_ref(s_path)["sha256"] != s_ref.get("sha256"):
        raise ValueError("scan_rows_sidecar_hash_mismatch")

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
    script = str(p.ROOT / "scripts/experiments/experiment_9270_v789_corpus_prefilter.py")
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
            RUN_DATE,
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
    parser = argparse.ArgumentParser(
        description="Exp9270 V789 scoped lossless corpus byte prefilter"
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=p.ROOT)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--from-work", action="store_true")
    parser.add_argument("--cold-replay", nargs="?", const="USE_OUTPUT", default=None)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    log("start_no_model_load")
    try:
        if args.date != RUN_DATE:
            raise ValueError(f"invalid_run_date:{args.date}")
        root = args.root.resolve()
        out_path = (args.output or (root / OUT)).resolve()
        raw_dir = (args.raw_dir or (root / RAW)).resolve()

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
            log("authenticate_inputs_before_measurement")
            inspected = inspect_v789(root, raw_dir, authenticate_only=True)
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
                scan_path.unlink(missing_ok=True)
                inspected = inspect_v789(
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

        log("publish_primary_start")
        pub = publish_primary(
            out_path, art, lambda cpath: terminal(cpath, raw_dir / "terminal", stop=stop)
        )
        atomic_json(raw_dir / "publication.json", pub)
        log(f"publish_primary_done honest_verdict={art['honest_verdict']}")
        print(
            json.dumps({"published": str(out_path), "honest_verdict": art["honest_verdict"]}),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as exc:
        log(f"refused:{exc}")
        return 1
