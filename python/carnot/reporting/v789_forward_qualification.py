"""REQ-REPORT-9271-V789 complete causally corrected validation revision seam for Exp9271."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import psutil

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.historical_contract_inputs import _digest, load_compact_primary
from carnot.reporting.primary_publication import publish_primary

ROOT = Path(__file__).resolve().parents[3]
EXP_ID = "experiment_9271_v789_forward_qualification"
TASK_ID = "exp9271-forward-qualification"
DATE = "20261121"
SEED = 9271
MAX_PRIMARY_BYTES = 10_485_760
OUT = ROOT / "results" / f"{EXP_ID}.json"
RAW = ROOT / "results" / "raw" / EXP_ID
SCRIPT = ROOT / "scripts" / "experiments" / "experiment_9271_v789_forward_qualification.py"
BUDGET = {
    "measurement_deadline_s": 3300,
    "verification_reserve_s": 1200,
    "publication_reserve_s": 300,
    "hard_cap_s": 4800,
}
OWNED = (
    "python/carnot/reporting/v789_forward_qualification.py",
    "scripts/experiments/experiment_9271_v789_forward_qualification.py",
)
TESTS = ("tests/python/test_v789_forward_qualification_9271.py",)
CONFIGS = (
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
)
OBLIGATION_SIDECAR_REL = "results/raw/experiment_9269_v789_contract/invocations/1795233294580319646/validation_obligations.json"
OBLIGATION_SIDECAR_SHA = "sha256:7a97383e5907b8225927cf63e1249fcebf5119e0529c99691f317a6eb005097e"
UPSTREAM_PINS: dict[str, tuple[str, str]] = {
    "exp9270-corpus-prefilter": (
        "results/experiment_9270_v789_corpus_prefilter.json",
        "sha256:2c13ec623618640f04468164f27cb315fca3aba06c8d2ad888919e19130e6831",
    ),
    "exp9269-v789-contract": (
        "results/experiment_9269_v789_contract.json",
        "sha256:0d0aa11d171cf91085491cf59b6add32b221068d487e110e5b0c1a3009ca4e4e",
    ),
    "exp9268-v788-capstone": (
        "results/experiment_9268_v788_capstone.json",
        "sha256:12059727652a55d93a267b186f01cbbb3ed1a043fc79799b67a455aae9eb6d17",
    ),
    "exp9258-victris-energy-bot": (
        "results/experiment_9258_v788_victris_energy_bot.json",
        "sha256:7c66714ad53ba5c88e38a94df68a74ab808f0b1773c3ce1f062d93b927ec5640",
    ),
    "exp9257-forward-qualification": (
        "results/experiment_9257_forward_qualification.json",
        "sha256:10bbcaf0b653f5b8b73f55393af7f8b805520316dc211911a089603c01b86ac6",
    ),
    "exp9256-corpus-prefilter": (
        "results/experiment_9256_v788_corpus_prefilter.json",
        "sha256:d1ac40d315f6a740c5cb30d853045f6c92ed51d2cebeef14e09b7b945f17481a",
    ),
    "exp9255-v788-contract": (
        "results/experiment_9255_v788_contract.json",
        "sha256:7c669d5df1331eb5d22211d22bca58db4959b76af6abcaa814ca5de5a5ea24ef",
    ),
    "exp9244-conformal-risk-control": (
        "results/experiment_9244_v787_conformal_risk_protocol.json",
        "sha256:36890bc36e8a84c4d91b8d8bff2d60698de061938f591dafbf7845c38fe2e693",
    ),
    "exp9241-v787-contract": (
        "results/experiment_9241_v787_contract.json",
        "sha256:32cdfef7df5fb0b829ef9eba8d68a44a0e6b7947afbd402d24129dae27a92f5b",
    ),
    "exp9240-v786-capstone": (
        "results/experiment_9240_v786_capstone.json",
        "sha256:b19a3b647f573f5b74c1dead4584137e25c0301dc47a08efb9cdd3b4e95c9500",
    ),
    "exp9225-selective-deferral-policy": (
        "results/experiment_9225_v785_selective_deferral_protocol.json",
        "sha256:73125a9151fc8be16793ccd1daf3e2b369c5ac0929eac5a088cea7863ed8a4af",
    ),
    "exp9211-label-ambiguity-audit": (
        "results/experiment_9211_v784_label_ambiguity.json",
        "sha256:00da04356c958a063793c21d85287653e38eda69c85c65ca1819426519852610",
    ),
}


def _progress(msg: str) -> None:
    print(f"[{EXP_ID} {datetime.now(UTC).isoformat()}] {msg}", flush=True)


def compact_guard(payload: dict[str, Any]) -> int:
    """Enforce that primary JSON stays under MAX_PRIMARY_BYTES (10 MB)."""
    size = len(json.dumps(payload, sort_keys=True).encode("utf-8"))
    if size >= MAX_PRIMARY_BYTES:
        raise ValueError(f"primary_size_exceeded:{size}>={MAX_PRIMARY_BYTES}")
    return size


def write_node_sidecar(
    raw: Path,
    node_rows: list[dict[str, Any]],
    shard_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write full per-node completion rows to a hashed sidecar under results/raw/."""
    raw.mkdir(parents=True, exist_ok=True)
    sidecar_path = raw / "node_completion_rows.json"
    body = {
        "experiment_id": EXP_ID,
        "task_id": TASK_ID,
        "row_count": len(node_rows),
        "shard_count": len(shard_rows),
        "shard_summaries": shard_rows,
        "node_completion_rows": node_rows,
    }
    atomic_json(sidecar_path, body)
    return {
        "path": str(sidecar_path),
        "sha256": sha256_file(sidecar_path),
        "row_count": len(node_rows),
        "shard_count": len(shard_rows),
        "bytes": sidecar_path.stat().st_size,
    }


def verify_node_sidecar(meta: dict[str, Any]) -> bool:
    """Verify hashed node completion sidecar integrity."""
    sidecar_path = Path(str(meta["path"]))
    actual_sha = sha256_file(sidecar_path)
    if actual_sha != meta["sha256"]:
        raise ValueError(f"node_sidecar_hash_mismatch:{actual_sha}!={meta['sha256']}")
    loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    rows = loaded.get("node_completion_rows", [])
    if len(rows) != int(meta["row_count"]):
        raise ValueError(f"node_sidecar_count_mismatch:{len(rows)}!={meta['row_count']}")
    return True


def _gate_file_sha(check: str, upstream: str, path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "check": check,
            "upstream": upstream,
            "path": str(path),
            "sha256": None,
            "artifact_field": "sha256",
            "op": "==",
            "expected": expected_sha,
            "observed": "missing_file",
            "passed": False,
        }
    actual = _digest(path)
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path),
        "sha256": actual,
        "artifact_field": "sha256",
        "op": "==",
        "expected": expected_sha,
        "observed": actual,
        "passed": actual == expected_sha,
    }


def _gate_field_eq(
    check: str,
    upstream: str,
    path: Path,
    file_sha: str | None,
    payload: dict[str, Any] | None,
    field: str,
    expected: Any,
) -> dict[str, Any]:
    if payload is None:
        observed: Any = "missing_file"
        passed = False
    elif field not in payload:
        observed = "missing_field"
        passed = False
    else:
        observed = payload[field]
        passed = observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path),
        "sha256": file_sha,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _discover_module_nodes(rel_path: str) -> list[str]:
    import ast

    tree = ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))
    return [
        f"{rel_path}::{node.name}"
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


def _build_obligation_manifest() -> dict[str, Any]:
    ob = json.loads((ROOT / OBLIGATION_SIDECAR_REL).read_text(encoding="utf-8"))
    orig_621: list[str] = list(ob["original_node_ids"])
    v779_53: list[str] = list(ob["v779_addition_ids"])
    v780_nodes: list[str] = [str(r["node_id"]) for r in ob["affected_v780_node_assertions"]]
    v782_hist_nodes: list[str] = [
        str(r.get("node_id") or r.get("unit_id")) for r in ob["historical_v782_validation_units"]
    ]
    v783_v789_modules = [
        "tests/python/test_arc_mechanic_routing_9192.py",
        "tests/python/test_v782_capstone_9193.py",
        "tests/python/test_v783_contract_9194.py",
        "tests/python/test_v783_capstone_9207.py",
        "tests/python/test_v784_contract_9208.py",
        "tests/python/test_v784_capstone_9221.py",
        "tests/python/test_label_ambiguity_9211.py",
        "tests/python/test_v785_contract_9222.py",
        "tests/python/test_selective_deferral_9225.py",
        "tests/python/test_v786_capstone_9240.py",
        "tests/python/test_v787_contract_9241.py",
        "tests/python/test_conformal_risk_9244.py",
        "tests/python/test_v788_contract_9255.py",
        "tests/python/test_v788_corpus_prefilter_9256.py",
        "tests/python/test_victris_energy_bot_9258.py",
        "tests/python/test_v788_capstone_9268.py",
        "tests/python/test_v789_contract_9269.py",
        "tests/python/test_v789_corpus_prefilter_9270.py",
        "tests/python/test_v789_forward_qualification_9271.py",
    ]
    v783_v789_nodes = [nid for mod in v783_v789_modules for nid in _discover_module_nodes(mod)]
    base_set = set(orig_621) | set(v779_53)
    affected_v780_v789: list[str] = []
    seen_aff: set[str] = set()
    for nid in v780_nodes + v782_hist_nodes + v783_v789_nodes:
        if nid not in base_set and nid not in seen_aff:
            seen_aff.add(nid)
            affected_v780_v789.append(nid)
    all_expected = orig_621 + v779_53 + affected_v780_v789
    categories: dict[str, str] = {}
    for nid in orig_621:
        categories[nid] = "original_621"
    for nid in v779_53:
        categories[nid] = "v779_addition_53"
    for nid in affected_v780_v789:
        categories[nid] = "affected_v780_v789"
    return {
        "sidecar_path": str(ROOT / OBLIGATION_SIDECAR_REL),
        "sidecar_sha256": OBLIGATION_SIDECAR_SHA,
        "original_node_ids": orig_621,
        "v779_addition_ids": v779_53,
        "affected_v780_v789_node_ids": affected_v780_v789,
        "all_expected_node_ids": all_expected,
        "node_categories": categories,
    }


def inspect_v789(root: Path = ROOT, raw: Path = RAW) -> dict[str, Any]:
    """Authenticate upstream artifacts, gate fields, runtime preconditions, and frozen obligations."""
    raw.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, Any] | None] = {}
    shas: dict[str, str | None] = {}
    gate_checks: list[dict[str, Any]] = []
    short_names = {
        "exp9270-corpus-prefilter": "exp9270",
        "exp9269-v789-contract": "exp9269",
        "exp9268-v788-capstone": "exp9268",
        "exp9258-victris-energy-bot": "exp9258",
        "exp9257-forward-qualification": "exp9257",
        "exp9256-corpus-prefilter": "exp9256",
        "exp9255-v788-contract": "exp9255",
        "exp9244-conformal-risk-control": "exp9244",
        "exp9241-v787-contract": "exp9241",
        "exp9240-v786-capstone": "exp9240",
        "exp9225-selective-deferral-policy": "exp9225",
        "exp9211-label-ambiguity-audit": "exp9211",
    }
    for task_key, (rel, expected_sha) in UPSTREAM_PINS.items():
        path = root / rel
        short = short_names[task_key]
        row = _gate_file_sha(f"{short}_sha256", task_key, path, expected_sha)
        gate_checks.append(row)
        shas[task_key] = row["sha256"]
        loaded[task_key] = (
            (
                json.loads(path.read_bytes())
                if path.stat().st_size < MAX_PRIMARY_BYTES
                else load_compact_primary(path)
            )
            if path.is_file()
            else None
        )

    gate_checks.append(
        _gate_file_sha(
            "exp9269_validation_obligations_sidecar_sha256",
            "exp9269-v789-contract",
            root / OBLIGATION_SIDECAR_REL,
            OBLIGATION_SIDECAR_SHA,
        )
    )
    p9270 = root / UPSTREAM_PINS["exp9270-corpus-prefilter"][0]
    d9270 = loaded["exp9270-corpus-prefilter"]
    s9270 = shas["exp9270-corpus-prefilter"]
    gate_checks.extend(
        [
            _gate_field_eq(
                "exp9270_corpus_scan_ready_score",
                "exp9270-corpus-prefilter",
                p9270,
                s9270,
                d9270,
                "corpus_scan_ready_score",
                1,
            ),
            _gate_field_eq(
                "exp9270_required_checks_passed",
                "exp9270-corpus-prefilter",
                p9270,
                s9270,
                d9270,
                "required_checks_passed",
                True,
            ),
        ]
    )
    p9269 = root / UPSTREAM_PINS["exp9269-v789-contract"][0]
    d9269 = loaded["exp9269-v789-contract"]
    s9269 = shas["exp9269-v789-contract"]
    for field in ("source_ready_score", "static_methods_ready_score", "online_methods_ready_score"):
        gate_checks.append(
            _gate_field_eq(
                f"exp9269_{field}", "exp9269-v789-contract", p9269, s9269, d9269, field, 1
            )
        )
    for key, fields in (
        ("exp9270-corpus-prefilter", ("corpus_manifest_reference", "scan_rows_sidecar_reference")),
        (
            "exp9269-v789-contract",
            ("validation_obligations", "validation_scope_manifest", "historical_input_roots"),
        ),
    ):
        for field in fields:
            reference = (loaded[key] or {}).get(field, {})
            path = root / str(reference.get("path", "__missing_sidecar__"))
            gate_checks.append(
                _gate_file_sha(
                    f"{short_names[key]}_{field}_sha256",
                    key,
                    path,
                    reference.get("sha256", "missing_field"),
                )
            )
    memory = psutil.virtual_memory()
    gate_checks.extend(
        [
            _gate_field_eq(
                "exp9269_contract_ready_score",
                "exp9269-v789-contract",
                p9269,
                s9269,
                d9269,
                "contract_ready_score",
                1,
            ),
            _gate_field_eq(
                "exp9269_replay_authority_ready_score",
                "exp9269-v789-contract",
                p9269,
                s9269,
                d9269,
                "replay_authority_ready_score",
                1,
            ),
        ]
    )
    free_b = shutil.disk_usage(raw).free
    cpus = os.cpu_count() or 1
    mods = {}
    for module in ("json", "pytest", "coverage", "carnot", "ruff", "mypy"):
        try:
            mods[module] = importlib.import_module(module).__name__
        except ImportError as exc:
            mods[module] = f"unavailable:{exc}"
    preconditions = [
        {
            "resource": "python_imports",
            "available": not any(v.startswith("unavailable:") for v in mods.values()),
            "receipt": mods,
        },
        {
            "resource": "memory_available_bytes",
            "available": memory.available >= 536_870_912,
            "receipt": {
                "available_bytes": memory.available,
                "total_bytes": memory.total,
                "address_space_limit": resource.getrlimit(resource.RLIMIT_AS),
            },
        },
        {
            "resource": "cpu_budget",
            "available": cpus >= 1,
            "receipt": {
                "cpu_count": cpus,
                "affinity": sorted(os.sched_getaffinity(0)),
                "cpu_time_limit": resource.getrlimit(resource.RLIMIT_CPU),
                "budget": BUDGET,
            },
        },
        {
            "resource": "disk_free_bytes",
            "available": free_b >= 536_870_912,
            "receipt": {"path": str(raw), "free_bytes": free_b},
        },
    ]
    manifest = (
        _build_obligation_manifest()
        if all(g["passed"] for g in gate_checks)
        else {
            "original_node_ids": [],
            "v779_addition_ids": [],
            "affected_v780_v789_node_ids": [],
            "all_expected_node_ids": [],
            "node_categories": {},
        }
    )
    return {
        "gate_checks": gate_checks,
        "preconditions": preconditions,
        "obligation_manifest": manifest,
        "upstream_shas": shas,
        "code_config_hashes": _code_config_hashes(raw),
        "source_hashes_reference": {
            "path": str(raw / "frozen_source_hashes.json"),
            "sha256": sha256_file(raw / "frozen_source_hashes.json"),
        },
    }


def build_scope(root: Path, raw: Path, obligation_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build disjoint whole-module shards, per-shard coverage paths, and tail checks."""
    raw.mkdir(parents=True, exist_ok=True)
    private_dir = Path(tempfile.mkdtemp(prefix="carnot_9271_priv_", dir="/var/tmp"))
    cov_ini = raw / "coverage_owned.ini"
    cov_ini.write_text(
        "[run]\nbranch = False\nparallel = True\npatch = subprocess\ninclude =\n    "
        + "\n    ".join(str(root / rel) for rel in OWNED)
        + "\n",
        encoding="utf-8",
    )
    shard_ini = raw / "coverage_shards.ini"
    shard_ini.write_text(
        "[run]\nbranch = False\nparallel = True\ninclude =\n    "
        + "\n    ".join(str(root / rel) for rel in OWNED)
        + "\n",
        encoding="utf-8",
    )
    orig_files = sorted({nid.split("::")[0] for nid in obligation_manifest["original_node_ids"]})
    v780_shared = {
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_current_work_receipt.py",
    }
    target_corpus = "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
    orig_adv = [f for f in orig_files if "test_adversarial_verify_" in f and f != target_corpus]
    orig_rep = [
        f for f in orig_files if "test_adversarial_verify_" not in f and f not in v780_shared
    ]
    v779_files = sorted({nid.split("::")[0] for nid in obligation_manifest["v779_addition_ids"]})
    exp4_files = sorted(
        {
            nid.split("::")[0]
            for nid in obligation_manifest["affected_v780_v789_node_ids"]
            if "test_experiment_4" in nid.split("::")[0]
        }
    )
    shard_defs: list[tuple[str, list[str], Path]] = [
        ("owned_tests", list(TESTS), cov_ini),
        ("shard_target_corpus", [target_corpus], shard_ini),
        ("shard_original_adversarial_a", orig_adv[:20], shard_ini),
        ("shard_original_adversarial_b", orig_adv[20:], shard_ini),
        ("shard_original_reporting_and_v779", orig_rep + v779_files, shard_ini),
        (
            "shard_v780_controls",
            [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
                "tests/python/test_conductor_gates.py",
                "tests/python/test_validation_schedule_9153.py",
            ],
            shard_ini,
        ),
        (
            "shard_v781_v782_causal_and_lint",
            [
                "tests/python/test_causal_validation_9167.py",
                "tests/python/test_causal_validation_9181.py",
                "tests/python/test_verdict_row_consistency_lint.py",
            ],
            shard_ini,
        ),
        (
            "shard_v781_v782_arc_and_regressions",
            [
                "tests/python/test_arc_mechanic_routing_9178.py",
                "tests/python/test_arc_mechanic_routing_9192.py",
                "tests/python/test_hard_exit_learning_qualification_8206.py",
                *exp4_files,
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v782_v783",
            [
                "tests/python/test_v782_contract_9180.py",
                "tests/python/test_v782_capstone_9193.py",
                "tests/python/test_v783_contract_9194.py",
                "tests/python/test_v783_capstone_9207.py",
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v784",
            [
                "tests/python/test_v784_contract_9208.py",
                "tests/python/test_v784_capstone_9221.py",
                "tests/python/test_label_ambiguity_9211.py",
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v785_v786",
            [
                "tests/python/test_v785_contract_9222.py",
                "tests/python/test_selective_deferral_9225.py",
                "tests/python/test_v786_capstone_9240.py",
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v787",
            [
                "tests/python/test_v787_contract_9241.py",
                "tests/python/test_conformal_risk_9244.py",
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v788_contract",
            ["tests/python/test_v788_contract_9255.py"],
            shard_ini,
        ),
        (
            "shard_consumers_v788_capstone_and_methods",
            [
                "tests/python/test_v788_capstone_9268.py",
                "tests/python/test_victris_energy_bot_9258.py",
                "tests/python/test_v788_corpus_prefilter_9256.py",
            ],
            shard_ini,
        ),
        (
            "shard_consumers_v789",
            [
                "tests/python/test_v789_contract_9269.py",
                "tests/python/test_v789_corpus_prefilter_9270.py",
            ],
            shard_ini,
        ),
        (
            "e2e018_021",
            [
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
            shard_ini,
        ),
    ]
    assigned = {f for _, files, _ in shard_defs for f in files}
    remaining = sorted(
        {nid.split("::")[0] for nid in obligation_manifest["all_expected_node_ids"]} - assigned
    )
    shard_defs.extend(
        (f"historical_module_{i}", [rel], shard_ini) for i, rel in enumerate(remaining)
    )
    all_expected = list(obligation_manifest["all_expected_node_ids"])
    shard_expected_nodes: dict[str, list[str]] = {}
    shards: list[dict[str, Any]] = []
    for name, files, ini_path in shard_defs:
        cov_file = raw / f".coverage.{name}"
        event_log = raw / f"events_{name}.jsonl"
        fset = set(files)
        expected_nodes = [nid for nid in all_expected if nid.split("::")[0] in fset]
        shard_expected_nodes[name] = expected_nodes
        cwd = private_dir if name == "e2e018_021" else root
        cmd = [
            "env",
            f"--chdir={cwd}",
            "--unset=PYTHONPATH",
            "JAX_PLATFORMS=cpu",
            f"COVERAGE_FILE={cov_file}",
            f"CARNOT_9125_NODE_LOG={event_log}",
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(root / "pyproject.toml"),
            f"--rootdir={root}",
            "-n0",
            "-o",
            "addopts=",
            "-p",
            "carnot.reporting.receipt_qualification_9125",
            f"--cov={root / 'python' / 'carnot' / 'reporting'}",
            f"--cov={root / 'scripts' / 'experiments'}",
            f"--cov-config={ini_path}",
            "--cov-report=",
            "-q",
            *[str(root / rel) for rel in files],
        ]
        shards.append(
            {
                "name": name,
                "test_files": files,
                "expected_node_count": len(expected_nodes),
                "coverage_shard_path": str(cov_file),
                "event_log": str(event_log),
                "chdir": str(cwd),
                "cmd": cmd,
            }
        )
    obligation_manifest["shard_expected_nodes"] = shard_expected_nodes
    scoped_targets = [str(root / rel) for rel in (*OWNED, *TESTS)]
    tail_checks = [
        (
            "outside_checkout_private_cli",
            [
                "env",
                f"--chdir={private_dir}",
                "--unset=PYTHONPATH",
                sys.executable,
                str(SCRIPT),
                "--help",
            ],
        ),
        ("ruff_check", [sys.executable, "-m", "ruff", "check", *scoped_targets]),
        ("ruff_format", [sys.executable, "-m", "ruff", "format", "--check", *scoped_targets]),
        ("strict_mypy", [sys.executable, "-m", "mypy", "--strict", *scoped_targets]),
        (
            "spec_coverage",
            [
                sys.executable,
                str(root / "scripts" / "check_spec_coverage.py"),
                str(root / TESTS[0]),
            ],
        ),
    ]
    return {
        "private_dir": str(private_dir),
        "coverage_config": str(cov_ini),
        "shards": shards,
        "tail_checks": tail_checks,
        "required_check_names": [s["name"] for s in shards] + [name for name, _ in tail_checks],
    }


def run_checked_command(
    name: str,
    cmd: list[str],
    raw_dir: Path,
    *,
    stop_monotonic: float | None = None,
) -> dict[str, Any]:
    """Execute one command with 60s heartbeat supervision via validation_diagnostics_9181.run."""
    from carnot.reporting.validation_diagnostics_9181 import run as diag_run

    raw_dir.mkdir(parents=True, exist_ok=True)
    deadline = (
        stop_monotonic
        if stop_monotonic is not None
        else (time.monotonic() + BUDGET["measurement_deadline_s"])
    )
    res = diag_run(name, cmd, raw_dir, deadline)
    out_text = Path(str(res["stdout_path"])).read_text(encoding="utf-8", errors="replace")
    err_text = Path(str(res["stderr_path"])).read_text(encoding="utf-8", errors="replace")
    return {
        **res,
        "cmd": cmd,
        "returncode": int(res["exit_code"]),
        "log": str(res["stdout_path"]),
        "stdout_tail": out_text[-1600:],
        "stderr_tail": err_text[-1600:],
    }


def parse_coverage_report(
    report: dict[str, Any],
    cov_json_path: Path,
    shard_paths: list[str],
) -> dict[str, Any]:
    """Evaluate 100% statement coverage and zero exclusions on OWNED files."""
    files_map = report.get("files", {})
    owned_stats: dict[str, Any] = {}
    cov_ok = True
    for rel in OWNED:
        matched = next((v for k, v in files_map.items() if k.endswith(rel)), None)
        if matched is None:
            cov_ok = False
            owned_stats[rel] = {
                "percent_covered": 0.0,
                "missing_lines": ["file_not_measured"],
                "excluded_lines": [],
                "num_statements": 0,
                "covered_lines": 0,
            }
        else:
            summary = matched.get("summary", {})
            pct = float(summary.get("percent_covered", 0.0))
            missing = list(matched.get("missing_lines", []))
            excluded = list(matched.get("excluded_lines", []))
            owned_stats[rel] = {
                "percent_covered": pct,
                "missing_lines": missing,
                "excluded_lines": excluded,
                "num_statements": int(summary.get("num_statements", 0)),
                "covered_lines": int(summary.get("covered_lines", 0)),
            }
            if pct < 100.0 or missing or excluded:
                cov_ok = False
    return {
        "passed": cov_ok,
        "coverage_json": str(cov_json_path),
        "shard_files": shard_paths,
        "owned_files": owned_stats,
    }


def combine_owned_coverage(
    raw: Path,
    shards: list[dict[str, Any]],
    *,
    run_cmd: Callable[..., Any] = subprocess.run,
    stop_monotonic: float | None = None,
) -> dict[str, Any]:
    """Combine per-shard .coverage.<shard> files and export JSON coverage report."""
    cov_ini = raw / "coverage_owned.ini"
    combined_cov = raw / ".coverage.combined"
    cov_json = raw / "coverage.json"
    shard_paths = [
        str(s["coverage_shard_path"])
        for s in shards
        if Path(str(s["coverage_shard_path"])).exists()
    ]
    env = {**os.environ, "COVERAGE_FILE": str(combined_cov)}
    if run_cmd is subprocess.run:

        def supervised(argv: list[str], **kwargs: Any) -> dict[str, Any]:
            return run_checked_command(
                "coverage_" + argv[3],
                ["env", f"COVERAGE_FILE={combined_cov}", *argv],
                raw,
                stop_monotonic=stop_monotonic,
            )

        run_cmd = supervised
    run_cmd(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            f"--rcfile={cov_ini}",
            "--keep",
            *shard_paths,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    run_cmd(
        [sys.executable, "-m", "coverage", "json", f"--rcfile={cov_ini}", "-o", str(cov_json)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(cov_json.read_text(encoding="utf-8")) if cov_json.is_file() else {}
    return parse_coverage_report(report, cov_json, shard_paths)


def reduce_node_completion(
    obligation_manifest: dict[str, Any],
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce per-shard event logs against the frozen obligation node union."""
    expected_nodes: list[str] = list(obligation_manifest["all_expected_node_ids"])
    expected_set = set(expected_nodes)
    raw_starts: dict[str, int] = {}
    start_counts: dict[str, int] = {}
    phases: dict[str, dict[str, dict[str, Any]]] = {}
    node_shards: dict[str, str] = {}
    for shard in shards:
        sname = str(shard["name"])
        for fpath in shard.get("test_files", []):
            node_shards.setdefault(str(fpath), sname)
        log_path = Path(str(shard["event_log"]))
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
        for line in lines:
            ev = json.loads(line)
            raw_nid = str(ev["nodeid"])
            nid = raw_nid if raw_nid in expected_set else raw_nid.split("[", 1)[0]
            node_shards[nid] = sname
            when = str(ev.get("when", ev.get("event", "")))
            if when == "start":
                raw_starts[raw_nid] = raw_starts.get(raw_nid, 0) + 1
                start_counts[nid] = max(start_counts.get(nid, 0), raw_starts[raw_nid])
            else:
                prev = phases.setdefault(nid, {}).get(when)
                prev_out = str(prev["outcome"]) if prev else "passed"
                outcome = str(ev["outcome"]) if prev_out == "passed" else prev_out
                dur = round(
                    float(prev["duration_s"] if prev else 0.0)
                    + float(ev.get("duration", ev.get("duration_s", 0.0))),
                    6,
                )
                phases[nid][when] = {"outcome": outcome, "duration_s": dur}

    categories: dict[str, str] = dict(obligation_manifest.get("node_categories", {}))
    node_rows: list[dict[str, Any]] = []
    completed_count = 0
    failed_count = 0
    censored_count = 0
    missing_count = 0
    duplicate_count = 0
    shard_buckets: dict[str, dict[str, Any]] = {
        str(s["name"]): {
            "shard": str(s["name"]),
            "module_count": len(s.get("test_files", [])),
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "missing": 0,
            "duplicate": 0,
            "duration_s": 0.0,
        }
        for s in shards
    }
    for nid in expected_nodes:
        mod = nid.split("::")[0]
        sname = node_shards.get(nid, node_shards.get(mod, "unassigned"))
        starts = start_counts.get(nid, 0)
        ph = phases.get(nid, {})
        has_all = all(w in ph for w in ("setup", "call", "teardown"))
        setup_s = float(ph.get("setup", {}).get("duration_s", 0.0))
        call_s = float(ph.get("call", {}).get("duration_s", 0.0))
        teardown_s = float(ph.get("teardown", {}).get("duration_s", 0.0))
        total_s = round(setup_s + call_s + teardown_s, 6)
        if starts == 0 and not ph:
            status = "missing"
            censor_reason: str | None = "missing_from_shard_events"
            missing_count += 1
        elif starts > 1:
            status = "duplicate"
            censor_reason = "duplicate_node_execution"
            duplicate_count += 1
        elif not has_all:
            status = "censored"
            censor_reason = "incomplete_phase_events"
            censored_count += 1
        elif any(ph[w]["outcome"] != "passed" for w in ("setup", "call", "teardown")):
            status = "failed"
            censor_reason = None
            failed_count += 1
        else:
            status = "completed"
            censor_reason = None
            completed_count += 1
        bucket = shard_buckets.get(sname)
        if bucket is not None:
            bucket[status] = int(bucket[status]) + 1
            bucket["duration_s"] = round(float(bucket["duration_s"]) + total_s, 6)
        node_rows.append(
            {
                "unit_id": nid,
                "node_id": nid,
                "module": mod,
                "shard": sname,
                "category": categories.get(nid, "affected_v780_v789"),
                "start_count": starts,
                "setup_outcome": ph.get("setup", {}).get("outcome", "missing"),
                "call_outcome": ph.get("call", {}).get("outcome", "missing"),
                "teardown_outcome": ph.get("teardown", {}).get("outcome", "missing"),
                "setup_s": setup_s,
                "call_s": call_s,
                "teardown_s": teardown_s,
                "duration_s": total_s,
                "status": status,
                "numerator": 1 if status == "completed" else 0,
                "denominator": 1,
                "censor_reason": censor_reason,
            }
        )
    union_passed = (
        completed_count == len(expected_nodes)
        and failed_count == 0
        and censored_count == 0
        and missing_count == 0
        and duplicate_count == 0
    )
    return {
        "union_passed": union_passed,
        "expected_count": len(expected_nodes),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "censored_count": censored_count,
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "shard_summaries": list(shard_buckets.values()),
        "node_rows": node_rows,
    }


def verify(
    root: Path = ROOT,
    raw: Path = RAW,
    *,
    obligation_manifest: dict[str, Any] | None = None,
    run_fn: Callable[..., dict[str, Any]] = run_checked_command,
    cov_fn: Callable[..., dict[str, Any]] = combine_owned_coverage,
    started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Execute the complete V789 validation union once under the current code revision."""
    raw.mkdir(parents=True, exist_ok=True)
    manifest = obligation_manifest or _build_obligation_manifest()
    scope = build_scope(root, raw, manifest)
    start = time.monotonic() if started_monotonic is None else started_monotonic
    stop_monotonic = start + BUDGET["measurement_deadline_s"]
    roster_path = raw / "frozen_obligation_manifest.json"
    atomic_json(roster_path, manifest)
    atomic_json(raw / "frozen_scope.json", scope)
    _progress(f"phase=measurement_shards_start shard_count={len(scope['shards'])}")

    def _run_shard(shard: dict[str, Any]) -> dict[str, Any]:
        return run_fn(str(shard["name"]), list(shard["cmd"]), raw, stop_monotonic=stop_monotonic)

    with ThreadPoolExecutor(max_workers=8) as pool:
        shard_checks = list(pool.map(_run_shard, scope["shards"]))

    _progress("phase=coverage_combine_start")
    verification_stop = stop_monotonic + BUDGET["verification_reserve_s"]
    coverage = (
        cov_fn(raw, scope["shards"], stop_monotonic=verification_stop)
        if cov_fn is combine_owned_coverage
        else cov_fn(raw, scope["shards"])
    )

    _progress("phase=tail_checks_start")

    def _run_tail(item: tuple[str, list[str]]) -> dict[str, Any]:
        return run_fn(item[0], item[1], raw, stop_monotonic=verification_stop)

    with ThreadPoolExecutor(max_workers=4) as pool:
        tail_checks = list(pool.map(_run_tail, scope["tail_checks"]))

    checks = shard_checks + tail_checks
    node_reduction = reduce_node_completion(manifest, scope["shards"])
    node_sidecar = write_node_sidecar(
        raw, node_reduction["node_rows"], node_reduction["shard_summaries"]
    )
    verify_node_sidecar(node_sidecar)
    e2e_ok = next((bool(c["passed"]) for c in shard_checks if c["name"] == "e2e018_021"), False)
    required_checks_passed = (
        all(bool(c["passed"]) for c in checks)
        and bool(coverage["passed"])
        and bool(node_reduction["union_passed"])
        and e2e_ok
    )
    _progress(f"phase=measurement_complete required_checks_passed={required_checks_passed}")
    return {
        "scope": scope,
        "shards": scope["shards"],
        "checks": checks,
        "coverage": coverage,
        "node_reduction": node_reduction,
        "node_sidecar": node_sidecar,
        "e2e_passed": e2e_ok,
        "required_checks_passed": required_checks_passed,
        "obligation_manifest_reference": {
            "path": str(roster_path),
            "sha256": sha256_file(roster_path),
        },
        "budget": dict(
            BUDGET, started_monotonic_s=start, measurement_stop_monotonic_s=stop_monotonic
        ),
        "scope_reference": {
            "path": str(raw / "frozen_scope.json"),
            "sha256": sha256_file(raw / "frozen_scope.json"),
        },
        "event_references": [
            {"path": str(s["event_log"]), "sha256": sha256_file(Path(s["event_log"]))}
            for s in scope["shards"]
            if Path(s["event_log"]).is_file()
        ],
    }


def simulate_verification_bundle(
    root: Path,
    raw: Path,
    obligation_manifest: dict[str, Any],
    *,
    all_pass: bool = True,
) -> dict[str, Any]:
    """Deterministic fast verification bundle for unit tests of publish/replay/CLI controls."""
    raw.mkdir(parents=True, exist_ok=True)
    scope = build_scope(root, raw, obligation_manifest)
    for shard in scope["shards"]:
        sname = str(shard["name"])
        lines: list[str] = []
        for nid in obligation_manifest["shard_expected_nodes"].get(sname, []):
            outcome = "passed" if all_pass else "failed"
            lines.extend(
                [
                    json.dumps({"event": "start", "nodeid": nid, "ts_ns": 1}),
                    json.dumps(
                        {
                            "event": "phase",
                            "nodeid": nid,
                            "when": "setup",
                            "outcome": "passed",
                            "duration_s": 0.001,
                            "ts_ns": 2,
                        }
                    ),
                    json.dumps(
                        {
                            "event": "phase",
                            "nodeid": nid,
                            "when": "call",
                            "outcome": outcome,
                            "duration_s": 0.002,
                            "ts_ns": 3,
                        }
                    ),
                    json.dumps(
                        {
                            "event": "phase",
                            "nodeid": nid,
                            "when": "teardown",
                            "outcome": "passed",
                            "duration_s": 0.001,
                            "ts_ns": 4,
                        }
                    ),
                ]
            )
        Path(str(shard["event_log"])).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    checks = [
        {
            "name": name,
            "cmd": ["simulated", name],
            "returncode": 0 if all_pass else 1,
            "passed": all_pass,
            "duration_s": 0.01,
            "log": str(raw / f"{name}.log"),
            "stdout_tail": "simulated_ok" if all_pass else "simulated_fail",
            "stderr_tail": "",
        }
        for name in scope["required_check_names"]
    ]
    coverage = parse_coverage_report(
        {
            "files": {
                rel: {
                    "summary": {
                        "num_statements": 100,
                        "covered_lines": 100 if all_pass else 90,
                        "percent_covered": 100.0 if all_pass else 90.0,
                    },
                    "missing_lines": [] if all_pass else [1],
                    "excluded_lines": [],
                }
                for rel in OWNED
            }
        },
        raw / "coverage.json",
        [str(s["coverage_shard_path"]) for s in scope["shards"]],
    )
    node_reduction = reduce_node_completion(obligation_manifest, scope["shards"])
    node_sidecar = write_node_sidecar(
        raw, node_reduction["node_rows"], node_reduction["shard_summaries"]
    )
    verify_node_sidecar(node_sidecar)
    return {
        "scope": scope,
        "shards": scope["shards"],
        "checks": checks,
        "coverage": coverage,
        "node_reduction": node_reduction,
        "node_sidecar": node_sidecar,
        "e2e_passed": all_pass,
        "required_checks_passed": all_pass
        and bool(coverage["passed"])
        and bool(node_reduction["union_passed"]),
    }


def _code_config_hashes(raw: Path | None = None) -> dict[str, Any]:
    owned_map = {rel: _digest(ROOT / rel) for rel in OWNED}
    test_map = {rel: _digest(ROOT / rel) for rel in TESTS}
    cfg_map = {
        rel: _digest(ROOT / rel) for rel in (*CONFIGS, "pyproject.toml", "tests/python/conftest.py")
    }
    affected = {
        str(p.relative_to(ROOT)): _digest(p)
        for base in ("python/carnot", "tests/python", "scripts")
        for p in sorted((ROOT / base).rglob("*.py"))
    }
    if raw is not None:
        atomic_json(raw / "frozen_source_hashes.json", affected)
    return {
        "owned_implementation": owned_map,
        "tests": test_map,
        "configuration": cfg_map,
        "affected_source_digest": canonical_hash(affected),
        "affected_source_count": len(affected),
        "combined_revision_hash": canonical_hash(
            {"owned": owned_map, "tests": test_map, "configuration": cfg_map, "affected": affected}
        ),
    }


def build_payload(
    inspected: dict[str, Any],
    verification: dict[str, Any],
    receipt: dict[str, Any],
    duration_s: float,
    *,
    date: str = DATE,
) -> dict[str, Any]:
    """Assemble the Exp9271 V789 forward qualification payload from rows and receipts."""
    gate_checks: list[dict[str, Any]] = list(inspected["gate_checks"])
    preconditions: list[dict[str, Any]] = list(inspected["preconditions"])
    manifest: dict[str, Any] = dict(inspected["obligation_manifest"])
    gates_ok = all(bool(g["passed"]) for g in gate_checks)
    pre_ok = all(bool(p["available"]) for p in preconditions)
    blocked = not (gates_ok and pre_ok)
    checks: list[dict[str, Any]] = list(verification["checks"])
    cov: dict[str, Any] = dict(verification["coverage"])
    node_red: dict[str, Any] = dict(verification["node_reduction"])
    node_sidecar: dict[str, Any] = dict(verification["node_sidecar"])
    receipts_ok = not validate_current_work_receipt(receipt, root=ROOT)
    req_ok = bool(verification["required_checks_passed"]) and not blocked
    node_ok = bool(node_red["union_passed"]) and not blocked
    cov_ok = bool(cov["passed"]) and not blocked
    e2e_ok = bool(verification["e2e_passed"]) and not blocked
    ready = gates_ok and pre_ok and req_ok and node_ok and cov_ok and e2e_ok and receipts_ok
    verdict_class = "blocked" if blocked else "null"
    honest_verdict = (
        "complete_blocked_v789_forward_qualification"
        if blocked
        else "complete_null_v789_forward_qualification"
    )
    rows = [
        {
            "unit_id": f"check::{c['name']}",
            "qid": str(c["name"]),
            "group": str(c["name"]),
            "sibling": "validation_shard_or_tail_check",
            "arm": "v789_complete_validation_union",
            "condition": "single_pass_whole_module_shards",
            "seed": SEED,
            "metric": "validation_check_passed",
            "absolute_metric": None if blocked else float(1 if c["passed"] else 0),
            "pass_score": 0 if blocked else (1 if c["passed"] else 0),
            "duration_s": round(float(c["duration_s"]), 6),
            "numerator": 0 if blocked else (1 if c["passed"] else 0),
            "denominator": 1,
            "status": "censored" if blocked else ("completed" if c["passed"] else "failed"),
            "exclusion_reason": None,
            "censor_reason": "gate_check_failed" if blocked else None,
        }
        for c in checks
    ]
    completed_rows = sum(1 for r in rows if r["status"] == "completed")
    failed_rows = sum(1 for r in rows if r["status"] == "failed")
    censored_rows = sum(1 for r in rows if r["status"] == "censored")
    attempted_rows = completed_rows + failed_rows
    code_hashes = inspected["code_config_hashes"]
    revision_unchanged = code_hashes == _code_config_hashes()
    ready = ready and revision_unchanged
    shard_summary_rows = [
        {
            **s,
            "phase_exits": {
                "setup": "passed"
                if s["failed"] == 0 and s["missing"] == 0 and s["censored"] == 0
                else "failed",
                "call": "passed"
                if s["failed"] == 0 and s["missing"] == 0 and s["censored"] == 0
                else "failed",
                "teardown": "passed"
                if s["failed"] == 0 and s["missing"] == 0 and s["censored"] == 0
                else "failed",
            },
            "sidecar_path": node_sidecar["path"],
            "sidecar_sha256": node_sidecar["sha256"],
            "sidecar_row_count": node_sidecar["row_count"],
        }
        for s in node_red["shard_summaries"]
    ]
    acceptance_gates = {
        "executed_revision_unchanged": {
            "condition": "Frozen code and configuration hashes equal the publication revision",
            "passed": revision_unchanged,
            "principle": "Validation qualifies only the revision actually executed.",
        },
        "upstream_gates_authenticated": {
            "condition": "All upstream V784-V789 artifacts and Exp9270 corpus_scan_ready_score==1 authenticate by SHA256 and field value",
            "passed": gates_ok,
            "principle": "Every blocked result names its failed operand; missing evidence differs from an observed zero.",
        },
        "preconditions_available": {
            "condition": "Python imports, CPU budget, and private disk space available before execution",
            "passed": pre_ok,
            "principle": "Unavailable resources cannot produce invented measurements.",
        },
        "complete_node_union_passed": {
            "condition": "All 621 original nodes, 53 V779 additions, and affected V780-V789 nodes complete setup/call/teardown once with zero missing, duplicate, failed, or censored nodes",
            "passed": node_ok,
            "principle": "No duplicated or absent node can satisfy the full union.",
        },
        "owned_coverage_100_percent": {
            "condition": "100 percent owned statement coverage with zero exclusions across combined per-shard coverage files",
            "passed": cov_ok,
            "principle": "Self-reported passing checks do not establish readiness.",
        },
        "outside_checkout_e2e_018_021_passed": {
            "condition": "Outside-checkout private CLI E2E-018/021 checks pass within budget",
            "passed": e2e_ok,
            "principle": "Readiness requires every affected check.",
        },
        "required_checks_passed": {
            "condition": "All whole-module pytest shards, ruff_check, ruff_format, strict_mypy, and spec_coverage pass",
            "passed": req_ok,
            "principle": "Readiness requires every affected check.",
        },
        "current_work_receipt_authenticated": {
            "condition": "Current work receipt validates with zero LLM invocations and zero generated tokens",
            "passed": receipts_ok,
            "principle": "Historical receipts cannot certify current inference.",
        },
    }
    source_refs = [
        {
            "producer": task_key,
            "path": rel,
            "sha256": expected_sha,
            "observed_sha256": inspected["upstream_shas"].get(task_key),
            "imported_fields": ["sha256", "honest_verdict", "verdict_class"],
        }
        for task_key, (rel, expected_sha) in UPSTREAM_PINS.items()
    ]
    payload: dict[str, Any] = {
        "schema": "carnot.v789_forward_qualification.v1",
        "experiment_id": 9271,
        "task_id": TASK_ID,
        "milestone": "2026.11.789",
        "date": date,
        "run_date": date,
        "timestamp": datetime.now(UTC).isoformat(),
        "title": "Exp9271 V789 Complete Causally Corrected Validation Revision",
        "summary": (
            f"Recorded the V789 validation union ({len(manifest['original_node_ids'])} original nodes, "
            f"{len(manifest['v779_addition_ids'])} V779 additions, and {len(manifest['affected_v780_v789_node_ids'])} "
            f"affected V780-V789 obligations = {node_red['expected_count']} total nodes) once across "
            f"{len(shard_summary_rows)} disjoint whole-module shards; forward_validation_ready_score={1 if ready else 0}."
        ),
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": gate_checks,
        "preconditions_checked": preconditions,
        "rows": rows,
        "intended_count": len(rows),
        "attempted_count": attempted_rows,
        "completed_count": completed_rows,
        "failed_count": failed_rows,
        "censored_count": censored_rows,
        "excluded_count": 0,
        "effective_independent_groups": completed_rows,
        "sample_size_budget": {
            "intended": len(rows),
            "attempted": attempted_rows,
            "completed": completed_rows,
            "failed": failed_rows,
            "censored": censored_rows,
            "effective_independent_groups": completed_rows,
        },
        "verifier_is_oracle": False,
        "independent_generalization_score": 0,
        "claim_scope": {
            "exposure_scope": "exposed_development_only",
            "independent_generalization_score": 0,
            "allowed_claims": [
                "Complete 621 original nodes, 53 V779 additions, and affected V780-V789 obligations executed once under current code revision with per-shard coverage and measured phase clocks."
            ]
            if ready
            else ["Recorded failed or blocked qualification operands and completed controls only."],
            "disallowed_claims": ["No current LLM load, forward pass, or token generation claim."],
        },
        "MODEL_SPECS": [],
        "model_invocation_status": "not_requested_no_model_load",
        "requested_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run" if blocked else "no_model_load",
        "inference_substrate": "no_model_load_causally_corrected_validation_union_orchestration",
        "model_invocation_counts": {
            "current_llm_invocations": 0,
            "loads_attempted": 0,
            "loads_completed": 0,
            "loads_failed": 0,
            "forward_calls": 0,
            "generated_tokens": 0,
        },
        "current_llm_invocations": 0,
        "generated_tokens": 0,
        "duration_s": round(duration_s, 6),
        "random_seed": SEED,
        "reproducibility_checksum": canonical_hash(
            {
                "seed": SEED,
                "upstream_pins": UPSTREAM_PINS,
                "code_config_hashes": code_hashes,
                "expected_node_count": node_red["expected_count"],
                "forward_validation_ready_score": 1 if ready else 0,
            }
        ),
        "source_references": source_refs,
        "code_config_hashes": code_hashes,
        "current_work_receipts": [receipt],
        "validation_receipts": {
            "source_hashes_reference": inspected["source_hashes_reference"],
            "obligation_manifest_reference": verification.get("obligation_manifest_reference"),
            "scope_reference": verification.get("scope_reference"),
            "event_references": verification.get("event_references", []),
            "budget": verification.get("budget", BUDGET),
            "commands": checks,
            "coverage": cov,
            "node_completion_sidecar": node_sidecar,
            "node_union_summary": {
                "original_621_count": len(manifest["original_node_ids"]),
                "v779_addition_53_count": len(manifest["v779_addition_ids"]),
                "affected_v780_v789_count": len(manifest["affected_v780_v789_node_ids"]),
                "expected_count": node_red["expected_count"],
                "completed_count": node_red["completed_count"],
                "failed_count": node_red["failed_count"],
                "censored_count": node_red["censored_count"],
                "missing_count": node_red["missing_count"],
                "duplicate_count": node_red["duplicate_count"],
                "union_passed": node_ok,
            },
            "e2e_018_021_passed": e2e_ok,
            "required_checks_passed": req_ok,
        },
        "required_checks_passed": req_ok and revision_unchanged,
        "acceptance_gates": acceptance_gates,
        "forward_validation_ready_score": 1 if ready else 0,
        "node_completion_rows": shard_summary_rows,
        "qualified_revision_reference": {
            "combined_revision_hash": code_hashes["combined_revision_hash"],
            "code_config_hashes": code_hashes,
            "node_completion_sidecar": node_sidecar,
            "coverage_json": cov["coverage_json"],
            "forward_validation_ready_score": 1 if ready else 0,
        },
    }
    payload["field_principles"] = {
        key: (
            "Bound by authenticated V784-V789 upstream provenance, single-pass whole-module shard execution "
            "of the 621+53+V780-V789 obligation union, 100% owned statement coverage, and hashed sidecars."
        )
        for key in payload
    }
    compact_guard(payload)
    return payload


def replay_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold replay check recomputing readiness and budget from stored rows and sidecar."""
    rows = payload["rows"]
    completed = sum(1 for r in rows if r["status"] == "completed")
    sidecar = payload["validation_receipts"]["node_completion_sidecar"]
    verify_node_sidecar(sidecar)
    validation = payload["validation_receipts"]
    current_evidence_ok = payload["code_config_hashes"] == _code_config_hashes()
    if validation.get("obligation_manifest_reference"):
        refs = [
            validation["obligation_manifest_reference"],
            validation["scope_reference"],
            validation["source_hashes_reference"],
            *validation["event_references"],
        ]
        for command in validation["commands"]:
            refs += [
                command["receipt_reference"],
                *[
                    {"path": command[f"{stream}_path"], "sha256": command[f"{stream}_sha256"]}
                    for stream in ("stdout", "stderr")
                ],
            ]
        current_evidence_ok = current_evidence_ok and all(
            Path(ref["path"]).is_file() and sha256_file(Path(ref["path"])) == ref["sha256"]
            for ref in refs
        )
        manifest = json.loads(
            Path(validation["obligation_manifest_reference"]["path"]).read_bytes()
        )
        scope = json.loads(Path(validation["scope_reference"]["path"]).read_bytes())
        reduced = reduce_node_completion(manifest, scope["shards"])
        current_evidence_ok = (
            current_evidence_ok
            and reduced["union_passed"] == validation["node_union_summary"]["union_passed"]
        )
    gates_ok = all(bool(g["passed"]) for g in payload["gate_check_summary"])
    pre_ok = all(bool(p["available"]) for p in payload["preconditions_checked"])
    cov_ok = bool(payload["validation_receipts"]["coverage"]["passed"])
    union_ok = bool(payload["validation_receipts"]["node_union_summary"]["union_passed"])
    e2e_ok = bool(payload["validation_receipts"]["e2e_018_021_passed"])
    all_rows_ok = completed == len(rows) and len(rows) > 0
    ready = (
        1
        if (
            gates_ok
            and pre_ok
            and cov_ok
            and union_ok
            and e2e_ok
            and all_rows_ok
            and current_evidence_ok
        )
        else 0
    )
    return {
        "passed": ready == payload.get("forward_validation_ready_score"),
        "completed_rows": completed,
        "forward_validation_ready_score": ready,
        "required_checks_passed": all_rows_ok and cov_ok and union_ok and e2e_ok,
    }


def terminal_validate(
    candidate: Path,
    logs: Path,
    *,
    run_linters: bool = True,
    stop_monotonic: float | None = None,
) -> dict[str, Any]:
    """Run adversarial_verify, verdict_row_consistency_lint --strict, and cold replay on candidate."""
    logs.mkdir(parents=True, exist_ok=True)
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    rep = replay_from_payload(loaded)
    if not run_linters:
        return {"passed": bool(rep["passed"]), "cold_replay": rep}
    deadline = (
        stop_monotonic
        if stop_monotonic is not None
        else time.monotonic() + BUDGET["publication_reserve_s"]
    )
    adv = run_checked_command(
        "adversarial_verify",
        [sys.executable, str(ROOT / "scripts" / "adversarial_verify.py"), "--json", str(candidate)],
        logs,
        stop_monotonic=deadline,
    )
    lint = run_checked_command(
        "verdict_row_consistency",
        [
            sys.executable,
            str(ROOT / "scripts" / "verdict_row_consistency_lint.py"),
            "--strict",
            str(candidate),
        ],
        logs,
        stop_monotonic=deadline,
    )
    return {
        "passed": bool(rep["passed"] and adv["passed"] and lint["passed"]),
        "cold_replay": rep,
        "adversarial_verify": adv,
        "verdict_row_consistency": lint,
    }


def publish(
    out: Path = OUT,
    raw: Path = RAW,
    *,
    root: Path = ROOT,
    date: str = DATE,
    verify_fn: Callable[[Path, Path], dict[str, Any]] = verify,
    require_ready: bool = True,
    tamper_receipt: bool = False,
    run_linters: bool = True,
) -> dict[str, Any]:
    """Execute inspection, verification, receipt generation, and atomic primary publication."""
    raw.mkdir(parents=True, exist_ok=True)
    started_ns = time.monotonic_ns()
    inv_dir = raw / "invocations" / str(started_ns)
    inv_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _progress("phase=authenticate_start")
    inspected = inspect_v789(root, inv_dir)
    allowed = all(g["passed"] for g in inspected["gate_checks"]) and all(
        p["available"] for p in inspected["preconditions"]
    )
    _progress(f"phase=authenticate_complete execution_allowed={allowed}")
    if not allowed:
        reduction = reduce_node_completion(inspected["obligation_manifest"], [])
        verification = {
            "checks": [],
            "coverage": parse_coverage_report({}, inv_dir / "coverage.json", []),
            "node_reduction": reduction,
            "node_sidecar": write_node_sidecar(inv_dir, reduction["node_rows"], []),
            "e2e_passed": False,
            "required_checks_passed": False,
        }
    elif verify_fn is verify:
        verification = verify(
            root,
            inv_dir,
            obligation_manifest=inspected["obligation_manifest"],
            started_monotonic=started_ns / 1_000_000_000,
        )
    else:
        verification = verify_fn(root, inv_dir)
    ended_ns = max(time.monotonic_ns(), started_ns + 1)
    duration_s = round((ended_ns - started_ns) / 1_000_000_000, 6)
    atomic_json(inv_dir / "gate_check_summary.json", {"gate_checks": inspected["gate_checks"]})
    atomic_json(
        inv_dir / "validation_summary.json",
        {"checks": verification["checks"], "coverage": verification["coverage"]},
    )
    receipt = build_current_work_receipt(
        run_id=f"{EXP_ID}-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load_causally_corrected_validation_union_orchestration",
        inference_substrate_details={"task_id": TASK_ID, "random_seed": SEED},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
    )
    if tamper_receipt:
        receipt["event_count"] = 1
    errors = validate_current_work_receipt(receipt, root=root)
    if errors:
        raise ValueError(f"receipt_validation_failed:{','.join(errors)}")
    payload = build_payload(inspected, verification, receipt, duration_s, date=date)
    if require_ready and payload["forward_validation_ready_score"] != 1:
        raise ValueError("forward_qualification_check_failed")
    pub = publish_primary(
        out,
        payload,
        lambda cpath: terminal_validate(
            cpath,
            inv_dir / "terminal",
            run_linters=run_linters,
            stop_monotonic=started_ns / 1_000_000_000 + BUDGET["hard_cap_s"],
        ),
    )
    atomic_json(raw / "publication_receipt.json", pub)
    return payload


def main(
    argv: list[str] | None = None,
    *,
    verify_fn: Callable[[Path, Path], dict[str, Any]] = verify,
    run_linters: bool = True,
) -> int:
    """CLI entrypoint for Exp9271 V789 forward qualification."""
    parser = argparse.ArgumentParser(
        description="Exp9271 V789 complete causally corrected validation revision"
    )
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args(argv)
    _progress(f"phase=start date={args.date} out={args.out}")
    payload = publish(
        args.out,
        args.raw_dir,
        date=str(args.date),
        verify_fn=verify_fn,
        require_ready=False,
        run_linters=run_linters,
    )
    _progress(
        f"phase=done honest_verdict={payload['honest_verdict']} "
        f"forward_validation_ready_score={payload['forward_validation_ready_score']}"
    )
    return 0
