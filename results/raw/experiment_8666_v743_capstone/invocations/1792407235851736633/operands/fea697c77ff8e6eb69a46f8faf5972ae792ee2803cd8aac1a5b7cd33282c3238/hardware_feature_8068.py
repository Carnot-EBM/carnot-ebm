"""REQ-REPORT-8068: bound compatible work without inventing device execution."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
CUSTODY = "results/experiment_8055_v697_hardware_guard_boundary.json"
FEATURE = "results/experiment_8066_v698_content_addressed_feature_service.json"
CUSTODY_PIN = "sha256:fca3709a4d51eb006f1a666163666663bd3e6af7fb0cb330858b74cdaadc3e67"
RESOURCES = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "python/carnot/experiment_8055_v697_hardware_guard_boundary.py",
    "python/carnot/reporting/precision_fallback_8042.py",
    "research-hardware-wishlist.md",
    "ops/hardware-bringup-prep.md",
    "research-references.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    ".venv/bin/python",
    ".venv/bin/pytest",
    ".venv/bin/ruff",
    ".venv/bin/mypy",
    ".venv/bin/coverage",
]


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Exact operands explain why an unavailable branch cannot earn credit."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def seal(path: Path, expected: str | None, raw: Path, data: Json, upstream: str) -> Path | None:
    """Keep exact historical bytes so later upstream replacement cannot change replay."""
    observed = sha256_file(path) if path.is_file() else None
    check = operand(
        upstream,
        path,
        "sha256" if expected else "exists",
        expected or True,
        observed if expected else observed is not None,
    )
    data["checks"].append(check)
    if not check["passed"]:
        return None
    assert observed is not None
    saved = raw / "inputs" / observed.split(":")[1] / path.name
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, saved)
    data["references"].append(dict(reference(saved), original_path=str(path)))
    return saved


def authenticate(root: Path, relative: str, eid: int, raw: Path, data: Json) -> tuple[Json, bool]:
    """Bind terminal reports to primary bytes before trusting a readiness field."""
    path = root / relative
    saved = seal(path, CUSTODY_PIN if eid == 8055 else None, raw, data, f"exp{eid}")
    if saved is None:
        return {}, False
    value = json.loads(saved.read_bytes())
    before = len(data["checks"])
    expected = dict(experiment_id=eid, flagged_adversarial=False)
    if eid == 8055:
        expected.update(hardware_custody_ready_score=1, owned_checks_passed=True)
    else:
        expected.update(feature_service_ready_score=1, required_checks_passed=True)
    for key, wanted in expected.items():
        data["checks"].append(operand(f"exp{eid}", path, key, wanted, value.get(key)))
    data["checks"].append(
        operand(
            f"exp{eid}",
            path,
            "verdict_class",
            "positive|null",
            "positive|null"
            if value.get("verdict_class") in {"positive", "null"}
            else value.get("verdict_class"),
        )
    )
    try:
        terminal = Path(value["terminal_validation_sidecar_path"])
        receipt = json.loads(terminal.read_bytes())
        binding = receipt["publication"]
        report = read_bound_sidecar(path, Path(binding["sidecar_path"]))
        for field, wanted, actual in (
            ("publication.primary_sha256", sha256_file(path), binding["primary_sha256"]),
            ("publication.primary_path", str(path), binding["primary_path"]),
            ("report.passed", True, report["report"]["passed"]),
        ):
            data["checks"].append(operand(f"exp{eid}", path, field, wanted, actual))
        seal(terminal, None, raw, data, f"exp{eid}")
        seal(Path(binding["sidecar_path"]), None, raw, data, f"exp{eid}")
    except (OSError, KeyError, ValueError) as error:
        data["checks"].append(
            operand(f"exp{eid}", path, "terminal_contract", "byte-bound pass", str(error))
        )
    return value, all(row["passed"] for row in data["checks"][before:])


def load(root: Path, raw: Path) -> Json:
    """Custody is available even when the optional timing study is disqualified."""
    data: Json = dict(
        boards=[],
        costs=[],
        checks=[],
        references=[],
        fixture=False,
        feature_qualified=False,
        numerical_obligations={},
    )
    for label in RESOURCES:
        seal(root / label, None, raw, data, "local_resource")
    prior, custody = authenticate(root, CUSTODY, 8055, raw, data)
    data["numerical_obligations"] = {
        key: prior.get(key)
        for key in (
            "guard_fallback_counts",
            "prediction_fallback_count",
            "prediction_fallback_denominator",
            "guard_predicate_fallback_counts",
            "guard_fallback_ready_score",
        )
    }
    data["numerical_obligations"]["required_policy"] = (
        "Keep input/accumulator overflow checks, threshold and guard uncertainty; "
        "use authoritative float64 CPU fallback before commit. No new quantization study."
    )
    for original in prior.get("board_rows", []):
        board = deepcopy(original)
        before = len(data["checks"])
        source = Path(board["source_path"])
        source = source if source.is_absolute() else root / source
        saved = seal(source, board["source_hash"], raw, data, board["board"])
        receipt = json.loads(saved.read_bytes()) if saved else {}
        transcript = receipt.get("kv260_terminal_transcript_path") or receipt.get(
            "raw_dispatch_transcript_path"
        )
        digest = receipt.get("kv260_terminal_transcript_sha256") or next(
            (
                r.get("latest_receipt_hash")
                for r in receipt.get("board_rows", [])
                if r.get("board") == board["board"]
            ),
            None,
        )
        if transcript and digest:
            p = Path(transcript)
            seal(
                p if p.is_absolute() else root / p,
                digest if digest.startswith("sha256:") else "sha256:" + digest,
                raw,
                data,
                board["board"],
            )
        board["custody_valid"] = custody and all(r["passed"] for r in data["checks"][before:])
        data["boards"].append(board)
    feature, qualified = authenticate(root, FEATURE, 8066, raw, data)
    data["feature_path"] = next(
        (r["path"] for r in data["references"] if r.get("original_path") == str(root / FEATURE)),
        str(root / FEATURE),
    )
    data["feature_qualified"] = qualified
    if qualified:
        for ref in feature["raw_shard_hashes"]:
            seal(Path(ref["path"]), ref["sha256"], raw, data, "exp8066")
        data["feature_qualified"] = all(
            r["passed"] for r in data["checks"] if r["upstream"] == "exp8066"
        )
        data["costs"] = [
            deepcopy(r)
            for r in feature["rows"]
            if r["status"] == "completed" and r["cached"] and r["repetition"] >= 0
        ]
    return data


def reduce(data: Json) -> Json:
    """Reduce historical obligations and estimates without converting either to speed."""
    checks = deepcopy(data["checks"])
    boards = deepcopy(data["boards"])
    contracts = {
        "KV260": ("fpga_fabric", 5, None),
        "PolarFire": ("linux_cpu", None, None),
        "GateMate": ("none", None, "0xffffffff"),
    }
    for board in boards:
        expected = contracts.get(board["board"])
        actual = (board.get("processor_class"), board.get("k_max"), board.get("blocker"))
        valid = expected == actual and board.get("current_hardware_execution") is False
        checks.append(
            operand(
                board["board"],
                Path(board["source_path"]),
                "board_scope",
                True,
                valid and board["custody_valid"],
            )
        )
        board["custody_valid"] = bool(valid and board["custody_valid"])
    custody = {b["board"] for b in boards} == set(contracts) and all(
        b["custody_valid"] for b in boards
    )
    bounds: list[Json] = []
    costs = data["costs"] if data["feature_qualified"] else []
    required = [
        "gradient_arithmetic_ns",
        "guard_scans_ns",
        "storage_fsync_ns",
        "cache_extraction_ns",
        "cache_hashing_ns",
        "cache_loading_ns",
    ]
    conditions = set()
    for row in costs:
        before = len(checks)
        condition = row.get("cache_condition", row.get("mode"))
        conditions.add(condition)
        components = row.get("components", {})
        values = dict(
            transaction_ns=row.get("transaction_ns"), **{k: components.get(k) for k in required}
        )
        for key, value in values.items():
            valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
            if key == "transaction_ns":
                valid = valid and value > 0
            if not valid:
                checks.append(
                    operand(
                        "exp8066",
                        Path(data.get("feature_path", str(ROOT / FEATURE))),
                        f"rows[{row['unit']}].{key}",
                        "finite nonnegative measured cost; positive transaction",
                        value,
                    )
                )
        if len(checks) != before:
            continue
        total, arithmetic = values["transaction_ns"], values["gradient_arithmetic_ns"]
        if arithmetic >= total:
            checks.append(
                operand(
                    "exp8066",
                    Path(data.get("feature_path", str(ROOT / FEATURE))),
                    f"rows[{row['unit']}].compatible_arithmetic_ns",
                    "strictly below complete transaction",
                    arithmetic,
                )
            )
            continue
        serial = total - arithmetic
        bounds.append(
            dict(
                unit=row["unit"],
                source=row.get("source"),
                arm=row["arm"],
                seed=row.get("seed"),
                condition=condition,
                numerator=total,
                denominator=serial + arithmetic / 100,
                status="hypothetical",
                exclusion_reason=None,
                complete_workload_ns=total,
                compatible_native_arithmetic_ns=arithmetic,
                compatible_fpga_arithmetic_ns=0,
                compatible_feature_ns=0,
                serial_host_storage_guard_residual_ns=serial,
                measured_components=components,
                device_transfer_ns=None,
                queue_ns=None,
                hypothetical100x_zero_overhead_ceiling=total / (serial + arithmetic / 100),
                infinite_kernel_zero_overhead_ceiling=total / serial,
                current_fpga_fabric_ceiling=1,
                break_even_transfer_queue_budget_ns=arithmetic * 0.99,
                break_even_condition="transfer_ns + queue_ns < budget_ns",
                symbolic_speedup="T / (serial_ns + arithmetic_ns/100 + transfer_ns + queue_ns)",
                claim_scope="conditional hypothetical native arithmetic; no measured acceleration",
            )
        )
    for condition in ("cold", "warm", "all_miss"):
        if condition not in conditions:
            checks.append(
                operand(
                    "exp8066",
                    Path(data.get("feature_path", str(ROOT / FEATURE))),
                    f"completed_cached_rows[{condition}]",
                    "at least one qualified current row",
                    0,
                )
            )
    workload = (
        bool(bounds)
        and data["feature_qualified"]
        and not any(not r["passed"] for r in checks if r["upstream"] == "exp8066")
    )
    if not workload:
        bounds = []
    resource_good = all(r["passed"] for r in checks if r["upstream"] == "local_resource")
    blocked = not custody or not workload or not resource_good
    rows = [
        dict(
            unit="historical_custody",
            source=b["board"],
            arm="read_only",
            seed=None,
            numerator=int(b["custody_valid"]),
            denominator=1,
            status="completed" if b["custody_valid"] else "excluded",
            exclusion_reason=None if b["custody_valid"] else "unqualified_custody",
        )
        for b in boards
    ]
    rows.append(
        dict(
            unit="new_workload_bound",
            source="exp8066",
            arm="conditional_reduction",
            seed=None,
            numerator=int(workload),
            denominator=1,
            status="completed" if workload else "excluded",
            exclusion_reason=None if workload else "missing_or_unqualified_current_costs",
        )
    )
    completed = sum(r["status"] == "completed" for r in rows)
    return dict(
        honest_verdict="complete_blocked_external_resource"
        if not resource_good
        else "complete_blocked_hardware_custody"
        if not custody
        else "complete_blocked_new_workload_bound"
        if blocked
        else "complete_circular_positive_private_fixture"
        if data.get("fixture")
        else "complete_null_conditional_hardware_boundary",
        verdict_class="blocked"
        if blocked
        else "circular_positive"
        if data.get("fixture")
        else "null",
        verifier_is_oracle=bool(data.get("fixture")),
        hardware_custody_ready_score=int(custody),
        current_device_execution_count=0,
        workload_bound_status="qualified" if workload else "blocked",
        acceleration_bounds=bounds,
        board_rows=boards,
        rows=rows,
        gate_check_summary=[r for r in checks if not r["passed"]],
        intended_count=len(rows),
        eligible_count=completed,
        completed_count=completed,
        independent_count=0,
        censored_count=0,
        failed_count=0,
        excluded_count=len(rows) - completed,
        sample_size_budget=dict(
            independent_unit="historical obligations; zero current scientific samples",
            intended_obligations=len(rows),
            model_calls=0,
            device_calls=0,
        ),
        generalized_learning_benefit_score=0,
        numerical_obligations=data["numerical_obligations"],
        missing_cost_components=[
            "device transfer",
            "device queue",
            "source acquisition",
            "external feedback",
        ],
        compatibility_map=dict(
            public_text_normalization_extraction="host operation",
            cache_integrity="host operation",
            small_head_arithmetic="potential native CPU path",
            quadratic_k_le_5="historical KV260 FPGA fabric boundary",
            spline_and_guarded_updates="no automatic mapping to existing quadratic fabric",
            PolarFire="Linux CPU dispatch only",
            GateMate="unchanged physical/JTAG0xffffffff blocker",
            TSU="unqualified access and vendor projections",
            NPU="unqualified access",
        ),
        purchase_recommendation="none; read-only custody and estimates grant no device benefit",
    )
