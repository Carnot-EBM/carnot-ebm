"""REQ-REPORT-8044: retain original bytes without assigning verdicts to absent work."""

import json
from pathlib import Path
import re
from typing import Any

from carnot.reporting import v685_authority_lifecycle as lifecycle
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
NAMED = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v696_capstone.py",
    "results/experiment_8031_v696_contract_methods.json",
    "results/experiment_8043_v696_capstone.json",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "openspec/change-proposals/research-roadmap-v696-preserved-20261002.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/changelog.md",
]


def operand(path: Path, upstream: str, field: str, expected: Any, observed: Any) -> Json:
    """Name a missing operand explicitly so it cannot look like a measured zero."""
    return dict(
        upstream_id=upstream,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        check_name=field,
        artifact_field=field,
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def history(root: Path, tasks: list[Json], durable: Path) -> Json:
    """Freeze final primaries, their sidecars, actual skips and earlier log claims."""
    refs: list[Json] = []
    rows: list[Json] = []
    gates: list[Json] = []

    def save(path: Path, role: str) -> Json:
        frozen = lifecycle._snapshot(
            path, path.read_bytes() if path.is_file() else None, durable, role
        )
        ref = dict(
            path=frozen.get("snapshot_path", str(path)),
            sha256=frozen["sha256"],
            original_path=str(path),
            role=role,
        )
        if frozen["exists"]:
            refs.append(ref)
        return ref

    for name in NAMED + [
        ".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")
    ]:
        path = root / name
        gates.append(operand(path, "local_preconditions", "resource_exists", True, path.is_file()))
        if path.is_file() and not name.startswith(".venv/"):
            save(path, "input_" + name.replace("/", "_"))
    authority = root / "results/experiment_8031_v696_contract_methods.json"
    old = json.loads(authority.read_text()) if authority.is_file() else {}
    invocation = old.get("authority_snapshots", {}).get("active", {}).get("snapshot_path")
    original_tasks = lifecycle._read(Path(invocation))[1].get("tasks", []) if invocation else []
    original_paths = {
        int(t["id"][3:].split("-")[0]): root / t["deliverable"] for t in original_tasks
    }
    for role, snapshot in old.get("authority_snapshots", {}).items():
        if snapshot["exists"]:
            p = Path(snapshot["snapshot_path"])
            ref = save(p, "v696_" + role)
            gates.append(
                operand(
                    p,
                    "exp8031-contract-methods",
                    "original_authority_sha256",
                    snapshot["sha256"],
                    ref["sha256"],
                )
            )
    prior_ids = {
        int(p["experiment_id"][3:].split("-")[0]) for t in tasks for p in t["prior_failures"]
    }
    by_id: dict[int, Json] = {}
    for n in sorted(set(range(8031, 8044)) | prior_ids):
        candidates = sorted((root / "results").glob(f"experiment_{n}_*.json"))
        path = (
            candidates[0]
            if len(candidates) == 1
            else root / "results" / f"absent_experiment_{n}.json"
        )
        if n in (8035, 8036, 8037) and not candidates:
            path = original_paths.get(n, path)
            rows.append(
                dict(
                    experiment_id=n,
                    evidence_kind="absent_evidence",
                    original_path=str(path),
                    sha256=None,
                    honest_verdict=None,
                    verdict_class=None,
                    measured=False,
                )
            )
            continue
        gates.append(operand(path, f"exp{n}", "unique_primary_or_skip", 1, len(candidates)))
        if len(candidates) != 1:
            continue
        ref = save(path, f"primary_{n}")
        value = json.loads(path.read_text())
        skip = value.get("honest_verdict") == "blocked_gate_check_failed"
        row = dict(
            experiment_id=n,
            evidence_kind="actual_skip_receipt" if skip else "final_primary",
            original_path=str(path),
            sha256=ref["sha256"],
            honest_verdict=value.get("honest_verdict"),
            verdict_class=value.get("verdict_class"),
            measured=not skip,
            gate_check_summary=value.get("gate_check_summary", []),
        )
        rows.append(row)
        by_id[n] = row
        if not skip:
            sidecar = Path(
                value.get("terminal_validation_sidecar_path", str(root / "absent_sidecar"))
            )
            side_ref = save(sidecar, f"terminal_{n}")
            terminal = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            binding = terminal.get("publication", terminal)
            gates.append(
                operand(
                    sidecar,
                    f"exp{n}",
                    "terminal_primary_sha256",
                    ref["sha256"],
                    binding.get("primary_sha256"),
                )
            )
            row["terminal_sidecar"] = side_ref
            validator = binding.get("sidecar_path")
            if validator:
                vpath = Path(validator)
                vref = save(vpath, f"validator_{n}")
                report = json.loads(vpath.read_text()) if vpath.is_file() else {}
                gates.append(
                    operand(
                        vpath,
                        f"exp{n}",
                        "validator_primary_sha256",
                        ref["sha256"],
                        report.get("primary_sha256"),
                    )
                )
                row["validator_sidecar"] = vref
    changelog = root / "ops/changelog.md"
    for line_number, line in enumerate(
        changelog.read_text().splitlines() if changelog.is_file() else [], 1
    ):
        match = re.search(
            r"honest_verdict=(complete_disqualified_\w+); results/experiment_(8032|8038)_", line
        )
        if match:
            rows.append(
                dict(
                    experiment_id=int(match[2]),
                    evidence_kind="intermediate_log",
                    honest_verdict=match[1],
                    verdict_class="disqualified",
                    original_path=str(changelog),
                    sha256=sha256_file(changelog),
                    line_number=line_number,
                    original_line=line,
                    measured=False,
                    superseded_by_final_primary=True,
                )
            )
    for n in (8032, 8038):
        gates.append(
            operand(
                changelog,
                f"exp{n}",
                "intermediate_disqualification_preserved",
                True,
                any(
                    r["experiment_id"] == n and r["evidence_kind"] == "intermediate_log"
                    for r in rows
                ),
            )
        )
    failure_rows = []
    for task in tasks:
        for prior in task["prior_failures"]:
            n = int(prior["experiment_id"][3:].split("-")[0])
            observed = by_id.get(n, {})
            gate = operand(
                Path(observed.get("original_path", str(root / "absent"))),
                prior["experiment_id"],
                "honest_verdict",
                prior["verdict"],
                observed.get("honest_verdict"),
            )
            gates.append(gate)
            failure_rows.append(dict(task_id=task["id"], prior=prior, **gate))
    return dict(
        ready=all(g["passed"] for g in gates),
        hashes=refs,
        rows=[],
        budget={},
        gate_check_summary=[g for g in gates if not g["passed"]],
        preconditions_checked=gates,
        historical_disposition_rows=rows,
        prior_failure_checks=failure_rows,
    )


def replay_history(rows: list[Json], refs: list[Json]) -> bool:
    """Reopen saved primaries and log lines so reported history is independently checked."""
    sources = {r["original_path"]: Path(r["path"]) for r in refs if "original_path" in r}
    for row in rows:
        kind = row["evidence_kind"]
        if kind == "absent_evidence":
            if (
                row["honest_verdict"] is not None
                or row["verdict_class"] is not None
                or row["measured"]
            ):
                return False
            continue
        path = sources[row["original_path"]]
        if kind == "intermediate_log":
            if path.read_text().splitlines()[row["line_number"] - 1] != row["original_line"]:
                return False
            continue
        original = json.loads(path.read_text())
        if (
            original.get("honest_verdict") != row["honest_verdict"]
            or original.get("verdict_class") != row["verdict_class"]
        ):
            return False
    return True
