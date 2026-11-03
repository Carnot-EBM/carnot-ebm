"""REQ-REPORT-8749: archived tests keep their own authority as the queue advances.

Only private fixture processes change historical module bindings. Production
readers and old artifact pins retain their original bytes and behavior.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.reporting import v747_contract_methods, v749_contract_methods as prior
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v749_contract_runner import custody

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8749_v750_archive_bound_contract"
TASK, MILESTONE = "exp8749-archive-bound-contract", "2026.11.750"
CLI = f"scripts/experiments/{NAME}.py"
DESIGN, ACTIVE, STAGED = prior.DESIGN, prior.ACTIVE, prior.STAGED
PIN = "9630ae14f5291cd2fc7a09f110e823b473cdbaf10ead05148f9ea1fb0d0372b6"
V748_PIN = "88a6e9a1c4822285e86c9b089047d4383f6e34d63430bdaff72c33beb979a07a"
ARCHIVES = [
    f"openspec/change-proposals/research-roadmap-v{v}-preserved-20261102.md" for v in [748, 749]
]
OWNED = [
    "python/carnot/reporting/v750_archive_bound.py",
    "python/carnot/reporting/v750_archive_runner.py",
    CLI,
]
TESTS = ["tests/python/test_v750_archive_bound_8749.py"]
HISTORICAL = [
    "tests/python/test_v748_contract_8723.py",
    "tests/python/test_v749_contract_methods_8737.py",
]
HISTORICAL_CLI = "scripts/experiments/experiment_8723_v748_collection_contract.py"
CONSUMERS = [
    "tests/python/test_v748_collection_8723.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
methods = namespace(v747_contract_methods, "v750_authority")
methods.NAME, methods.TASK, methods.MILESTONE = NAME, TASK, MILESTONE
methods.FIRST_ID, methods.VERSION, methods.PREVIOUS_MILESTONE = 8749, "750", "2026.11.749"
methods.PIN = PIN
methods.REQUIRE_PRIORS = True
authority, tasks_digest = methods.authority, prior.digest
freeze_learning = prior.learning


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a supervisor can observe this model-free producer."""
    print(f"[exp8749] phase={phase} completed={completed} pending={pending}", flush=True)


def scoped_ready(validation: Json) -> bool:
    """An unrelated repository diagnostic cannot erase or replace required checks."""
    return (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
        and validation["union"]["passed"]
    )


def mutations(root: Path) -> list[Json]:
    """Changing any field of any complete task must fail the independently pinned contract."""
    root.mkdir(parents=True, exist_ok=True)
    text = (ROOT / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    design = root / DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text(text)
    rows = []
    for index, task in enumerate(tasks):
        for field in task:
            changed = deepcopy(tasks)
            changed[index][field] = "forged"
            (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
            rows.append(dict(task_id=task["id"], field=field, passed=not authority(root)["ready"]))
        progress("full_object_controls", index + 1, len(tasks) - index - 1)
    for mode in ["valid", "count", "order", "table", "digest", "rehashed", "start", "end"]:
        changed, body = deepcopy(tasks), text
        if mode == "count":
            changed.pop()
        elif mode == "order":
            changed.reverse()
        elif mode == "table":
            body = body.replace("| 1 |", "| 99 |", 1)
        elif mode == "digest":
            body = body.replace(PIN, "0" * 64)
        elif mode == "rehashed":
            changed[0]["prompt"] += "forged"
            head, tail = body.split("```json", 1)
            _, end = tail.split("```", 1)
            body = (
                head
                + "```json\n"
                + json.dumps(dict(milestone=MILESTONE, tasks=changed))
                + "\n```"
                + end
            ).replace(PIN, tasks_digest(changed))
        elif mode in {"start", "end"}:
            body = body.replace(f"<!-- V750_TASK_CONTRACT_{mode.upper()} -->", "")
        design.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
        rows.append(dict(mutation=mode, passed=authority(root)["ready"] == (mode == "valid")))
    return rows


def archived_root(root: Path, version: int = 748) -> Path:
    """Supply all historical dependencies privately, without editing the current queue."""
    root.mkdir(parents=True)
    cli = (
        HISTORICAL_CLI
        if version == 748
        else "scripts/experiments/experiment_8737_v749_contract_methods.py"
    )
    for source in ROOT.iterdir():
        if source.name not in {"openspec", "scripts", ACTIVE, STAGED}:
            (root / source.name).symlink_to(source, target_is_directory=source.is_dir())
    for relative in ["openspec", "openspec/change-proposals", "scripts", "scripts/experiments"]:
        target = root / relative
        target.mkdir(exist_ok=True)
        for source in (ROOT / relative).iterdir():
            if source.name in {"change-proposals", "experiments"} or str(
                source.relative_to(ROOT)
            ) in {DESIGN, cli}:
                continue
            (target / source.name).symlink_to(source, target_is_directory=source.is_dir())
    text = (ROOT / ARCHIVES[version - 748]).read_text()
    (root / DESIGN).write_text(text)
    milestone = f"2026.11.{version}"
    tasks = parse_design(text, milestone=milestone)[1]
    assert tasks_digest(tasks) == (V748_PIN if version == 748 else prior.PIN)
    for path in [ACTIVE, STAGED]:
        (root / path).write_text(yaml.safe_dump(dict(milestone=milestone, tasks=tasks)))
    wrapper = root / cli
    wrapper.write_text(
        "from pathlib import Path\nimport sys\n"
        f"sys.path[:0] = {[str(ROOT / 'python'), str(ROOT)]!r}\n"
        "from carnot.reporting.v750_archive_bound import historical_main\n"
        f"raise SystemExit(historical_main(Path({str(root)!r}), {version}))\n"
    )
    return root


def historical_main(root: Path, version: int = 748) -> int:
    """The fixture CLI binds the same archived root as its original pytest parent."""
    from carnot.reporting import v748_collection_contract as historical

    if version == 748:
        historical.m.ROOT = root
        historical.m.CLI = str(root / HISTORICAL_CLI)
        return int(historical.main())
    from carnot.reporting import v749_contract_runner

    prior.ROOT = root
    v749_contract_runner.CLI = str(
        root / "scripts/experiments/experiment_8737_v749_contract_methods.py"
    )
    return int(v749_contract_runner.main())


def historical_suite(root: Path, target: Path, version: int = 748) -> str:
    """Copy original assertion bytes and inject only archived dependency bindings."""
    target.mkdir(parents=True)
    name = HISTORICAL[version - 748]
    shutil.copyfile(ROOT / name, target / Path(name).name)
    binding = "api.m" if version == 748 else "api"
    imported = "v748_collection_contract" if version == 748 else "v749_contract_methods"
    cli_binding = "api.m" if version == 748 else "runner"
    cli = (
        HISTORICAL_CLI
        if version == 748
        else "scripts/experiments/experiment_8737_v749_contract_methods.py"
    )
    (target / "conftest.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        f"from carnot.reporting import {imported} as api\n"
        "from carnot.reporting import v749_contract_runner as runner\n"
        "@pytest.fixture(autouse=True)\n"
        "def archived_dependencies(monkeypatch):\n"
        f'    monkeypatch.setattr({binding}, "ROOT", Path({str(root)!r}))\n'
        f'    monkeypatch.setattr({cli_binding}, "CLI", {str(root / cli)!r})\n'
    )
    return str(target / Path(name).name)


def freeze(raw: Path, root: Path) -> Json:
    """Seal exact sources, original methods and failed historical receipts before checks."""
    learning = custody(freeze_learning, ROOT / prior.original.PROTOCOL)
    protocol = custody(prior.original.protocol, ROOT / prior.original.PROTOCOL)
    old_primary = next((ROOT / "results").glob("experiment_8737_*.json"))
    old = load(reference(old_primary))
    old_validation = load(old["validation_reference"])
    health = next(r for r in old_validation["receipts"] if r["name"] == "full_python_suite")
    failure = next(r for r in old_validation["receipts"] if r["name"] == "child_5")
    refs = [
        reference(ROOT / p)
        for p in ARCHIVES + [DESIGN, ACTIVE, prior.original.PROTOCOL, "ops/exclusion_manifest.yaml"]
    ]
    refs += learning["references"] + [reference(old_primary), old["validation_reference"]]
    preserved_primaries = []
    for number in [8737, 8738, 8743, 8745, 8746, 8747, 8748]:
        path = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        ref = reference(path)
        value = load(ref)
        refs.append(ref)
        preserved_primaries.append(
            dict(
                reference=ref,
                experiment_id=number,
                honest_verdict=value["honest_verdict"],
                verdict_class=value["verdict_class"],
                promoted=False,
            )
        )
    protocols = {
        name: old[name + "_protocol_reference"]
        for name in ["learning", "verifier_transfer", "cost"]
    }
    for ref in protocols.values():
        load(ref)
        refs.append(ref)
    current_health = ROOT / "results/raw" / NAME / "startup/full_python_suite.receipt.json"
    if current_health.is_file():
        refs.append(reference(current_health))
        current_receipt = load(refs[-1])
        refs += [
            dict(
                path=str((ROOT / current_receipt[s + "_path"]).absolute()),
                sha256=current_receipt[s + "_sha256"],
            )
            for s in ["stdout", "stderr"]
        ]
    else:
        current_receipt = None
    for receipt in [health, failure]:
        refs += [
            dict(path=str((ROOT / receipt[s + "_path"]).absolute()), sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    sources = [
        reference(ROOT / p)
        for p in OWNED
        + TESTS
        + HISTORICAL
        + CONSUMERS
        + [
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/experiment_template.py",
        ]
    ]
    contract = authority(root)
    manifest = ROOT / "ops/exclusion_manifest.yaml"
    excluded = yaml.safe_load(manifest.read_bytes())
    retired = {
        int(r["experiment_id"])
        for r in excluded["retired"] + excluded.get("retired_experiments", [])
        if isinstance(r, dict) and str(r.get("experiment_id", "")).isdigit()
    }
    return dict(
        contract=contract,
        learning=learning,
        protocol=protocol,
        references=refs,
        sources=sources,
        preserved_primary_rows=preserved_primaries,
        exclusion_check=dict(
            upstream=TASK,
            **reference(manifest),
            check="not_retired",
            field="retired_experiment_ids",
            operator="not_contains",
            expected=8749,
            observed=sorted(retired),
            passed=8749 not in retired,
        ),
        repository_health_reference=dict(
            receipt=health,
            primary_reference=reference(old_primary),
            validation_reference=old["validation_reference"],
            full_suite_pass=False,
            current_invocation_diagnostic=current_receipt,
        ),
        historical_failure_reference=failure,
        branch_method_references=[
            dict(
                task_id=r["unit_id"],
                task_object=r.get("task_object"),
                method_reference=protocols[
                    "verifier_transfer"
                    if r["unit_id"].startswith(("exp8754-", "exp8755-"))
                    else "cost"
                    if r["unit_id"].startswith(
                        ("exp8756-", "exp8757-", "exp8758-", "exp8759-", "exp8761-")
                    )
                    else "learning"
                ],
                original_numerical_protocol_reference=reference(ROOT / prior.original.PROTOCOL),
                own_input_authentication_required=True,
                upstream_gates=(r.get("task_object") or {}).get("gated_on", []),
                inherited_producer_promoted=False,
                execution_authorized=False,
                gate_check_summary=learning.get("gate_check_summary", [])
                if not r["unit_id"].startswith(
                    (
                        "exp8754-",
                        "exp8755-",
                        "exp8756-",
                        "exp8757-",
                        "exp8758-",
                        "exp8759-",
                        "exp8761-",
                    )
                )
                else [],
            )
            for r in contract["rows"]
        ],
        archived_validation_recipe=dict(
            archive_reference=refs[0],
            canonical_tasks_sha256=V748_PIN,
            original_test_references=[reference(ROOT / p) for p in HISTORICAL],
            bindings=["ROOT", "DESIGN", "ACTIVE", "STAGED"],
            original_assertions_unchanged=True,
        ),
        raw=str(raw),
    )
