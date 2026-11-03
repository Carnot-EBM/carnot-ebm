"""REQ-REPORT-V751-PLAN / REQ-REPORT-8763: reuse archived qualification privately."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v750_archive_bound as prior
from carnot.reporting import v750_archive_runner as runner
from carnot.reporting import v751_branch_evidence as arc
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8763_v751_branch_contract",
    "exp8763-branch-contract",
    "2026.11.751",
)
CLI = f"scripts/experiments/{NAME}.py"
PIN = "c698abb7ef71beda98a24ffe4ba914c91b51556db5e13b760e9eb46c154bd4bc"
ARCHIVE = "openspec/change-proposals/research-roadmap-v750-preserved-20261103.md"
ARCHIVE_PIN = "sha256:3b3bd887c283ae194b97a39c038eb0bf514f2f1e75599082558c817fbc2fbda9"
OLD_PRIMARY = "results/experiment_8749_v750_archive_bound_contract.json"
OLD_PIN = "sha256:d007049f2a219af6248c2d6120b47a87e13fd7bc3cf2c24a26f09ca33c301c08"
TESTS = ["tests/python/test_v751_branch_contract_8763.py"]
OWNED = [
    "python/carnot/reporting/v751_branch_contract.py",
    "python/carnot/reporting/v751_branch_evidence.py",
    CLI,
]
scope = namespace(prior, "v751_scope")
runtime = namespace(runner, "v751_runtime")
base_freeze, base_build, base_validate = scope.freeze, runtime.build, runtime.validate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report boundaries and exact counts while bounded children retain their own heartbeat."""
    print(f"[exp8763] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """The independent digest prevents coordinated edits to all mutable task copies."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes())
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        for roadmap in [active, staged]:
            for supplied, expected in zip(roadmap["tasks"], tasks):
                if supplied == dict(expected, prompt=prior.prior.encode(expected["prompt"])):
                    supplied["prompt"] = expected["prompt"]
        result = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8763, count=13
        )
        exact = len(tasks) == len(table) == len(active["tasks"]) == len(staged["tasks"]) == 13
        for i, row in enumerate(result["rows"]):
            full = (
                i < len(tasks)
                and i < len(active["tasks"])
                and i < len(staged["tasks"])
                and tasks[i] == active["tasks"][i] == staged["tasks"][i]
            )
            row["checks"]["full_object"] = full
            row["task_object"] = deepcopy(tasks[i]) if i < len(tasks) else None
            row.update(
                matched=all(row["checks"].values()),
                raw_numerator=sum(row["checks"].values()),
                raw_denominator=len(row["checks"]),
                effective_independent_groups=0,
            )
            row.update(
                absolute_metric=int(row["matched"]), censored=not row["matched"], excluded=False
            )
        ready = (
            exact
            and all(r["matched"] for r in result["rows"])
            and prior.tasks_digest(tasks) == PIN
            and f"Canonical full-task SHA-256: `{PIN}`" in text
            and "<!-- V751_TASK_CONTRACT_END -->" in text
        )
        return dict(
            ready=ready,
            rows=result["rows"],
            visible_table=table,
            references=[
                prior.reference(root / p)
                for p in [DESIGN, ACTIVE] + ([STAGED] if (root / STAGED).is_file() else [])
            ],
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(ready=False, rows=[], visible_table=[], references=[])


def mutations(root: Path) -> list[Json]:
    """Exercise every complete task field and coordinated rehashing using private copies."""
    root.mkdir(parents=True)
    text = (ROOT / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    design = root / DESIGN
    design.parent.mkdir(parents=True)
    rows = []
    for i, task in enumerate(tasks):
        for field in task:
            changed = deepcopy(tasks)
            changed[i][field] = "forged"
            design.write_text(text)
            (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
            rows.append(dict(task_id=task["id"], field=field, passed=not authority(root)["ready"]))
        progress("full_object_controls", i + 1, 12 - i)
    for mode in [
        "valid",
        "encoded",
        "count",
        "order",
        "table",
        "digest",
        "rehashed",
        "start",
        "end",
        "staged",
    ]:
        changed, body = deepcopy(tasks), text
        if mode == "encoded":
            for task in changed:
                task["prompt"] = prior.prior.encode(task["prompt"])
        elif mode == "count":
            changed.pop()
        elif mode == "order":
            changed.reverse()
        elif mode == "table":
            body = body.replace("| 1 |", "| 99 |", 1)
        elif mode == "digest":
            body = body.replace(PIN, "0" * 64)
        elif mode == "rehashed":
            changed[0]["prompt"] += "forged"
            head, tail = text.split("```json", 1)
            _, end = tail.split("```", 1)
            body = (
                head
                + "```json\n"
                + json.dumps(dict(milestone=MILESTONE, tasks=changed))
                + "\n```"
                + end
            ).replace(PIN, prior.tasks_digest(changed))
        elif mode in {"start", "end"}:
            body = body.replace(f"<!-- V751_TASK_CONTRACT_{mode.upper()} -->", "")
        design.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
        if mode == "staged":
            changed[0]["routing"] = "forged"
            (root / STAGED).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
        rows.append(
            dict(mutation=mode, passed=authority(root)["ready"] == (mode in {"valid", "encoded"}))
        )
    return rows


def archived_root(root: Path, version: int = 748) -> Path:
    """Private result copies keep fixture cleanup away from production evidence."""
    facade = root.with_name(root.name + "-sources")
    facade.mkdir(parents=True)
    for source in ROOT.iterdir():
        if source.name != "results":
            (facade / source.name).symlink_to(source, target_is_directory=source.is_dir())
    results = facade / "results"
    results.mkdir()
    for number in [8737, 8738, 8743, 8745, 8746, 8747, 8748]:
        source = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        (results / source.name).write_bytes(source.read_bytes())
    old = json.loads(next(results.glob("experiment_8737_*.json")).read_bytes())
    validation = prior.load(old["validation_reference"])
    for receipt in validation["receipts"]:
        for stream in ["stdout", "stderr"]:
            relative = Path(receipt[stream + "_path"])
            if not relative.is_absolute():
                target = facade / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
    recipe = namespace(prior, "v751_private_archive")
    recipe.ROOT = facade
    return Path(recipe.archived_root(root, version))


def archive750(root: Path) -> Path:
    """Reuse the qualified archive recipe, replacing only private authority dependencies."""
    archived_root(root, 749)
    text = (ROOT / ARCHIVE).read_text()
    (root / DESIGN).write_text(text)
    tasks = parse_design(text, milestone=prior.MILESTONE)[1]
    for name in [ACTIVE, STAGED]:
        (root / name).write_text(yaml.safe_dump(dict(milestone=prior.MILESTONE, tasks=tasks)))
    cli = root / prior.CLI
    cli.unlink()
    cli.write_text(
        "from pathlib import Path\nimport sys\n"
        f"sys.path[:0] = {[str(ROOT / 'python'), str(ROOT)]!r}\n"
        "from carnot.reporting.v751_branch_contract import historical750_main\n"
        f"raise SystemExit(historical750_main(Path({str(root)!r})))\n"
    )
    return root


def historical750_main(root: Path) -> int:
    """The historical child shares its parent's private archive without changing consumers."""
    prior.ROOT = root
    runner.terminal_runner.CLI = str(root / prior.CLI)
    return int(runner.main())


def suite750(root: Path, target: Path) -> str:
    """Retain every original assertion and bind only its private dependency root."""
    target.mkdir(parents=True)
    source = ROOT / prior.TESTS[0]
    copy = target / source.name
    copy.write_bytes(source.read_bytes())
    (target / "conftest.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "from carnot.reporting import v750_archive_bound as a, v750_archive_runner as r\n"
        "@pytest.fixture(autouse=True)\ndef archived_dependencies(monkeypatch):\n"
        f'    monkeypatch.setattr(a, "ROOT", Path({str(root)!r}))\n'
        f'    monkeypatch.setattr(r.terminal_runner, "CLI", {str(root / prior.CLI)!r})\n'
    )
    return str(copy)


def freeze(raw: Path, root: Path) -> Json:
    """Freeze unchanged methods and failed primaries before the current checks run."""
    work = base_freeze(raw, root)
    archive_ref = dict(path=str(ROOT / ARCHIVE), sha256=ARCHIVE_PIN)
    if prior.reference(ROOT / ARCHIVE) != archive_ref:
        raise ValueError("v750_archive_changed")
    old_ref = dict(path=str(ROOT / OLD_PRIMARY), sha256=OLD_PIN)
    old = prior.load(old_ref)
    work["references"] += [archive_ref, old_ref]
    work["preserved_primary_rows"].append(
        dict(
            reference=old_ref,
            experiment_id=8749,
            honest_verdict=old["honest_verdict"],
            verdict_class=old["verdict_class"],
            promoted=False,
        )
    )
    work["exclusion_check"].update(
        expected=8763, passed=8763 not in work["exclusion_check"]["observed"]
    )
    work["branch_method_references"] = [
        dict(
            task_id=r["unit_id"],
            task_object=r.get("task_object"),
            original_numerical_protocol_reference=prior.reference(
                ROOT / prior.prior.original.PROTOCOL
            ),
            execution_authorized=False,
            own_input_authentication_required=True,
            upstream_gates=(r.get("task_object") or {}).get("gated_on", []),
        )
        for r in work["contract"]["rows"]
    ]
    progress("arc_frontier_and_locator_once")
    work["arc"] = arc.freeze(raw)
    work["references"] += work["arc"]["references"]
    health = ROOT / "results/raw" / NAME / "startup/full_python_suite.receipt.json"
    work["repository_health_reference"]["current_invocation_diagnostic"] = (
        prior.reference(health) if health.is_file() else None
    )
    if health.is_file():
        receipt = prior.load(prior.reference(health))
        work["references"] += [prior.reference(health)] + [
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    work["sources"] += [
        prior.reference(ROOT / p)
        for p in prior.OWNED
        + [__file__]
        + [
            "python/carnot/reporting/v751_branch_evidence.py",
            "python/carnot/reporting/collection_contract.py",
            "python/carnot/reporting/v746_retained_evidence.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/roadmap_contract.py",
            "pyproject.toml",
        ]
    ]
    return work


def build(work: Json, validation: Json, output: Path) -> Json:
    """ARC absence blocks its own branch while owned failure suppresses every readiness."""
    value = base_build(work, validation, output)
    value.update(
        experiment_id=8763,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7518763,
        honest_verdict="complete_" + value["verdict_class"] + "_v751_branch_contract",
        intended_count=13,
        original_node_union=validation["union"],
        arc_new_outcomes_available_score=int(
            bool(work["arc"]["ready"]) and bool(value["contract_ready_score"])
        ),
        arc_frontier_reference=work["arc"]["frontier_reference"],
        arc_new_outcome_refs=work["arc"]["new_outcome_refs"],
        arc_gate_check_summary=work["arc"]["gate_check_summary"],
        contract_gate_check_summary=[
            r for r in value["gate_check_summary"] if r["field"] != "learning_inputs"
        ],
        learning_gate_check_summary=work["learning"].get("gate_check_summary", []),
        methodology_note="Bind thirteen complete V751 task objects to an independent canonical digest. Preserve V750 bytes and disqualified Exp8749. Execute the complete unchanged historical assertion union under private archived roots, including actual private-control failure publication and exit under CLI coverage. Freeze unchanged V744 numerical estimands, partitions, thresholds and current learner source identities. One bounded authoritative ARC delta read admits only actual selected/applied outcomes beyond the qualified Exp8760 frontier. Empty ARC evidence gates only Exp8773. Administrative custody establishes no learning benefit or generalization. Zero model loads, live games or policy changes.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind actual invocation and immutable source bytes. Administrative, learning custody and fresh ARC evidence are separate predicates; owned failure suppresses readiness.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def validate(raw: Path, tracer: Any, archived_suites: list[str]) -> Json:
    """Add the entire unchanged V750 suite to the original V748/V749 and consumer union."""
    with TemporaryDirectory(prefix="v751-original750-") as directory:
        root = archive750(Path(directory) / "archive")
        suite = suite750(root, raw / "original_tests_750")
        result: Json = base_validate(raw, tracer, archived_suites + [suite])
    return result


scope.NAME, scope.TASK, scope.MILESTONE, scope.PIN = NAME, TASK, MILESTONE, PIN
scope.OWNED, scope.TESTS, scope.CLI = OWNED + prior.OWNED, TESTS, CLI
scope.authority, scope.freeze, scope.mutations, scope.progress = (
    authority,
    freeze,
    mutations,
    progress,
)
runtime.a = scope
scope.archived_root = archived_root
runtime.build, runtime.validate = build, validate
runtime.terminal_runner.CLI = CLI


def main(argv: list[str] | None = None) -> int:
    """Use the established atomic publisher and fresh replay for the current identity."""
    return int(runtime.main(argv))
