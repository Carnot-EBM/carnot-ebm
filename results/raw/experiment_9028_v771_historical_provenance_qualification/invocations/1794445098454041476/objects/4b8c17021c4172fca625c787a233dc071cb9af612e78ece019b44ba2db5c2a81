"""REQ-REPORT-9013-V770: preserve failed history without rerunning frozen science."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v769_branch_evidence as old
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT, LEDGER = old.ROOT, old.LEDGER
BRANCH = "results/experiment_8999_v769_branch_contract.json"
ARCHIVE = "openspec/change-proposals/research-roadmap-v769-preserved-20261111.md"
QUEUE = "openspec/change-proposals/research-roadmap-v769-queue-preserved-20261111.yaml"
PINS = dict(
    old.PINS,
    **{
        ARCHIVE: "sha256:79d6bb2e282041edebc5d00278eeefb5e0dced11ac54fbfbb43e07d595218ad6",
        QUEUE: "sha256:0bc335cc33ac2f7c0618389f4e92eee0921a4fef77b063dcf7ac99da681eb1a9",
        BRANCH: "sha256:6e60393a99efb680765b8d1f6cd5993efdf4717ec0a091da6624a0cf4d9997d7",
        "results/experiment_9000_v769_invocation_provenance.json": "sha256:e92dddafa07eb2b28911b7ece011db3b58607f7e4050e5dd3ee26a2ab2558679",
        "results/experiment_9001_v769_human_source_authority.json": "sha256:9a6d75d73f3bcce3d2f57da61bd679d5f49a70785648a434f418b1fd4ae0256f",
        "results/experiment_9012_v769_capstone.json": "sha256:1993533fc52485731137aebb706bd40326a19ae8eb65e3f0da213e7cd446d5fd",
    },
)
GRAPH, ONLINE, PROTOCOL, ONLINE_PROTOCOL = old.GRAPH, old.ONLINE, old.PROTOCOL, old.ONLINE_PROTOCOL
reference, check, science = old.reference, old.check, old.science


def load(root: Path, name: str) -> Json:
    """Changed upstream bytes are unavailable, rather than silently becoming history."""
    path = root / name
    return json.loads(path.read_bytes()) if reference(path)["sha256"] == PINS[name] else {}


def health(root: Path) -> Json:
    """Retain original rejection receipts and authenticate each failed log independently."""
    rows, checks, refs = [], [], []
    for name in list(PINS)[-3:]:
        value = load(root, name)
        checks.append(
            check(
                root / name,
                "verdict_class",
                "disqualified",
                value.get("verdict_class", "missing_file"),
                "historical_health",
            )
        )
        health = value.get("repository_health", {})
        receipts = health if isinstance(health, list) else health.get("receipts", [])
        for original in receipts:
            receipt = original
            if "receipt_reference" in original:
                ref = original["receipt_reference"]
                path = root / Path(ref["path"]).relative_to(ROOT)
                refs.append(dict(path=ref["path"], sha256=ref["sha256"]))
                checks.append(
                    check(
                        path,
                        "sha256",
                        ref["sha256"],
                        reference(path)["sha256"] or "missing_file",
                        "historical_health",
                    )
                )
                receipt = json.loads(path.read_bytes()) if path.is_file() else {}
            paths = [
                (receipt[k], receipt[k.replace("_path", "_sha256")])
                for k in ["log_path", "stdout_path", "stderr_path"]
                if k in receipt
            ]
            for path_text, pin in paths:
                path = Path(path_text)
                relative = path.relative_to(ROOT) if path.is_absolute() else path
                retained = root / relative
                refs.append(dict(path=str(ROOT / relative), sha256=pin))
                checks.append(
                    check(
                        retained,
                        "sha256",
                        pin,
                        reference(retained)["sha256"] or "missing_file",
                        "historical_health",
                    )
                )
            checks.append(
                check(
                    root / name,
                    "repository_health.passed",
                    False,
                    original.get("passed", "missing_field"),
                    "historical_health",
                )
            )
            rows.append(
                dict(
                    experiment_id=value["experiment_id"],
                    original_verdict=value["honest_verdict"],
                    original_verdict_class=value["verdict_class"],
                    original_receipt=original,
                    log_references=refs[-len(paths) :] if paths else [],
                )
            )
    return dict(
        rows=rows,
        checks=checks,
        references=refs,
        passed=False,
        affects_current_owned_readiness=False,
    )


def operands(root: Path) -> list[Json]:
    """Discover original state, ledger and failed-log bytes before making snapshots."""
    refs = old.operands(root)
    branch = load(root, BRANCH)
    refs += [
        r
        for r in branch.get("raw_shard_hashes", [])
        if r.get("source", {}).get("path") == str(ROOT / LEDGER)
    ]
    refs += health(root)["references"]
    return refs


def frontier(root: Path) -> Json:
    """Read a single saved ledger against Exp8999; an empty delta stays scientific zero."""
    checks: list[Json] = []
    rows: list[Json] = []
    before: Json = {}
    after: Json = {}
    try:
        branch = load(root, BRANCH)
        ref = next(
            r
            for r in branch["raw_shard_hashes"]
            if r.get("source", {}).get("path") == str(ROOT / LEDGER)
        )
        held_path = root / Path(ref["path"]).relative_to(ROOT)
        held = old.arc.read(dict(path=str(held_path), sha256=ref["sha256"]))
        before = branch["arc_frontier"]["frontier_after"]
        if sorted(held["entries"]) != before["ledger_receipt_identities"]:
            raise ValueError("v769_frontier_identity_mismatch")
        live = json.loads((root / LEDGER).read_bytes())
        closed, _ = old.arc.authenticate()
        rows, after = old.arc.delta(held, live, closed)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        checks.append(
            check(root / LEDGER, "new_native_receipts", "authenticated", str(error), "optional_arc")
        )
    support = old.arc_support(rows)
    return dict(
        ready=not checks and support["ready"],
        checks=checks,
        frontier_before=before,
        frontier_after=after,
        new_event_rows=rows,
        support=support,
        baseline_experiment_id=8999,
        cheap_skip_count=int(not support["ready"]),
        panel_rerun_count=0,
        delta_sha256=canonical_hash(rows),
    )
