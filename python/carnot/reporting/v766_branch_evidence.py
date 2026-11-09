"""REQ-REPORT-8957: retained graph evidence and new live ARC evidence have separate custody."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting import arc_supervisor_frontier_8720 as arc
from carnot.reporting import v765_capstone_sources as retained
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v755_branch_contract import reference
from carnot.reporting.v763_capstone import references, resolve

Json = dict[str, Any]
ROOT = arc.ROOT
ARCHIVE = "openspec/change-proposals/research-roadmap-v765-preserved-20261109.md"
QUEUE = "openspec/change-proposals/research-roadmap-v765-queue-preserved-20261109.yaml"
PINS = {f"results/{name}": pin for name, pin in retained.PINS.values()}
PINS.update(
    {
        ARCHIVE: "sha256:ca966a330a3f99ec374ed17bc356d5336f0c7db8e3d0518b7b3fb01dabffb9e7",
        QUEUE: "sha256:bc06ee6a979092fa5fdc0e9607aed4788fd1283d23309c28dac0ffa10b2b3b3e",
        "results/experiment_8956_v765_capstone.json": "sha256:dec8ae9e48204127619cc7d20ce56888c81d2d4836c98f2f22acdbbcf52f9871",
    }
)
ARC_PRIMARY = "results/experiment_8720_v747_arc_supervisor_frontier.json"
ARC_PIN = "sha256:bcef7715e5181dd49b89cf019a184465783803a6ff7e51b3d5a79bb182a308f7"
GRAPH_GATES = {
    8946: "matched_source_ready_score",
    8948: "depth_source_ready_score",
    8949: "shard_ready_score",
    8950: "shard_ready_score",
    8951: "corpus_ready_score",
}


def diagnostic(
    path: Path,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    check: str = "rejected_producer",
) -> Json:
    """Record the precise failed operand rather than treating absence as a measured zero."""
    return dict(
        **reference(path),
        upstream=upstream,
        check=check,
        artifact_field=field,
        expected=expected,
        observed=observed,
    )


def history(root: Path = ROOT) -> Json:
    """Authenticate each graph producer directly; a quarantined diagnostic admits no science."""
    refs, failures, values = [], [], {}
    path, field = root / ARCHIVE, "sha256"
    expected: Any = "pinned qualified original"
    observed: Any = "missing_file"
    try:
        for name, pin in PINS.items():
            path, field = root / name, "sha256"
            expected, observed = pin, reference(path)["sha256"]
            if observed != expected:
                raise ValueError("original_bytes_changed")
            refs.append(reference(path))
            if path.suffix == ".json":
                identity = int(path.name.split("_")[1])
                if identity in {8945, 8956, *GRAPH_GATES}:
                    values[identity] = json.loads(path.read_bytes())
        graph = {n: values[n] for n in GRAPH_GATES}
        catalog = {
            r["sha256"]: r
            for v in graph.values()
            for r in references(v)
            if isinstance(r, dict) and r.get("sha256") and Path(r["path"]).is_file()
        }
        for n, value in graph.items():
            path, field = root / "results" / retained.PINS[n][0], GRAPH_GATES[n]
            expected, observed = 1, value[field]
            if not (
                type(value[field]) is int
                and value[field] == 1
                and value["flagged_adversarial"] is False
                and value["required_checks_passed"] is True
                and value["owned_statement_coverage"]["complete"] is True
                and all(r["passed"] for r in value["validation_receipts"])
            ):
                raise ValueError("unqualified_graph_producer")
            for ref in references(value):
                refs.append(resolve(ref, catalog))
        quarantine = values[8945]
        field = "flagged_adversarial"
        path = root / "results" / retained.PINS[8945][0]
        expected, observed = True, quarantine[field]
        if quarantine[field] is not True:
            raise ValueError("quarantine_changed")
        path, field = root / "results" / retained.PINS[8951][0], "graph_corpus_counts"
        corpus = {
            k: values[8951][k]
            for k in [
                "independent_count",
                "independent_error_count",
                "semantic_error_count",
                "out_of_graph_error_count",
                "format_error_count",
            ]
        }
        expected = dict(
            independent_count=320,
            independent_error_count=233,
            semantic_error_count=227,
            out_of_graph_error_count=6,
            format_error_count=0,
        )
        observed = corpus
        if corpus != expected:
            raise ValueError("retained_corpus_census")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            diagnostic(
                path,
                "v765_retained_producers",
                field,
                expected,
                "missing_file"
                if not path.is_file()
                else "missing_field"
                if isinstance(error, KeyError)
                else observed,
            )
        )
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        outcome_rows=values.get(8956, {}).get("outcome_rows", []),
        graph_corpus=corpus if not failures else {},
        quarantine={
            k: values.get(8945, {}).get(k)
            for k in [
                "flagged_adversarial",
                "inference_substrate",
                "inference_substrate_class",
                "execution_venue",
            ]
        },
        scientific_results_admitted=False,
    )


def arc_support(rows: list[Json]) -> Json:
    """Count unique completed episodes and require both arms within every held family."""
    cells: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    episodes = set()
    for row in rows:
        if not (
            row["status"] == "completed"
            and row["selected"] is True
            and row["applied"] is True
            and row["arm"] in ARM_ORDER
        ):
            continue
        episode = canonical_hash([row["invocation_id"], row["game"], row["seed"]])
        cells[row["game"]][row["arm"]].add(episode)
        episodes.add(episode)
    arms = sorted({a for family in cells.values() for a in family})
    support = [
        dict(
            held_family=game,
            arm=arm,
            held_episode_count=len(cells[game].get(arm, set())),
            training_episode_count=sum(
                len(other.get(arm, set())) for key, other in cells.items() if key != game
            ),
        )
        for game in sorted(cells)
        for arm in arms
    ]
    ready = (
        len(episodes) >= 30
        and len(cells) >= 3
        and len(arms) == 2
        and all(r["held_episode_count"] and r["training_episode_count"] for r in support)
    )
    return dict(
        ready=bool(ready),
        new_uncensored_episode_count=len(episodes),
        game_family_count=len(cells),
        compared_arms=arms,
        support_rows=support,
    )


def arc_evidence(root: Path = ROOT, ledger_path: Path | None = None) -> Json:
    """Open only native receipt deltas beyond the last qualified identity frontier."""
    refs, failures, rows, before, after = [], [], [], {}, {}
    path, field = root / ARC_PRIMARY, "qualified_terminal"
    try:
        producer, qualified_refs = arc.qualified(path, ARC_PIN)
        refs += qualified_refs
        path, field = root / arc.BASELINE, "retained_ledger_bytes"
        baseline = arc.read(dict(path=str(path), sha256=arc.LEDGER_PIN))
        refs.append(reference(path))
        before = producer["frontier_after"]
        if before["ledger_receipt_identities"] != sorted(baseline["entries"]):
            raise ValueError("qualified_frontier_identity_mismatch")
        path, field = ledger_path or root.joinpath(*arc.DEFAULT_LEDGER_PARTS), "ledger.entries"
        ledger = json.loads(path.read_bytes())
        refs.append(reference(path))
        closed, closed_refs = arc.authenticate()
        refs += closed_refs
        rows, after = arc.delta(baseline, ledger, closed)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        failures.append(
            diagnostic(
                path,
                "exp8720-arc-supervisor-frontier",
                field,
                "authenticated new live selected/applied receipts",
                "missing_file"
                if not path.is_file()
                else "missing_field"
                if isinstance(error, KeyError)
                else str(error),
            )
        )
    support = arc_support(rows)
    if not failures and not support["ready"]:
        failures.append(
            diagnostic(
                path,
                "live_supervisor_receipts",
                "new_uncensored_episode_count",
                ">=30; >=3 families; both arms in each held family",
                support["new_uncensored_episode_count"],
                "observed_zero"
                if not support["new_uncensored_episode_count"]
                else "insufficient_support",
            )
        )
    return dict(
        ready=not failures and support["ready"],
        references=refs,
        failures=failures,
        frontier_before=before,
        frontier_after=after,
        new_event_rows=rows,
        support=support,
        baseline_primary=dict(path=str(root / ARC_PRIMARY), sha256=ARC_PIN),
        baseline_ledger=dict(path=str(root / arc.BASELINE), sha256=arc.LEDGER_PIN),
    )
