"""Locate real producer receipts before counting new observations. REQ-REPORT-8215.

The producer's existing ledger and output directory supply paths. A locator seals
those operands; it never creates a replacement state file. Native receipt IDs
bind game, seed and measured fields, while recorded clocks bound the delta.
"""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.agentic.arc_supervisor_refinement import (
    DEFAULT_LEDGER_PARTS,
    EVAL_RUNS_DIR_NAME,
    LEDGER_SCHEMA,
    _evidence_from_row,
    classify_receipt,
    extract_rows,
    receipt_id_for_row,
    scan_inputs,
)
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting.arc_supervisor_v688_receipts import event_order
from carnot.reporting.arc_supervisor_v689_delta import operand
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.arc.authority_locator.v1"
TASK_ID = "exp8215-arc-authoritative-frontier"
PRIOR = ROOT / "results/experiment_8189_v707_arc_supervisor_frontier.json"
INVENTORY = PRIOR.parent / "raw" / PRIOR.stem / "receipt_inventory.json"
TERMINAL = INVENTORY.parent / "terminal_reports.json"
PINNED_PRIOR = "sha256:bc298ca8b5e58f082e8cb714b8ffe1cf808432cbb4f54e5490fdc6f5468d75c0"
PINNED_INVENTORY = "sha256:f0ac50659c9c568823d68caac31c5031ae7dc3efd2bc812c87eccc6bc4f2338b"
PINNED_TERMINAL = "sha256:226f79a1fa1addb5968fffb278aef8b3f02b1961906d13186e848c0c070fb207"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
PINNED_REGISTRY = "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947"
PRODUCERS = [
    "python/carnot/agentic/arc_supervisor_refinement.py",
    "python/carnot/agentic/arc_trajectory_supervisor.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "scripts/arc_leaderboard_eval.py",
    "scripts/arc_scored_path_lever_harness.py",
    "scripts/arc_bench.py",
]


def producer_hashes() -> dict[str, str]:
    """Seal the current locator contract, without relabeling historical execution."""
    return {str(ROOT / p): sha256_file(ROOT / p) for p in PRODUCERS}


def reference(path: Path) -> dict[str, Any]:
    """Absence remains a literal missing operand, never an empty measurement."""
    return dict(path=str(path), sha256=sha256_file(path) if path.is_file() else None)


def discover() -> dict[str, Any]:
    """Use only names exported by producers and source pointers in their ledger."""
    ledger_path = ROOT.joinpath(*DEFAULT_LEDGER_PARTS)
    try:
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
        ledger = ledger if isinstance(ledger, dict) else {}
    except ValueError:
        ledger = {}
    sources: dict[str, list[str]] = {}
    entries = ledger.get("entries", {})
    entries = entries if isinstance(entries, dict) else {}
    for identity, entry in entries.items():
        sources.setdefault(entry["source"], []).append(identity)
    optional = []
    native = ROOT / "results" / EVAL_RUNS_DIR_NAME
    if native.is_dir():
        for path in scan_inputs([native]):
            document = json.loads(path.read_text())
            if document.get("policy") == "explorer":
                optional.append(
                    dict(
                        reference(path),
                        required=False,
                        disposition="explorer_not_live_supervisor_authority",
                    )
                )
                continue
            sources.setdefault(str(path), [])
    return dict(
        schema=SCHEMA,
        task_id=TASK_ID,
        ledger=reference(ledger_path),
        ledger_definition=list(DEFAULT_LEDGER_PARTS),
        producer_code_hashes=producer_hashes(),
        prior=dict(path=str(PRIOR), sha256=PINNED_PRIOR),
        inventory=dict(path=str(INVENTORY), sha256=PINNED_INVENTORY),
        terminal=dict(path=str(TERMINAL), sha256=PINNED_TERMINAL),
        sources=[
            dict(
                reference(Path(p)),
                receipt_ids=ids,
                kind="harness" if Path(p).name == "rows.json" else "eval",
            )
            for p, ids in sorted(sources.items())
        ],
        optional_dispositions=optional,
    )


def reduction(rows: list[dict[str, Any]], prior: dict[str, Any]) -> dict[str, Any]:
    """Describe curated arms using games as groups; no causal claim follows."""
    eligible = [r for r in rows if r["status"] in {"completed", "censored"}]
    arms: dict[str, Any] = {}
    games: dict[str, Any] = {}
    for row in eligible:
        for cells, label in ((arms, row["arm"]), (games, row["game"])):
            cell = cells.setdefault(
                label,
                dict(fired=0, helped=0, censored=0, resolved_by_levelup=0, actions_to_levelup=[]),
            )
            cell["fired"] += 1
            cell["helped"] += int(row["helped"])
            cell["resolved_by_levelup"] += int(row["resolved_by_levelup"])
            cell["censored"] += int(row["status"] == "censored")
            cell["actions_to_levelup"].append(row["actions_to_levelup"])
    proposals = []
    for game in sorted(games):
        training = [r for r in eligible if r["game"] != game and r["status"] == "completed"]
        support = {
            a: dict(
                fired=sum(r["arm"] == a for r in training),
                helped=sum(r["arm"] == a and r["helped"] for r in training),
            )
            for a in ARM_ORDER
            if any(r["arm"] == a for r in training)
        }
        selected = (
            max(support, key=lambda a: support[a]["helped"] / support[a]["fired"])
            if support
            else None
        )
        proposals.append(
            dict(
                held_game=game,
                selected_arm=selected,
                training_support=support,
                training_games=sorted({r["game"] for r in training}),
                held_support=sum(r["game"] == game for r in eligible),
                claim="descriptive_only_separate_live_comparison_required",
            )
        )
    statuses = Counter(r["status"] for r in rows)
    return dict(
        rows=rows,
        new_outcome_count=len(eligible),
        per_game_results=games,
        per_arm_results=arms,
        arm_selection_proposal=proposals,
        intended_count=len(rows),
        completed_count=statuses["completed"],
        censored_count=statuses["censored"],
        failed_count=statuses["failed"],
        excluded_count=statuses["excluded"],
        independent_count=len(games),
        prior_frontier=prior,
        current_frontier=dict(
            receipt_ids=sorted(
                set(prior.get("receipt_ids", [])) | {r["receipt_id"] for r in eligible}
            ),
            event_ids=sorted(set(prior.get("event_ids", [])) | {r["event_id"] for r in eligible}),
            legacy_receipt_inventory=prior.get("receipt_inventory", {}),
        ),
    )


def read_delta(
    locator_path: Path,
    *,
    precondition_failures: list[dict[str, Any]] | None = None,
    current_date: str = "20261006",
) -> dict[str, Any]:
    """Authenticate every required operand before the bounded outcome reduction.

    The locator is an explicit invocation input. Its source hashes bind native
    rows; historical ledger IDs cross-check copied game/seed/redirect fields.
    Missing scientific clocks exclude old observations instead of using mtimes.
    """
    started = time.monotonic()
    checks: list[dict[str, Any]] = list(precondition_failures or [])
    hashes: dict[str, str] = {}

    def check(path: Path, field: str, expected: Any, observed: Any, digest: str | None) -> None:
        checks.append(
            dict(operand(path, field, expected, observed, digest), passed=expected == observed)
        )

    def load(ref: dict[str, Any]) -> Any:
        path = Path(ref["path"])
        digest = sha256_file(path) if path.is_file() else None
        check(path, "sha256", ref["sha256"], digest, digest)
        check(path, "is_file", True, path.is_file(), digest)
        if digest is None or digest != ref["sha256"]:
            return {}
        hashes[str(path)] = digest
        try:
            if path.stat().st_size > 33554432:
                raise ValueError("source_exceeds_32MiB")
            document = json.loads(path.read_text())
            check(path, "json_object", True, isinstance(document, dict), digest)
            return document if isinstance(document, dict) else {}
        except (ValueError, OSError) as error:
            check(path, "bounded_json", True, str(error), digest)
            return {}

    registry_hash = sha256_file(REGISTRY) if REGISTRY.is_file() else None
    check(REGISTRY, "registry_sha256", PINNED_REGISTRY, registry_hash, registry_hash)
    registry = yaml.safe_load(REGISTRY.read_text()) if registry_hash == PINNED_REGISTRY else {}
    games = registry.get("games", {})
    games = {r["game"]: r for r in games} if isinstance(games, list) else games
    if registry_hash:
        hashes[str(REGISTRY)] = registry_hash
    locator = load(reference(locator_path))
    check(locator_path, "schema", SCHEMA, locator.get("schema"), hashes.get(str(locator_path)))
    check(locator_path, "task_id", TASK_ID, locator.get("task_id"), hashes.get(str(locator_path)))
    check(
        locator_path,
        "ledger_definition",
        list(DEFAULT_LEDGER_PARTS),
        locator.get("ledger_definition"),
        hashes.get(str(locator_path)),
    )
    check(
        locator_path,
        "producer_paths",
        sorted(producer_hashes()),
        sorted(locator.get("producer_code_hashes", {})),
        hashes.get(str(locator_path)),
    )
    for label, expected in locator.get("producer_code_hashes", {}).items():
        path = Path(label)
        actual = sha256_file(path) if path.is_file() else None
        check(path, "producer_code_sha256", expected, actual, actual)
        if actual:
            hashes[label] = actual
    documents = {
        k: load(locator[k]) for k in ("ledger", "prior", "inventory", "terminal") if k in locator
    }
    ledger = documents.get("ledger", {})
    previous = documents.get("prior", {})
    prior = documents.get("inventory", {})
    check(locator_path, "ledger.schema", LEDGER_SCHEMA, ledger.get("schema"), None)
    check(
        locator_path, "ledger.entries_object", True, isinstance(ledger.get("entries"), dict), None
    )
    for key, expected in [
        ("experiment_id", 8189),
        ("supervisor_reader_ready_score", 1),
        ("required_checks_passed", True),
        ("verdict_class", "null"),
    ]:
        check(
            Path(locator.get("prior", {}).get("path", str(locator_path))),
            key,
            expected,
            previous.get(key),
            locator.get("prior", {}).get("sha256"),
        )
    terminal = documents.get("terminal", {})
    check(
        locator_path,
        "terminal.primary_sha256",
        locator.get("prior", {}).get("sha256"),
        terminal.get("primary_sha256"),
        None,
    )
    check(
        locator_path, "terminal.report.passed", True, terminal.get("report", {}).get("passed"), None
    )
    refs = locator.get("sources", [])
    check(locator_path, "source_count_at_most_64", True, len(refs) <= 64, None)
    entries = ledger.get("entries", {})
    entries = entries if isinstance(entries, dict) else {}
    check(
        locator_path,
        "all_ledger_sources_named",
        True,
        {e["source"] for e in entries.values()} <= {r["path"] for r in refs},
        None,
    )
    sources = [(ref, load(ref)) for ref in refs[:64]]
    rows: list[dict[str, Any]] = []
    seen_receipts = set(prior.get("receipt_ids", []))
    seen_events = set(prior.get("event_ids", []))
    for index, (ref, doc) in enumerate(sources):
        print(
            f"[exp8215] phase=authority_source completed={index + 1} pending={len(sources) - index - 1}",
            flush=True,
        )
        if time.monotonic() - started > 120:
            check(Path(ref["path"]), "scan_deadline_s", 120, "expired", ref["sha256"])
            break
        native = (
            ref.get("kind") == "eval"
            and isinstance(doc, dict)
            and doc.get("experiment") == "arc_leaderboard_eval"
            and doc.get("policy") == "e3"
        )
        harness = (
            ref.get("kind") == "harness"
            and isinstance(doc, dict)
            and isinstance(doc.get("rows"), list)
        )
        check(Path(ref["path"]), "live_producer_schema", True, native or harness, ref["sha256"])
        episodes = extract_rows(doc)
        check(
            Path(ref["path"]),
            "redirect_count_at_most_4096",
            True,
            sum(len(r.get("trajectory_supervisor", {}).get("redirects", [])) for r in episodes)
            <= 4096,
            ref["sha256"],
        )
        check(
            Path(ref["path"]),
            "ledger_receipt_set",
            sorted(k for k, e in entries.items() if e["source"] == ref["path"]),
            sorted(ref.get("receipt_ids", [])),
            ref["sha256"],
        )
        by_id = {receipt_id_for_row(r): r for r in episodes}
        for identity in ref.get("receipt_ids", []):
            expected = (
                ledger.get("entries", {}).get(identity, {})
                if isinstance(ledger.get("entries"), dict)
                else {}
            )
            observed = (
                _evidence_from_row(by_id[identity], ref["path"])
                if identity in by_id and classify_receipt(by_id[identity]) == "applied"
                else {}
            )
            valid = (
                bool(expected)
                and expected.get("receipt_id") == identity
                and all(
                    observed.get(k) == v
                    for k, v in expected.items()
                    if k in observed and k != "redirects"
                )
                and all(
                    all(actual.get(k) == v for k, v in old.items())
                    for actual, old in zip(
                        observed.get("redirects", []), expected.get("redirects", []), strict=False
                    )
                )
                and len(observed.get("redirects", [])) == len(expected.get("redirects", []))
            )
            check(Path(ref["path"]), "receipt_identity:" + identity, True, valid, ref["sha256"])
    failures = [r for r in checks if not r["passed"]]
    if failures:
        rows = [
            dict(
                r,
                status="failed",
                reason="authority_precondition",
                condition="required_authority_operand",
                metric="operand_authenticated",
                numerator=0,
                denominator=1,
            )
            for r in failures
        ]
    if not failures:
        for index, (ref, doc) in enumerate(sources):
            print(
                f"[exp8215] phase=redirect_source completed={index} pending={len(sources) - index}",
                flush=True,
            )
            for episode in extract_rows(doc):
                identity = receipt_id_for_row(episode)
                receipt = episode.get("trajectory_supervisor", {})
                if classify_receipt(episode) != "applied":
                    continue
                chronology = event_order(
                    dict(episode, event_timestamp=episode.get("finished_at")),
                    dict(event_timestamp=previous.get("finished_at")),
                    current_date,
                )
                for redirect in receipt["redirects"]:
                    event_id = canonical_hash(
                        dict(game=episode.get("game"), seed=episode.get("seed"), redirect=redirect)
                    )
                    valid = (
                        episode.get("game") in games
                        and type(episode.get("seed")) is int
                        and redirect.get("arm") in ARM_ORDER
                        and type(redirect.get("resolved_by_levelup")) is bool
                        and (
                            type(redirect.get("actions_to_levelup")) is int
                            and redirect["actions_to_levelup"] > 0
                            if redirect.get("resolved_by_levelup")
                            else redirect.get("actions_to_levelup") is None
                        )
                    )
                    reason = (
                        "outcome_schema"
                        if not valid
                        else "duplicate_receipt"
                        if identity in seen_receipts or event_id in seen_events
                        else "chronology_absent"
                        if chronology == "unknown"
                        else "before_frontier"
                        if chronology != "after_cutoff"
                        else None
                    )
                    status = (
                        "excluded"
                        if reason
                        else "completed"
                        if redirect["resolved_by_levelup"]
                        else "censored"
                    )
                    rows.append(
                        dict(
                            redirect,
                            game=episode.get("game"),
                            seed=episode.get("seed"),
                            receipt_id=identity,
                            event_id=event_id,
                            status=status,
                            reason=reason,
                            source_path=ref["path"],
                            source_sha256=ref["sha256"],
                            entrypoint="E3AgentPolicy/make_carnot_agent",
                            chronology=chronology,
                            event_timestamp=episode.get("finished_at"),
                            fired=True,
                            helped=redirect.get("resolved_by_levelup") is True,
                            metric="resolved_by_levelup",
                            numerator=int(redirect.get("resolved_by_levelup") is True),
                            denominator=1,
                            condition="observational_curated_redirect",
                            solve_provenance="live_agent_self_discovery",
                            stagnations_unredirected=receipt.get("stagnations_unredirected"),
                            unredirected_windows=receipt.get("unredirected_windows", []),
                        )
                    )
                    seen_events.add(event_id)
                seen_receipts.add(identity)
    value = reduction(rows, prior)
    return dict(
        value,
        failures=failures,
        authority_locator_rows=checks,
        source_artifact_hashes=hashes,
        historical_receipt_count=len(ledger["entries"])
        if isinstance(ledger.get("entries"), dict)
        else None,
        source_count=len(sources),
        prior_finished_at=previous.get("finished_at"),
        registry_precheck={g: r.get("levels_reproduced", 0) for g, r in games.items()},
        optional_dispositions=locator.get("optional_dispositions", []),
    )
