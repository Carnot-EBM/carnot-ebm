"""REQ-REPORT-8077: publish an independent null or supported projected audit.

The qualified publication driver supplies bounded children and validators. Only
this audit's worker and cold equations determine its scientific evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import coverage as statement_coverage

from carnot import experiment_8076_v699_projected_online_learning as driver
from carnot import experiment_8064_v698_fresh_feedback_learning as historical
from carnot import experiment_8065_v698_fresh_learning_audit as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import evidence_features_7980 as features
from carnot.verify import fresh_learning_audit_8065 as independent
from carnot.verify import learning_retention_audit_8026 as math
from carnot.verify import projected_learning_audit_8077 as a

Json = dict[str, Any]
ROOT = driver.ROOT
NAME = "experiment_8077_v699_projected_learning_audit"
TASK = "exp8077-projected-learning-audit"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_projected_learning_audit_8077.py"
OWNED = [MODULE, "python/carnot/verify/projected_learning_audit_8077.py", CLI]
INPUTS = [
    *driver.INPUTS,
    prior.MODULE,
    prior.OWNED[2],
    "python/carnot/reporting/learning_recovery_8052.py",
    "results/" + prior.NAME + ".json",
    "results/" + driver.NAME + ".json",
]
BASE_BUILD = driver.build
BASE_MANIFEST = driver.manifest


def manifest(private: Path) -> list[Json]:
    """Keep the established validation contract, with an explicit audit deadline."""
    commands = BASE_MANIFEST(private)
    return commands


def recovery_state(expected: list[Any], prefix: list[Any], actual: list[Any]) -> Json:
    """Compare causal restart state without assuming the V698 candidate schema."""
    math.equal("recovery_prefix", expected[: len(prefix)], prefix)
    math.equal("recovery_events", expected, actual)
    issued = {r["slot"] for k, r in prefix if k == "issue"}
    released = {r["slot"] for k, r in prefix if k == "release"}
    consumed = [r["slot"] for k, r in prefix if k == "consume"]
    candidates = [r for k, r in prefix if k == "candidate"]
    pending = [
        r["slot"]
        for k, r in expected
        if k == "consume"
        and candidates
        and r["candidate_slot"] == candidates[-1]["slot"]
        and r["slot"] not in consumed
    ]
    next_update = next((r for k, r in actual[len(prefix) :] if k == "commit"), None)
    return dict(
        pending_ids=sorted(issued - released),
        consumed_ids=consumed,
        pending_candidate_ids=pending,
        prefix_hash=canonical_hash(prefix),
        next_update_hash=canonical_hash(next_update),
        predictions_hash=canonical_hash([r for k, r in actual if k == "issue"]),
    )


def recovery(data: Json, raw: Path, *, budget_s: float = 900) -> list[Json]:
    """Kill real private learner children and compare durable prefixes and labels."""
    deadline = time.monotonic() + budget_s
    raw.mkdir(parents=True, exist_ok=True)
    source = raw / "input.json"
    d = {k: deepcopy(data[k]) for k in ("head", "sources", "seeds")}
    d.update(seeds=[data["seeds"][0]], labels=prior.labels(data, "labels"))
    atomic_json(source, d)
    script = raw / "worker.py"
    script.write_text(
        "import json,os,signal,sys\nfrom pathlib import Path\n"
        "from carnot.verify import projected_online_8076 as m\n"
        "d=json.loads(Path(sys.argv[1]).read_text())\noriginal=m.old.Journal.emit\n"
        "project=m.kernel.project\n"
        "if sys.argv[5]=='force':\n"
        " def forced(*args,**kwargs): return project(*args,**dict(kwargs,budget=0))\n"
        " m.kernel.project=forced\n"
        "def emit(self,kind,row):\n"
        " hit=kind==sys.argv[3] and (kind!='fallback' or row['reason']=='projection')\n"
        " if hit and sys.argv[4]=='before': os.kill(os.getpid(),signal.SIGKILL)\n"
        " original(self,kind,row)\n"
        " if hit and sys.argv[4]=='after': os.kill(os.getpid(),signal.SIGKILL)\n"
        "m.old.Journal.emit=emit\nm.measure(d,Path(sys.argv[2]),budget_s=60)\n"
    )

    def run(
        name: str,
        folder: Path,
        event: str = "none",
        side: str = "none",
        expected: int = 0,
        mode: str = "natural",
    ) -> Json:
        if time.monotonic() >= deadline:
            raise TimeoutError("recovery_budget")
        prior.progress("subprocess_before_8077_" + name)
        receipt = run_check(
            ROOT,
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(script),
                    str(source),
                    str(folder),
                    event,
                    side,
                    mode,
                ],
                deadline_s=min(90, max(1, deadline - time.monotonic())),
                expected_exit=expected,
            ),
            raw / "scratch",
            raw / "logs",
        )
        prior.progress("subprocess_after_8077_" + name, int(receipt["passed"]), 0)
        return receipt

    baseline = raw / "uninterrupted"
    receipts = [run("baseline", baseline)]
    ledger = Path(f"seed-{d['seeds'][0]}/ledger.sqlite")
    expected = prior.journal_events(baseline / ledger)
    mode = "natural"
    if not any(k == "fallback" and r["reason"] == "projection" for k, r in expected):
        baseline, mode = raw / "forced_uninterrupted", "force"
        receipts.append(run("forced_fallback_baseline", baseline, mode=mode))
        expected = prior.journal_events(baseline / ledger)
    rows = []
    for event in ("addition", "fallback", "candidate", "consume", "commit"):
        for side in ("before", "after"):
            folder = raw / (event + "-" + side)
            death = run(event + "_" + side + "_kill", folder, event, side, -9, mode)
            prefix = prior.journal_events(folder / ledger)
            resumed = run(event + "_" + side + "_resume", folder, mode=mode)
            actual = prior.journal_events(folder / ledger)
            state = recovery_state(expected, prefix, actual)
            consumed = [r["slot"] for k, r in actual if k == "consume"]
            row = dict(
                event=event,
                boundary=side,
                exit_code=death["exit_code"],
                **state,
                uninterrupted_sha256=sha256_file(baseline / ledger),
                recovered_sha256=sha256_file(folder / ledger),
                exactly_once=len(set(consumed)) == len(consumed),
                passed=all(r["passed"] for r in [*receipts, death, resumed])
                and len(set(consumed)) == len(consumed),
                validation_receipts=[death, resumed],
                scope="Private first-seed recovery control; independent_count=0",
                projection_control=mode,
            )
            math.equal(
                "recovered_durable_bytes", row["uninterrupted_sha256"], row["recovered_sha256"]
            )
            rows.append(row)
            prior.progress("8077_recovery_boundary", len(rows), 10 - len(rows))
    atomic_json(raw / "receipts.json", dict(rows=rows, baseline_receipts=receipts))
    return rows


def measure(data: Json, raw: Path) -> Json:
    """Copy primitives and seal retention before the target accessor opens."""
    deadline = time.monotonic() + 900
    trajectory = raw / "trajectory"
    shutil.copytree(Path(data["trajectory"]), trajectory)
    inputs = json.loads((trajectory / "inputs.json").read_text())
    for key in ("head", "sources", "seeds"):
        math.equal("input." + key, data[key], inputs[key])
    math.equal("timeline", 256, len(data["sources"]))
    math.equal("unique_sources", 256, len({r["source_cluster_id"] for r in data["sources"]}))
    for row in [*data["sources"], *data["retention"]]:
        math.equal(
            "public_label_fields",
            [],
            sorted(set(row) & {"y", "label", "labels", "eligible_y", "human_label"}),
        )
        if "source_bytes" in row:
            rebuilt = features.extract(
                {k: row[k] for k in ("source_bytes", "answer_bytes", "family_id")}
            )
            math.equal("public_features", row["features"], rebuilt["values"])
    prior.progress("benchmark_before_independent_8077")
    result = a.reconstruct(
        trajectory, prior.labels(data, "labels"), budget_s=deadline - time.monotonic()
    )
    prior.progress("benchmark_after_independent_8077", len(result["rows"]), 0)
    prior.progress("retention_predictions_before_8077", 0, len(data["retention"]))
    result["retention_rows"] = [
        dict(r, condition="sealed_retention")
        for r in independent.retention(
            data, result["final_head_seals"], raw, lambda: prior.labels(data, "retention_labels")
        )
    ]
    prior.progress("retention_predictions_after_8077", len(result["retention_rows"]), 0)
    result.update(a.comparisons(result["later_source_rows"], result["retention_rows"]))
    result["beneficial_changed_sources"] = result["H2"]["beneficial_changed_sources"]
    result["uncertainty_scope"] = result["H2"]["uncertainty_scope"]
    cells = {
        (r["source"], r["seed"], r["arm"]): r
        for r in result["later_source_rows"]
        if r["denominator"]
    }
    changed_sources = {
        r["source"]
        for r in result["later_source_rows"]
        if r["denominator"]
        and r["arm"] == "projected_fresh"
        and r["action"] != cells[r["source"], r["seed"], "ray_fresh"]["action"]
    }
    result["unchanged_sources"] = sorted(
        {r["source"] for r in result["later_source_rows"] if r["denominator"]} - changed_sources
    )
    result["recovery_rows"] = recovery(data, raw / "recovery", budget_s=deadline - time.monotonic())
    math.equal("recovery_complete", True, all(r["passed"] for r in result["recovery_rows"]))
    atomic_json(raw / "independent_reduction.json", result)
    return result


def worker(root: Path, raw: Path, fixture_input: Path | None = None) -> Json:
    """Authenticate external operands before owned reduction and private recovery."""
    started = time.monotonic()
    prior.progress("8077_preconditions_before")
    refs, failures = driver.prerequisites(root, raw)
    data: Json = {}
    if fixture_input:
        data = json.loads(fixture_input.read_text())
        refs.append(reference(fixture_input))
        failures = []
        from carnot.verify import projected_online_8076 as producer

        producer.measure(data, raw / "fixture_producer", budget_s=120)
        data["trajectory"] = str(raw / "fixture_producer")
    elif not failures:
        prior.progress("8077_numerical_head_load_before")
        data, failures = historical.load_inputs(root, raw / "historical")
        refs.extend(data.get("references", []))
        path = root / "results" / "experiment_8076_v699_projected_online_learning.json"
        parent = json.loads(path.read_text())
        for field, expected in [
            ("experiment_id", 8076),
            ("task_id", "exp8076-projected-online-learning"),
            ("learning_trajectory_ready_score", 1),
            ("required_checks_passed", True),
            ("verifier_is_oracle", False),
        ]:
            if parent.get(field) != expected:
                failures.append(prior.failure(path, field, expected, parent.get(field)))
        for ref in parent["raw_shard_hashes"] + parent["source_artifact_hashes"]:
            snapshot = Path(ref.get("snapshot_path", ref["path"]))
            observed = sha256_file(snapshot) if snapshot.is_file() else None
            if observed != ref["sha256"]:
                failures.append(prior.failure(snapshot, "sha256", ref["sha256"], observed))
        data["trajectory"] = parent["trajectory_directory"]
        prior.progress("8077_numerical_head_load_after", int(bool(data.get("head"))), 0)
    frozen = time.monotonic()
    prior.progress("8077_preconditions_after", len(refs), len(failures))
    evidence: Json = {}
    owned_failure = False
    if not failures:
        try:
            evidence = measure(data, raw)
        except (OSError, ValueError, KeyError, IndexError, TimeoutError) as error:
            owned_failure = True
            failures.append(
                prior.failure(raw / "trajectory", "owned_reduction", True, str(error), owned=True)
            )
    work = dict(
        data=data,
        evidence=evidence,
        failures=failures,
        owned_failure=owned_failure,
        source_artifact_hashes=refs,
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                prior.OWNED[2],
                prior.MODULE,
                "python/carnot/experiment_8076_v699_projected_online_learning.py",
                "python/carnot/verify/learning_retention_audit_8026.py",
                "python/carnot/verify/evidence_features_7980.py",
                "python/carnot/verify/projected_online_8076.py",
                "python/carnot/verify/constraint_projection_8075.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v686_contract_validation.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
            ]
        },
        phase_spans=[
            dict(phase="preconditions", duration_s=frozen - started),
            dict(
                phase="independent_reduction_retention_recovery",
                duration_s=time.monotonic() - frozen,
            ),
        ],
        duration_s=time.monotonic() - started,
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name not in {"work.json", "validation_commands.json"}
            and "inputs" not in p.parts
        ],
    )
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, *, fixture: bool) -> Json:
    """Audit readiness permits a null; failed safeguards never award benefit."""
    value = BASE_BUILD(work, raw, receipts, coverage, fixture=fixture)
    h = work["evidence"].get("H2", {})
    ready = (
        value["required_checks_passed"]
        and bool(h)
        and not work["failures"]
        and not work["owned_failure"]
    )
    benefit = int(bool(ready and h.get("qualified_benefit") and not fixture))
    if benefit:
        value["verdict_class"] = "positive"
    value.update(
        experiment_id=8077,
        task_id=TASK,
        schema="carnot.v699.projected_learning_audit.v1",
        honest_verdict="complete_" + value["verdict_class"] + "_projected_learning_audit"
        if value["verdict_class"] != "blocked"
        else value["honest_verdict"],
        learning_audit_ready_score=int(bool(ready and not fixture)),
        projected_learning_benefit_score=benefit,
        H2=h,
        random_seed=a.CONFIG["seed"],
        config=a.CONFIG,
        claim_scope="Independent replay of historically exposed cached development sources; conditional H2, retention and private recovery only; no unseen-environment or generalized learning claim.",
        methodology_note="Cold independent chronological equations and public features; seed-averaged common later sources on the original256-slot moving-block timeline; both guarded retention arms; private SIGKILL recovery. No current model inference.",
        substrate_declaration=dict(
            inference_substrate=value["inference_substrate"],
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            trained_head_specs=value["trained_head_specs"],
        ),
    )
    for key in (
        "later_source_rows",
        "retention_rows",
        "per_seed_false_accept_rows",
        "recovery_rows",
        "reconstructed_projection_residuals",
    ):
        value.setdefault(key, [])
    for key in (
        "qualified_constraint_additions",
        "beneficial_changed_sources",
        "resets",
    ):
        value.setdefault(key, 0)
    value.setdefault("unchanged_sources", [])
    value.setdefault("uncertainty_scope", "No qualified operands; no inference")
    if h:
        rows = [
            r
            for r in value["rows"]
            if r["seed"] == work["data"]["seeds"][0] and r["arm"] == "frozen"
        ]
        counts = dict(
            intended_count=256,
            eligible_count=sum(r["denominator"] for r in rows),
            independent_count=h["support_count"],
            completed_count=sum(r["status"] == "completed" for r in rows),
            censored_count=sum(r["status"] == "censored" for r in rows),
            excluded_count=sum(r["status"] == "excluded" for r in rows),
            failed_count=0,
        )
        value.update(
            counts,
            sample_size_budget=dict(
                counts,
                later_independent_count=h["support_count"],
                retention_independent_count=h["retention_support_count"],
                seeds_are_independent=False,
                bootstrap_draws_are_independent=False,
                independent_datasets=1,
            ),
        )
        checks = [
            ("support_passed", True, h["support_passed"], "=="),
            ("safety_passed", True, h["safety_passed"], "=="),
            ("beneficial_changed_sources", 5, h["beneficial_changed_sources"], ">="),
            ("cost_gain", 0.02, h["tests"][0]["gain"], ">="),
            ("margin_p", 0.05, h["tests"][0]["raw_p"], "<"),
        ]
        for arm in h["retention_checks"]:
            checks += [
                ("retention." + arm["arm"] + ".brier", 0.01, arm["brier_drift"], "<="),
                ("retention." + arm["arm"] + ".cost", 0.02, arm["cost_drift"], "<="),
            ]
        reduction_path = raw / "independent_reduction.json"
        for field, expected, observed, op in checks:
            passed = observed is not None and (
                {
                    "==": lambda: observed == expected,
                    ">=": lambda: observed >= expected,
                    "<=": lambda: observed <= expected,
                    "<": lambda: observed < expected,
                }[op]()
            )
            if not passed:
                value["gate_check_summary"].append(
                    dict(
                        prior.failure(reduction_path, "H2." + field, expected, observed),
                        op=op,
                        classification="scientific",
                    )
                )
        if not h["support_passed"] and ready and not fixture:
            value.update(
                verdict_class="blocked",
                honest_verdict="complete_blocked_later_or_retention_support",
                learning_audit_ready_score=0,
                projected_learning_benefit_score=0,
            )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            code=value["code_config_hashes"],
            inputs=value["source_artifact_hashes"],
            raw=value["raw_shard_hashes"],
            config=a.CONFIG,
        )
    )
    value["field_principles"].update(
        {
            k: "Bind "
            + k
            + " to independent primitive development evidence; repetitions cannot create unseen-environment credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        H2="Original timeline masks and paired seed means prevent sample inflation; both retention safeguards block positive benefit.",
        learning_audit_ready_score="A completed independent null is usable evidence and must not trigger accidental retries.",
        projected_learning_benefit_score="Local geometry or restored frozen heads cannot substitute for later utility and retention.",
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild reductions and recovery from immutable bytes, rejecting forged readiness."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            math.equal(
                "raw_hash", ref["sha256"], sha256_file(Path(ref.get("snapshot_path", ref["path"])))
            )
        math.equal(
            "code_hashes",
            False,
            any(
                sha256_file(ROOT / p) != digest for p, digest in value["code_config_hashes"].items()
            ),
        )
        validation = json.loads((raw / "validation.json").read_text())
        for receipt in validation["receipts"]:
            math.equal(
                "validation_log_hash", receipt["log_sha256"], sha256_file(Path(receipt["log_path"]))
            )
        work = json.loads((raw / "work.json").read_text())
        if work["evidence"]:
            math.equal(
                "work_reduction",
                work["evidence"],
                json.loads((raw / "independent_reduction.json").read_text()),
            )
            rebuilt = a.reconstruct(raw / "trajectory", prior.labels(work["data"], "labels"))
            with tempfile.TemporaryDirectory(prefix="carnot-8077-retention-") as temp:
                retained = [
                    dict(r, condition="sealed_retention")
                    for r in independent.retention(
                        work["data"],
                        rebuilt["final_head_seals"],
                        Path(temp),
                        lambda: prior.labels(work["data"], "retention_labels"),
                    )
                ]
            rebuilt.update(
                retention_rows=retained, **a.comparisons(rebuilt["later_source_rows"], retained)
            )
            math.equal(
                "independent_reduction",
                False,
                any(
                    work["evidence"][k] != v for k, v in rebuilt.items() if k != "unchanged_sources"
                ),
            )
            for row in work["evidence"]["recovery_rows"]:
                recovery_raw = raw / "recovery"
                baseline = recovery_raw / (
                    "uninterrupted"
                    if row["projection_control"] == "natural"
                    else "forced_uninterrupted"
                )
                ledger = Path(f"seed-{work['data']['seeds'][0]}/ledger.sqlite")
                expected = prior.journal_events(baseline / ledger)
                actual_path = recovery_raw / (row["event"] + "-" + row["boundary"]) / ledger
                actual = prior.journal_events(actual_path)
                count = next(
                    i
                    for i, (kind, payload) in enumerate(expected)
                    if kind == row["event"]
                    and (kind != "fallback" or payload["reason"] == "projection")
                ) + int(row["boundary"] == "after")
                state = recovery_state(expected, expected[:count], actual)
                math.equal(
                    "recovery_state",
                    False,
                    any(row[k] != v for k, v in state.items())
                    or row["exit_code"] != -9
                    or not row["exactly_once"]
                    or not row["passed"],
                )
                math.equal(
                    "recovery_hashes",
                    False,
                    sha256_file(baseline / ledger) != row["uninterrupted_sha256"]
                    or sha256_file(actual_path) != row["recovered_sha256"],
                )
        return bool(
            build(
                work,
                raw,
                validation["receipts"],
                validation["coverage"],
                fixture=validation["fixture"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError, StopIteration):
        return False


def main(argv: list[str] | None = None) -> int:
    """Reuse the qualified terminal driver while restoring its namespace on exit."""
    overrides: Json = dict(
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        MODULE=MODULE,
        TEST=TEST,
        OWNED=OWNED,
        INPUTS=INPUTS,
        worker=worker,
        build=build,
        replay=replay,
        manifest=manifest,
        m=SimpleNamespace(CONFIG=a.CONFIG, FIELDS=a.FIELDS, progress=prior.progress),
    )
    saved = {key: getattr(driver, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(driver, key, value)
        return int(driver.main(argv))
    finally:
        for key, value in saved.items():
            setattr(driver, key, value)
        config = os.environ.get("CARNOT_8076_COVERAGE_CONFIG")
        current = statement_coverage.Coverage.current()
        if config and current is not None:
            current.save()
            shutil.copyfile(
                current.get_data().data_filename(),
                Path(config).parent / (".coverage.owned-" + str(os.getpid())),
            )
