"""REQ-VERIFY-8143: apply qualified causal heads to historical public stream slots.

The reused engine owns the numerical protocol. This layer opens only released
targets, preserves state evidence and seals retention before its labels open.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import learning_protocol_8138 as engine

Json = dict[str, Any]
ROOT = engine.ROOT
NAME = "experiment_8143_v704_delayed_energy_memory"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/delayed_energy_memory_8143.py"
RUNNER = "python/carnot/reporting/delayed_energy_execution_8143.py"
TEST = "tests/python/test_delayed_energy_memory_8143.py"
UPSTREAM = "results/experiment_8138_v704_learning_protocol.json"
UPSTREAM_HASH = "sha256:df6366aceaab34ba88e1a84a8e7a850bbfd40a91a9635dddf283eb10c3f7810d"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real flushed counts keep finite CPU work visible to the supervisor."""
    print(f"[exp8143] phase={phase} completed={completed} pending={pending}", flush=True)


class DelayedLabels(list[int | None]):
    """Decode one evaluator fragment only after its issuing state is durable."""

    def __init__(self, path: Path, rows: list[Json]):
        super().__init__()
        self.vault = engine.historical.LabelVault(path, rows)
        self.state: Json = {}
        self.opened: Json = {}

    def __len__(self) -> int:
        return 256

    def __getitem__(self, index: Any) -> Any:
        slot = int(index) + 1
        if self.state.get("phase") != "release" or self.state["cursor"] != slot + 20:
            raise ValueError("unsealed_release")
        label = self.vault.release(slot, self.state["cursor"], sealed=True)
        self.opened[str(slot)] = label
        return label["y"]


def run_seed(
    rows: list[Json],
    label_path: Path,
    seed: int,
    raw: Path,
    *,
    state: Json | None = None,
    stop: int = 256,
    crash_slot: int = 0,
) -> Json:
    """Persist each issuing boundary and checkpoints without decoding future labels."""
    raw.mkdir(parents=True, exist_ok=True)
    labels = DelayedLabels(label_path, rows)
    labels.state = state or {}
    started = time.monotonic()
    journal = raw / "durable.jsonl"
    with journal.open("a") as stream:

        def seal(kind: str, current: Json) -> None:
            labels.state = current
            if time.monotonic() - started > 1200:
                raise TimeoutError("cpu_science_budget")
            record = engine.durable_record(kind, current)
            record["issued_ref"] = engine.shard(raw, {"issued": current["issued"][-1]})
            record["heads_ref"] = engine.shard(raw, {"arms": current["arms"]})
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            slot = current["cursor"] - int(kind == "durable_update")
            if (kind == "durable_update" and slot % 32 == 0) or (
                kind == "commit_candidate" and slot in [64, 128, 192]
            ):
                atomic_json(raw / f"checkpoint-{slot}-{kind}.json", current)
                progress("checkpoint", slot, len(current["pending"]))
            if kind == "durable_commit" and current["cursor"] == crash_slot:
                atomic_json(raw / "crash_checkpoint.json", current)
                progress("intentional_crash", slot, len(current["pending"]))
                os._exit(73)

        final = engine.run(rows, labels, seed, state=state, stop=stop, seal=seal)
    atomic_json(raw / "opened_labels.json", labels.opened)
    atomic_json(raw / "final_state.json", final)
    return final


def head_diagnostics(state: Json) -> Json:
    """Repeated centers remain installed; rank reports their actual information."""
    return {
        arm: dict(
            installed_capacity=len(head["centers"]),
            duplicate_centers=len(head["centers"])
            - len({canonical_hash(c["x"]) for c in head["centers"]}),
            effective_rank=int(
                np.linalg.matrix_rank(
                    np.exp(
                        -np.sum(
                            (
                                np.array([c["x"] for c in head["centers"]])[:, None, :]
                                - np.array([c["x"] for c in head["centers"]])[None, :, :]
                            )
                            ** 2,
                            axis=2,
                        )
                        / (2 * state["geometry"]["sigma"] ** 2)
                    )
                )
            ),
            optimizer_step=head["optimizer_step"],
        )
        for arm, head in state["arms"].items()
    }


def authenticate(root: Path, raw: Path, b: engine.methods.Custody) -> Json:
    """Qualified exact protocol and slot masks authorize historical reuse only."""
    b.upstream = "exp8138-learning-protocol"
    path = root / UPSTREAM
    value = b.read(path)
    terminal = engine.methods.historical.terminal(path, value, b)
    for key, expected in [
        ("experiment_id", 8138),
        ("learning_protocol_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        b.require(path, key, expected, value.get(key))
    b.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    b.require(path, "qualification_sha256", UPSTREAM_HASH, sha256_file(path))
    side = b.read(Path(value["terminal_validation_sidecar_path"]))
    b.require(path, "normal_process_exit", True, side.get("normal_process_exit"))
    b.require(
        path,
        "numerical_protocol",
        dict(
            delayed_memory=engine.protocol(),
            statistical_plan=engine.methods.protocol()["statistical_plan"],
        ),
        value["numerical_protocol"],
    )
    upstream, _ = engine.authenticate(root, raw, b)
    for role in ["stream", "retention"]:
        b.require(
            path,
            role + "_feature_manifest",
            value[role + "_feature_manifest"],
            upstream[role + "_feature_manifest"],
        )
    b.require(
        path,
        "original_slot_mask",
        value["original_slot_mask"],
        upstream["direct_stream_authentication"]["original_slot_mask"],
    )
    return upstream


def restart_specs(raw: Path) -> list[Json]:
    """Fix crash and fresh resume argv before historical measurement begins."""
    base = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / CLI),
        "--seed-input",
        str(raw / "restart_input.json"),
    ]
    return [
        dict(
            name="real_crash",
            argv=base + ["--seed-output", str(raw / "crash"), "--crash-slot", "64"],
            expected_exit=73,
            deadline_s=60,
            classification="required",
        ),
        dict(
            name="fresh_resume",
            argv=base
            + [
                "--seed-output",
                str(raw / "resume"),
                "--resume-state",
                str(raw / "crash/crash_checkpoint.json"),
            ],
            expected_exit=0,
            deadline_s=60,
            classification="required",
        ),
    ]


def restart_check(rows: list[Json], labels: Path, raw: Path, baseline: Json) -> Json:
    """A real abrupt exit must resume with identical natural predictions and state."""
    atomic_json(raw / "restart_input.json", dict(rows=rows, label_path=str(labels), seed=101))
    private = Path(mkdtemp(prefix="carnot-8143-restart-"))
    receipts = []
    for index, spec in enumerate(restart_specs(raw)):
        progress("before_subprocess_" + spec["name"], index, 2 - index)
        row = run_check(ROOT, spec, private, private / "sealed_logs", heartbeat_s=30)
        row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
        row["intentional_abrupt_exit"] = spec["name"] == "real_crash"
        receipts.append(row)
        progress("after_subprocess_" + spec["name"], index + 1, 1 - index)
    resumed = raw / "resume/final_state.json"
    matched = resumed.exists() and json.loads(resumed.read_text()) == baseline
    return dict(
        passed=all(r["passed"] for r in receipts) and matched,
        predictions_and_state_identical=matched,
        uninterrupted_state_hash=canonical_hash(baseline),
        validation_receipts=receipts,
    )


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Run the exact historical protocol; private targets confer only fixture credit."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    b = engine.methods.Custody(raw / "inputs")
    work: Json = dict(
        input_ready=0,
        fixture_mode=fixture,
        rows=[],
        learning_rows=[],
        admission_rows=[],
        state_manifest=[],
        final_head_manifest=[],
        retention_predictions=[],
        lost_feedback_rows=[],
        cited_upstream_artifacts=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        gate_check_summary=[],
        phase_spans=[],
        numerical_protocol=dict(
            delayed_memory=engine.protocol(),
            statistical_plan=engine.methods.protocol()["statistical_plan"],
        ),
    )
    progress("before_input_custody")
    work["restart_commands"] = restart_specs(raw)
    atomic_json(raw / "restart_commands.json", dict(commands=work["restart_commands"]))
    try:
        if fixture:
            features, labels = {}, {}
            for role, count in [("stream", 256), ("retention", 64)]:
                rows, targets = engine.fixture(count, role)
                features[role] = engine.shard(raw, dict(rows=rows))
                labels[role] = engine.shard(
                    raw, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)])
                )
            upstream = dict(
                evaluator_label_manifests=labels,
                stream_feature_manifest=features["stream"],
                retention_feature_manifest=features["retention"],
            )
        else:
            upstream = authenticate(root, raw, b)
            work["cited_upstream_artifacts"] = [
                dict(
                    path=str(root / UPSTREAM),
                    sha256=sha256_file(root / UPSTREAM),
                    scope="qualified_protocol",
                ),
                dict(
                    path=str(root / engine.historical.UPSTREAM),
                    scope="historical_model_receipts",
                    historical_model_provenance=upstream.get("historical_model_provenance"),
                ),
            ]
        work["input_ready"] = 1
        work["input_manifests"] = {
            k: upstream[k]
            for k in [
                "stream_feature_manifest",
                "retention_feature_manifest",
                "evaluator_label_manifests",
            ]
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not b.failures:
            b.failures.append(
                dict(
                    check="qualified_protocol_custody",
                    upstream="exp8138",
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="authenticated_primitives",
                    op="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                )
            )
    work["gate_check_summary"] = b.checks + [r for r in b.failures if r not in b.checks]
    work["source_artifact_hashes"] = b.refs
    work["phase_spans"].append(dict(phase="custody", duration_s=time.monotonic() - started))
    progress("after_input_custody", work["input_ready"])
    if work["input_ready"]:
        public = {
            r: json.loads(Path(upstream[r + "_feature_manifest"]["path"]).read_text())["rows"]
            for r in ["stream", "retention"]
        }
        work["original_slot_mask"] = {
            r: [v["values"] is not None for v in rows] for r, rows in public.items()
        }
        all_heads, all_predictions = [], []
        progress("before_head_benchmark", 0, 20)
        for index, seed in enumerate(range(101, 121)):
            if time.monotonic() - started >= 1200:
                raise TimeoutError("cpu_science_budget")
            seed_raw = raw / f"seed-{seed}"
            state = run_seed(
                public["stream"],
                Path(upstream["evaluator_label_manifests"]["stream"]["path"]),
                seed,
                seed_raw,
            )
            final_ref = dict(
                path=str(seed_raw / "final_state.json"),
                sha256=sha256_file(seed_raw / "final_state.json"),
            )
            work["state_manifest"].append(
                dict(seed=seed, state=final_ref, diagnostics=head_diagnostics(state))
            )
            work["learning_rows"].extend(
                dict(seed=seed, **r)
                for r in state["events"]
                if r["kind"] in ["commit_candidate", "durable_update"]
            )
            work["admission_rows"].extend(
                dict(seed=seed, **r)
                for r in state["events"]
                if r["kind"] in ["admit_once", "defer_candidate"]
            )
            work["lost_feedback_rows"].extend(
                dict(seed=seed, slot=s, reason="oldest_pending_overflow") for s in state["lost"]
            )
            all_heads.append(
                dict(
                    seed=seed,
                    heads=state["arms"],
                    geometry=state["geometry"],
                    state_hash=canonical_hash(state),
                )
            )
            predictions = [
                dict(
                    slot=r["slot"],
                    predictions={
                        a: None
                        if r["values"] is None
                        else engine.probability(h, state["geometry"], r["values"])
                        for a, h in state["arms"].items()
                    },
                )
                for r in public["retention"]
            ]
            for pred in predictions:
                pred["prediction_hash"] = canonical_hash(pred["predictions"])
            all_predictions.append(dict(seed=seed, rows=predictions))
            opened = json.loads((seed_raw / "opened_labels.json").read_text())
            work["rows"].extend(score_stream(public["stream"], state, opened, seed))
            progress("seed_complete", index + 1, 19 - index)
        work["final_head_manifest"] = engine.shard(raw, dict(rows=all_heads))
        retention_seal = engine.shard(raw, dict(rows=all_predictions))
        work["retention_prediction_manifest"] = retention_seal
        work["retention_predictions"] = all_predictions
        progress("all_heads_and_retention_predictions_sealed", 20)
        baseline = json.loads(Path(work["state_manifest"][0]["state"]["path"]).read_text())
        work["restart_receipt"] = restart_check(
            public["stream"],
            Path(upstream["evaluator_label_manifests"]["stream"]["path"]),
            raw,
            baseline,
        )
        vault = engine.historical.LabelVault(
            Path(upstream["evaluator_label_manifests"]["retention"]["path"]), public["retention"]
        )
        for group in all_predictions:
            for row, pred in zip(public["retention"], group["rows"], strict=True):
                label = vault.release(row["slot"], 256, sealed=True, retention=True)
                work["rows"].extend(
                    engine.historical.scored(row, pred, label, "retention", group["seed"])
                )
        progress("after_head_benchmark", 20)
    work["reductions"] = engine.historical.reductions(work["rows"])
    work["code_config_hashes"] = {
        p: sha256_file(ROOT / p)
        for p in [
            MODULE,
            RUNNER,
            CLI,
            TEST,
            engine.MODULE,
            engine.historical.MODULE,
            engine.methods.MODULE,
            "python/carnot/verify/radial_memory_8085.py",
            "openspec/change-proposals/v702-methods-and-stream-protocol.md",
        ]
    }
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="causal_heads_and_retention",
            duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"],
        )
    )
    work["raw_shard_hashes"] = [
        dict(path=str(p), sha256=sha256_file(p))
        for p in sorted(raw.rglob("*"))
        if p.is_file() and "inputs" not in p.relative_to(raw).parts
    ]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", len(work["state_manifest"]))
    return work


def score_stream(rows: list[Json], state: Json, opened: Json, seed: int) -> list[Json]:
    """Warmup sources never enter the later-stream headline; unresolved tail stays."""
    result = []
    for row, pred in zip(rows[64:], state["issued"][64:], strict=True):
        label = opened.get(
            str(row["slot"]), dict(y=None, exclusion_reason="feedback_unresolved_tail")
        )
        result.extend(engine.historical.scored(row, pred, label, "later_stream", seed))
    return result


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """A causal trajectory can be ready without statistical support or benefit."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and (not work["input_ready"] or work["restart_receipt"]["passed"])
    )
    ready = int(owned and work["input_ready"] and len(work["state_manifest"]) == 20)
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
        if work["fixture_mode"]
        else "null"
    )
    operand = next(
        (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
        "trajectory_complete_audit_pending",
    )
    units = [
        r
        for r in work["rows"]
        if r["seed"] == 101 and r["arm"] == engine.ARMS[0] and r["metric"] == "brier"
    ]
    counts = {
        s: sum(r["status"] == s for r in units) for s in ["completed", "excluded", "censored"]
    }
    value = dict(
        work,
        experiment_id=8143,
        task_id="exp8143-delayed-energy-memory",
        milestone="2026.10.704",
        honest_verdict="complete_" + verdict + "_" + ("owned_validation" if not owned else operand),
        verdict_class=verdict,
        learning_trajectory_ready_score=ready,
        continuous_self_learning_task=True,
        verifier_is_oracle=work["fixture_mode"],
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(engine.historical.ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(
                kind="small_Gaussian_residual",
                arms=engine.ARMS[1:],
                seeds=list(range(101, 121)),
                scope="private_fixture" if work["fixture_mode"] else "exposed_historical_stream",
                initial_centers=16,
                maximum_centers=28,
                optimizer="four_clipped_SGD_steps",
                learning_rate=0.05,
                ridge=0.01,
            )
        ]
        if work["input_ready"]
        else [],
        claim_scope="Causal persistent constraints on exposed development; statistical benefit reserved for the independent audit",
        exposure_scope="private_circular_fixture"
        if work["fixture_mode"]
        else "exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        intended_count=256,
        eligible_count=counts["completed"],
        independent_count=counts["completed"] if not work["fixture_mode"] else 0,
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=0,
        sample_size_budget=dict(
            original_stream=256,
            public_warmup=64,
            evaluated_stream=192,
            retention=64,
            robustness_seeds=20,
            arms=4,
            repeats_add_independent_sources=0,
        ),
        run_date="20261004",
        random_seed=101,
        acceptance_gates=dict(
            custody="Exp8138 readiness1; exact primitive slots/masks and V702 protocol",
            conformance="Qualified engine, delay20, capacity32, no tail flush, sealed retention and cold state replay",
            validation="Normal owned checks and100 percent changed statement coverage; audit decides benefit",
        ),
        methodology_note="No LLM is loaded. Historical Qwen logits remain fixed offsets. Public first64 geometry and zero residual "
        "start equally budgeted learned arms. Newest released updates train proposals; future bucket0 labels admit once. "
        "Seeds repeat sources. Completed trajectories do not establish external generalization or statistical benefit.",
    )
    value["field_principles"] = {
        k: "Exact causal evidence; repeats do not add sources or generalization credit."
        for k in value
    }
    value["field_principles"].update(
        learning_trajectory_ready_score="Custody, conformance and owned checks; valid null may be ready.",
        verdict_class="External operand blocks are terminal; owned failures disqualify.",
        model_invocation_counts="Current zero calls exclude imported historical model provenance.",
        retention_predictions="Every seed prediction and final head sealed before retention targets open.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Independent scalar events and full reexecution reject even rehashed drift."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if (
            checksum != canonical_hash(value)
            or value["numerical_protocol"]["delayed_memory"] != engine.protocol()
        ):
            return False
        for p, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / p) != digest:
                return False
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if engine.historical.reductions(value["rows"]) != value["reductions"]:
            return False
        recomputed = []
        if value["input_ready"]:
            manifests = value["input_manifests"]
            public = {
                r: json.loads(Path(manifests[r + "_feature_manifest"]["path"]).read_text())["rows"]
                for r in ["stream", "retention"]
            }
            heads = json.loads(Path(value["final_head_manifest"]["path"]).read_text())["rows"]
            saved_predictions = json.loads(
                Path(value["retention_prediction_manifest"]["path"]).read_text()
            )["rows"]
            if saved_predictions != value["retention_predictions"]:
                return False
            for index, entry in enumerate(value["state_manifest"]):
                state = json.loads(Path(entry["state"]["path"]).read_text())
                labels = DelayedLabels(
                    Path(manifests["evaluator_label_manifests"]["stream"]["path"]), public["stream"]
                )

                def seal(kind: str, current: Json) -> None:
                    labels.state = current

                replayed = engine.run(public["stream"], labels, entry["seed"], seal=seal)
                if (
                    state != replayed
                    or heads[index]["state_hash"] != canonical_hash(state)
                    or entry["diagnostics"] != head_diagnostics(state)
                ):
                    return False
                recomputed.extend(
                    score_stream(public["stream"], replayed, labels.opened, entry["seed"])
                )
                vault = engine.historical.LabelVault(
                    Path(manifests["evaluator_label_manifests"]["retention"]["path"]),
                    public["retention"],
                )
                for row, pred in zip(
                    public["retention"], saved_predictions[index]["rows"], strict=True
                ):
                    for arm, head in state["arms"].items():
                        expected = (
                            None
                            if row["values"] is None
                            else engine.scalar_probability(head, state["geometry"], row["values"])
                        )
                        actual = pred["predictions"][arm]
                        if (actual is None) != (expected is None) or (
                            expected is not None and abs(actual - expected) > 1e-12
                        ):
                            return False
                    recomputed.extend(
                        engine.historical.scored(
                            row,
                            pred,
                            vault.release(row["slot"], 256, sealed=True, retention=True),
                            "retention",
                            entry["seed"],
                        )
                    )
                progress("cold_seed_complete", index + 1, 19 - index)
        if recomputed != value["rows"]:
            # Measurement stores all stream seeds before all retention seeds.
            key = lambda r: (r["seed"], r["condition"], r["slot"], r["arm"], r["metric"])
            if sorted(recomputed, key=key) != sorted(value["rows"], key=key):
                return False
        expected = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        return all(
            expected[k] == value[k]
            for k in [
                "learning_trajectory_ready_score",
                "verdict_class",
                "completed_count",
                "censored_count",
                "excluded_count",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
