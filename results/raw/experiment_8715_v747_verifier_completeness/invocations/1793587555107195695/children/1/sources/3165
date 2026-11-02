"""Bounded online fixture for REQ-LEARN-7760; no natural learning claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7742_v674_bank_qualification import sentence_features
from carnot.experiment_7732_v673_causal_admission import admission_decision
from carnot.reporting.constraint_bank_protocol import Bank, grammar
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "results/experiment_7742_v674_bank_qualification.json"
MANIFEST = (
    "results/raw/experiment_7740_v674_sentence_label_protocol/sentence_protocol_manifest.json"
)
BANK_MODULE = "python/carnot/reporting/constraint_bank_protocol.py"
FEATURE_MODULE = "python/carnot/experiment_7742_v674_bank_qualification.py"
MODULE = "python/carnot/experiment_7760_v675_online_runner.py"
CLI = "scripts/experiments/experiment_7760_v675_online_runner.py"
TEST = "tests/python/test_experiment_7760_v675_online_runner.py"
RAW = Path("results/raw/experiment_7760_v675_online_runner")
OUTPUT = Path("results/experiment_7760_v675_online_runner.json")
ARMS = ("adaptive", "frozen", "complete_static", "shuffled")


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print a flushed boundary with real elapsed time and completed units."""
    print(
        f"[exp7760] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def check(
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    artifact_sha256: str | None = None,
) -> dict:
    """Keep literal operands for failed custody and validation checks."""
    return {
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "artifact_sha256": artifact_sha256,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict], dict, list[str]]:
    """Authenticate the qualified bank and the declared dictionary bytes."""
    checks: list[dict] = []
    hashes: dict[str, Any] = {
        "eligible_producers": {},
        "module_hashes": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
    }
    names: list[str] = []
    for relative, upstream in (
        (SOURCE, "7742"),
        (MANIFEST, "7740"),
        (BANK_MODULE, "bank"),
        (FEATURE_MODULE, "7742-module"),
    ):
        path = root / relative
        exists = path.is_file()
        checks.append(check(upstream, relative, "exists", True, exists))
        if not exists:
            hashes["missing_inputs"].append(relative)
            continue
        digest = sha256_file(path)
        category = (
            "eligible_producers"
            if relative == SOURCE
            else "pre_gate_receipts"
            if relative == MANIFEST
            else "module_hashes"
        )
        hashes[category][relative] = digest
        try:
            value = json.loads(path.read_text()) if relative in {SOURCE, MANIFEST} else {}
        except (OSError, ValueError):
            value = {}
        if relative == SOURCE:
            checks.append(
                check(
                    upstream,
                    relative,
                    "sha256",
                    "sha256:b7ca268ac6a32e1ea9fb86503104c678754e1b11a09e18edc834182f69cb0d9c",
                    digest,
                )
            )
            for field, expected in (
                ("experiment_id", 7742),
                ("verdict_class", "circular_positive"),
                ("flagged_adversarial", False),
                ("acquisition_protocol_ready_score", 1),
            ):
                checks.append(check(upstream, relative, field, expected, value.get(field)))
            checks.append(
                check(
                    upstream,
                    relative,
                    "bank_module_sha256",
                    "sha256:3151f595df7d5719e8b52b691dd9d9c5f4fe4c51ccb06253eeeb898a4612372c",
                    sha256_file(root / BANK_MODULE) if (root / BANK_MODULE).is_file() else None,
                )
            )
        elif relative == MANIFEST:
            checks.append(
                check(
                    upstream,
                    relative,
                    "sha256",
                    "sha256:cf1f115f4efaddd39dcd135fb0f8a72ea6f0c89505fa514169e514f6cba35a3a",
                    digest,
                )
            )
            candidate = value.get("advisory_dictionary") if isinstance(value, dict) else None
            names = (
                candidate
                if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate)
                else []
            )
            checks.append(check(upstream, relative, "unique_predicates", 16, len(set(names))))
        elif relative == FEATURE_MODULE:
            checks.append(
                check(
                    upstream,
                    relative,
                    "sha256",
                    "sha256:b2e4eacc54782ed4391d800a937b92943cfe3186988f6f63325c19f79861ba9d",
                    digest,
                )
            )
    hashes["source_details"] = {
        SOURCE: {
            "sha256": hashes["eligible_producers"].get(SOURCE),
            "date": "20260927",
            "imported_fields": [
                "experiment_id",
                "verdict_class",
                "flagged_adversarial",
                "acquisition_protocol_ready_score",
            ],
            "eligible": not any(
                not row["passed"] and row["artifact_path"] == SOURCE for row in checks
            ),
        },
        MANIFEST: {
            "sha256": hashes["pre_gate_receipts"].get(MANIFEST),
            "date": "20260927",
            "imported_fields": ["advisory_dictionary"],
            "eligible": len(names) == 16,
        },
    }
    checks.append(check("host", str(root), "cpu_available", True, (os.cpu_count() or 0) > 0))
    checks.append(
        check(
            "host",
            str(root),
            "free_bytes_at_least_10m",
            True,
            os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 10_000_000,
        )
    )
    all_hashes = {
        **hashes["eligible_producers"],
        **hashes["module_hashes"],
        **hashes["pre_gate_receipts"],
    }
    for row in checks:
        row["artifact_sha256"] = all_hashes.get(row["artifact_path"])
    return checks, hashes, names


class OnlineRunner:
    """Keep the label gate, numerical model, and queue beside a durable bank."""

    def __init__(
        self, state_path: Path, bank_path: Path, names: list[str], arm: str, admit: bool = True
    ) -> None:
        if arm not in ARMS or len(names) != 16 or len(set(names)) != 16:
            raise ValueError(
                "complete_static_empty"
                if arm == "complete_static" and not names
                else "model_config_invalid"
            )
        self.state_path, self.bank_path, self.names, self.arm, self.admit = (
            state_path,
            bank_path,
            names,
            arm,
            admit,
        )
        self.bank = Bank(
            bank_path,
            grammar(),
            "priority" if arm in {"adaptive", "shuffled"} else "read_only",
            0.1,
            2,
        )
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            if (
                self.state.get("schema") != "exp7760-runner-v1"
                or self.state.get("arm") != arm
                or self.state.get("names") != names
                or self.state.get("bank_hash") != self.bank.state_hash
            ):
                raise ValueError("state_invalid")
        else:
            active = names[:] if arm == "complete_static" else []
            self.state = {
                "schema": "exp7760-runner-v1",
                "arm": arm,
                "names": names,
                "initial_active": active[:],
                "active": active,
                "weights": dict.fromkeys(active, 0.0),
                "queue": [],
                "rows": [],
                "lifecycle": [],
                "completed_blocks": [],
                "released_blocks": [],
                "credits": 0,
                "overflow": 0,
                "expired": 0,
                "update_calls": 0,
                "rng_counter": 0,
                "admitted_predicates": [],
                "lookup_s": 0.0,
                "update_s": 0.0,
                "serialization_s": 0.0,
                "acknowledgement_s": 0.0,
                "bank_hash": self.bank.state_hash,
            }
            self._save()

    def _save(self) -> None:
        begin = time.monotonic()
        self.state["bank_hash"] = self.bank.state_hash
        atomic_json(self.state_path, self.state)
        self.state["serialization_s"] += time.monotonic() - begin

    def label(self, block: int, index: int) -> int:
        """Reveal a synthetic label only after a full block is durable."""
        if block not in self.state["completed_blocks"]:
            raise ValueError("premature_label_access")
        return int(index % 4 == 3)

    def _features(self, index: int) -> dict[str, float]:
        return sentence_features({"unit_id": f"fixture-{index}"}, self.names)

    def _probability(self, features: dict[str, float]) -> float:
        return max(
            1e-6,
            min(
                1 - 1e-6,
                0.4
                + sum(
                    self.state["weights"][name] * features[name] for name in self.state["active"]
                ),
            ),
        )

    def predict_block(self, block: int) -> None:
        """Save twelve bank forecasts and eight admission forecasts before labels open."""
        if block != len(self.state["completed_blocks"]):
            raise ValueError("block_order")
        if len(self.state["queue"]) + 12 > 12:
            self.state["overflow"] += 1
            for kind, count in (("update", 12), ("admission", 8)):
                for index in range(count):
                    self.state["rows"].append(
                        {
                            "arm": self.arm,
                            "block": block,
                            "kind": kind,
                            "index": index,
                            "event_id": f"{self.arm}-{block}-{kind}-{index}",
                            "features": None,
                            "probability": None,
                            "decision": None,
                            "label": None,
                            "label_arrival_block": None,
                            "excluded": True,
                            "censored": True,
                            "status": "unstarted_overflow",
                        }
                    )
            self.state["lifecycle"].append(
                {
                    "block": block,
                    "event": "overflow",
                    "count": 20,
                    "bank_hash": self.bank.state_hash,
                }
            )
            self._save()
            raise ValueError("pending_overflow")
        for kind, count in (("update", 12), ("admission", 8)):
            for index in range(count):
                started = time.monotonic()
                features = self._features(index)
                probability = self._probability(features)
                self.state["lookup_s"] += time.monotonic() - started
                event_id = f"{self.arm}-{block}-{kind}-{index}"
                row = {
                    "arm": self.arm,
                    "block": block,
                    "kind": kind,
                    "index": index,
                    "event_id": event_id,
                    "features": features,
                    "probability": probability,
                    "decision": probability >= 0.5,
                    "label": None,
                    "label_arrival_block": None,
                    "excluded": False,
                    "censored": False,
                }
                if kind == "update":
                    primitives = grammar()["primitives"]
                    payload = {
                        name: features[self.names[i]] if i < 4 else 0.0
                        for i, name in enumerate(primitives)
                    }
                    ack = time.monotonic()
                    self.bank.predict(
                        event_id, block * 100 + index, payload, 0.4, "unknown", "update", event_id
                    )
                    self.state["acknowledgement_s"] += time.monotonic() - ack
                    self.state["queue"].append(event_id)
                self.state["rows"].append(row)
        self.state["completed_blocks"].append(block)
        self.state["lifecycle"].append(
            {
                "block": block,
                "event": "predictions_saved",
                "bank_hash": self.bank.state_hash,
                "model_hash": canonical_hash(self.state["weights"]),
            }
        )
        self._save()

    def release_block(self, block: int, arrival: int) -> None:
        """Spend one block credit after releasing only that block's saved labels."""
        if arrival != block + 1 or block in self.state["released_blocks"]:
            raise ValueError("credit_reused")
        if block not in self.state["completed_blocks"]:
            raise ValueError("premature_label_access")
        block_rows = [row for row in self.state["rows"] if row["block"] == block]
        if len(block_rows) != 20 or len(self.state["queue"]) != 12:
            raise ValueError("queue_invalid")
        for row in block_rows:
            index = row["index"]
            mapped = (
                index
                if self.arm != "shuffled"
                else (index * 5 + 3) % 12
                if row["kind"] == "update"
                else (index * 3 + 1) % 8
            )
            label = self.label(block, mapped)
            row["label"] = label
            row["label_arrival_block"] = arrival
            self.state["rng_counter"] += 1
            if row["kind"] == "update":
                started = time.monotonic()
                ack = time.monotonic()
                self.bank.release(row["event_id"], label, arrival * 100 + index)
                self.state["acknowledgement_s"] += time.monotonic() - ack
                self.state["queue"].remove(row["event_id"])
                for name in self.state["active"]:
                    feature = row["features"][name]
                    self.state["weights"][name] = max(
                        -0.4,
                        min(
                            0.4,
                            self.state["weights"][name]
                            - 0.05 * (row["probability"] - label) * feature,
                        ),
                    )
                self.state["update_calls"] += 1
                self.state["update_s"] += time.monotonic() - started
        if self.arm in {"adaptive", "shuffled"} and self.state["credits"] < 8:
            proposal = self.bank.propose()
            if proposal is not None:
                primitives = grammar()["primitives"]
                pair = tuple(primitives.index(name) for name in proposal["pair"])
                pair_map = {(0, 1): 4, (0, 2): 5, (0, 3): 6, (1, 2): 7, (1, 3): 8, (2, 3): 9}
                name = self.names[pair_map[pair]] if pair in pair_map else None
                admission_rows = [row for row in block_rows if row["kind"] == "admission"]
                frozen = [
                    {
                        "unit_id": row["event_id"],
                        "base": row["probability"],
                        "candidate": max(
                            1e-6,
                            min(
                                1 - 1e-6,
                                row["probability"]
                                + proposal["weight"] * row["features"].get(name, 0.0),
                            ),
                        ),
                    }
                    for row in admission_rows
                ]
                result = admission_decision(frozen, [row["label"] for row in admission_rows])
                accepted = bool(
                    self.admit and name and name not in self.state["active"] and result["accepted"]
                )
                ack = time.monotonic()
                self.bank.admit(accepted)
                self.state["acknowledgement_s"] += time.monotonic() - ack
                if accepted:
                    self.state["active"].append(name)
                    self.state["weights"][name] = proposal["weight"]
                    self.state["admitted_predicates"].append(name)
                self.state["lifecycle"].append(
                    {
                        "block": block,
                        "event": "admission",
                        "predicate": name,
                        "accepted": accepted,
                        "reason": "admitted" if accepted else "rejected",
                        "credit": self.state["credits"] + 1,
                        "bank_hash": self.bank.state_hash,
                    }
                )
        self.state["credits"] += 1
        self.state["released_blocks"].append(block)
        self.state["lifecycle"].append(
            {
                "block": block,
                "arrival_block": arrival,
                "event": "feedback_released",
                "count": 20,
                "bank_hash": self.bank.state_hash,
            }
        )
        self._save()

    def expire_block(self, block: int, arrival: int) -> None:
        """Close missing feedback with one visible censored row per item."""
        if arrival != block + 1 or block in self.state["released_blocks"]:
            raise ValueError("credit_reused")
        if block not in self.state["completed_blocks"]:
            raise ValueError("premature_label_access")
        for row in (row for row in self.state["rows"] if row["block"] == block):
            if row["kind"] == "update":
                self.bank.mark_missing(row["event_id"], arrival * 100 + row["index"])
                self.state["queue"].remove(row["event_id"])
            row["censored"] = True
            row["label_arrival_block"] = arrival
            self.state["expired"] += 1
        self.state["credits"] += 1
        self.state["released_blocks"].append(block)
        self.state["lifecycle"].append(
            {
                "block": block,
                "arrival_block": arrival,
                "event": "feedback_expired",
                "count": 20,
                "bank_hash": self.bank.state_hash,
            }
        )
        self._save()

    def finish(self) -> dict:
        """Reduce final row and state hashes without multiplying family counts."""
        if self.state["queue"] or len(self.state["released_blocks"]) != 8:
            raise ValueError("unfinished_feedback")
        rows = self.state["rows"]
        changed = any(
            row["decision"] and row["probability"] > 0.5 and row["block"] > 0
            for row in rows
            if row["kind"] == "update"
        )
        return {
            "arm": self.arm,
            "rows": rows,
            "lifecycle": self.state["lifecycle"],
            "initial_active": self.state["initial_active"],
            "admitted_predicates": self.state["admitted_predicates"],
            "decision_changed_after_admission": changed if self.arm == "adaptive" else None,
            "update_calls": self.state["update_calls"],
            "parameter_ceiling": 16,
            "credits": self.state["credits"],
            "rng_counter": self.state["rng_counter"],
            "overflow": self.state["overflow"],
            "expired": self.state["expired"],
            "decision_hash": canonical_hash(
                [(r["event_id"], r["decision"], r["probability"]) for r in rows]
            ),
            "model_hash": canonical_hash(self.state["weights"]),
            "queue_hash": canonical_hash(self.state["queue"]),
            "bank_hash": self.bank.state_hash,
            "state_path": str(self.state_path),
            "bank_path": str(self.bank_path),
            "timings_s": {
                key: self.state[key]
                for key in ("lookup_s", "update_s", "serialization_s", "acknowledgement_s")
            },
        }


def run_arm(folder: Path, names: list[str], arm: str) -> dict:
    """Run one arm with a private production bank and fixed call budget."""
    folder.mkdir(parents=True, exist_ok=True)
    runner = OnlineRunner(folder / "runner.json", folder / "bank.json", names, arm)
    started = time.monotonic()
    for block in range(len(runner.state["completed_blocks"]), 8):
        if block:
            runner.release_block(block - 1, block)
        runner.predict_block(block)
        progress(started, f"{arm}_block", "complete", block + 1)
    runner.release_block(7, 8)
    return runner.finish()


def run_fixture(folder: Path, names: list[str]) -> dict:
    """Freeze the same eight blocks for four bounded comparison arms."""
    folder.mkdir(parents=True, exist_ok=True)
    arms = {arm: run_arm(folder / arm, names, arm) for arm in ARMS}
    raw = {
        "schema": "exp7760-raw-v1",
        "names": names,
        "arms": arms,
        "seed": 7760,
        "blocks": 8,
        "update_items_per_block": 12,
        "admission_items_per_block": 8,
        "raw_path": str(folder / "raw.json"),
    }
    atomic_json(folder / "raw.json", raw)
    return raw


def cold_reduce(path: Path) -> dict:
    """Reopen raw rows and bank ledgers without using the runner's summaries."""
    raw = json.loads(path.read_text())
    if raw.get("schema") != "exp7760-raw-v1" or set(raw["arms"]) != set(ARMS):
        raise ValueError("raw_scope_invalid")
    for arm, result in raw["arms"].items():
        rows = result["rows"]
        state = json.loads(Path(result["state_path"]).read_text())
        bank = Bank(
            Path(result["bank_path"]),
            grammar(),
            "priority" if arm in {"adaptive", "shuffled"} else "read_only",
            0.1,
            2,
        )
        if len(rows) != 160 or state["rows"] != rows or bank.state_hash != result["bank_hash"]:
            raise ValueError("raw_rows_invalid")
        if result["decision_hash"] != canonical_hash(
            [(r["event_id"], r["decision"], r["probability"]) for r in rows]
        ):
            raise ValueError("decision_hash_invalid")
        if result["model_hash"] != canonical_hash(state["weights"]):
            raise ValueError("model_hash_invalid")
        if result["queue_hash"] != canonical_hash(state["queue"]) or state["queue"]:
            raise ValueError("queue_hash_invalid")
        if result["credits"] != 8 or result["update_calls"] != 96 or result["rng_counter"] != 160:
            raise ValueError("budget_invalid")
        for row in rows:
            if row["label_arrival_block"] != row["block"] + 1:
                raise ValueError("feedback_delay_invalid")
            mapped = (
                row["index"]
                if arm != "shuffled"
                else (row["index"] * 5 + 3) % 12
                if row["kind"] == "update"
                else (row["index"] * 3 + 1) % 8
            )
            if row["label"] != int(mapped % 4 == 3):
                raise ValueError("label_invalid")
        bank.replay_ledger()
    return {"valid": True, "arm_count": 4, "row_count": 640, "raw_sha256": sha256_file(path)}


def hard_exit_commit_check(folder: Path, names: list[str]) -> bool:
    """Inspect state after owned hard exits before and after durable commit."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "bank.json"
    started = time.monotonic()
    children: list[dict] = []

    def record(stage: str, argv: list[str], result: subprocess.CompletedProcess[str]) -> None:
        log = folder / f"{stage}.log"
        log.write_text(result.stdout + result.stderr)
        children.append(
            {
                "stage": stage,
                "command_argv": argv,
                "exit_code": result.returncode,
                "log_path": str(log),
                "log_sha256": sha256_file(log),
            }
        )
        atomic_json(folder / "receipt.json", {"children": children})

    code = """from pathlib import Path
import os,sys
from carnot.reporting.constraint_bank_protocol import Bank,grammar
b=Bank(Path(sys.argv[1]),grammar(),'priority',0.1,2)
f={n:float(i<2) for i,n in enumerate(grammar()['primitives'])}
b.predict('case',0,f,0.4,'unknown','update','case')
b.release('case',1,10)
assert b.propose() is not None
os._exit(0)
"""
    progress(started, "hard_exit_before", "before_subprocess", 0)
    before_argv = [sys.executable, "-u", "-c", code, str(path)]
    child = subprocess.run(
        before_argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    progress(started, "hard_exit_before", "after_subprocess", 1)
    record("before_commit", before_argv, child)
    if child.returncode:
        return False
    reopened = Bank(path, grammar(), "priority", 0.1, 2)
    before = not reopened.state["templates"] and reopened.state["proposal"] is not None
    after_code = """from pathlib import Path
import os,sys
from carnot.reporting.constraint_bank_protocol import Bank,grammar
b=Bank(Path(sys.argv[1]),grammar(),'priority',0.1,2)
assert b.state['proposal'] is not None
b.admit(True)
os._exit(0)
"""
    progress(started, "hard_exit_after", "before_subprocess", 1)
    after_argv = [sys.executable, "-u", "-c", after_code, str(path)]
    child = subprocess.run(after_argv, check=False, capture_output=True, text=True, timeout=30)
    progress(started, "hard_exit_after", "after_subprocess", 2)
    record("after_commit", after_argv, child)
    if child.returncode:
        return False
    reopened = Bank(path, grammar(), "priority", 0.1, 2)
    return bool(
        before
        and len(reopened.state["templates"]) == 1
        and reopened.state["proposal"] is None
        and reopened.replay_ledger()
        and names
    )


def build_artifact(
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    measured: dict | None,
    spans: list[dict],
    scope: dict,
) -> dict:
    """Keep protocol readiness separate from natural scientific benefit."""
    failed = [row for row in checks if not row["passed"]]
    rows = [row for arm in measured["arms"].values() for row in arm["rows"]] if measured else []
    lifecycle = (
        [row for arm in measured["arms"].values() for row in arm["lifecycle"]] if measured else []
    )
    timings = (
        {
            key: sum(arm["timings_s"][key] for arm in measured["arms"].values())
            for key in ("lookup_s", "update_s", "serialization_s", "acknowledgement_s")
        }
        if measured
        else {}
    )
    principles = {
        "experiment_id": "An artifact must have a unique current owner.",
        "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
        "verdict_class": "The claim class travels with the evidence.",
        "flagged_adversarial": "Invalid evidence must not open downstream gates.",
        "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
        "rows": "Aggregates must be recomputable without rerunning science.",
        "acceptance_gate_results": "A working protocol is not evidence of benefit.",
        "duration_s": "Duration must describe actual work without padding.",
        "random_seed": "A third party needs the same experiment inputs.",
        "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
        "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
        "preconditions_checked": "Access and validity must be established before expensive work.",
        "validation_receipts": "All registered checks must pass before readiness opens.",
        "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
        "inference_substrate_class": "Duration floors must match the invoked substrate.",
        "MODEL_SPECS": "A cited upstream model is not a current invocation.",
        "online_runtime_ready_score": "A bank file alone does not prove a working learning loop.",
        "online_protocol_path": "Natural learning must follow a tested causal contract.",
        "lifecycle_rows": "Every update needs a prior prediction and authorized feedback.",
    }
    return {
        "schema": "carnot.exp7760.v675.online_runner.v1",
        "experiment_id": 7760,
        "milestone": "2026.09.675",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_precondition"
        if failed
        else "complete_circular_positive_online_fixture",
        "verdict_class": "blocked" if failed else "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "lifecycle_rows": lifecycle,
        "acceptance_gate_results": {
            "validity": not failed,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": 7760,
        "reproducibility_checksum": canonical_hash(
            {
                "seed": 7760,
                "inputs": hashes,
                "code": sha256_file(ROOT / MODULE),
                "roles": ["update", "admission"],
                "parameters": {"blocks": 8, "updates": 12, "admissions": 8, "capacity": 12},
            }
        ),
        "sample_size_budget": {
            "intended": 640,
            "eligible": 640 if measured else 0,
            "started": len(rows),
            "completed": sum(row["label"] is not None for row in rows),
            "excluded": sum(bool(row["excluded"]) for row in rows),
            "censored": sum(bool(row["censored"]) for row in rows),
            "effective_independent_n": 0,
            "fixture_blocks": 8,
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {"frozen_affected_scope": scope},
        "verifier_is_oracle": True,
        "claim_scope": "synthetic_delayed_feedback_fixture_only; natural_benefit_unmeasured",
        "field_principles": principles,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "online_runtime_ready_score": 0,
        "online_protocol_path": {
            "queue_capacity": 12,
            "delay_blocks": 1,
            "update_items_per_block": 12,
            "admission_items_per_block": 8,
            "max_proposal_credits": 8,
            "proposal_credits_per_block": 1,
            "complete_static_predicates": 16,
            "production_bank_sha256": hashes["module_hashes"].get(BANK_MODULE),
            "runner_sha256": sha256_file(ROOT / MODULE),
        },
        "timings_s": timings,
        "raw_rows_path": str(measured["raw_path"]) if measured else None,
    }


def run_experiment(
    root: Path, date: str, output: Path, *, raw: Path | None = None, validate: bool = True
) -> dict:
    """Measure, replay, validate, and atomically publish the terminal record."""
    started = time.monotonic()
    raw = raw or root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope_path = raw / "affected_scope.json"
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    spans: list[dict] = []
    progress(started, "preflight", "before", 0)
    begin = time.monotonic()
    checks, hashes, names = preflight(root)
    spans.append(
        {
            "phase": "preflight",
            "start_s": begin - started,
            "duration_s": time.monotonic() - begin,
            "completed_units": len(checks),
        }
    )
    progress(started, "preflight", "after", len(checks))
    if any(not row["passed"] for row in checks):
        artifact = build_artifact(date, started, checks, hashes, None, spans, scope)
        atomic_json(output, artifact)
        return artifact
    progress(started, "measurement", "before", 0)
    begin = time.monotonic()
    measurement = Path(tempfile.mkdtemp(prefix="fixture-", dir=raw))
    measured = run_fixture(measurement, names)
    spans.append(
        {
            "phase": "measurement",
            "start_s": begin - started,
            "duration_s": time.monotonic() - begin,
            "completed_units": 640,
            "checkpoint_sha256": sha256_file(measurement / "raw.json"),
        }
    )
    progress(started, "measurement", "after", 640)
    progress(started, "cold_replay", "before", 0)
    begin = time.monotonic()
    replay = cold_reduce(measurement / "raw.json")
    spans.append(
        {
            "phase": "cold_replay",
            "start_s": begin - started,
            "duration_s": time.monotonic() - begin,
            "completed_units": 640,
            "checkpoint_sha256": replay["raw_sha256"],
        }
    )
    progress(started, "cold_replay", "after", 640)
    artifact = build_artifact(date, started, checks, hashes, measured, spans, scope)
    artifact["validation_receipts"]["raw_reduction"] = replay
    if validate:
        progress(started, "cold_restart", "before", 0)
        begin = time.monotonic()
        staged = OnlineRunner(
            measurement / "restart.json", measurement / "restart-bank.json", names, "adaptive"
        )
        for block in range(4):
            if block:
                staged.release_block(block - 1, block)
            staged.predict_block(block)
        python = str(root / ".venv/bin/python")
        restart = run_commands(
            root,
            [
                CommandSpec(
                    "cold_restart",
                    (
                        python,
                        "-u",
                        "-m",
                        "carnot.experiment_7760_v675_online_runner",
                        "--resume-private",
                        str(staged.state_path),
                        str(staged.bank_path),
                    ),
                    "owned_restart",
                    300,
                )
            ],
            log_dir=measurement / "validation/restart",
            heartbeat_s=30,
        )
        try:
            restarted = (
                json.loads(restart[0]["output_tail"].splitlines()[-1])
                if restart[0]["passed"]
                else {}
            )
        except (ValueError, IndexError):
            restarted = {}
        reference = measured["arms"]["adaptive"]
        parity_fields = (
            "decision_hash",
            "bank_hash",
            "queue_hash",
            "credits",
            "rng_counter",
            "model_hash",
        )
        parity = restart[0]["passed"] and all(
            restarted.get(key) == reference[key] for key in parity_fields
        )
        artifact["validation_receipts"]["cold_restart"] = {
            "child": restart[0],
            "parity": parity,
            "fields": list(parity_fields),
        }
        spans.append(
            {
                "phase": "cold_restart",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": 4,
            }
        )
        progress(started, "cold_restart", "after", 4)
        progress(started, "hard_exit", "before", 0)
        begin = time.monotonic()
        hard_exit = hard_exit_commit_check(measurement / "hard_exit", names)
        hard_exit_receipt = measurement / "hard_exit/receipt.json"
        artifact["validation_receipts"]["hard_exit_commit"] = {
            "passed": hard_exit,
            "path": str(hard_exit_receipt),
            "sha256": sha256_file(hard_exit_receipt) if hard_exit_receipt.is_file() else None,
        }
        spans.append(
            {
                "phase": "hard_exit",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": 2,
            }
        )
        progress(started, "hard_exit", "after", 2)
        progress(started, "validation", "before", 0)
        begin = time.monotonic()
        private = Path(tempfile.mkdtemp(prefix="exp7760-validation-"))
        basetemp = private / "basetemp"
        basetemp.mkdir(parents=True)
        probe = run_commands(
            root,
            [
                CommandSpec(
                    "basetemp_parent_probe",
                    (
                        python,
                        "-u",
                        "-c",
                        "from pathlib import Path; import sys; assert Path(sys.argv[1]).is_dir(); print('basetemp_ok')",
                        str(basetemp),
                    ),
                    "private_basetemp_parent",
                    30,
                )
            ],
            log_dir=measurement / "validation/probe",
            heartbeat_s=30,
        )
        commands = build_scoped_commands(
            root,
            [TEST],
            [MODULE],
            static_paths=[CLI],
            basetemp=basetemp,
            coverage_file=private / ".coverage",
        )
        affected = run_commands(
            root,
            commands,
            log_dir=measurement / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["affected"] = affected
        artifact["validation_receipts"]["basetemp_parent_probe"] = probe
        spans.append(
            {
                "phase": "validation",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": len(affected) + 1,
            }
        )
        progress(started, "validation", "after", len(affected) + 1)
        progress(started, "full_suite", "before_subprocess", 0)
        begin = time.monotonic()
        full_suite = run_commands(
            root,
            [
                CommandSpec(
                    "full_python_suite",
                    (
                        str(root / ".venv/bin/pytest"),
                        "tests/python",
                        "-q",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={basetemp / 'full'}",
                    ),
                    "full_python_suite",
                    2400,
                )
            ],
            log_dir=measurement / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["full_python_suite"] = full_suite
        artifact["validation_receipts"]["repository_health"] = {
            "full_python_suite_passed": all(row["passed"] for row in full_suite),
            "scope": "repository_wide_observation_separate_from_affected_validation",
        }
        spans.append(
            {
                "phase": "full_suite",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": 1,
            }
        )
        progress(started, "full_suite", "after_subprocess", 1)
        progress(started, "task_e2e", "before_subprocess", 0)
        begin = time.monotonic()
        e2e = run_commands(
            root,
            [
                CommandSpec(
                    "task_e2e",
                    (python, "-u", CLI, "--private-e2e"),
                    "real_entrypoint_private_state",
                    300,
                )
            ],
            log_dir=measurement / "validation/e2e",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["task_e2e"] = e2e
        spans.append(
            {
                "phase": "task_e2e",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": 1,
            }
        )
        progress(started, "task_e2e", "after_subprocess", 1)
        prerequisite_passed = (
            parity and hard_exit and all(row["passed"] for row in [*probe, *affected, *e2e])
        )
        artifact["online_runtime_ready_score"] = int(prerequisite_passed)
        artifact["acceptance_gate_results"]["readiness"] = bool(prerequisite_passed)
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        candidate = measurement / "terminal_candidate.json"
        atomic_json(candidate, artifact)
        progress(started, "terminal_readers", "before_subprocess", 0)
        begin = time.monotonic()
        terminal = run_commands(
            root,
            [
                CommandSpec(
                    "cold_reduce",
                    (
                        python,
                        "-u",
                        "-m",
                        "carnot.experiment_7760_v675_online_runner",
                        "--cold-reduce",
                        str(candidate),
                    ),
                    "exact_candidate",
                    120,
                ),
                CommandSpec(
                    "adversarial_verify",
                    (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                    "exact_candidate",
                    120,
                ),
                CommandSpec(
                    "verdict_row_consistency",
                    (
                        python,
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "exact_candidate",
                    120,
                ),
            ],
            log_dir=measurement / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        spans.append(
            {
                "phase": "terminal_readers",
                "start_s": begin - started,
                "duration_s": time.monotonic() - begin,
                "completed_units": len(terminal),
                "checkpoint_sha256": sha256_file(candidate),
            }
        )
        progress(started, "terminal_readers", "after_subprocess", len(terminal))
        failed = [row for row in [*probe, *affected, *e2e, *terminal] if not row["passed"]]
        if not parity:
            artifact["gate_check_summary"].append(
                check("current", str(staged.state_path), "cold_restart_parity", True, False)
            )
        if not hard_exit:
            artifact["gate_check_summary"].append(
                check("current", str(measurement / "hard_exit"), "hard_exit_commit", True, False)
            )
        artifact["gate_check_summary"].extend(
            check("current", row["log_path"], "exit_code", 0, row["exit_code"]) for row in failed
        )
        if failed or not parity or not hard_exit:
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["online_runtime_ready_score"] = 0
            artifact["acceptance_gate_results"]["validity"] = False
            artifact["acceptance_gate_results"]["readiness"] = False
        artifact["flagged_adversarial"] = any(
            row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
        )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before_atomic_write", len(artifact["rows"]))
    atomic_json(output, artifact)
    progress(started, "publish", "after_atomic_write", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Dispatch the production run and bounded private child checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--resume-private", nargs=2, type=Path)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_reduce:
        candidate = json.loads(args.cold_reduce.read_text())
        sources = candidate["source_artifact_hashes"]
        for category in ("eligible_producers", "module_hashes", "pre_gate_receipts"):
            for relative, expected in sources[category].items():
                if sha256_file(ROOT / relative) != expected:
                    raise ValueError("candidate_source_hash_invalid")
        raw_path = Path(candidate["raw_rows_path"])
        raw = json.loads(raw_path.read_text())
        expected_rows = {
            row["event_id"]: row for arm in raw["arms"].values() for row in arm["rows"]
        }
        actual_rows = {row["event_id"]: row for row in candidate["rows"]}
        if len(expected_rows) != len(candidate["rows"]) or expected_rows != actual_rows:
            raise ValueError("candidate_rows_invalid")
        print(json.dumps(cold_reduce(raw_path), sort_keys=True), flush=True)
        return 0
    if args.resume_private:
        state = json.loads(args.resume_private[0].read_text())
        runner = OnlineRunner(
            args.resume_private[0], args.resume_private[1], state["names"], state["arm"]
        )
        for block in range(len(runner.state["completed_blocks"]), 8):
            runner.release_block(block - 1, block)
            runner.predict_block(block)
        runner.release_block(7, 8)
        result = runner.finish()
        parity_fields = (
            "decision_hash",
            "bank_hash",
            "queue_hash",
            "credits",
            "rng_counter",
            "model_hash",
        )
        print(
            json.dumps(
                {key: result[key] for key in parity_fields if key in result}, sort_keys=True
            ),
            flush=True,
        )
        return 0
    if args.private_e2e:
        with tempfile.TemporaryDirectory(prefix="exp7760-e2e-") as directory:
            result = run_fixture(Path(directory), preflight(args.root)[2])
            reduced = cold_reduce(Path(result["raw_path"]))
            print(json.dumps(reduced, sort_keys=True), flush=True)
        return 0
    run_experiment(
        args.root,
        args.date,
        args.root / args.output,
        raw=args.root / args.raw,
        validate=not args.no_validate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by real child entrypoint.
    raise SystemExit(main())
