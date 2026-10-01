"""REQ-REPORT-7981: capture new stream inputs through the qualified protocol.

The adapter keeps the existing worker, lease, validation and publication
machinery fixed. Only public input roles and their separate readiness change.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot import experiment_7969_v691_qwen_calibration_capture as legacy
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_stream_capture_7981 as capture

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_7981_v692_qwen_stream_capture"
TASK = "exp7981-qwen-stream-capture"
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
UPSTREAM = "results/experiment_7980_v692_evidence_features.json"
UPSTREAM_PIN = "sha256:dc06fadccb5a0bfce0b545256a9e8133e61f002df642b2d71d829753322438ed"
HISTORY = "results/experiment_7969_v691_qwen_calibration_capture.json"
HISTORY_PIN = "sha256:3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51"
REVISION = "fe1e2a23d973adb629709749dc4f6756df66ef10"
MODEL_PIN = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_stream_capture_7981.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_qwen_stream_capture_7981.py",
    f"tests/python/test_{NAME}.py",
    *legacy.TESTS,
]
IMPORTS = [
    *legacy.IMPORTS,
    "carnot.experiment_7969_v691_qwen_calibration_capture",
    "carnot.verify.qwen_calibration_capture_7969",
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
reference, operand = legacy.reference, legacy.operand
_BASE, _BUILD, _FREEZE = legacy.base, legacy.build, legacy.freeze_commands
_LIVE, _VALIDATE, _REPLAY = legacy.live_capture, legacy.apply_validation, legacy.replay


def load_public(manifests: Json) -> Json:
    """Open authenticated public files only; evaluator annotations stay sealed."""
    views = {}
    for role, item in manifests.items():
        path = legacy.prior.targets.prior.checked_reference(item)
        view = json.loads(path.read_text())
        if role == capture.FRESH:
            view = dict(
                role=role,
                request_rows=view["request_rows"],
                boundaries=[
                    dict(family_id=r["family_id"], public_eligible=True, exclusion_reason=None)
                    for r in view["request_rows"]
                ],
            )
        views[role] = view
    capture.freeze(views)
    return views


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Bind both producers' original dates and exact bytes before admission."""
    checks: list[Json] = []
    upstream: Json = {}
    for label, relative, pin, expected in [
        (
            "exp7980",
            UPSTREAM,
            UPSTREAM_PIN,
            dict(
                experiment_id=7980,
                task_id="exp7980-evidence-features",
                milestone="2026.10.692",
                run_date="20261001",
                execution_date="20261001",
                verdict_class="null",
                flagged_adversarial=False,
            ),
        ),
        (
            "exp7969",
            HISTORY,
            HISTORY_PIN,
            dict(
                experiment_id=7969,
                task_id="exp7969-qwen-calibration-capture",
                milestone="2026.10.691",
                run_date="20261001",
                execution_date="20261001",
                qwen_capture_ready_score=1,
                verdict_class="null",
                flagged_adversarial=False,
                model_revision=REVISION,
                gguf_sha256=MODEL_PIN,
            ),
        ),
    ]:
        path = root / relative
        checks.append(
            operand(label, path, "sha256", pin, sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if checks[-1]["passed"] else {}
        upstream[label] = value
        checks.extend(operand(label, path, k, v, value.get(k)) for k, v in expected.items())
    upstream["branch_gate_checks"] = [
        operand("exp7980", root / UPSTREAM, field, 1, upstream["exp7980"].get(field))
        for field in ("fresh_panel_ready_score", "feature_views_ready_score")
    ]
    if all(c["passed"] for c in checks):
        try:
            source, history = upstream["exp7980"], upstream["exp7969"]
            manifests = {}
            for field, roles in [
                ("feature_views_ready_score", list(capture.ROLES)),
                ("fresh_panel_ready_score", [capture.FRESH]),
            ]:
                if source.get(field) != 1:
                    continue
                branch: Json = {}
                branch_checks = []
                try:
                    for role in roles:
                        item = (
                            source["fresh_public_manifest"]
                            if role == capture.FRESH
                            else source["public_role_manifests"][role]
                        )
                        path = Path(item["path"])
                        branch_checks.append(
                            operand(
                                "exp7980",
                                path,
                                "sha256",
                                item["sha256"],
                                sha256_file(path) if path.is_file() else None,
                            )
                        )
                        branch_checks.append(
                            operand(
                                "exp7980",
                                path,
                                "declared_role",
                                "fresh" if role == capture.FRESH else role,
                                path.stem,
                            )
                        )
                        branch[role] = item
                    load_public(branch)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    branch_checks.append(
                        operand(
                            "exp7980",
                            root / UPSTREAM,
                            field + ".authenticated_public",
                            True,
                            str(error),
                        )
                    )
                upstream["branch_gate_checks"].extend(branch_checks)
                if all(c["passed"] for c in branch_checks):
                    manifests.update(branch)
                    checks.extend(branch_checks)
            checks.append(
                operand(
                    "exp7980", root / UPSTREAM, "any_public_branch_available", True, bool(manifests)
                )
            )
            if manifests:
                load_public(manifests)
            decoder = {
                k: capture.config()[k]
                for k in ("seed", "temperature", "max_tokens", "input_tokens", "grammar_sha256")
            }
            checks.append(
                operand(
                    "exp7969",
                    root / HISTORY,
                    "decoder_protocol",
                    decoder,
                    {k: history["capture_budget"].get(k) for k in decoder},
                )
            )
            protocol = dict(
                decoder=decoder,
                system=capture.risk.SYSTEM,
                model_revision=REVISION,
                gguf_sha256=MODEL_PIN,
            )
            checks.append(
                operand(
                    "exp7969",
                    root / HISTORY,
                    "protocol_fingerprint",
                    canonical_hash(protocol),
                    history["protocol_fingerprint"],
                )
            )
            upstream.update(public_role_manifests=manifests, protocol=protocol, exp7958=history)
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks.append(
                operand(
                    "exp7980", root / UPSTREAM, "authenticated_public_protocol", True, str(error)
                )
            )
    import yaml

    exclusion = root / "ops/exclusion_manifest.yaml"
    document = yaml.safe_load(exclusion.read_text()) if exclusion.is_file() else {}
    for eid in (7969, 7980):
        retired = (
            any(
                r.get("experiment_id") == eid
                for key in ("retired", "retired_experiments")
                for r in document.get(key, [])
            )
            if exclusion.is_file()
            else None
        )
        checks.append(operand("exclusion_manifest", exclusion, f"exp{eid}_retired", False, retired))
    upstream["checks"] = checks
    return [r for r in checks if not r["passed"]], upstream


def base(failures: list[Json]) -> Json:
    """Keep actual blocked model calls zero and planned compute explicit."""
    value = _BASE(failures)
    value.update(capture.reduce([]))
    value.update(
        schema="carnot.exp7981.qwen_stream_capture.v1",
        experiment_id=7981,
        task_id=TASK,
        milestone="2026.10.692",
        capture_budget=capture.config(),
        token_budget=capture.config(),
        raw_shard_hashes=[],
        branch_gate_checks=[],
        planned_model_specs=MODEL_SPECS,
        cleanup=dict(
            leak_free=True, owned_processes_started=0, unrelated_process_kill_count_delta=0
        ),
        claim_scope="Bounded full-source stream and reserved judgments only; no calibration, decision benefit or transport quality claim.",
    )
    return value


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Reuse owned hardware work while separating previously started calls."""
    previous = {
        json.loads(p.read_text())["family_id"]
        for p in (raw / "slots").glob("slot-*.json")
        if json.loads(p.read_text())["started"]
    }
    result = _LIVE(plan, raw, scratch)
    result["current_family_ids"] = [
        r["family_id"] for r in result["rows"] if r["started"] and r["family_id"] not in previous
    ]
    if result.get("model_loads_completed"):
        current = set(result["current_family_ids"])
        expected = [
            dict(
                request_sha256=canonical_hash(r["request"]),
                response_sha256=canonical_hash(r["raw_response"]),
                output_tokens=r["raw_response"].get("usage", {}).get("completion_tokens"),
            )
            for r in result["rows"]
            if r["family_id"] in current and r["status"] == "generated"
        ]
        observed = [
            {k: r.get(k) for k in ("request_sha256", "response_sha256", "output_tokens")}
            for r in result.get("runtime_receipts", [])
        ]
        result["checks"].append(
            operand(
                "owned_cuda",
                raw / "request_manifest.json",
                "generated_token_receipt_binding",
                expected,
                observed,
            )
        )
        result["checks"].append(
            operand(
                "owned_cuda",
                raw / "request_manifest.json",
                "bounded_actual_output_tokens",
                True,
                all(
                    isinstance(r["output_tokens"], int) and 0 < r["output_tokens"] <= 96
                    for r in observed
                ),
            )
        )
    return result


def build(raw: Path, result: Json, upstream: Json, *, fixture: Path | None = None) -> Json:
    """Transport success supplies judgments and cannot imply quality benefit."""
    value = _BUILD(raw, result, upstream, fixture=fixture)
    rows = result.get("rows", [])
    value.update(
        experiment_id=7981,
        task_id=TASK,
        milestone="2026.10.692",
        schema="carnot.exp7981.qwen_stream_capture.v1",
    )
    value["raw_shard_hashes"] = value["raw_response_shards"]
    value["request_rows"] = [
        dict(
            family_id=r["family_id"],
            role=r["role"],
            request=r["request"],
            public_hash=r["public_hash"],
        )
        for r in rows
    ]
    value["branch_gate_checks"] = upstream.get("branch_gate_checks", [])
    value["gate_check_summary"] += [r for r in value["branch_gate_checks"] if not r["passed"]]
    history = upstream.get("exp7969", {})
    value["historical_evaluation_provenance"] = dict(
        producer_id=7969,
        primary_path=str(ROOT / HISTORY),
        primary_sha256=HISTORY_PIN,
        scope="historical_protocol_only_no_current_model_calls",
    )
    value["prior_verdicts_unchanged"] = [
        dict(experiment_id=eid, honest_verdict=upstream.get(f"exp{eid}", {}).get("honest_verdict"))
        for eid in (7969, 7980)
    ]
    value["historical_required_failures"] = history.get("historical_required_failures", [])
    value["cited_upstream_artifacts"] = [
        dict(
            reference(ROOT / relative),
            producer_id=eid,
            producer_invocation_date=upstream.get(f"exp{eid}", {}).get("run_date"),
            imported_fields=fields,
        )
        for eid, relative, fields in [
            (
                7980,
                UPSTREAM,
                [
                    "public_role_manifests",
                    "feature_views_ready_score",
                    "fresh_panel_ready_score",
                    "fresh_public_manifest",
                ],
            ),
            (
                7969,
                HISTORY,
                ["model_revision", "gguf_sha256", "protocol_fingerprint", "capture_budget"],
            ),
        ]
        if (ROOT / relative).is_file()
    ] + list(upstream.get("public_role_manifests", {}).values())
    value["claim_scope"] = base([])["claim_scope"]
    if fixture:
        value.update(stream_capture_ready_score=0, fresh_capture_ready_score=0)
    else:
        current = set(
            result.get("current_family_ids", [r["family_id"] for r in rows if r["started"]])
        )
        for destination, selected in [
            (value["model_invocation_counts"], [r for r in rows if r["family_id"] in current]),
            (
                value["resumed_invocation_counts"],
                [r for r in rows if r["family_id"] not in current],
            ),
        ]:
            destination.update(
                generation_calls_attempted=sum(r["started"] for r in selected),
                generation_calls_completed=sum(r["status"] == "generated" for r in selected),
                generation_calls_failed=sum(
                    r["started"] and r["status"] == "failed" for r in selected
                ),
            )
        if value["verdict_class"] == "disqualified" or any(
            not c["passed"] for c in upstream.get("checks", []) + result.get("checks", [])
        ):
            value.update(stream_capture_ready_score=0, fresh_capture_ready_score=0)
        elif rows:
            valid = bool(value["stream_capture_ready_score"] or value["fresh_capture_ready_score"])
            value.update(
                verdict_class="null" if valid else "blocked",
                honest_verdict="complete_null_qwen_stream_capture"
                if valid
                else "complete_blocked_capture_source_floors",
            )
            if not valid:
                value["gate_check_summary"].append(
                    operand(
                        "owned_capture",
                        raw / "request_manifest.json",
                        "any_branch_source_floor",
                        True,
                        False,
                    )
                )
        if value["fresh_capture_ready_score"] and result.get("measured_duration_s", 0) < 10:
            value.update(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_duration_floor",
                stream_capture_ready_score=0,
                fresh_capture_ready_score=0,
            )
    value["qwen_capture_ready_score"] = value["stream_capture_ready_score"]
    value["acceptance_gate_results"].update(
        validity=value["verdict_class"] in {"null", "circular_positive"},
        readiness=bool(value["stream_capture_ready_score"] or value["fresh_capture_ready_score"]),
    )
    return value


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned failed checks clear both branches while health stays separate."""
    _VALIDATE(value, receipts)
    if value["verdict_class"] == "disqualified":
        value.update(stream_capture_ready_score=0, fresh_capture_ready_score=0)


def replay(value: Json) -> Json:
    """Rebuild measured counts even when validation suppressed readiness."""
    with qualified_protocol():
        shadow = dict(value)
        if value["verdict_class"] != "null" and value["raw_response_shards"]:
            rows = [
                json.loads(legacy.prior.targets.prior.checked_reference(item).read_text())
                for item in value["raw_response_shards"]
            ]
            measured = capture.reduce(rows)
            shadow.update(
                {
                    k: measured[k]
                    for k in ("stream_capture_ready_score", "fresh_capture_ready_score")
                }
            )
        reduced = _REPLAY(shadow)
        if value["fresh_capture_ready_score"]:
            if (
                value["verdict_class"] != "null"
                or value["inference_mode"] != "live_gpu"
                or value["flagged_adversarial"]
            ):
                raise ValueError("unsafe_readiness")
            if not value.get("validation_pending"):
                legacy.prior.targets.check_receipts(value)
        return reduced


def freeze_commands(raw: Path, scratch: Path, views: Json) -> Json:
    """Freeze the exact required checks before any model outcome is known."""
    manifest = _FREEZE(raw, scratch, views)
    manifest["commands"] = [r for r in manifest["commands"] if not r["name"].startswith("e2e016")]
    manifest["commands"].insert(
        1,
        dict(
            name="e2e019_private",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "e2e019"),
                "-q",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ],
            expected_exit=0,
            required=True,
            deadline_s=300,
            failure_reason=None,
        ),
    )
    for command in manifest["commands"]:
        if command["name"] == "repository_health":
            command["deadline_s"] = 300
    manifest["historical_e2e016_date"] = None
    atomic_json(raw / "validation_command_manifest.json", manifest)
    return manifest


def publish(output: Path, value: Json) -> None:
    """Bind final validators and both real readers to one atomic primary."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        k: "Bind current identity, exact public bytes, measured work and separate branch validity; capture makes no benefit claim."
        for k in value
    }
    published = publish_primary(output, value, legacy.terminal_check)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            primary_path=str(output),
            primary_sha256=published["primary_sha256"],
            validator=published["sidecar_path"],
        ),
    )
    receipt = reader_receipt(
        TASK,
        output.parent,
        field="stream_capture_ready_score",
        expected=value["stream_capture_ready_score"],
    )
    if not receipt["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", receipt)


@contextmanager
def qualified_protocol() -> Iterator[None]:
    """Scope the input adapter so existing producers keep their own behavior."""
    names = [
        "NAME",
        "TASK",
        "MODEL_SPECS",
        "UPSTREAM",
        "UPSTREAM_PIN",
        "HISTORY",
        "HISTORY_PIN",
        "OWNED",
        "TESTS",
        "IMPORTS",
        "INCLUDE",
        "capture",
        "load_public",
        "authenticate",
        "base",
        "live_capture",
        "build",
        "apply_validation",
        "replay",
        "freeze_commands",
        "publish",
    ]
    with ExitStack() as stack:
        for name in names:
            stack.enter_context(patch.object(legacy, name, globals()[name]))
        yield


def main(argv: list[str] | None = None) -> int:
    """Use the qualified real CLI, including private fixtures and cold replay."""
    with qualified_protocol():
        return legacy.main(argv)
