#!/usr/bin/env python3
"""Private event-occupancy executor for the isolated V644 fixture.

The evaluator implements the domain directly. It intentionally imports no
learner predicate and no V643 rule helper. Its IPC server returns only a query
identity and one Boolean result.

Spec refs: REQ-CL-7330 and SCENARIO-CL-7330-EXECUTOR.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
PUBLIC_SEED = 7_330_101
TOKEN_SEED = 7_330_211
PRIVATE_RULE_SEED = 7_330_307
COHORTS = (
    "stable_rules",
    "announced_changes",
    "return_to_prior_version",
    "unannounced_changes",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _opaque_token(generator: random.Random) -> str:
    """Generate token spelling without consulting rule or request state."""

    return f"opaque-{generator.getrandbits(96):024x}"


def _rule_assignment(generator: random.Random) -> JsonDict:
    """Draw one authority state from the private seed before request evaluation."""

    pairs = [["a", "b"], ["c", "d"]]
    selected = pairs[: generator.randint(1, 2)]
    return {
        "capacity": generator.randint(3, 5),
        "pair_gaps": [{"pair": pair, "minimum_gap": generator.randint(0, 1)} for pair in selected],
        "forbidden_compounds": [],
    }


def _version_schedule(cohort: str, first: str, second: str) -> list[tuple[str, str]]:
    """Return public tokens and private authority labels as separate values."""

    if cohort == "stable_rules":
        return [(first, "authority-a")] * 12
    if cohort == "announced_changes":
        return [(first, "authority-a")] * 6 + [(second, "authority-b")] * 6
    if cohort == "return_to_prior_version":
        return (
            [(first, "authority-a")] * 4
            + [(second, "authority-b")] * 4
            + [(first, "authority-a")] * 4
        )
    return [(first, "authority-a")] * 6 + [(first, "authority-b")] * 6


def _public_request(
    generator: random.Random,
    request_id: str,
    version_token: str,
    index: int,
) -> JsonDict:
    """Build varied public windows with one guaranteed non-overlap assignment."""

    count = 4 + ((index + generator.randint(0, 2)) % 3)
    activities = list("abcdef"[:count])
    horizon = 22
    durations = {name: 1 + generator.randint(0, 1) for name in activities}
    weights = {name: 1 + generator.randint(0, 2) for name in activities}
    allowed: dict[str, list[int]] = {}
    for position, name in enumerate(activities):
        safe = position * 3 + index % 2
        candidates = {
            index % 3,
            (index + position) % 5,
            safe,
            min(safe + 1, horizon - durations[name]),
        }
        while len(candidates) < 4:
            candidates.add(generator.randint(0, horizon - durations[name]))
        allowed[name] = sorted(candidates)[:4]
    return {
        "request_id": request_id,
        "version_token": version_token,
        "activities": activities,
        "allowed_starts": allowed,
        "durations": durations,
        "weights": weights,
        "horizon": horizon,
        "public_revision": index,
    }


def _interval_gap(
    request: Mapping[str, Any], assignments: Mapping[str, int], left: str, right: str
) -> int:
    left_start = int(assignments[left])
    right_start = int(assignments[right])
    left_end = left_start + int(request["durations"][left])
    right_end = right_start + int(request["durations"][right])
    if left_end <= right_start:
        return right_start - left_end
    if right_end <= left_start:
        return left_start - right_end
    return -min(left_end, right_end) + max(left_start, right_start)


def _check(request: Mapping[str, Any], plan: Mapping[str, Any], rules: Mapping[str, Any]) -> bool:
    """Evaluate allowed starts, event occupancy, private gaps, and compounds."""

    if not isinstance(plan, Mapping) or plan.get("request_id") != request.get("request_id"):
        return False
    assignments = plan.get("assignments")
    if not isinstance(assignments, Mapping) or not assignments:
        return False
    activities = set(request.get("activities", []))
    if not set(assignments) <= activities:
        return False
    for name, value in assignments.items():
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if value not in request["allowed_starts"][name]:
            return False
    occupancy = [0] * int(request["horizon"])
    active_by_time: list[set[str]] = [set() for _ in occupancy]
    for name, start in assignments.items():
        end = int(start) + int(request["durations"][name])
        if end > len(occupancy):
            return False
        for moment in range(int(start), end):
            occupancy[moment] += int(request["weights"][name])
            active_by_time[moment].add(str(name))
    if max(occupancy, default=0) > int(rules["capacity"]):
        return False
    for row in rules["pair_gaps"]:
        left, right = row["pair"]
        if left in assignments and right in assignments:
            if _interval_gap(request, assignments, left, right) < int(row["minimum_gap"]):
                return False
    for compound in rules.get("forbidden_compounds", []):
        names = set(compound)
        if names <= set(assignments) and any(names <= active for active in active_by_time):
            return False
    return True


def _acceptance_witness(request: Mapping[str, Any], rules: Mapping[str, Any]) -> JsonDict:
    names = list(request["activities"])
    for values in itertools.product(*(request["allowed_starts"][name] for name in names)):
        plan = {
            "request_id": request["request_id"],
            "assignments": dict(zip(names, values, strict=True)),
        }
        if _check(request, plan, rules):
            return plan
    raise RuntimeError(f"empty_acceptance_region:{request['request_id']}")


def _renamed_twin(request: Mapping[str, Any], twin_id: str) -> tuple[JsonDict, dict[str, str]]:
    names = list(request["activities"])
    rename = {name: f"unit-{index + 1}" for index, name in enumerate(names)}
    twin = {
        **deepcopy(dict(request)),
        "request_id": twin_id,
        "activities": [rename[name] for name in names],
        "allowed_starts": {rename[name]: request["allowed_starts"][name] for name in names},
        "durations": {rename[name]: request["durations"][name] for name in names},
        "weights": {rename[name]: request["weights"][name] for name in names},
    }
    return twin, rename


def _rename_rules(rules: Mapping[str, Any], rename: Mapping[str, str]) -> JsonDict:
    return {
        "capacity": rules["capacity"],
        "pair_gaps": [
            {
                "pair": [rename.get(name, name) for name in row["pair"]],
                "minimum_gap": row["minimum_gap"],
            }
            for row in rules["pair_gaps"]
        ],
        "forbidden_compounds": [
            [rename.get(name, name) for name in names]
            for names in rules.get("forbidden_compounds", [])
        ],
    }


def _stream(
    public_rng: random.Random,
    token_rng: random.Random,
    private_rng: random.Random,
    stream_id: str,
    cohort: str,
) -> tuple[JsonDict, dict[str, JsonDict]]:
    first = _opaque_token(token_rng)
    second = _opaque_token(token_rng)
    schedule = _version_schedule(cohort, first, second)
    authorities = {
        "authority-a": _rule_assignment(private_rng),
        "authority-b": _rule_assignment(private_rng),
    }
    requests: list[JsonDict] = []
    private_records: dict[str, JsonDict] = {}
    for index, (token, authority) in enumerate(schedule):
        request = _public_request(public_rng, f"{stream_id}-request-{index:02d}", token, index)
        rules = authorities[authority]
        witness = _acceptance_witness(request, rules)
        requests.append(request)
        private_records[str(request["request_id"])] = {
            "version_token": token,
            "authority_label": authority,
            "private_rules": rules,
            "acceptance_witness": witness,
            "witness_label": True,
        }
    return {"stream_id": stream_id, "cohort": cohort, "requests": requests}, private_records


def build_manifests(public_path: Path, private_path: Path) -> JsonDict:
    """Seal public requests before any learner proposal can be generated."""

    public_rng = random.Random(PUBLIC_SEED)
    token_rng = random.Random(TOKEN_SEED)
    private_rng = random.Random(PRIVATE_RULE_SEED)
    development: list[JsonDict] = []
    held_out: list[JsonDict] = []
    records: dict[str, JsonDict] = {}
    for index, cohort in enumerate(COHORTS):
        stream, private = _stream(
            public_rng, token_rng, private_rng, f"development-{index:02d}", cohort
        )
        development.append(stream)
        records.update(private)
    for cohort_index, cohort in enumerate(COHORTS):
        for local_index in range(4):
            stream, private = _stream(
                public_rng,
                token_rng,
                private_rng,
                f"held-out-{cohort_index:01d}-{local_index:02d}",
                cohort,
            )
            held_out.append(stream)
            records.update(private)

    live_panel: list[JsonDict] = []
    for cohort_index, cohort in enumerate(COHORTS):
        first = _opaque_token(token_rng)
        second = _opaque_token(token_rng)
        schedule = _version_schedule(cohort, first, second)
        authorities = {
            "authority-a": _rule_assignment(private_rng),
            "authority-b": _rule_assignment(private_rng),
        }
        for local_index in range(6):
            token, authority = schedule[local_index + 4]
            original = _public_request(
                public_rng,
                f"live-{cohort_index}-{local_index}-original",
                token,
                local_index + 4,
            )
            twin, rename = _renamed_twin(original, f"live-{cohort_index}-{local_index}-twin")
            rules = authorities[authority]
            twin_rules = _rename_rules(rules, rename)
            records[str(original["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": rules,
                "acceptance_witness": _acceptance_witness(original, rules),
                "witness_label": True,
            }
            records[str(twin["request_id"])] = {
                "version_token": token,
                "authority_label": authority,
                "private_rules": twin_rules,
                "acceptance_witness": _acceptance_witness(twin, twin_rules),
                "witness_label": True,
            }
            live_panel.append(
                {
                    "panel_id": f"live-{cohort_index}-{local_index}",
                    "cohort": cohort,
                    "presentation_order": (
                        "original_first" if local_index % 2 == 0 else "twin_first"
                    ),
                    "original": original,
                    "twin": twin,
                    "renaming_map": rename,
                }
            )

    challenge_token = _opaque_token(token_rng)
    challenge_request: JsonDict = {
        "request_id": "compound-challenge",
        "version_token": challenge_token,
        "activities": ["a", "b", "c"],
        "allowed_starts": {name: [0, 3] for name in "abc"},
        "durations": {name: 2 for name in "abc"},
        "weights": {name: 1 for name in "abc"},
        "horizon": 6,
        "public_revision": 0,
    }
    challenge_plan = {
        "request_id": challenge_request["request_id"],
        "assignments": {name: 0 for name in "abc"},
    }
    challenge_rules = {
        "capacity": 6,
        "pair_gaps": [],
        "forbidden_compounds": [["a", "b", "c"]],
    }
    records["compound-challenge"] = {
        "version_token": challenge_token,
        "authority_label": "compound-outside-language",
        "private_rules": challenge_rules,
        "acceptance_witness": _acceptance_witness(challenge_request, challenge_rules),
        "witness_label": True,
    }
    public_manifest: JsonDict = {
        "schema": "carnot.experiment_7330.public-manifest.v1",
        "sealed_before_proposals": True,
        "public_seed": PUBLIC_SEED,
        "development_streams": development,
        "held_out_streams": held_out,
        "warmup_requests_per_stream": 4,
        "later_requests_per_stream": 8,
        "live_proposal_panel": live_panel,
        "compound_conflict_challenge": {
            "request": challenge_request,
            "candidate_plan": challenge_plan,
            "outside_acquisition_language": True,
        },
        "downstream_query_budget_per_request": 24,
        "downstream_arm_distribution_identical": True,
    }
    public_manifest["manifest_hash"] = _sha256_json(public_manifest)
    _atomic_json(public_path, public_manifest)
    private_manifest: JsonDict = {
        "schema": "carnot.experiment_7330.evaluator-private-manifest.v1",
        "public_manifest_sha256": _sha256_file(public_path),
        "token_seed_commitment": _sha256_json(TOKEN_SEED),
        "private_rule_seed_commitment": _sha256_json(PRIVATE_RULE_SEED),
        "evaluator_records": records,
        "label_count": len(records),
        "all_acceptance_witnesses_nonempty": all(
            bool(row["acceptance_witness"]["assignments"]) for row in records.values()
        ),
    }
    private_manifest["manifest_hash"] = _sha256_json(private_manifest)
    _atomic_json(private_path, private_manifest)
    return {
        "public_manifest": str(public_path),
        "private_manifest": str(private_path),
        "public_sha256": _sha256_file(public_path),
        "private_sha256": _sha256_file(private_path),
        "development_stream_count": len(development),
        "held_out_stream_count": len(held_out),
        "live_pair_count": len(live_panel),
        "evaluator_label_count": len(records),
    }


def _public_request_index(manifest: Mapping[str, Any]) -> dict[str, JsonDict]:
    index: dict[str, JsonDict] = {}
    for key in ("development_streams", "held_out_streams"):
        for stream in manifest[key]:
            for request in stream["requests"]:
                index[str(request["request_id"])] = request
    for pair in manifest["live_proposal_panel"]:
        for side in ("original", "twin"):
            request = pair[side]
            index[str(request["request_id"])] = request
    challenge = manifest["compound_conflict_challenge"]["request"]
    index[str(challenge["request_id"])] = challenge
    return index


def run_server(
    public_path: Path, private_path: Path, endpoint: Path, receipt_path: Path
) -> JsonDict:
    """Serve one learner connection and retain boundary evidence without rule data."""

    opened: list[str] = []

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            opened.append(os.fsdecode(args[0]))

    sys.addaudithook(audit)
    public_manifest = json.loads(public_path.read_text(encoding="utf-8"))
    private_manifest = json.loads(private_path.read_text(encoding="utf-8"))
    if private_manifest["public_manifest_sha256"] != _sha256_file(public_path):
        raise RuntimeError("public_manifest_hash_mismatch")
    public_requests = _public_request_index(public_manifest)
    private_records = private_manifest["evaluator_records"]
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    if endpoint.exists():
        endpoint.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)
    print(f"[exp7330-evaluator] ready pid={os.getpid()} endpoint={endpoint}", flush=True)
    connection, _address = server.accept()
    reader = connection.makefile("r", encoding="utf-8")
    writer = connection.makefile("w", encoding="utf-8")
    query_rows: list[JsonDict] = []
    for line in reader:
        message = json.loads(line)
        request_id = str(message.get("request_id", ""))
        accepted = False
        if message.get("op") == "check" and request_id in public_requests:
            record = private_records[request_id]
            accepted = _check(
                public_requests[request_id], message.get("plan", {}), record["private_rules"]
            )
        response = {"query_id": message.get("query_id"), "accepted": bool(accepted)}
        writer.write(_canonical_bytes(response).decode("utf-8") + "\n")
        writer.flush()
        query_rows.append(
            {
                "query_id": message.get("query_id"),
                "request_id": request_id,
                "accepted": bool(accepted),
                "response_keys": sorted(response),
            }
        )
    writer.close()
    reader.close()
    connection.close()
    server.close()
    endpoint.unlink(missing_ok=True)
    import_closure = sorted(
        {
            str(Path(module.__file__).resolve())
            for module in sys.modules.values()
            if getattr(module, "__file__", None)
        }
    )
    receipt: JsonDict = {
        "schema": "carnot.experiment_7330.evaluator-boundary-receipt.v1",
        "evaluator_pid": os.getpid(),
        "allowed_input_paths": [str(public_path.resolve()), str(private_path.resolve())],
        "allowed_output_paths": [str(receipt_path.resolve()), str(endpoint.resolve())],
        "public_manifest_sha256": _sha256_file(public_path),
        "private_manifest_sha256": _sha256_file(private_path),
        "import_closure": import_closure,
        "open_file_receipts": sorted(set(opened)),
        "query_rows": query_rows,
        "query_count": len(query_rows),
        "response_keys": ["accepted", "query_id"],
        "returned_private_fields": False,
    }
    _atomic_json(receipt_path, receipt)
    print(f"[exp7330-evaluator] complete queries={len(query_rows)}", flush=True)
    return receipt


def self_test() -> JsonDict:
    """Exercise executor semantics without importing any learned predicate."""

    request = {
        "request_id": "tiny",
        "activities": ["a", "b", "c"],
        "allowed_starts": {name: [0, 1] for name in "abc"},
        "durations": {name: 1 for name in "abc"},
        "weights": {name: 1 for name in "abc"},
        "horizon": 2,
    }
    rules = {
        "capacity": 2,
        "pair_gaps": [{"pair": ["a", "b"], "minimum_gap": 0}],
        "forbidden_compounds": [],
    }
    labels = []
    for values in itertools.product((0, 1), repeat=3):
        plan = {"request_id": "tiny", "assignments": dict(zip("abc", values, strict=True))}
        labels.append(_check(request, plan, rules))
    expected = [False, False, True, True, True, True, False, False]
    malformed = [
        {},
        {"request_id": "wrong", "assignments": {"a": 0}},
        {"request_id": "tiny", "assignments": {}},
        {"request_id": "tiny", "assignments": {"z": 0}},
        {"request_id": "tiny", "assignments": {"a": "0"}},
        {"request_id": "tiny", "assignments": {"a": 9}},
    ]
    probe = {"request_id": "tiny", "assignments": {"a": 0, "b": 0}}
    permissive = {"capacity": 3, "pair_gaps": [], "forbidden_compounds": []}
    different = _check(request, probe, rules) != _check(request, probe, permissive)
    same_rules = _check(request, probe, permissive) == _check(request, probe, deepcopy(permissive))
    suffix_a = "opaque-control-a"
    suffix_b = "opaque-control-b"
    old_suffix_differs = suffix_a.endswith("-b") != suffix_b.endswith("-b")
    controls: JsonDict = {
        "exhaustive_tiny_domain": {
            "enumerated": len(labels),
            "accepted": sum(labels),
            "rejected": len(labels) - sum(labels),
            "expected_labels_match": labels == expected,
        },
        "constraint_mutation_changed_label_count": sum(
            _check(
                request,
                {"request_id": "tiny", "assignments": dict(zip("abc", values, strict=True))},
                rules,
            )
            != _check(
                request,
                {"request_id": "tiny", "assignments": dict(zip("abc", values, strict=True))},
                permissive,
            )
            for values in itertools.product((0, 1), repeat=3)
        ),
        "malformed_rejection_count": sum(not _check(request, plan, rules) for plan in malformed),
        "different_rules_change_label": different,
        "same_rules_different_tokens_same_label": same_rules,
        "old_suffix_lookup_mismatch_exposed": old_suffix_differs and same_rules,
    }
    controls["passed"] = (
        controls["exhaustive_tiny_domain"]["expected_labels_match"]
        and controls["constraint_mutation_changed_label_count"] > 0
        and controls["malformed_rejection_count"] == len(malformed)
        and different
        and same_rules
        and controls["old_suffix_lookup_mismatch_exposed"]
    )
    return controls


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-manifests", action="store_true")
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True), flush=True)
        return 0
    if args.public_manifest is None or args.private_manifest is None:
        raise SystemExit("manifest paths are required")
    if args.build_manifests:
        print(
            json.dumps(
                build_manifests(args.public_manifest, args.private_manifest), sort_keys=True
            ),
            flush=True,
        )
        return 0
    if args.endpoint is None or args.receipt is None:
        raise SystemExit("server endpoint and receipt are required")
    run_server(args.public_manifest, args.private_manifest, args.endpoint, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
