"""REQ-VERIFY-8098: seal exposed sources before human outcomes are decoded.

Source separation limits within-run reuse. It cannot undo earlier corpus access
or establish that a pretrained generator never saw these public examples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import selected_sources
from carnot.verify import fresh_cohort_8084 as qualified
from carnot.verify.evidence_features_7980 import FEATURES
from carnot.verify.sentence_labels_7942 import checked_spans

Json = dict[str, Any]
ROLES = dict(fit=128, tune=64, evaluation=128, stream=256, retention=64)
RESPONSE_KEYS = frozenset({"id", "source_id", "split", "model", "response"})
PUBLIC_KEYS = frozenset({"family_id", "source_bytes", "answer_bytes"})
render = qualified.text_bytes
normalize = qualified.normalize
immutable = qualified.immutable
operand = qualified.operand


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts and flushed output let a supervisor distinguish slow work."""
    print(f"[exp8098] {phase} completed={completed} pending={pending}", flush=True)


def sha(value: bytes) -> str:
    """Salted exact bytes, rather than numeric IDs, determine public ordering."""
    return hashlib.sha256(value).hexdigest()


def public_fields(line: str) -> Json:
    """Skip private JSON values lexically so selection never decodes outcomes."""
    decoder, result, i = json.JSONDecoder(), {}, 1
    while i < len(line):
        while i < len(line) and line[i] in " \t\r\n,":
            i += 1
        if i == len(line) or line[i] == "}":
            break
        key, i = decoder.raw_decode(line, i)
        i = line.index(":", i) + 1
        while line[i].isspace():
            i += 1
        if key in RESPONSE_KEYS:
            value, i = decoder.raw_decode(line, i)
            if key in result or not isinstance(value, str):
                raise ValueError("public_metadata")
            result[key] = value
        else:
            depth, quoted, escaped = 0, False, False
            while i < len(line):
                char = line[i]
                if not quoted and depth == 0 and char in ",}":
                    break
                if quoted:
                    if char == '"' and not escaped:
                        quoted = False
                    escaped = char == "\\" and not escaped
                elif char == '"':
                    quoted = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                i += 1
    if set(result) != RESPONSE_KEYS:
        raise ValueError("public_metadata")
    return result


def public_training(root: Path) -> tuple[list[Json], list[Json]]:
    """Only official response TRAIN rows authorize a source's public projection."""
    responses = []
    for index, line in enumerate((root / "data/ragtruth/response.jsonl").open()):
        row = public_fields(line)
        if row["split"] == "train":
            responses.append(row)
        if index % 1024 == 0:
            progress("public_response_scan", index + 1)
    sources = selected_sources(
        root / "data/ragtruth/source_info.jsonl", {r["source_id"] for r in responses}
    )
    return [
        {k: s[k] for k in ("source_id", "task_type", "source_info")} for s in sources
    ], responses


def select(
    sources: list[Json], responses: list[Json]
) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Transitive public similarity clusters stay intact across all five roles."""
    if any(set(r) != RESPONSE_KEYS for r in responses):
        raise ValueError("public_metadata")
    indexed: dict[str, Json] = {}
    content: dict[str, bytes] = {}
    exclusions: list[Json] = []
    for source in sources:
        if not isinstance(source["source_info"], (str, dict, list)) or source["source_info"] in (
            {},
            [],
        ):
            exclusions.append(
                dict(source_id=source["source_id"], exclusion_reason="malformed_public_source")
            )
            continue
        blob = render(source["source_info"])
        norm = normalize(blob)
        if not norm:
            exclusions.append(dict(source_id=source["source_id"], exclusion_reason="empty_source"))
            continue
        key = sha(norm)
        if key in content and content[key] != norm:
            raise ValueError("source_hash_collision")
        content[key] = norm
        indexed[source["source_id"]] = dict(source, blob=blob, normalized=norm, key=key)
    keys = sorted(content)
    parents = {k: k for k in keys}

    def find(k: str) -> str:
        """Following component roots keeps similarity chains in one role."""
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    shingle_sets = {k: qualified.shingles(content[k]) for k in keys}
    postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for k in keys:
        for shingle in shingle_sets[k]:
            postings[shingle].append(k)
    intersections: Counter[tuple[str, str]] = Counter()
    for members in postings.values():
        intersections.update(combinations(members, 2))
    for (a, b), n in intersections.items():
        if 5 * n >= 4 * (len(shingle_sets[a]) + len(shingle_sets[b]) - n):
            parents[find(b)] = find(a)
    groups: dict[str, list[Json]] = defaultdict(list)
    for r in responses:
        if r["split"] == "train" and r["source_id"] in indexed:
            groups[find(indexed[r["source_id"]]["key"])].append(r)
    clusters: list[Json] = []
    for candidates in groups.values():
        smallest = min(indexed[r["source_id"]]["normalized"] for r in candidates)
        winner = min(
            candidates,
            key=lambda r: (sha(b"V701-response" + r["id"].encode()), r["source_id"], r["id"]),
        )
        clusters.append(
            dict(
                cluster_id=canonical_hash(
                    sorted({indexed[r["source_id"]]["key"] for r in candidates})
                ),
                source_ids=sorted({r["source_id"] for r in candidates}),
                selection_hash=sha(b"V701" + smallest),
                winner=winner,
            )
        )
    clusters.sort(key=lambda c: (c["selection_hash"], c["cluster_id"]))
    roster, public = [], []
    if len(clusters) >= 640:
        offset = 0
        for role, count in ROLES.items():
            for j, c in enumerate(clusters[offset : offset + count]):
                r = c["winner"]
                fid = canonical_hash([c["cluster_id"], r["id"]])
                roster.append(
                    dict(
                        family_id=fid,
                        unit_id=fid,
                        source_id=r["source_id"],
                        response_id=r["id"],
                        source_cluster_id=c["cluster_id"],
                        source_ids=c["source_ids"],
                        role=role,
                        slot=j + 1,
                        order=offset + j,
                        model=r["model"],
                        task_type=indexed[r["source_id"]]["task_type"],
                    )
                )
                public.append(
                    dict(
                        family_id=fid,
                        source_bytes=indexed[r["source_id"]]["blob"].hex(),
                        answer_bytes=r["response"].encode().hex(),
                    )
                )
            offset += count
    return roster, public, exclusions, clusters


def separation(roster: list[Json], public: list[Json]) -> None:
    """A copied source or injected outcome must fail before annotations open."""
    qualified.separation(public, roster)
    if (
        len({r["source_cluster_id"] for r in roster}) != len(roster)
        or any(
            r["family_id"] != p["family_id"] or r["role"] != role or r["slot"] != j + 1
            for role in ROLES
            for j, (r, p) in enumerate(
                (pair for pair in zip(roster, public, strict=True) if pair[0]["role"] == role)
            )
        )
        or {role: sum(r["role"] == role for r in roster) for role in ROLES} != ROLES
    ):
        raise ValueError("immutable_roles")


def target(response: Json, answer: bytes) -> tuple[Json, list[Json]]:
    """Unknown or incomplete annotation custody cannot become a negative label."""
    try:
        spans = checked_spans(response, answer)
        if response["quality"] != "good" or not answer.strip():
            raise ValueError("incomplete_response")
        return dict(y=int(bool(spans)), status="completed", exclusion_reason=None), spans
    except ValueError as error:
        return dict(
            y=None, status="excluded", exclusion_reason="annotation_custody:" + str(error)
        ), []


def empty_plan() -> Json:
    """A failed child still needs truthful zero readiness and preserved missing slots."""
    return dict(
        rows=[],
        failures=[],
        preconditions_checked=[],
        raw_shard_hashes=[],
        role_manifests={},
        evaluator_label_manifests={},
        public_model_strata={},
        class_support={},
        cohort_ready_score=0,
        fit_support_ready_score=0,
        owned_failure=False,
        selection_sealed_before_labels=False,
        exclusions=[],
    )


def seal(root: Path, raw: Path, mutation: str = "") -> Json:
    """Persist public selection first, then preserve complete evaluator span custody."""
    plan = empty_plan()
    for name in ("source_info", "response"):
        path = root / "data/ragtruth" / (name + ".jsonl")
        check = operand(path, "exists", True, path.is_file())
        plan["preconditions_checked"].append(check)
        if not check["passed"]:
            plan["failures"].append(check)
        else:
            plan["raw_shard_hashes"].append(
                dict(path=str(path.absolute()), sha256=sha256_file(path))
            )
    if plan["failures"]:
        return plan
    progress("public_selection_before_labels")
    sources, responses = public_training(root)
    roster, public, exclusions, clusters = select(sources, responses)
    plan.update(
        exclusions=exclusions,
        public_cluster_count=len(clusters),
        train_source_count=len({r["source_id"] for r in responses}),
    )
    plan["raw_shard_hashes"].append(
        immutable(raw / "clusters.json", dict(rows=clusters, exclusions=exclusions))
    )
    check = operand(
        root / "data/ragtruth/response.jsonl", "public_clusters", 640, len(clusters), ">="
    )
    plan["preconditions_checked"].append(check)
    if not check["passed"]:
        plan["failures"].append(check)
        return plan
    if mutation == "contamination":
        public[0]["y"] = 1
    if mutation == "duplicate":
        public[128]["source_bytes"] = public[0]["source_bytes"]
    if mutation == "roles":
        roster[0]["role"] = "tune"
    try:
        separation(roster, public)
    except ValueError as error:
        plan["failures"].append(
            operand(raw / "selection.json", "public_separation", "passed", str(error))
        )
        plan["owned_failure"] = True
        return plan
    plan["selection"] = immutable(raw / "selection.json", dict(roster=roster, public=public))
    plan["raw_shard_hashes"].append(plan["selection"])
    for role in ROLES:
        pairs = [(r, p) for r, p in zip(roster, public, strict=True) if r["role"] == role]
        ref = immutable(
            raw / "public" / (role + ".json"),
            dict(request_rows=[p for _, p in pairs], roster=[r for r, _ in pairs]),
        )
        plan["role_manifests"][role] = ref
        plan["raw_shard_hashes"].append(ref)
    plan["public_model_strata"] = {
        role: dict(
            generator=dict(Counter(r["model"] for r in roster if r["role"] == role)),
            task_type=dict(Counter(r["task_type"] for r in roster if r["role"] == role)),
        )
        for role in ROLES
    }
    plan["raw_shard_hashes"].append(
        immutable(raw / "public/strata.json", plan["public_model_strata"])
    )
    plan.update(cohort_ready_score=1, selection_sealed_before_labels=True)
    progress("public_sealed_evaluator_open", 640)
    selected = {r["response_id"] for r in roster}
    annotated = {}
    for index, line in enumerate((root / "data/ragtruth/response.jsonl").open()):
        metadata = public_fields(line)
        if metadata["id"] in selected:
            annotated[metadata["id"]] = json.loads(line)
        if index % 1024 == 0:
            progress("evaluator_scan", index + 1)
    for role, intended in ROLES.items():
        rows: list[Json] = []
        annotations: list[Json] = []
        for r, p in zip(roster, public, strict=True):
            if r["role"] != role:
                continue
            result, spans = target(
                annotated.get(r["response_id"], {}), bytes.fromhex(p["answer_bytes"])
            )
            rows.append(dict(r, **result))
            annotations.extend(
                dict(
                    s,
                    family_id=r["family_id"],
                    response_id=r["response_id"],
                    response_sha256="sha256:" + sha(bytes.fromhex(p["answer_bytes"])),
                )
                for s in spans
            )
            plan["rows"].append(
                dict(
                    source_id=r["source_id"],
                    unit_id=r["unit_id"],
                    family_id=r["family_id"],
                    role=role,
                    slot=r["slot"],
                    arm="cohort_preparation",
                    condition="exposed_development",
                    issued_state="methods_sealed",
                    metric="complete_human_target",
                    numerator=int(result["y"] is not None),
                    denominator=1,
                    status=result["status"],
                    exclusion_reason=result["exclusion_reason"],
                )
            )
        path = raw / "evaluator" / (role + ".json")
        ref = immutable(
            path,
            dict(
                rows=rows,
                annotation_rows=annotations,
                original_response_records=[
                    annotated[r["response_id"]] for r in roster if r["role"] == role
                ],
                access_policy="evaluator_only",
                public_seal_sha256=plan["selection"]["sha256"],
            ),
        )
        path.parent.chmod(0o700)
        path.chmod(0o600)
        plan["evaluator_label_manifests"][role] = ref
        plan["raw_shard_hashes"].append(ref)
        plan["class_support"][role] = dict(
            intended=intended,
            eligible=sum(r["y"] is not None for r in rows),
            classes={str(y): sum(r["y"] == y for r in rows) for y in (0, 1)},
        )
    plan["fit_support_ready_score"] = int(
        all(
            plan["class_support"][role]["eligible"] >= total
            and min(plan["class_support"][role]["classes"].values()) >= each
            for role, total, each in (("fit", 96, 16), ("tune", 48, 8))
        )
    )
    plan["raw_shard_hashes"].append(
        immutable(
            raw / "public/capture_gate.json",
            dict(fit_support_ready_score=plan["fit_support_ready_score"]),
        )
    )
    progress("evaluator_completed", 640)
    return plan


def methods(design: Path) -> Json:
    """Seal full design text as well as exact basis choices before outcome access."""
    protocol = (
        design.read_text()
        .split("## Frozen data and numerical protocol", 1)[1]
        .split("## Dependency graph", 1)[0]
        .split("<!-- V701_TASK_CONTRACT_START -->", 1)[0]
    )
    return dict(
        protocol=protocol,
        protocol_sha256=canonical_hash(protocol),
        roles=ROLES,
        features=["bounded_qwen_unsupported_logit", *FEATURES],
        additive_basis=dict(
            module="carnot.verify.multivariate_energy_7982",
            function="design",
            degree=3,
            fit_quantiles=[0.25, 0.5, 0.75],
            endpoint_multiplicity=4,
            endpoint_padding=1e-8,
            clipping="fit endpoints",
            channels=9,
            basis_columns=63,
            intercept="separate unpenalized",
            qwen_offset=False,
        ),
        kernel_differences=dict(
            kernel="multivariate squared Euclidean Gaussian",
            old_solver="8085 penalizes intercept and fixes ridge .01; V701 excludes intercept and uses four-fold ridge selection",
            online="V701 four SGD steps replace 8085 bounded batch solve",
        ),
        response_selection_rule="smallest SHA256(V701-response + response_id); ties source_id then response_id",
        statistical_plan=dict(
            draws=10000,
            valid_minimum=9500,
            one_sided_confidence=0.975,
            margin=0.02,
            primary_block=16,
            sensitivity_blocks=[8, 32],
            seeds=list(range(101, 121)),
            unit="source_cluster",
            exposed_development_only=True,
        ),
        precision_bounds={
            str(n): dict(
                worst_case_95_normal_half_width=1.96 * (0.25 / n) ** 0.5,
                zero_event_95_upper=1 - 0.05 ** (1 / n),
                zero_event_97_5_upper=1 - 0.025 ** (1 / n),
            )
            for n in (64, 128, 192, 256)
        },
        rare_event_safety_certified=False,
        no_paper_guarantee_transfers=True,
    )
