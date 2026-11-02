"""REQ-REPORT-8698: qualify constructed worlds without inferring natural truth.

Questions declare the relation vocabulary before candidate text is inspected.
An integer evaluator receives only facts and predicates, never answer wording.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v746_contract_authority as authority
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.verify.source_alignment import sentence_spans
from carnot.verify import cuda_reader_8410 as cuda

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = (
    "experiment_8698_v746_relational_evidence_controls",
    "exp8698-relational-evidence-controls",
)
PROTOCOL, PROTOCOL_PIN = authority.original.PROTOCOL, authority.original.PROTOCOL_PIN
MODEL_SPECS: list[Json] = []
FAMILIES = ("directed_role", "negation_attachment", "temporal_order")
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/relational_{s}_8698.py" for s in ["evidence", "runner"]] + [CLI]
TESTS = ["tests/python/test_relational_evidence_8698.py"]
reference, load = retained.reference, retained.load


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a supervisor can distinguish work from a stall."""
    print(f"[exp8698] phase={phase} completed={completed} pending={pending}", flush=True)


def scalar(path: Path, field: str, expected: Any) -> Json:
    """Absence and measured zero require different reopening actions."""
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    state = (
        "missing_file"
        if not path.is_file()
        else "missing_field"
        if field not in value
        else "observed_zero"
        if value[field] == 0
        else "observed_value"
    )
    return dict(
        authority.gate(path, field, expected, value.get(field), TASK),
        operand_state=state,
        passed=field in value and value[field] == expected,
    )


def authenticate(ref: Json) -> Json:
    """Compare bytes with an independently supplied pin before interpreting them."""
    path = Path(ref["path"])
    observed = reference(path)["sha256"] if path.is_file() else None
    return dict(
        authority.gate(path, "sha256", ref["sha256"], observed, TASK),
        operand_state="missing_file"
        if observed is None
        else "authenticated"
        if observed == ref["sha256"]
        else "changed_bytes",
        passed=observed == ref["sha256"],
    )


def question(world: Json) -> Json:
    """Expose only the relation family and entity roster when grounding the question."""
    return dict(
        instruction="Assess registered relations among these entities.",
        family=world["family"],
        entities=list(world["entities"]),
    )


def ground(prompt: Json) -> list[list[int]]:
    """Freeze every allowable parameterized query before any candidate answer is opened."""
    return [
        [a, b, polarity]
        for a in range(len(prompt["entities"]))
        for b in range(len(prompt["entities"]))
        if a != b
        for polarity in [0, 1]
    ]


def render(world: Json) -> str:
    """Render the fixed grammar independently of the integer truth evaluator."""
    a, b, c = world["entities"]
    text = {
        "directed_role": f"{a} guides {b}.",
        "negation_attachment": f"{a} helps {b} and does not help {c}.",
        "temporal_order": f"{a} arrives before {b}.",
    }[world["family"]]
    return (
        "A neutral event happened. " * (66 if int(world["world_id"].split(":")[1]) % 2 else 0)
    ) + text


def parse_claim(claim: str, world: Json) -> Any:
    """Recognize only registered constructed wording; unknown text cannot become accepted."""
    a, b, c = world["entities"]
    forms = {
        "directed_role": {
            f"{a} guides {b}.": [0, 1, 1],
            f"{b} is guided by {a}.": [0, 1, 1],
            f"{b} guides {a}.": [1, 0, 1],
        },
        "negation_attachment": {
            f"{a} helps {b} and does not help {c}.": [[0, 1, 1], [0, 2, 0]],
            f"{a} does not help {c} and helps {b}.": [[0, 2, 0], [0, 1, 1]],
            f"{a} helps {c} and does not help {b}.": [[0, 2, 1], [0, 1, 0]],
        },
        "temporal_order": {
            f"{a} arrives before {b}.": [0, 1, 1],
            f"{b} arrives after {a}.": [0, 1, 1],
            f"{b} arrives before {a}.": [1, 0, 1],
        },
    }
    return forms[world["family"]].get(claim)


def evaluate(facts: list[list[int]], query: Any) -> int:
    """Use explicit premises only; an absent fact keeps the whole conjunction unknown."""
    if query is None:
        return -1
    predicates = query if isinstance(query[0], list) else [query]
    table = {(a, b): value for a, b, value in facts}
    if any((a, b) not in table for a, b, _ in predicates):
        return -1
    return int(all(table[a, b] == expected for a, b, expected in predicates))


def qualify(worlds: list[Json]) -> Json:
    """Keep every original row and require truth, labels, grounding and exact byte alignment."""
    rows, spans, world_rows, independence = [], [], [], []
    for index, world in enumerate(worlds):
        prompt = question(world)
        queries = ground(prompt)
        query_hash = canonical_hash(queries)
        source = render(world)
        decisive_sentence = len(sentence_spans(source.encode()))
        world_rows.append(
            dict(
                world_id=world["world_id"],
                family=world["family"],
                facts=world["facts"],
                question=prompt,
                grounded_queries=queries,
                decisive_sentence=decisive_sentence,
            )
        )
        independence.append(
            dict(
                world_id=world["world_id"],
                query_sha256=query_hash,
                candidate_answer_read_before_grounding=False,
                answer_permutation_query_sha256=canonical_hash(ground(prompt)),
                passed=queries == ground(prompt),
            )
        )
        for view in world["views"]:
            actual_query = parse_claim(view["claim"], world)
            actual = evaluate(world["facts"], actual_query)
            predicates = (
                actual_query
                if isinstance(actual_query, list) and isinstance(actual_query[0], list)
                else [actual_query]
            )
            grounded = actual_query is None or all(p in queries for p in predicates)
            document = view["document"].encode()
            ss, cs = view["source_span"], view["claim_span"]
            aligned = (
                document[ss[0] : ss[1]] == view["source"].encode() == source.encode()
                and document[cs[0] : cs[1]] == view["claim"].encode()
            )
            view_labels = dict(
                supported=1, equivalent_wording=1, contradicted=0, explicit_uncertainty=-1
            )
            known_wording = (
                actual_query is not None
                or view["claim"] == "It is uncertain whether " + source.rsplit(". ", 1)[-1]
            )
            valid = (
                actual == view["truth"]
                and actual_query == view["query"]
                and aligned
                and grounded
                and known_wording
                and view_labels.get(view["view"]) == view["truth"]
            )
            identity = dict(world_id=world["world_id"], view=view["view"])
            spans.append(
                dict(
                    identity,
                    source_span=ss,
                    claim_span=cs,
                    document_sha256=canonical_hash(view["document"]),
                    passed=aligned,
                )
            )
            rows.append(
                dict(
                    identity,
                    original_truth=view["truth"],
                    actual_truth=actual,
                    original_query=view["query"],
                    query=actual_query,
                    query_sha256=query_hash,
                    status="completed" if valid else "failed",
                    passed=valid,
                    absolute_metric=int(valid),
                    censored=False,
                    excluded=False,
                )
            )
        progress("worlds", index + 1, len(worlds) - index - 1)
    ready = (
        len(rows) == 480
        and len({w["world_id"] for w in worlds}) == 120
        and Counter(w["family"] for w in worlds) == dict.fromkeys(FAMILIES, 40)
        and all(r["passed"] for r in rows)
        and sum(w["decisive_sentence"] > 65 for w in world_rows) == 60
    )
    return dict(
        ready=ready,
        rows=rows,
        world_rows=world_rows,
        source_span_rows=spans,
        query_independence_rows=independence,
    )


def custody(ref: Json, field: str, number: int) -> Json:
    """Read original terminal acceptance as well as the producer's scalar and identity."""
    refs, failures = [ref], []
    try:
        value = load(ref)
        path = Path(ref["path"])
        for key, expected in [
            (field, 1),
            ("experiment_id", number),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = scalar(path, key, expected)
            if not row["passed"]:
                failures.append(row)
        side_ref = reference(Path(value["terminal_validation_sidecar_path"]))
        refs.append(side_ref)
        side = load(side_ref)
        publication = side.get("publication", side)
        bound = publication.get("primary_sha256", side.get("candidate_reference", {}).get("sha256"))
        if publication.get("sidecar_path"):
            refs.append(reference(Path(publication["sidecar_path"])))
            accepted = load(refs[-1])["report"]["passed"] is True
        else:
            accepted = side.get("report", side).get("passed") is True
        if bound != ref["sha256"] or not accepted:
            failures.append(
                authority.gate(
                    Path(side_ref["path"]),
                    "terminal_hash_and_acceptance",
                    ref["sha256"],
                    bound,
                    f"exp{number}",
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(authenticate(ref), check="producer_terminal", passed=False, diagnostic=str(error))
        )
    return dict(passed=not failures, references=refs, gate_check_summary=failures)


def inputs(root: Path) -> Json:
    """Authenticate only this branch's original operands, without consulting global readiness."""
    refs = [dict(path=str(root / PROTOCOL), sha256=PROTOCOL_PIN)]
    failures: list[Json] = []
    try:
        p = load(refs[0])
        protocol = p["relational_evidence_protocol"]
        refs += [protocol["worlds_reference"], protocol["independent_evaluator_reference"]]
        refs += p["operand_references"]["extraction"]
        refs.append(
            dict(
                path=str(
                    root / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
                ),
                sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
            )
        )
        terminals = []
        for number, field in p["branch_requirements"]["extraction_inputs_ready_score"].items():
            ref = p["v743_producers"][number]
            terminal = custody(ref, field, int(number))
            terminals.append(terminal)
            refs += terminal["references"]
            failures += terminal["gate_check_summary"]
        extractor = load(p["v743_producers"]["8656"])
        for row in extractor["code_config_hashes"]:
            if Path(row["origin_path"]).name in {"source_alignment.py", "streaming_evidence.py"}:
                refs += [
                    row["reference"],
                    dict(path=row["origin_path"], sha256=row["reference"]["sha256"]),
                ]
        failures += [check for ref in refs if not (check := authenticate(ref))["passed"]]
        evaluator = load(protocol["independent_evaluator_reference"])
        worlds = load(protocol["worlds_reference"])["worlds"]
        assertions = (
            protocol["worlds"] == 120
            and protocol["worlds_per_family"] == 40
            and protocol["views_per_world"] == 4
            and protocol["half_decisive_evidence_beyond_sentence"] == 65
            and protocol["families"] == list(FAMILIES)
            and evaluator["no_extractor_or_expected_answer_input"] is True
            and all(w["facts"] == evaluator["truth_tables"][w["family"]] for w in worlds)
        )
        if not assertions:
            failures.append(
                authority.gate(root / PROTOCOL, "original_relational_assertions", True, False)
            )
        return dict(
            passed=not failures,
            failures=failures,
            references=refs,
            protocol=protocol,
            worlds=worlds,
            producer_terminals=terminals,
            historical_quarantines=p["historical_quarantines"],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            passed=False,
            failures=[dict(authenticate(refs[0]), diagnostic=str(error))],
            references=refs,
            worlds=[],
            protocol={},
            historical_quarantines={},
        )


def runtime(root: Path, raw: Path) -> Json:
    """Use the shipped supplied-receipt reader; no missing runtime can veto CPU work."""
    from carnot.reporting.v725_contract_methods import runtime as supplied_runtime

    raw.mkdir(parents=True, exist_ok=True)
    result = dict(supplied_runtime(root))
    path = root / "ops/cuda-runtime-repair-receipt.json"
    validator = dict(
        path=str(ROOT / "results/experiment_8580_v737_current_validator_qualification.json"),
        sha256="sha256:7656f6f49b69237528263cd6dc38ac9503e853b1f53d1b5b1037f6510d2892f4",
    )
    terminal = custody(validator, "current_validator_ready_score", 8580)
    changed = any(
        r["changed"] and r["field"] in {"libraries", "environment"}
        for r in result.get("changes", [])
    )
    ready = int(result["ready"] and changed and terminal["passed"])
    failures = result["failures"] + terminal["gate_check_summary"]
    if not ready:
        failures.append(scalar(path, "causal_library_or_environment_repair", 1))
    try:
        cached = cuda.cached(ROOT / cuda.CACHED, raw / "cached")
    except (OSError, ValueError, KeyError, TypeError) as error:
        ready = 0
        cached = dict(refs=[], diagnostic=str(error))
        failures.append(
            dict(
                authority.gate(
                    ROOT / cuda.CACHED, "cached_runtime_diagnostic", True, str(error), "exp8410"
                ),
                check="cached_runtime_diagnostic",
            )
        )
    atomic_json(raw / "diagnostics.json", dict(supplied=result, cached=cached, validator=terminal))
    return dict(
        ready=ready,
        gate_check_summary=failures,
        diagnostic_reference=reference(raw / "diagnostics.json"),
        references=terminal["references"] + cached["refs"],
        current_device_probe_count=0,
    )


def checksum(value: Json) -> str:
    """Bind every field except the checksum itself so cold replay can detect report edits."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(work: Json, output: Path) -> Json:
    """Publish constructed capacity only after owned execution and original custody agree."""
    corpus = deepcopy(work["corpus"])
    validation = work["validation"]
    owned = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    owned = owned and validation["coverage"]["complete"] and validation["custody"]["passed"]
    private = work["private_control"]
    ready = int(
        owned
        and not private
        and work["inputs"]["passed"]
        and work["contract"]["ready"]
        and corpus["ready"]
        and not work["adversarial_findings"]
    )
    verdict = (
        "null"
        if private
        else "disqualified"
        if not owned or work["adversarial_findings"]
        else "circular_positive"
        if ready
        else "blocked"
    )
    failures = deepcopy(work["inputs"]["failures"] + work["contract"]["gate_check_summary"])
    if not owned:
        failures.append(authority.gate(Path(work["raw"]), "owned_checks_and_coverage", True, False))
    if not corpus["ready"]:
        failures.append(authority.gate(Path(work["raw"]), "all_480_constructed_views", True, False))
    value: Json = dict(
        experiment_id=8698,
        task_id=TASK,
        milestone="2026.11.746",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v746_relational_evidence_controls",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=None,
        historical_model_provenance=dict(
            scope="Historical source producers only; no numerical head evaluation."
        ),
        rows=corpus["rows"],
        intended_count=480,
        completed_count=sum(r["passed"] for r in corpus["rows"]),
        failed_count=sum(not r["passed"] for r in corpus["rows"]),
        censored_count=480 - len(corpus["rows"]),
        excluded_count=0,
        independent_count=len(corpus["world_rows"]),
        sample_size_budget=dict(
            independent_constructed_worlds=120,
            views=480,
            natural_questions=0,
            repeats_multiply_evidence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="Fully exposed constructed worlds; capacity diagnostic only.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(not private and (not owned or work["adversarial_findings"])),
        acceptance_gates=dict(
            original_branch_inputs=work["inputs"]["passed"],
            full_v746_contract=work["contract"]["ready"],
            all_truth_views_and_spans=corpus["ready"],
            owned_validation=bool(owned),
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        retained_child_custody=validation["custody"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["resources"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7448670,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=dict(
            corpus_ready_score=ready, runtime_repair_ready_score=work["runtime"]["ready"]
        ),
        corpus_ready_score=ready,
        runtime_repair_ready_score=work["runtime"]["ready"],
        world_rows=corpus["world_rows"],
        source_span_rows=corpus["source_span_rows"],
        query_independence_rows=corpus["query_independence_rows"],
        optional_runtime_gate_check_summary=work["runtime"]["gate_check_summary"],
        optional_runtime_diagnostics=work["runtime"],
        original_protocol_reference=work["inputs"]["references"][0],
        historical_quarantines=work["inputs"]["historical_quarantines"],
        preserved_nulls=[
            "V733_source_information",
            "V735_cold_cost",
            "V737_supervisor",
            "V743_batch_one",
        ],
        current_device_probe_count=0,
        physical_fpga_execution=False,
        execution_authority=work["contract"],
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology_note="Compile question predicates before candidate parsing. Compare 480 immutable views against explicit integer facts and original byte spans. This tests constructed verifier capacity, not natural extraction, useful learning, independent generalization or hardware acceleration.",
    )
    value["field_principles"] = dict.fromkeys(
        value,
        "Bind this invocation to original byte authority; readiness grants no independent scientific benefit.",
    )
    value["field_principles"].update(
        corpus_ready_score="One only after all 480 original truth/view/span checks, 120 distinct worlds, independent producer custody and complete owned validation.",
        runtime_repair_ready_score="Separate exact optional runtime gate, consumed only by Exp8704; absence does not lower corpus readiness.",
        independent_count="120 constructed world IDs; four repeated views do not multiply independence.",
        field_principles="Explain each field without changing its evidence scope.",
        reproducibility_checksum="Hash every field except this checksum.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value
