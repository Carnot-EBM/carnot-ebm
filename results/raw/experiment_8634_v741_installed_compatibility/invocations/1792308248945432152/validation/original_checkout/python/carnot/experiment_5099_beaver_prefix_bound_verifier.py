"""Exp 5099: BEAVER-style prefix bounds over a finite verdict schema.

Spec refs: REQ-VERIFY-5099, SCENARIO-VERIFY-5099.

This module intentionally proves the probability-bound abstraction on a tiny
finite language before touching live model logprobs. The finite language is
canonical ASCII JSON for Carnot verifier verdict records. The prefix-closed
part is the admissible-prefix set: a prefix is still admissible exactly when it
can be extended to at least one terminal record that satisfies the semantic
constraint. That gives deterministic lower and upper probability bounds over a
frontier without claiming anything about a live GGUF endpoint when Exp5097 is
blocked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ONE = Fraction(1, 1)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5099
EXPERIMENT_NAME = "experiment_5099_beaver_prefix_bound_verifier"
SCHEMA = "carnot.experiment_5099_beaver_prefix_bound_verifier.v468"
RESULT_RELATIVE_PATH = "results/experiment_5099_beaver_prefix_bound_verifier_v468.json"
EXP5097_RELATIVE_PATH = "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json"
SPEC_REFS = ["REQ-VERIFY-5099", "SCENARIO-VERIFY-5099"]
RUN_DATE = "20260701"
RANDOM_SEED = 20260701

FINITE_SCHEMA_NAME = "verifier_verdict_schema_v1"
CONSTRAINT_NAME = "evidence_backed_non_abstain_medium_or_high_confidence"
TOY_INFERENCE_SUBSTRATE = "deterministic_toy_finite_distribution"
DEFAULT_FRONTIER_DEPTH = 72
DEFAULT_MONOTONIC_DEPTHS = (0, 16, 40, 72, 96, 10_000)
EOS_TOKEN_ID = 128

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)
MANDATED_MODEL_IDS = tuple(row["hf_id"] for row in MODEL_SPECS)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "Terminal prefix states whether bounds are sound on the finite schema and whether the run stayed toy-only because Exp5097 was not clean."
    },
    "duration_s": {
        "principle": "Wall-clock duration for deterministic schema enumeration, trie traversal, and artifact assembly."
    },
    "inference_substrate": {
        "principle": "Declares deterministic_toy_finite_distribution unless a real live logprob frontier is actually invoked."
    },
    "preconditions_checked": {
        "principle": "Records finite schema, prefix-closed constraint, byte-token assumptions, and Exp5097 logprob cleanliness before bound computation."
    },
    "model_specs": {
        "principle": "Carries the three mandated GGUF IDs and any Exp5097 resolved paths without implying live use."
    },
    "prefix_closed_constraint": {
        "principle": "Defines the semantic terminal predicate and the admissible-prefix rule used for BEAVER-style bounds."
    },
    "backend_used": {
        "principle": "Names toy_distribution unless a clean Exp5097 substrate is explicitly used for live logprob bounds."
    },
    "lower_bound": {
        "principle": "Probability mass of frontier nodes whose every terminal continuation satisfies the constraint."
    },
    "upper_bound": {
        "principle": "Probability mass of frontier nodes with at least one satisfying terminal continuation."
    },
    "bound_gap": {
        "principle": "Upper minus lower; the gap shrinks monotonically as the frontier is refined."
    },
    "exact_probability_if_enumerable": {
        "principle": "Exact satisfying probability from terminal enumeration of the finite toy distribution."
    },
    "monotonic_bounds": {
        "principle": "Checks that refined frontiers never decrease lower bounds, never increase upper bounds, and always contain the exact probability."
    },
    "soundness_checks_passed": {
        "principle": "True only when lower <= exact <= upper, monotonic checks pass, and no live/toy provenance inconsistency is present."
    },
    "frontier_nodes": {
        "principle": "Auditable frontier node summaries showing mass, satisfying mass, and all/mixed/none classification."
    },
    "live_llm_invoked": {
        "principle": "True only if local GGUF logprobs were actually queried for this artifact."
    },
    "flagged_adversarial": {
        "principle": "True only when the artifact detects an internal contradiction such as live inference claimed without live invocation."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class DistributionRow:
    """One terminal finite-schema output with exact rational probability."""

    text: str
    probability: Fraction


@dataclass(frozen=True)
class ExactProbability:
    """Exact probability reported in both rational and float forms."""

    value: float
    fraction: str


@dataclass(frozen=True)
class TrieNode:
    """One prefix trie node with total and satisfying mass under the prefix."""

    transitions: Mapping[int, int]
    total_mass: Fraction
    satisfying_mass: Fraction


@dataclass(frozen=True)
class PrefixTrie:
    """Trie over canonical ASCII byte tokens for the finite distribution."""

    nodes: tuple[TrieNode, ...]


@dataclass(frozen=True)
class FrontierBound:
    """Lower/upper bound result for one frontier depth."""

    lower_mass: Fraction
    upper_mass: Fraction
    frontier_nodes: tuple[JsonDict, ...]

    @property
    def lower_bound(self) -> float:
        return _fraction_to_float(self.lower_mass)

    @property
    def upper_bound(self) -> float:
        return _fraction_to_float(self.upper_mass)

    @property
    def bound_gap(self) -> float:
        return round(self.upper_bound - self.lower_bound, 12)

    @property
    def frontier_node_count(self) -> int:
        return len(self.frontier_nodes)


def finite_verifier_verdict_outputs() -> tuple[str, ...]:
    """Enumerate the finite canonical JSON language for Exp5099."""

    verdicts = ("accept", "reject", "abstain")
    evidence_labels = (
        "schema_valid",
        "schema_missing_field",
        "solver_verified",
        "solver_counterexample",
        "arithmetic_mismatch",
        "citation_gap",
    )
    confidences = ("low", "medium", "high")
    outputs: list[str] = []
    for verdict in verdicts:
        for evidence_label in evidence_labels:
            for confidence in confidences:
                outputs.append(
                    json.dumps(
                        {
                            "confidence": confidence,
                            "evidence_label": evidence_label,
                            "schema": FINITE_SCHEMA_NAME,
                            "verdict": verdict,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
    return tuple(outputs)


def finite_schema_descriptor() -> JsonDict:
    return {
        "schema_name": FINITE_SCHEMA_NAME,
        "output_format": "canonical_ascii_json",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'))",
        "fields": {
            "schema": [FINITE_SCHEMA_NAME],
            "verdict": ["accept", "reject", "abstain"],
            "evidence_label": [
                "schema_valid",
                "schema_missing_field",
                "solver_verified",
                "solver_counterexample",
                "arithmetic_mismatch",
                "citation_gap",
            ],
            "confidence": ["low", "medium", "high"],
        },
        "tokenization": {
            "kind": "ascii_byte_tokens_plus_eos",
            "byte_token_ids": "0..127",
            "eos_token_id": EOS_TOKEN_ID,
            "bpe_tokenizer_used": False,
        },
    }


def terminal_satisfies_constraint(text: str) -> bool:
    """Return whether a terminal verdict JSON record satisfies the constraint."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("schema") == FINITE_SCHEMA_NAME
        and payload.get("verdict") in {"accept", "reject"}
        and payload.get("evidence_label") in {"schema_valid", "solver_verified"}
        and payload.get("confidence") in {"medium", "high"}
    )


def prefix_closed_constraint() -> JsonDict:
    outputs = finite_verifier_verdict_outputs()
    satisfying = [text for text in outputs if terminal_satisfies_constraint(text)]
    return {
        "constraint_name": CONSTRAINT_NAME,
        "prefix_closed": True,
        "terminal_predicate": (
            "schema is verifier_verdict_schema_v1; verdict is accept or reject; "
            "evidence_label is schema_valid or solver_verified; confidence is medium or high"
        ),
        "admissible_prefix_rule": (
            "a prefix is admissible iff at least one satisfying canonical terminal output starts with it"
        ),
        "satisfied_terminal_count": len(satisfying),
        "total_terminal_count": len(outputs),
    }


def toy_finite_distribution(outputs: Sequence[str]) -> tuple[DistributionRow, ...]:
    """Return the exact fallback distribution used when live logprobs are unclean."""

    if not outputs:
        raise ValueError("finite output set must not be empty")
    probability = Fraction(1, len(outputs))
    return tuple(DistributionRow(text=text, probability=probability) for text in outputs)


def exact_probability(
    distribution: Sequence[DistributionRow],
    constraint: Mapping[str, Any],
) -> ExactProbability:
    """Enumerate terminal outputs and return exact satisfying probability."""

    del constraint
    mass = sum(
        (row.probability for row in distribution if terminal_satisfies_constraint(row.text)),
        Fraction(0, 1),
    )
    return ExactProbability(value=_fraction_to_float(mass), fraction=_format_fraction(mass))


def build_prefix_trie(
    distribution: Sequence[DistributionRow],
    terminal_predicate: Any,
) -> PrefixTrie:
    """Build a byte-token trie with exact total and satisfying mass at each node."""

    transitions: list[dict[int, int]] = [{}]
    total_mass: list[Fraction] = [Fraction(0, 1)]
    satisfying_mass: list[Fraction] = [Fraction(0, 1)]

    for row in distribution:
        encoded = row.text.encode("ascii")
        satisfies = bool(terminal_predicate(row.text))
        state = 0
        total_mass[state] += row.probability
        if satisfies:
            satisfying_mass[state] += row.probability
        for token_id in encoded:
            next_state = transitions[state].get(token_id)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][token_id] = next_state
                transitions.append({})
                total_mass.append(Fraction(0, 1))
                satisfying_mass.append(Fraction(0, 1))
            state = next_state
            total_mass[state] += row.probability
            if satisfies:
                satisfying_mass[state] += row.probability

    return PrefixTrie(
        nodes=tuple(
            TrieNode(
                transitions=dict(sorted(children.items())),
                total_mass=total,
                satisfying_mass=satisfying,
            )
            for children, total, satisfying in zip(transitions, total_mass, satisfying_mass)
        )
    )


def bound_frontier(trie: PrefixTrie, *, max_depth: int) -> FrontierBound:
    """Traverse the trie frontier and return lower/upper satisfying bounds."""

    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    lower = Fraction(0, 1)
    upper = Fraction(0, 1)
    frontier: list[JsonDict] = []
    stack: list[tuple[int, bytes, int]] = [(0, b"", 0)]
    while stack:
        state, prefix, depth = stack.pop()
        node = trie.nodes[state]
        if depth >= max_depth or not node.transitions:
            classification = _classify_node(node)
            if classification == "all_satisfying":
                lower += node.total_mass
                upper += node.total_mass
            elif classification == "mixed":
                upper += node.total_mass
            frontier.append(_frontier_node_summary(state, prefix, depth, node, classification))
            continue
        for token_id, target in sorted(node.transitions.items(), reverse=True):
            stack.append((target, prefix + bytes((token_id,)), depth + 1))
    frontier.sort(key=lambda row: (row["depth"], row["prefix"]))
    return FrontierBound(lower_mass=lower, upper_mass=upper, frontier_nodes=tuple(frontier))


def check_monotonic_bounds(
    trie: PrefixTrie,
    *,
    exact_probability: float,
    depths: Sequence[int],
) -> JsonDict:
    """Check that refinement tightens bounds while containing the exact mass."""

    rows: list[JsonDict] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    gap_values: list[float] = []
    for depth in depths:
        bound = bound_frontier(trie, max_depth=int(depth))
        lower_values.append(bound.lower_bound)
        upper_values.append(bound.upper_bound)
        gap_values.append(bound.bound_gap)
        rows.append(
            {
                "depth": int(depth),
                "lower_bound": bound.lower_bound,
                "upper_bound": bound.upper_bound,
                "bound_gap": bound.bound_gap,
                "frontier_node_count": bound.frontier_node_count,
            }
        )
    lower_ok = all(a <= b + 1e-12 for a, b in zip(lower_values, lower_values[1:]))
    upper_ok = all(a + 1e-12 >= b for a, b in zip(upper_values, upper_values[1:]))
    gap_ok = all(a + 1e-12 >= b for a, b in zip(gap_values, gap_values[1:]))
    contains = all(
        lower - 1e-12 <= exact_probability <= upper + 1e-12
        for lower, upper in zip(lower_values, upper_values)
    )
    return {
        "passed": bool(lower_ok and upper_ok and gap_ok and contains),
        "depths": list(map(int, depths)),
        "rows": rows,
        "lower_non_decreasing": lower_ok,
        "upper_non_increasing": upper_ok,
        "gap_non_increasing": gap_ok,
        "exact_probability_between_all_depths": contains,
    }


def load_preconditions(*, root: Path | str = REPO_ROOT) -> JsonDict:
    """Record the substrate and finite-schema assumptions before computing bounds."""

    root_path = Path(root)
    exp5097_path = root_path / EXP5097_RELATIVE_PATH
    exp5097 = _read_json_object(exp5097_path)
    exists = exp5097 is not None
    clean = bool(
        exists
        and exp5097.get("logprob_endpoint_clean") is True
        and exp5097.get("live_llm_invoked") is True
        and exp5097.get("flagged_adversarial") is not True
    )
    if not exists:
        reason = "exp5097_artifact_missing"
    elif exp5097.get("flagged_adversarial") is True:
        reason = "exp5097_flagged_adversarial"
    elif not exp5097.get("logprob_endpoint_clean"):
        reason = "exp5097_not_clean"
    elif exp5097.get("live_llm_invoked") is not True:
        reason = "exp5097_no_live_llm_invocation"
    else:
        reason = None
    return {
        "selected_finite_schema": FINITE_SCHEMA_NAME,
        "prefix_closed_constraint_definition": CONSTRAINT_NAME,
        "tokenization_assumptions": {
            "serialization": "canonical_ascii_json",
            "tokenization": "one ASCII byte per token plus EOS",
            "byte_token_ids": "0..127",
            "eos_token_id": EOS_TOKEN_ID,
            "bpe_tokenizer_used": False,
            "gguf_tokenizer_used": False,
        },
        "exp5097_logprob_substrate": {
            "artifact_path": EXP5097_RELATIVE_PATH,
            "exists": exists,
            "artifact_sha256": _sha256_file(exp5097_path),
            "clean": clean,
            "logprob_endpoint_clean": bool(exp5097.get("logprob_endpoint_clean")) if exists else False,
            "live_llm_invoked": bool(exp5097.get("live_llm_invoked")) if exists else False,
            "flagged_adversarial": bool(exp5097.get("flagged_adversarial")) if exists else False,
            "usable_for_live_frontier": clean,
            "unusable_reason": reason,
        },
    }


def model_specs_from_exp5097(*, root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Return mandated model specs plus any resolved paths carried by Exp5097."""

    exp5097 = _read_json_object(Path(root) / EXP5097_RELATIVE_PATH) or {}
    mandatory = []
    model_specs = exp5097.get("model_specs")
    if isinstance(model_specs, Mapping) and isinstance(model_specs.get("mandatory_models"), list):
        mandatory = [row for row in model_specs["mandatory_models"] if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    for base in MODEL_SPECS:
        source = next((row for row in mandatory if row.get("hf_id") == base["hf_id"]), {})
        rows.append(
            {
                "role": base["role"],
                "hf_id": base["hf_id"],
                "preferred_quant": source.get("preferred_quant") or base["preferred_quant"],
                "resolved_path": source.get("resolved_path"),
                "cache_status": source.get("cache_status") or ("resolved" if source.get("resolved_path") else "unknown"),
                "live_llm_invoked": False,
            }
        )
    return rows


def run(*, root: Path | str = REPO_ROOT, frontier_depth: int = DEFAULT_FRONTIER_DEPTH) -> JsonDict:
    """Run Exp5099 and return the terminal artifact payload."""

    started = time.perf_counter()
    root_path = Path(root)
    outputs = finite_verifier_verdict_outputs()
    distribution = toy_finite_distribution(outputs)
    constraint = prefix_closed_constraint()
    trie = build_prefix_trie(distribution, terminal_satisfies_constraint)
    exact = exact_probability(distribution, constraint)
    bound = bound_frontier(trie, max_depth=frontier_depth)
    monotonic = check_monotonic_bounds(
        trie,
        exact_probability=exact.value,
        depths=DEFAULT_MONOTONIC_DEPTHS,
    )
    preconditions = load_preconditions(root=root_path)
    exp5097_clean = bool(preconditions["exp5097_logprob_substrate"]["clean"])
    live_llm_invoked = False
    backend_used = "toy_distribution"
    inference_substrate = TOY_INFERENCE_SUBSTRATE
    soundness = _soundness_details(bound, exact.value, monotonic, live_llm_invoked, inference_substrate)
    honest_verdict = (
        "success_beaver_prefix_bounds_sound_on_finite_schema"
        if exp5097_clean and soundness["passed"]
        else "complete_beaver_prefix_bounds_toy_only_runtime_not_clean"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "duration_s": round(time.perf_counter() - started, 6),
        "inference_substrate": inference_substrate,
        "preconditions_checked": preconditions,
        "model_specs": model_specs_from_exp5097(root=root_path),
        "prefix_closed_constraint": constraint,
        "backend_used": backend_used,
        "lower_bound": bound.lower_bound,
        "upper_bound": bound.upper_bound,
        "bound_gap": bound.bound_gap,
        "exact_probability_if_enumerable": exact.value,
        "monotonic_bounds": monotonic,
        "soundness_checks_passed": bool(soundness["passed"]),
        "frontier_nodes": list(bound.frontier_nodes),
        "live_llm_invoked": live_llm_invoked,
        "flagged_adversarial": False,
        "finite_schema": finite_schema_descriptor(),
        "frontier_depth": int(frontier_depth),
        "frontier_node_count": bound.frontier_node_count,
        "exact_probability_fraction": exact.fraction,
        "toy_distribution": {
            "kind": "uniform_over_finite_schema",
            "terminal_count": len(distribution),
            "probability_per_terminal": _format_fraction(distribution[0].probability),
            "checksum": _sha256_payload([row.text for row in distribution]),
        },
        "soundness_checks": soundness,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str | None = None,
    frontier_depth: int = DEFAULT_FRONTIER_DEPTH,
) -> JsonDict:
    """Write the stable Exp5099 result JSON."""

    root_path = Path(root)
    destination = Path(output_path) if output_path is not None else root_path / RESULT_RELATIVE_PATH
    artifact = run(root=root_path, frontier_depth=frontier_depth)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5099 terminal artifact violates the contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(
        (
            "success_beaver_prefix_bounds_sound_on_finite_schema",
            "complete_beaver_prefix_bounds_toy_only_runtime_not_clean",
        )
    ):
        raise ValueError("honest_verdict has no accepted Exp5099 terminal prefix")
    if not isinstance(artifact["live_llm_invoked"], bool):
        raise ValueError("live_llm_invoked must be a boolean")
    if not isinstance(artifact["flagged_adversarial"], bool):
        raise ValueError("flagged_adversarial must be a boolean")
    if artifact["inference_substrate"] == "live_llm_inference" and not artifact["live_llm_invoked"]:
        raise ValueError("live_llm_inference cannot be claimed when live_llm_invoked=false")
    if not artifact["live_llm_invoked"] and artifact["inference_substrate"] != TOY_INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be toy when live_llm_invoked=false")
    if not artifact["live_llm_invoked"] and artifact["backend_used"] != "toy_distribution":
        raise ValueError("backend_used must be toy_distribution when live_llm_invoked=false")
    lower = _validated_rate(artifact["lower_bound"], "lower_bound")
    upper = _validated_rate(artifact["upper_bound"], "upper_bound")
    exact = _validated_rate(
        artifact["exact_probability_if_enumerable"],
        "exact_probability_if_enumerable",
    )
    if lower - 1e-12 > upper:
        raise ValueError("lower_bound cannot exceed upper_bound")
    if not (lower - 1e-12 <= exact <= upper + 1e-12):
        raise ValueError("exact_probability_if_enumerable must be inside bounds")
    if abs(float(artifact["bound_gap"]) - (upper - lower)) > 1e-9:
        raise ValueError("bound_gap must equal upper_bound - lower_bound")
    if artifact["soundness_checks_passed"] is not True:
        raise ValueError("soundness_checks_passed must be true for terminal artifact")
    monotonic = artifact.get("monotonic_bounds")
    if not isinstance(monotonic, Mapping) or monotonic.get("passed") is not True:
        raise ValueError("monotonic_bounds must pass")
    frontier_nodes = artifact.get("frontier_nodes")
    if not isinstance(frontier_nodes, list) or not frontier_nodes:
        raise ValueError("frontier_nodes must be a non-empty list")
    if int(artifact.get("frontier_node_count") or -1) != len(frontier_nodes):
        raise ValueError("frontier_node_count must match frontier_nodes")
    constraint = artifact.get("prefix_closed_constraint")
    if not isinstance(constraint, Mapping) or constraint.get("prefix_closed") is not True:
        raise ValueError("prefix_closed_constraint must declare prefix_closed=true")
    model_ids = {
        str(row.get("hf_id"))
        for row in artifact.get("model_specs", [])
        if isinstance(row, Mapping)
    }
    if set(MANDATED_MODEL_IDS) - model_ids:
        raise ValueError("model_specs must include all mandated GGUF IDs")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        raise ValueError("field_principles must annotate every required field")


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = write_artifact()
    print(json.dumps({field: artifact[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


def _classify_node(node: TrieNode) -> str:
    if node.satisfying_mass == 0:
        return "no_satisfying"
    if node.satisfying_mass == node.total_mass:
        return "all_satisfying"
    return "mixed"


def _frontier_node_summary(
    state: int,
    prefix: bytes,
    depth: int,
    node: TrieNode,
    classification: str,
) -> JsonDict:
    return {
        "state": state,
        "prefix": prefix.decode("ascii"),
        "depth": depth,
        "classification": classification,
        "child_count": len(node.transitions),
        "probability_mass": _fraction_to_float(node.total_mass),
        "probability_fraction": _format_fraction(node.total_mass),
        "satisfying_mass": _fraction_to_float(node.satisfying_mass),
        "satisfying_fraction": _format_fraction(node.satisfying_mass),
    }


def _soundness_details(
    bound: FrontierBound,
    exact_probability_value: float,
    monotonic: Mapping[str, Any],
    live_llm_invoked: bool,
    inference_substrate: str,
) -> JsonDict:
    exact_inside = bound.lower_bound <= exact_probability_value + 1e-12 and exact_probability_value <= bound.upper_bound + 1e-12
    live_consistent = live_llm_invoked or inference_substrate == TOY_INFERENCE_SUBSTRATE
    return {
        "passed": bool(exact_inside and monotonic.get("passed") is True and live_consistent),
        "lower_le_exact": bool(bound.lower_bound <= exact_probability_value + 1e-12),
        "exact_le_upper": bool(exact_probability_value <= bound.upper_bound + 1e-12),
        "monotonic_bounds_passed": bool(monotonic.get("passed") is True),
        "live_provenance_consistent": live_consistent,
    }


def _read_json_object(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    basis = {
        "schema": artifact.get("schema"),
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "backend_used": artifact.get("backend_used"),
        "lower_bound": artifact.get("lower_bound"),
        "upper_bound": artifact.get("upper_bound"),
        "exact_probability_if_enumerable": artifact.get("exact_probability_if_enumerable"),
        "frontier_depth": artifact.get("frontier_depth"),
        "toy_distribution": artifact.get("toy_distribution"),
        "random_seed": artifact.get("random_seed"),
    }
    return _sha256_payload(basis)


def _validated_rate(value: Any, field: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return parsed


def _fraction_to_float(value: Fraction) -> float:
    return round(float(value), 12)


def _format_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
