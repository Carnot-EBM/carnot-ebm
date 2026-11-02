"""Exp 3886 defabricated graph-grounding fact verifier.

This module reruns the facts-domain graph verifier with a live local SOTA
GGUF extraction pass. The graph score follows the HalluGraph decomposition:
Entity Grounding plus Relation Preservation produce a Composite Fidelity
Index, and the hallucination score is ``1 - CFI``. The same rows are scored
with the math-bound Exp 3862 baseline so the artifact can report the real
facts-domain delta without moving the frozen FoVer math headline.

Spec refs: REQ-VERIFY-3886, SCENARIO-VERIFY-3886,
SCENARIO-VERIFY-3886-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import importlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.corrected_cross_domain_remeasurement_v4 import tie_aware_auroc
from carnot.verify.gguf_inference import load_gguf_generator
from carnot.verify.graph_grounding_probe import (
    FACTS_CORPUS_CANDIDATES,
    GraphGroundingProbe,
    load_facts_rows,
    score_rows_math_bound_ensemble,
)


JsonDict = dict[str, Any]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ResolveGguf = Callable[[str], str | None]
Clock = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3886_graph_grounding_fact_verifier_defabricated.json")
PER_ITEM_REL_PATH = Path(
    "results/experiment_3886_graph_grounding_fact_verifier_defabricated_scores.jsonl"
)
PRIMARY_REASONER_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FALLBACK_REASONER_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
RANDOM_SEED = 3886
DEFAULT_SAMPLE_SIZE = 120
MIN_REAL_DURATION_S = 60.0
CFI_ENTITY_WEIGHT = 0.7
GRAPH_GROUNDING_PREFER_ORDER = (
    "gemma-4-26B-A4B-it",
    "Qwen3.6-35B-A3B",
    "gemma-4-31B-it",
    "Qwen3.5-0.8B",
)
INFERENCE_SUBSTRATE = (
    "live_llama_cpp_sota_gguf_entity_relation_extraction_plus_hallugraph_cfi_"
    "and_same_row_math_bound_ensemble_baseline"
)
METHODOLOGY_PRINCIPLE = (
    "Pre-Launch + Adversarial-Verify + Inference-Substrate - a real graph-grounding "
    "run invokes the local GGUF and takes >=60s; sub-60s is blocked."
)

REQUIRED_PRINCIPLE_FIELDS = (
    "facts_catch_delta",
    "graph_auroc",
    "math_baseline_auroc",
    "per_item_scores_path",
    "model_invoked",
    "n_items",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    *REQUIRED_PRINCIPLE_FIELDS,
    "field_principles",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "facts_catch_delta": (
        "BARE FLOAT - graph_auroc - math_baseline_auroc; exp3887 disk-reads this. "
        "Must reproduce >0 with a non-fabricated run to be bankable."
    ),
    "graph_auroc": (
        "The new-architecture verifier's facts AUROC - must reproduce exp3862's "
        "~0.64 with a real run."
    ),
    "math_baseline_auroc": (
        "The math ensemble on facts - the earned-negative baseline graph-grounding must beat."
    ),
    "per_item_scores_path": (
        "Persist per-item graph+ensemble scores so exp3887 complementarity can run "
        "(the exp3863 blocker)."
    ),
    "model_invoked": (
        "BARE BOOL - the de-fabrication assertion; a real run invokes the model and takes >=60s."
    ),
    "n_items": METHODOLOGY_PRINCIPLE,
    "preconditions_checked": METHODOLOGY_PRINCIPLE,
    "model_specs": METHODOLOGY_PRINCIPLE,
    "random_seed": METHODOLOGY_PRINCIPLE,
    "reproducibility_checksum": METHODOLOGY_PRINCIPLE,
    "duration_s": METHODOLOGY_PRINCIPLE,
    "inference_substrate": METHODOLOGY_PRINCIPLE,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class PreconditionCheck:
    """One live-resource precondition probe."""

    resource: str
    available: bool
    detail: Any

    def as_dict(self) -> JsonDict:
        return {"resource": self.resource, "available": bool(self.available), "detail": self.detail}


@dataclass(frozen=True)
class PreflightResult:
    """Resolved live resources and blocked state before scoring."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None
    model_specs: JsonDict
    corpus_path: Path | None


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3886."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    per_item_scores_path: Path | None = None
    sample_size: int = DEFAULT_SAMPLE_SIZE
    min_duration_s: float = MIN_REAL_DURATION_S
    random_seed: int = RANDOM_SEED
    cuda_probe_timeout_s: int = 20
    started_at: float | None = None
    clock: Clock = time.perf_counter

    def resolved_output_path(self) -> Path:
        return _repo_path(self.repo_root, self.output_path or OUTPUT_REL_PATH)

    def resolved_per_item_scores_path(self) -> Path:
        return _repo_path(self.repo_root, self.per_item_scores_path or PER_ITEM_REL_PATH)

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"

    def start_time(self) -> float:
        return float(self.started_at if self.started_at is not None else self.clock())


@dataclass(frozen=True)
class RelationTriple:
    """One model-extracted relation assertion."""

    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class ExtractedGraph:
    """Entity and relation graph extracted from a text span."""

    entities: tuple[str, ...]
    relations: tuple[RelationTriple, ...]
    raw_response: str = ""


@dataclass(frozen=True)
class GraphExtractionResult:
    """Model-backed graph extraction for one answer/evidence pair."""

    response_graph: ExtractedGraph
    evidence_graph: ExtractedGraph
    prompt_sha256: str
    completion_tokens: int
    parse_fallback_used: bool = False


@dataclass(frozen=True)
class HalluGraphScore:
    """HalluGraph metric decomposition for one extracted graph pair."""

    entity_grounding: float
    relation_preservation: float
    composite_fidelity_index: float
    hallucination_score: float
    missing_entities: tuple[str, ...]
    unsupported_relations: tuple[JsonDict, ...]


class LlamaGraphExtractor:
    """Live llama.cpp extractor for response/evidence entity-relation graphs."""

    def __init__(
        self,
        model_specs: Mapping[str, Any],
        *,
        llama_factory: Callable[..., Any] | None = None,
        max_tokens: int = 220,
    ) -> None:
        if llama_factory is None:
            from llama_cpp import Llama  # pragma: no cover - exercised by the live run.

            llama_factory = Llama  # pragma: no cover
        self.model_specs = dict(model_specs)
        self.max_tokens = max_tokens
        self.invocation_count = 0
        self._llm = llama_factory(
            model_path=str(model_specs["model_path"]),
            n_gpu_layers=int(model_specs.get("n_gpu_layers", -1)),
            n_ctx=int(model_specs.get("n_ctx", 4096)),
            n_batch=int(model_specs.get("n_batch", 256)),
            verbose=False,
        )

    def extract_pair(self, answer: str, evidence: str, item_index: int) -> GraphExtractionResult:
        prompt = graph_extraction_prompt(answer, evidence)
        self.invocation_count += 1
        raw = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED + item_index,
        )
        text = _extract_text(raw)
        completion_tokens = _completion_tokens(raw, text)
        parsed = parse_model_graph_response(
            text,
            prompt_sha256=sha256_text(prompt),
            completion_tokens=completion_tokens,
        )
        if _graph_empty(parsed.response_graph) and _graph_empty(parsed.evidence_graph):
            return _fallback_graph_extraction(
                answer,
                evidence,
                raw_response=text,
                prompt_sha256=sha256_text(prompt),
                completion_tokens=completion_tokens,
            )
        return parsed


class RobustGeneratorGraphExtractor:
    """Graph extractor backed by the Exp 3915 robust generator object."""

    def __init__(
        self,
        generator: Any,
        model_specs: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = 160,
    ) -> None:
        self.generator = generator
        self.model_specs = dict(model_specs or {})
        self.max_tokens = max_tokens
        self.invocation_count = 0
        self.completion_tokens_total = 0

    def extract_pair(self, answer: str, evidence: str, item_index: int) -> GraphExtractionResult:
        prompt = graph_extraction_prompt(answer, evidence)
        self.invocation_count += 1
        raw = self.generator(
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        text = _extract_text(raw)
        completion_tokens = _completion_tokens(raw, text)
        self.completion_tokens_total += completion_tokens
        parsed = parse_model_graph_response(
            text,
            prompt_sha256=sha256_text(prompt),
            completion_tokens=completion_tokens,
        )
        if _graph_empty(parsed.response_graph) and _graph_empty(parsed.evidence_graph):
            return _fallback_graph_extraction(
                answer,
                evidence,
                raw_response=text,
                prompt_sha256=sha256_text(prompt),
                completion_tokens=completion_tokens,
            )
        return parsed


def graph_extraction_prompt(answer: str, evidence: str) -> str:
    """Build the strict JSON extraction prompt for llama.cpp."""

    return (
        "Extract factual knowledge graphs for hallucination checking.\n"
        "Return ONLY compact JSON with keys response and evidence.\n"
        "Each graph has entities: [string] and relations: "
        "[{\"subject\": string, \"relation\": string, \"object\": string}].\n"
        "Include names, dates, locations, organizations, and named objects as entities.\n"
        "Normalize relation phrases to short predicates such as discovered, wrote, signed_in, "
        "launched_aboard, produced, received, erupted_in. Do not judge truth; only extract.\n"
        "If a sentence says X discovered Y, the relation subject is X and object is Y.\n\n"
        f"Response:\n{_clip_text(answer, 700)}\n\n"
        f"Evidence:\n{_clip_text(evidence, 1200)}\n"
    )


def graph_ground_score(
    item: Mapping[str, Any],
    generator: Any | None = None,
    *,
    model_path: str | None = None,
    llama_factory: Callable[..., Any] | None = None,
    max_tokens: int = 220,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_batch: int = 256,
) -> JsonDict:
    """Score one claim/source pair with a live graph extraction call."""

    answer, evidence = _item_answer_evidence(item)
    if generator is not None and not isinstance(generator, (str, Path)):
        extractor: Any = RobustGeneratorGraphExtractor(
            generator,
            {"loader": "carnot.verify.gguf_inference.load_gguf_generator"},
            max_tokens=max_tokens,
        )
    else:
        resolved_model_path = str(model_path or generator or "")
        if not resolved_model_path:
            raise ValueError("graph_ground_score requires a robust generator or model_path")
        extractor = LlamaGraphExtractor(
            {
                "model_path": resolved_model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "max_tokens": max_tokens,
            },
            llama_factory=llama_factory,
            max_tokens=max_tokens,
        )
    extraction = extractor.extract_pair(answer, evidence, 0)
    score = compute_hallugraph_score(extraction)
    payload = _score_payload(score, extraction, extractor.invocation_count > 0)
    payload["model_call_count"] = extractor.invocation_count
    return payload


def load_robust_graph_grounding_generator(
    prefer_order: Sequence[str] | None = None,
    *,
    n_ctx: int = 2048,
    max_n_gpu_layers: int = 0,
) -> tuple[Any, JsonDict]:
    """Load the Exp 3915 robust GGUF generator for graph-grounding calls."""

    generator, meta = load_gguf_generator(
        prefer_order=tuple(prefer_order) if prefer_order is not None else GRAPH_GROUNDING_PREFER_ORDER,
        n_ctx=n_ctx,
        max_n_gpu_layers=max_n_gpu_layers,
    )
    return generator, dict(meta)


def build_graph_grounding_fixture() -> tuple[JsonDict, ...]:
    """Build the Exp 3896 positive-control claim/source fixture."""

    fixture = (
        _fixture_item(
            item_id="grounded-lovelace-notes",
            claim="Ada Lovelace wrote notes about the Analytical Engine.",
            source="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Ada Lovelace", "Analytical Engine"],
                "relations": [
                    {"subject": "Ada Lovelace", "relation": "wrote", "object": "Analytical Engine"}
                ],
            },
            evidence_graph={
                "entities": ["Ada Lovelace", "Charles Babbage", "Analytical Engine"],
                "relations": [
                    {"subject": "Ada Lovelace", "relation": "wrote", "object": "Analytical Engine"}
                ],
            },
        ),
        _fixture_item(
            item_id="planted-relation-curie-radium",
            claim="Marie Curie discovered radium.",
            source="Marie Curie discovered polonium. Pierre Curie studied radium.",
            gold_hallucinated=True,
            planted_hallucinated_relation=True,
            response_graph={
                "entities": ["Marie Curie", "radium"],
                "relations": [
                    {"subject": "Marie Curie", "relation": "discovered", "object": "radium"}
                ],
            },
            evidence_graph={
                "entities": ["Marie Curie", "polonium", "Pierre Curie", "radium"],
                "relations": [
                    {"subject": "Marie Curie", "relation": "discovered", "object": "polonium"},
                    {"subject": "Pierre Curie", "relation": "studied", "object": "radium"},
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-hubble-discovery",
            claim="The Hubble Space Telescope launched aboard Discovery.",
            source="The Hubble Space Telescope launched aboard the space shuttle Discovery.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
            evidence_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
        ),
        _fixture_item(
            item_id="planted-entity-hubble-atlantis",
            claim="The Hubble Space Telescope launched aboard Atlantis.",
            source="The Hubble Space Telescope launched aboard the space shuttle Discovery.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Hubble Space Telescope", "Atlantis"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Atlantis",
                    }
                ],
            },
            evidence_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-versailles-1919",
            claim="The Treaty of Versailles was signed in 1919.",
            source="The Treaty of Versailles was signed in 1919 after World War I.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Treaty of Versailles", "1919"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
            evidence_graph={
                "entities": ["Treaty of Versailles", "1919", "World War I"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
        ),
        _fixture_item(
            item_id="planted-date-versailles-1929",
            claim="The Treaty of Versailles was signed in 1929.",
            source="The Treaty of Versailles was signed in 1919 after World War I.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Treaty of Versailles", "1929"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1929"}
                ],
            },
            evidence_graph={
                "entities": ["Treaty of Versailles", "1919", "World War I"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-franklin-photo51",
            claim="Rosalind Franklin produced Photo 51.",
            source="Rosalind Franklin produced Photo 51, an X-ray diffraction image of DNA.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Rosalind Franklin", "Photo 51"],
                "relations": [
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"}
                ],
            },
            evidence_graph={
                "entities": ["Rosalind Franklin", "Photo 51", "DNA"],
                "relations": [
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"}
                ],
            },
        ),
        _fixture_item(
            item_id="planted-relation-franklin-nobel",
            claim="Rosalind Franklin received the 1962 Nobel Prize in Physiology or Medicine.",
            source=(
                "James Watson, Francis Crick, and Maurice Wilkins received the 1962 Nobel "
                "Prize in Physiology or Medicine. Rosalind Franklin produced Photo 51."
            ),
            gold_hallucinated=True,
            planted_hallucinated_relation=True,
            response_graph={
                "entities": [
                    "Rosalind Franklin",
                    "1962 Nobel Prize in Physiology or Medicine",
                ],
                "relations": [
                    {
                        "subject": "Rosalind Franklin",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    }
                ],
            },
            evidence_graph={
                "entities": [
                    "James Watson",
                    "Francis Crick",
                    "Maurice Wilkins",
                    "1962 Nobel Prize in Physiology or Medicine",
                    "Rosalind Franklin",
                    "Photo 51",
                ],
                "relations": [
                    {
                        "subject": "James Watson",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {
                        "subject": "Francis Crick",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {
                        "subject": "Maurice Wilkins",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"},
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-vesuvius-79",
            claim="Mount Vesuvius erupted in AD 79.",
            source="Mount Vesuvius erupted in AD 79 and buried Pompeii.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Mount Vesuvius", "AD 79"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 79"}
                ],
            },
            evidence_graph={
                "entities": ["Mount Vesuvius", "AD 79", "Pompeii"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 79"}
                ],
            },
        ),
        _fixture_item(
            item_id="planted-date-vesuvius-179",
            claim="Mount Vesuvius erupted in AD 179.",
            source="Mount Vesuvius erupted in AD 79 and buried Pompeii.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Mount Vesuvius", "AD 179"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 179"}
                ],
            },
            evidence_graph={
                "entities": ["Mount Vesuvius", "AD 79", "Pompeii"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 79"}
                ],
            },
        ),
    )
    return tuple(dict(item) for item in fixture)


def build_nonseparable_graph_grounding_fixture() -> tuple[JsonDict, ...]:
    """Build the Exp 3920 non-separable graph-grounding fixture."""

    fixture = (
        _fixture_item(
            item_id="grounded-lovelace-notes",
            claim="Ada Lovelace wrote notes about the Analytical Engine.",
            source="Ada Lovelace wrote notes about Charles Babbage's Analytical Engine.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Ada Lovelace", "Analytical Engine"],
                "relations": [
                    {"subject": "Ada Lovelace", "relation": "wrote", "object": "Analytical Engine"}
                ],
            },
            evidence_graph={
                "entities": ["Ada Lovelace", "Charles Babbage", "Analytical Engine"],
                "relations": [
                    {"subject": "Ada Lovelace", "relation": "wrote", "object": "Analytical Engine"}
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-hubble-discovery",
            claim="The Hubble Space Telescope launched aboard Discovery.",
            source="The Hubble Space Telescope launched aboard the space shuttle Discovery.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
            evidence_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-versailles-1919",
            claim="The Treaty of Versailles was signed in 1919.",
            source="The Treaty of Versailles was signed in 1919 after World War I.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Treaty of Versailles", "1919"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
            evidence_graph={
                "entities": ["Treaty of Versailles", "1919", "World War I"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-franklin-photo51",
            claim="Rosalind Franklin produced Photo 51.",
            source="Rosalind Franklin produced Photo 51, an X-ray diffraction image of DNA.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Rosalind Franklin", "Photo 51"],
                "relations": [
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"}
                ],
            },
            evidence_graph={
                "entities": ["Rosalind Franklin", "Photo 51", "DNA"],
                "relations": [
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"}
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-vesuvius-79",
            claim="Mount Vesuvius erupted in AD 79.",
            source="Mount Vesuvius erupted in AD 79 and buried Pompeii.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Mount Vesuvius", "AD 79"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 79"}
                ],
            },
            evidence_graph={
                "entities": ["Mount Vesuvius", "AD 79", "Pompeii"],
                "relations": [
                    {"subject": "Mount Vesuvius", "relation": "erupted_in", "object": "AD 79"}
                ],
            },
        ),
        _fixture_item(
            item_id="grounded-apollo-moon",
            claim="Apollo 11 landed on the Moon in 1969.",
            source="Apollo 11 landed on the Moon in July 1969.",
            gold_hallucinated=False,
            response_graph={
                "entities": ["Apollo 11", "Moon", "1969"],
                "relations": [
                    {"subject": "Apollo 11", "relation": "landed_on", "object": "Moon"},
                    {"subject": "Apollo 11", "relation": "landed_in", "object": "1969"},
                ],
            },
            evidence_graph={
                "entities": ["Apollo 11", "Moon", "July 1969"],
                "relations": [
                    {"subject": "Apollo 11", "relation": "landed_on", "object": "Moon"},
                    {"subject": "Apollo 11", "relation": "landed_in", "object": "July 1969"},
                ],
            },
        ),
        _fixture_item(
            item_id="planted-relation-curie-radium",
            claim="Marie Curie discovered radium.",
            source="Marie Curie discovered polonium. Pierre Curie studied radium.",
            gold_hallucinated=True,
            planted_hallucinated_relation=True,
            response_graph={
                "entities": ["Marie Curie", "radium"],
                "relations": [
                    {"subject": "Marie Curie", "relation": "discovered", "object": "radium"}
                ],
            },
            evidence_graph={
                "entities": ["Marie Curie", "polonium", "Pierre Curie", "radium"],
                "relations": [
                    {"subject": "Marie Curie", "relation": "discovered", "object": "polonium"},
                    {"subject": "Pierre Curie", "relation": "studied", "object": "radium"},
                ],
            },
        ),
        _fixture_item(
            item_id="planted-entity-hubble-atlantis",
            claim="The Hubble Space Telescope launched aboard Atlantis.",
            source="The Hubble Space Telescope launched aboard the space shuttle Discovery.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Hubble Space Telescope", "Atlantis"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Atlantis",
                    }
                ],
            },
            evidence_graph={
                "entities": ["Hubble Space Telescope", "Discovery"],
                "relations": [
                    {
                        "subject": "Hubble Space Telescope",
                        "relation": "launched_aboard",
                        "object": "Discovery",
                    }
                ],
            },
        ),
        _fixture_item(
            item_id="planted-date-versailles-1929",
            claim="The Treaty of Versailles was signed in 1929.",
            source="The Treaty of Versailles was signed in 1919 after World War I.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Treaty of Versailles", "1929"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1929"}
                ],
            },
            evidence_graph={
                "entities": ["Treaty of Versailles", "1919", "World War I"],
                "relations": [
                    {"subject": "Treaty of Versailles", "relation": "signed_in", "object": "1919"}
                ],
            },
        ),
        _fixture_item(
            item_id="planted-relation-franklin-nobel",
            claim="Rosalind Franklin received the 1962 Nobel Prize in Physiology or Medicine.",
            source=(
                "James Watson, Francis Crick, and Maurice Wilkins received the 1962 Nobel "
                "Prize in Physiology or Medicine. Rosalind Franklin produced Photo 51."
            ),
            gold_hallucinated=True,
            planted_hallucinated_relation=True,
            response_graph={
                "entities": ["Rosalind Franklin", "1962 Nobel Prize in Physiology or Medicine"],
                "relations": [
                    {
                        "subject": "Rosalind Franklin",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    }
                ],
            },
            evidence_graph={
                "entities": [
                    "James Watson",
                    "Francis Crick",
                    "Maurice Wilkins",
                    "1962 Nobel Prize in Physiology or Medicine",
                    "Rosalind Franklin",
                    "Photo 51",
                ],
                "relations": [
                    {
                        "subject": "James Watson",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {
                        "subject": "Francis Crick",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {
                        "subject": "Maurice Wilkins",
                        "relation": "received",
                        "object": "1962 Nobel Prize in Physiology or Medicine",
                    },
                    {"subject": "Rosalind Franklin", "relation": "produced", "object": "Photo 51"},
                ],
            },
        ),
        _fixture_item(
            item_id="planted-apollo-mars",
            claim="Apollo 11 landed on Mars in 1969.",
            source="Apollo 11 landed on the Moon in July 1969.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["Apollo 11", "Mars", "1969"],
                "relations": [
                    {"subject": "Apollo 11", "relation": "landed_on", "object": "Mars"},
                    {"subject": "Apollo 11", "relation": "landed_in", "object": "1969"},
                ],
            },
            evidence_graph={
                "entities": ["Apollo 11", "Moon", "July 1969"],
                "relations": [
                    {"subject": "Apollo 11", "relation": "landed_on", "object": "Moon"},
                    {"subject": "Apollo 11", "relation": "landed_in", "object": "July 1969"},
                ],
            },
        ),
        _fixture_item(
            item_id="subtle-negation-mayor-bridge",
            claim="The mayor said the bridge was unsafe.",
            source="The mayor denied the bridge was unsafe.",
            gold_hallucinated=True,
            response_graph={
                "entities": ["mayor", "bridge", "unsafe"],
                "relations": [
                    {"subject": "bridge", "relation": "was", "object": "unsafe"}
                ],
            },
            evidence_graph={
                "entities": ["mayor", "bridge", "unsafe"],
                "relations": [
                    {"subject": "bridge", "relation": "was", "object": "unsafe"}
                ],
            },
        ),
    )
    return tuple(dict(item) for item in fixture)


def score_graph_grounding_fixture(
    fixture: Sequence[Mapping[str, Any]] | None,
    *,
    model_path: str | None = None,
    generator: Any | None = None,
    llama_factory: Callable[..., Any] | None = None,
    max_tokens: int = 220,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_batch: int = 256,
    consistency_passes: int = 1,
) -> JsonDict:
    """Run one shared live extractor over the Exp 3896 positive-control fixture."""

    rows = tuple(fixture or build_graph_grounding_fixture())
    pass_count = max(1, int(consistency_passes))
    if generator is not None:
        extractor: Any = RobustGeneratorGraphExtractor(
            generator,
            {"loader": "carnot.verify.gguf_inference.load_gguf_generator"},
            max_tokens=max_tokens,
        )
    else:
        if not model_path:
            raise ValueError("score_graph_grounding_fixture requires generator or model_path")
        extractor = LlamaGraphExtractor(
            {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "max_tokens": max_tokens,
            },
            llama_factory=llama_factory,
            max_tokens=max_tokens,
        )

    labels: list[int] = []
    scores: list[float] = []
    per_item: list[JsonDict] = []
    consistency_scores: list[list[float]] = []
    planted_hallucinated_relation_flagged = False
    for pass_index in range(pass_count):
        pass_scores: list[float] = []
        for index, item in enumerate(rows):
            answer, evidence = _item_answer_evidence(item)
            extraction = extractor.extract_pair(answer, evidence, pass_index * len(rows) + index)
            score = compute_hallugraph_score(extraction)
            pass_scores.append(score.hallucination_score)
            if pass_index != 0:
                continue

            label = int(bool(item.get("gold_hallucinated") or item.get("is_hallucination")))
            labels.append(label)
            scores.append(score.hallucination_score)
            unsupported = list(score.unsupported_relations)
            if bool(item.get("planted_hallucinated_relation")) and unsupported:
                planted_hallucinated_relation_flagged = True
            per_item.append(
                {
                    "id": str(item.get("id") or f"fixture-{index}"),
                    "claim": answer,
                    "source": evidence,
                    "gold_hallucinated": bool(label),
                    "planted_hallucinated_relation": bool(
                        item.get("planted_hallucinated_relation")
                    ),
                    "graph_score": score.hallucination_score,
                    "eg": score.entity_grounding,
                    "rp": score.relation_preservation,
                    "cfi": score.composite_fidelity_index,
                    "missing_entities": list(score.missing_entities),
                    "unsupported_relations": unsupported,
                    "completion_tokens": extraction.completion_tokens,
                    "prompt_sha256": extraction.prompt_sha256,
                    "raw_response_sha256": sha256_text(extraction.response_graph.raw_response),
                    "raw_response_chars": len(extraction.response_graph.raw_response),
                    "raw_response_excerpt": _clip_text(extraction.response_graph.raw_response, 300),
                    "parse_fallback_used": extraction.parse_fallback_used,
                }
            )
        consistency_scores.append([round(float(score), 6) for score in pass_scores])

    fixture_auroc = round(float(tie_aware_auroc(labels, scores)), 6) if labels else None
    unique_scores = {round(score, 6) for score in scores}
    has_model_text = any(
        int(item["completion_tokens"]) > 0 or int(item["raw_response_chars"]) > 0
        for item in per_item
    )
    model_invoked = extractor.invocation_count >= len(rows) and len(rows) > 0
    stub_rejected = bool(model_invoked and has_model_text and len(unique_scores) > 1)
    return {
        "fixture_auroc": fixture_auroc,
        "model_invoked": model_invoked,
        "fixture_n_items": len(rows),
        "fixture_n_hallucinated": int(sum(labels)),
        "labels": labels,
        "graph_scores": [round(float(score), 6) for score in scores],
        "per_item_scores": per_item,
        "planted_hallucinated_relation_flagged": planted_hallucinated_relation_flagged,
        "parse_fallback_count": sum(1 for item in per_item if item["parse_fallback_used"]),
        "model_call_count": extractor.invocation_count,
        "fixture_token_count": int(sum(int(item["completion_tokens"]) for item in per_item)),
        "consistency_passes": pass_count,
        "consistency_scores": consistency_scores,
        "stub_rejected": stub_rejected,
    }


def score_nonseparable_graph_grounding_fixture(
    fixture: Sequence[Mapping[str, Any]] | None = None,
    *,
    generator: Any,
    max_tokens: int = 160,
) -> JsonDict:
    """Run the Exp 3920 non-separable fixture through the robust generator."""

    return score_graph_grounding_fixture(
        tuple(fixture or build_nonseparable_graph_grounding_fixture()),
        generator=generator,
        max_tokens=max_tokens,
        consistency_passes=1,
    )


def compute_hallugraph_score(extraction: GraphExtractionResult) -> HalluGraphScore:
    """Compute Entity Grounding, Relation Preservation, CFI, and error score."""

    response_entities = tuple(_dedupe_keep_order(extraction.response_graph.entities))
    evidence_entities = tuple(_dedupe_keep_order(extraction.evidence_graph.entities))
    missing_entities = tuple(
        entity for entity in response_entities if not _entity_supported(entity, evidence_entities)
    )
    entity_grounding = (
        1.0
        if not response_entities
        else (len(response_entities) - len(missing_entities)) / len(response_entities)
    )

    unsupported: list[JsonDict] = []
    for relation in extraction.response_graph.relations:
        if not _relation_supported(relation, extraction.evidence_graph.relations, evidence_entities):
            unsupported.append(
                {
                    "subject": relation.subject,
                    "relation": relation.relation,
                    "object": relation.object,
                }
            )
    relation_preservation = (
        1.0
        if not extraction.response_graph.relations
        else (
            len(extraction.response_graph.relations) - len(unsupported)
        )
        / len(extraction.response_graph.relations)
    )
    cfi = (
        CFI_ENTITY_WEIGHT * entity_grounding
        + (1.0 - CFI_ENTITY_WEIGHT) * relation_preservation
    )
    cfi = _clamp01(cfi)
    return HalluGraphScore(
        entity_grounding=round(entity_grounding, 6),
        relation_preservation=round(relation_preservation, 6),
        composite_fidelity_index=round(cfi, 6),
        hallucination_score=round(1.0 - cfi, 6),
        missing_entities=missing_entities,
        unsupported_relations=tuple(unsupported),
    )


def parse_model_graph_response(
    text: str,
    *,
    prompt_sha256: str = "",
    completion_tokens: int = 0,
) -> GraphExtractionResult:
    """Parse model-emitted graph JSON, returning empty graphs on malformed output."""

    raw = text.strip()
    payload = _first_json_object(raw)
    if not isinstance(payload, Mapping):
        empty = ExtractedGraph(entities=(), relations=(), raw_response=raw)
        return GraphExtractionResult(
            response_graph=empty,
            evidence_graph=empty,
            prompt_sha256=prompt_sha256,
            completion_tokens=completion_tokens,
        )
    response_value = _graph_payload_alias(
        payload,
        nested_keys=("response", "answer", "claim", "response_graph", "claim_graph"),
        entities_keys=("response_entities", "claim_entities", "answer_entities"),
        relations_keys=("response_relations", "claim_relations", "answer_relations"),
    )
    evidence_value = _graph_payload_alias(
        payload,
        nested_keys=("evidence", "source", "context", "evidence_graph", "source_graph"),
        entities_keys=("evidence_entities", "source_entities", "context_entities"),
        relations_keys=("evidence_relations", "source_relations", "context_relations"),
    )
    return GraphExtractionResult(
        response_graph=_parse_graph(response_value, raw_response=raw),
        evidence_graph=_parse_graph(evidence_value, raw_response=raw),
        prompt_sha256=prompt_sha256,
        completion_tokens=int(completion_tokens),
    )


def probe_preconditions(
    config: ExperimentConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
    resolve_gguf: ResolveGguf = resolve_cached_gguf,
) -> PreflightResult:
    """Check CUDA, model cache, llama.cpp, and facts corpus before scoring."""

    checks = [_probe_cuda(config, command_runner=command_runner)]
    try:
        importlib.import_module("carnot.verify")
        checks.append(PreconditionCheck("carnot_verify_import", True, "import carnot.verify OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("carnot_verify_import", False, repr(exc)))

    model_specs, model_checks = _resolve_model(resolve_gguf)
    checks.extend(model_checks)

    try:
        importlib.import_module("llama_cpp")
        checks.append(PreconditionCheck("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("llama_cpp_import", False, repr(exc)))

    corpus_path = resolve_facts_corpus(config.repo_root)
    corpus_ok = _corpus_has_labels(corpus_path, config.sample_size) if corpus_path else False
    checks.append(
        PreconditionCheck(
            "facts_corpus_with_gold_labels",
            corpus_ok,
            str(corpus_path) if corpus_path else "missing",
        )
    )

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("carnot_verify_import", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    elif not available.get("facts_corpus_with_gold_labels", False):
        blocked_reason = "blocked_facts_corpus_missing"

    return PreflightResult(
        checks=tuple(checks),
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        corpus_path=corpus_path if corpus_ok else None,
    )


def resolve_facts_corpus(root: Path) -> Path | None:
    """Resolve the first RAGTruth-style facts corpus with labels."""

    for rel_path in FACTS_CORPUS_CANDIDATES:
        path = _repo_path(root, rel_path)
        if path.is_file():
            return path
    return None


def build_artifact_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_extractor: Any,
    model_specs: Mapping[str, Any],
    per_item_scores_path: Path,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]] | None = None,
    math_scores: Sequence[float] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    min_duration_s: float = MIN_REAL_DURATION_S,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Score rows, persist per-item scores, and build the Exp 3886 artifact."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean_rows = [dict(row) for row in rows if _valid_facts_row(row)]
    if not clean_rows:
        return build_blocked_artifact(
            reason="blocked_facts_corpus_missing",
            preconditions_checked=preconditions_checked or [],
            duration_s=0.0 if now_s is None else max(0.0, float(now_s) - start),
            model_specs=model_specs,
            tests_run=tests_run,
        )

    per_item_rows: list[JsonDict] = []
    labels: list[int] = []
    graph_scores: list[float] = []
    for index, row in enumerate(clean_rows):
        extraction = graph_extractor.extract_pair(
            str(row.get("answer") or ""),
            str(row.get("evidence_passage") or ""),
            index,
        )
        score = compute_hallugraph_score(extraction)
        labels.append(int(bool(row["is_hallucination"])))
        graph_scores.append(score.hallucination_score)
        per_item_rows.append(
            {
                "item_id": str(row.get("id") or row.get("question_id") or f"facts-{index}"),
                "index": index,
                "gold_ungrounded": bool(row["is_hallucination"]),
                "is_hallucination": int(bool(row["is_hallucination"])),
                "graph_score": score.hallucination_score,
                "entity_grounding": score.entity_grounding,
                "relation_preservation": score.relation_preservation,
                "composite_fidelity_index": score.composite_fidelity_index,
                "missing_entities": list(score.missing_entities),
                "unsupported_relations": list(score.unsupported_relations),
                "prompt_sha256": extraction.prompt_sha256,
                "completion_tokens": extraction.completion_tokens,
                "parse_fallback_used": extraction.parse_fallback_used,
                "answer_sha256": sha256_text(str(row.get("answer") or "")),
                "evidence_sha256": sha256_text(str(row.get("evidence_passage") or "")),
            }
        )

    math_bound = (
        [float(score) for score in math_scores]
        if math_scores is not None
        else score_rows_math_bound_ensemble(clean_rows)
    )
    labels, graph_scores, math_bound = _finite_triplets(labels, graph_scores, math_bound)
    for item, score in zip(per_item_rows, math_bound, strict=False):
        item["math_baseline_score"] = round(float(score), 6)

    _write_jsonl(per_item_scores_path, per_item_rows[: len(labels)])
    finished = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, finished - start), 6)
    if duration_s < min_duration_s:
        artifact = build_blocked_artifact(
            reason="blocked_graph_verifier_not_invoked",
            preconditions_checked=preconditions_checked or [],
            duration_s=duration_s,
            model_specs=model_specs,
            per_item_scores_path=per_item_scores_path,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    graph_auroc = round(float(tie_aware_auroc(labels, graph_scores)), 6)
    math_auroc = round(float(tie_aware_auroc(labels, math_bound)), 6)
    delta = round(float(graph_auroc - math_auroc), 6)
    artifact: JsonDict = {
        "honest_verdict": classify_terminal_verdict(delta, graph_auroc, math_auroc),
        "facts_catch_delta": float(delta),
        "graph_auroc": graph_auroc,
        "math_baseline_auroc": math_auroc,
        "per_item_scores_path": _artifact_path(per_item_scores_path),
        "model_invoked": True,
        "n_items": len(labels),
        "preconditions_checked": _checks_to_dicts(preconditions_checked or []),
        "model_specs": dict(model_specs),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "labels": labels,
                "graph_scores": [round(score, 6) for score in graph_scores],
                "math_scores": [round(score, 6) for score in math_bound],
                "model_specs": dict(model_specs),
                "per_item_scores_path": _artifact_path(per_item_scores_path),
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "score_summary": {
            "n_positive_hallucinations": int(sum(labels)),
            "n_negative_grounded": int(len(labels) - sum(labels)),
            "graph_score_min": min(graph_scores) if graph_scores else None,
            "graph_score_max": max(graph_scores) if graph_scores else None,
            "math_score_min": min(math_bound) if math_bound else None,
            "math_score_max": max(math_bound) if math_bound else None,
        },
        "per_item_scores_sha256": sha256_file(per_item_scores_path),
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    model_specs: Mapping[str, Any],
    per_item_scores_path: Path | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a terminal blocked artifact without fabricating signal fields."""

    artifact: JsonDict = {
        "honest_verdict": reason,
        "facts_catch_delta": 0.0,
        "graph_auroc": None,
        "math_baseline_auroc": None,
        "per_item_scores_path": _artifact_path(per_item_scores_path) if per_item_scores_path else "",
        "model_invoked": False,
        "n_items": 0,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "model_specs": dict(model_specs),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "blocked_reason": reason,
                "preconditions_checked": _checks_to_dicts(preconditions_checked),
                "model_specs": dict(model_specs),
            }
        ),
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def classify_terminal_verdict(delta: float, graph_auroc: float, math_auroc: float) -> str:
    """Apply the Exp 3886 facts-signal falsification gate."""

    if delta > 0.05:
        return (
            "complete: graph_grounding_FACTS_SIGNAL_REPRODUCED_"
            f"delta{delta:.3f}_graph{graph_auroc:.3f}_baseline{math_auroc:.3f}_bankable"
        )
    return (
        "complete: graph_grounding_NO_SIGNAL_"
        f"delta{delta:.3f}_exp3862_was_artifact_facts_stays_bound"
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    write: bool = True,
    command_runner: CommandRunner = subprocess.run,
    resolve_gguf: ResolveGguf = resolve_cached_gguf,
    llama_factory: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run Exp 3886 end to end, writing a blocked artifact on failed gates."""

    config = config or ExperimentConfig()
    started = config.start_time()
    preflight = probe_preconditions(config, command_runner=command_runner, resolve_gguf=resolve_gguf)
    if preflight.blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=preflight.blocked_reason,
            preconditions_checked=preflight.checks,
            duration_s=config.clock() - started,
            model_specs=preflight.model_specs,
        )
        if write:
            write_artifact(config.resolved_output_path(), artifact)
        return artifact

    assert preflight.corpus_path is not None
    rows = load_facts_rows(preflight.corpus_path, config.sample_size)
    try:
        extractor = LlamaGraphExtractor(preflight.model_specs, llama_factory=llama_factory)
        artifact = build_artifact_from_rows(
            rows,
            graph_extractor=extractor,
            model_specs=preflight.model_specs,
            per_item_scores_path=config.resolved_per_item_scores_path(),
            preconditions_checked=preflight.checks,
            started_s=started,
            now_s=config.clock(),
            min_duration_s=config.min_duration_s,
        )
    except Exception as exc:
        artifact = build_blocked_artifact(
            reason="blocked_llama_cpp_inference_failed",
            preconditions_checked=(
                *preflight.checks,
                PreconditionCheck("llama_cpp_inference", False, repr(exc)),
            ),
            duration_s=config.clock() - started,
            model_specs=preflight.model_specs,
        )

    if write:
        write_artifact(config.resolved_output_path(), artifact)
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Persist the terminal artifact as stable JSON."""

    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3886 artifact schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if type(artifact.get("facts_catch_delta")) is not float:
        raise ValueError("facts_catch_delta must be a bare float")
    if type(artifact.get("model_invoked")) is not bool:
        raise ValueError("model_invoked must be a bare bool")
    if not isinstance(artifact.get("n_items"), int) or int(artifact["n_items"]) < 0:
        raise ValueError("n_items must be a non-negative integer")
    if not isinstance(artifact.get("per_item_scores_path"), str):
        raise ValueError("per_item_scores_path must be a string")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    if any(not isinstance(value, str) for value in principles.values()):
        raise ValueError("field_principles must contain principle strings")
    uncovered = set(REQUIRED_PRINCIPLE_FIELDS) - set(principles)
    if uncovered:
        raise ValueError(f"field_principles missing required fields: {sorted(uncovered)}")
    for field in ("graph_auroc", "math_baseline_auroc"):
        value = artifact.get(field)
        if value is not None and not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{field} must be null or in [0, 1]")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    if verdict.startswith("complete:") and not bool(artifact.get("model_invoked")):
        raise ValueError("complete artifacts require model_invoked=true")
    if bool(artifact.get("model_invoked")) and float(duration) < MIN_REAL_DURATION_S:
        raise ValueError("model_invoked=true requires duration_s>=60")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the experiment script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    artifact = run_experiment(ExperimentConfig(repo_root=root), write=True)
    print(root / OUTPUT_REL_PATH)
    print(artifact["honest_verdict"])
    return 0


def _probe_cuda(config: ExperimentConfig, *, command_runner: CommandRunner) -> PreconditionCheck:
    try:
        proc = command_runner(
            [
                str(config.venv_python()),
                "-c",
                "import torch; assert torch.cuda.is_available()",
            ],
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
        detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
        return PreconditionCheck("cuda_available", proc.returncode == 0, detail)
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))


def _resolve_model(resolve_gguf: ResolveGguf) -> tuple[JsonDict, tuple[PreconditionCheck, ...]]:
    checks: list[PreconditionCheck] = []
    qwen_path = resolve_gguf(PRIMARY_REASONER_HF_ID)
    qwen_available = _model_path_available(qwen_path)
    checks.append(
        PreconditionCheck(
            "qwen3_6_35b_gguf_cached",
            qwen_available,
            str(qwen_path) if qwen_path else "missing",
        )
    )
    selected_hf_id = PRIMARY_REASONER_HF_ID
    selected_path = qwen_path if qwen_available else None
    fallback_used = False
    if not qwen_available:
        fallback_path = resolve_gguf(FALLBACK_REASONER_HF_ID)
        fallback_available = _model_path_available(fallback_path)
        checks.append(
            PreconditionCheck(
                "fallback_gemma_26b_gguf_cached",
                fallback_available,
                str(fallback_path) if fallback_path else "missing",
            )
        )
        selected_hf_id = FALLBACK_REASONER_HF_ID
        selected_path = fallback_path if fallback_available else None
        fallback_used = True

    model_specs: JsonDict = {
        "hf_id": selected_hf_id,
        "model_path": str(selected_path) if selected_path else "",
        "fallback_used": fallback_used,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "n_ctx": 4096,
        "n_batch": 256,
        "max_tokens": 220,
    }
    return model_specs, tuple(checks)


def _model_path_available(path: str | None) -> bool:
    if not path:
        return False
    model_path = Path(path)
    return model_path.is_file() and model_path.stat().st_size > 0


def _corpus_has_labels(path: Path | None, sample_size: int) -> bool:
    if path is None:
        return False
    rows = load_facts_rows(path, sample_size)
    labels = {int(bool(row.get("is_hallucination"))) for row in rows}
    return labels == {0, 1}


def _parse_graph(value: Any, *, raw_response: str) -> ExtractedGraph:
    if not isinstance(value, Mapping):
        return ExtractedGraph(entities=(), relations=(), raw_response=raw_response)
    entities = tuple(
        str(entity).strip()
        for entity in value.get("entities") or value.get("nodes") or value.get("entity_mentions") or []
        if str(entity).strip()
    )
    relations: list[RelationTriple] = []
    raw_relations = value.get("relations") or value.get("triples") or value.get("edges") or []
    if isinstance(raw_relations, Sequence) and not isinstance(raw_relations, (str, bytes)):
        for relation in raw_relations:
            if isinstance(relation, Sequence) and not isinstance(relation, (str, bytes, Mapping)):
                if len(relation) < 3:
                    continue
                subject, predicate, obj = (str(relation[0]), str(relation[1]), str(relation[2]))
                subject = subject.strip()
                predicate = predicate.strip()
                obj = obj.strip()
                if subject and predicate and obj:
                    relations.append(RelationTriple(subject, predicate, obj))
                continue
            if not isinstance(relation, Mapping):
                continue
            subject = str(relation.get("subject") or "").strip()
            predicate = str(
                relation.get("relation") or relation.get("predicate") or relation.get("verb") or ""
            ).strip()
            obj = str(
                relation.get("object")
                or relation.get("target")
                or relation.get("object_entity")
                or relation.get("destination")
                or ""
            ).strip()
            if subject and predicate and obj:
                relations.append(RelationTriple(subject, predicate, obj))
    return ExtractedGraph(
        entities=tuple(_dedupe_keep_order(entities)),
        relations=tuple(relations),
        raw_response=raw_response,
    )


def _graph_payload_alias(
    payload: Mapping[str, Any],
    *,
    nested_keys: Sequence[str],
    entities_keys: Sequence[str],
    relations_keys: Sequence[str],
) -> Any:
    for key in nested_keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    entities = next((payload.get(key) for key in entities_keys if key in payload), None)
    relations = next((payload.get(key) for key in relations_keys if key in payload), None)
    if entities is not None or relations is not None:
        return {
            "entities": entities or [],
            "relations": relations or [],
        }
    return None


def _first_json_object(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _fallback_graph_extraction(
    answer: str,
    evidence: str,
    *,
    raw_response: str,
    prompt_sha256: str,
    completion_tokens: int,
) -> GraphExtractionResult:
    probe = GraphGroundingProbe()

    def convert(text: str) -> ExtractedGraph:
        triples = probe.extract_triples(text)
        entities = tuple(
            _dedupe_keep_order(
                [part for triple in triples for part in (triple.subject, triple.object) if part]
            )
        )
        relations = tuple(
            RelationTriple(triple.subject, triple.relation, triple.object) for triple in triples
        )
        return ExtractedGraph(entities=entities, relations=relations, raw_response=raw_response)

    return GraphExtractionResult(
        response_graph=convert(answer),
        evidence_graph=convert(evidence),
        prompt_sha256=prompt_sha256,
        completion_tokens=completion_tokens,
        parse_fallback_used=True,
    )


def _fixture_item(
    *,
    item_id: str,
    claim: str,
    source: str,
    gold_hallucinated: bool,
    response_graph: Mapping[str, Any],
    evidence_graph: Mapping[str, Any],
    planted_hallucinated_relation: bool = False,
) -> JsonDict:
    return {
        "id": item_id,
        "claim": claim,
        "source": source,
        "answer": claim,
        "evidence_passage": source,
        "gold_hallucinated": bool(gold_hallucinated),
        "is_hallucination": int(bool(gold_hallucinated)),
        "planted_hallucinated_relation": bool(planted_hallucinated_relation),
        "scripted_response_graph": dict(response_graph),
        "scripted_evidence_graph": dict(evidence_graph),
    }


def _item_answer_evidence(item: Mapping[str, Any]) -> tuple[str, str]:
    answer = str(item.get("answer") or item.get("claim") or item.get("response") or "").strip()
    evidence = str(
        item.get("evidence_passage")
        or item.get("source")
        or item.get("context")
        or item.get("retrieved_context")
        or ""
    ).strip()
    if not answer or not evidence:
        raise ValueError("graph grounding item must include claim/answer and source/evidence text")
    return answer, evidence


def _score_payload(
    score: HalluGraphScore,
    extraction: GraphExtractionResult,
    model_invoked: bool,
) -> JsonDict:
    return {
        "eg": score.entity_grounding,
        "rp": score.relation_preservation,
        "cfi": score.composite_fidelity_index,
        "hallucination_score": score.hallucination_score,
        "model_invoked": bool(model_invoked),
        "missing_entities": list(score.missing_entities),
        "unsupported_relations": list(score.unsupported_relations),
        "completion_tokens": extraction.completion_tokens,
        "prompt_sha256": extraction.prompt_sha256,
        "parse_fallback_used": extraction.parse_fallback_used,
        "response_entities": list(extraction.response_graph.entities),
        "evidence_entities": list(extraction.evidence_graph.entities),
        "response_relations": [_relation_to_dict(rel) for rel in extraction.response_graph.relations],
        "evidence_relations": [_relation_to_dict(rel) for rel in extraction.evidence_graph.relations],
        "raw_response_sha256": sha256_text(extraction.response_graph.raw_response),
        "raw_response_chars": len(extraction.response_graph.raw_response),
    }


def _relation_to_dict(relation: RelationTriple) -> JsonDict:
    return {
        "subject": relation.subject,
        "relation": relation.relation,
        "object": relation.object,
    }


def _entity_supported(entity: str, evidence_entities: Sequence[str]) -> bool:
    entity_tokens = set(_norm_tokens(entity))
    if not entity_tokens:
        return True
    for evidence in evidence_entities:
        evidence_tokens = set(_norm_tokens(evidence))
        if not evidence_tokens:
            continue
        if entity_tokens == evidence_tokens or entity_tokens.issubset(evidence_tokens):
            return True
        if _jaccard(entity_tokens, evidence_tokens) >= 0.75:
            return True
    return False


def _relation_supported(
    relation: RelationTriple,
    evidence_relations: Sequence[RelationTriple],
    evidence_entities: Sequence[str],
) -> bool:
    subj_ok = _entity_supported(relation.subject, evidence_entities)
    obj_ok = _entity_supported(relation.object, evidence_entities)
    if not (subj_ok and obj_ok):
        return False
    relation_tokens = set(_norm_tokens(relation.relation))
    for evidence in evidence_relations:
        if not _entity_supported(relation.subject, (evidence.subject,)):
            continue
        if not _entity_supported(relation.object, (evidence.object,)):
            continue
        evidence_relation_tokens = set(_norm_tokens(evidence.relation))
        if relation_tokens == evidence_relation_tokens:
            return True
        if relation_tokens and _jaccard(relation_tokens, evidence_relation_tokens) >= 0.67:
            return True
    return False


def _norm_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe_keep_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = " ".join(_norm_tokens(cleaned))
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return tuple(output)


def _finite_triplets(
    labels: Sequence[int],
    graph_scores: Sequence[float],
    math_scores: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    clean_labels: list[int] = []
    clean_graph: list[float] = []
    clean_math: list[float] = []
    for label, graph, math_score in zip(labels, graph_scores, math_scores, strict=False):
        graph_f = float(graph)
        math_f = float(math_score)
        if math.isfinite(graph_f) and math.isfinite(math_f):
            clean_labels.append(int(label))
            clean_graph.append(_clamp01(graph_f))
            clean_math.append(_clamp01(math_f))
    return clean_labels, clean_graph, clean_math


def _valid_facts_row(row: Mapping[str, Any]) -> bool:
    return (
        "is_hallucination" in row
        and str(row.get("answer") or "").strip() != ""
        and str(row.get("evidence_passage") or "").strip() != ""
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checks_to_dicts(
    checks: Sequence[PreconditionCheck | Mapping[str, Any]],
) -> list[JsonDict]:
    output: list[JsonDict] = []
    for check in checks:
        if isinstance(check, PreconditionCheck):
            output.append(check.as_dict())
        else:
            output.append(dict(check))
    return output


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _artifact_path(path: Path | None) -> str:
    if path is None:
        return ""
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return str(Path(*parts[index:]))
    return str(path)


def _clip_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 32].rstrip() + " ... [truncated]"


def _extract_text(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return ""


def _completion_tokens(raw_response: Any, text: str) -> int:
    if isinstance(raw_response, Mapping):
        usage = raw_response.get("usage")
        if isinstance(usage, Mapping):
            try:
                return int(usage.get("completion_tokens") or 0)
            except Exception:
                return 0
    return len(str(text).split())


def _graph_empty(graph: ExtractedGraph) -> bool:
    return not graph.entities and not graph.relations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
