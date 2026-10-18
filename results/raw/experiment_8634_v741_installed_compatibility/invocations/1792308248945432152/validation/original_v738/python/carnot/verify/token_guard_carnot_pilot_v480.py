"""Exp 5251 Token-Guard-inspired fragment gate pilot.

Spec refs: REQ-VERIFY-5251, SCENARIO-VERIFY-5251.

This module tests Token-Guard's self-check/regenerate idea at Carnot's
fragment boundary. It deliberately uses deterministic prompt-provenance and
semantic-grounding checks plus an internal consistency-energy score. It does
not call logprob APIs, external text scorers, or the retired Phase D scoring
path.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair
from carnot.pipeline.semantic_grounding import verify_semantic_grounding
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5251_token_guard_carnot_pilot_v480"
EXPERIMENT_ID = 5251
SCHEMA = "carnot.token_guard_carnot_pilot.v480"
RESULT_RELATIVE_PATH = "results/experiment_5251_token_guard_carnot_pilot_v480.json"
SPEC_REFS = ("REQ-VERIFY-5251", "SCENARIO-VERIFY-5251")
RANDOM_SEED = 5251
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
DEFAULT_LLAMA_COMPLETION = Path(
    "/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-completion"
)
MANDATED_HEADLINE_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DEFAULT_FIXTURE_IDS = (
    "exp214-followup-omitted-premise-5",
    "exp214-followup-omitted-premise-6",
    "exp214-followup-omitted-premise-7",
    "exp214-followup-omitted-premise-8",
    "exp214-followup-entity-binding-1",
    "exp214-followup-entity-binding-2",
    "exp214-followup-entity-binding-3",
    "exp214-followup-entity-binding-4",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "The terminal value must say whether fragment self-checking helped, was null, "
        "was harmful, or blocked before inference."
    ),
    "inference_substrate": (
        "Declares that this artifact used local SOTA GGUF inference rather than cached "
        "candidate scoring or external text scoring."
    ),
    "model_specs": (
        "Names the exact mandated model, quantization, runtime, seed, and hashes so the "
        "live inference claim is replayable."
    ),
    "retired_phase_d_path_reopened": (
        "Guards against reviving off-path generated-text/logprob scoring as the primary metric."
    ),
    "fixtures_count": "Bounds the pilot to the requested existing deterministic fixture panel.",
    "unsupported_claim_delta": (
        "Primary harm metric: gated unsupported claims minus no-gate unsupported claims."
    ),
    "deterministic_violation_delta": (
        "Secondary metric: gated deterministic verifier violations minus no-gate violations."
    ),
    "regeneration_count": "Measures how often the fragment gate actually exercised regeneration.",
    "false_accepts": "Counts accepted fragments that later led to deterministic final violations.",
    "consumer_recommendation": "Converts the measured delta into keep, retire, or redesign guidance.",
}
REQUIRED_OBJECT_FIELDS = tuple(FIELD_PRINCIPLES)
UNSUPPORTED_SEMANTIC_TYPES = {"unsupported_reference", "answer_target_mismatch"}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
TIMESTAMP_LOG_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+\s+[IWE](?:\s+|$)")
INLINE_TIMESTAMP_LOG_RE = re.compile(r"\d+\.\d+\.\d+\.\d+\s+[IWE](?:\s+[^\n]*)?")
LLAMA_LOG_CONTINUATION_PREFIXES = (
    "repeat_last_n =",
    "dry_multiplier =",
    "top_k =",
    "min_p =",
    "typ_p =",
    "mirostat =",
    "generate:",
    "main:",
    "common_perf_",
    "llama_perf_",
    "llama_context:",
    "llama_model_loader:",
    "llama_kv_cache:",
    "ggml_cuda_",
    "load_tensors:",
    "print_info:",
)
WORD_NUMBERS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "half": 0.5,
    "quarter": 0.25,
    "twice": 2.0,
    "double": 2.0,
}


@dataclass(frozen=True)
class PilotFixture:
    """One existing Exp 214 fixture with deterministic final-answer support."""

    fixture_id: str
    prompt: str
    expected_outcome: str
    source_artifact: str
    taxonomy_label: str

    @property
    def expected_numbers(self) -> tuple[float, ...]:
        return tuple(extract_numbers(self.expected_outcome))


@dataclass(frozen=True)
class GenerationReceipt:
    """One live or fake generation call with replayable hashes."""

    tag: str
    prompt: str
    text: str
    seed: int
    command: Sequence[str]
    duration_s: float
    returncode: int
    stderr_tail: str
    stdout_tail: str

    @property
    def prompt_checksum(self) -> str:
        return sha16(self.prompt)

    @property
    def completion_checksum(self) -> str:
        return sha16(self.text)

    def compact(self) -> JsonDict:
        return {
            "tag": self.tag,
            "prompt_checksum": self.prompt_checksum,
            "completion_checksum": self.completion_checksum,
            "seed": self.seed,
            "duration_s": round(float(self.duration_s), 6),
            "returncode": int(self.returncode),
            "command": list(self.command),
            "stderr_tail": self.stderr_tail[-500:],
            "stdout_tail": self.stdout_tail[-500:],
        }


@dataclass(frozen=True)
class GateDecision:
    """Fragment-level deterministic gate result."""

    accepted: bool
    energy_score: float
    unsupported_claim_count: int
    deterministic_violation_count: int
    reasons: tuple[str, ...]
    unsupported_numbers: tuple[float, ...]
    semantic_violation_types: tuple[str, ...]


@dataclass(frozen=True)
class FinalCheck:
    """Final deterministic support check for one fixture answer."""

    unsupported_claim_count: int
    deterministic_violation_count: int
    accuracy: bool
    energy_score: float
    final_numbers: tuple[float, ...]
    semantic_violation_types: tuple[str, ...]


@dataclass(frozen=True)
class PreconditionReport:
    """Resources checked before any live inference begins."""

    ok: bool
    checks: list[JsonDict]
    selected_model: JsonDict | None
    runtime_command: Sequence[str]
    blocked_reason: str = ""


class LlamaCompletionRunner:
    """Small subprocess wrapper around a CUDA-capable `llama-completion` binary."""

    def __init__(
        self,
        *,
        model_path: str,
        runtime_path: str | Path,
        n_ctx: int = 1536,
        n_gpu_layers: int = 999,
        timeout_s: int = 240,
    ) -> None:
        self.model_path = str(model_path)
        self.runtime_path = str(runtime_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.timeout_s = int(timeout_s)

    def command(self, prompt: str, *, max_tokens: int, seed: int) -> list[str]:
        rendered = render_gemma_turn_prompt(prompt)
        return [
            self.runtime_path,
            "-m",
            self.model_path,
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.n_ctx),
            "-n",
            str(int(max_tokens)),
            "--temp",
            "0",
            "--seed",
            str(int(seed)),
            "-p",
            rendered,
            "-no-cnv",
            "--no-display-prompt",
            "--simple-io",
            "--no-warmup",
        ]

    def generate(self, prompt: str, *, max_tokens: int, seed: int, tag: str) -> GenerationReceipt:
        started = time.monotonic()
        cmd = self.command(prompt, max_tokens=max_tokens, seed=seed)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            exc_stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            exc_stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
            stdout = exc_stdout or ""
            stderr = f"{exc_stderr or ''}\ntimeout_s={self.timeout_s}"
            returncode = -124
        duration = time.monotonic() - started
        raw = f"{stdout}\n{stderr}"
        text = clean_llama_completion_output(raw)
        return GenerationReceipt(
            tag=tag,
            prompt=prompt,
            text=text,
            seed=seed,
            command=cmd,
            duration_s=duration,
            returncode=returncode,
            stderr_tail=stderr[-500:],
            stdout_tail=stdout[-500:],
        )


def sha16(text: str | bytes) -> str:
    payload = text if isinstance(text, bytes) else text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_gemma_turn_prompt(prompt: str) -> str:
    return f"<|turn>user\n{prompt.strip()}<turn|>\n<|turn>model\n"


def clean_llama_completion_output(raw: str) -> str:
    lines = []
    for line in raw.replace("\r", "\n").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if TIMESTAMP_LOG_RE.match(stripped):
            continue
        stripped = re.sub(r"\[end of text\].*$", " ", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"Exiting\.\.\..*$", " ", stripped).strip()
        stripped = INLINE_TIMESTAMP_LOG_RE.sub(" ", stripped).strip()
        if not stripped:
            continue
        if stripped.startswith(("build ", "model ", "modalities ", "available commands")):
            continue
        if stripped.startswith(LLAMA_LOG_CONTINUATION_PREFIXES):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    for token in (
        "<|channel>thought",
        "<|channel>final",
        "<channel|>",
        "<turn|>",
        "<|turn>model",
        "<|turn>user",
        "</s>",
    ):
        text = text.replace(token, " ")
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0)
        scale = 0.01 if raw.endswith("%") else 1.0
        cleaned = raw.rstrip("%").replace(",", "")
        values.append(float(cleaned) * scale)
    lowered = text.lower()
    for word, value in WORD_NUMBERS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            values.append(value)
    return values


def final_numbers(text: str) -> tuple[float, ...]:
    matches = list(re.finditer(r"\b(?:answer|final)\b\s*:?\s*([^\n.。]*)", text, re.IGNORECASE))
    if matches:
        nums = extract_numbers(matches[-1].group(1))
        if nums:
            return tuple(nums)
    nums = extract_numbers(text)
    return tuple(nums[-1:]) if nums else ()


def close_number(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)


def derived_number_closure(seed_numbers: Iterable[float], rounds: int = 2) -> set[float]:
    values = {round(float(v), 6) for v in seed_numbers if math.isfinite(float(v))}
    for _ in range(rounds):
        current = list(values)
        for a, b in itertools.product(current, repeat=2):
            candidates = [a + b, a - b, b - a, a * b]
            if abs(b) > 1e-9:
                candidates.append(a / b)
            if abs(a) > 1e-9:
                candidates.append(b / a)
            for candidate in candidates:
                if math.isfinite(candidate) and abs(candidate) <= 1_000_000:
                    values.add(round(float(candidate), 6))
    return values


def unsupported_numbers_for(
    prompt: str, text: str, prior_fragments: Sequence[str] | None = None
) -> tuple[float, ...]:
    prior_fragments = prior_fragments or []
    support = derived_number_closure(
        [*extract_numbers(prompt), *[n for frag in prior_fragments for n in extract_numbers(frag)]]
    )
    unsupported = []
    for number in extract_numbers(text):
        if not any(close_number(number, candidate) for candidate in support):
            unsupported.append(float(number))
    return tuple(unsupported)


def load_selected_fixtures(
    repo_root: Path | str = REPO_ROOT, fixture_ids: Sequence[str] = DEFAULT_FIXTURE_IDS
) -> list[PilotFixture]:
    corpus_path = Path(repo_root) / "data" / "research" / "semantic_failure_corpus_214.jsonl"
    wanted = set(fixture_ids)
    fixtures: list[PilotFixture] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("example_id") not in wanted:
            continue
        diagnosis = record.get("gold_diagnosis") or {}
        fixtures.append(
            PilotFixture(
                fixture_id=str(record["example_id"]),
                prompt=str(record["prompt"]),
                expected_outcome=str(diagnosis.get("expected_outcome") or ""),
                source_artifact=str(record.get("source_artifact") or ""),
                taxonomy_label=str(diagnosis.get("taxonomy_label") or ""),
            )
        )
    order = {fixture_id: index for index, fixture_id in enumerate(fixture_ids)}
    return sorted(fixtures, key=lambda fixture: order[fixture.fixture_id])


def score_fragment(
    fixture: PilotFixture, fragment: str, prior_fragments: Sequence[str] | None = None
) -> GateDecision:
    prior_fragments = prior_fragments or []
    grounding = verify_semantic_grounding(fixture.prompt, fragment)
    semantic_types = tuple(v.violation_type for v in grounding.violations)
    unsupported_semantic = [t for t in semantic_types if t in UNSUPPORTED_SEMANTIC_TYPES]
    unsupported_nums = unsupported_numbers_for(fixture.prompt, fragment, prior_fragments)
    consistency_energy = SemanticConsistencyVerifier().score(fragment)
    unsupported_count = len(unsupported_semantic) + len(unsupported_nums)
    deterministic_count = len(unsupported_semantic) + int(consistency_energy > 0.25)
    energy_score = float(consistency_energy + unsupported_count + 0.25 * deterministic_count)
    reasons: list[str] = []
    if unsupported_nums:
        reasons.append("unsupported_numeric_claim")
    if unsupported_semantic:
        reasons.append("unsupported_semantic_claim")
    if consistency_energy > 0.25:
        reasons.append("semantic_consistency_energy")
    accepted = not reasons
    return GateDecision(
        accepted=accepted,
        energy_score=round(energy_score, 6),
        unsupported_claim_count=unsupported_count,
        deterministic_violation_count=deterministic_count,
        reasons=tuple(reasons),
        unsupported_numbers=unsupported_nums,
        semantic_violation_types=semantic_types,
    )


def check_final_answer(fixture: PilotFixture, response: str) -> FinalCheck:
    grounding = verify_semantic_grounding(fixture.prompt, response)
    semantic_types = tuple(v.violation_type for v in grounding.violations)
    unsupported_semantic = [t for t in semantic_types if t in UNSUPPORTED_SEMANTIC_TYPES]
    unsupported_nums = unsupported_numbers_for(fixture.prompt, response)
    finals = final_numbers(response)
    accuracy = bool(finals) and any(
        close_number(answer, expected) for answer in finals for expected in fixture.expected_numbers
    )
    answer_mismatch = 0 if accuracy else 1
    unsupported_count = len(unsupported_semantic) + len(unsupported_nums)
    deterministic_count = len(grounding.violations) + answer_mismatch
    energy_score = float(SemanticConsistencyVerifier().score(response) + deterministic_count)
    return FinalCheck(
        unsupported_claim_count=unsupported_count,
        deterministic_violation_count=deterministic_count,
        accuracy=accuracy,
        energy_score=round(energy_score, 6),
        final_numbers=finals,
        semantic_violation_types=semantic_types,
    )


def baseline_prompt(fixture: PilotFixture) -> str:
    return (
        "Solve the word problem. Keep the answer concise. End with 'FINAL: <number>'.\n\n"
        f"Problem: {fixture.prompt}"
    )


def fragment_prompt(fixture: PilotFixture) -> str:
    return (
        "Generate exactly one short reasoning fragment for this word problem. "
        "Use only quantities stated in the problem or directly derived by arithmetic. "
        "Do not give the final answer unless the fragment proves it.\n\n"
        f"Problem: {fixture.prompt}"
    )


def repair_fragment_prompt(fixture: PilotFixture, rejected: str, reasons: Sequence[str]) -> str:
    return (
        "The previous fragment was rejected by deterministic Carnot gates for "
        f"{', '.join(reasons)}. Regenerate one corrected short fragment using only supported facts.\n\n"
        f"Problem: {fixture.prompt}\nRejected fragment: {rejected}"
    )


def gated_final_prompt(fixture: PilotFixture, accepted_fragments: Sequence[str]) -> str:
    context = (
        "\n".join(f"- {fragment}" for fragment in accepted_fragments) or "- no fragment accepted"
    )
    return (
        "Solve the problem using only the accepted fragments and the original problem. "
        "End with 'FINAL: <number>'.\n\n"
        f"Problem: {fixture.prompt}\nAccepted fragments:\n{context}"
    )


def run_fixture_pair(
    fixture: PilotFixture,
    *,
    generator: Any,
    seed: int,
) -> JsonDict:
    baseline_receipt = generator.generate(
        baseline_prompt(fixture), max_tokens=192, seed=seed, tag=f"{fixture.fixture_id}:baseline"
    )
    baseline_check = check_final_answer(fixture, baseline_receipt.text)

    first_fragment = generator.generate(
        fragment_prompt(fixture),
        max_tokens=128,
        seed=seed + 1,
        tag=f"{fixture.fixture_id}:fragment0",
    )
    first_gate = score_fragment(fixture, first_fragment.text, prior_fragments=[])
    regeneration_count = 0
    accepted_fragments: list[str] = []
    gate_receipts = [first_fragment]
    gate_decisions = [first_gate]

    if first_gate.accepted:
        accepted_fragments.append(first_fragment.text)
    else:
        regeneration_count = 1
        repaired = generator.generate(
            repair_fragment_prompt(fixture, first_fragment.text, first_gate.reasons),
            max_tokens=128,
            seed=seed + 2,
            tag=f"{fixture.fixture_id}:fragment1",
        )
        repaired_gate = score_fragment(fixture, repaired.text, prior_fragments=[])
        gate_receipts.append(repaired)
        gate_decisions.append(repaired_gate)
        if repaired_gate.accepted:
            accepted_fragments.append(repaired.text)

    gated_receipt = generator.generate(
        gated_final_prompt(fixture, accepted_fragments),
        max_tokens=192,
        seed=seed + 3,
        tag=f"{fixture.fixture_id}:gated_final",
    )
    gated_check = check_final_answer(fixture, gated_receipt.text)
    false_accept = bool(accepted_fragments) and gated_check.deterministic_violation_count > 0

    return {
        "fixture_id": fixture.fixture_id,
        "taxonomy_label": fixture.taxonomy_label,
        "expected_outcome": fixture.expected_outcome,
        "baseline": {
            "receipt": baseline_receipt.compact(),
            "check": baseline_check.__dict__,
        },
        "gated": {
            "receipts": [receipt.compact() for receipt in gate_receipts + [gated_receipt]],
            "gate_decisions": [decision.__dict__ for decision in gate_decisions],
            "accepted_fragment_count": len(accepted_fragments),
            "regeneration_count": regeneration_count,
            "final_check": gated_check.__dict__,
            "false_accept": false_accept,
        },
    }


def wrapped(value: Any, field: str) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def stable_checksum(payload: JsonDict) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha16(encoded)


def build_blocked_artifact(
    *,
    preconditions: PreconditionReport,
    started_at: str,
    finished_at: str,
    duration_s: float,
) -> JsonDict:
    reason = preconditions.blocked_reason or "blocked_precondition"
    model_value = {
        "headline_model": None,
        "quantization": None,
        "runtime_command": list(preconditions.runtime_command),
        "seed": RANDOM_SEED,
        "prompt_checksums": [],
        "completion_checksums": [],
    }
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions.checks,
        "honest_verdict": {
            "value": f"{reason}: fragment self-checking not tested because preconditions failed",
            "principle": FIELD_PRINCIPLES["honest_verdict"],
        },
        "inference_substrate": wrapped(INFERENCE_SUBSTRATE, "inference_substrate"),
        "model_specs": wrapped(model_value, "model_specs"),
        "retired_phase_d_path_reopened": wrapped(False, "retired_phase_d_path_reopened"),
        "fixtures_count": wrapped(0, "fixtures_count"),
        "unsupported_claim_delta": wrapped(0.0, "unsupported_claim_delta"),
        "deterministic_violation_delta": wrapped(0.0, "deterministic_violation_delta"),
        "regeneration_count": wrapped(0, "regeneration_count"),
        "false_accepts": wrapped(0, "false_accepts"),
        "consumer_recommendation": {
            "value": "blocked_precondition: do not headline; fix local CUDA GGUF runtime before retesting",
            "principle": FIELD_PRINCIPLES["consumer_recommendation"],
        },
        "accuracy_change": {
            "value": 0.0,
            "principle": "Blocked runs have no measured accuracy delta.",
        },
        "rows": [],
    }
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    rows: Sequence[JsonDict],
    preconditions: PreconditionReport,
    started_at: str,
    finished_at: str,
    duration_s: float,
) -> JsonDict:
    baseline_unsupported = sum(row["baseline"]["check"]["unsupported_claim_count"] for row in rows)
    gated_unsupported = sum(row["gated"]["final_check"]["unsupported_claim_count"] for row in rows)
    baseline_violations = sum(
        row["baseline"]["check"]["deterministic_violation_count"] for row in rows
    )
    gated_violations = sum(
        row["gated"]["final_check"]["deterministic_violation_count"] for row in rows
    )
    baseline_accuracy = sum(1 for row in rows if row["baseline"]["check"]["accuracy"]) / len(rows)
    gated_accuracy = sum(1 for row in rows if row["gated"]["final_check"]["accuracy"]) / len(rows)
    regeneration_count = sum(row["gated"]["regeneration_count"] for row in rows)
    false_accepts = sum(1 for row in rows if row["gated"]["false_accept"])
    unsupported_delta = float(gated_unsupported - baseline_unsupported)
    violation_delta = float(gated_violations - baseline_violations)
    accuracy_change = float(gated_accuracy - baseline_accuracy)
    helped = unsupported_delta < 0 and violation_delta <= 0 and accuracy_change >= 0
    harmful = unsupported_delta > 0 or violation_delta > 0 or accuracy_change < 0
    if helped:
        verdict_tail = "fragment self-checking helped on this bounded panel"
        recommendation = (
            "keep_as_pilot: fragment gate reduced unsupported claims without accuracy loss"
        )
    elif harmful:
        verdict_tail = "fragment self-checking was harmful on this bounded panel"
        recommendation = "redesign_or_retire: fragment gate worsened at least one primary metric"
    else:
        verdict_tail = "fragment self-checking was null on this bounded panel"
        recommendation = "retire_or_redesign: no measured unsupported-claim reduction"

    prompt_checksums = [
        receipt["prompt_checksum"]
        for row in rows
        for receipt in [row["baseline"]["receipt"], *row["gated"]["receipts"]]
    ]
    completion_checksums = [
        receipt["completion_checksum"]
        for row in rows
        for receipt in [row["baseline"]["receipt"], *row["gated"]["receipts"]]
    ]
    selected_model = preconditions.selected_model or {}
    model_value = {
        "headline_model": selected_model.get("hf_id"),
        "model_name": selected_model.get("name"),
        "model_path": selected_model.get("model_path"),
        "quantization": selected_model.get("quantization")
        or infer_quantization(str(selected_model.get("model_path") or "")),
        "runtime_command": list(preconditions.runtime_command),
        "seed": RANDOM_SEED,
        "prompt_checksums": prompt_checksums,
        "completion_checksums": completion_checksums,
    }
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions.checks,
        "honest_verdict": {
            "value": f"complete: {verdict_tail}",
            "principle": FIELD_PRINCIPLES["honest_verdict"],
        },
        "inference_substrate": wrapped(INFERENCE_SUBSTRATE, "inference_substrate"),
        "model_specs": wrapped(model_value, "model_specs"),
        "retired_phase_d_path_reopened": wrapped(False, "retired_phase_d_path_reopened"),
        "fixtures_count": wrapped(len(rows), "fixtures_count"),
        "unsupported_claim_delta": wrapped(unsupported_delta, "unsupported_claim_delta"),
        "deterministic_violation_delta": wrapped(violation_delta, "deterministic_violation_delta"),
        "regeneration_count": wrapped(regeneration_count, "regeneration_count"),
        "false_accepts": wrapped(false_accepts, "false_accepts"),
        "consumer_recommendation": {
            "value": recommendation,
            "principle": FIELD_PRINCIPLES["consumer_recommendation"],
        },
        "accuracy_change": {
            "value": round(accuracy_change, 6),
            "principle": "Accuracy guards against reducing unsupported claims by damaging answer correctness.",
        },
        "baseline_accuracy": round(baseline_accuracy, 6),
        "gated_accuracy": round(gated_accuracy, 6),
        "baseline_unsupported_claims": baseline_unsupported,
        "gated_unsupported_claims": gated_unsupported,
        "baseline_deterministic_violations": baseline_violations,
        "gated_deterministic_violations": gated_violations,
        "rows": list(rows),
    }
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonDict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_OBJECT_FIELDS:
        item = artifact.get(field)
        if not isinstance(item, dict):
            errors.append(f"missing_object_field:{field}")
            continue
        if "value" not in item:
            errors.append(f"missing_value:{field}")
        if item.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"missing_principle:{field}")
    verdict = artifact.get("honest_verdict", {}).get("value", "")
    if not str(verdict).startswith(("complete:", "blocked_")):
        errors.append("honest_verdict_prefix")
    if artifact.get("retired_phase_d_path_reopened", {}).get("value") is not False:
        errors.append("retired_phase_d_reopened")
    model = artifact.get("model_specs", {}).get("value", {})
    headline = model.get("headline_model")
    if headline is not None and headline not in MANDATED_HEADLINE_MODELS:
        errors.append("headline_model_not_mandated_sota")
    count = artifact.get("fixtures_count", {}).get("value")
    if str(verdict).startswith("complete:") and not (8 <= int(count) <= 12):
        errors.append("complete_fixture_count_out_of_bounds")
    return errors


def infer_quantization(model_path: str) -> str | None:
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "IQ2_M", "IQ2_XXS"):
        if token.lower() in model_path.lower():
            return token
    return None


def select_headline_model(specs: Sequence[JsonDict]) -> JsonDict | None:
    for preferred in ("unsloth/gemma-4-26B-A4B-it-GGUF", "unsloth/Qwen3.6-35B-A3B-GGUF"):
        for spec in specs:
            if spec.get("hf_id") == preferred:
                return {
                    **spec,
                    "quantization": infer_quantization(str(spec.get("model_path") or "")),
                }
    return None


def check_preconditions(repo_root: Path | str = REPO_ROOT) -> PreconditionReport:
    del repo_root
    checks: list[JsonDict] = []
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        detail = f"torch_cuda_devices={torch.cuda.device_count()}"
    except Exception as exc:  # pragma: no cover - depends on host packages.
        cuda_ok = False
        detail = repr(exc)
    checks.append({"resource": "cuda_gpu", "available": cuda_ok, "detail": detail})

    runtime = Path(os.environ.get("CARNOT_LLAMA_COMPLETION", str(DEFAULT_LLAMA_COMPLETION)))
    runtime_ok = runtime.is_file() and os.access(runtime, os.X_OK)
    ldd_text = ""
    if runtime_ok:
        ldd = subprocess.run(["ldd", str(runtime)], capture_output=True, text=True, timeout=10)
        ldd_text = f"{ldd.stdout}\n{ldd.stderr}"
        runtime_ok = "libggml-cuda" in ldd_text and "libcuda.so" in ldd_text
    checks.append(
        {
            "resource": "local_gguf_runtime",
            "available": runtime_ok,
            "path": str(runtime),
            "cuda_linked": "libggml-cuda" in ldd_text,
        }
    )

    specs = cached_sota_pair(gpu_indices=(0, 1)) or []
    selected = select_headline_model(specs)
    model_ok = selected is not None and Path(str(selected.get("model_path"))).is_file()
    checks.append(
        {
            "resource": "mandated_sota_gguf",
            "available": model_ok,
            "model": selected.get("hf_id") if selected else None,
            "model_path": selected.get("model_path") if selected else None,
        }
    )

    ok = all(check["available"] for check in checks)
    missing = next((check["resource"] for check in checks if not check["available"]), "")
    runtime_command = (str(runtime),)
    if selected:
        runtime_command = (
            str(runtime),
            "-m",
            str(selected.get("model_path")),
            "-ngl",
            "999",
            "-c",
            "1536",
            "-no-cnv",
            "--no-display-prompt",
            "--simple-io",
        )
    return PreconditionReport(
        ok=ok,
        checks=checks,
        selected_model=selected if model_ok else None,
        runtime_command=runtime_command,
        blocked_reason=f"blocked_precondition_{missing}" if missing else "",
    )


def run_pilot(
    *,
    repo_root: Path | str = REPO_ROOT,
    generator: Any | None = None,
    fixtures: Sequence[PilotFixture] | None = None,
    preconditions: PreconditionReport | None = None,
    write: bool = True,
) -> JsonDict:
    started_mono = time.monotonic()
    started_at = utc_now()
    preconditions = preconditions or check_preconditions(repo_root)
    if not preconditions.ok:
        finished_at = utc_now()
        artifact = build_blocked_artifact(
            preconditions=preconditions,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=time.monotonic() - started_mono,
        )
    else:
        fixture_rows = list(fixtures) if fixtures is not None else load_selected_fixtures(repo_root)
        if generator is None:
            selected = preconditions.selected_model or {}
            generator = LlamaCompletionRunner(
                model_path=str(selected["model_path"]),
                runtime_path=preconditions.runtime_command[0],
            )
        rows = [
            run_fixture_pair(fixture, generator=generator, seed=RANDOM_SEED + index * 10)
            for index, fixture in enumerate(fixture_rows)
        ]
        finished_at = utc_now()
        artifact = build_complete_artifact(
            rows=rows,
            preconditions=preconditions,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=time.monotonic() - started_mono,
        )
    if write:
        output = Path(repo_root) / RESULT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - exercised by live artifact command.
    artifact = run_pilot(write=True)
    print(
        json.dumps(
            {"result_path": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    main()
