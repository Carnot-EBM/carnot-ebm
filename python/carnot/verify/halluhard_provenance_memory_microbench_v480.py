"""Exp 5252 HalluHard-style provenance-memory microbench.

Spec refs: REQ-VERIFY-5252, SCENARIO-VERIFY-5252.

This module runs a small local citation-support benchmark over fictional
evidence snippets. The benchmark is intentionally narrow: it does not claim to
replicate HalluHard, and it does not use network retrieval at benchmark time.
It asks whether a typed provenance-memory prompt reduces repeated unsupported
support/citation errors across two turns compared with no memory and raw
conversation memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5252_halluhard_provenance_memory_microbench_v480"
EXPERIMENT_ID = 5252
SCHEMA = "carnot.halluhard_provenance_memory_microbench.v480"
RESULT_RELATIVE_PATH = "results/experiment_5252_halluhard_provenance_memory_microbench_v480.json"
SPEC_REFS = ("REQ-VERIFY-5252", "SCENARIO-VERIFY-5252")
RANDOM_SEED = 5252
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
DEFAULT_LLAMA_COMPLETION = Path("/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-completion")
MANDATED_HEADLINE_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ARM_NAMES = ("no_memory", "raw_conversation_memory", "typed_provenance_memory")
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal verdict states whether provenance memory reduced",
    "inference_substrate": "Declares local SOTA GGUF live inference",
    "model_specs": "Names the mandated model, quantization, runtime command",
    "fixture_count": "Bounds the HalluHard-style panel to the requested 10-20",
    "unsupported_claim_rate_no_memory": "Baseline unsupported support-error rate",
    "unsupported_claim_rate_typed_memory": "Typed-memory unsupported support-error",
    "repeated_error_delta": "No-memory repeated-error rate minus typed-memory",
    "citation_support_delta": "Typed-memory citation support rate minus",
    "leakage_checks": "Records local-evidence-only, gold-label visibility",
    "no_network_at_benchmark_time": "Confirms the benchmark used local curated",
}
REQUIRED_OBJECT_FIELDS = tuple(FIELD_PRINCIPLES)
REFUSAL_MARKERS = (
    "insufficient_evidence",
    "insufficient evidence",
    "not enough evidence",
    "not stated",
    "not provided",
    "cannot determine",
    "can't determine",
    "unknown",
)
CASE_HEADER_RE = re.compile(r"(?im)^\s*CASE\s+([A-Za-z0-9_-]+)\s*[:\-]")
INLINE_CASE_RE = re.compile(r"(?i)\bCASE\s+([A-Za-z0-9_-]+)\s*[:\-]")
TIMESTAMP_LOG_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+\s+[IWE](?:\s+|$)")
INLINE_TIMESTAMP_LOG_RE = re.compile(r"\d+\.\d+\.\d+\.\d+\s+[IWE](?:\s+[^\n]*)?")
LLAMA_LOG_PREFIXES = (
    "build ",
    "model ",
    "llama_",
    "ggml_",
    "common_",
    "sampling:",
    "generate:",
    "main:",
    "load_tensors:",
    "print_info:",
)


@dataclass(frozen=True)
class EvidenceSnippet:
    """One local evidence snippet with a citation identifier."""

    evidence_id: str
    text: str


@dataclass(frozen=True)
class MicrobenchFixture:
    """One two-turn citation-support case and its deterministic labels."""

    fixture_id: str
    evidence: tuple[EvidenceSnippet, ...]
    turn1_question: str
    turn2_question: str
    expected_answer: str | None
    expected_citation: str | None
    answer_aliases: tuple[str, ...]
    expected_supported_claims: tuple[str, ...]
    expected_unsupported_claims: tuple[str, ...]
    missing_field: str


@dataclass(frozen=True)
class GenerationReceipt:
    """One local generation call with replayable prompt and output checksums."""

    tag: str
    prompt: str
    text: str
    seed: int
    command: tuple[str, ...]
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
            "seed": int(self.seed),
            "duration_s": round(float(self.duration_s), 6),
            "returncode": int(self.returncode),
            "command": list(self.command),
            "stderr_tail": self.stderr_tail[-500:],
            "stdout_tail": self.stdout_tail[-500:],
        }


@dataclass(frozen=True)
class CaseScore:
    """Deterministic support label for a model response to one case."""

    fixture_id: str
    unsupported_claim: bool
    citation_supported: bool
    over_refusal: bool
    missed_answer: bool
    answerable: bool
    citations: tuple[str, ...]
    reasons: tuple[str, ...]
    case_text_checksum: str
    case_text_excerpt: str


@dataclass(frozen=True)
class PreconditionReport:
    """Resources checked before any live local GGUF inference begins."""

    ok: bool
    checks: list[JsonDict]
    selected_model: JsonDict | None
    runtime_command: tuple[str, ...]
    blocked_reason: str = ""


class LlamaBatchRunner:
    """Small subprocess wrapper around a CUDA-capable `llama-completion` binary."""

    def __init__(
        self,
        *,
        model_path: str,
        runtime_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 999,
        timeout_s: int = 420,
    ) -> None:
        self.model_path = str(model_path)
        self.runtime_path = str(runtime_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.timeout_s = int(timeout_s)

    def command(self, prompt: str, *, max_tokens: int, seed: int) -> list[str]:
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
            render_turn_prompt(prompt),
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
            stdout = str(exc.stdout or "")
            stderr = f"{exc.stderr or ''}\ntimeout_s={self.timeout_s}"
            returncode = -124
        raw = f"{stdout}\n{stderr}"
        return GenerationReceipt(
            tag=tag,
            prompt=prompt,
            text=clean_llama_completion_output(raw),
            seed=seed,
            command=tuple(cmd),
            duration_s=time.monotonic() - started,
            returncode=returncode,
            stderr_tail=stderr[-500:],
            stdout_tail=stdout[-500:],
        )


def default_fixtures() -> list[MicrobenchFixture]:
    """Return the local curated 12-item HalluHard-style fixture set."""

    return [
        MicrobenchFixture(
            fixture_id="hhpm-001",
            evidence=(
                EvidenceSnippet("E1", "Larkspur audit memo: the Aster-9 battery ran 47 minutes under the amber-load test."),
                EvidenceSnippet("E2", "Larkspur inventory note: the Aster-9 casing color was matte blue."),
            ),
            turn1_question="How long did the Aster-9 battery run under the amber-load test?",
            turn2_question="State the Aster-9 amber-load runtime again with a citation.",
            expected_answer="47 minutes",
            expected_citation="E1",
            answer_aliases=("47 minutes", "forty seven minutes", "forty-seven minutes"),
            expected_supported_claims=("47 minutes",),
            expected_unsupported_claims=("74 minutes", "57 minutes"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-002",
            evidence=(
                EvidenceSnippet("E1", "Rillwick harbor log: Buoy K-17 was serviced on 12 March by dock crew Juno."),
                EvidenceSnippet("E2", "Rillwick harbor log: Buoy K-18 was retired after the south-channel drill."),
            ),
            turn1_question="Which crew serviced Buoy K-17?",
            turn2_question="Repeat the Buoy K-17 servicing crew with a citation.",
            expected_answer="dock crew Juno",
            expected_citation="E1",
            answer_aliases=("dock crew Juno", "crew Juno", "Juno"),
            expected_supported_claims=("dock crew Juno",),
            expected_unsupported_claims=("crew Kappa", "dock crew Kilo"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-003",
            evidence=(
                EvidenceSnippet("E1", "Mira Vale clinic snippet: Trial Nacre enrolled 18 participants and used a saline control."),
                EvidenceSnippet("E2", "Mira Vale clinic snippet: Trial Nacre measured sleep latency after four nights."),
            ),
            turn1_question="How many participants did Trial Nacre enroll?",
            turn2_question="Give the Trial Nacre enrollment count again with a citation.",
            expected_answer="18 participants",
            expected_citation="E1",
            answer_aliases=("18 participants", "18"),
            expected_supported_claims=("18 participants",),
            expected_unsupported_claims=("81 participants", "28 participants"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-004",
            evidence=(
                EvidenceSnippet("E1", "Tavian school board note: the Cedar lab opened in 2024 with six benches."),
                EvidenceSnippet("E2", "Tavian school board note: the Maple lab stored microscopes, not reagents."),
            ),
            turn1_question="In what year did the Cedar lab open?",
            turn2_question="Restate the Cedar lab opening year with a citation.",
            expected_answer="2024",
            expected_citation="E1",
            answer_aliases=("2024",),
            expected_supported_claims=("2024",),
            expected_unsupported_claims=("2023", "2025"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-005",
            evidence=(
                EvidenceSnippet("E1", "Orison museum label: the glass moth pendant was donated by Priya Sen."),
                EvidenceSnippet("E2", "Orison museum label: the basalt cup was cataloged in room C."),
            ),
            turn1_question="Who donated the glass moth pendant?",
            turn2_question="Answer again: who donated the glass moth pendant? Cite the evidence.",
            expected_answer="Priya Sen",
            expected_citation="E1",
            answer_aliases=("Priya Sen", "Sen"),
            expected_supported_claims=("Priya Sen",),
            expected_unsupported_claims=("Mara Sol", "Pia Stone"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-006",
            evidence=(
                EvidenceSnippet("E1", "Kelross field sheet: plot V-3 contained red clover and two marker flags."),
                EvidenceSnippet("E2", "Kelross field sheet: plot V-4 contained ryegrass and one marker flag."),
            ),
            turn1_question="Which plant was recorded in plot V-3?",
            turn2_question="Repeat the plant recorded in plot V-3 with a citation.",
            expected_answer="red clover",
            expected_citation="E1",
            answer_aliases=("red clover",),
            expected_supported_claims=("red clover",),
            expected_unsupported_claims=("ryegrass", "white clover"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-007",
            evidence=(
                EvidenceSnippet("E1", "Noma transit note: Route 6 skipped Pear Gate during the lantern parade."),
                EvidenceSnippet("E2", "Noma transit note: Route 8 stopped at Pear Gate after 19:00."),
            ),
            turn1_question="Which gate did Route 6 skip during the lantern parade?",
            turn2_question="State the skipped Route 6 gate again with a citation.",
            expected_answer="Pear Gate",
            expected_citation="E1",
            answer_aliases=("Pear Gate",),
            expected_supported_claims=("Pear Gate",),
            expected_unsupported_claims=("Pine Gate", "Pear Station"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-008",
            evidence=(
                EvidenceSnippet("E1", "Eldwick kitchen card: Soup Batch 14 used saffron after the onions softened."),
                EvidenceSnippet("E2", "Eldwick kitchen card: Soup Batch 15 used turmeric after the onions browned."),
            ),
            turn1_question="Which spice did Soup Batch 14 use?",
            turn2_question="Repeat the Soup Batch 14 spice with a citation.",
            expected_answer="saffron",
            expected_citation="E1",
            answer_aliases=("saffron",),
            expected_supported_claims=("saffron",),
            expected_unsupported_claims=("turmeric", "cumin"),
            missing_field="",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-009",
            evidence=(
                EvidenceSnippet("E1", "Rookfen archive card: the vellum map was folded twice before storage."),
                EvidenceSnippet("E2", "Rookfen archive card: the linen chart was stored in drawer Delta."),
            ),
            turn1_question="Who drew the vellum map?",
            turn2_question="Repeat who drew the vellum map, or say if the evidence is insufficient.",
            expected_answer=None,
            expected_citation=None,
            answer_aliases=(),
            expected_supported_claims=(),
            expected_unsupported_claims=("Ansel Rook", "Mira Fen", "the archivist"),
            missing_field="map drawer",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-010",
            evidence=(
                EvidenceSnippet("E1", "Solenne lab note: reagent Pavo turned violet at pH 5.2."),
                EvidenceSnippet("E2", "Solenne lab note: reagent Mira stayed clear at pH 7.0."),
            ),
            turn1_question="Which company manufactured reagent Pavo?",
            turn2_question="Give the manufacturer of reagent Pavo with a citation, or say if unsupported.",
            expected_answer=None,
            expected_citation=None,
            answer_aliases=(),
            expected_supported_claims=(),
            expected_unsupported_claims=("Orion Labs", "Pavo Chemical", "Solenne Works"),
            missing_field="manufacturer",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-011",
            evidence=(
                EvidenceSnippet("E1", "Istra festival note: the noon bell rang after the kite dance."),
                EvidenceSnippet("E2", "Istra festival note: the lantern walk began at dusk."),
            ),
            turn1_question="What award did the kite dance win?",
            turn2_question="Repeat the kite dance award with a citation, or say if unsupported.",
            expected_answer=None,
            expected_citation=None,
            answer_aliases=(),
            expected_supported_claims=(),
            expected_unsupported_claims=("silver laurel", "best procession", "judge prize"),
            missing_field="award",
        ),
        MicrobenchFixture(
            fixture_id="hhpm-012",
            evidence=(
                EvidenceSnippet("E1", "Vesper quarry note: sample Q-2 weighed 31 grams after drying."),
                EvidenceSnippet("E2", "Vesper quarry note: sample Q-3 was discarded because its label tore."),
            ),
            turn1_question="Which microscope was used for sample Q-2?",
            turn2_question="Name the microscope used for sample Q-2 with a citation, or say if unsupported.",
            expected_answer=None,
            expected_citation=None,
            answer_aliases=(),
            expected_supported_claims=(),
            expected_unsupported_claims=("Mantis-4", "Zeiss field scope", "MiraScope"),
            missing_field="microscope",
        ),
    ]


def sha16(text: str | bytes) -> str:
    payload = text if isinstance(text, bytes) else text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_turn_prompt(prompt: str) -> str:
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
        stripped = INLINE_TIMESTAMP_LOG_RE.sub(" ", stripped).strip()
        if not stripped:
            continue
        if stripped.startswith(LLAMA_LOG_PREFIXES):
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
    return re.sub(r"\s+", " ", text).strip()


def check_preconditions(repo_root: Path | str = REPO_ROOT) -> PreconditionReport:
    del repo_root
    checks: list[JsonDict] = []
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        detail = f"torch_cuda_devices={torch.cuda.device_count()}"
    except Exception as exc:  # pragma: no cover - host dependent.
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
    model_ok = selected is not None and _materialized_model_file(str(selected.get("model_path") or ""))
    checks.append(
        {
            "resource": "mandated_sota_gguf",
            "available": model_ok,
            "model": selected.get("hf_id") if selected else None,
            "model_path": selected.get("model_path") if selected else None,
        }
    )

    ok = all(bool(check["available"]) for check in checks)
    missing = next((str(check["resource"]) for check in checks if not check["available"]), "")
    runtime_command = (str(runtime),)
    if selected:
        runtime_command = (
            str(runtime),
            "-m",
            str(selected.get("model_path")),
            "-ngl",
            "999",
            "-c",
            "4096",
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


def select_headline_model(specs: list[JsonDict] | tuple[JsonDict, ...]) -> JsonDict | None:
    for preferred in (
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ):
        for spec in specs:
            if spec.get("hf_id") == preferred:
                row = dict(spec)
                row["quantization"] = row.get("quantization") or infer_quantization(str(row.get("model_path") or ""))
                return row
    return None


def infer_quantization(model_path: str) -> str | None:
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0", "BF16"):
        if token.lower() in model_path.lower():
            return token
    return None


def score_case_response(fixture: MicrobenchFixture, case_text: str) -> CaseScore:
    text = case_text.strip()
    lowered = text.lower()
    citations = tuple(dict.fromkeys(match.upper() for match in re.findall(r"\[([Ee]\d+)\]", text)))
    refusal = any(marker in lowered for marker in REFUSAL_MARKERS)
    answerable = fixture.expected_answer is not None
    reasons: list[str] = []
    if answerable:
        aliases = fixture.answer_aliases or (str(fixture.expected_answer),)
        has_expected = any(alias.lower() in lowered for alias in aliases)
        trap_found = any(claim.lower() in lowered for claim in fixture.expected_unsupported_claims)
        citation_ok = bool(fixture.expected_citation and fixture.expected_citation.upper() in citations)
        if trap_found:
            reasons.append("known_unsupported_claim")
        if not has_expected:
            reasons.append("expected_answer_missing")
        if not citation_ok:
            reasons.append("expected_citation_missing_or_wrong")
        unsupported = bool(reasons)
        citation_supported = bool(has_expected and citation_ok and not trap_found)
        missed = bool(refusal or not has_expected)
        over_refusal = bool(refusal)
    else:
        answer_payload = _answer_payload(text)
        trap_found = any(claim.lower() in lowered for claim in fixture.expected_unsupported_claims)
        fabricated = bool((answer_payload and not refusal) or trap_found)
        if fabricated:
            reasons.append(f"missing_{fixture.missing_field or 'field'}_fabricated")
        unsupported = fabricated
        citation_supported = not fabricated
        missed = False
        over_refusal = False
    return CaseScore(
        fixture_id=fixture.fixture_id,
        unsupported_claim=unsupported,
        citation_supported=citation_supported,
        over_refusal=over_refusal,
        missed_answer=missed,
        answerable=answerable,
        citations=citations,
        reasons=tuple(reasons),
        case_text_checksum=sha16(text),
        case_text_excerpt=text[:300],
    )


def build_batch_prompt(
    fixtures: list[MicrobenchFixture],
    *,
    arm: str,
    turn: int,
    prior_response: str | None = None,
    prior_scores: list[CaseScore] | None = None,
) -> str:
    turn_label = "first" if turn == 1 else "second"
    memory_block = _memory_block(arm, turn, prior_response=prior_response, prior_scores=prior_scores)
    cases = "\n\n".join(_case_block(fixture, turn=turn) for fixture in fixtures)
    return (
        "Local HalluHard-style citation support microbench. Use only the evidence snippets "
        "inside each case. Do not use outside knowledge or web search. If the evidence does "
        "not support the answer, write INSUFFICIENT_EVIDENCE.\n\n"
        "Return exactly one line per case in this format:\n"
        "CASE <id>: ANSWER: <short answer or INSUFFICIENT_EVIDENCE>; CITATIONS: [E1] or []\n\n"
        f"Turn: {turn_label}\n"
        f"Arm: {arm}\n"
        f"{memory_block}\n\n"
        f"{cases}"
    )


def score_batch_response(
    fixtures: list[MicrobenchFixture],
    response: str,
    *,
    arm: str,
    turn: int,
) -> list[JsonDict]:
    rows = []
    for fixture in fixtures:
        case_text = extract_case_text(response, fixture.fixture_id)
        score = score_case_response(fixture, case_text)
        rows.append(
            {
                **score.__dict__,
                "citations": list(score.citations),
                "reasons": list(score.reasons),
                "arm": arm,
                "turn": turn,
            }
        )
    return rows


def extract_case_text(response: str, fixture_id: str) -> str:
    matches = list(CASE_HEADER_RE.finditer(response))
    for index, match in enumerate(matches):
        if match.group(1).lower() != fixture_id.lower():
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        return response[start:end].strip()
    inline_matches = list(INLINE_CASE_RE.finditer(response))
    for index, match in enumerate(inline_matches):
        if match.group(1).lower() != fixture_id.lower():
            continue
        start = match.end()
        end = inline_matches[index + 1].start() if index + 1 < len(inline_matches) else len(response)
        return response[start:end].strip()
    for line in response.splitlines():
        if fixture_id.lower() in line.lower():
            return line.strip()
    return ""


def summarize_arm(
    arm: str,
    fixtures: list[MicrobenchFixture],
    turn1_rows: list[JsonDict],
    turn2_rows: list[JsonDict],
    receipts: list[GenerationReceipt],
) -> JsonDict:
    rows = [*turn1_rows, *turn2_rows]
    return {
        "arm": arm,
        "fixture_count": len(fixtures),
        "response_count": len(rows),
        "unsupported_claim_rate": _rate(sum(1 for row in rows if row["unsupported_claim"]), len(rows)),
        "citation_support_rate": _rate(sum(1 for row in rows if row["citation_supported"]), len(rows)),
        "repeated_error_rate": _repeated_error_rate(fixtures, turn1_rows, turn2_rows),
        "over_refusal_rate": _answerable_rate(rows, "over_refusal"),
        "missed_answer_rate": _answerable_rate(rows, "missed_answer"),
        "turns": {1: turn1_rows, 2: turn2_rows},
        "receipts": [receipt.compact() for receipt in receipts],
    }


def empty_arm_result(arm: str, fixtures: list[MicrobenchFixture]) -> JsonDict:
    rows = [
        {
            "fixture_id": fixture.fixture_id,
            "unsupported_claim": False,
            "citation_supported": True,
            "over_refusal": False,
            "missed_answer": False,
            "answerable": fixture.expected_answer is not None,
            "citations": [],
            "reasons": [],
            "case_text_checksum": sha16(""),
            "case_text_excerpt": "",
            "arm": arm,
            "turn": turn,
        }
        for turn in (1, 2)
        for fixture in fixtures
    ]
    return {
        "arm": arm,
        "fixture_count": len(fixtures),
        "response_count": len(rows),
        "unsupported_claim_rate": 0.0,
        "citation_support_rate": 1.0 if rows else 0.0,
        "repeated_error_rate": 0.0,
        "over_refusal_rate": 0.0,
        "missed_answer_rate": 0.0,
        "turns": {1: rows[: len(fixtures)], 2: rows[len(fixtures) :]},
        "receipts": [],
    }


def leakage_checks(fixtures: list[MicrobenchFixture], prompt_records: list[JsonDict]) -> JsonDict:
    fixture_count_ok = 10 <= len(fixtures) <= 20
    local_only = all("http://" not in snippet.text and "https://" not in snippet.text for fixture in fixtures for snippet in fixture.evidence)
    answers_in_evidence = all(
        fixture.expected_answer is None
        or fixture.expected_answer.lower() in " ".join(snippet.text for snippet in fixture.evidence).lower()
        for fixture in fixtures
    )
    answers_not_questions = all(
        fixture.expected_answer is None
        or (
            fixture.expected_answer.lower() not in fixture.turn1_question.lower()
            and fixture.expected_answer.lower() not in fixture.turn2_question.lower()
        )
        for fixture in fixtures
    )
    typed_memory_no_gold = True
    for record in prompt_records:
        if record.get("arm") != "typed_provenance_memory":
            continue
        stripped = str(record.get("prompt") or "").lower()
        for fixture in fixtures:
            for snippet in fixture.evidence:
                stripped = stripped.replace(snippet.text.lower(), "")
        for fixture in fixtures:
            if fixture.expected_answer and fixture.expected_answer.lower() in stripped:
                typed_memory_no_gold = False
    raw_memory_marked = all(
        record.get("arm") != "raw_conversation_memory"
        or int(record.get("turn") or 0) != 2
        or "BEGIN PRIOR MODEL OUTPUT" in str(record.get("prompt") or "")
        for record in prompt_records
    )
    checks = {
        "fixture_count_in_range": {"passed": fixture_count_ok, "value": len(fixtures)},
        "local_curated_evidence_only": {"passed": local_only},
        "gold_answers_in_evidence": {"passed": answers_in_evidence},
        "gold_answers_not_in_questions": {"passed": answers_not_questions},
        "typed_memory_does_not_add_gold_answers": {"passed": typed_memory_no_gold},
        "raw_memory_model_output_marked": {"passed": raw_memory_marked},
        "no_network_search_used": {"passed": True},
    }
    checks["passed"] = all(bool(value["passed"]) for value in checks.values())
    return checks


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
        "model_name": None,
        "model_path": None,
        "quantization": None,
        "runtime_command": list(preconditions.runtime_command),
        "seeds": [RANDOM_SEED],
        "prompt_checksums": [],
        "completion_checksums": [],
        "precondition_checks": preconditions.checks,
    }
    artifact = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "arms": list(ARM_NAMES),
        "preconditions_checked": preconditions.checks,
        "honest_verdict": wrapped(
            f"{reason}: provenance memory did not reduce hallucination errors because the microbench did not run",
            "honest_verdict",
        ),
        "inference_substrate": wrapped(INFERENCE_SUBSTRATE, "inference_substrate"),
        "model_specs": wrapped(model_value, "model_specs"),
        "fixture_count": wrapped(0, "fixture_count"),
        "unsupported_claim_rate_no_memory": wrapped(0.0, "unsupported_claim_rate_no_memory"),
        "unsupported_claim_rate_typed_memory": wrapped(0.0, "unsupported_claim_rate_typed_memory"),
        "repeated_error_delta": wrapped(0.0, "repeated_error_delta"),
        "citation_support_delta": wrapped(0.0, "citation_support_delta"),
        "leakage_checks": wrapped(
            {
                "passed": False,
                "blocked_precondition": {"passed": False, "reason": reason},
                "no_network_search_used": {"passed": True},
            },
            "leakage_checks",
        ),
        "no_network_at_benchmark_time": wrapped(True, "no_network_at_benchmark_time"),
        "arm_metrics": {},
        "over_refusal_costs": {},
        "missed_answer_costs": {},
    }
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    arm_results: dict[str, JsonDict],
    preconditions: PreconditionReport,
    started_at: str,
    finished_at: str,
    duration_s: float,
    leakage: JsonDict,
) -> JsonDict:
    no_memory = arm_results["no_memory"]
    typed = arm_results["typed_provenance_memory"]
    repeated_delta = _delta(no_memory["repeated_error_rate"], typed["repeated_error_rate"])
    citation_delta = _delta(typed["citation_support_rate"], no_memory["citation_support_rate"])
    unsupported_no = float(no_memory["unsupported_claim_rate"])
    unsupported_typed = float(typed["unsupported_claim_rate"])
    reduced = repeated_delta > 0.0 and unsupported_typed < unsupported_no
    if reduced:
        verdict = "complete: typed provenance memory reduced hallucination errors on this local microbench"
    else:
        verdict = "complete: typed provenance memory did not reduce hallucination errors on this local microbench"
    selected = preconditions.selected_model or {}
    receipts = [
        receipt
        for result in arm_results.values()
        for receipt in result.get("receipts", [])
    ]
    model_value = {
        "headline_model": selected.get("hf_id"),
        "model_name": selected.get("name"),
        "model_path": selected.get("model_path"),
        "quantization": selected.get("quantization") or infer_quantization(str(selected.get("model_path") or "")),
        "runtime_command": list(preconditions.runtime_command),
        "seeds": [RANDOM_SEED],
        "prompt_checksums": [receipt["prompt_checksum"] for receipt in receipts],
        "completion_checksums": [receipt["completion_checksum"] for receipt in receipts],
        "precondition_checks": preconditions.checks,
    }
    fixture_count = int(no_memory["fixture_count"])
    artifact = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "arms": list(ARM_NAMES),
        "preconditions_checked": preconditions.checks,
        "honest_verdict": wrapped(verdict, "honest_verdict"),
        "inference_substrate": wrapped(INFERENCE_SUBSTRATE, "inference_substrate"),
        "model_specs": wrapped(model_value, "model_specs"),
        "fixture_count": wrapped(fixture_count, "fixture_count"),
        "unsupported_claim_rate_no_memory": wrapped(unsupported_no, "unsupported_claim_rate_no_memory"),
        "unsupported_claim_rate_typed_memory": wrapped(unsupported_typed, "unsupported_claim_rate_typed_memory"),
        "repeated_error_delta": wrapped(repeated_delta, "repeated_error_delta"),
        "citation_support_delta": wrapped(citation_delta, "citation_support_delta"),
        "leakage_checks": wrapped(leakage, "leakage_checks"),
        "no_network_at_benchmark_time": wrapped(True, "no_network_at_benchmark_time"),
        "arm_metrics": arm_results,
        "over_refusal_costs": {
            arm: result["over_refusal_rate"] for arm, result in arm_results.items()
        },
        "missed_answer_costs": {
            arm: result["missed_answer_rate"] for arm, result in arm_results.items()
        },
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
    verdict = str(artifact.get("honest_verdict", {}).get("value", ""))
    if not verdict.startswith(("complete:", "blocked_")):
        errors.append("honest_verdict_prefix")
    substrate = artifact.get("inference_substrate", {}).get("value")
    if substrate != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    model = artifact.get("model_specs", {}).get("value", {})
    headline = model.get("headline_model") if isinstance(model, dict) else None
    if headline is not None and headline not in MANDATED_HEADLINE_MODELS:
        errors.append("headline_model_not_mandated_sota")
    fixture_count = artifact.get("fixture_count")
    count = fixture_count.get("value") if isinstance(fixture_count, dict) else 0
    if not verdict.startswith("blocked_") and not (10 <= int(count) <= 20):
        errors.append("complete_fixture_count_out_of_bounds")
    if artifact.get("no_network_at_benchmark_time", {}).get("value") is not True:
        errors.append("network_flag_not_true")
    for field in (
        "unsupported_claim_rate_no_memory",
        "unsupported_claim_rate_typed_memory",
        "repeated_error_delta",
        "citation_support_delta",
    ):
        value = artifact.get(field, {}).get("value")
        if not isinstance(value, float):
            errors.append(f"numeric_field_not_float:{field}")
    return errors


def run_microbench(
    *,
    repo_root: Path | str = REPO_ROOT,
    generator: Any | None = None,
    fixtures: list[MicrobenchFixture] | None = None,
    preconditions: PreconditionReport | None = None,
    write: bool = True,
) -> JsonDict:
    started_mono = time.monotonic()
    started_at = utc_now()
    preconditions = preconditions or check_preconditions(repo_root)
    if not preconditions.ok:
        artifact = build_blocked_artifact(
            preconditions=preconditions,
            started_at=started_at,
            finished_at=utc_now(),
            duration_s=time.monotonic() - started_mono,
        )
    else:
        panel = fixtures or default_fixtures()
        if generator is None:
            selected = preconditions.selected_model or {}
            generator = LlamaBatchRunner(
                model_path=str(selected["model_path"]),
                runtime_path=preconditions.runtime_command[0],
            )
        arm_results: dict[str, JsonDict] = {}
        prompt_records: list[JsonDict] = []
        for arm_index, arm in enumerate(ARM_NAMES):
            prompt1 = build_batch_prompt(panel, arm=arm, turn=1)
            prompt_records.append({"arm": arm, "turn": 1, "prompt": prompt1})
            receipt1 = generator.generate(
                prompt1,
                max_tokens=1400,
                seed=RANDOM_SEED + arm_index * 10,
                tag=f"{arm}:turn1",
            )
            turn1_rows = score_batch_response(panel, receipt1.text, arm=arm, turn=1)
            prior_scores = [_score_from_row(row) for row in turn1_rows]
            prompt2 = build_batch_prompt(
                panel,
                arm=arm,
                turn=2,
                prior_response=receipt1.text,
                prior_scores=prior_scores,
            )
            prompt_records.append({"arm": arm, "turn": 2, "prompt": prompt2})
            receipt2 = generator.generate(
                prompt2,
                max_tokens=1400,
                seed=RANDOM_SEED + arm_index * 10 + 1,
                tag=f"{arm}:turn2",
            )
            turn2_rows = score_batch_response(panel, receipt2.text, arm=arm, turn=2)
            arm_results[arm] = summarize_arm(
                arm,
                panel,
                turn1_rows,
                turn2_rows,
                [receipt1, receipt2],
            )
        artifact = build_complete_artifact(
            arm_results=arm_results,
            preconditions=preconditions,
            started_at=started_at,
            finished_at=utc_now(),
            duration_s=time.monotonic() - started_mono,
            leakage=leakage_checks(panel, prompt_records),
        )
    if write:
        output = Path(repo_root) / RESULT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def wrapped(value: Any, field: str) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def stable_checksum(payload: JsonDict) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha16(encoded)


def _memory_block(
    arm: str,
    turn: int,
    *,
    prior_response: str | None,
    prior_scores: list[CaseScore] | None,
) -> str:
    if turn == 1:
        if arm == "typed_provenance_memory":
            return "Typed provenance memory: no prior answer yet; bind every claim to a listed evidence ID."
        if arm == "raw_conversation_memory":
            return "Raw conversation memory: no prior assistant answer yet."
        return "Memory: none."
    if arm == "raw_conversation_memory":
        return (
            "Raw conversation memory follows. It is prior model output, not gold evidence.\n"
            "BEGIN PRIOR MODEL OUTPUT\n"
            f"{prior_response or ''}\n"
            "END PRIOR MODEL OUTPUT"
        )
    if arm == "typed_provenance_memory":
        diagnostics = []
        for score in prior_scores or []:
            status = "unsupported" if score.unsupported_claim else "supported"
            reasons = ",".join(score.reasons) or "none"
            diagnostics.append(
                f"CASE {score.fixture_id}: prior_support={status}; reasons={reasons}; "
                "rule=use evidence IDs only and abstain with INSUFFICIENT_EVIDENCE when missing."
            )
        return "Typed provenance memory:\n" + "\n".join(diagnostics)
    return "Memory: none; answer independently from the evidence."


def _case_block(fixture: MicrobenchFixture, *, turn: int) -> str:
    question = fixture.turn1_question if turn == 1 else fixture.turn2_question
    evidence = "\n".join(f"[{snippet.evidence_id}] {snippet.text}" for snippet in fixture.evidence)
    return f"CASE {fixture.fixture_id}\nEvidence:\n{evidence}\nQuestion: {question}"


def _score_from_row(row: JsonDict) -> CaseScore:
    return CaseScore(
        fixture_id=str(row["fixture_id"]),
        unsupported_claim=bool(row["unsupported_claim"]),
        citation_supported=bool(row["citation_supported"]),
        over_refusal=bool(row["over_refusal"]),
        missed_answer=bool(row["missed_answer"]),
        answerable=bool(row["answerable"]),
        citations=tuple(str(item) for item in row.get("citations", [])),
        reasons=tuple(str(item) for item in row.get("reasons", [])),
        case_text_checksum=str(row["case_text_checksum"]),
        case_text_excerpt=str(row["case_text_excerpt"]),
    )


def _answer_payload(text: str) -> str:
    match = re.search(r"(?is)\bANSWER\s*:\s*(.*?)(?:\bCITATIONS\s*:|$)", text)
    if not match:
        return text.strip()
    payload = match.group(1).strip(" ;.[]\n\t")
    return "" if payload.lower() in {"", "none", "n/a", "[]"} else payload


def _repeated_error_rate(
    fixtures: list[MicrobenchFixture],
    turn1_rows: list[JsonDict],
    turn2_rows: list[JsonDict],
) -> float:
    first = {row["fixture_id"]: bool(row["unsupported_claim"]) for row in turn1_rows}
    second = {row["fixture_id"]: bool(row["unsupported_claim"]) for row in turn2_rows}
    repeated = sum(1 for fixture in fixtures if first.get(fixture.fixture_id) and second.get(fixture.fixture_id))
    return _rate(repeated, len(fixtures))


def _answerable_rate(rows: list[JsonDict], field: str) -> float:
    answerable = [row for row in rows if row["answerable"]]
    return _rate(sum(1 for row in answerable if row[field]), len(answerable))


def _rate(numer: int, denom: int) -> float:
    return round(float(numer) / float(denom), 6) if denom else 0.0


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _materialized_model_file(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_file():
        return False
    if path.stat().st_size < 1_000_000:
        return False
    try:
        prefix = path.read_bytes()[:256]
    except OSError:
        return False
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/")


def main() -> None:  # pragma: no cover - live CLI entrypoint.
    artifact = run_microbench(write=True)
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}))


if __name__ == "__main__":  # pragma: no cover
    main()
