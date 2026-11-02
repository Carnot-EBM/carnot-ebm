"""REQ-VERIFY-8263: intact inputs and one focal record have separate truth limits.

Exact vocabulary counts decide admission and deletion matching. An address only
binds a predicted relation to surviving source bytes; it never proves that claim.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import re
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify.evidence_view_kernel_8249 import construct_view
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
INSTRUCTION = "Emit only the requested original focal answer index|E/C/B|p_unsupported to two decimals|[zero to two distinct source-view indices]. Relations and citations are predictions, not truth certificates."
ENCODING = "SHA256(UTF8(JSON([original_source_sha256, seed], ensure_ascii=False, separators=(',',':')))); unsigned big-endian integer modulo 8; seeds 101/102/103"


def request(
    view: Json,
    focal: int,
    count: Callable[[str], int],
    *,
    context_tokens: int = 6000,
    output_tokens: int = 64,
) -> Json:
    """Reserve the entire bounded output before admitting the intact raw prompt.

    This is an explicit raw-completion endpoint, so no uncounted chat wrapper
    enters context. Original answer offsets are retained even beyond index eight.
    """
    source, answer = (bytes.fromhex(view[k]) for k in ["source_bytes", "answer_bytes"])
    segments, answers = transport.partition(source), transport.partition(answer)
    if type(focal) is not int or not any(
        s["sentence_index"] == focal and s["complete"] for s in answers
    ):
        raise ValueError("focal_index")
    schema = transport.grammar([focal], len(segments))
    if not segments:
        schema = (
            "\n".join(
                'refs ::= "[]"' if line.startswith("refs ::=") else line
                for line in schema.splitlines()
                if not line.startswith("source ::=")
            )
            + "\n"
        )
    payload = dict(
        instruction=INSTRUCTION,
        source=source.decode(),
        answer=answer.decode(),
        source_segments=segments,
        answer_offsets=answers,
        answer_sentence_indices=[focal],
    )
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    bound = transport.output_budget([focal], max(1, len(segments)), count)
    tokens = count(prompt)
    if (
        type(tokens) is not int
        or tokens < 0
        or not 1 <= output_tokens <= 64
        or bound["maximum_encoded_output_tokens"] > output_tokens
        or tokens + output_tokens > context_tokens
    ):
        raise ValueError("token_budget")
    return dict(
        prompt=prompt,
        grammar=schema,
        sentence_indices=[focal],
        source_count=len(segments),
        input_tokens=tokens,
        max_tokens=output_tokens,
        context_tokens=context_tokens,
        output_bound=bound,
        temperature=0,
        seed=7138250,
        endpoint="/completion",
        prompt_format="raw_completion",
        template_sha256=canonical_hash(INSTRUCTION),
        response_schema_sha256=canonical_hash(schema),
    )


def parse(view: Json, transcript: str, request_hash: str) -> Json:
    """Require the original focal position and translate surviving citations only."""
    req = view["request"]
    if canonical_hash(req) != request_hash:
        raise ValueError("request_hash")
    result = transport.parse(transcript, req["sentence_indices"], req["source_count"])
    for row in result["rows"]:
        if len(row["source_indices"]) != len(set(row["source_indices"])):
            result["status"] = "escalated"
        row["original_source_indices"] = [
            m["original_index"]
            for i in row["source_indices"]
            for m in view["sentence_map"]
            if m["view_index"] == i
        ]
    return dict(result)


def views(row: Json, cached: list[Json], count: Callable[[str], int]) -> Json:
    """Apply the frozen maximum-Jaccard selector without the inherited splitter."""
    source, answer = (bytes.fromhex(row[k]) for k in ["source_bytes", "answer_bytes"])
    sources = [s for s in transport.partition(source) if s["complete"]]
    answers = {s["sentence_index"]: s for s in transport.partition(answer) if s["complete"]}
    usable = [
        r
        for r in cached
        if r["sentence_index"] in answers
        and isinstance(r.get("p_unsupported"), (int, float))
        and math.isfinite(r["p_unsupported"])
        and 0 <= r["p_unsupported"] <= 1
    ]
    if not usable or len(sources) < 2:
        return dict(
            status="unavailable",
            reason="missing_focal_cached_probability"
            if not usable
            else "no_second_complete_source_sentence",
            views={},
        )
    focal = min(usable, key=lambda r: (-r["p_unsupported"], r["sentence_index"]))
    sentence = answers[focal["sentence_index"]]
    words: Callable[[str], set[str]] = lambda text: set(re.findall(r"\w+", text.casefold()))
    target = words(answer[sentence["byte_start"] : sentence["byte_end"]].decode())
    diagnostics = []
    for s in sources:
        text = source[s["byte_start"] : s["byte_end"]].decode()
        tokens = words(text)
        overlap = len(tokens & target) / len(tokens | target) if tokens | target else 0.0
        diagnostics.append(dict(s, lexical_overlap=overlap, embedded_tokens=count(text)))
    selected = min(diagnostics, key=lambda s: (-s["lexical_overlap"], s["sentence_index"]))
    control = min(
        (s for s in diagnostics if s["sentence_index"] != selected["sentence_index"]),
        key=lambda s: (
            abs(s["embedded_tokens"] - selected["embedded_tokens"]),
            s["sentence_index"],
        ),
    )
    mapped = {}
    for name, removed in [
        ("original", []),
        ("selected_evidence_deleted", [selected["sentence_index"]]),
        ("nonselected_deletion", [control["sentence_index"]]),
    ]:
        view = construct_view(row, removed)
        view["request"] = request(view, focal["sentence_index"], count)
        mapped[name] = view
    return dict(
        status="completed",
        focal_index=focal["sentence_index"],
        selected_sentence_index=selected["sentence_index"],
        control_sentence_index=control["sentence_index"],
        cached_relation=focal.get("relation", "B"),
        diagnostics=diagnostics,
        views=mapped,
    )
