"""REQ-VERIFY-8179: carry evidence addresses without rewriting answer text.

Short records remove copied quotes from the output budget. The original bytes
remain in every request, so syntax qualification cannot silently lose qualifiers.
An address is a prediction about evidence location, never an entailment label.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import re
from typing import Any

from carnot.verify.sentence_evidence_8166 import partition

Json = dict[str, Any]


def grammar(indices: list[int], source_count: int) -> str:
    """Fix each requested sentence position and forbid unbounded output whitespace.

    Literal source alternatives prevent out-of-range addresses during decoding.
    The independent parser still checks cached or unconstrained transcripts.
    """
    rules = ["root ::= " + ' "\\n" '.join(f"r{i}" for i in indices)]
    rules += [f'r{i} ::= "{i}|" relation "|" probability "|" refs' for i in indices]
    rules += [
        'relation ::= "E" | "C" | "B"',
        'probability ::= "0." [0-9] [0-9] | "1.00"',
        'refs ::= "[]" | "[" source "]" | "[" source "," source "]"',
        "source ::= " + " | ".join(json.dumps(str(i)) for i in range(source_count)),
    ]
    return "\n".join(rules) + "\n"


def requests(row: Json, count: Callable[[str], int] | None = None) -> Json:
    """Retain full inputs and complete sentence groups; refuse excess capacity.

    Byte counting is useful for private fixtures only. Live protocol preflight
    supplies the exact embedded tokenizer and records that identity separately.
    """
    source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
    segments, sentences = partition(source), partition(answer)
    reason = (
        "missing_source"
        if not source.strip()
        else "missing_sentences"
        if not sentences
        else "incomplete_sentence"
        if not all(s["complete"] for s in sentences)
        else "sentence_capacity"
        if len(sentences) > 16
        else None
    )
    calls = []
    if reason is None:
        for start in range(0, len(sentences), 8):
            indices = [s["sentence_index"] for s in sentences[start : start + 8]]
            payload = json.dumps(
                dict(
                    instruction="For each requested answer sentence emit index|E/C/B|p_unsupported with two decimals|[zero to two source segment indices]. One record per line in input order; no other text. Relations and addresses are predictions.",
                    source=source.decode(),
                    answer=answer.decode(),
                    source_segments=[
                        dict(
                            index=s["sentence_index"],
                            byte_start=s["byte_start"],
                            byte_end=s["byte_end"],
                        )
                        for s in segments
                    ],
                    answer_sentence_indices=indices,
                    answer_offsets=[
                        dict(
                            index=s["sentence_index"],
                            byte_start=s["byte_start"],
                            byte_end=s["byte_end"],
                        )
                        for s in sentences
                    ],
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tokens = count(payload) if count else len(payload.encode())
            if type(tokens) is not int or tokens < 0 or tokens > 6000:
                reason = "input_token_limit"
            calls.append(
                dict(
                    request_index=start // 8,
                    payload=payload,
                    sentence_indices=indices,
                    input_tokens=tokens,
                    token_count_measured=count is not None,
                    maximum_output_tokens=256,
                    grammar=grammar(indices, len(segments)),
                )
            )
    return dict(
        source_segments=segments,
        sentences=sentences,
        requests=[] if reason else calls,
        status="escalated" if reason else "completed",
        exclusion_reason=reason,
        human_target=None,
    )


def parse(transcript: str, indices: list[int], source_count: int) -> Json:
    """Check complete ordered records independently of any claimed semantic truth.

    A partial group escalates the source. Empty evidence lists are valid transport
    and cannot be interpreted as a gold baseless judgment.
    """
    rows = []
    valid = bool(indices) and len(transcript.split("\n")) == len(indices)
    for line, index in zip(transcript.split("\n"), indices):
        match = re.fullmatch(
            r"(0|[1-9][0-9]*)\|([ECB])\|(0\.[0-9]{2}|1\.00)\|\[((?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))?)?\]",
            line,
        )
        if match is None:
            valid = False
            continue
        refs = [int(x) for x in match[4].split(",")] if match[4] else []
        valid = valid and int(match[1]) == index and all(0 <= x < source_count for x in refs)
        rows.append(
            dict(
                sentence_index=int(match[1]),
                relation=match[2],
                p_unsupported=float(match[3]),
                source_indices=refs,
                human_target=None,
            )
        )
    return dict(rows=rows, status="completed" if valid else "escalated", semantic_gold=False)


def output_budget(indices: list[int], source_count: int, count: Callable[[str], int]) -> Json:
    """Enumerate longest records and bind a safe bound for every grammar path.

    Every grammar character is ASCII and every emitted token consumes at least
    one character. Thus maximal byte length plus EOS bounds even noncanonical
    token sequences. Canonical tokenizer counts are measured for all101 allowed
    probabilities and all three relations at the longest reference encoding.
    Mixed relations or probabilities cannot exceed this byte-length bound.
    """
    encodings = [
        "\n".join(
            f"{i}|{relation}|{p / 100:.2f}|[{source_count - 1},{source_count - 1}]" for i in indices
        )
        for relation in "ECB"
        for p in range(101)
    ]
    measured = [count(text) for text in encodings]
    return dict(
        maximum_encoded_output_tokens=max(len(text.encode()) for text in encodings) + 1,
        maximum_measured_tokens=max(measured),
        enumerated_count=len(encodings),
        worst_encoding=encodings[measured.index(max(measured))],
        bound_includes_eos=True,
        proof="ASCII grammar; nonempty decoded token consumes >=1 byte; longest references and every probability/relation enumerated",
    )
