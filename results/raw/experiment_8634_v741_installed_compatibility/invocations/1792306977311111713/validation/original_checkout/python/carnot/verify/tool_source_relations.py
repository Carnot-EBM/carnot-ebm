"""Bind narrow source tuples without treating lexical overlap as proof.

The reader handles only visible records and explicit answer forms. This is an
exact fixture protocol, not a learned verifier or whole-answer authority.
Spec: REQ-VERIFY-7672 and SCENARIO-VERIFY-7672-TUPLE/QUALIFIER/REPLAY.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from carnot.verify.tool_source_atoms import parse_atoms


FRAME = re.compile(r"\bat (?P<function>[\w.$<>-]+) \((?P<path>[^\s():]+):(?P<line>\d+):\d+\)")
QUOTE = re.compile(r"(?P<mark>['\"])(?P<value>[^'\"]+)\1")
STACK_CLAIM = re.compile(r"`(?P<function>[\w.$<>-]+)` at `(?P<path>[^`\s:]+):(?P<line>\d+)`")
GREP_CLAIM = re.compile(
    r"`(?P<path>[^`\s:]+):(?P<line>\d+)` contains (?P<mark>['\"])(?P<quote>[^'\"]+)\3"
)
AST_CLAIM = re.compile(
    r"`(?P<name>\w+)` is (?P<negative>not )?defined in `(?P<scope>\w+)` at line (?P<line>\d+)"
)
QUALIFIER = re.compile(
    r"\b(?:because|caus(?:e|ed)|might|may|could|would|always|never|every|all|and|or|only|if)\b",
    re.I,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _byte(value: str, position: int) -> int:
    return len(value[:position].encode("utf-8"))


def index_relations(source: str) -> list[dict[str, Any]]:
    """Retain one tuple per visible record with its exact evidence bytes."""
    if source.count("```") < 2:
        return []
    records: list[dict[str, Any]] = []
    for atom in parse_atoms(source):
        base = {
            "dialect": atom["dialect"],
            "polarity": "positive",
            "complete": atom["complete"],
            "evidence_span": [atom["byte_start"], atom["byte_end"]],
            "source_id": atom["source_id"],
        }
        if atom["dialect"] == "stack":
            frame = FRAME.search(atom["text"])
            if frame:
                records.append(
                    {
                        **base,
                        "kind": "stack_frame",
                        "arguments": {
                            "function": frame["function"],
                            "path": frame["path"],
                            "line": int(frame["line"]),
                        },
                    }
                )
        elif atom["dialect"] == "grep":
            quote = QUOTE.search(atom["text"])
            if quote:
                records.append(
                    {
                        **base,
                        "kind": "grep_quote",
                        "arguments": {
                            "path": atom["source_id"],
                            "line": atom["line"],
                            "quote": quote["value"],
                        },
                    }
                )
        elif atom["dialect"] in {"plain", "numbered"}:
            for name in atom["definitions"]:
                parts = name.split(".")
                records.append(
                    {
                        **base,
                        "kind": "definition_in_scope",
                        "arguments": {
                            "name": parts[-1],
                            "scope": ".".join(parts[:-1]),
                            "line": atom["line"],
                        },
                    }
                )
    return records


def _claims(answer: str) -> list[dict[str, Any]]:
    """Read only explicit forms and keep their original byte ranges."""
    found = []
    for kind, pattern in (
        ("stack_frame", STACK_CLAIM),
        ("grep_quote", GREP_CLAIM),
        ("definition_in_scope", AST_CLAIM),
    ):
        for match in pattern.finditer(answer):
            arguments = {
                key: value
                for key, value in match.groupdict().items()
                if key in {"function", "path", "line", "quote", "name", "scope"}
            }
            if "line" in arguments:
                arguments["line"] = int(arguments["line"])
            found.append(
                {
                    "kind": kind,
                    "arguments": arguments,
                    "polarity": "negative"
                    if kind == "definition_in_scope" and match.group("negative")
                    else "positive",
                    "answer_span": [_byte(answer, match.start()), _byte(answer, match.end())],
                    "text": match.group(),
                }
            )
    return sorted(found, key=lambda row: row["answer_span"])


def verify_relations(source: str, answer: str) -> dict[str, Any]:
    """Decide each explicit proposition; leave all extra meaning unresolved."""
    records = index_relations(source)
    claims = _claims(answer)
    answer_bytes = answer.encode("utf-8")
    covered = bytearray(answer_bytes)
    findings = []
    for claim in claims:
        start, end = claim["answer_span"]
        covered[start:end] = b" " * (end - start)
        kind = claim["kind"]
        args = claim["arguments"]
        relevant = [row for row in records if row["kind"] == kind]
        if kind in {"stack_frame", "grep_quote"}:
            relevant = [
                row
                for row in relevant
                if row["arguments"]["path"] == args["path"]
                and row["arguments"]["line"] == args["line"]
            ]
        else:
            relevant = [row for row in relevant if row["arguments"]["line"] == args["line"]]
        ambiguous = len({tuple(sorted(row["arguments"].items())) for row in relevant}) > 1
        exact = [row for row in relevant if row["arguments"] == args]
        if ambiguous:
            decision = "unknown"
            reason = "cross_record_ambiguity"
            evidence = None
        elif exact:
            decision = "contradicted" if claim["polarity"] == "negative" else "supported"
            reason = "exact_bound_tuple"
            evidence = exact[0]["evidence_span"]
        elif (
            kind == "definition_in_scope" and relevant and all(row["complete"] for row in relevant)
        ):
            decision = "supported" if claim["polarity"] == "negative" else "contradicted"
            reason = "complete_local_definition_line"
            evidence = relevant[0]["evidence_span"]
        else:
            decision = "unknown"
            reason = "partial_or_absent_evidence"
            evidence = None
        findings.append({**claim, "status": decision, "reason": reason, "evidence_span": evidence})
    remainder = bytes(covered).decode("utf-8").strip(" .,!?:;\n\t") if claims else answer
    if remainder or QUALIFIER.search(answer):
        for finding in findings:
            if finding["status"] == "supported":
                finding["status"] = "unknown"
                finding["reason"] = "unhandled_qualifier_or_clause"
    if not findings:
        findings = [
            {
                "kind": "unhandled_clause",
                "arguments": {},
                "polarity": "unknown",
                "answer_span": [0, len(answer_bytes)],
                "text": answer,
                "status": "unknown",
                "reason": "unhandled_answer_form",
                "evidence_span": None,
            }
        ]
    decisions = {finding["status"] for finding in findings}
    status = (
        "contradicted"
        if "contradicted" in decisions
        else "supported"
        if decisions == {"supported"} and not remainder
        else "unknown"
    )
    return {
        "source_sha256": _sha(source),
        "answer_sha256": _sha(answer),
        "relations": records,
        "findings": findings,
        "status": status,
        "checked_spans": [row["answer_span"] for row in findings if row["status"] != "unknown"],
        "unknown_spans": [row["answer_span"] for row in findings if row["status"] == "unknown"],
        "residual_unverified_text": remainder,
        "whole_answer_certified": False,
    }


def replay_relations(source: str, answer: str, saved: dict[str, Any]) -> None:
    """Reject altered bytes, offsets, relations, or decisions after reload."""
    if saved != verify_relations(source, answer):
        raise ValueError("relation_replay_mismatch")
