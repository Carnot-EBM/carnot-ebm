"""Normalize bytes or text at one strict GGUF output boundary.

The boundary removes transport encoding only. It never repairs certificate
syntax or meaning. Callers can therefore measure transport recovery separately
from parser and verifier outcomes.

Spec: REQ-VERIFY-6755 and SCENARIO-VERIFY-6755-BOUNDARY.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Any


class OutputTextNormalizationError(ValueError):
    """Report one exact reason that output bytes cannot become lossless text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _sha256_bytes(value: bytes) -> str:
    """Hash bytes before any decode so transport evidence stays auditable."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _decode_utf8(value: bytes) -> str:
    """Decode once and reject invalid bytes instead of replacing evidence."""

    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise OutputTextNormalizationError("invalid_utf8") from error


def _starts_bytes_literal(value: str) -> bool:
    """Identify the only legacy envelope prefix that may need inspection."""

    return value.startswith(("b'", 'b"'))


def normalize_gguf_output_text(
    value: str | bytes, *, unwrap_legacy_envelope: bool = False
) -> dict[str, Any]:
    """Return one lossless text receipt for a real or legacy GGUF response.

    A JSON artifact can preserve a prior bytes object as text such as
    ``b'SAT x1=1'``. The optional legacy path accepts only one complete Python
    bytes constant. It rejects expressions and a second nested envelope.
    """

    if isinstance(value, bytes):
        source_bytes = value
        text = _decode_utf8(value)
        kind = "utf8_bytes"
    elif isinstance(value, str):
        source_bytes = value.encode("utf-8")
        text = value
        kind = "text"
        if unwrap_legacy_envelope and _starts_bytes_literal(value):
            if value.startswith(("b'''", 'b"""')):
                raise OutputTextNormalizationError("ambiguous_bytes_literal")
            try:
                expression = ast.parse(value, mode="eval")
            except (SyntaxError, ValueError) as error:
                raise OutputTextNormalizationError("ambiguous_bytes_literal") from error
            node = expression.body
            if not isinstance(node, ast.Constant) or not isinstance(node.value, bytes):
                raise OutputTextNormalizationError("ambiguous_bytes_literal")
            literal_bytes = node.value
            text = _decode_utf8(literal_bytes)
            if _starts_bytes_literal(text):
                raise OutputTextNormalizationError("nested_bytes_literal")
            if (
                text.encode("utf-8") != literal_bytes
                or ast.literal_eval(repr(literal_bytes)) != literal_bytes
            ):
                raise OutputTextNormalizationError("ambiguous_bytes_literal")
            kind = "legacy_python_bytes_literal"
    else:
        raise OutputTextNormalizationError("unsupported_output_type")

    normalized_bytes = text.encode("utf-8")
    return {
        "text": text,
        "normalization_kind": kind,
        "source_bytes_sha256": _sha256_bytes(source_bytes),
        "normalized_text_sha256": _sha256_bytes(normalized_bytes),
        "semantic_edits_performed": 0,
    }
