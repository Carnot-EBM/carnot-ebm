"""REQ-REPORT-8527: give absent gate evidence the same explicit schema as present evidence."""

from typing import Any

TASK = "exp8527-source-consumer-qualification"
DIAGNOSTIC_FIELDS = {
    "check",
    "upstream",
    "path",
    "sha256",
    "field",
    "operator",
    "expected",
    "observed",
}


def diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate legacy field names without changing the observation or its threshold.

    Older authority readers use artifact_field and upstream_id. Missing upstream
    evidence still needs those names translated, because diagnostic consumers
    must be able to identify the absent producer without raising a KeyError.
    """
    return [
        dict(
            row,
            check=row.get("check", row.get("artifact_field", row.get("field"))),
            upstream=row.get("upstream", row.get("upstream_id", TASK)),
            path=row.get("path", row.get("artifact_path")),
            sha256=row.get("sha256", row.get("artifact_hash")),
            field=row.get("field", row.get("artifact_field")),
            operator=row.get("operator", row.get("op", "==")),
            expected=row.get("expected"),
            observed=row.get("observed"),
        )
        for row in rows
    ]
