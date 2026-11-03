"""REQ-VERIFY-8770: opt-in leaves keep the optimized service arithmetic intact.

The native call includes its Python crossing and native residual arithmetic.
These cannot be separated by a Python timer, so the report names that limit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from . import installed_profile_8733 as base


def install(recorder: base.Recorder) -> None:
    """Pause enclosing work while each observed leaf owns the clock."""
    numeric = base.wire.numeric
    for owner, name, label in [
        (base.wire, "read_frame", "transport_read"),
        (base.wire, "write_frame", "transport_write"),
        (base.wire, "sha256_file", "configuration_digest"),
        (base.wire, "configuration", "configuration"),
        (base.wire, "decide", "protocol_validation"),
        (base.batch, "evaluate", "batch_validation"),
        (numeric, "extract", "feature_extraction"),
        (numeric, "canonical_hash", "digesting"),
        (numeric, "normalized", "text_normalization_and_digest"),
        (numeric, "scale", "numerical_scoring"),
        (numeric, "basis", "numerical_scoring"),
        (numeric, "parameters", "numerical_scoring"),
        (numeric, "predict", "numerical_scoring"),
    ]:
        original = getattr(owner, name)

        def wrapped(*args: Any, _fn: Any = original, _label: str = label, **kw: Any) -> Any:
            return recorder.call(_label, _fn, *args, **kw)

        setattr(owner, name, wrapped)
    original_batch = numeric.batch

    def score(head: Any, rows: Any, native: Any = None, **kwargs: Any) -> Any:
        if native is not None:
            original_native = native.nine_input_residual_8533

            def cross(*args: Any) -> Any:
                return recorder.call("native_crossing_and_residual", original_native, *args)

            native = SimpleNamespace(nine_input_residual_8533=cross)
        return recorder.call("numerical_scoring", original_batch, head, rows, native, **kwargs)

    numeric.batch = score
    for owner in (base.wire, base.batch, numeric):

        def dumps(*args: Any, **kw: Any) -> Any:
            return recorder.call("json_serialization", base.json.dumps, *args, **kw)

        def loads(*args: Any, **kw: Any) -> Any:
            return recorder.call("json_deserialization", base.json.loads, *args, **kw)

        owner.json = SimpleNamespace(dumps=dumps, loads=loads)


def main(argv: list[str] | None = None) -> int:
    """Restore shared bindings so importing this diagnostic cannot alter old studies."""
    original = base.install
    base.install = install
    try:
        return base.main(argv)
    finally:
        base.install = original


if __name__ == "__main__":
    raise SystemExit(main())
