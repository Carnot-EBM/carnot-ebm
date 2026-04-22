"""incremental_test_selector.py — Thin shim re-exporting from carnot.pipeline.

All logic lives in python/carnot/pipeline/incremental_test_selector.py which is
under coverage tracking.  This shim allows conductor_pre_flight.py to import via
a scripts/ path.insert without duplicating the implementation.

Spec: REQ-INFRA-041, SCENARIO-INFRA-050, SCENARIO-INFRA-051
"""

from __future__ import annotations

from carnot.pipeline.incremental_test_selector import (  # noqa: F401
    IncrementalTestSelector,
    _FULL_SUITE_DIFF_THRESHOLD,
    _any_rust_changed,
    _build_import_map,
    _collect_test_imports,
    _get_changed_files,
    _python_modules_changed,
)
