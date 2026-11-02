"""REQ-VERIFY-8722: configure the versioned collector without changing its historical module."""

from importlib.util import module_from_spec, spec_from_file_location
from typing import Any

from carnot.reporting import v746_capstone as retained
from carnot.reporting import v747_capstone_inputs as e
from carnot.reporting import v747_capstone_report as report
from carnot.reporting.v747_capstone_measure import measure

# A separate namespace keeps historical consumers and source bytes unchanged.
spec = spec_from_file_location("carnot.reporting._v747_capstone_custody", retained.__file__)
assert spec is not None and spec.loader is not None
runtime: Any = module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.e, runtime.report, runtime.measure = e, report, measure
main = runtime.main
