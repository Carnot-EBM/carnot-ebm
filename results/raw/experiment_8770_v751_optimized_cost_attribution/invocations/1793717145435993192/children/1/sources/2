"""REQ-VERIFY-8770: reuse qualified freezing, child custody and terminal checks."""

from carnot.reporting import feature_reuse_cost_runner_8757 as previous
from carnot.reporting import optimized_cost_attribution_8770 as e
from carnot.reporting import optimized_cost_attribution_measure_8770 as m
from carnot.reporting.v748_collection_contract import namespace

runtime = namespace(previous, "optimized_cost_8770")
runtime.e, runtime.m = e, m
runtime.shared = namespace(previous.shared, "optimized_validation_8770")
runtime.SUITES = e.TESTS + [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_installed_batch_8702.py",
    "tests/python/test_single_extraction_8743.py",
]
main, replay = runtime.main, runtime.replay
