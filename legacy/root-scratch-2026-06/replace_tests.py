import re
with open("tests/python/test_experiment_3381_kv260_latency_benchmark.py", "r") as f:
    text = f.read()

text = text.replace("experiment_2898", "experiment_3381_kv260_latency_benchmark")
text = text.replace("Exp 2898", "Exp 3381")

# Update test_blocked_artifact_has_required_hardware_smoke_fields
text = text.replace('assert "speedup" not in {key.lower() for key in _all_keys(artifact)}', 'assert "fpga_speedup" in artifact')

text = text.replace("exp.build_success_artifact(\n            preconditions_checked=_fake_preconditions(),", "exp.build_success_artifact(\n            preconditions_checked=_fake_preconditions(),\n            cpu_baseline_latency_us=5000.0,")

# Update run_experiment test to mock _measure_cpu_baseline
text = text.replace('monkeypatch.setattr(exp, "run_board_harness", lambda payload, transcript: _fake_board_payload())', 'monkeypatch.setattr(exp, "run_board_harness", lambda payload, transcript: _fake_board_payload())\n        monkeypatch.setattr(exp, "_measure_cpu_baseline", lambda: 5000.0)')

with open("tests/python/test_experiment_3381_kv260_latency_benchmark.py", "w") as f:
    f.write(text)
