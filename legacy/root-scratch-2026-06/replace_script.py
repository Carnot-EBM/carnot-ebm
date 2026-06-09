import re
with open("scripts/experiment_3381_kv260_latency_benchmark.py", "r") as f:
    text = f.read()

text = text.replace("experiment_2898", "experiment_3381")
text = text.replace("EXPERIMENT_ID = 2898", "EXPERIMENT_ID = 3381")
text = text.replace("results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json", "results/experiment_3381_kv260_latency_benchmark.json")

# Add CPU baseline function
cpu_func = """
def _measure_cpu_baseline(n_spins: int = N_SPINS, n_trials: int = 10) -> float:
    import jax.numpy as jnp
    import jax.random as jrandom
    import time
    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    sampler = ParallelIsingSampler(
        n_warmup=200,
        n_samples=1,
        steps_per_sample=5,
        schedule=AnnealingSchedule(beta_init=0.5, beta_final=5.0),
        use_checkerboard=True,
    )
    key = jrandom.PRNGKey(0)
    biases = jnp.zeros(n_spins, dtype=jnp.float32)
    couplings = jnp.zeros((n_spins, n_spins), dtype=jnp.float32)

    _ = sampler.sample(key, biases, couplings)

    latencies_us: list[float] = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        _ = sampler.sample(jrandom.PRNGKey(i), biases, couplings)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    return float(sum(latencies_us) / len(latencies_us))
"""

text = text.replace("def _run(cmd: list[str], timeout: int | float) -> CommandResult:", cpu_func + "\n\ndef _run(cmd: list[str], timeout: int | float) -> CommandResult:")

# Update build_success_artifact to include latencies
success_art = """
def build_success_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    uptime: str,
    overlay_loaded: str,
    overlay_load_command: str,
    uio_devices_present: list[str],
    bitstream_sha256: str | None,
    problem_payload: dict[str, Any],
    board_payload: dict[str, Any],
    duration_s: float,
    transcript_path: Path,
    cpu_baseline_latency_us: float,
) -> dict[str, Any]:
    per_seed = _success_per_seed_results(board_payload)
    hardware_latency_us = sum(row["per_sample_wall_clock_us_median"] for row in per_seed) / len(per_seed) if per_seed else 0.0
    fpga_speedup = cpu_baseline_latency_us / hardware_latency_us if hardware_latency_us > 0 else 0.0

    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp3381-kv260-latency-benchmark",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "hardware_latency_us": hardware_latency_us,
        "cpu_baseline_latency_us": cpu_baseline_latency_us,
        "fpga_speedup": fpga_speedup,
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": uptime,
        "kv260_overlay_loaded": overlay_loaded,
        "kv260_overlay_load_command": overlay_load_command,
        "kv260_uio_devices_present": uio_devices_present,
        "bitstream_sha256": bitstream_sha256,
        "bitstream_sha256_source": "board:/lib/firmware/xilinx/carnot_ising_v4",
        "ising_problem_spec": _primary_problem_spec(problem_payload),
        "problem_payload": problem_payload,
        "per_seed_results": per_seed,
        "sample_count_sweep_results": _sample_count_sweep(board_payload),
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": _reproducibility_checksum(
            problem_payload, overlay_loaded, bitstream_sha256
        ),
        "board_transcript_path": _path_for_artifact(transcript_path),
        "board_harness_summary": {
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            "uio0_mmap_checked": board_payload.get("uio0_mmap_checked"),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
        "duration_s": duration_s,
    }
"""
text = re.sub(r"def build_success_artifact\(.*?\) -> dict\[str, Any\]:.*?(?=def _validate_success_artifact)", success_art, text, flags=re.DOTALL)

# Update run_experiment to call cpu_baseline and pass it
run_exp_replace = """
    problem_payload = build_problem_payload()
    board_payload = run_board_harness(problem_payload, transcript)

    cpu_baseline_latency_us = _measure_cpu_baseline()

    artifact = build_success_artifact(
        preconditions_checked=preconditions,
        uptime=provenance.get("uptime", ""),
        overlay_loaded=load_details.get("loaded_overlay") or "carnot_ising_v2_n64",
        overlay_load_command=OVERLAY_LOAD_COMMAND,
        uio_devices_present=provenance.get("uio_devices", []),
        bitstream_sha256=provenance.get("bitstream_sha256"),
        problem_payload=problem_payload,
        board_payload=board_payload,
        duration_s=time.perf_counter() - started,
        transcript_path=TRANSCRIPT_PATH,
        cpu_baseline_latency_us=cpu_baseline_latency_us,
    )
"""
text = re.sub(r"    problem_payload = build_problem_payload\(\).*?transcript_path=TRANSCRIPT_PATH,\n    \)", run_exp_replace, text, flags=re.DOTALL)

# Also block artifact
blocked_art = """
def build_blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    transcript_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp3381-kv260-latency-benchmark",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "hardware_latency_us": None,
        "cpu_baseline_latency_us": None,
        "fpga_speedup": None,
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": "",
        "kv260_overlay_loaded": "",
        "kv260_overlay_load_command": OVERLAY_LOAD_COMMAND,
        "kv260_uio_devices_present": [],
        "bitstream_sha256": None,
        "ising_problem_spec": {
            "n_spins": N_SPINS,
            "j_matrix_sha256": "",
            "h_vector_sha256": "",
            "random_seed": RANDOM_SEEDS[0],
        },
        "per_seed_results": [],
        "sample_count_sweep_results": [],
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "board_transcript_path": _path_for_artifact(transcript_path),
        "duration_s": duration_s,
    }
"""
text = re.sub(r"def build_blocked_artifact\(.*?\) -> dict\[str, Any\]:.*?(?=def build_success_artifact)", blocked_art, text, flags=re.DOTALL)

with open("scripts/experiment_3381_kv260_latency_benchmark.py", "w") as f:
    f.write(text)

