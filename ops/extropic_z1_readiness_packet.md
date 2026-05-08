# Extropic Z1 Access-Readiness Packet

Spec refs: REQ-SAMPLE-055, SCENARIO-SAMPLE-083.

## Status

- status: access_readiness_packet_only
- milestone: 2026.04.118
- run_date: 20260508
- no_hardware_execution_claim: true

## Benchmark Case List

| case_id | n_spins | topology | seed/schedule manifest |
|---|---:|---|---|
| n256_schedule_stress:low_beta_short_warmup | 256 | signed_ring_chord | seeds=[20260508,20360508]; beta=0.9, n_warmup=384, steps_per_sample=3, use_checkerboard=True |
| n256_schedule_stress:baseline_n128_style | 256 | signed_ring_chord | seeds=[20260525,20360525]; beta=1.0, n_warmup=512, steps_per_sample=4, use_checkerboard=True |
| n256_schedule_stress:high_beta_longer_thinning | 256 | signed_ring_chord | seeds=[20260539,20360539]; beta=1.1, n_warmup=640, steps_per_sample=6, use_checkerboard=True |
| n64_diverse_topology:complete | 64 | complete | seeds=[20260508,20260509,20260510,20260511,20260512]; beta=1.05, n_warmup=512, steps_per_sample=4, use_checkerboard=True |
| n64_diverse_topology:sparse_random | 64 | sparse_random | seeds=[20260508,20260509,20260510,20260511,20260512]; beta=1.05, n_warmup=512, steps_per_sample=4, use_checkerboard=True |
| n64_diverse_topology:lattice | 64 | lattice | seeds=[20260508,20260509,20260510,20260511,20260512]; beta=1.05, n_warmup=512, steps_per_sample=4, use_checkerboard=True |
| n64_diverse_topology:scale_free | 64 | scale_free | seeds=[20260508,20260509,20260510,20260511,20260512]; beta=1.05, n_warmup=512, steps_per_sample=4, use_checkerboard=True |

## Required Device Metadata

- transcript_schema_version
- run_date
- authenticated_access_proof
- access_grant_reference
- provider_or_lab_operator
- device_family
- device_identifier
- device_firmware_or_runtime
- sdk_package_name
- sdk_version
- thrml_version
- device_discovery_command
- execution_timestamp_utc
- host_identifier
- benchmark_case_id
- schedule_id
- topology
- n_spins
- sample_count
- state_encoding
- sample_shape
- sample_dtype
- output_samples_sha256
- energy_trace_sha256
- energy_metric_fields
- latency_metric_fields
- hardware_execution_performed
- simulator_fallback_used
- claim_boundary_acknowledged

## Transcript Schema

- schema_path: `ops/extropic_z1_transcript_schema.json`
- schema requires authenticated access proof, device identity, SDK versions, latency fields, sample shape, output checksums, metric fields, and claim-boundary acknowledgement.

## Expected Output Checksums Or Metric Fields

- mean_energy
- magnetization
- energy_autocorrelation_lag1
- kl_divergence_vs_simulator
- sample_shape
- sample_dtype
- output_samples_sha256
- energy_trace_sha256
- host_to_device_latency_us
- device_sampling_latency_us
- device_to_host_latency_us
- end_to_end_latency_us

## Simulator Artifacts Referenced

| artifact | sha256 | boundary |
|---|---|---|
| results/experiment_1543_thrml_carnot_parity_n256_schedule_stress.json | d966a96368454cad5edb9304a2dbb4fb7543230ae7f1cbf94d514c0395f53a70 | software parity only; hardware_execution=false |
| results/experiment_1544_thrml_diverse_topology_parity_n64.json | 302680028da845ba0cdbd75ad4a4a7403dca8b67a7013cba3777ad2c813b0b96 | software parity only; hardware_execution=false |

## No Hardware Execution Claim

This packet does not report Extropic Z1, XTR-0, TSU, board, synthesis, bitstream, latency, or device sample execution. The referenced evidence is software/simulator parity only.

## Access Blockers

- no_authenticated_extropic_z1_or_xtr0_device_access
- no_authenticated_hardware_run_transcript
- no_device_latency_or_sample_quality_evidence_from_z1
- no_extropic_sdk_credentials_or_device_discovery_transcript
- public_thrml_material_only_simulator_parity_artifacts
- source_docs_confirm_no_authenticated_extropic_access
- thrml_independent_rng_followup_not_completed

## Rollback Criteria

- Missing authenticated_access_proof, device_identifier, or device_discovery_command.
- Any transcript sets hardware_execution_performed=false or simulator_fallback_used=true.
- Sample shape, state encoding, or checksum fields are absent or inconsistent.
- Mean-energy, KL, magnetization, or autocorrelation metrics exceed the simulator gates.
- Latency fields are missing, impossible, or mixed across host/device scopes.
- The SDK path silently falls back to THRML/JAX/CPU simulation.

## Source Context Checks

- hardware_wishlist_mentions_no_access: True
- references_request_packet_not_claim: True
- known_issue_rng_followup: True
