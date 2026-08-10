//! Tests for the fixed Exp6166/Exp6180 mode-jump sampler port.
//!
//! Spec refs: REQ-SAMPLE-6194, SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY,
//! SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY,
//! SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION.

use carnot_samplers::mode_jump::{
    ModeJumpConfig, ModeJumpCore, ModeJumpState, ModeJumpStateMetadata,
};

fn frozen_config() -> ModeJumpConfig {
    ModeJumpConfig::new(
        vec![
            "left_peak".to_string(),
            "left_shoulder".to_string(),
            "valley_left".to_string(),
            "valley_right".to_string(),
            "right_peak".to_string(),
            "right_shoulder".to_string(),
        ],
        vec![0.36, 0.24, 0.025, 0.025, 0.245, 0.105],
        vec![
            vec![0.0, 0.75, 0.0, 0.0, 0.25, 0.0],
            vec![0.375, 0.0, 0.375, 0.0, 0.0, 0.25],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.25, 0.0, 0.0, 0.0, 0.0, 0.75],
            vec![0.0, 0.25, 0.0, 0.375, 0.375, 0.0],
        ],
    )
    .unwrap()
}

#[test]
fn req_sample_6194_replays_exact_exp6166_short_chain() {
    let core = ModeJumpCore::new(frozen_config());
    let mut state = ModeJumpState::new("left_peak".to_string(), 6194, 0, 0).unwrap();
    let expected = [
        ("left_shoulder", false, "left_peak"),
        ("left_shoulder", false, "left_peak"),
        ("left_shoulder", false, "left_peak"),
        ("left_shoulder", false, "left_peak"),
        ("left_shoulder", false, "left_peak"),
        ("left_shoulder", false, "left_peak"),
        ("right_peak", true, "right_peak"),
        ("right_shoulder", false, "right_peak"),
        ("left_peak", true, "left_peak"),
        ("left_shoulder", false, "left_peak"),
    ];

    for (index, (proposed, accepted, after)) in expected.iter().enumerate() {
        let outcome = core.step_trace(&state).unwrap();
        assert_eq!(outcome.proposed_label, *proposed);
        assert_eq!(outcome.accepted, *accepted);
        assert_eq!(outcome.state.current_label, *after);
        assert_eq!(outcome.state.step, index + 1);
        state = outcome.state;
    }

    assert_eq!(state.current_label, "left_peak");
    assert_eq!(state.accepted_count, 2);
    assert_eq!(state.rng_state, 17181667396803735766);
}

#[test]
fn req_sample_6194_uses_exact_mh_acceptance_equation() {
    let core = ModeJumpCore::new(frozen_config());
    let state = ModeJumpState::new("left_peak".to_string(), 6194, 0, 0).unwrap();
    let outcome = core.step_trace(&state).unwrap();

    assert!((outcome.proposal_uniform - 0.011404724311910863).abs() < 1e-15);
    assert!((outcome.acceptance_uniform - 0.4492787037262084).abs() < 1e-15);
    assert!((outcome.current_energy - -0.36_f64.ln()).abs() < 1e-12);
    assert!((outcome.proposed_energy - -0.24_f64.ln()).abs() < 1e-12);
    assert!((outcome.proposal_log_forward - 0.75_f64.ln()).abs() < 1e-12);
    assert!((outcome.proposal_log_reverse - 0.375_f64.ln()).abs() < 1e-12);
    assert!((outcome.log_acceptance + 1.0986122886681096).abs() < 1e-12);
    assert!((outcome.acceptance_probability - (1.0 / 3.0)).abs() < 1e-12);
    assert!(!outcome.accepted);
}

#[test]
fn scenario_sample_6194_distribution_quality_matches_frozen_target() {
    let core = ModeJumpCore::new(frozen_config());
    let state = ModeJumpState::new("left_peak".to_string(), 6194, 0, 0).unwrap();
    let summary = core.run(&state, 51_000, 1_000).unwrap();

    assert_eq!(summary.sample_count, 50_000);
    assert!(summary.total_variation_to_target <= 0.01);
    assert!(summary.kl_target_to_empirical <= 0.001);
    assert!(summary.acceptance_rate > 0.45);
    assert!(summary.acceptance_rate < 0.60);
    assert!(summary.effective_sample_size > 10_000.0);
    assert!((summary.frequency("left_peak").unwrap() - 0.35592).abs() < 1e-12);
}

#[test]
fn scenario_sample_6194_serialization_and_errors_fail_closed() {
    let config = frozen_config();
    let core = ModeJumpCore::new(config.clone());
    let state = ModeJumpState::new("left_peak".to_string(), 6194, 0, 0).unwrap();
    let serialized = state.serialize();
    let restored = core.state_from_serialized(&serialized).unwrap();
    assert_eq!(restored, state);

    assert!(ModeJumpConfig::new(vec![], vec![], vec![]).is_err());
    assert!(ModeJumpConfig::new(
        config.labels.clone(),
        vec![0.36, 0.24, 0.025, 0.025, 0.245, 0.0],
        config.proposal_probabilities.clone(),
    )
    .is_err());
    assert!(ModeJumpState::new("unsupported_shadow".to_string(), 1, 0, 0).is_ok());
    assert!(core
        .step_trace(&ModeJumpState::new("unsupported_shadow".to_string(), 1, 0, 0).unwrap())
        .is_err());
    assert!(core
        .state_from_serialized("mode_jump_state_v1|bad|1|0|0")
        .is_err());
    assert!(core.state_from_serialized("not-a-mode-jump-state").is_err());
    assert!(core.run(&state, 0, 0).is_err());
}

#[test]
fn req_sample_6208_runtime_adapter_uses_fixed_kernel_contract() {
    // REQ-SAMPLE-6208-FIXED-KERNEL, REQ-SAMPLE-6208-SHAPE-CONTRACT:
    // the runtime adapter is allowed to route this exact kernel, not mutate it.
    let config = frozen_config();
    assert_eq!(config.labels.len(), 6);
    assert_eq!(config.target_probabilities.len(), 6);
    assert_eq!(config.proposal_probabilities.len(), 6);
    assert!(config
        .proposal_probabilities
        .iter()
        .all(|row| row.len() == 6));

    let core = ModeJumpCore::new(config.clone());
    let state = ModeJumpState::new("left_peak".to_string(), 6208, 0, 0).unwrap();
    let summary = core.run(&state, 128, 8).unwrap();

    assert_eq!(summary.sample_count, 120);
    assert_eq!(summary.attempted_count, 128);
    assert!(summary.accepted_count <= summary.attempted_count);
    assert_eq!(summary.frequencies.len(), 6);
    assert!(summary.effective_sample_size > 0.0);
    assert!(core
        .state_from_serialized(&summary.final_state.serialize())
        .is_ok());
}

#[test]
fn req_sample_6208_runtime_controls_reject_malformed_fixed_config() {
    // REQ-SAMPLE-6208-RUNTIME-ACCOUNTING: broken runtime controls fail closed
    // before a caller can treat malformed probabilities as the qualified kernel.
    let config = frozen_config();
    let mut duplicate_labels = config.labels.clone();
    duplicate_labels[1] = duplicate_labels[0].clone();
    assert!(ModeJumpConfig::new(
        duplicate_labels,
        config.target_probabilities.clone(),
        config.proposal_probabilities.clone(),
    )
    .is_err());

    let mut non_normalized_target = config.target_probabilities.clone();
    non_normalized_target[0] += 0.25;
    assert!(ModeJumpConfig::new(
        config.labels.clone(),
        non_normalized_target,
        config.proposal_probabilities.clone(),
    )
    .is_err());

    let mut asymmetric = config.proposal_probabilities.clone();
    asymmetric[0][1] = 0.0;
    asymmetric[0][4] = 1.0;
    assert!(ModeJumpConfig::new(
        config.labels.clone(),
        config.target_probabilities.clone(),
        asymmetric,
    )
    .is_err());

    let mut bad_row = config.proposal_probabilities.clone();
    bad_row[0] = vec![1.0];
    assert!(ModeJumpConfig::new(config.labels, config.target_probabilities, bad_row).is_err());
}

fn complete_no_self_proposal(n: usize) -> Vec<Vec<f64>> {
    let off_diagonal = 1.0 / (n - 1) as f64;
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 0.0 } else { off_diagonal })
                .collect()
        })
        .collect()
}

#[test]
fn req_sampler_6280_variable_cardinality_metadata_roundtrips_and_runs() {
    // REQ-SAMPLER-6280-METADATA and REQ-SAMPLER-6280-PARITY:
    // typed rank-1 metadata validates before the generic MH kernel runs.
    let labels = vec![
        "-1,0".to_string(),
        "-1,1".to_string(),
        "-1,2".to_string(),
        "+1,0".to_string(),
        "+1,1".to_string(),
        "+1,2".to_string(),
    ];
    let state_values = vec![
        vec![0, 0],
        vec![0, 1],
        vec![0, 2],
        vec![1, 0],
        vec![1, 1],
        vec![1, 2],
    ];
    let metadata = ModeJumpStateMetadata::new(
        "carnot.mode_jump.typed_state_metadata.v1".to_string(),
        vec![2],
        vec![2, 3],
        "mixed_radix_rank1".to_string(),
        labels.clone(),
        state_values.clone(),
        "explicit_support_complete_no_self".to_string(),
        6,
    )
    .unwrap();

    assert_eq!(metadata.encode_label("-1,2").unwrap(), 2);
    assert_eq!(metadata.decode_index(4).unwrap(), " +1,1".trim());
    assert_eq!(metadata.state_value("+1,2").unwrap(), vec![1, 2]);
    assert!(metadata.decode_index(6).is_err());

    let config = ModeJumpConfig::new_with_metadata(
        labels.clone(),
        vec![0.10, 0.15, 0.20, 0.25, 0.12, 0.18],
        complete_no_self_proposal(labels.len()),
        metadata,
    )
    .unwrap();
    let core = ModeJumpCore::new(config);
    let state = ModeJumpState::new("-1,0".to_string(), 6280, 0, 0).unwrap();
    let first = core.step_trace(&state).unwrap();
    let replay = core.step_trace(&state).unwrap();

    assert_eq!(first, replay);
    assert_eq!(first.state.step, 1);
    assert!(labels.contains(&first.proposed_label));
    assert!(core.run(&state, 16, 2).is_ok());
}

#[test]
fn req_sampler_6280_metadata_controls_fail_closed() {
    // REQ-SAMPLER-6280-CONTROLS: inconsistent metadata and proposal domains
    // are rejected before any state can be interpreted under the wrong labels.
    let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
    let good_metadata = ModeJumpStateMetadata::new(
        "carnot.mode_jump.typed_state_metadata.v1".to_string(),
        vec![1],
        vec![3],
        "zero_based_rank1".to_string(),
        labels.clone(),
        vec![vec![0], vec![1], vec![2]],
        "explicit_support_complete_no_self".to_string(),
        3,
    )
    .unwrap();

    let mut permuted = labels.clone();
    permuted.reverse();
    assert!(ModeJumpConfig::new_with_metadata(
        permuted,
        vec![0.2, 0.3, 0.5],
        complete_no_self_proposal(labels.len()),
        good_metadata.clone(),
    )
    .is_err());

    assert!(ModeJumpStateMetadata::new(
        "carnot.mode_jump.typed_state_metadata.v1".to_string(),
        vec![1, 1],
        vec![3],
        "zero_based_rank1".to_string(),
        labels.clone(),
        vec![vec![0], vec![1], vec![2]],
        "explicit_support_complete_no_self".to_string(),
        3,
    )
    .is_err());
    assert!(ModeJumpStateMetadata::new(
        "carnot.mode_jump.typed_state_metadata.v1".to_string(),
        vec![1],
        vec![3],
        "zero_based_rank1".to_string(),
        labels.clone(),
        vec![vec![0], vec![1], vec![3]],
        "explicit_support_complete_no_self".to_string(),
        3,
    )
    .is_err());
    assert!(ModeJumpConfig::new_with_metadata(
        labels,
        vec![0.2, 0.3, 0.5],
        vec![vec![0.0, 1.0]],
        good_metadata,
    )
    .is_err());
}
