//! Tests for the fixed Exp6166/Exp6180 mode-jump sampler port.
//!
//! Spec refs: REQ-SAMPLE-6194, SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY,
//! SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY,
//! SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION.

use carnot_samplers::mode_jump::{ModeJumpConfig, ModeJumpCore, ModeJumpState};

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
