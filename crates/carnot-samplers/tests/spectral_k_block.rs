use carnot_samplers::spectral_k_block::{
    SpectralKBlockConfig, SpectralKBlockCore, SpectralKBlockState,
};

fn config() -> SpectralKBlockConfig {
    SpectralKBlockConfig::new(
        vec![
            vec![0.0, 0.8, -0.35, 0.0],
            vec![0.8, 0.0, 0.45, -0.25],
            vec![-0.35, 0.45, 0.0, 0.7],
            vec![0.0, -0.25, 0.7, 0.0],
        ],
        vec![0.1, -0.05, 0.0, 0.08],
        0.9,
        vec![vec![0, 1], vec![2, 3]],
    )
    .expect("REQ-SAMPLER-6612 fixture must validate")
}

#[test]
fn req_sampler_6612_exact_block_transition_replays_seed() {
    let core = SpectralKBlockCore::new(config());
    let initial = SpectralKBlockState::new(vec![1, -1, 1, -1], 6612, 0, 0)
        .expect("SCENARIO-SAMPLER-6612 state must validate");

    let first = core
        .run_chain(&initial, 7, 32)
        .expect("REQ-SAMPLER-6612 chain must run");
    let replay = core
        .run_chain(&initial, 7, 32)
        .expect("REQ-RUSTPY-6612 chain must replay");

    assert_eq!(first, replay);
    assert_eq!(first.samples.len(), 32 * 4);
    assert_eq!(first.transitions, 39);
    assert_eq!(first.spins_updated, 78);
    assert_eq!(first.final_state.transition, 39);
    assert!(first
        .samples
        .iter()
        .all(|value| *value == -1 || *value == 1));
}

#[test]
fn scenario_rustpy_6612_single_spin_and_block_kernels_preserve_contract() {
    let block = SpectralKBlockCore::new(config());
    let singles = SpectralKBlockCore::new(
        SpectralKBlockConfig::new(
            config().couplings,
            config().fields,
            config().temperature,
            vec![vec![0], vec![1], vec![2], vec![3]],
        )
        .unwrap(),
    );
    let initial = SpectralKBlockState::new(vec![-1, -1, 1, 1], 77, 0, 0).unwrap();

    let block_run = block.run_chain(&initial, 3, 11).unwrap();
    let single_run = singles.run_chain(&initial, 3, 11).unwrap();

    assert_eq!(block_run.transitions, single_run.transitions);
    assert_eq!(block_run.spins_updated, 2 * single_run.spins_updated);
    assert_ne!(block_run.samples, single_run.samples);
}

#[test]
fn req_rustpy_6612_malformed_inputs_fail_closed() {
    assert!(
        SpectralKBlockConfig::new(vec![vec![0.0, 0.2]], vec![0.0], 1.0, vec![vec![0]],).is_err()
    );
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.1, 0.0]],
        vec![0.0, 0.0],
        1.0,
        vec![vec![0], vec![1]],
    )
    .is_err());
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.2, 0.0]],
        vec![0.0, 0.0],
        0.0,
        vec![vec![0], vec![1]],
    )
    .is_err());
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.2, 0.0]],
        vec![0.0, 0.0],
        1.0,
        vec![vec![0], vec![0]],
    )
    .is_err());
    assert!(SpectralKBlockState::new(vec![1, 0], 1, 0, 0).is_err());

    let core = SpectralKBlockCore::new(config());
    let state = SpectralKBlockState::new(vec![1, 1, -1, -1], 1, 0, 0).unwrap();
    assert!(core.run_chain(&state, 0, 0).is_err());
}

#[test]
fn req_sampler_6612_validation_and_counter_guards_are_covered() {
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.2, 0.0]],
        vec![0.0],
        1.0,
        vec![vec![0], vec![1]],
    )
    .is_err());
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, f64::NAN], vec![f64::NAN, 0.0]],
        vec![0.0, 0.0],
        1.0,
        vec![vec![0], vec![1]],
    )
    .is_err());
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.2, 0.0]],
        vec![0.0, 0.0],
        1.0,
        vec![vec![], vec![0, 1]],
    )
    .is_err());
    assert!(SpectralKBlockConfig::new(
        vec![vec![0.0, 0.2], vec![0.2, 0.0]],
        vec![0.0, 0.0],
        1.0,
        vec![vec![0], vec![2]],
    )
    .is_err());
    assert!(SpectralKBlockState::new(vec![], 1, 0, 0).is_err());

    let core = SpectralKBlockCore::new(config());
    assert!((core.energy(&[1, -1, 1, -1]).unwrap() - 2.48).abs() < 1.0e-12);
    assert!(core.energy(&[1, -1]).is_err());

    let state = SpectralKBlockState::new(vec![1, 1, -1, -1], 1, 0, 0).unwrap();
    assert!(core.run_chain(&state, usize::MAX, 1).is_err());
    assert!(core.run_chain(&state, 0, usize::MAX).is_err());

    let transition_overflow =
        SpectralKBlockState::new(vec![1, 1, -1, -1], 1, usize::MAX, 0).unwrap();
    assert!(core.run_chain(&transition_overflow, 0, 1).is_err());
    let update_overflow = SpectralKBlockState::new(vec![1, 1, -1, -1], 1, 0, usize::MAX).unwrap();
    assert!(core.run_chain(&update_overflow, 0, 1).is_err());
}
