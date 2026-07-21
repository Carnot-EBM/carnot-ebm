use carnot_samplers::one_axis_tempering::{
    OneAxisTemperingConfig, OneAxisTemperingCore, OneAxisTemperingState,
};

fn config() -> OneAxisTemperingConfig {
    OneAxisTemperingConfig::new(
        vec![
            vec![0.0, 0.58, -0.47],
            vec![0.58, 0.0, 0.52],
            vec![-0.47, 0.52, 0.0],
        ],
        vec![0.12, -0.18, 0.09],
        vec![0.45, 0.8, 1.25],
        0.72,
        0.17,
    )
    .expect("REQ-SAMPLE-5714 fixture config must validate")
}

#[test]
fn req_sample_5714_energy_proposal_and_swap_are_deterministic() {
    let core = OneAxisTemperingCore::new(config());
    let state = vec![1, -1, 1];
    let target = vec![-1, -1, 1];

    assert!((core.energy(&state).unwrap() - 1.18).abs() < 1e-12);
    assert!(core
        .proposal_log_probability(&state, &target, 0.8)
        .unwrap()
        .is_finite());

    let decision = core
        .corrected_step(&state, 0.8, &[0.07, 0.61, 0.44, 0.19])
        .unwrap();
    assert_eq!(decision.proposed_state.len(), 3);
    assert!(decision.log_acceptance.is_finite());

    let states = vec![vec![1, -1, 1], vec![-1, -1, 1], vec![1, 1, -1]];
    let labels = vec![2, 0, 1];
    let swap = core.swap_decision(&states, &labels, &[1, 2], 0.13).unwrap();
    assert_eq!(swap.proposed_labels, vec![1, 0, 2]);
    assert!(swap.log_ratio.is_finite());
}

#[test]
fn scenario_sample_5714_checkpoint_restart_reproduces_schedule() {
    let core = OneAxisTemperingCore::new(config());
    let initial = OneAxisTemperingState::new(
        vec![vec![1, -1, 1], vec![-1, -1, 1], vec![1, 1, -1]],
        vec![0, 1, 2],
        5714,
        0,
    )
    .expect("SCENARIO-SAMPLE-5714 fixture state must validate");

    assert_eq!(
        core.scheduler_trace(),
        vec!["within:0", "within:1", "within:2", "swap:0-1", "swap:1-2"]
    );
    assert_eq!(core.target_state(&initial).unwrap(), vec![1, 1, -1]);

    let next = core.step(&initial).unwrap();
    let restarted = OneAxisTemperingState::new(
        next.states.clone(),
        next.labels.clone(),
        next.rng_state,
        next.sweep,
    )
    .unwrap();
    assert_eq!(core.step(&next).unwrap(), core.step(&restarted).unwrap());
}

#[test]
fn req_sample_5764_compact_sweeps_reuse_buffers_and_match_step_replay() {
    let core = OneAxisTemperingCore::new(config());
    let initial = OneAxisTemperingState::new(
        vec![vec![1, -1, 1], vec![-1, -1, 1], vec![1, 1, -1]],
        vec![0, 1, 2],
        5764,
        0,
    )
    .expect("REQ-SAMPLE-5764 fixture state must validate");

    let compact = core.run_compact_sweeps(&initial, 1, 3).unwrap();
    let mut replay = initial.clone();
    let mut expected_samples = Vec::new();
    for sweep in 0..4 {
        replay = core.step(&replay).unwrap();
        if sweep >= 1 {
            expected_samples.extend(core.target_state(&replay).unwrap());
        }
    }

    assert_eq!(compact.samples_spin, expected_samples);
    assert_eq!(compact.final_state, replay);
    assert_eq!(compact.counters.rust_per_sample_heap_allocations, 0);
    assert_eq!(compact.counters.workspace_allocations, 4);
    assert_eq!(compact.counters.output_allocations, 1);
    assert!(compact.buffer_reuse.contiguous_samples);
    assert_eq!(compact.worker_pool.fixed_worker_count, 1);
}

#[test]
fn req_sample_5714_malformed_inputs_fail_closed() {
    assert!(OneAxisTemperingConfig::new(
        vec![vec![0.0, 0.1]],
        vec![0.0, 0.1],
        vec![0.45, 0.8],
        0.72,
        0.17,
    )
    .is_err());

    assert!(
        OneAxisTemperingConfig::new(vec![vec![0.0]], vec![0.0], vec![0.8, 0.8], 0.72, 0.17,)
            .is_err()
    );

    let core = OneAxisTemperingCore::new(config());
    assert!(core.energy(&[1, 0, -1]).is_err());
    assert!(OneAxisTemperingState::new(
        vec![vec![1, -1, 1], vec![1, -1, 1], vec![1, -1, 1]],
        vec![0, 0, 2],
        7,
        0,
    )
    .is_err());
}
