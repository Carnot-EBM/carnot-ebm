use carnot_samplers::fixed_cardinality::{
    PairSwapConfig, PairSwapCore, PairSwapDraw, PairSwapSeededState, StoredEdge,
};

fn fixture() -> PairSwapCore {
    let config = PairSwapConfig::new(
        vec![
            StoredEdge::new(0, 1, 1.0),
            StoredEdge::new(0, 2, 1.0),
            StoredEdge::new(1, 2, -1.0),
        ],
        vec![0.1, -0.2, 0.3, -0.2],
        2,
        2.0,
    )
    .expect("SCENARIO-SAMPLER-7189-REPLAY fixture is valid");
    PairSwapCore::new(config)
}

#[test]
fn replay_uses_caller_tape_and_preserves_cardinality() {
    // SCENARIO-SAMPLER-7189-REPLAY
    let core = fixture();
    let initial = vec![1, 1, -1, -1];
    let tape = vec![
        PairSwapDraw::new(0, 1, 0.9).expect("valid draw"),
        PairSwapDraw::new(1, 0, 0.0).expect("valid draw"),
        PairSwapDraw::new(0, 0, 0.5).expect("valid draw"),
    ];
    let outcome = core.run_replay(&initial, &tape).expect("replay succeeds");
    assert_eq!(outcome.steps.len(), tape.len());
    assert_eq!(
        outcome
            .final_state
            .iter()
            .filter(|spin| **spin == 1)
            .count(),
        2
    );
    assert!(outcome.steps.iter().all(|step| step.cardinality == 2));
    assert!(outcome
        .steps
        .iter()
        .all(|step| step.delta_energy.is_finite()));
    assert_eq!(core.energy(&initial).expect("energy"), -0.8);
}

#[test]
fn singleton_slices_are_identity_kernels() {
    // REQ-SAMPLER-7189-KERNEL
    for cardinality in [0, 4] {
        let config = PairSwapConfig::new(
            vec![
                StoredEdge::new(0, 1, 1.0),
                StoredEdge::new(0, 2, 1.0),
                StoredEdge::new(1, 2, -1.0),
            ],
            vec![0.1, -0.2, 0.3, -0.4],
            cardinality,
            1.0,
        )
        .expect("singleton config");
        let core = PairSwapCore::new(config);
        let initial = vec![if cardinality == 0 { -1 } else { 1 }; 4];
        let step = core
            .step_from_draw(&initial, &PairSwapDraw::new(0, 0, 0.5).expect("draw"))
            .expect("singleton step");
        assert_eq!(step.state, initial);
        assert_eq!(step.proposed_state, initial);
        assert!(step.accepted);
        assert_eq!(step.delta_energy, 0.0);
    }
}

#[test]
fn seeded_stream_is_reproducible_and_keeps_the_slice() {
    // SCENARIO-SAMPLER-7189-DISTRIBUTION
    let core = fixture();
    let initial = vec![1, 1, -1, -1];
    let mut first = PairSwapSeededState::new(initial.clone(), 7189).expect("seeded state");
    let mut second = PairSwapSeededState::new(initial, 7189).expect("seeded state");
    for _ in 0..100 {
        let left = core.step_seeded(&mut first).expect("seeded step");
        let right = core.step_seeded(&mut second).expect("seeded step");
        assert_eq!(left, right);
        assert_eq!(left.cardinality, 2);
    }
    assert_eq!(first, second);
}

#[test]
fn energy_budget_charges_each_full_energy_call() {
    // REQ-SAMPLER-7189-THROUGHPUT uses Exp7187's 160-evaluation budget.
    let core = fixture();
    let outcome = core
        .run_seeded_energy_budget(&[1, 1, -1, -1], 7189, 40)
        .expect("budgeted chain succeeds");
    assert_eq!(outcome.energy_evaluations, 40);
    assert_eq!(outcome.attempted, 39);
    assert_eq!(outcome.energies.len(), 40);
    assert_eq!(outcome.samples.len(), 40);
}

#[test]
fn malformed_models_states_and_draws_fail_closed() {
    // SCENARIO-SAMPLER-7189-ARTIFACT relies on strict kernel inputs.
    assert!(StoredEdge::try_new(1, 0, 1.0).is_err());
    assert!(StoredEdge::try_new(0, 0, 1.0).is_err());
    assert!(StoredEdge::try_new(0, 1, f64::NAN).is_err());
    assert!(PairSwapDraw::new(0, 0, -0.1).is_err());
    assert!(PairSwapDraw::new(0, 0, 1.0).is_err());
    assert!(PairSwapConfig::new(vec![], vec![], 0, 1.0).is_err());
    assert!(PairSwapConfig::new(vec![], vec![0.1, 0.2], 3, 1.0).is_err());
    assert!(PairSwapConfig::new(vec![], vec![0.1, 0.0], 1, 1.0).is_err());
    assert!(PairSwapConfig::new(vec![], vec![0.1, 0.2], 1, 0.0).is_err());
    assert!(PairSwapConfig::new(vec![StoredEdge::new(0, 3, 1.0)], vec![0.1, 0.2], 1, 1.0).is_err());

    let core = fixture();
    assert!(core.energy(&[1, -1]).is_err());
    assert!(core.energy(&[1, 0, -1, 1]).is_err());
    assert!(core
        .step_from_draw(
            &[1, 1, -1, -1],
            &PairSwapDraw::new(2, 0, 0.5).expect("syntactically valid draw")
        )
        .is_err());
    assert!(core
        .step_from_draw(
            &[1, 1, -1, -1],
            &PairSwapDraw::new(0, 2, 0.5).expect("syntactically valid draw")
        )
        .is_err());
    assert!(PairSwapSeededState::new(vec![1, 0], 1).is_err());
}
