use carnot_boltzmann::{soft_bellman_solve, soft_bellman_solve_path};
use carnot_core::Float;

fn assert_close(actual: Float, expected: Float, tol: Float) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn soft_bellman_probability_gauge_maps_logprobs_to_energy() {
    // REQ-INFER-2056, SCENARIO-INFER-2056-001
    let logprobs = vec![-0.1, -1.25, -0.05];

    let solution = soft_bellman_solve(&logprobs).unwrap();

    assert_eq!(solution.immediate_rewards, logprobs);
    assert_eq!(solution.soft_values.len(), logprobs.len() + 1);
    for value in &solution.soft_values {
        assert_close(*value, 0.0, 1e-7);
    }
    assert_eq!(solution.token_energies, vec![0.1, 1.25, 0.05]);
    assert_close(solution.sequence_affinity, -1.4, 1e-6);
    assert_close(solution.sequence_energy, 1.4, 1e-6);
    assert_close(solution.log_probability, -1.4, 1e-6);
    assert_close(solution.log_partition, 0.0, 1e-7);
    assert_close(solution.max_abs_bellman_residual, 0.0, 1e-7);
}

#[test]
fn soft_bellman_empty_sequence_has_zero_energy() {
    // REQ-INFER-2056
    let solution = soft_bellman_solve(&[]).unwrap();

    assert!(solution.immediate_rewards.is_empty());
    assert!(solution.token_energies.is_empty());
    assert_eq!(solution.soft_values, vec![0.0]);
    assert_eq!(solution.sequence_affinity, 0.0);
    assert_eq!(solution.sequence_energy, 0.0);
    assert_eq!(solution.log_probability, 0.0);
}

#[test]
fn soft_bellman_rejects_nonfinite_logprobs() {
    // REQ-INFER-2056
    let err = soft_bellman_solve(&[-0.1, Float::NAN]).unwrap_err();
    assert!(err.to_string().contains("finite"));
}

#[test]
fn soft_bellman_rejects_positive_logprobs() {
    // REQ-INFER-2056
    let err = soft_bellman_solve(&[-0.1, 0.25]).unwrap_err();
    assert!(err.to_string().contains("positive"));
}

#[test]
fn soft_bellman_path_extracts_chosen_logprobs_from_rows() {
    // REQ-INFER-2056, SCENARIO-INFER-2056-001
    let rows = vec![
        vec![0.6_f32.ln(), 0.4_f32.ln()],
        vec![0.2_f32.ln(), 0.8_f32.ln()],
    ];
    let token_ids = vec![1, 0];

    let solution = soft_bellman_solve_path(&rows, &token_ids).unwrap();

    assert_close(solution.immediate_rewards[0], 0.4_f32.ln(), 1e-6);
    assert_close(solution.immediate_rewards[1], 0.2_f32.ln(), 1e-6);
    assert_close(
        solution.sequence_energy,
        -(0.4_f32.ln() + 0.2_f32.ln()),
        1e-6,
    );
    assert!(solution.max_abs_bellman_residual <= 1e-6);
}

#[test]
fn soft_bellman_path_rejects_bad_shapes_and_token_ids() {
    // REQ-INFER-2056
    let rows = vec![vec![0.0_f32]];

    let length_err = soft_bellman_solve_path(&rows, &[]).unwrap_err();
    assert!(length_err.to_string().contains("Dimension mismatch"));

    let id_err = soft_bellman_solve_path(&rows, &[1]).unwrap_err();
    assert!(id_err.to_string().contains("out of range"));

    let empty_err = soft_bellman_solve_path(&[vec![]], &[0]).unwrap_err();
    assert!(empty_err.to_string().contains("empty"));
}

#[test]
fn soft_bellman_path_rejects_unnormalized_logprob_rows() {
    // REQ-INFER-2056
    let rows = vec![vec![-0.2, -0.3]];

    let err = soft_bellman_solve_path(&rows, &[0]).unwrap_err();

    assert!(err.to_string().contains("normalized"));
}

#[test]
fn soft_bellman_path_rejects_nonfinite_and_positive_row_values() {
    // REQ-INFER-2056
    let nonfinite_err = soft_bellman_solve_path(&[vec![Float::NAN]], &[0]).unwrap_err();
    assert!(nonfinite_err.to_string().contains("finite"));

    let positive_err = soft_bellman_solve_path(&[vec![0.1, -3.0]], &[0]).unwrap_err();
    assert!(positive_err.to_string().contains("positive"));
}

#[test]
fn soft_bellman_exhaustive_sequence_energies_normalize() {
    // REQ-INFER-2056, SCENARIO-INFER-2056-002
    let step0 = [0.7_f32.ln(), 0.3_f32.ln()];
    let step1 = [0.25_f32.ln(), 0.75_f32.ln()];
    let mut total_mass = 0.0;
    let mut high_likelihood_energy = None;
    let mut low_likelihood_energy = None;

    for first in 0..2 {
        for second in 0..2 {
            let logprobs = vec![step0[first], step1[second]];
            let solution = soft_bellman_solve(&logprobs).unwrap();
            total_mass += (-solution.sequence_energy).exp();

            if first == 0 && second == 1 {
                high_likelihood_energy = Some(solution.sequence_energy);
            }
            if first == 1 && second == 0 {
                low_likelihood_energy = Some(solution.sequence_energy);
            }
        }
    }

    assert_close(total_mass, 1.0, 1e-6);
    assert!(high_likelihood_energy.unwrap() < low_likelihood_energy.unwrap());
}
