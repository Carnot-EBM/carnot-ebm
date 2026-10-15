//! SCENARIO-VERIFY-8533-PARITY: exercise every owned residual rejection path.
use carnot_core::nine_input_8533::residuals;

#[test]
fn partition_and_endpoints() {
    let coefficients = vec![1.0; 109];
    for value in [0.0, f64::from_bits(1), 0.25, 0.5, 1.0] {
        let result = residuals(&coefficients, &[vec![value; 9]]).unwrap();
        assert!((result[0] - 10.0).abs() < 1e-13);
    }
    let mut coefficients = vec![0.0; 109];
    coefficients[0] = 2.0;
    coefficients[107] = 3.0;
    assert_eq!(residuals(&coefficients, &[vec![0.0; 9]]).unwrap(), [2.0]);
    assert_eq!(residuals(&coefficients, &[vec![1.0; 9]]).unwrap(), [3.0]);
}

#[test]
fn reject_invalid_operands() {
    let coefficients = vec![0.0; 109];
    for bad in [vec![], vec![0.0; 108], vec![0.0; 110], vec![f64::NAN; 109]] {
        assert_eq!(residuals(&bad, &[vec![0.5; 9]]), Err("coefficients"));
    }
    assert_eq!(residuals(&coefficients, &[]), Err("features"));
    for bad in [
        vec![],
        vec![0.0; 8],
        vec![0.0; 10],
        vec![f64::NAN; 9],
        vec![f64::INFINITY; 9],
        vec![-0.1; 9],
        vec![1.1; 9],
    ] {
        assert_eq!(residuals(&coefficients, &[bad]), Err("features"));
    }
    assert_eq!(
        residuals(&vec![f64::MAX; 109], &[vec![0.5; 9]]),
        Err("residual")
    );
}
