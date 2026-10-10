//! REQ-VERIFY-8379: pure Rust tests keep invalid operands away from native arithmetic.

use carnot_core::direct_spline_8379::{design, logits, update};

#[test]
fn req_verify_8379_basis_endpoints_and_knots() {
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.387] {
        let d = design(&[0.7, v, v, v, v]).unwrap();
        assert!((d[2..10].iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert_eq!(d.len(), 34);
    }
    let c = vec![1.0; 34];
    assert_eq!(logits(&c, &[vec![0.0; 5]], 2.0).unwrap(), [2.5]);
    assert_eq!(design(&[0.0; 5]).unwrap()[2], 1.0);
    assert_eq!(design(&[0.0, 1.0, 1.0, 1.0, 1.0]).unwrap()[9], 1.0);
}

#[test]
fn req_verify_8379_update_and_validation() {
    let c = vec![0.0; 34];
    assert!(logits(&c, &[], 1.0).is_err());
    assert_eq!(update(&c, &[0.0; 5], 0.0).unwrap(), c);
    let changed = update(&c, &[0.0; 5], 2.0).unwrap();
    assert_eq!(changed[0..2], [0.0, 0.0]);
    assert_eq!(changed[2], -0.005);
    let bound = vec![4.0; 34];
    assert_eq!(update(&bound, &[0.0; 5], -1.0).unwrap(), bound);
    for x in [
        vec![],
        vec![f64::NAN; 5],
        vec![f64::INFINITY; 5],
        vec![0.0, -1.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0, 0.0],
    ] {
        assert!(design(&x).is_err());
    }
    for bad in [vec![], vec![f64::NAN; 34], vec![5.0; 34]] {
        assert!(logits(&bad, &[vec![0.0; 5]], 1.0).is_err());
        assert!(update(&bad, &[0.0; 5], 0.0).is_err());
    }
    for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(logits(&c, &[vec![0.0; 5]], t).is_err());
    }
    assert!(logits(&c, &[vec![]], 1.0).is_err());
    assert!(update(&c, &[0.0; 5], f64::NAN).is_err());
    assert!(update(&c, &[], 0.0).is_err());
}
