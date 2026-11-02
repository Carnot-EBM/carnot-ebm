//! REQ-REPORT-8731: resident commits preserve the original checked kernel.
use super::*;

#[test]
fn resident_sequential_and_errors() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "resident_test").unwrap();
        register(&module).unwrap();
        assert!(module
            .getattr("ResidentSparse8731")
            .unwrap()
            .call0()
            .is_err());
        let context = serde_json::to_string(&Fixed::default()).unwrap();
        let bound = module
            .getattr("ResidentSparse8731")
            .unwrap()
            .call1((vec![0_i64; 109], UNIT, UNIT, &context))
            .unwrap();
        for name in ["audit", "telemetry", "serialize"] {
            assert!(bound.call_method1(name, (0,)).is_ok());
            assert!(bound.call_method1(name, (1,)).is_err());
        }
        let mut r = ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, &context).unwrap();
        assert!(r.apply(UNIT / 2, 1, 0).is_err());
        assert!(r.prepare(vec![0; 16], vec![-1; 9]).is_err());
        let (ids, weights, _, token) = r.prepare(vec![0; 16], vec![0; 9]).unwrap();
        let mut f = serde_json::from_str(&serde_json::to_string(&r.r.f).unwrap()).unwrap();
        let expected = update(&mut f, &[0; 109], UNIT, UNIT, UNIT / 2, 1, &ids, &weights).unwrap();
        assert!(r.apply(-1, 0, token).is_err());
        assert!(r.apply(0, 2, token).is_err());
        assert!(r.apply(0, 0, token - 1).is_err());
        let actual = r.apply(UNIT / 2, 1, token).unwrap();
        assert_eq!(r.r.parameters, expected.0);
        assert_eq!((actual.0, actual.2), (expected.1, expected.2));
        assert_eq!(
            serde_json::to_value(&r.r.f).unwrap(),
            serde_json::to_value(&f).unwrap()
        );
        assert_eq!(r.serialize(token).unwrap().len(), 444);
        for (n, expected) in [(-7, -4), (-5, -2), (5, 2), (7, 4)] {
            assert_eq!(r.control("ratio", n, 2).unwrap(), expected);
        }
        assert!(r.control("ratio", 1, 0).is_err());
        assert!(r.control("product", i64::MAX, 2).is_err());
        assert!(ResidentSparse8731::new(vec![0; 109], 1, UNIT, &context).is_err());
        assert!(ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, "{}").is_ok());
        assert!(ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, "bad").is_err());
        r.quantized(0, false);
        assert_eq!(r.audit(token).unwrap().0.len(), 109);
        assert!(r.prepare(vec![-1; 16], vec![0; 9]).is_err());
        assert!(r.control("serialize", 0, 0).is_err());
        r.token = u64::MAX;
        assert!(r.prepare(vec![0; 16], vec![0; 9]).is_err());
        r.r.parameters.clear();
        assert!(r.serialize(u64::MAX).is_err());
        // Corrupt internal state to prove checked failures cannot commit coefficients.
        let mut corrupt = ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, &context).unwrap();
        corrupt.r.parameters[108] = i64::MAX;
        assert!(corrupt.prepare(vec![0; 16], vec![0; 9]).is_err());
        let mut corrupt = ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, &context).unwrap();
        let (_, _, _, token) = corrupt.prepare(vec![0; 16], vec![0; 9]).unwrap();
        corrupt.r.decay_scale = i64::MAX;
        assert!(corrupt.apply(UNIT / 2, 1, token).is_err());
        assert_eq!(corrupt.r.parameters, vec![0; 109]);
    });
}
