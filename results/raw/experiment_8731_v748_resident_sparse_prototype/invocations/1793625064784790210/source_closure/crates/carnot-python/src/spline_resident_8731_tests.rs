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
            assert!(bound.call_method0(name).is_ok());
        }
        // Exercise the generated no-argument ABI wrappers as well as Python's vector-call route.
        // The GIL and live bound object keep every input pointer valid.
        let pointers = unsafe {
            [
                ResidentSparse8731::__pymethod_audit__(py, bound.as_ptr()).unwrap(),
                ResidentSparse8731::__pymethod_telemetry__(py, bound.as_ptr()).unwrap(),
                ResidentSparse8731::__pymethod_serialize__(py, bound.as_ptr()).unwrap(),
            ]
        };
        for pointer in pointers {
            let result = unsafe { Bound::<PyAny>::from_owned_ptr(py, pointer) };
            assert!(!result.is_none());
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
        assert_eq!(r.serialize().unwrap().len(), 444);
        for (n, expected) in [(-7, -4), (-5, -2), (5, 2), (7, 4)] {
            assert_eq!(r.control("ratio", n, 2).unwrap(), expected);
        }
        assert!(r.control("ratio", 1, 0).is_err());
        assert!(r.control("product", i64::MAX, 2).is_err());
        assert!(ResidentSparse8731::new(vec![0; 109], 1, UNIT, &context).is_err());
        assert!(ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, "{}").is_ok());
        assert!(ResidentSparse8731::new(vec![0; 109], UNIT, UNIT, "bad").is_err());
        r.quantized(0, false);
        assert_eq!(r.audit().0.len(), 109);
        assert!(r.prepare(vec![-1; 16], vec![0; 9]).is_err());
        assert!(r.control("serialize", 0, 0).is_err());
        r.token = u64::MAX;
        assert!(r.prepare(vec![0; 16], vec![0; 9]).is_err());
        r.r.parameters.clear();
        assert!(r.serialize().is_err());
    });
}
