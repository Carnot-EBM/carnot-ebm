//! REQ-REPORT-8719: negative ties must retain Python floor semantics.
use super::*;

#[test]
fn ties_checks_and_saturation() {
    let mut f = Fixed::default();
    for (n, d, expected) in [(-5, 2, -2), (-7, 2, -4), (5, 2, 2), (7, 2, 4)] {
        assert_eq!(f.ratio(n, d).unwrap(), expected);
    }
    assert_eq!(f.ratio(1, 0), Err("positive_denominator"));
    assert_eq!(f.checked(i64::MAX as i128 + 1), Err("signed64"));
    assert_eq!(f.sat(i32::MAX as i64 + 1), i32::MAX as i64);
    assert_eq!(f.sat(i32::MIN as i64 - 1), i32::MIN as i64);
    assert_eq!(f.product(i64::MAX, 2), Err("signed64"));
    assert_eq!(f.div(1, -1), Err("positive_denominator"));
    assert_eq!(f.mul(UNIT, UNIT).unwrap(), UNIT);
}

#[test]
fn local_support_and_dense_trigger() {
    let mut f = Fixed::default();
    let knots = vec![
        0,
        0,
        0,
        0,
        UNIT / 9,
        2 * UNIT / 9,
        3 * UNIT / 9,
        4 * UNIT / 9,
        5 * UNIT / 9,
        6 * UNIT / 9,
        7 * UNIT / 9,
        8 * UNIT / 9,
        UNIT,
        UNIT,
        UNIT,
        UNIT,
    ];
    for x in [0, UNIT / 2, UNIT] {
        let (ids, weights) = basis(&mut f, &knots, &[x; 9]).unwrap();
        assert!(ids.len() <= 37);
        assert_eq!(weights[0], UNIT);
        let p = vec![0; 109];
        assert_eq!(residual(&mut f, &p, UNIT, &ids, &weights).unwrap(), 0);
        let (_, scale, writes) = update(
            &mut f,
            &p,
            UNIT / 256 - 1,
            UNIT,
            UNIT / 2,
            1,
            &ids,
            &weights,
        )
        .unwrap();
        assert!(scale > UNIT / 256);
        assert_eq!(writes, 109);
    }
}

#[test]
fn binding_and_domain_rejections() {
    // These controls enter the same dispatcher that the loaded extension uses.
    Python::with_gil(|_| {
        assert!(fixed_sparse_8719("ratio", "not json").is_err());
        for (op, payload, expected) in [
            ("unknown", "{}", "operation"),
            ("basis", "{}", "supported_domain"),
            ("basis", r#"{"knots":[1,0]}"#, "supported_domain"),
            ("basis", r#"{"knots":[-1]}"#, "supported_domain"),
            ("residual", "{}", "supported_domain"),
            ("update", "{}", "supported_domain"),
            ("serialize", "{}", "supported_domain"),
            ("ratio", r#"{"a":1,"b":0}"#, "positive_denominator"),
            ("mul", r#"{"a":9223372036854775807,"b":2}"#, "signed64"),
            ("div", r#"{"a":9223372036854775807,"b":1}"#, "signed64"),
            ("ratio", r#"{"f":{"counts":{}}}"#, "context_contract"),
        ] {
            let out = fixed_sparse_8719(op, payload).unwrap();
            let value: Value = serde_json::from_str(&out).unwrap();
            assert_eq!(value["error"], expected);
        }
        let p = vec![0; 109];
        for (field, operand) in [("label", json!(2)), ("probability", json!(-1))] {
            let mut payload = json!({"parameters":p});
            payload[field] = operand;
            let value: Value =
                serde_json::from_str(&fixed_sparse_8719("update", &payload.to_string()).unwrap())
                    .unwrap();
            assert_eq!(value["error"], "supported_domain");
        }
    });
}
