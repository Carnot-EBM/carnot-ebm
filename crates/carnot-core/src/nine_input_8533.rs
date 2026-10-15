//! REQ-VERIFY-8533: nine scaled inputs and 109 effective spline coefficients.
//! Python owns scaling, offset, temperature, normalization and decisions.

const KNOTS: [f64; 16] = [
    0.0,
    0.0,
    0.0,
    0.0,
    1.0 / 9.0,
    2.0 / 9.0,
    3.0 / 9.0,
    4.0 / 9.0,
    5.0 / 9.0,
    6.0 / 9.0,
    7.0 / 9.0,
    8.0 / 9.0,
    1.0,
    1.0,
    1.0,
    1.0,
];

/// Evaluate the clamped cubic recurrence without fused multiply/add operations.
fn basis(value: f64) -> [f64; 15] {
    let mut current = [0.0; 15];
    if value == 1.0 {
        current[11] = 1.0;
        return current;
    }
    for i in 0..15 {
        current[i] = f64::from(KNOTS[i] <= value && value < KNOTS[i + 1]);
    }
    for degree in 1..=3 {
        let mut next = [0.0; 15];
        for i in 0..15 - degree {
            let left_width = KNOTS[i + degree] - KNOTS[i];
            let right_width = KNOTS[i + degree + 1] - KNOTS[i + 1];
            let left = if left_width == 0.0 {
                0.0
            } else {
                ((value - KNOTS[i]) / left_width) * current[i]
            };
            let right = if right_width == 0.0 {
                0.0
            } else {
                ((KNOTS[i + degree + 1] - value) / right_width) * current[i + 1]
            };
            next[i] = left + right;
        }
        current = next;
    }
    current
}

/// Return only the residual, with explicit operand and finite-output validation.
pub fn residuals(c: &[f64], rows: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {
    if c.len() != 109 || c.iter().any(|v| !v.is_finite()) {
        return Err("coefficients");
    }
    if rows.is_empty() {
        return Err("features");
    }
    let mut output = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 9
            || row
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("features");
        }
        let mut residual = 0.0;
        for dimension in 0..9 {
            let weights = basis(row[dimension]);
            for i in 0..12 {
                let product = weights[i] * c[dimension * 12 + i];
                residual += product;
            }
        }
        residual += c[108];
        if !residual.is_finite() {
            return Err("residual");
        }
        output.push(residual);
    }
    Ok(output)
}
