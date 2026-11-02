use pyo3::prelude::*;
#[path = "/home/ianblenke/github.com/ianblenke/carnot/crates/carnot-python/src/spline_fixed_8719.rs"]
mod arithmetic;
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    arithmetic::register(m)
}
