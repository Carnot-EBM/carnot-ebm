use pyo3::prelude::*;
mod nine_input_8533;
#[pymodule]
fn _nine_input_8533(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nine_input_8533::register(m)
}
