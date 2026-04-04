use pyo3::prelude::*;
mod stats;
mod info;

#[pyfunction]
fn add(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}

#[pymodule]
fn interlinked(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;

    let stats_mod = PyModule::new(py, "stats")?;
    stats_mod.add_function(wrap_pyfunction!(stats::pearson,  &stats_mod)?)?;
    stats_mod.add_function(wrap_pyfunction!(stats::spearman, &stats_mod)?)?;
    m.add_submodule(&stats_mod)?;

    //let info_mod = PyModule::new(py, "info")?;
    //info_mod.add_function(wrap_pyfunction!(stats::disc_H,  &info_mod)?)?;
    //m.add_submodule(&info_mod)?;

    Ok(())
}

