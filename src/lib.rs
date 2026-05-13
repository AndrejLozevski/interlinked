use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;
mod stats;
mod info;

#[pymodule]
fn interlinked(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    let stats_mod = PyModule::new(py, "stats")?;
    stats_mod.add_function(wrap_pyfunction!(stats::pearson_corr,  &stats_mod)?)?;
    stats_mod.add_function(wrap_pyfunction!(stats::spearman_corr, &stats_mod)?)?;
    stats_mod.add_function(wrap_pyfunction!(stats::phi_coef,      &stats_mod)?)?;
    m.add_submodule(&stats_mod)?;

    let info_mod = PyModule::new(py, "info")?;
    info_mod.add_function(wrap_pyfunction!(info::ksg_mi,  &info_mod)?)?;
    info_mod.add_function(wrap_pyfunction!(info::ksg_cmi, &info_mod)?)?;
    info_mod.add_function(wrap_pyfunction!(info::ksg_ii,  &info_mod)?)?;
    info_mod.add_function(wrap_pyfunction!(info::kl_h,    &info_mod)?)?;
    m.add_submodule(&info_mod)?;

    Ok(())
}

