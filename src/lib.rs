use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use statrs::distribution::{ContinuousCDF, StudentsT};

#[pyfunction]
fn add(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}


#[pyfunction]
fn pearson(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }

    let n = x.len() as f64;

    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut num   = 0.0;
    let mut x_den = 0.0;
    let mut y_den = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        num += dx * dy;
        x_den += dx * dx;
        y_den += dy * dy;
    }

    if x_den == 0.0 || y_den == 0.0 {
        return Err(PyValueError::new_err("arrays must have nonzero variance"));
    }

    let r = num / (x_den.sqrt() * y_den.sqrt());

    let df = n - 2.0;
    let t = r * (df / (1.0 - r * r)).sqrt();

    let st = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| PyValueError::new_err(format!("failed to create t distribution: {e}")))?;
    let p = 2.0 * (1.0 - st.cdf(t.abs()));
    Ok((r, p))
}





#[pymodule]
fn interlinked(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(pearson, m)?)?;
    Ok(())
}

