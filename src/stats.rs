use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use statrs::distribution::{ContinuousCDF, StudentsT, ChiSquared};

fn rank(x: &[f64]) -> Vec<f64> {
    let mut pairs: Vec<(usize, f64)> = x.iter().cloned().enumerate().collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;

    while i < pairs.len() {
        let start = i;
        let mut end = i;

        while end + 1 < pairs.len() && pairs[end].1 == pairs[end + 1].1 {
            end += 1;
        }

        let rank_mean = (start + end) as f64 / 2.0 + 1.0;

        for j in start..=end {
            ranks[pairs[j].0] = rank_mean;
        }

        i = end + 1;
    }

    ranks
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Python-usable scripts
/////////////////////////////////////////////////////////////////////////////////////////

#[pyfunction]
pub fn pearson_corr(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }

    let n = x.len() as f64;
    if n < 3.0 {
        return Err(PyValueError::new_err("array must contain >= 3 elements"));
    }

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

#[pyfunction]
pub fn spearman_corr(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }

    let n = x.len() as f64;
    if n < 3.0 {
        return Err(PyValueError::new_err("array must contain >= 3 elements"));
    }

    let rx = rank(x);
    let ry = rank(y);

    let nf = n as f64;
    let rx_mean = rx.iter().sum::<f64>() / nf;
    let ry_mean = ry.iter().sum::<f64>() / nf;

    let mut num = 0.0;
    let mut x_den = 0.0;
    let mut y_den = 0.0;

    for (&xi, &yi) in rx.iter().zip(ry.iter()) {
        let dx = xi - rx_mean;
        let dy = yi - ry_mean;
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

#[pyfunction]
pub fn phi_coef(x: PyReadonlyArray1<bool>, y: PyReadonlyArray1<bool>) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }

    let n = x.len() as f64;
    if n < 3.0 {
        return Err(PyValueError::new_err("array must contain >= 3 elements"));
    }

    let mut f11: u64 = 0;
    let mut f00: u64 = 0;
    let mut f10: u64 = 0;
    let mut f01: u64 = 0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        match (xi, yi) {
            (true,  true)  => f11 += 1,
            (false, false) => f00 += 1,
            (true,  false) => f10 += 1,
            (false, true)  => f01 += 1,
        }
    }

    let (f11, f00, f10, f01) = (
        f11 as f64,
        f00 as f64,
        f10 as f64,
        f01 as f64
    );
    
    let f1_ = f11 + f10;
    let f0_ = f00 + f01;
    let f_1 = f11 + f01;
    let f_0 = f00 + f10;

    let den = (f1_*f0_ * f_1*f_0).sqrt();
    if den == 0.0 {
        return Err(PyValueError::new_err("phi is undefined: 1+ marginal totals are zero"))};

    let phi = (f11*f00 - f01*f10) / den;
    let chi2_dist = ChiSquared::new(1.0)
        .map_err(|e| PyValueError::new_err(format!("failed to build Chi Squared distribution {e}")))?;

    let chi2_stat = n * phi * phi;
    let p = 1.0 - chi2_dist.cdf(chi2_stat);
    Ok((phi, p))
}

