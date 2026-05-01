use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use rayon::prelude::*;


fn digamma(
    x: f64
) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut result = 0.0_f64;
    let mut z = x;
    while z < 6.0 {
        result -= 1.0 / z;
        z += 1.0;
    }

    let inv_z2 = 1.0 / (z * z);
    result += z.ln()
        - 0.5 / z
        - inv_z2 * (1.0 / 12.0
            - inv_z2 * (1.0 / 120.0
                - inv_z2 / 252.0));
    result
}

fn knn_1d(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| if i == j {0.0} else {(x[i] - x[j]).abs()})
                .collect();
            row.select_nth_unstable_by(k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            row[k].max(1e-7)
        })
        .collect()
}

fn entropy_kl_1d(x: &[f64], k: usize) -> f64 {
    let n = x.len() as f64;
    let log_sum: f64 = knn_1d(x, k).iter().map(|&r| r.ln()).sum();
    -digamma(k as f64) + digamma(n) + std::f64::consts::LN_2 + log_sum / n
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Python-usable scripts
/////////////////////////////////////////////////////////////////////////////////////////

#[pyfunction]
pub fn kl_h(
    x: PyReadonlyArray1<f64>,
    k: usize,
) -> PyResult<f64> {
    let x = x.as_array();
    let n = x.len();

    if n == 0 {
        return Err(PyValueError::new_err("arrays must be non-empty"));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!("k must satisfy 1 ≤ k < n (got k={k}, n={n})")));
    }

    //let x: Vec<f64> = x.as_array().iter().copied().collect();
    let x: Vec<f64> = x.iter().copied().collect();
    let result = entropy_kl_1d(&x, k);
    Ok(result)
}

#[pyfunction]
pub fn ksg_mi(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    k: usize,
    normalize: bool,
) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let n = x_arr.len();

    if n != y_arr.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("arrays must be non-empty"));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!("k must satisfy 1 ≤ k < n (got k={k}, n={n})")));
    }

    let x: Vec<f64> = x_arr.iter().copied().collect();
    let y: Vec<f64> = y_arr.iter().copied().collect();

    let eps: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        (x[i] - x[j]).abs().max((y[i] - y[j]).abs())
                    }
                }).collect();
            row.select_nth_unstable_by(k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            row[k].max(1e-7)
        })
        .collect();

    let sum_dig: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps[i];
            let nx = x.iter().filter(|&&v| (v - x[i]).abs() < e).count() - 1;
            let ny = y.iter().filter(|&&v| (v - y[i]).abs() < e).count() - 1;
            digamma(nx as f64 + 1.0) + digamma(ny as f64 + 1.0)
        })
        .sum();

    let mi_raw = (digamma(k as f64) + digamma(n as f64) - sum_dig / n as f64)
        .max(0.0);

    let mi: f64 = if normalize {
        let h_x = entropy_kl_1d(&x, k);
        let h_y = entropy_kl_1d(&y, k);
        let den = (h_x * h_y).sqrt();

        if !den.is_finite() || den < 1e-10 {
            mi_raw
        } else {
            (mi_raw / den).clamp(0.0, 1.0)
        }
    } else {
        mi_raw
    };

    Ok(mi)
}

