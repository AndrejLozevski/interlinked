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
                .map(|j| 
                    if i == j {0.0} 
                    else {(x[i] - x[j]).abs()})
                .collect();
            row.select_nth_unstable_by(k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            row[k].max(1e-7)
        })
        .collect()
}

fn knn_2d(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| 
                    if i == j {0.0} 
                    else {(x[i] - x[j]).abs().max((y[i] - y[j]).abs())})
                .collect();
            row.select_nth_unstable_by(k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            row[k].max(1e-7)
        })
        .collect()
}

fn knn_3d(x: &[f64], y: &[f64], z: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| {
                    if i == j {0.0} 
                    else {
                        (x[i] - x[j]).abs()
                            .max((y[i] - y[j]).abs())
                            .max((z[i] - z[j]).abs())
                    }
                })
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

fn entropy_kl_2d(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len() as f64;
    let log_sum: f64 = knn_2d(x, y, k).iter().map(|&r| r.ln()).sum();
    -digamma(k as f64) + digamma(n) + 2.0 * std::f64::consts::LN_2 + 2.0 * log_sum / n
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
        return Err(PyValueError::new_err("x and y must have same length"));
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
            let n_x = x.iter().filter(|&&v| (v - x[i]).abs() < e).count() - 1;
            let n_y = y.iter().filter(|&&v| (v - y[i]).abs() < e).count() - 1;
            digamma(n_x as f64 + 1.0) + digamma(n_y as f64 + 1.0)
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

#[pyfunction]
pub fn ksg_cmi(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    k: usize,
    normalize: bool,
) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let z_arr = z.as_array();
    let n = x_arr.len();

    if n != y_arr.len() || n != z_arr.len() {
        return Err(PyValueError::new_err("x, y, and z must have same length"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("arrays must be non-empty"));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!("k must satisfy 1 ≤ k < n (got k={k}, n={n})")));
    }
    
    let x: Vec<f64> = x_arr.iter().copied().collect();
    let y: Vec<f64> = y_arr.iter().copied().collect();
    let z: Vec<f64> = z_arr.iter().copied().collect();

    let eps: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        (x[i] - x[j]).abs()
                            .max((y[i] - y[j]).abs())
                            .max((z[i] - z[j]).abs())
                    }
                })
                .collect();
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

            let n_xz = x.iter().zip(z.iter())
                .filter(|&(&xv, &zv)| (xv - x[i]).abs() < e && (zv - z[i]).abs() < e)
                .count() - 1;
            let n_yz = y.iter().zip(z.iter())
                .filter(|&(&yv, &zv)| (yv - y[i]).abs() < e && (zv - z[i]).abs() < e)
                .count() - 1;
            let n_z = z.iter()
                .filter(|&&zv| (zv - z[i]).abs() < e)
                .count() - 1;
            digamma(n_z as f64 + 1.0)
                - digamma(n_xz as f64 + 1.0)
                - digamma(n_yz as f64 + 1.0)
        })
        .sum();

    let cmi_raw = (digamma(k as f64) + sum_dig / n as f64).max(0.0);

    let cmi: f64 = if normalize {
        let h_xz = entropy_kl_2d(&x, &z, k);
        let h_yz = entropy_kl_2d(&y, &z, k);
        let h_z  = entropy_kl_1d(&z, k);

        let h_x_gz = h_xz - h_z;
        let h_y_gz = h_yz - h_z;
        let den = (h_x_gz * h_y_gz).sqrt();

        if !den.is_finite() || den < 1e-10 {
            cmi_raw
        } else {
            (cmi_raw / den).clamp(0.0, 1.0)
        }
    } else {
        cmi_raw
    };

    Ok(cmi)
}

#[pyfunction]
pub fn ksg_ii(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    k: usize,
    normalize: bool,
) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let z_arr = z.as_array();
    let n = x_arr.len();

    if n != y_arr.len() || n != z_arr.len() {
        return Err(PyValueError::new_err("x, y, and z must have same length"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("arrays must be non-empty"));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!("k must satisfy 1 ≤ k < n (got k={k}, n={n})")));
    }
    
    let x: Vec<f64> = x_arr.iter().copied().collect();
    let y: Vec<f64> = y_arr.iter().copied().collect();
    let z: Vec<f64> = z_arr.iter().copied().collect();

    let eps_xyz = knn_3d(&x, &y, &z, k);

    let sum_dig_cmi: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps_xyz[i];

            let n_xz = x.iter().zip(z.iter())
                .filter(|&(&xv, &zv)| (xv - x[i]).abs() < e && (zv - z[i]).abs() < e)
                .count() - 1;
            let n_yz = y.iter().zip(z.iter())
                .filter(|&(&yv, &zv)| (yv - y[i]).abs() < e && (zv - z[i]).abs() < e)
                .count() - 1;
            let n_z = z.iter()
                .filter(|&&zv| (zv - z[i]).abs() < e)
                .count() - 1;
            digamma(n_z as f64 + 1.0)
                - digamma(n_xz as f64 + 1.0)
                - digamma(n_yz as f64 + 1.0)
        })
        .sum();

    let cmi = digamma(k as f64) + sum_dig_cmi / n as f64;

    let eps_xy = knn_2d(&x, &y, k);

    let sum_dig_mi: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps_xy[i];
            let n_x = x.iter().filter(|&&v| (v - x[i]).abs() < e).count() - 1;
            let n_y = y.iter().filter(|&&v| (v - y[i]).abs() < e).count() - 1;
            digamma(n_x as f64 + 1.0) + digamma(n_y as f64 + 1.0)
        })
        .sum();

    let mi = digamma(k as f64) + digamma(n as f64) - sum_dig_mi / n as f64;

    let ii_raw = cmi - mi;

    let ii = if normalize {
        let den = cmi.abs() + mi.abs();
        if den < 1e-10 {
            0.0
        } else {
            ii_raw / den
        }
    } else {
        ii_raw
    };

    Ok(ii)
}

