use std::f64::consts::LN_2;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use rayon::prelude::*;
use numpy::PyReadonlyArray1;

use kiddo::KdTree;
use kiddo::distance_metric::DistanceMetric;


#[inline(always)]
fn digamma(x: f64) -> f64 {
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

struct Chebyshev;
impl DistanceMetric<f64, 1> for Chebyshev {
    #[inline(always)]
    fn dist(a: &[f64; 1], b: &[f64; 1]) -> f64 {
        (a[0] - b[0]).abs()
    }
    #[inline(always)]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}
impl DistanceMetric<f64, 2> for Chebyshev {
    #[inline(always)]
    fn dist(a: &[f64; 2], b: &[f64; 2]) -> f64 {
        (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
    }
    #[inline(always)]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}
impl DistanceMetric<f64, 3> for Chebyshev {
    #[inline(always)]
    fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        (a[0] - b[0]).abs()
            .max((a[1] - b[1]).abs())
            .max((a[2] - b[2]).abs())
    }
    #[inline(always)]
    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}

#[inline(always)]
fn count_within_1d(sorted: &[f64], query: f64, eps: f64) -> usize {
    let lo = sorted.partition_point(|&v| v <= query - eps);
    let hi = sorted.partition_point(|&v| v <  query + eps);
    (hi - lo).saturating_sub(1)
}

fn knn_eps_2d(tree: &KdTree<f64, 2>, x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| {
            tree.nearest_n::<Chebyshev>(&[xi, yi], k + 1)
                .last()
                .map(|nb| nb.distance)
                .unwrap_or(0.0)
                .max(1e-7)
        })
        .collect()
}

fn knn_eps_3d(tree: &KdTree<f64, 3>, x: &[f64], y: &[f64], z: &[f64], k: usize) -> Vec<f64> {
    x.par_iter()
        .zip(y.par_iter())
        .zip(z.par_iter())
        .map(|((&xi, &yi), &zi)| {
            tree.nearest_n::<Chebyshev>(&[xi, yi, zi], k + 1)
                .last()
                .map(|nb| nb.distance)
                .unwrap_or(0.0)
                .max(1e-7)
        })
        .collect()
}

fn entropy_kl_1d_tree(tree: &KdTree<f64, 1>, x: &[f64], k: usize) -> f64 {
    let n = x.len() as f64;
    let log_sum: f64 = x.iter()
        .map(|&xi| {
            tree.nearest_n::<Chebyshev>(&[xi], k + 1)
                .last()
                .map(|nb| nb.distance.max(1e-7).ln())
                .unwrap_or(f64::NEG_INFINITY)
        })
        .sum();
    -digamma(k as f64) + digamma(n) + LN_2 + log_sum / n
}

fn entropy_kl_2d_tree(tree: &KdTree<f64, 2>, x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len() as f64;
    let log_sum: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            tree.nearest_n::<Chebyshev>(&[xi, yi], k + 1)
                .last()
                .map(|nb| nb.distance.max(1e-7).ln())
                .unwrap_or(f64::NEG_INFINITY)
        })
        .sum();
    -digamma(k as f64) + digamma(n) + (2.0 * LN_2) + (2.0 * log_sum / n)
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

    let x: Vec<f64> = x.iter().copied().collect();

    let mut tree: KdTree<f64, 1> = KdTree::with_capacity(n);
    for (i, &xi) in x.iter().enumerate() {
        tree.add(&[xi], i as u64);
    }

    Ok(entropy_kl_1d_tree(&tree, &x, k))
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

    let mut tree_xy: KdTree<f64, 2> = KdTree::with_capacity(n);
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        tree_xy.add(&[xi, yi], i as u64);
    }

    let mut x_sort = x.clone();
    let mut y_sort = y.clone();
    x_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let eps = knn_eps_2d(&tree_xy, &x, &y, k);
    let sum_dig: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps[i];
            let n_x = count_within_1d(&x_sort, x[i], e);
            let n_y = count_within_1d(&y_sort, y[i], e);
            digamma(n_x as f64 + 1.0) + digamma(n_y as f64 + 1.0)
        })
        .sum();

    let mi_raw = (digamma(k as f64) + digamma(n as f64) - (sum_dig / n as f64)).max(0.0);
    
    let mi = if normalize {
        let mut tree_x: KdTree<f64, 1> = KdTree::with_capacity(n);
        let mut tree_y: KdTree<f64, 1> = KdTree::with_capacity(n);
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            tree_x.add(&[xi], i as u64);
            tree_y.add(&[yi], i as u64);
        }

        let h_x = entropy_kl_1d_tree(&tree_x, &x, k);
        let h_y = entropy_kl_1d_tree(&tree_y, &y, k);

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
        return Err(PyValueError::new_err("x, y and z must have same length"));
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

    let mut tree_xyz: KdTree<f64, 3> = KdTree::with_capacity(n);
    let mut tree_xz:  KdTree<f64, 2> = KdTree::with_capacity(n);
    let mut tree_yz:  KdTree<f64, 2> = KdTree::with_capacity(n);
    for (i, ((&xi, &yi), &zi)) in x.iter().zip(y.iter()).zip(z.iter()).enumerate() {
        tree_xyz.add(&[xi, yi, zi], i as u64);
        tree_xz.add(&[xi, zi], i as u64);
        tree_yz.add(&[yi, zi], i as u64);
    }

    let mut z_sort = z.clone();
    z_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
      
   let eps = knn_eps_3d(&tree_xyz, &x, &y, &z, k);
   let sum_dig: f64 = (0..n)
       .into_par_iter()
       .map(|i| {
           let e = eps[i];
           let n_xz = tree_xz
               .within_unsorted::<Chebyshev>(&[x[i], z[i]], e)
               .len()
               .saturating_sub(1);
           let n_yz = tree_yz
               .within_unsorted::<Chebyshev>(&[y[i], z[i]], e)
               .len()
               .saturating_sub(1);
           let n_z = count_within_1d(&z_sort, z[i], e);
           digamma(n_z as f64 + 1.0)
               - digamma(n_xz as f64 + 1.0)
               - digamma(n_yz as f64 + 1.0)
       })
       .sum();

    let cmi_raw = (digamma(k as f64) + (sum_dig / n as f64)).max(0.0);

    let cmi = if normalize {
        let mut tree_z: KdTree<f64, 1> = KdTree::with_capacity(n);
        for (i, &zi) in z.iter().enumerate() {
            tree_z.add(&[zi], i as u64);
        }
        let h_xz = entropy_kl_2d_tree(&tree_xz, &x, &z, k);
        let h_yz = entropy_kl_2d_tree(&tree_yz, &y, &z, k);
        let h_z  = entropy_kl_1d_tree(&tree_z,  &z, k);
        
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
        return Err(PyValueError::new_err("x, y and z must have same length"));
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

    let mut tree_xyz: KdTree<f64, 3> = KdTree::with_capacity(n);
    let mut tree_xy:  KdTree<f64, 2> = KdTree::with_capacity(n);
    let mut tree_xz:  KdTree<f64, 2> = KdTree::with_capacity(n);
    let mut tree_yz:  KdTree<f64, 2> = KdTree::with_capacity(n);
    for (i, ((&xi, &yi), &zi)) in x.iter().zip(y.iter()).zip(z.iter()).enumerate() {
        tree_xyz.add(&[xi, yi, zi], i as u64);
        tree_xy.add(&[xi, yi],      i as u64);
        tree_xz.add(&[xi, zi],      i as u64);
        tree_yz.add(&[yi, zi],      i as u64);
    }

    let mut x_sort = x.clone();
    let mut y_sort = y.clone();
    let mut z_sort = z.clone();
    x_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    z_sort.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let eps_xy = knn_eps_2d(&tree_xy, &x, &y, k);
    let sum_dig_mi: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps_xy[i];
            let n_x = count_within_1d(&x_sort, x[i], e);
            let n_y = count_within_1d(&y_sort, y[i], e);
            digamma(n_x as f64 + 1.0) + digamma(n_y as f64 + 1.0)
        })
        .sum();
    let mi = digamma(k as f64) + digamma(n as f64) - (sum_dig_mi / n as f64);

    let eps_xyz = knn_eps_3d(&tree_xyz, &x, &y, &z, k);
    let sum_dig_cmi: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let e = eps_xyz[i];
            let n_xz = tree_xz
                .within_unsorted::<Chebyshev>(&[x[i], z[i]], e)
                .len()
                .saturating_sub(1);
            let n_yz = tree_yz
                .within_unsorted::<Chebyshev>(&[y[i], z[i]], e)
                .len()
                .saturating_sub(1);
            let n_z = count_within_1d(&z_sort, z[i], e);
            digamma(n_z as f64 + 1.0)
                - digamma(n_xz as f64 + 1.0)
                - digamma(n_yz as f64 + 1.0)
        })
        .sum();
    let cmi = digamma(k as f64) + (sum_dig_cmi / n as f64);

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



