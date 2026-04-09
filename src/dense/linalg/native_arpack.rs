use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::RlstScalar;
use num::complex::Complex;
use num::traits::{NumAssign, One, Zero};
use num::Float;
use rand::thread_rng;

#[derive(Clone, Copy)]
pub(super) enum WhichCriterion {
    LM,
    SM,
    LR,
    SR,
    LI,
    SI,
}

pub(super) struct ArnoldiState<Item: RlstScalar> {
    pub basis_size: usize,
    pub processed_cols: usize,
    pub max_process_steps: usize,
    pub nev: usize,
    pub tol: <Item as RlstScalar>::Real,
    pub which: WhichCriterion,
    pub hessenberg: Vec<Item>,
}

pub(super) struct SelectedEigenpairs<Real: Float + NumAssign> {
    pub arnoldi_dim: usize,
    pub values: Vec<Complex<Real>>,
    pub vectors: Vec<Complex<Real>>,
}

thread_local! {
    static STATES_F32: RefCell<HashMap<usize, ArnoldiState<f32>>> = RefCell::new(HashMap::new());
    static STATES_F64: RefCell<HashMap<usize, ArnoldiState<f64>>> = RefCell::new(HashMap::new());
    static STATES_C32: RefCell<HashMap<usize, ArnoldiState<Complex<f32>>>> = RefCell::new(HashMap::new());
    static STATES_C64: RefCell<HashMap<usize, ArnoldiState<Complex<f64>>>> = RefCell::new(HashMap::new());
}

pub(super) fn with_states_f32<R>(f: impl FnOnce(&mut HashMap<usize, ArnoldiState<f32>>) -> R) -> R {
    STATES_F32.with(|states| {
        let mut states = states.borrow_mut();
        f(&mut states)
    })
}

pub(super) fn with_states_f64<R>(f: impl FnOnce(&mut HashMap<usize, ArnoldiState<f64>>) -> R) -> R {
    STATES_F64.with(|states| {
        let mut states = states.borrow_mut();
        f(&mut states)
    })
}

pub(super) fn with_states_c32<R>(
    f: impl FnOnce(&mut HashMap<usize, ArnoldiState<Complex<f32>>>) -> R,
) -> R {
    STATES_C32.with(|states| {
        let mut states = states.borrow_mut();
        f(&mut states)
    })
}

pub(super) fn with_states_c64<R>(
    f: impl FnOnce(&mut HashMap<usize, ArnoldiState<Complex<f64>>>) -> R,
) -> R {
    STATES_C64.with(|states| {
        let mut states = states.borrow_mut();
        f(&mut states)
    })
}

pub(super) fn parse_which(which: &str) -> Option<WhichCriterion> {
    Some(match which {
        "LM" => WhichCriterion::LM,
        "SM" => WhichCriterion::SM,
        "LR" => WhichCriterion::LR,
        "SR" => WhichCriterion::SR,
        "LI" => WhichCriterion::LI,
        "SI" => WhichCriterion::SI,
        _ => return None,
    })
}

pub(super) fn naupd_with_state<Item: RlstScalar<Complex = Complex<<Item as RlstScalar>::Real>>>(
    states: &mut HashMap<usize, ArnoldiState<Item>>,
    ido: &mut i32,
    bmat: &str,
    dim: i32,
    which: &str,
    nev: i32,
    tol: <Item as RlstScalar>::Real,
    resid: &mut [Item],
    ncv: i32,
    v: &mut [Item],
    ldv: i32,
    iparam: &mut [i32; 11],
    ipntr: &mut [i32; 14],
    workd: &mut [Item],
    info: &mut i32,
) {
    let key = v.as_ptr() as usize;
    let dim = dim as usize;
    let ldv = ldv as usize;
    let nev = nev as usize;
    let ncv = ncv as usize;

    if *ido == 0 || !states.contains_key(&key) {
        let Some(which) = parse_which(which) else {
            *info = -5;
            *ido = 99;
            return;
        };

        if bmat != "I" {
            *info = -2;
            *ido = 99;
            return;
        }

        if dim == 0 || nev == 0 || ncv == 0 || ldv < dim || workd.len() < 2 * dim {
            *info = -1;
            *ido = 99;
            return;
        }

        let max_process_steps = ncv.min(dim).min(iparam[2].max(1) as usize);
        if max_process_steps == 0 {
            *info = -1;
            *ido = 99;
            return;
        }

        let tol = if tol > num::Zero::zero() {
            tol
        } else {
            machine_tol::<Item::Real>()
        };

        initialise_residual(resid, dim);

        let beta = vector_norm(resid);
        if beta <= tol {
            *info = -1;
            *ido = 99;
            return;
        }

        let beta_item = Item::from_real(beta);
        for elem in resid.iter_mut().take(dim) {
            *elem /= beta_item;
        }

        write_basis_vector(v, ldv, 0, dim, &resid[..dim]);
        workd[..dim].copy_from_slice(&resid[..dim]);
        zero_slice(&mut workd[dim..2 * dim]);

        ipntr.fill(0);
        ipntr[0] = 1;
        ipntr[1] = dim as i32 + 1;
        ipntr[2] = 1;

        states.insert(
            key,
            ArnoldiState {
                basis_size: 1,
                processed_cols: 0,
                max_process_steps,
                nev,
                tol,
                which,
                hessenberg: vec![Item::zero(); (max_process_steps + 1) * max_process_steps],
            },
        );

        *info = 0;
        *ido = -1;
        return;
    }

    let Some(state) = states.get_mut(&key) else {
        *info = -1;
        *ido = 99;
        return;
    };

    if state.processed_cols >= state.max_process_steps {
        *info = 0;
        *ido = 99;
        return;
    }

    let col = state.processed_cols;
    let mut w = workd[dim..2 * dim].to_vec();

    orthogonalize_against_basis(
        &mut w,
        v,
        ldv,
        dim,
        col + 1,
        &mut state.hessenberg,
        state.max_process_steps + 1,
        col,
    );

    let beta = vector_norm(&w);
    let h_index = hessenberg_index(state.max_process_steps + 1, col + 1, col);
    state.hessenberg[h_index] = Item::from_real(beta);
    state.processed_cols += 1;

    let converged = state.processed_cols >= state.nev && count_converged(state) >= state.nev;

    let reached_capacity = state.processed_cols >= state.max_process_steps;
    let breakdown = beta <= state.tol;

    if !(converged || reached_capacity || breakdown) {
        let next_col = state.processed_cols;
        let beta_item = Item::from_real(beta);
        for elem in &mut w {
            *elem /= beta_item;
        }

        write_basis_vector(v, ldv, next_col, dim, &w);
        workd[..dim].copy_from_slice(&w);
        zero_slice(&mut workd[dim..2 * dim]);
        state.basis_size = next_col + 1;

        *info = 0;
        *ido = 1;
        return;
    }

    state.basis_size = state.basis_size.min(state.processed_cols);
    *info = 0;
    *ido = 99;
}

pub(super) fn extract_selected_eigenpairs<
    Item: RlstScalar<Complex = Complex<<Item as RlstScalar>::Real>>,
>(
    states: &mut HashMap<usize, ArnoldiState<Item>>,
    key: usize,
    nev: usize,
) -> Option<SelectedEigenpairs<<Item as RlstScalar>::Real>> {
    let state = states.remove(&key)?;
    let eig = projected_eigendecomposition(&state);
    let order = sorted_indices(&eig.values, state.which);
    let count = nev.min(order.len());
    let m = state.processed_cols;
    let mut values = Vec::with_capacity(count);
    let mut vectors = vec![Complex::zero(); m * count];

    for (out_col, eig_col) in order.into_iter().take(count).enumerate() {
        values.push(eig.values[eig_col]);
        for row in 0..m {
            vectors[m * out_col + row] = eig.vectors[m * eig_col + row];
        }
    }

    Some(SelectedEigenpairs {
        arnoldi_dim: m,
        values,
        vectors,
    })
}

pub(super) fn combine_basis_real<Real: Float + NumAssign>(
    v: &[Real],
    ldv: usize,
    dim: usize,
    m: usize,
    coeffs: &[Complex<Real>],
) -> (Vec<Real>, Vec<Real>) {
    let mut re = vec![Real::zero(); dim];
    let mut im = vec![Real::zero(); dim];

    for col in 0..m {
        let coeff = coeffs[col];
        for row in 0..dim {
            let basis = v[basis_index(ldv, row, col)];
            re[row] += basis * coeff.re;
            im[row] += basis * coeff.im;
        }
    }

    (re, im)
}

pub(super) fn combine_basis_complex<Real: Float + NumAssign>(
    v: &[Complex<Real>],
    ldv: usize,
    dim: usize,
    m: usize,
    coeffs: &[Complex<Real>],
) -> Vec<Complex<Real>> {
    let mut out = vec![Complex::zero(); dim];
    for col in 0..m {
        let coeff = coeffs[col];
        for row in 0..dim {
            out[row] += v[basis_index(ldv, row, col)] * coeff;
        }
    }
    out
}

pub(super) fn approx_eq_complex<Real: Float + NumAssign>(
    lhs: Complex<Real>,
    rhs: Complex<Real>,
    tol: Real,
) -> bool {
    (lhs - rhs).norm() <= tol
}

fn initialise_residual<Item: RlstScalar<Complex = Complex<<Item as RlstScalar>::Real>>>(
    resid: &mut [Item],
    dim: usize,
) {
    let tol = machine_tol::<Item::Real>();
    let norm = vector_norm(&resid[..dim]);

    if norm > tol {
        return;
    }

    let mut rng = thread_rng();
    for elem in resid.iter_mut().take(dim) {
        *elem = Item::rand(&mut rng);
    }
}

fn count_converged<Item: RlstScalar<Complex = Complex<<Item as RlstScalar>::Real>>>(
    state: &ArnoldiState<Item>,
) -> usize {
    let eig = projected_eigendecomposition(state);
    let order = sorted_indices(&eig.values, state.which);
    let beta = if state.processed_cols == 0 {
        Item::Real::zero()
    } else {
        state.hessenberg[hessenberg_index(
            state.max_process_steps + 1,
            state.processed_cols,
            state.processed_cols - 1,
        )]
        .abs()
    };

    order
        .into_iter()
        .take(state.nev)
        .filter(|&col| {
            let last_component = eig.vectors[state.processed_cols * col + state.processed_cols - 1];
            let residual = beta * last_component.norm();
            let value_norm = eig.values[col].norm();
            let scale = if value_norm > Item::Real::one() {
                value_norm
            } else {
                Item::Real::one()
            };
            residual <= state.tol * scale
        })
        .count()
}

struct DenseEigenResult<Real: Float + NumAssign> {
    values: Vec<Complex<Real>>,
    vectors: Vec<Complex<Real>>,
}

fn projected_eigendecomposition<Item: RlstScalar<Complex = Complex<<Item as RlstScalar>::Real>>>(
    state: &ArnoldiState<Item>,
) -> DenseEigenResult<<Item as RlstScalar>::Real> {
    let m = state.processed_cols;
    let mut h = vec![Complex::zero(); m * m];

    for col in 0..m {
        for row in 0..m {
            h[m * col + row] =
                state.hessenberg[hessenberg_index(state.max_process_steps + 1, row, col)].as_c();
        }
    }

    dense_eigendecomposition(h, m, state.tol)
}

fn dense_eigendecomposition<Real: Float + NumAssign>(
    mut a: Vec<Complex<Real>>,
    n: usize,
    tol: Real,
) -> DenseEigenResult<Real> {
    let original = a.clone();
    let eps = tol.max(machine_tol::<Real>());
    let max_iters = 256 * n.max(1);
    let mut active = n;
    let mut iter = 0;

    while active > 1 && iter < max_iters {
        let subdiag = a[matrix_index(n, active - 1, active - 2)].norm();
        let diag_scale = a[matrix_index(n, active - 1, active - 1)].norm()
            + a[matrix_index(n, active - 2, active - 2)].norm()
            + Real::one();

        if subdiag <= eps * diag_scale {
            a[matrix_index(n, active - 1, active - 2)] = Complex::zero();
            active -= 1;
            continue;
        }

        let shift = a[matrix_index(n, active - 1, active - 1)];
        let mut block = vec![Complex::zero(); active * active];

        for col in 0..active {
            for row in 0..active {
                block[matrix_index(active, row, col)] = a[matrix_index(n, row, col)];
            }
            block[matrix_index(active, col, col)] -= shift;
        }

        let (q_sub, r_sub) = qr_decompose(&block, active, eps);
        let mut next_block = matrix_multiply(&r_sub, &q_sub, active);
        for col in 0..active {
            next_block[matrix_index(active, col, col)] += shift;
        }

        for col in 0..active {
            for row in 0..active {
                a[matrix_index(n, row, col)] = next_block[matrix_index(active, row, col)];
            }
        }
        iter += 1;
    }

    let values = (0..n)
        .map(|idx| a[matrix_index(n, idx, idx)])
        .collect::<Vec<_>>();

    let mut vectors = vec![Complex::zero(); n * n];
    for col in 0..n {
        let vector = inverse_iteration(&original, n, values[col], eps);
        let norm = complex_vector_norm(&vector);
        let scale = if norm > eps {
            Complex::new(norm, Real::zero())
        } else {
            Complex::one()
        };
        for row in 0..n {
            vectors[matrix_index(n, row, col)] = vector[row] / scale;
        }
    }

    DenseEigenResult { values, vectors }
}

fn qr_decompose<Real: Float + NumAssign>(
    a: &[Complex<Real>],
    n: usize,
    eps: Real,
) -> (Vec<Complex<Real>>, Vec<Complex<Real>>) {
    let mut q = vec![Complex::zero(); n * n];
    let mut r = vec![Complex::zero(); n * n];
    let mut col = vec![Complex::zero(); n];

    for j in 0..n {
        for row in 0..n {
            col[row] = a[matrix_index(n, row, j)];
        }

        for i in 0..j {
            let coeff = dot_complex_column(&q, n, i, &col);
            r[matrix_index(n, i, j)] = coeff;
            for row in 0..n {
                col[row] -= coeff * q[matrix_index(n, row, i)];
            }
        }

        let norm = complex_vector_norm(&col);
        if norm <= eps {
            orthonormal_fallback(&mut q, n, j, eps);
            continue;
        }

        r[matrix_index(n, j, j)] = Complex::new(norm, Real::zero());
        let scale = Complex::new(norm, Real::zero());
        for row in 0..n {
            q[matrix_index(n, row, j)] = col[row] / scale;
        }
    }

    (q, r)
}

fn orthonormal_fallback<Real: Float + NumAssign>(
    q: &mut [Complex<Real>],
    n: usize,
    col_index: usize,
    eps: Real,
) {
    let mut trial = vec![Complex::zero(); n];

    for offset in 0..n {
        trial.fill(Complex::zero());
        trial[(col_index + offset) % n] = Complex::one();

        for prev in 0..col_index {
            let coeff = dot_complex_column(q, n, prev, &trial);
            for row in 0..n {
                trial[row] -= coeff * q[matrix_index(n, row, prev)];
            }
        }

        let norm = complex_vector_norm(&trial);
        if norm > eps {
            let scale = Complex::new(norm, Real::zero());
            for row in 0..n {
                q[matrix_index(n, row, col_index)] = trial[row] / scale;
            }
            return;
        }
    }
}

fn sorted_indices<Real: Float + NumAssign>(
    values: &[Complex<Real>],
    which: WhichCriterion,
) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|&lhs, &rhs| compare_values(values[lhs], values[rhs], which));
    indices
}

fn compare_values<Real: Float + NumAssign>(
    lhs: Complex<Real>,
    rhs: Complex<Real>,
    which: WhichCriterion,
) -> Ordering {
    let primary = match which {
        WhichCriterion::LM | WhichCriterion::SM => lhs.norm().partial_cmp(&rhs.norm()),
        WhichCriterion::LR | WhichCriterion::SR => lhs.re.partial_cmp(&rhs.re),
        WhichCriterion::LI | WhichCriterion::SI => lhs.im.partial_cmp(&rhs.im),
    }
    .unwrap_or(Ordering::Equal);

    let tie_break = lhs.im.partial_cmp(&rhs.im).unwrap_or(Ordering::Equal);

    match which {
        WhichCriterion::LM | WhichCriterion::LR | WhichCriterion::LI => {
            primary.reverse().then(tie_break.reverse())
        }
        WhichCriterion::SM | WhichCriterion::SR | WhichCriterion::SI => {
            primary.then(tie_break.reverse())
        }
    }
}

fn orthogonalize_against_basis<Item: RlstScalar>(
    w: &mut [Item],
    basis: &[Item],
    ldv: usize,
    dim: usize,
    cols: usize,
    h: &mut [Item],
    h_rows: usize,
    h_col: usize,
) {
    for pass in 0..2 {
        for basis_col in 0..cols {
            let coeff = dot_basis_vector(basis, ldv, dim, basis_col, w);
            let index = hessenberg_index(h_rows, basis_col, h_col);
            if pass == 0 {
                h[index] = coeff;
            } else {
                h[index] += coeff;
            }
            axpy_basis_vector(w, basis, ldv, dim, basis_col, -coeff);
        }
    }
}

fn dot_basis_vector<Item: RlstScalar>(
    basis: &[Item],
    ldv: usize,
    dim: usize,
    basis_col: usize,
    w: &[Item],
) -> Item {
    let mut sum = Item::zero();
    for row in 0..dim {
        sum += basis[basis_index(ldv, row, basis_col)].conj() * w[row];
    }
    sum
}

fn axpy_basis_vector<Item: RlstScalar>(
    w: &mut [Item],
    basis: &[Item],
    ldv: usize,
    dim: usize,
    basis_col: usize,
    alpha: Item,
) {
    for row in 0..dim {
        w[row] += alpha * basis[basis_index(ldv, row, basis_col)];
    }
}

fn write_basis_vector<Item: RlstScalar>(
    basis: &mut [Item],
    ldv: usize,
    basis_col: usize,
    dim: usize,
    data: &[Item],
) {
    for row in 0..dim {
        basis[basis_index(ldv, row, basis_col)] = data[row];
    }
}

fn basis_index(ldv: usize, row: usize, col: usize) -> usize {
    row + ldv * col
}

fn hessenberg_index(rows: usize, row: usize, col: usize) -> usize {
    row + rows * col
}

fn matrix_index(size: usize, row: usize, col: usize) -> usize {
    row + size * col
}

fn zero_slice<Item: RlstScalar>(slice: &mut [Item]) {
    for item in slice.iter_mut() {
        *item = Item::zero();
    }
}

fn vector_norm<Item: RlstScalar>(x: &[Item]) -> <Item as RlstScalar>::Real {
    let mut sum = Item::Real::zero();
    for value in x {
        sum += value.square();
    }
    Float::sqrt(sum)
}

fn dot_complex_column<Real: Float + NumAssign>(
    q: &[Complex<Real>],
    n: usize,
    col: usize,
    vec: &[Complex<Real>],
) -> Complex<Real> {
    let mut sum = Complex::zero();
    for row in 0..n {
        sum += q[matrix_index(n, row, col)].conj() * vec[row];
    }
    sum
}

fn complex_vector_norm<Real: Float + NumAssign>(x: &[Complex<Real>]) -> Real {
    let mut sum = Real::zero();
    for value in x {
        sum += value.norm_sqr();
    }
    Float::sqrt(sum)
}

fn inverse_iteration<Real: Float + NumAssign>(
    a: &[Complex<Real>],
    n: usize,
    lambda: Complex<Real>,
    eps: Real,
) -> Vec<Complex<Real>> {
    let shift = lambda + Complex::new(eps, eps);
    let mut x = vec![Complex::one(); n];

    for idx in 0..n {
        x[idx] = Complex::new(Real::one(), Real::zero());
    }

    for _ in 0..8 {
        let mut shifted = a.to_vec();
        for idx in 0..n {
            shifted[matrix_index(n, idx, idx)] -= shift;
        }

        let y = solve_linear_system(&shifted, n, &x, eps);
        let norm = complex_vector_norm(&y);
        if norm <= eps {
            break;
        }

        let scale = Complex::new(norm, Real::zero());
        for (target, value) in x.iter_mut().zip(y.into_iter()) {
            *target = value / scale;
        }
    }

    x
}

fn solve_linear_system<Real: Float + NumAssign>(
    a: &[Complex<Real>],
    n: usize,
    b: &[Complex<Real>],
    eps: Real,
) -> Vec<Complex<Real>> {
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();

    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_norm = mat[matrix_index(n, pivot_col, pivot_col)].norm();
        for row in pivot_col + 1..n {
            let candidate = mat[matrix_index(n, row, pivot_col)].norm();
            if candidate > pivot_norm {
                pivot_norm = candidate;
                pivot_row = row;
            }
        }

        if pivot_row != pivot_col {
            for col in pivot_col..n {
                mat.swap(
                    matrix_index(n, pivot_col, col),
                    matrix_index(n, pivot_row, col),
                );
            }
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = if mat[matrix_index(n, pivot_col, pivot_col)].norm() <= eps {
            Complex::new(eps, eps)
        } else {
            mat[matrix_index(n, pivot_col, pivot_col)]
        };

        for row in pivot_col + 1..n {
            let factor = mat[matrix_index(n, row, pivot_col)] / pivot;
            let rhs_pivot = rhs[pivot_col];
            for col in pivot_col..n {
                let pivot_entry = mat[matrix_index(n, pivot_col, col)];
                mat[matrix_index(n, row, col)] -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs_pivot;
        }
    }

    let mut x = vec![Complex::zero(); n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for col in row + 1..n {
            sum -= mat[matrix_index(n, row, col)] * x[col];
        }
        let diag = if mat[matrix_index(n, row, row)].norm() <= eps {
            Complex::new(eps, eps)
        } else {
            mat[matrix_index(n, row, row)]
        };
        x[row] = sum / diag;
    }

    x
}

fn matrix_multiply<Real: Float + NumAssign>(
    lhs: &[Complex<Real>],
    rhs: &[Complex<Real>],
    n: usize,
) -> Vec<Complex<Real>> {
    let mut out = vec![Complex::zero(); n * n];
    for col in 0..n {
        for inner in 0..n {
            let rhs_coeff = rhs[matrix_index(n, inner, col)];
            for row in 0..n {
                out[matrix_index(n, row, col)] += lhs[matrix_index(n, row, inner)] * rhs_coeff;
            }
        }
    }
    out
}

fn machine_tol<Real: Float + NumAssign>() -> Real {
    Float::sqrt(Real::epsilon())
}
