#![cfg(any(
    not(target_os = "macos"),
    all(target_os = "macos", feature = "macos-legacy-arpack")
))]

use rlst::dense::linalg::naupd::ArnoldiBackend;
use rlst::operator::operations::eigenvalues::eigs::Which;
use rlst::prelude::*;
use rlst::{Eigs, Operator};

fn sort_by_abs_desc(values: &mut [num::Complex<f64>]) {
    values.sort_by(|lhs, rhs| {
        rhs.norm()
            .partial_cmp(&lhs.norm())
            .unwrap()
            .then(rhs.re.partial_cmp(&lhs.re).unwrap())
            .then(rhs.im.partial_cmp(&lhs.im).unwrap())
    });
}

fn residual_norm(matrix: &[[f64; 5]; 5], eigenvalue: f64, eigenvector: &[f64]) -> f64 {
    let mut accum = 0.0;
    for row in 0..5 {
        let mut value = 0.0;
        for col in 0..5 {
            value += matrix[row][col] * eigenvector[col];
        }
        value -= eigenvalue * eigenvector[row];
        accum += value * value;
    }
    accum.sqrt()
}

#[test]
fn test_compare_native_and_legacy_eigs_on_dense_real_matrix() {
    let entries = [
        [4.0, 1.2, -0.7, 0.4, 1.1],
        [1.2, -1.5, 0.9, 0.3, -0.8],
        [-0.7, 0.9, 3.2, -1.1, 0.5],
        [0.4, 0.3, -1.1, 2.7, 1.4],
        [1.1, -0.8, 0.5, 1.4, -2.0],
    ];

    let mut mat_native = rlst_dynamic_array2!(f64, [5, 5]);
    let mut mat_legacy = rlst_dynamic_array2!(f64, [5, 5]);

    for row in 0..5 {
        for col in 0..5 {
            mat_native[[row, col]] = entries[row][col];
            mat_legacy[[row, col]] = entries[row][col];
        }
    }

    let mut eigs_native = Eigs::new(
        Operator::from(mat_native),
        1e-10,
        Some(80),
        None,
        Some(Which::LM),
    );
    let mut eigs_legacy = Eigs::new(
        Operator::from(mat_legacy),
        1e-10,
        Some(80),
        None,
        Some(Which::LM),
    );

    let (mut vals_native, vecs_native) =
        eigs_native.run_with_backend(None, 3, None, true, ArnoldiBackend::Native);
    let (mut vals_legacy, vecs_legacy) =
        eigs_legacy.run_with_backend(None, 3, None, true, ArnoldiBackend::Legacy);

    for col in 0..3 {
        assert!(
            vals_native[col].im.abs() < 1e-8,
            "native backend returned a complex eigenvalue for the test matrix"
        );
        assert!(
            vals_legacy[col].im.abs() < 1e-8,
            "legacy backend returned a complex eigenvalue for the test matrix"
        );

        let native_vec = &vecs_native[col * 5..(col + 1) * 5];
        let legacy_vec = &vecs_legacy[col * 5..(col + 1) * 5];

        assert!(
            residual_norm(&entries, vals_native[col].re, native_vec) < 1e-6,
            "native backend returned an inaccurate eigenpair"
        );
        assert!(
            residual_norm(&entries, vals_legacy[col].re, legacy_vec) < 1e-6,
            "legacy backend returned an inaccurate eigenpair"
        );
    }

    sort_by_abs_desc(&mut vals_native);
    sort_by_abs_desc(&mut vals_legacy);

    for (native, legacy) in vals_native.iter().zip(vals_legacy.iter()) {
        approx::assert_relative_eq!(native.re, legacy.re, epsilon = 1e-7);
        approx::assert_relative_eq!(native.im, legacy.im, epsilon = 1e-7);
    }
}
