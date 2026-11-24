//! Native matrix-free Arnoldi eigensolver.
//!
//! The expensive operator is used only through [`OperatorBase::apply`]. The
//! eigenproblem for the small projected upper-Hessenberg matrix is solved by
//! RLST's LAPACK layer. No ARPACK library or reverse-communication interface is
//! required.

use num::{Float, One, Zero};

use crate::{
    Inner, InnerProductSpace, Norm, NormedSpace, RlstError, RlstResult, RlstScalar,
    abstract_operator::OperatorBase,
    dense::{array::DynArray, linalg::lapack::eigenvalue_decomposition::EigMode},
    operator::{abstract_operator::Operator, element::Element},
    traits::linalg::{decompositions::EigenvalueDecomposition, lapack::Lapack},
};

/// Which Ritz values should be returned by Arnoldi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArnoldiWhich {
    /// Largest absolute value first.
    LargestMagnitude,
    /// Smallest absolute value first.
    SmallestMagnitude,
    /// Largest real part first.
    LargestReal,
    /// Smallest real part first.
    SmallestReal,
}

/// Configuration for one Arnoldi factorization.
#[derive(Clone, Copy, Debug)]
pub struct ArnoldiOptions<Real> {
    /// Maximum Krylov-space dimension.
    pub max_basis: usize,
    /// Maximum number of Ritz pairs returned.
    pub eigenpairs: usize,
    /// Relative Ritz residual tolerance.
    pub tolerance: Real,
    /// Ritz-value ordering.
    pub which: ArnoldiWhich,
}

/// Configuration for explicitly thick-restarted Arnoldi.
#[derive(Clone, Copy, Debug)]
pub struct ThickRestartArnoldiOptions<Real> {
    /// Maximum number of vectors resident in the active Krylov basis.
    pub max_basis: usize,
    /// Number of unconverged Ritz vectors retained between restart cycles.
    pub restart_keep: usize,
    /// Maximum number of restart cycles after the initial cycle.
    pub max_restarts: usize,
    /// Number of requested eigenpairs.
    pub eigenpairs: usize,
    /// Relative true-residual tolerance.
    pub tolerance: Real,
    /// Ritz-value ordering.
    pub which: ArnoldiWhich,
}

impl<Real: Float> Default for ThickRestartArnoldiOptions<Real> {
    fn default() -> Self {
        Self {
            max_basis: 30,
            restart_keep: 8,
            max_restarts: 20,
            eigenpairs: 6,
            tolerance: Real::from(1.0e-6).unwrap(),
            which: ArnoldiWhich::LargestMagnitude,
        }
    }
}

impl<Real: Float> Default for ArnoldiOptions<Real> {
    fn default() -> Self {
        Self {
            max_basis: 30,
            eigenpairs: 6,
            tolerance: Real::from(1.0e-6).unwrap(),
            which: ArnoldiWhich::LargestMagnitude,
        }
    }
}

/// One Arnoldi Ritz pair and its true operator residual.
pub struct ArnoldiRitzPair<'a, Scalar, Space>
where
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
{
    /// Ritz value.
    pub value: Scalar,
    /// Normalized Ritz vector.
    pub vector: Element<'a, Space>,
    /// `||A v - lambda v||_2`.
    pub residual: Scalar::Real,
    /// Whether the requested relative tolerance was attained.
    pub converged: bool,
}

/// Result of a native Arnoldi factorization.
pub struct ArnoldiResult<'a, Scalar, Space>
where
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
{
    /// Requested Ritz pairs in the selected order.
    pub pairs: Vec<ArnoldiRitzPair<'a, Scalar, Space>>,
    /// Number of operator applications.
    pub operator_applications: usize,
    /// Dimension reached by the Krylov basis.
    pub basis_dimension: usize,
    /// Whether Arnoldi terminated because the Krylov space became invariant.
    pub invariant_subspace: bool,
    /// Number of completed restart operations.
    pub restart_count: usize,
    /// Number of converged pairs preserved by locking.
    pub locked_pairs: usize,
}

/// Compute Ritz pairs of a general complex operator with native Arnoldi.
///
/// General real matrices can have complex eigenvectors that cannot be stored in
/// their real domain space. Consequently this interface deliberately requires
/// a complex scalar (`Scalar::Complex = Scalar`). A real operator can be
/// represented in a complex space without changing its mathematical action.
/// Modified Gram--Schmidt with a second orthogonalization pass is used.
pub fn arnoldi<'a, Scalar, Space, OpImpl>(
    operator: &'a Operator<OpImpl>,
    initial: &'a Element<'a, Space>,
    options: ArnoldiOptions<Scalar::Real>,
) -> RlstResult<ArnoldiResult<'a, Scalar, Space>>
where
    Scalar: RlstScalar<Complex = Scalar> + Lapack,
    Space: InnerProductSpace<F = Scalar> + NormedSpace<Output = Scalar::Real>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
{
    if options.max_basis == 0 || options.eigenpairs == 0 {
        return Err(RlstError::GeneralError(
            "Arnoldi requires nonzero max_basis and eigenpairs".into(),
        ));
    }
    if options.tolerance < Scalar::Real::zero() || !options.tolerance.is_finite() {
        return Err(RlstError::GeneralError(
            "Arnoldi tolerance must be finite and nonnegative".into(),
        ));
    }
    let initial_norm = initial.norm();
    if initial_norm == Scalar::Real::zero() || !initial_norm.is_finite() {
        return Err(RlstError::GeneralError(
            "Arnoldi initial vector must have finite nonzero norm".into(),
        ));
    }

    let mut basis = Vec::with_capacity(options.max_basis + 1);
    let mut first = initial.clone();
    first *= Scalar::from_real(Scalar::Real::one() / initial_norm);
    basis.push(first);
    let mut hessenberg =
        DynArray::<Scalar, 2>::from_shape([options.max_basis + 1, options.max_basis]);
    let breakdown = Scalar::Real::epsilon() * Scalar::real(options.max_basis.max(10));
    let mut columns = 0;
    let mut invariant_subspace = false;

    for column in 0..options.max_basis {
        let mut work = operator.dot(&basis[column]);
        for _ in 0..2 {
            for (row, vector) in basis.iter().take(column + 1).enumerate() {
                let coefficient = work.inner(vector);
                hessenberg[[row, column]] += coefficient;
                operator
                    .domain()
                    .axpy_inplace(&(-coefficient), vector, &mut work);
            }
        }
        let beta = work.norm();
        hessenberg[[column + 1, column]] = Scalar::from_real(beta);
        columns = column + 1;
        if beta <= breakdown || !beta.is_finite() {
            invariant_subspace = true;
            break;
        }
        work *= Scalar::from_real(Scalar::Real::one() / beta);
        basis.push(work);
    }

    let mut projected = DynArray::<Scalar, 2>::from_shape([columns, columns]);
    for column in 0..columns {
        for row in 0..columns {
            projected[[row, column]] = hessenberg[[row, column]];
        }
    }
    let (values, right_vectors, _) = projected.eig(EigMode::RightEigenvectors)?;
    let right_vectors = right_vectors.expect("right eigenvectors were requested");
    let mut order: Vec<usize> = (0..columns).collect();
    order.sort_by(|&left, &right| {
        let lhs = values[[left]];
        let rhs = values[[right]];
        let ordering = match options.which {
            ArnoldiWhich::LargestMagnitude | ArnoldiWhich::SmallestMagnitude => {
                lhs.abs().partial_cmp(&rhs.abs())
            }
            ArnoldiWhich::LargestReal | ArnoldiWhich::SmallestReal => {
                lhs.re().partial_cmp(&rhs.re())
            }
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        match options.which {
            ArnoldiWhich::LargestMagnitude | ArnoldiWhich::LargestReal => ordering.reverse(),
            ArnoldiWhich::SmallestMagnitude | ArnoldiWhich::SmallestReal => ordering,
        }
    });

    let mut pairs = Vec::with_capacity(options.eigenpairs.min(columns));
    for index in order.into_iter().take(options.eigenpairs.min(columns)) {
        let mut vector = operator.domain().zero();
        for row in 0..columns {
            operator
                .domain()
                .axpy_inplace(&right_vectors[[row, index]], &basis[row], &mut vector);
        }
        let vector_norm = vector.norm();
        if vector_norm != Scalar::Real::zero() {
            vector *= Scalar::from_real(Scalar::Real::one() / vector_norm);
        }
        let value = values[[index]];
        let mut residual_vector = operator.dot(&vector);
        operator
            .domain()
            .axpy_inplace(&(-value), &vector, &mut residual_vector);
        let residual = residual_vector.norm();
        let scale = Scalar::Real::one().max(value.abs());
        pairs.push(ArnoldiRitzPair {
            value,
            vector,
            residual,
            converged: residual <= options.tolerance * scale,
        });
    }

    let returned_pairs = pairs.len();
    Ok(ArnoldiResult {
        pairs,
        operator_applications: columns + returned_pairs,
        basis_dimension: columns,
        invariant_subspace,
        restart_count: 0,
        locked_pairs: 0,
    })
}

fn arnoldi_order<Scalar: RlstScalar>(
    values: &DynArray<Scalar, 1>,
    which: ArnoldiWhich,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.shape()[0]).collect();
    order.sort_by(|&left, &right| {
        let lhs = values[[left]];
        let rhs = values[[right]];
        let ordering = match which {
            ArnoldiWhich::LargestMagnitude | ArnoldiWhich::SmallestMagnitude => {
                lhs.abs().partial_cmp(&rhs.abs())
            }
            ArnoldiWhich::LargestReal | ArnoldiWhich::SmallestReal => {
                lhs.re().partial_cmp(&rhs.re())
            }
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        match which {
            ArnoldiWhich::LargestMagnitude | ArnoldiWhich::LargestReal => ordering.reverse(),
            ArnoldiWhich::SmallestMagnitude | ArnoldiWhich::SmallestReal => ordering,
        }
    });
    order
}

/// Compute general complex eigenpairs using bounded-memory thick restarts.
///
/// At a restart the best unconverged Ritz vectors are retained as an
/// orthonormal block. Converged Ritz pairs are locked and excluded from later
/// search spaces. Expansion then applies the operator successively to the
/// retained block and its descendants. This is an explicit thick-restart
/// Rayleigh--Ritz formulation: unlike ARPACK it does not require an implicit QR
/// bulge chase or a reverse-communication ABI.
pub fn thick_restarted_arnoldi<'a, Scalar, Space, OpImpl>(
    operator: &'a Operator<OpImpl>,
    initial: &'a Element<'a, Space>,
    options: ThickRestartArnoldiOptions<Scalar::Real>,
) -> RlstResult<ArnoldiResult<'a, Scalar, Space>>
where
    Scalar: RlstScalar<Complex = Scalar> + Lapack,
    Space: InnerProductSpace<F = Scalar> + NormedSpace<Output = Scalar::Real>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
{
    if options.eigenpairs == 0
        || options.restart_keep == 0
        || options.max_basis <= options.restart_keep
        || options.restart_keep < options.eigenpairs
    {
        return Err(RlstError::GeneralError(
            "thick-restarted Arnoldi requires 0 < eigenpairs <= restart_keep < max_basis".into(),
        ));
    }
    if options.tolerance < Scalar::Real::zero() || !options.tolerance.is_finite() {
        return Err(RlstError::GeneralError(
            "Arnoldi tolerance must be finite and nonnegative".into(),
        ));
    }
    let initial_norm = initial.norm();
    if initial_norm == Scalar::Real::zero() || !initial_norm.is_finite() {
        return Err(RlstError::GeneralError(
            "Arnoldi initial vector must have finite nonzero norm".into(),
        ));
    }

    let breakdown = Scalar::Real::epsilon() * Scalar::real(options.max_basis.max(10));
    let mut first = initial.clone();
    first *= Scalar::from_real(Scalar::Real::one() / initial_norm);
    let mut basis = vec![first];
    let mut locked: Vec<ArnoldiRitzPair<'a, Scalar, Space>> = Vec::new();
    let mut operator_applications = 0;
    let mut restarts = 0;
    let mut invariant_subspace = false;
    let mut final_candidates = Vec::new();

    for cycle in 0..=options.max_restarts {
        let mut source = 0;
        while basis.len() < options.max_basis && source < basis.len() {
            let mut work = operator.dot(&basis[source]);
            operator_applications += 1;
            for _ in 0..2 {
                for pair in &locked {
                    let coefficient = work.inner(&pair.vector);
                    operator
                        .domain()
                        .axpy_inplace(&(-coefficient), &pair.vector, &mut work);
                }
                for vector in &basis {
                    let coefficient = work.inner(vector);
                    operator
                        .domain()
                        .axpy_inplace(&(-coefficient), vector, &mut work);
                }
            }
            let norm = work.norm();
            if norm > breakdown && norm.is_finite() {
                work *= Scalar::from_real(Scalar::Real::one() / norm);
                basis.push(work);
            }
            source += 1;
        }
        invariant_subspace = basis.len() < options.max_basis;

        let dimension = basis.len();
        let mut images = Vec::with_capacity(dimension);
        for vector in &basis {
            images.push(operator.dot(vector));
            operator_applications += 1;
        }
        let mut projected = DynArray::<Scalar, 2>::from_shape([dimension, dimension]);
        for column in 0..dimension {
            for row in 0..dimension {
                projected[[row, column]] = images[column].inner(&basis[row]);
            }
        }
        let (values, right_vectors, _) = projected.eig(EigMode::RightEigenvectors)?;
        let right_vectors = right_vectors.expect("right eigenvectors were requested");
        let order = arnoldi_order(&values, options.which);
        let mut candidates = Vec::with_capacity(dimension);
        for index in order {
            let mut vector = operator.domain().zero();
            for row in 0..dimension {
                operator.domain().axpy_inplace(
                    &right_vectors[[row, index]],
                    &basis[row],
                    &mut vector,
                );
            }
            let vector_norm = vector.norm();
            if vector_norm <= breakdown || !vector_norm.is_finite() {
                continue;
            }
            vector *= Scalar::from_real(Scalar::Real::one() / vector_norm);
            let value = values[[index]];
            let mut residual_vector = operator.dot(&vector);
            operator_applications += 1;
            operator
                .domain()
                .axpy_inplace(&(-value), &vector, &mut residual_vector);
            let residual = residual_vector.norm();
            let converged = residual <= options.tolerance * Scalar::Real::one().max(value.abs());
            candidates.push(ArnoldiRitzPair {
                value,
                vector,
                residual,
                converged,
            });
        }

        let mut remaining = Vec::new();
        for candidate in candidates {
            if candidate.converged && locked.len() < options.eigenpairs {
                let duplicate = locked.iter().any(|pair| {
                    candidate.vector.inner(&pair.vector).abs()
                        > Scalar::Real::one() - Scalar::real(1.0e-4)
                });
                if !duplicate {
                    locked.push(candidate);
                    continue;
                }
            }
            remaining.push(candidate);
        }
        if locked.len() >= options.eigenpairs || cycle == options.max_restarts {
            final_candidates = remaining;
            break;
        }

        let mut retained = Vec::with_capacity(options.restart_keep);
        for candidate in remaining.iter().take(options.restart_keep) {
            let mut vector = candidate.vector.clone();
            for _ in 0..2 {
                for pair in &locked {
                    let coefficient = vector.inner(&pair.vector);
                    operator
                        .domain()
                        .axpy_inplace(&(-coefficient), &pair.vector, &mut vector);
                }
                for previous in &retained {
                    let coefficient = vector.inner(previous);
                    operator
                        .domain()
                        .axpy_inplace(&(-coefficient), previous, &mut vector);
                }
            }
            let norm = vector.norm();
            if norm > breakdown && norm.is_finite() {
                vector *= Scalar::from_real(Scalar::Real::one() / norm);
                retained.push(vector);
            }
        }
        if retained.is_empty() {
            final_candidates = remaining;
            invariant_subspace = true;
            break;
        }
        basis = retained;
        restarts += 1;
    }

    let locked_pairs = locked.len();
    let mut pairs = locked;
    pairs.extend(
        final_candidates
            .into_iter()
            .take(options.eigenpairs.saturating_sub(pairs.len())),
    );
    Ok(ArnoldiResult {
        pairs,
        operator_applications,
        basis_dimension: options.max_basis,
        invariant_subspace,
        restart_count: restarts,
        locked_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ArnoldiOptions, ArnoldiWhich, ThickRestartArnoldiOptions, arnoldi, thick_restarted_arnoldi,
    };
    use crate::{
        LinearSpace,
        abstract_operator::OperatorBase,
        base_types::{c32, c64},
        dense::array::DynArray,
        operator::abstract_operator::Operator,
    };

    macro_rules! arnoldi_complex_test {
        ($name:ident, $scalar:ty, $real:ty, $tol:expr) => {
            #[test]
            fn $name() {
                let mut matrix = DynArray::<$scalar, 2>::from_shape([4, 4]);
                matrix[[0, 0]] = <$scalar>::new(4.0, 1.0);
                matrix[[1, 1]] = <$scalar>::new(-2.0, 0.5);
                matrix[[2, 2]] = <$scalar>::new(1.0, -0.25);
                matrix[[3, 3]] = <$scalar>::new(0.2, 0.1);
                matrix[[0, 1]] = <$scalar>::new(0.3, -0.1);
                matrix[[1, 2]] = <$scalar>::new(-0.2, 0.4);
                let operator = Operator::from(&matrix);
                let mut initial = operator.domain().zero();
                for (entry, value) in initial.imp_mut().iter_mut().zip(1..=4) {
                    *entry = <$scalar>::new(value as $real, (5 - value) as $real * (0.1 as $real));
                }
                let result = arnoldi(
                    &operator,
                    &initial,
                    ArnoldiOptions {
                        max_basis: 4,
                        eigenpairs: 2,
                        tolerance: $tol,
                        which: ArnoldiWhich::LargestMagnitude,
                    },
                )
                .unwrap();
                assert_eq!(result.basis_dimension, 4);
                assert_eq!(result.pairs.len(), 2);
                assert!((result.pairs[0].value - <$scalar>::new(4.0, 1.0)).norm() < $tol * 10.0);
                assert!(result.pairs.iter().all(|pair| pair.residual < $tol * 20.0));
            }
        };
    }

    arnoldi_complex_test!(general_c32, c32, f32, 2.0e-5_f32);
    arnoldi_complex_test!(general_c64, c64, f64, 2.0e-12_f64);

    #[test]
    fn thick_restart_converges_beyond_one_krylov_window() {
        let dimension = 24;
        let mut matrix = DynArray::<c64, 2>::from_shape([dimension, dimension]);
        for index in 0..dimension {
            let value = if index < 3 {
                20.0 - 1.0e-3 * index as f64
            } else {
                18.0 - index as f64 * 0.7
            };
            matrix[[index, index]] = c64::new(value, 0.05 * index as f64);
        }
        let operator = Operator::from(&matrix);
        let mut initial = operator.domain().zero();
        for (index, entry) in initial.imp_mut().iter_mut().enumerate() {
            *entry = c64::new(1.0 + index as f64 * 0.01, 0.2);
        }
        let result = thick_restarted_arnoldi(
            &operator,
            &initial,
            ThickRestartArnoldiOptions {
                max_basis: 8,
                restart_keep: 3,
                max_restarts: 40,
                eigenpairs: 2,
                tolerance: 1.0e-7,
                which: ArnoldiWhich::LargestMagnitude,
            },
        )
        .unwrap();
        assert!(result.restart_count > 0);
        assert!(result.locked_pairs > 0);
        assert!(result.basis_dimension <= 8);
        assert_eq!(result.pairs.len(), 2);
        assert!(result.pairs.iter().all(|pair| pair.converged));
        assert!(result.pairs.iter().all(|pair| pair.residual < 3.0e-6));
    }
}
