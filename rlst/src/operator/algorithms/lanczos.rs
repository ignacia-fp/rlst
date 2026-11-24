//! Native matrix-free Lanczos eigensolver for self-adjoint operators.
//!
//! Lanczos uses only operator applications and inner products. The small real
//! symmetric tridiagonal eigenproblem is solved with RLST's LAPACK layer. Full
//! reorthogonalization is used to prevent duplicate (“ghost”) Ritz values.

use num::{Float, One, Zero};

use crate::{
    Inner, InnerProductSpace, Norm, NormedSpace, RlstError, RlstResult, RlstScalar, UpLo,
    abstract_operator::OperatorBase,
    dense::{array::DynArray, linalg::lapack::symmeig::SymmEigMode},
    operator::{abstract_operator::Operator, element::Element},
    traits::linalg::{decompositions::SymmEig, lapack::Lapack},
};

/// Which Hermitian Ritz values should be returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanczosWhich {
    /// Largest algebraic value first.
    LargestAlgebraic,
    /// Smallest algebraic value first.
    SmallestAlgebraic,
    /// Largest magnitude first.
    LargestMagnitude,
    /// Smallest magnitude first.
    SmallestMagnitude,
}

/// Configuration for one Lanczos factorization.
#[derive(Clone, Copy, Debug)]
pub struct LanczosOptions<Real> {
    /// Maximum Krylov-space dimension.
    pub max_basis: usize,
    /// Maximum number of Ritz pairs returned.
    pub eigenpairs: usize,
    /// Relative Ritz residual tolerance.
    pub tolerance: Real,
    /// Ritz-value ordering.
    pub which: LanczosWhich,
}

/// Configuration for bounded-memory thick-restarted Lanczos.
#[derive(Clone, Copy, Debug)]
pub struct ThickRestartLanczosOptions<Real> {
    /// Maximum number of vectors resident in the active search basis.
    pub max_basis: usize,
    /// Number of unconverged Ritz vectors retained at restart.
    pub restart_keep: usize,
    /// Maximum number of restart cycles after the initial cycle.
    pub max_restarts: usize,
    /// Number of requested eigenpairs.
    pub eigenpairs: usize,
    /// Relative true-residual tolerance.
    pub tolerance: Real,
    /// Ritz-value ordering.
    pub which: LanczosWhich,
}

impl<Real: Float> Default for ThickRestartLanczosOptions<Real> {
    fn default() -> Self {
        Self {
            max_basis: 30,
            restart_keep: 8,
            max_restarts: 20,
            eigenpairs: 6,
            tolerance: Real::from(1.0e-6).unwrap(),
            which: LanczosWhich::LargestMagnitude,
        }
    }
}

impl<Real: Float> Default for LanczosOptions<Real> {
    fn default() -> Self {
        Self {
            max_basis: 30,
            eigenpairs: 6,
            tolerance: Real::from(1.0e-6).unwrap(),
            which: LanczosWhich::LargestMagnitude,
        }
    }
}

/// One Lanczos Ritz pair and its true operator residual.
pub struct LanczosRitzPair<'a, Scalar, Space>
where
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
{
    /// Real Ritz value.
    pub value: Scalar::Real,
    /// Normalized Ritz vector.
    pub vector: Element<'a, Space>,
    /// `||A v - lambda v||_2`.
    pub residual: Scalar::Real,
    /// Whether the requested relative tolerance was attained.
    pub converged: bool,
}

/// Result of a native Lanczos factorization.
pub struct LanczosResult<'a, Scalar, Space>
where
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
{
    /// Requested Ritz pairs in the selected order.
    pub pairs: Vec<LanczosRitzPair<'a, Scalar, Space>>,
    /// Number of operator applications.
    pub operator_applications: usize,
    /// Dimension reached by the Krylov basis.
    pub basis_dimension: usize,
    /// Whether Lanczos terminated because the Krylov space became invariant.
    pub invariant_subspace: bool,
    /// Number of completed restart operations.
    pub restart_count: usize,
    /// Number of converged pairs preserved by locking.
    pub locked_pairs: usize,
}

/// Compute Ritz pairs of a real symmetric or complex Hermitian operator.
///
/// The caller is responsible for supplying a self-adjoint operator. The small
/// imaginary part of `q^* A q` caused by roundoff is discarded. A materially
/// non-Hermitian operator should instead be passed to native Arnoldi.
pub fn lanczos<'a, Scalar, Space, OpImpl>(
    operator: &'a Operator<OpImpl>,
    initial: &'a Element<'a, Space>,
    options: LanczosOptions<Scalar::Real>,
) -> RlstResult<LanczosResult<'a, Scalar, Space>>
where
    Scalar: RlstScalar + Lapack,
    Scalar::Real: Lapack,
    Space: InnerProductSpace<F = Scalar> + NormedSpace<Output = Scalar::Real>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
{
    if options.max_basis == 0 || options.eigenpairs == 0 {
        return Err(RlstError::GeneralError(
            "Lanczos requires nonzero max_basis and eigenpairs".into(),
        ));
    }
    if options.tolerance < Scalar::Real::zero() || !options.tolerance.is_finite() {
        return Err(RlstError::GeneralError(
            "Lanczos tolerance must be finite and nonnegative".into(),
        ));
    }
    let initial_norm = initial.norm();
    if initial_norm == Scalar::Real::zero() || !initial_norm.is_finite() {
        return Err(RlstError::GeneralError(
            "Lanczos initial vector must have finite nonzero norm".into(),
        ));
    }

    let mut basis = Vec::with_capacity(options.max_basis + 1);
    let mut first = initial.clone();
    first *= Scalar::from_real(Scalar::Real::one() / initial_norm);
    basis.push(first);
    let mut diagonal = Vec::with_capacity(options.max_basis);
    let mut off_diagonal = Vec::with_capacity(options.max_basis.saturating_sub(1));
    let breakdown = Scalar::Real::epsilon() * Scalar::real(options.max_basis.max(10));
    let mut invariant_subspace = false;

    for column in 0..options.max_basis {
        let mut work = operator.dot(&basis[column]);
        if column > 0 {
            let previous_beta = off_diagonal[column - 1];
            operator.domain().axpy_inplace(
                &(-Scalar::from_real(previous_beta)),
                &basis[column - 1],
                &mut work,
            );
        }
        let alpha = work.inner(&basis[column]).re();
        operator
            .domain()
            .axpy_inplace(&(-Scalar::from_real(alpha)), &basis[column], &mut work);
        diagonal.push(alpha);

        // Full two-pass reorthogonalization prevents loss of orthogonality and
        // duplicate converged Ritz values. Corrections are roundoff-sized for
        // a Hermitian operator and exact Lanczos recurrence.
        for _ in 0..2 {
            for vector in basis.iter().take(column + 1) {
                let coefficient = work.inner(vector);
                operator
                    .domain()
                    .axpy_inplace(&(-coefficient), vector, &mut work);
            }
        }
        let beta = work.norm();
        if column + 1 == options.max_basis || beta <= breakdown || !beta.is_finite() {
            invariant_subspace = beta <= breakdown || !beta.is_finite();
            break;
        }
        off_diagonal.push(beta);
        work *= Scalar::from_real(Scalar::Real::one() / beta);
        basis.push(work);
    }

    let dimension = diagonal.len();
    let mut tridiagonal = DynArray::<Scalar::Real, 2>::from_shape([dimension, dimension]);
    for index in 0..dimension {
        tridiagonal[[index, index]] = diagonal[index];
        if index + 1 < dimension {
            tridiagonal[[index + 1, index]] = off_diagonal[index];
            tridiagonal[[index, index + 1]] = off_diagonal[index];
        }
    }
    let (values, vectors) =
        tridiagonal.eigh(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors)?;
    let vectors = vectors.expect("eigenvectors were requested");
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|&left, &right| {
        let lhs = values[[left]];
        let rhs = values[[right]];
        let ordering = match options.which {
            LanczosWhich::LargestAlgebraic | LanczosWhich::SmallestAlgebraic => {
                lhs.partial_cmp(&rhs)
            }
            LanczosWhich::LargestMagnitude | LanczosWhich::SmallestMagnitude => {
                Float::abs(lhs).partial_cmp(&Float::abs(rhs))
            }
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        match options.which {
            LanczosWhich::LargestAlgebraic | LanczosWhich::LargestMagnitude => ordering.reverse(),
            LanczosWhich::SmallestAlgebraic | LanczosWhich::SmallestMagnitude => ordering,
        }
    });

    let mut pairs = Vec::with_capacity(options.eigenpairs.min(dimension));
    for index in order.into_iter().take(options.eigenpairs.min(dimension)) {
        let mut vector = operator.domain().zero();
        for row in 0..dimension {
            operator.domain().axpy_inplace(
                &Scalar::from_real(vectors[[row, index]]),
                &basis[row],
                &mut vector,
            );
        }
        let vector_norm = vector.norm();
        if vector_norm != Scalar::Real::zero() {
            vector *= Scalar::from_real(Scalar::Real::one() / vector_norm);
        }
        let value = values[[index]];
        let mut residual_vector = operator.dot(&vector);
        operator
            .domain()
            .axpy_inplace(&(-Scalar::from_real(value)), &vector, &mut residual_vector);
        let residual = residual_vector.norm();
        let scale = Scalar::Real::one().max(Float::abs(value));
        pairs.push(LanczosRitzPair {
            value,
            vector,
            residual,
            converged: residual <= options.tolerance * scale,
        });
    }

    Ok(LanczosResult {
        operator_applications: dimension + pairs.len(),
        basis_dimension: dimension,
        invariant_subspace,
        pairs,
        restart_count: 0,
        locked_pairs: 0,
    })
}

fn lanczos_order<Real: Float + Default>(
    values: &DynArray<Real, 1>,
    which: LanczosWhich,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.shape()[0]).collect();
    order.sort_by(|&left, &right| {
        let lhs = values[[left]];
        let rhs = values[[right]];
        let ordering = match which {
            LanczosWhich::LargestAlgebraic | LanczosWhich::SmallestAlgebraic => {
                lhs.partial_cmp(&rhs)
            }
            LanczosWhich::LargestMagnitude | LanczosWhich::SmallestMagnitude => {
                Float::abs(lhs).partial_cmp(&Float::abs(rhs))
            }
        }
        .unwrap_or(std::cmp::Ordering::Equal);
        match which {
            LanczosWhich::LargestAlgebraic | LanczosWhich::LargestMagnitude => ordering.reverse(),
            LanczosWhich::SmallestAlgebraic | LanczosWhich::SmallestMagnitude => ordering,
        }
    });
    order
}

/// Compute Hermitian eigenpairs with bounded-memory thick-restarted Lanczos.
///
/// Restarted cycles retain a block of the best unconverged Ritz vectors and
/// lock converged vectors. Full reorthogonalization is performed against both
/// active and locked vectors. The projected matrix is formed explicitly after
/// restart; this is the stable block Rayleigh--Ritz form of thick restarting.
pub fn thick_restarted_lanczos<'a, Scalar, Space, OpImpl>(
    operator: &'a Operator<OpImpl>,
    initial: &'a Element<'a, Space>,
    options: ThickRestartLanczosOptions<Scalar::Real>,
) -> RlstResult<LanczosResult<'a, Scalar, Space>>
where
    Scalar: RlstScalar + Lapack,
    Scalar::Real: Lapack,
    Space: InnerProductSpace<F = Scalar> + NormedSpace<Output = Scalar::Real>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
{
    if options.eigenpairs == 0
        || options.restart_keep == 0
        || options.max_basis <= options.restart_keep
        || options.restart_keep < options.eigenpairs
    {
        return Err(RlstError::GeneralError(
            "thick-restarted Lanczos requires 0 < eigenpairs <= restart_keep < max_basis".into(),
        ));
    }
    if options.tolerance < Scalar::Real::zero() || !options.tolerance.is_finite() {
        return Err(RlstError::GeneralError(
            "Lanczos tolerance must be finite and nonnegative".into(),
        ));
    }
    let initial_norm = initial.norm();
    if initial_norm == Scalar::Real::zero() || !initial_norm.is_finite() {
        return Err(RlstError::GeneralError(
            "Lanczos initial vector must have finite nonzero norm".into(),
        ));
    }

    let breakdown = Scalar::Real::epsilon() * Scalar::real(options.max_basis.max(10));
    let mut first = initial.clone();
    first *= Scalar::from_real(Scalar::Real::one() / initial_norm);
    let mut basis = vec![first];
    let mut locked: Vec<LanczosRitzPair<'a, Scalar, Space>> = Vec::new();
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
        // Explicitly symmetrize the small projection to suppress accumulated
        // roundoff without hiding a non-Hermitian large operator.
        for column in 0..dimension {
            projected[[column, column]] = Scalar::from_real(projected[[column, column]].re());
            for row in 0..column {
                let value = (projected[[row, column]] + projected[[column, row]].conj())
                    .mul_real(Scalar::real(0.5));
                projected[[row, column]] = value;
                projected[[column, row]] = value.conj();
            }
        }
        let (values, vectors) =
            projected.eigh(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors)?;
        let vectors = vectors.expect("eigenvectors were requested");
        let order = lanczos_order(&values, options.which);
        let mut candidates = Vec::with_capacity(dimension);
        for index in order {
            let mut vector = operator.domain().zero();
            for row in 0..dimension {
                operator
                    .domain()
                    .axpy_inplace(&vectors[[row, index]], &basis[row], &mut vector);
            }
            let vector_norm = vector.norm();
            if vector_norm <= breakdown || !vector_norm.is_finite() {
                continue;
            }
            vector *= Scalar::from_real(Scalar::Real::one() / vector_norm);
            let value = values[[index]];
            let mut residual_vector = operator.dot(&vector);
            operator_applications += 1;
            operator.domain().axpy_inplace(
                &(-Scalar::from_real(value)),
                &vector,
                &mut residual_vector,
            );
            let residual = residual_vector.norm();
            let converged =
                residual <= options.tolerance * Scalar::Real::one().max(Float::abs(value));
            candidates.push(LanczosRitzPair {
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
    let missing = options.eigenpairs.saturating_sub(pairs.len());
    pairs.extend(final_candidates.into_iter().take(missing));
    Ok(LanczosResult {
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
        LanczosOptions, LanczosWhich, ThickRestartLanczosOptions, lanczos, thick_restarted_lanczos,
    };
    use crate::{
        LinearSpace, RlstScalar, UpLo,
        abstract_operator::OperatorBase,
        base_types::{c32, c64},
        dense::{array::DynArray, linalg::lapack::symmeig::SymmEigMode},
        operator::abstract_operator::Operator,
        traits::linalg::decompositions::SymmEig,
    };

    macro_rules! lanczos_test {
        ($name:ident, $scalar:ty, $make:expr, $tol:expr) => {
            #[test]
            fn $name() {
                let make = $make;
                let mut matrix = DynArray::<$scalar, 2>::from_shape([4, 4]);
                for (index, value) in [1.0, 2.0, 4.0, 7.0].into_iter().enumerate() {
                    matrix[[index, index]] = make(value, 0.0);
                }
                matrix[[0, 1]] = make(0.4, -0.25);
                matrix[[1, 0]] = matrix[[0, 1]].conj();
                matrix[[1, 2]] = make(-0.3, 0.2);
                matrix[[2, 1]] = matrix[[1, 2]].conj();
                matrix[[2, 3]] = make(0.2, 0.1);
                matrix[[3, 2]] = matrix[[2, 3]].conj();
                let (exact, _) = matrix
                    .eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly)
                    .unwrap();
                let operator = Operator::from(&matrix);
                let mut initial = operator.domain().zero();
                for (entry, value) in initial.imp_mut().iter_mut().zip(1..=4) {
                    *entry = make(value as f64, (5 - value) as f64 * 0.13);
                }
                let result = lanczos(
                    &operator,
                    &initial,
                    LanczosOptions {
                        max_basis: 4,
                        eigenpairs: 2,
                        tolerance: $tol,
                        which: LanczosWhich::LargestAlgebraic,
                    },
                )
                .unwrap();
                assert_eq!(result.basis_dimension, 4);
                assert_eq!(result.pairs.len(), 2);
                assert!((result.pairs[0].value - exact[[3]]).abs() < $tol * 10.0);
                assert!((result.pairs[1].value - exact[[2]]).abs() < $tol * 10.0);
                assert!(result.pairs.iter().all(|pair| pair.residual < $tol * 20.0));
            }
        };
    }

    lanczos_test!(hermitian_f32, f32, |re: f64, _| re as f32, 3.0e-5_f32);
    lanczos_test!(hermitian_f64, f64, |re: f64, _| re, 3.0e-12_f64);
    lanczos_test!(
        hermitian_c32,
        c32,
        |re: f64, im: f64| c32::new(re as f32, im as f32),
        5.0e-5_f32
    );
    lanczos_test!(
        hermitian_c64,
        c64,
        |re: f64, im: f64| c64::new(re, im),
        5.0e-12_f64
    );

    #[test]
    fn thick_restart_locks_clustered_extreme_pairs() {
        let dimension = 30;
        let mut matrix = DynArray::<f64, 2>::from_shape([dimension, dimension]);
        for index in 0..dimension {
            matrix[[index, index]] = if index < 3 {
                30.0 - 1.0e-4 * index as f64
            } else {
                25.0 - index as f64 * 0.8
            };
        }
        let operator = Operator::from(&matrix);
        let mut initial = operator.domain().zero();
        for (index, entry) in initial.imp_mut().iter_mut().enumerate() {
            *entry = 1.0 + index as f64 * 0.01;
        }
        let result = thick_restarted_lanczos(
            &operator,
            &initial,
            ThickRestartLanczosOptions {
                max_basis: 9,
                restart_keep: 3,
                max_restarts: 50,
                eigenpairs: 2,
                tolerance: 1.0e-8,
                which: LanczosWhich::LargestAlgebraic,
            },
        )
        .unwrap();
        assert!(result.restart_count > 0);
        assert!(result.locked_pairs > 0);
        assert!(result.basis_dimension <= 9);
        assert_eq!(result.pairs.len(), 2);
        assert!(result.pairs.iter().all(|pair| pair.converged));
        assert!(result.pairs.iter().all(|pair| pair.residual < 3.0e-7));
        assert!((result.pairs[0].value - 30.0).abs() < 1.0e-7);
        assert!((result.pairs[1].value - 29.9999).abs() < 1.0e-6);
    }
}
