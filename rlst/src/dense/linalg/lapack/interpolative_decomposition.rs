//! Matrix interpolative decomposition based on pivoted QR.

use num::{One, Zero};
use serde::{Deserialize, Serialize};

use crate::{
    RlstScalar, Shape, TransMode, UnsafeRandom1DAccessByValue, UnsafeRandomAccessByRef, UpLo,
    base_types::{RlstResult, c32, c64},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::qr::EnablePivoting,
    },
    traits::linalg::{
        decompositions::{Qr, SolveTriangular},
        lapack::Lapack,
    },
};

/// Rank-selection policy for an interpolative decomposition.
///
/// This is the primary API. The names state precisely whether a tolerance is
/// combined with a lower or upper rank bound.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum RankSelection<Real> {
    /// Select all pivots larger than `tolerance * max(abs(diag(R)))`.
    Tolerance {
        /// Relative threshold applied to the diagonal of the R factor.
        tolerance: Real,
    },
    /// Use exactly `rank` skeletons, capped by the candidate count.
    Fixed {
        /// Requested number of skeletons.
        rank: usize,
    },
    /// Apply the tolerance and select at least `minimum_rank` skeletons.
    ToleranceWithMinimumRank {
        /// Relative threshold applied to the diagonal of the R factor.
        tolerance: Real,
        /// Lower bound for the selected rank.
        minimum_rank: usize,
    },
    /// Apply the tolerance and select at most `maximum_rank` skeletons.
    ToleranceWithMaximumRank {
        /// Relative threshold applied to the diagonal of the R factor.
        tolerance: Real,
        /// Upper bound for the selected rank.
        maximum_rank: usize,
    },
}

/// Historical rank-policy spelling retained for downstream compatibility.
///
/// New code should use [`RankSelection`]. In particular, the historical
/// `MaxRank` and `MinRank` names describe how the tolerance-derived rank is
/// combined, not the bound they impose, and are therefore easy to misuse.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum Accuracy<Real> {
    /// Select all pivots larger than `tol * max(abs(diag(R)))`.
    Tol(Real),
    /// Use exactly this many skeletons, capped by the candidate count.
    FixedRank(usize),
    /// Preserve the historical RLST behavior: use at least `rank` skeletons.
    MaxRank(Real, usize),
    /// Preserve the historical RLST behavior: use at most `rank` skeletons.
    MinRank(Real, usize),
}

impl<Real> From<Accuracy<Real>> for RankSelection<Real> {
    fn from(value: Accuracy<Real>) -> Self {
        match value {
            Accuracy::Tol(tolerance) => Self::Tolerance { tolerance },
            Accuracy::FixedRank(rank) => Self::Fixed { rank },
            Accuracy::MaxRank(tolerance, minimum_rank) => Self::ToleranceWithMinimumRank {
                tolerance,
                minimum_rank,
            },
            Accuracy::MinRank(tolerance, maximum_rank) => Self::ToleranceWithMaximumRank {
                tolerance,
                maximum_rank,
            },
        }
    }
}

/// Rank-revealing QR variant.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum RankRevealingQrType<Real> {
    /// LAPACK column-pivoted QR.
    RRQR,
    /// Strong RRQR with coefficient bound `f`.
    ///
    /// The initial factorization is column-pivoted QR. The strong refinement
    /// is implemented as a separate postprocessing step so the public choice
    /// remains compatible with the historical fork.
    SRRQR(Real),
}

/// Scalar types supporting interpolative decomposition.
pub trait MatrixId: Lapack {}

/// Marker retained for compatibility with code that requests an ID without
/// materializing the skeleton matrix.
pub trait MatrixIdNoSkel: MatrixId {}

macro_rules! impl_matrix_id {
    ($($scalar:ty),+ $(,)?) => {$(
        impl MatrixId for $scalar {}
        impl MatrixIdNoSkel for $scalar {}
    )+};
}

impl_matrix_id!(f32, f64, c32, c64);

/// Result of a row or column interpolative decomposition.
pub struct IdDecomposition<Item: RlstScalar> {
    /// Selected candidate vectors, stored as rows.
    pub skel: DynArray<Item, 2>,
    /// Pivot order. The first `rank` entries are skeleton indices.
    pub perm: Vec<usize>,
    /// Numerical rank selected by the requested policy.
    pub rank: usize,
    /// Interpolation coefficients with shape `(candidate_count-rank, rank)`.
    pub id_mat: DynArray<Item, 2>,
}

impl<Item, ArrayImpl> Array<ArrayImpl, 2>
where
    Item: MatrixId,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + Shape<2>,
{
    /// Compute an interpolative decomposition using the current RLST API.
    pub fn interpolative_decomposition(
        &self,
        rank_selection: RankSelection<Item::Real>,
        qr_type: RankRevealingQrType<Item::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        id_decomposition(self, rank_selection, qr_type, trans_mode, true)
    }

    /// Compute an interpolative decomposition without copying the skeletons.
    pub fn interpolative_decomposition_without_skeletons(
        &self,
        rank_selection: RankSelection<Item::Real>,
        qr_type: RankRevealingQrType<Item::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        id_decomposition(self, rank_selection, qr_type, trans_mode, false)
    }

    /// Compute an interpolative decomposition and materialize skeletons.
    pub fn into_id_alloc(
        &self,
        accuracy: Accuracy<Item::Real>,
        qr_type: RankRevealingQrType<Item::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        self.interpolative_decomposition(accuracy.into(), qr_type, trans_mode)
    }

    /// Compute an interpolative decomposition without materializing skeletons.
    pub fn into_id_alloc_no_skel(
        &self,
        accuracy: Accuracy<Item::Real>,
        qr_type: RankRevealingQrType<Item::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        self.interpolative_decomposition_without_skeletons(accuracy.into(), qr_type, trans_mode)
    }
}

fn id_decomposition<Item, ArrayImpl>(
    array: &Array<ArrayImpl, 2>,
    rank_selection: RankSelection<Item::Real>,
    qr_type: RankRevealingQrType<Item::Real>,
    trans_mode: TransMode,
    build_skeleton: bool,
) -> RlstResult<IdDecomposition<Item>>
where
    Item: MatrixId,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + Shape<2>,
{
    let [rows, columns] = array.shape();
    let (candidate_count, feature_count) = match trans_mode {
        TransMode::NoTrans | TransMode::ConjNoTrans => (rows, columns),
        TransMode::Trans | TransMode::ConjTrans => (columns, rows),
    };

    // Candidate vectors are columns of the QR input.
    let mut work = DynArray::<Item, 2>::from_shape([feature_count, candidate_count]);
    for candidate in 0..candidate_count {
        for feature in 0..feature_count {
            let value = match trans_mode {
                TransMode::NoTrans => array[[candidate, feature]],
                TransMode::ConjNoTrans => array[[candidate, feature]].conj(),
                TransMode::Trans => array[[feature, candidate]],
                TransMode::ConjTrans => array[[feature, candidate]].conj(),
            };
            work[[feature, candidate]] = value;
        }
    }

    let qr = work.qr(EnablePivoting::Yes)?;
    let mut r = qr.r_mat()?;
    let mut perm = qr.perm();

    let diagonal_count = feature_count.min(candidate_count);
    let tolerance_rank =
        numerical_rank(&r, diagonal_count, rank_selection_tolerance(rank_selection));
    let rank = select_rank(rank_selection, tolerance_rank, candidate_count);

    if let RankRevealingQrType::SRRQR(bound) = qr_type {
        r = strong_rrqr_refine(r, &mut perm, rank, bound)?;
    }

    let skel = if build_skeleton {
        let mut selected = DynArray::<Item, 2>::from_shape([rank, feature_count]);
        for (selected_index, &candidate) in perm.iter().take(rank).enumerate() {
            for feature in 0..feature_count {
                selected[[selected_index, feature]] = work[[feature, candidate]];
            }
        }
        selected
    } else {
        DynArray::<Item, 2>::from_shape([0, 0])
    };

    let id_mat = if rank == 0 || rank >= candidate_count {
        let mut identity = DynArray::<Item, 2>::from_shape([candidate_count, candidate_count]);
        identity.set_identity();
        identity
    } else {
        let remainder = candidate_count - rank;
        let mut r11 = DynArray::<Item, 2>::from_shape([rank, rank]);
        let mut r12 = DynArray::<Item, 2>::from_shape([rank, remainder]);
        for column in 0..rank {
            for row in 0..rank {
                r11[[row, column]] = r[[row, column]];
            }
        }
        for column in 0..remainder {
            for row in 0..rank {
                r12[[row, column]] = r[[row, rank + column]];
            }
        }
        let coefficients = r11.solve_triangular(UpLo::Upper, &r12)?;
        let mut interpolation = DynArray::<Item, 2>::from_shape([remainder, rank]);
        for row in 0..remainder {
            for column in 0..rank {
                interpolation[[row, column]] = coefficients[[column, row]].conj();
            }
        }
        interpolation
    };

    Ok(IdDecomposition {
        skel,
        perm,
        rank,
        id_mat,
    })
}

fn rank_selection_tolerance<Real: num::Float>(rank_selection: RankSelection<Real>) -> Option<Real> {
    match rank_selection {
        RankSelection::Tolerance { tolerance }
        | RankSelection::ToleranceWithMinimumRank { tolerance, .. }
        | RankSelection::ToleranceWithMaximumRank { tolerance, .. } => Some(tolerance),
        RankSelection::Fixed { .. } => None,
    }
}

fn numerical_rank<Item: RlstScalar>(
    r: &DynArray<Item, 2>,
    diagonal_count: usize,
    tolerance: Option<Item::Real>,
) -> usize {
    let Some(tolerance) = tolerance else {
        return diagonal_count;
    };
    let maximum = (0..diagonal_count)
        .map(|index| r[[index, index]].abs())
        .fold(Item::Real::zero(), num::Float::max);
    if maximum == Item::Real::zero() {
        return 0;
    }
    (0..diagonal_count)
        .filter(|&index| r[[index, index]].abs() > tolerance * maximum)
        .count()
}

fn select_rank<Real: num::Float>(
    rank_selection: RankSelection<Real>,
    tolerance_rank: usize,
    candidate_count: usize,
) -> usize {
    match rank_selection {
        RankSelection::Tolerance { .. } => tolerance_rank,
        RankSelection::Fixed { rank } => rank.min(candidate_count),
        RankSelection::ToleranceWithMinimumRank { minimum_rank, .. } => {
            tolerance_rank.max(minimum_rank).min(candidate_count)
        }
        RankSelection::ToleranceWithMaximumRank { maximum_rank, .. } => {
            tolerance_rank.min(maximum_rank).min(candidate_count)
        }
    }
}

fn strong_rrqr_refine<Item: MatrixId>(
    mut r: DynArray<Item, 2>,
    perm: &mut [usize],
    rank: usize,
    bound: Item::Real,
) -> RlstResult<DynArray<Item, 2>> {
    assert!(
        bound >= Item::Real::one(),
        "SRRQR bound must be at least one"
    );
    let [rows, columns] = r.shape();
    if rank == 0 || rank >= columns || rank > rows {
        return Ok(r);
    }

    // Gu--Eisenstat strong RRQR condition. Once A P = Q R has been formed, a
    // further column interchange needs only a QR factorization of R S:
    //
    //     A P S = Q (R S) = (Q Q_s) R_s.
    //
    // Refactorizing the compact R factor is mathematically equivalent to
    // refactorizing the tall original matrix, but avoids revisiting all of its
    // rows after every strong-RRQR swap. This is particularly important for
    // the tall sketch matrices used by RSRS and works unchanged for complex
    // scalars.
    let max_swaps = rank * (columns - rank) * 4 + 1;
    for _ in 0..max_swaps {
        let mut r11 = DynArray::<Item, 2>::from_shape([rank, rank]);
        let mut r12 = DynArray::<Item, 2>::from_shape([rank, columns - rank]);
        for column in 0..rank {
            for row in 0..rank {
                r11[[row, column]] = r[[row, column]];
            }
        }
        for column in rank..columns {
            for row in 0..rank {
                r12[[row, column - rank]] = r[[row, column]];
            }
        }
        let coefficients = r11.solve_triangular(UpLo::Upper, &r12)?;

        let mut identity = DynArray::<Item, 2>::from_shape([rank, rank]);
        identity.set_identity();
        let inverse = r11.solve_triangular(UpLo::Upper, &identity)?;
        let mut offending = None;
        'search: for trailing in 0..(columns - rank) {
            let gamma_squared = (rank..rows.min(columns))
                .map(|row| r[[row, rank + trailing]].abs().powi(2))
                .fold(Item::Real::zero(), |sum, value| sum + value);
            for skeleton in 0..rank {
                let inverse_row_squared = (0..rank)
                    .map(|column| inverse[[skeleton, column]].abs().powi(2))
                    .fold(Item::Real::zero(), |sum, value| sum + value);
                let omega_squared = Item::Real::one() / inverse_row_squared;
                let criterion = coefficients[[skeleton, trailing]].abs().powi(2)
                    + gamma_squared / omega_squared;
                if criterion > bound.powi(2) {
                    offending = Some((skeleton, trailing));
                    break 'search;
                }
            }
        }

        match offending {
            Some((skeleton, trailing)) => {
                let trailing = rank + trailing;
                perm.swap(skeleton, trailing);
                for row in 0..rows {
                    let value = r[[row, skeleton]];
                    r[[row, skeleton]] = r[[row, trailing]];
                    r[[row, trailing]] = value;
                }
                r = r.qr(EnablePivoting::No)?.r_mat()?;
            }
            None => return Ok(r),
        }
    }
    panic!("strong RRQR did not converge after {max_swaps} column swaps")
}

#[cfg(test)]
mod tests {
    use super::{Accuracy, MatrixId, RankRevealingQrType, RankSelection};
    use crate::{
        TransMode,
        base_types::{c32, c64},
        dense::array::DynArray,
    };

    fn rank_two_matrix() -> DynArray<f64, 2> {
        let mut matrix = DynArray::<f64, 2>::from_shape([3, 4]);
        let values = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        for row in 0..3 {
            for column in 0..4 {
                matrix[[row, column]] = values[row][column];
            }
        }
        matrix
    }

    #[test]
    fn tolerance_and_fixed_rank_column_id() {
        let matrix = rank_two_matrix();
        let tolerance_id = matrix
            .into_id_alloc_no_skel(
                Accuracy::Tol(1.0e-12),
                RankRevealingQrType::RRQR,
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(tolerance_id.rank, 2);
        assert_eq!(tolerance_id.perm.len(), 4);
        assert_eq!(tolerance_id.id_mat.shape(), [2, 2]);

        let fixed_id = matrix
            .into_id_alloc_no_skel(
                Accuracy::FixedRank(1),
                RankRevealingQrType::RRQR,
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(fixed_id.rank, 1);
        assert_eq!(fixed_id.id_mat.shape(), [3, 1]);
    }

    #[test]
    fn min_rank_caps_tolerance_rank() {
        let matrix = rank_two_matrix();
        let id = matrix
            .into_id_alloc_no_skel(
                Accuracy::MinRank(1.0e-12, 1),
                RankRevealingQrType::RRQR,
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(id.rank, 1);
    }

    #[test]
    fn explicit_rank_bounds_have_unambiguous_semantics() {
        let matrix = rank_two_matrix();
        let minimum = matrix
            .interpolative_decomposition_without_skeletons(
                RankSelection::ToleranceWithMinimumRank {
                    tolerance: 2.0,
                    minimum_rank: 1,
                },
                RankRevealingQrType::RRQR,
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(minimum.rank, 1);

        let maximum = matrix
            .interpolative_decomposition_without_skeletons(
                RankSelection::ToleranceWithMaximumRank {
                    tolerance: 1.0e-12,
                    maximum_rank: 1,
                },
                RankRevealingQrType::RRQR,
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(maximum.rank, 1);
    }

    fn check_srrqr_reconstruction<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: MatrixId,
        Make: Fn(f64, f64) -> Item,
    {
        // Store six candidate vectors as columns. The first two vectors span
        // the matrix and all remaining candidates are nontrivial linear
        // combinations. For complex scalars both the basis and coefficients
        // have nonzero imaginary parts.
        let feature_count = 5;
        let candidate_count = 6;
        let rank = 2;
        let mut matrix = DynArray::<Item, 2>::from_shape([feature_count, candidate_count]);
        let basis = [
            [make(1.0, 0.5), make(-0.3, 0.2)],
            [make(0.2, -0.7), make(1.1, 0.1)],
            [make(-0.8, 0.4), make(0.6, -0.9)],
            [make(0.5, 0.3), make(-1.2, 0.2)],
            [make(1.3, -0.2), make(0.4, 0.8)],
        ];
        let coefficients = [
            [make(1.0, 0.0), make(0.0, 0.0)],
            [make(0.0, 0.0), make(1.0, 0.0)],
            [make(0.4, 0.3), make(-0.2, 0.1)],
            [make(-0.6, 0.2), make(0.7, -0.4)],
            [make(0.1, -0.5), make(0.8, 0.2)],
            [make(0.9, 0.1), make(0.3, -0.6)],
        ];
        for feature in 0..feature_count {
            for candidate in 0..candidate_count {
                matrix[[feature, candidate]] = basis[feature][0] * coefficients[candidate][0]
                    + basis[feature][1] * coefficients[candidate][1];
            }
        }

        let bound = Item::real(2.0);
        let id = matrix
            .into_id_alloc(
                Accuracy::FixedRank(rank),
                RankRevealingQrType::SRRQR(bound),
                TransMode::Trans,
            )
            .unwrap();
        assert_eq!(id.rank, rank);
        assert_eq!(id.skel.shape(), [rank, feature_count]);
        assert_eq!(id.id_mat.shape(), [candidate_count - rank, rank]);

        // id_mat stores the conjugate-transposed interpolation coefficients,
        // matching the historical RSRS factor convention.
        for ordered_candidate in 0..candidate_count {
            let original_candidate = id.perm[ordered_candidate];
            for feature in 0..feature_count {
                let reconstructed = if ordered_candidate < rank {
                    id.skel[[ordered_candidate, feature]]
                } else {
                    (0..rank)
                        .map(|skeleton| {
                            id.skel[[skeleton, feature]]
                                * id.id_mat[[ordered_candidate - rank, skeleton]].conj()
                        })
                        .fold(Item::zero(), |sum, value| sum + value)
                };
                let error = (reconstructed - matrix[[feature, original_candidate]]).abs();
                assert!(
                    error <= tolerance,
                    "ID reconstruction error {error:e} exceeds {tolerance:e}"
                );
            }
        }

        // The Gu--Eisenstat criterion is stronger than this entrywise check,
        // hence every interpolation coefficient must obey the requested bound.
        for coefficient in id.id_mat.iter_value() {
            assert!(coefficient.abs() <= bound + tolerance);
        }
    }

    #[test]
    fn srrqr_id_reconstructs_f32() {
        check_srrqr_reconstruction::<f32, _>(|re, _| re as f32, 2.0e-5);
    }

    #[test]
    fn srrqr_id_reconstructs_f64() {
        check_srrqr_reconstruction::<f64, _>(|re, _| re, 2.0e-12);
    }

    #[test]
    fn srrqr_id_reconstructs_genuinely_complex_c32() {
        check_srrqr_reconstruction::<c32, _>(|re, im| c32::new(re as f32, im as f32), 5.0e-5);
    }

    #[test]
    fn srrqr_id_reconstructs_genuinely_complex_c64() {
        check_srrqr_reconstruction::<c64, _>(c64::new, 5.0e-12);
    }
}
