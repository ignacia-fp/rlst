//! Interpolative decomposition of a matrix.
use crate::dense::array::Array;
use crate::dense::traits::{
    RawAccessMut, Shape, Stride, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::dense::types::{Side, TransMode, TriangularType};
use crate::DynamicArray;
use crate::QrTolerance;
use crate::RankParam;
use crate::RankRevealingQrType;
use crate::{empty_array, rlst_dynamic_array2, BaseArray, VectorContainer};
use crate::{TriangularMatrix, TriangularOperations};
/// Compute the matrix interpolative decomposition, by providing a rank and an interpolation matrix.
///
/// The matrix interpolative decomposition is defined for a two dimensional 'long' array `arr` of
/// shape `[m, n]`, where `n>m`.
///
/// # Example
///
/// The following command computes the interpolative decomposition of an array `a` for a given tolerance, tol.
/// ```
/// # use rlst::rlst_dynamic_array2;
/// # use rlst::dense::types::TransMode;
/// # use rlst::dense::linalg::interpolative_decomposition::Accuracy;
/// # let tol: f64 = 1e-5;
/// # let mut a = rlst_dynamic_array2!(f64, [50, 100]);
/// # a.fill_from_seed_equally_distributed(0);
/// # let res = a.r_mut().into_id_alloc(Accuracy::Tol(tol), TransMode::NoTrans).unwrap();
/// ```
/// This method allocates memory for the interpolative decomposition.
pub trait MatrixId: RlstScalar {
    ///This method allocates space for ID
    fn into_id_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        rank_param: Accuracy<<Self as RlstScalar>::Real>,
        qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Self>>;
}

/// Compute the matrix interpolative decomposition without materializing the
/// skeleton matrix.
pub trait MatrixIdNoSkel: RlstScalar {
    /// This method allocates space for ID while skipping the skeleton array.
    fn into_id_alloc_no_skel<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        rank_param: Accuracy<<Self as RlstScalar>::Real>,
        qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Self>>;
}

macro_rules! implement_into_id {
    ($scalar:ty) => {
        impl MatrixId for $scalar {
            fn into_id_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + UnsafeRandomAccessMut<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                rank_param: Accuracy<<Self as RlstScalar>::Real>,
                qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<IdDecomposition<Self>> {
                IdDecomposition::<$scalar>::new(arr, rank_param, qr_type, trans_mode)
            }
        }

        impl MatrixIdNoSkel for $scalar {
            fn into_id_alloc_no_skel<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + UnsafeRandomAccessMut<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                rank_param: Accuracy<<Self as RlstScalar>::Real>,
                qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<IdDecomposition<Self>> {
                IdDecomposition::<$scalar>::new_no_skel(arr, rank_param, qr_type, trans_mode)
            }
        }
    };
}

implement_into_id!(f32);
implement_into_id!(f64);
implement_into_id!(c32);
implement_into_id!(c64);

impl<
        Item: RlstScalar + MatrixId,
        ArrayImplId: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImplId, 2>
{
    /// Compute the interpolative decomposition of a given 2-dimensional array.
    pub fn into_id_alloc(
        self,
        rank_param: Accuracy<<Item as RlstScalar>::Real>,
        qr_type: RankRevealingQrType<<Item as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        <Item as MatrixId>::into_id_alloc(self, rank_param, qr_type, trans_mode)
    }

    /// Compute the interpolative decomposition without materializing the
    /// skeleton matrix.
    pub fn into_id_alloc_no_skel(
        self,
        rank_param: Accuracy<<Item as RlstScalar>::Real>,
        qr_type: RankRevealingQrType<<Item as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>>
    where
        Item: MatrixIdNoSkel,
    {
        <Item as MatrixIdNoSkel>::into_id_alloc_no_skel(self, rank_param, qr_type, trans_mode)
    }
}

/// Compute the matrix interpolative decomposition
pub trait MatrixIdDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Create a new Interpolative Decomposition from a given array.
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
        rank_param: Accuracy<<Self::Item as RlstScalar>::Real>,
        qr_type: RankRevealingQrType<<Self::Item as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<Self>;

    ///Compute the permutation matrix associated to the Interpolative Decomposition
    fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );
}

///Stores the relevant features regarding interpolative decomposition.
pub struct IdDecomposition<Item: RlstScalar> {
    /// skel: skeleton of the interpolative decomposition
    pub skel: DynamicArray<Item, 2>,
    /// perm: permutation associated to the pivoting indiced interpolative decomposition
    pub perm: Vec<usize>,
    /// rank: rank of the matrix associated to the interpolative decomposition for a given tolerance
    pub rank: usize,
    ///id_mat: interpolative matrix calculated for a given tolerance
    pub id_mat: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

#[derive(Debug, Clone)]
///Options to decide the matrix rank
pub enum Accuracy<T> {
    /// Indicates that the rank of the decomposition will be computed from a given tolerance
    Tol(T),
    /// Indicates that the rank of the decomposition is given beforehand by the user
    FixedRank(usize),
    /// Computes the rank from the tolerance, and if this one is smaller than a user set range, then we stick to the user set range
    MaxRank(T, usize),
    /// Computes the rank from the tolerance, and if this one is larger than a user set range, then we stick to the user set range
    MinRank(T, usize),
}

fn id_matrix_shape(shape: [usize; 2], trans_mode: TransMode) -> [usize; 2] {
    match trans_mode {
        TransMode::NoTrans | TransMode::ConjNoTrans => shape,
        TransMode::Trans | TransMode::ConjTrans => [shape[1], shape[0]],
    }
}

fn gather_skeleton<Item, ArrayImpl>(
    arr: Array<Item, ArrayImpl, 2>,
    perm: &[usize],
    rank: usize,
    trans_mode: TransMode,
) -> DynamicArray<Item, 2>
where
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
{
    let [_, cols] = id_matrix_shape(arr.shape(), trans_mode);
    let mut skel = rlst_dynamic_array2!(Item, [rank, cols]);

    for (row, &perm_index) in perm.iter().take(rank).enumerate() {
        match trans_mode {
            TransMode::NoTrans => skel
                .r_mut()
                .into_subview([row, 0], [1, cols])
                .fill_from(arr.r().into_subview([perm_index, 0], [1, cols])),
            TransMode::Trans => skel
                .r_mut()
                .into_subview([row, 0], [1, cols])
                .fill_from(arr.r().transpose().into_subview([perm_index, 0], [1, cols])),
            TransMode::ConjNoTrans => skel
                .r_mut()
                .into_subview([row, 0], [1, cols])
                .fill_from(arr.r().conj().into_subview([perm_index, 0], [1, cols])),
            TransMode::ConjTrans => skel.r_mut().into_subview([row, 0], [1, cols]).fill_from(
                arr.r()
                    .transpose()
                    .conj()
                    .into_subview([perm_index, 0], [1, cols]),
            ),
        }
    }

    skel
}

macro_rules! impl_id {
    ($scalar:ty) => {
        impl IdDecomposition<$scalar> {
            fn from_array<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImpl, 2>,
                rank_param: Accuracy<<$scalar as RlstScalar>::Real>,
                qr_type: RankRevealingQrType<<$scalar as RlstScalar>::Real>,
                trans_mode: TransMode,
                build_skeleton: bool,
            ) -> RlstResult<Self> {
                //We compute the QR decomposition using rlst QR decomposition
                let mut arr_work = empty_array();
                //let mut u_tri = empty_array();

                match trans_mode {
                    TransMode::Trans => arr_work.fill_from_resize(arr.r()),
                    TransMode::NoTrans => arr_work.fill_from_resize(arr.r().transpose()),
                    TransMode::ConjNoTrans => arr_work.fill_from_resize(arr.r().conj()),
                    TransMode::ConjTrans => arr_work.fill_from_resize(arr.r().transpose().conj()),
                };

                let (mut r, perm, rank) = match qr_type {
                    RankRevealingQrType::RRQR => match rank_param {
                        Accuracy::Tol(tol) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::RRQR,
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            (rrqr.r, rrqr.perm, rrqr.rank)
                        }
                        Accuracy::FixedRank(k) => {
                            let rrqr = arr_work
                                .r_mut()
                                .into_rrqr_alloc(RankRevealingQrType::RRQR, RankParam::Rank(k));
                            (rrqr.r, rrqr.perm, rrqr.rank)
                        }
                        Accuracy::MaxRank(tol, k) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::RRQR,
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            let rank = std::cmp::max(k, rrqr.rank);
                            (rrqr.r, rrqr.perm, rank)
                        }
                        Accuracy::MinRank(tol, k) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::RRQR,
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            let rank = std::cmp::min(k, rrqr.rank);
                            (rrqr.r, rrqr.perm, rank)
                        }
                    },
                    RankRevealingQrType::SRRQR(f) => match rank_param {
                        Accuracy::Tol(tol) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::SRRQR(f),
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            (rrqr.r, rrqr.perm, rrqr.rank)
                        }
                        Accuracy::FixedRank(k) => {
                            let rrqr = arr_work
                                .r_mut()
                                .into_rrqr_alloc(RankRevealingQrType::SRRQR(f), RankParam::Rank(k));
                            (rrqr.r, rrqr.perm, rrqr.rank)
                        }
                        Accuracy::MaxRank(tol, k) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::SRRQR(f),
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            let rank = std::cmp::max(k, rrqr.rank);
                            (rrqr.r, rrqr.perm, rank)
                        }
                        Accuracy::MinRank(tol, k) => {
                            let rrqr = arr_work.r_mut().into_rrqr_alloc(
                                RankRevealingQrType::SRRQR(f),
                                RankParam::Tol(tol, QrTolerance::Rel),
                            );
                            let rank = std::cmp::min(k, rrqr.rank);
                            (rrqr.r, rrqr.perm, rank)
                        }
                    },
                };

                let dim = arr_work.shape()[1];

                let skel = if build_skeleton {
                    gather_skeleton(arr, &perm, rank, trans_mode)
                } else {
                    empty_array::<$scalar, 2>()
                };

                //In the case the matrix is full rank or we get a matrix of rank 0, then return the identity matrix.
                //If not, compute the Interpolative Decomposition matrix.
                if rank == 0 || rank >= dim {
                    let mut id_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                    id_mat.set_identity();
                    Ok(Self {
                        skel,
                        perm,
                        rank,
                        id_mat,
                    })
                } else {
                    let mut id_mat: DynamicArray<$scalar, 2> =
                        rlst_dynamic_array2!($scalar, [dim - rank, rank]);
                    let r11 = TriangularMatrix::<$scalar>::new(
                        &r.r_mut().into_subview([0, 0], [rank, rank]),
                        TriangularType::Upper,
                    )
                    .unwrap();

                    let mut r12 = r.r_mut().into_subview([0, rank], [rank, dim - rank]);
                    r11.solve(&mut r12, Side::Left, TransMode::NoTrans);

                    id_mat.fill_from(r12.r().conj().transpose().r());
                    Ok(Self {
                        skel,
                        perm,
                        rank,
                        id_mat,
                    })
                }
            }

            fn new_no_skel<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImpl, 2>,
                rank_param: Accuracy<<$scalar as RlstScalar>::Real>,
                qr_type: RankRevealingQrType<<$scalar as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<Self> {
                Self::from_array(arr, rank_param, qr_type, trans_mode, false)
            }
        }

        impl MatrixIdDecomposition for IdDecomposition<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImpl, 2>,
                rank_param: Accuracy<<$scalar as RlstScalar>::Real>,
                qr_type: RankRevealingQrType<<$scalar as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<Self> {
                Self::from_array(arr, rank_param, qr_type, trans_mode, true)
            }

            fn get_p<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                arr.set_zero();
                let mut view = arr.r_mut();

                for (index, &elem) in self.perm.iter().enumerate() {
                    view[[index, elem]] = <$scalar as num::One>::one();
                }
            }
        }
    };
}

impl_id!(f64);
impl_id!(f32);
impl_id!(c32);
impl_id!(c64);

#[cfg(test)]
mod tests {
    use super::*;

    fn rank_two_test_matrix() -> DynamicArray<f64, 2> {
        let mut arr = rlst_dynamic_array2!(f64, [2, 4]);
        arr.r_mut()[[0, 0]] = 1.0;
        arr.r_mut()[[0, 2]] = 1.0;
        arr.r_mut()[[1, 1]] = 1.0;
        arr.r_mut()[[1, 3]] = 1.0;
        arr
    }

    #[test]
    fn min_rank_caps_rrqr_rank() {
        let mut arr = rank_two_test_matrix();
        let id = arr
            .r_mut()
            .into_id_alloc(
                Accuracy::MinRank(1e-12, 1),
                RankRevealingQrType::RRQR,
                TransMode::NoTrans,
            )
            .unwrap();
        assert_eq!(id.rank, 1);
    }

    #[test]
    fn min_rank_caps_srrqr_rank() {
        let mut arr = rank_two_test_matrix();
        let id = arr
            .r_mut()
            .into_id_alloc(
                Accuracy::MinRank(1e-12, 1),
                RankRevealingQrType::SRRQR(2.0),
                TransMode::NoTrans,
            )
            .unwrap();
        assert_eq!(id.rank, 1);
    }
}
