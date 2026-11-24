//! Lapack LU Decomposition.

use num::One;

use crate::base_types::{RlstError, RlstResult, TransMode};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::getrs::GetrsTransMode;
use crate::traits::linalg::decompositions::Lu;
use crate::traits::linalg::lapack::Lapack;
use crate::{Shape, UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByValue};

impl<Item, ArrayImpl> Lu for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;
    fn lu(&self) -> RlstResult<LuDecomposition<Item>> {
        let mut lu_mat = DynArray::new_from(self);
        let (m, n, lda) = (lu_mat.shape()[0], lu_mat.shape()[1], lu_mat.shape()[0]);
        let mut ipiv = vec![0_i32; std::cmp::min(m, n)];

        Item::getrf(m, n, lu_mat.data_mut().unwrap(), lda, &mut ipiv)?;

        Ok(LuDecomposition { lu: lu_mat, ipiv })
    }
}

/// Store the result of an LU decomposition of a matrix.
/// The LU Decomposition is represented as `A = P * L * U`, where:
/// - `A` is the original matrix,
/// - `P` is a permutation matrix,
/// - `L` is a lower triangular matrix with unit diagonal,
/// - `U` is an upper triangular matrix.
pub struct LuDecomposition<Item> {
    lu: DynArray<Item, 2>,
    ipiv: Vec<i32>,
}
/// Owned square LU factors for repeated solves and applications.
///
/// This compatibility wrapper owns only the packed LAPACK factors. Matrix
/// application uses the triangular factors and pivots directly, avoiding a
/// second dense `n x n` matrix.
pub struct SquareLuFactors<Item: Lapack> {
    lu: DynArray<Item, 2>,
    ipiv: Vec<i32>,
}

impl<Item: Lapack> SquareLuFactors<Item> {
    /// Copy reusable factors from a square LU decomposition.
    pub fn from_lu(lu: &LuDecomposition<Item>) -> RlstResult<Self> {
        let [rows, columns] = lu.lu.shape();
        assert_eq!(rows, columns, "reusable LU factors require a square matrix");
        Ok(Self {
            lu: DynArray::new_from(&lu.lu),
            ipiv: lu.ipiv.clone(),
        })
    }

    /// Return the row permutation in the same convention as
    /// [`LuDecomposition::perm_vec`].
    pub fn perm(&self) -> Vec<usize> {
        let mut permutation = (0..self.lu.shape()[0]).collect::<Vec<_>>();
        for (index, &pivot) in self.ipiv.iter().enumerate() {
            permutation.swap(index, pivot as usize - 1);
        }
        permutation
    }

    /// Overwrite a vector with the factorized matrix applied to it.
    pub fn mul_vec<ArrayImpl>(&self, trans: TransMode, rhs: Array<ArrayImpl, 1>) -> RlstResult<()>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
            + UnsafeDisjointRandom1DAccessMut<Item = Item>
            + Shape<1>,
    {
        self.mul_mat(
            trans,
            rhs.insert_empty_axis(crate::dense::array::empty_axis::AxisPosition::Back),
        )
    }

    /// Overwrite one or more right-hand sides with the factorized matrix
    /// applied to them.
    pub fn mul_mat<ArrayImpl>(
        &self,
        trans: TransMode,
        mut rhs: Array<ArrayImpl, 2>,
    ) -> RlstResult<()>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
            + UnsafeDisjointRandom1DAccessMut<Item = Item>
            + Shape<2>,
    {
        let [rows, columns] = rhs.shape();
        assert_eq!(rows, self.lu.shape()[0]);
        let conjugate = matches!(trans, TransMode::ConjNoTrans | TransMode::ConjTrans);
        let transpose = matches!(trans, TransMode::Trans | TransMode::ConjTrans);
        let factor = |row: usize, column: usize| {
            let value = self.lu[[row, column]];
            if conjugate { value.conj() } else { value }
        };

        let mut work = DynArray::new_from(&rhs);
        for column in 0..columns {
            if transpose {
                // A^T = U^T L^T P^T (and analogously for A^H). P^T is the
                // forward LAPACK pivot sequence.
                for (index, &pivot) in self.ipiv.iter().enumerate() {
                    let pivot = pivot as usize - 1;
                    let value = work[[index, column]];
                    work[[index, column]] = work[[pivot, column]];
                    work[[pivot, column]] = value;
                }
                // Unit-lower L transposed is unit-upper. Ascending order keeps
                // every input needed by subsequent rows intact.
                for row in 0..rows {
                    let mut value = work[[row, column]];
                    for inner in row + 1..rows {
                        value += factor(inner, row) * work[[inner, column]];
                    }
                    work[[row, column]] = value;
                }
                // Upper U transposed is lower. Descending order is in-place.
                for row in (0..rows).rev() {
                    let mut value = Item::zero();
                    for inner in 0..=row {
                        value += factor(inner, row) * work[[inner, column]];
                    }
                    work[[row, column]] = value;
                }
            } else {
                // Apply U in ascending row order, then unit-lower L in
                // descending row order; both operations are safe in-place.
                for row in 0..rows {
                    let mut value = Item::zero();
                    for inner in row..rows {
                        value += factor(row, inner) * work[[inner, column]];
                    }
                    work[[row, column]] = value;
                }
                for row in (0..rows).rev() {
                    let mut value = work[[row, column]];
                    for inner in 0..row {
                        value += factor(row, inner) * work[[inner, column]];
                    }
                    work[[row, column]] = value;
                }
                // A = P L U. P is the inverse of LAPACK's forward pivot
                // sequence, hence the reverse traversal.
                for (index, &pivot) in self.ipiv.iter().enumerate().rev() {
                    let pivot = pivot as usize - 1;
                    let value = work[[index, column]];
                    work[[index, column]] = work[[pivot, column]];
                    work[[pivot, column]] = value;
                }
            }
        }
        rhs.fill_from(&work);
        Ok(())
    }

    /// Overwrite a vector with the solution of the factorized system.
    pub fn solve_vec<ArrayImpl>(&self, trans: TransMode, rhs: Array<ArrayImpl, 1>) -> RlstResult<()>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
            + UnsafeDisjointRandom1DAccessMut<Item = Item>
            + Shape<1>,
    {
        self.solve_mat(
            trans,
            rhs.insert_empty_axis(crate::dense::array::empty_axis::AxisPosition::Back),
        )
    }

    /// Overwrite one or more right-hand sides with the solution of the
    /// factorized system.
    pub fn solve_mat<ArrayImpl>(
        &self,
        trans: TransMode,
        mut rhs: Array<ArrayImpl, 2>,
    ) -> RlstResult<()>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
            + UnsafeDisjointRandom1DAccessMut<Item = Item>
            + Shape<2>,
    {
        assert_eq!(rhs.shape()[0], self.lu.shape()[0]);
        let mode = match trans {
            TransMode::NoTrans => GetrsTransMode::NoTranspose,
            TransMode::Trans => GetrsTransMode::Transpose,
            TransMode::ConjTrans => GetrsTransMode::ConjugateTranspose,
            TransMode::ConjNoTrans => {
                return Err(RlstError::GeneralError(
                    "conjugate-without-transpose LU solve is unsupported".to_string(),
                ));
            }
        };
        let mut solution = DynArray::new_from(&rhs);
        let rows = self.lu.shape()[0];
        let nrhs = rhs.shape()[1];
        Item::getrs(
            mode,
            rows,
            nrhs,
            self.lu.data().unwrap(),
            rows,
            &self.ipiv,
            solution.data_mut().unwrap(),
            rows,
        )?;
        rhs.fill_from(&solution);
        Ok(())
    }
}

impl<Item> LuDecomposition<Item>
where
    Item: Lapack,
{
    /// Copy these factors into an owned reusable square-factor object.
    ///
    /// This adapter is useful when factors must outlive the decomposition that
    /// produced them. New code that keeps the decomposition should prefer
    /// [`Self::solve_in_place`] and avoid the copy.
    pub fn to_owned_square_factors(&self) -> RlstResult<SquareLuFactors<Item>> {
        SquareLuFactors::from_lu(self)
    }

    /// Solve `op(A) x = b` in place from the stored factors.
    ///
    /// The factorization remains unchanged and `b` is overwritten by the
    /// solution. One- and two-dimensional owned arrays are supported.
    pub fn solve_in_place<const NDIM: usize>(
        &self,
        trans: TransMode,
        b: &mut DynArray<Item, NDIM>,
    ) -> RlstResult<()> {
        if NDIM > 2 {
            return Err(RlstError::GeneralError(
                "LU solve is only implemented for 1D and 2D arrays.".to_string(),
            ));
        }
        assert_eq!(
            self.lu.shape()[0],
            self.lu.shape()[1],
            "Matrix must be square for LU solve."
        );
        assert_eq!(
            self.lu.shape()[0],
            b.shape()[0],
            "Right-hand side vector length does not match LU matrix size."
        );
        let trans = match trans {
            TransMode::NoTrans => GetrsTransMode::NoTranspose,
            TransMode::Trans => GetrsTransMode::Transpose,
            TransMode::ConjTrans => GetrsTransMode::ConjugateTranspose,
            TransMode::ConjNoTrans => {
                return Err(RlstError::GeneralError(
                    "conjugate-without-transpose LU solve is unsupported".to_string(),
                ));
            }
        };
        let rows = self.lu.shape()[0];
        let right_hand_sides = if NDIM == 1 { 1 } else { b.shape()[1] };
        Item::getrs(
            trans,
            rows,
            right_hand_sides,
            self.lu.data().unwrap(),
            rows,
            &self.ipiv,
            b.data_mut().unwrap(),
            rows,
        )?;
        Ok(())
    }

    /// Solve `op(A) x = b` from the stored factors.
    ///
    /// `trans` selects A, its transpose, or its conjugate transpose. The
    /// factorized matrix must be square. A vector RHS has length `n`; a matrix
    /// RHS has `n` rows. The RHS is copied and the factors are unchanged.
    ///
    /// # Errors
    /// Returns an error for RHS rank above two or a LAPACK solve failure.
    ///
    /// # Panics
    /// Panics for a nonsquare factorization, incompatible RHS shape, or an
    /// unsupported transpose mode.
    pub fn solve<ArrayImpl, const NDIM: usize>(
        &self,
        trans: TransMode,
        b: &Array<ArrayImpl, NDIM>,
    ) -> RlstResult<DynArray<Item, NDIM>>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let mut sol = DynArray::new_from(b);
        self.solve_in_place(trans, &mut sol)?;
        Ok(sol)
    }

    /// Return the L matrix of `A = P * L * U`.
    pub fn l_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut l_mat = DynArray::from_shape([m, k]);

        let lu_data = self.lu.data().unwrap();

        for col in 0..k {
            for row in col..m {
                if col == row {
                    unsafe { *l_mat.get_unchecked_mut([row, col]) = <Item as One>::one() };
                } else {
                    unsafe {
                        *l_mat.get_unchecked_mut([row, col]) =
                            *lu_data.get_unchecked(col * m + row);
                    };
                }
            }
        }

        Ok(l_mat)
    }

    /// Return the U matrix of `A = P * L * U`.
    pub fn u_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut u_mat = DynArray::from_shape([k, n]);

        let lu_data = self.lu.data().unwrap();

        for col in 0..n {
            for row in 0..=std::cmp::min(col, k - 1) {
                unsafe {
                    *u_mat.get_unchecked_mut([row, col]) = *lu_data.get_unchecked(col * m + row);
                }
            }
        }

        Ok(u_mat)
    }

    /// Return the P matrix of `A = P * L * U`.
    pub fn p_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, _] = self.lu.shape();
        let mut p_mat = DynArray::from_shape([m, m]);

        let perm = self.perm_vec()?;

        for (i, &j) in perm.iter().enumerate() {
            unsafe {
                *p_mat.get_unchecked_mut([j, i]) = <Item as One>::one();
            }
        }

        Ok(p_mat)
    }

    /// Return the zero-based permutation vector defining P.
    ///
    /// If `perm[i] = j`, row `i` of `L U` corresponds to row `j` of A.
    pub fn perm_vec(&self) -> RlstResult<Vec<usize>> {
        let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

        let mut perm = (0..self.lu.shape()[0]).collect::<Vec<_>>();

        for (index, &elem) in ipiv.iter().enumerate() {
            perm.swap(index, elem);
        }

        Ok(perm)
    }

    /// Return the determinant of A from its factors.
    ///
    /// # Panics
    /// Panics unless the factorized matrix is square (and currently also for
    /// an empty `0 x 0` factorization).
    pub fn det(&self) -> Item {
        let [m, n] = self.lu.shape();
        assert_eq!(m, n, "Matrix must be square to compute determinant.");
        let lu_data = self.lu.data().unwrap();
        let mut det = lu_data[0];
        if self.ipiv[0] != 1 {
            det = -det;
        }
        for i in 1..m {
            det *= unsafe { *lu_data.get_unchecked(i * m + i) };
            if self.ipiv[i] != (i + 1) as i32 {
                det = -det;
            }
        }
        det
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;

    use crate::MultInto;
    use crate::MultIntoResize;
    use crate::RlstScalar;
    use crate::empty_array;
    use paste::paste;
    #[test]
    fn reusable_factors_apply_and_solve_all_transpose_modes() {
        let mut matrix = DynArray::<c64, 2>::from_shape([3, 3]);
        matrix[[0, 0]] = c64::new(4.0, 1.0);
        matrix[[1, 0]] = c64::new(-1.0, 0.5);
        matrix[[2, 0]] = c64::new(0.5, -0.2);
        matrix[[0, 1]] = c64::new(1.0, -0.5);
        matrix[[1, 1]] = c64::new(3.0, 0.0);
        matrix[[2, 1]] = c64::new(-1.0, 0.3);
        matrix[[0, 2]] = c64::new(0.2, 0.1);
        matrix[[1, 2]] = c64::new(1.0, -0.4);
        matrix[[2, 2]] = c64::new(2.5, 0.7);
        let factors = SquareLuFactors::from_lu(&matrix.lu().unwrap()).unwrap();

        let mut input = DynArray::<c64, 2>::from_shape([3, 2]);
        input.fill_from_seed_normally_distributed(19);
        for mode in [
            TransMode::NoTrans,
            TransMode::ConjNoTrans,
            TransMode::Trans,
            TransMode::ConjTrans,
        ] {
            let mut expected = DynArray::<c64, 2>::from_shape([3, 2]);
            if mode == TransMode::ConjNoTrans {
                for column in 0..2 {
                    for row in 0..3 {
                        for inner in 0..3 {
                            expected[[row, column]] +=
                                matrix[[row, inner]].conj() * input[[inner, column]];
                        }
                    }
                }
            } else {
                expected.r_mut().mult_into(
                    mode,
                    TransMode::NoTrans,
                    c64::new(1.0, 0.0),
                    matrix.r(),
                    input.r(),
                    c64::new(0.0, 0.0),
                );
            }
            let mut actual = DynArray::new_from(&input);
            factors.mul_mat(mode, actual.r_mut()).unwrap();
            crate::assert_array_relative_eq!(actual, expected, 1.0e-12);

            if mode != TransMode::ConjNoTrans {
                factors.solve_mat(mode, actual.r_mut()).unwrap();
                crate::assert_array_relative_eq!(actual, input, 1.0e-12);
            }
        }
    }

    macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([8, 20]);

                    arr.fill_from_seed_normally_distributed(0);

                    let lu = DynArray::new_from(&arr).lu().unwrap();


                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        crate::empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_square_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 12]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_solve_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 12]);
                    arr.fill_from_seed_equally_distributed(0);
                    let mut x_expected = DynArray::<$scalar, 1>::from_shape([dim[0]]);
                    x_expected.fill_from_seed_equally_distributed(1);
                    let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());

                    let x_actual = arr.lu().unwrap().solve(TransMode::NoTrans, &rhs).unwrap();

                    crate::assert_array_relative_eq!(x_actual, x_expected, $tol)
                }



                #[test]
                fn [<test_lu_thin_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 8]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_det_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([2, 2]);
                    arr[[0, 1]] = $scalar::from_real(3.0);
                    arr[[1, 0]] = $scalar::from_real(2.0);

                    let det = arr.lu().unwrap().det();

                    approx::assert_relative_eq!(det, $scalar::from_real(-6.0), epsilon=$tol);
                }



            }
        };
    }

    impl_lu_tests!(f64, 1E-12);
    impl_lu_tests!(f32, 1E-4);
    impl_lu_tests!(c64, 1E-12);
    impl_lu_tests!(c32, 1E-4);
}
