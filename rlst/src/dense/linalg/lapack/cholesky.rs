//! Implement Cholesky decomposition and solve for positive definite matrices.

use crate::{
    UnsafeRandom1DAccessByValue,
    base_types::{RlstResult, UpLo},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::{potrf::PotrfUplo, potrs::PotrsUplo},
    },
    traits::{
        base_operations::Shape,
        linalg::{
            decompositions::{Cholesky, CholeskySolve},
            lapack::Lapack,
        },
    },
};

/// Reusable Cholesky factors of a Hermitian positive-definite matrix.
pub struct CholeskyDecomposition<Item: Lapack> {
    factor: DynArray<Item, 2>,
    uplo: UpLo,
}

impl<Item: Lapack> CholeskyDecomposition<Item> {
    /// Return the triangular factor.
    pub fn factor(&self) -> &DynArray<Item, 2> {
        &self.factor
    }

    /// Consume the decomposition and return its triangular factor.
    pub fn into_factor(self) -> DynArray<Item, 2> {
        self.factor
    }

    /// Overwrite one or more right-hand sides with the solution.
    pub fn solve_in_place<const NDIM: usize>(
        &self,
        rhs: &mut DynArray<Item, NDIM>,
    ) -> RlstResult<()> {
        assert!(
            NDIM == 1 || NDIM == 2,
            "RHS must be one- or two-dimensional"
        );
        let n = self.factor.shape()[0];
        assert_eq!(rhs.shape()[0], n, "RHS row count must match the factor");
        let nrhs = if NDIM == 1 { 1 } else { rhs.shape()[1] };
        let uplo = match self.uplo {
            UpLo::Upper => PotrsUplo::Upper,
            UpLo::Lower => PotrsUplo::Lower,
        };
        Item::potrs(
            uplo,
            n,
            nrhs,
            self.factor.data().unwrap(),
            n,
            rhs.data_mut().unwrap(),
            n,
        )?;
        Ok(())
    }

    /// Copy `rhs`, solve the system, and return the solution.
    pub fn solve<ArrayImpl, const NDIM: usize>(
        &self,
        rhs: &Array<ArrayImpl, NDIM>,
    ) -> RlstResult<DynArray<Item, NDIM>>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let mut solution = DynArray::new_from(rhs);
        self.solve_in_place(&mut solution)?;
        Ok(solution)
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    /// Compute reusable Cholesky factors.
    pub fn cholesky_decomposition(&self, uplo: UpLo) -> RlstResult<CholeskyDecomposition<Item>> {
        let [m, n] = self.shape();
        assert_eq!(m, n, "Cholesky decomposition requires a square matrix");
        let mut factor = DynArray::new_from(self);
        let potrf_uplo = match uplo {
            UpLo::Upper => PotrfUplo::Upper,
            UpLo::Lower => PotrfUplo::Lower,
        };
        Item::potrf(potrf_uplo, m, factor.data_mut().unwrap(), m)?;

        // Keep the returned object a genuine triangular matrix, rather than
        // exposing LAPACK's unused triangle containing stale input values.
        match uplo {
            UpLo::Upper => {
                for column in 0..n {
                    for row in column + 1..m {
                        factor[[row, column]] = Item::zero();
                    }
                }
            }
            UpLo::Lower => {
                for column in 0..n {
                    for row in 0..column {
                        factor[[row, column]] = Item::zero();
                    }
                }
            }
        }
        Ok(CholeskyDecomposition { factor, uplo })
    }
}

impl<Item, ArrayImpl> Cholesky for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    type Item = Item;

    fn cholesky(&self, uplo: UpLo) -> RlstResult<DynArray<Self::Item, 2>> {
        Ok(self.cholesky_decomposition(uplo)?.into_factor())
    }
}

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> CholeskySolve<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    RhsArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: Lapack,
{
    type Output = DynArray<Item, NDIM>;

    fn cholesky_solve(
        &self,
        uplo: UpLo,
        rhs: &Array<RhsArrayImpl, NDIM>,
    ) -> RlstResult<Self::Output> {
        self.cholesky_decomposition(uplo)?.solve(rhs)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot_blas;
    use crate::traits::base_operations::*;
    use paste::paste;

    macro_rules! implement_cholesky_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_cholesky_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_normally_distributed(0);

                // Make it symmetric positive definite
                a = dot_blas!(a.r().conj().transpose().eval(), a.r());

                let z = a.cholesky(UpLo::Upper).unwrap();

                let actual = dot_blas!(z.r().conj().transpose().eval(), z.r());

                crate::assert_array_abs_diff_eq!(actual, a, $tol);

                // Now solve a linear system with Cholesky

                let mut x_expected = DynArray::<$scalar, 2>::from_shape([n, 2]);
                x_expected.fill_from_seed_equally_distributed(1);

                let b = dot_blas!(a.r(), x_expected.r());

                let x_actual = a.cholesky_solve(UpLo::Upper, &b).unwrap();

                crate::assert_array_abs_diff_eq!(x_actual, x_expected, $tol);

                let factors = a.cholesky_decomposition(UpLo::Upper).unwrap();
                let mut x_reused = DynArray::new_from(&b);
                factors.solve_in_place(&mut x_reused).unwrap();
                crate::assert_array_abs_diff_eq!(x_reused, x_expected, $tol);

                // Reusing the same factors must not mutate them.
                let x_reused_again = factors.solve(&b).unwrap();
                crate::assert_array_abs_diff_eq!(x_reused_again, x_expected, $tol);
            }

                    }
        };
    }

    implement_cholesky_test!(f32, 1E-3);
    implement_cholesky_test!(f64, 1E-10);
    implement_cholesky_test!(c32, 1E-3);
    implement_cholesky_test!(c64, 1E-10);
}
