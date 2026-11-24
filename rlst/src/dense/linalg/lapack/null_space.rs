//! Dense null-space bases computed with LAPACK.

use num::Zero;

use crate::{
    RlstScalar, Shape, UnsafeRandom1DAccessByValue,
    base_types::RlstResult,
    dense::{
        array::{Array, DynArray},
        linalg::lapack::{
            interface::gesvd::{JobU, JobVt},
            qr::{EnablePivoting, QMode},
        },
    },
    traits::linalg::{base::Gemm, decompositions::Qr, lapack::Lapack},
};

/// Which null space to construct and which factorization to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullSpaceMethod {
    /// Right null space, `A N = 0`, from right singular vectors.
    RightSvd,
    /// Left null space, `A^H N = 0`, from a complete pivoted QR basis.
    LeftQr,
}

/// Method used to construct the null space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Right null space from a full singular-value decomposition.
    Svd,
    /// Left null space (the orthogonal complement of the columns) from QR.
    ///
    /// This intentionally preserves the convention used by the historical
    /// RSRS fork. Callers wanting a right null space with QR should transpose
    /// the input first.
    Qr,
}

/// Result of a null-space computation.
pub struct NullSpace<Item: RlstScalar> {
    /// Orthonormal basis vectors stored as columns.
    pub null_space_arr: DynArray<Item, 2>,
}

impl<Item, ArrayImpl> Array<ArrayImpl, 2>
where
    Item: Lapack + Gemm,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    /// Compute an orthonormal null-space basis.
    pub fn null_space(
        &self,
        tolerance: Item::Real,
        method: NullSpaceMethod,
    ) -> RlstResult<NullSpace<Item>> {
        assert!(
            tolerance >= Item::Real::zero(),
            "null-space tolerance must be nonnegative"
        );
        let [rows, columns] = self.shape();
        let null_space_arr = match method {
            NullSpaceMethod::RightSvd => {
                // GESVD permits requesting V^H without U. The historical
                // implementation used a full divide-and-conquer SVD and
                // allocated an unused m x m U matrix.
                let mut work = DynArray::new_from(self);
                let diagonal = rows.min(columns);
                let mut singular_values = DynArray::<Item::Real, 1>::from_shape([diagonal]);
                let mut vh = DynArray::<Item, 2>::from_shape([columns, columns]);
                Item::gesvd(
                    JobU::N,
                    JobVt::A,
                    rows,
                    columns,
                    work.data_mut().unwrap(),
                    rows,
                    singular_values.data_mut().unwrap(),
                    None,
                    1,
                    vh.data_mut(),
                    columns,
                )?;
                let rank = relative_rank::<Item>(&singular_values, tolerance);
                let mut basis = DynArray::<Item, 2>::from_shape([columns, columns - rank]);
                for vector in rank..columns {
                    for component in 0..columns {
                        basis[[component, vector - rank]] = vh[[vector, component]].conj();
                    }
                }
                basis
            }
            NullSpaceMethod::LeftQr => {
                let qr = self.qr(EnablePivoting::Yes)?;
                let r = qr.r_mat()?;
                let rank = diagonal_rank(&r, tolerance);
                let q = qr.q_mat(QMode::Full)?;
                let mut basis = DynArray::<Item, 2>::from_shape([rows, rows - rank]);
                for vector in rank..rows {
                    for component in 0..rows {
                        basis[[component, vector - rank]] = q[[component, vector]];
                    }
                }
                basis
            }
        };
        Ok(NullSpace { null_space_arr })
    }

    /// Compute a null-space basis using the historical RSRS conventions.
    pub fn into_null_alloc(
        &self,
        tolerance: Item::Real,
        method: Method,
    ) -> RlstResult<NullSpace<Item>> {
        let method = match method {
            Method::Svd => NullSpaceMethod::RightSvd,
            Method::Qr => NullSpaceMethod::LeftQr,
        };
        self.null_space(tolerance, method)
    }
}

fn relative_rank<Item: RlstScalar>(
    values: &DynArray<Item::Real, 1>,
    tolerance: Item::Real,
) -> usize {
    let maximum = values
        .iter_value()
        .fold(Item::Real::zero(), num::Float::max);
    if maximum == Item::Real::zero() {
        return 0;
    }
    values
        .iter_value()
        .filter(|&value| value > tolerance * maximum)
        .count()
}

fn diagonal_rank<Item: RlstScalar>(r: &DynArray<Item, 2>, tolerance: Item::Real) -> usize {
    let diagonal = r.shape()[0].min(r.shape()[1]);
    let maximum = (0..diagonal)
        .map(|index| r[[index, index]].abs())
        .fold(Item::Real::zero(), num::Float::max);
    if maximum == Item::Real::zero() {
        return 0;
    }
    (0..diagonal)
        .filter(|&index| r[[index, index]].abs() > tolerance * maximum)
        .count()
}

#[cfg(test)]
mod tests {
    use super::Method;
    use crate::{MultInto, TransMode, dense::array::DynArray};

    #[test]
    fn svd_returns_right_null_space_for_rectangular_matrix() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 3]);
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 1.0;
        let null = a
            .into_null_alloc(1.0e-12, Method::Svd)
            .unwrap()
            .null_space_arr;
        assert_eq!(null.shape(), [3, 1]);
        let mut residual = DynArray::<f64, 2>::from_shape([2, 1]);
        residual.r_mut().mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            1.0,
            a.r(),
            null.r(),
            0.0,
        );
        assert!(residual.iter_value().all(|value| value.abs() < 1.0e-12));
    }

    #[test]
    fn qr_returns_left_null_space_and_handles_rank_deficiency() {
        let mut a = DynArray::<f64, 2>::from_shape([4, 2]);
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 1.0;
        let null = a
            .into_null_alloc(1.0e-12, Method::Qr)
            .unwrap()
            .null_space_arr;
        assert_eq!(null.shape(), [4, 2]);
        let mut residual = DynArray::<f64, 2>::from_shape([2, 2]);
        residual.r_mut().mult_into(
            TransMode::Trans,
            TransMode::NoTrans,
            1.0,
            a.r(),
            null.r(),
            0.0,
        );
        assert!(residual.iter_value().all(|value| value.abs() < 1.0e-12));
    }
}
