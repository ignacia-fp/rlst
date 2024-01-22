//! Other operators that do not require special containers

use crate::array::operators::{addition::ArrayAddition, scalar_mult::ArrayScalarMult};
use crate::array::*;
use num::One;

impl<
        Item: RlstScalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > std::ops::Sub<Array<Item, ArrayImpl2, NDIM>> for Array<Item, ArrayImpl1, NDIM>
where
    Item: std::ops::Mul<
        Array<Item, ArrayImpl2, NDIM>,
        Output = Array<Item, ArrayScalarMult<Item, ArrayImpl2, NDIM>, NDIM>,
    >,
{
    type Output = Array<
        Item,
        ArrayAddition<Item, ArrayImpl1, ArrayScalarMult<Item, ArrayImpl2, NDIM>, NDIM>,
        NDIM,
    >;
    fn sub(self, rhs: Array<Item, ArrayImpl2, NDIM>) -> Self::Output {
        let minus_one = -<Item as One>::one();
        let minus_rhs = minus_one * rhs;
        self + minus_rhs
    }
}
