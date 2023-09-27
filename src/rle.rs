use core::ops::RangeBounds;
pub trait Sliceable<T: Copy = usize>: HasLength<T> {
    #[must_use]
    fn slice(&self, range: impl RangeBounds<T>) -> Self;
    /// slice in-place
    fn slice_(&mut self, range: impl RangeBounds<T>)
    where
        Self: Sized,
    {
        *self = self.slice(range);
    }

    #[must_use]
    fn split(&mut self, pos: T) -> Self
    where
        Self: Sized,
    {
        let right = self.slice(pos..);
        self.slice_(..pos);
        right
    }
}

pub trait Mergeable {
    /// Whether self can merge rhs with self on the left.
    ///
    /// Note: This is not symmetric.
    fn can_merge(&self, rhs: &Self) -> bool;
    fn merge_right(&mut self, rhs: &Self);
    fn merge_left(&mut self, left: &Self);
}

pub trait HasLength<T = usize> {
    fn rle_len(&self) -> T;
}
