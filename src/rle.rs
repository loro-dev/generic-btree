use core::ops::RangeBounds;
use std::ops::Range;

/// For better performance, it's advised to impl split
pub trait Sliceable: HasLength + Sized {
    #[must_use]
    fn _slice(&self, range: Range<usize>) -> Self;

    #[must_use]
    #[inline(always)]
    fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.rle_len(),
        };

        self._slice(start..end)
    }
    /// slice in-place
    fn slice_(&mut self, range: impl RangeBounds<usize>) {
        *self = self.slice(range);
    }

    #[must_use]
    fn split(&mut self, pos: usize) -> Self {
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

pub trait HasLength {
    fn rle_len(&self) -> usize;
}
