use std::{ops::Range, usize};

use generic_btree::{
    rle::{HasLength, Mergeable, Sliceable},
    BTree, BTreeTrait, LengthFinder, UseLengthFinder,
};

/// This struct keep the mapping of ranges to numbers
#[derive(Debug)]
pub struct RangeNumMap(BTree<RangeNumMapTrait>);

struct RangeNumMapTrait;

#[derive(Debug)]
struct Elem {
    value: Option<isize>,
    len: usize,
}

impl RangeNumMap {
    pub fn new() -> Self {
        Self(BTree::new())
    }

    pub fn insert(&mut self, range: Range<usize>, value: isize) {
        self.reserve_range(&range);
        if let Some(range) = self.0.range::<LengthFinder>(range.clone()) {
            self.0
                .update(range.start.cursor..range.end.cursor, &mut |slice| {
                    slice.value = Some(value);
                    None
                });
        } else {
            assert_eq!(range.start, 0);
            self.0.push(Elem {
                value: Some(value),
                len: range.len(),
            });
        }
    }

    pub fn get(&mut self, index: usize) -> Option<isize> {
        let Some(result) = self.0.query::<LengthFinder>(&index) else {
            return None;
        };

        if !result.found {
            return None;
        }

        self.0.get_elem(result.leaf()).and_then(|x| x.value)
    }

    pub fn iter(&mut self) -> impl Iterator<Item = (Range<usize>, isize)> + '_ {
        let mut index = 0;
        self.0.iter().filter_map(move |elem| {
            let len = elem.len;
            let value = elem.value?;
            let range = index..index + len;
            index += len;
            Some((range, value))
        })
    }

    pub fn plus(&mut self, range: Range<usize>, change: isize) {
        self.reserve_range(&range);
        if let Some(range) = self.0.range::<LengthFinder>(range) {
            self.0
                .update(range.start.cursor..range.end.cursor, &mut |slice| {
                    if let Some(v) = &mut slice.value {
                        *v += change;
                    }

                    None
                });
        }
    }

    fn reserve_range(&mut self, range: &Range<usize>) {
        if self.len() < range.end {
            self.0.push(Elem {
                value: None,
                len: range.end - self.len() + 10,
            });
        }
    }

    pub fn drain(
        &mut self,
        range: Range<usize>,
    ) -> impl Iterator<Item = (Range<usize>, isize)> + '_ {
        let mut index = range.start;
        let self1 = &self.0;
        let from = self1.query::<LengthFinder>(&range.start);
        let to = self1.query::<LengthFinder>(&range.end);
        generic_btree::iter::Drain::new(&mut self.0, from, to).filter_map(move |elem| {
            let len = elem.len;
            let value = elem.value?;
            let range = index..index + len;
            index += len;
            Some((range, value))
        })
    }

    pub fn len(&self) -> usize {
        *self.0.root_cache()
    }
}

impl UseLengthFinder<RangeNumMapTrait> for RangeNumMapTrait {
    fn get_len(cache: &usize) -> usize {
        *cache
    }
}

impl HasLength for Elem {
    fn rle_len(&self) -> usize {
        self.len
    }
}

impl Mergeable for Elem {
    fn can_merge(&self, rhs: &Self) -> bool {
        self.value == rhs.value || rhs.len == 0
    }

    fn merge_right(&mut self, rhs: &Self) {
        self.len += rhs.len
    }

    fn merge_left(&mut self, left: &Self) {
        self.len += left.len;
    }
}

impl Sliceable for Elem {
    fn _slice(&self, range: Range<usize>) -> Self {
        Elem {
            value: self.value,
            len: range.len(),
        }
    }

    fn slice_(&mut self, range: impl std::ops::RangeBounds<usize>)
    where
        Self: Sized,
    {
        let len = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len,
        } - match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };

        self.len = len;
    }
}

impl BTreeTrait for RangeNumMapTrait {
    /// value
    type Elem = Elem;
    /// len
    type Cache = usize;

    type CacheDiff = isize;

    #[inline(always)]
    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[generic_btree::Child<Self>],
    ) -> isize {
        let new_cache = caches.iter().map(|c| c.cache).sum();
        let diff = new_cache as isize - *cache as isize;
        *cache = new_cache;
        diff
    }

    #[inline(always)]
    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff) {
        *diff1 += diff2;
    }

    #[inline(always)]
    fn get_elem_cache(elem: &Self::Elem) -> Self::Cache {
        elem.len
    }

    #[inline(always)]
    fn apply_cache_diff(cache: &mut Self::Cache, diff: &Self::CacheDiff) {
        *cache = (*cache as isize + diff) as usize;
    }

    #[inline(always)]
    fn new_cache_to_diff(cache: &Self::Cache) -> Self::CacheDiff {
        *cache as isize
    }

    fn sub_cache(cache_lhs: &Self::Cache, cache_rhs: &Self::Cache) -> Self::CacheDiff {
        *cache_lhs as isize - *cache_rhs as isize
    }
}
