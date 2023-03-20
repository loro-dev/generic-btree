use std::{ops::Range, usize};

use generic_btree::{
    rle::{
        delete_range_in_elements, scan_and_merge, update_slice, HasLength, Mergeable, Sliceable,
    },
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
        let range = self.0.range::<LengthFinder>(range);
        self.0.update(&range.start..&range.end, &mut |mut slice| {
            let before_len = if cfg!(debug_assert) {
                slice.elements.iter().map(|x| x.len).sum()
            } else {
                0
            };
            let ans = update_slice::<Elem, _>(&mut slice, &mut |x| {
                x.value = Some(value);
                false
            });
            scan_and_merge(slice.elements, slice.start.map(|x| x.0).unwrap_or(0));
            if cfg!(debug_assert) {
                let after_len = slice.elements.iter().map(|x| x.len).sum();
                assert_eq!(before_len, after_len);
            }
            (ans, None)
        });
    }

    pub fn get(&mut self, index: usize) -> Option<isize> {
        let result = self.0.query::<LengthFinder>(&index);
        self.0.get_elem(&result).and_then(|x| x.value)
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
        let range = self.0.range::<LengthFinder>(range);
        self.0.update(&range.start..&range.end, &mut |mut slice| {
            let before_len = if cfg!(debug_assert) {
                slice.elements.iter().map(|x| x.len).sum()
            } else {
                0
            };
            let ans = update_slice::<Elem, _>(&mut slice, &mut |x| {
                if let Some(v) = &mut x.value {
                    *v += change;
                }
                false
            });
            scan_and_merge(slice.elements, slice.start.map(|x| x.0).unwrap_or(0));
            if cfg!(debug_assert) {
                let after_len = slice.elements.iter().map(|x| x.len).sum();
                assert_eq!(before_len, after_len);
            }
            (ans, None)
        });
    }

    fn reserve_range(&mut self, range: &Range<usize>) {
        if self.len() < range.end {
            self.0.insert::<LengthFinder>(
                &range.end,
                Elem {
                    value: None,
                    len: range.end - self.len() + 10,
                },
            );
        }
    }

    pub fn drain(
        &mut self,
        range: Range<usize>,
    ) -> impl Iterator<Item = (Range<usize>, isize)> + '_ {
        let mut index = range.start;
        self.0.drain::<LengthFinder>(range).filter_map(move |elem| {
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

    fn find_element_by_offset(elements: &[Elem], offset: usize) -> generic_btree::FindResult {
        let mut left = offset;
        for (i, elem) in elements.iter().enumerate() {
            if left >= elem.len {
                left -= elem.len;
            } else {
                return generic_btree::FindResult::new_found(i, left);
            }
        }

        generic_btree::FindResult::new_missing(elements.len(), left)
    }

    #[inline]
    fn finder_drain_range(
        elements: &mut generic_btree::HeapVec<<RangeNumMapTrait as BTreeTrait>::Elem>,
        start: Option<generic_btree::QueryResult>,
        end: Option<generic_btree::QueryResult>,
    ) -> Box<dyn Iterator<Item = Elem> + '_> {
        Box::new(delete_range_in_elements(elements, start, end).into_iter())
    }

    fn finder_delete_range(
        elements: &mut generic_btree::HeapVec<<RangeNumMapTrait as BTreeTrait>::Elem>,
        start: Option<generic_btree::QueryResult>,
        end: Option<generic_btree::QueryResult>,
    ) {
        delete_range_in_elements(elements, start, end);
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
    fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> Self {
        let len = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len,
        } - match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        Elem {
            value: self.value,
            len,
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

    const MAX_LEN: usize = 8;

    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[generic_btree::Child<Self>],
        diff: Option<isize>,
    ) -> Option<isize> {
        match diff {
            Some(diff) => {
                *cache = (*cache as isize + diff) as usize;
                Some(diff)
            }
            None => {
                let new_cache = caches.iter().map(|c| c.cache).sum();
                let diff = new_cache as isize - *cache as isize;
                *cache = new_cache;
                Some(diff)
            }
        }
    }

    fn calc_cache_leaf(
        cache: &mut Self::Cache,
        elements: &[Self::Elem],
        _: Option<Self::CacheDiff>,
    ) -> isize {
        let new_cache = elements.iter().map(|c| c.len).sum();
        let diff = new_cache as isize - *cache as isize;
        *cache = new_cache;
        diff
    }

    type CacheDiff = isize;

    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff) {
        *diff1 += diff2;
    }
}
