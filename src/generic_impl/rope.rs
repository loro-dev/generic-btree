use core::ops::{Range, RangeBounds};
extern crate alloc;

use alloc::string::{String, ToString};
use smallvec::SmallVec;

use crate::{BTree, BTreeTrait, FindResult, HeapVec, LengthFinder};

use super::len_finder::UseLengthFinder;

#[derive(Debug)]
struct RopeTrait;

#[derive(Debug)]
pub struct Rope {
    tree: BTree<RopeTrait>,
}

impl UseLengthFinder<RopeTrait> for RopeTrait {
    fn get_len(cache: &<Self as BTreeTrait>::Cache) -> usize {
        *cache
    }

    fn find_element_by_offset(
        elements: &[<Self as BTreeTrait>::Elem],
        offset: usize,
    ) -> crate::FindResult {
        if offset < elements.len() {
            FindResult::new_found(offset, 0)
        } else {
            FindResult::new_missing(elements.len(), offset - elements.len())
        }
    }
}

impl Rope {
    #[inline]
    pub fn len(&self) -> usize {
        self.tree.root_cache
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, index: usize, elem: &str) {
        let result = self.tree.query::<LengthFinder>(&index);
        self.tree
            .batch_insert_by_query_result(result, elem.chars().collect::<SmallVec<[char; 16]>>());
    }

    pub fn delete_range(&mut self, range: impl RangeBounds<usize>) {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(&x) => x + 1,
            core::ops::Bound::Excluded(&x) => x,
            core::ops::Bound::Unbounded => self.len(),
        };
        self.tree.drain::<LengthFinder>(start..end);
    }

    pub fn iter(&self) -> impl Iterator<Item = &char> {
        self.tree.iter()
    }

    pub fn iter_range(&self, range: Range<usize>) -> impl Iterator<Item = char> + '_ {
        self.tree
            .iter_range(self.tree.range::<LengthFinder>(range))
            .map(|x| *x.elem)
    }

    pub fn new() -> Self {
        Self { tree: BTree::new() }
    }

    pub fn node_len(&self) -> usize {
        self.tree.node_len()
    }

    fn update_in_place(&mut self, pos: usize, new: &str) {
        let mut iter = new.chars();
        let start = self.tree.query::<LengthFinder>(&pos);
        let end = self.tree.query::<LengthFinder>(&(pos + new.len()));
        self.tree.update::<_>(start..end, &mut |slice| {
            let start = slice.start.map(|x| x.0).unwrap_or(0);
            let end = slice.end.map(|x| x.0).unwrap_or(slice.elements.len());
            for c in slice.elements[start..end].iter_mut() {
                *c = iter.next().unwrap();
            }
            false
        });
    }

    pub fn clear(&mut self) {
        self.tree.clear();
    }

    pub fn check(&self) {
        self.tree.check()
    }
}

impl Default for Rope {
    fn default() -> Self {
        Self::new()
    }
}

impl ToString for Rope {
    fn to_string(&self) -> String {
        let mut ans = String::with_capacity(self.len());
        for &s in self.iter() {
            ans.push(s);
        }

        ans
    }
}

impl BTreeTrait for RopeTrait {
    type Elem = char;
    type WriteBuffer = ();
    type Cache = usize;

    const MAX_LEN: usize = 32;

    fn element_to_cache(_: &Self::Elem) -> Self::Cache {
        1
    }

    fn calc_cache_internal(caches: &[crate::Child<Self>]) -> Self::Cache {
        caches.iter().map(|x| x.cache).sum::<usize>()
    }

    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache {
        elements.len()
    }

    fn insert(elements: &mut HeapVec<Self::Elem>, index: usize, _: usize, elem: Self::Elem) {
        elements.insert(index, elem);
    }

    fn insert_batch(
        elements: &mut HeapVec<Self::Elem>,
        index: usize,
        _: usize,
        elem: impl IntoIterator<Item = Self::Elem>,
    ) where
        Self::Elem: Clone,
    {
        elements.insert_many(index, elem.into_iter());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let mut rope = Rope::new();
        rope.insert(0, "123");
        rope.insert(1, "x");
        assert_eq!(rope.len(), 4);
        rope.delete_range(2..4);
        assert_eq!(&rope.to_string(), "1x");
        rope.delete_range(..1);
        assert_eq!(&rope.to_string(), "x");
        rope.delete_range(..);
        assert_eq!(&rope.to_string(), "");
        assert_eq!(rope.len(), 0);
    }

    #[test]
    fn test_update() {
        let mut rope = Rope::new();
        rope.insert(0, "123");
        rope.insert(3, "xyz");
        rope.update_in_place(1, "kkkk");
        assert_eq!(&rope.to_string(), "1kkkkz");
    }

    #[test]
    fn test_clear() {
        let mut rope = Rope::new();
        rope.insert(0, "123");
        assert_eq!(rope.len(), 3);
        rope.clear();
        assert_eq!(rope.len(), 0);
        assert_eq!(&rope.to_string(), "");
        rope.insert(0, "kkk");
        assert_eq!(&rope.to_string(), "kkk");
    }

    #[test]
    fn test_update_1() {
        let mut rope = Rope::new();
        for i in 0..100 {
            rope.insert(i, &(i % 10).to_string());
        }

        rope.update_in_place(15, "kkkkk");
        assert_eq!(&rope.to_string()[10..20], "01234kkkkk");
    }

    #[derive(Debug)]
    enum Action {
        Insert { pos: u8, content: u8 },
        Delete { pos: u8, len: u8 },
    }

    fn fuzz(data: HeapVec<Action>) {
        let mut rope = Rope::new();
        let mut truth = String::new();
        for action in data {
            match action {
                Action::Insert { pos, content } => {
                    let pos = pos as usize % (truth.len() + 1);
                    let s = content.to_string();
                    truth.insert_str(pos, &s);
                    rope.insert(pos, &s);
                }
                Action::Delete { pos, len } => {
                    let pos = pos as usize % (truth.len() + 1);
                    let mut len = len as usize % 10;
                    len = len.min(truth.len() - pos);
                    // dbg!(&rope);
                    // dbg!(rope.to_string(), pos, len);
                    rope.delete_range(pos..(pos + len));
                    // dbg!(rope.to_string());
                    // dbg!(&rope);
                    truth.drain(pos..pos + len);
                }
            }

            rope.check();
        }

        assert_eq!(rope.to_string(), truth);
    }

    use ctor::ctor;
    use smallvec::smallvec as vec;
    use Action::*;

    #[test]
    fn fuzz_0() {
        fuzz(vec![
            Insert {
                pos: 0,
                content: 128,
            },
            Insert {
                pos: 0,
                content: 249,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 192, len: 193 },
            Insert {
                pos: 106,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 100,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 8 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 111,
                content: 127,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 36 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 135, len: 169 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
        ])
    }

    #[test]
    fn fuzz_1() {
        fuzz(vec![
            Insert {
                pos: 157,
                content: 108,
            },
            Insert {
                pos: 255,
                content: 255,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 8,
                content: 101,
            },
            Insert {
                pos: 111,
                content: 127,
            },
            Delete { pos: 255, len: 169 },
        ])
    }

    #[test]
    fn fuzz_2() {
        fuzz(vec![
            Insert {
                pos: 0,
                content: 128,
            },
            Insert {
                pos: 0,
                content: 249,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 0,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 249,
            },
            Insert {
                pos: 135,
                content: 255,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 169, len: 169 },
        ])
    }

    #[test]
    fn fuzz_empty() {
        fuzz(smallvec::smallvec![])
    }

    #[ctor]
    fn init_color_backtrace() {
        color_backtrace::install();
    }
}
