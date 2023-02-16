use std::ops::RangeBounds;

use smallvec::SmallVec;

use crate::{BTree, BTreeTrait, FindResult, Query, SmallElemVec};

struct Finder {
    left: usize,
}

#[derive(Debug)]
struct RopeTrait;

#[derive(Debug)]
pub struct Rope {
    tree: BTree<RopeTrait>,
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
        let result = self.tree.query::<Finder>(&index);
        self.tree
            .batch_insert_by_query_result(result, &elem.chars().collect::<SmallVec<[char; 16]>>());
    }

    pub fn delete_range(&mut self, range: impl RangeBounds<usize>) {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
            std::ops::Bound::Unbounded => self.len(),
        };
        self.tree.drain::<Finder>(start..end);
    }

    pub fn iter(&self) -> impl Iterator<Item = &char> {
        self.tree.iter()
    }

    pub fn iter_range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = &char> {
        self.tree.iter_range::<Finder>(range)
    }

    pub fn new() -> Self {
        Self { tree: BTree::new() }
    }

    pub fn node_len(&self) -> usize {
        self.tree.node_len()
    }

    fn update_in_place(&mut self, pos: usize, new: &str) {
        let mut iter = new.chars();
        self.tree
            .update::<Finder, _>(pos..pos + new.len(), &mut |slice| {
                for c in slice.elements.iter_mut() {
                    *c = iter.next().unwrap();
                }
                false
            });
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

    type Cache = usize;

    const MAX_LEN: usize = 128;

    fn element_to_cache(_: &Self::Elem) -> Self::Cache {
        1
    }

    fn calc_cache_internal(caches: &[crate::Child<Self::Cache>]) -> Self::Cache {
        caches.iter().map(|x| x.cache).sum::<usize>()
    }

    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache {
        elements.len()
    }

    fn insert(elements: &mut Vec<Self::Elem>, index: usize, _: usize, elem: Self::Elem) {
        elements.insert(index, elem);
    }

    fn insert_batch(elements: &mut Vec<Self::Elem>, index: usize, _: usize, elem: &[Self::Elem])
    where
        Self::Elem: Clone,
    {
        elements.splice(index..index, elem.iter().cloned());
    }
}

impl Query<RopeTrait> for Finder {
    type QueryArg = usize;

    fn find_node(
        &mut self,
        _: &Self::QueryArg,
        child_caches: &[crate::Child<usize>],
    ) -> FindResult {
        for (i, cache) in child_caches.iter().enumerate() {
            if self.left > cache.cache {
                self.left -= cache.cache;
            } else {
                return FindResult::new_found(i, self.left);
            }
        }

        FindResult::new_missing(child_caches.len(), self.left)
    }

    fn find_element(&mut self, _: &Self::QueryArg, elements: &[char]) -> crate::FindResult {
        if self.left >= elements.len() {
            self.left -= elements.len();
            return FindResult::new_missing(elements.len(), self.left);
        }

        FindResult::new_found(self.left, 0)
    }

    fn init(target: &Self::QueryArg) -> Self {
        Self { left: *target }
    }

    fn delete(
        elements: &mut Vec<char>,
        _: &Self::QueryArg,
        elem_index: usize,
        _: usize,
    ) -> Option<char> {
        if elem_index >= elements.len() {
            return None;
        }

        if elem_index < elements.len() {
            Some(elements.remove(elem_index))
        } else {
            None
        }
    }

    fn delete_range(
        elements: &mut Vec<char>,
        _: &Self::QueryArg,
        _: &Self::QueryArg,
        start: Option<crate::QueryResult>,
        end: Option<crate::QueryResult>,
    ) -> SmallElemVec<char> {
        fn drain_start(start: crate::QueryResult, elements: &mut [char]) -> usize {
            if start.offset == 0 || start.elem_index >= elements.len() {
                start.elem_index
            } else if start.offset == 1 {
                start.elem_index + 1
            } else {
                unreachable!()
            }
        }

        fn drain_end(end: crate::QueryResult, elements: &mut [char]) -> usize {
            if end.offset == 0 || end.elem_index >= elements.len() {
                end.elem_index
            } else if 1 == end.offset {
                end.elem_index + 1
            } else {
                unreachable!()
            }
        }

        if elements.is_empty() {
            return SmallElemVec::new();
        }

        match (start, end) {
            (None, None) => elements.drain(..).collect(),
            (None, Some(end)) => {
                let end = drain_end(end, elements);
                elements.drain(..end).collect()
            }
            (Some(start), None) => {
                let start = drain_start(start, elements);
                elements.drain(start..).collect()
            }
            (Some(start), Some(end)) => {
                if start.elem_index == end.elem_index {
                    if elements.len() <= start.elem_index || start.offset == end.offset {
                        SmallElemVec::new()
                    } else {
                        assert_eq!(start.offset, 0);
                        assert_eq!(end.offset, 1);
                        let mut ans = SmallElemVec::new();
                        ans.push(elements[start.elem_index]);
                        elements.remove(start.elem_index);
                        ans
                    }
                } else {
                    let start = drain_start(start, elements);
                    let end = drain_end(end, elements);
                    elements.drain(start..end).collect()
                }
            }
        }
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

    #[derive(Debug)]
    enum Action {
        Insert { pos: u8, content: u8 },
        Delete { pos: u8, len: u8 },
    }

    fn fuzz(data: Vec<Action>) {
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
        fuzz(vec![])
    }

    #[ctor]
    fn init_color_backtrace() {
        color_backtrace::install();
    }
}
