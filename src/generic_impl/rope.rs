use core::ops::{Range, RangeBounds};
use std::mem::take;
extern crate alloc;

use alloc::string::{String, ToString};

use crate::{BTree, BTreeTrait, FindResult, HeapVec, LengthFinder};

const MAX_ELEM_SIZE: usize = 800;

use super::len_finder::UseLengthFinder;

#[derive(Debug)]
struct RopeTrait;

#[derive(Debug, Clone)]
struct RopeElem {
    left: Vec<u8>,
    /// this stored the reversed u8 of the right part
    right: Vec<u8>,
}

impl RopeElem {
    #[inline(always)]
    fn len(&self) -> usize {
        self.left.len() + self.right.len()
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.left.capacity()
    }

    #[inline(always)]
    fn new() -> Self {
        Self {
            left: Vec::new(),
            right: Vec::new(),
        }
    }

    fn from_str(s: &str) -> Self {
        let mut left = Vec::with_capacity(closest_2_power(s.len()));
        let right = Vec::new();
        for c in s.as_bytes() {
            left.push(*c);
        }
        Self { left, right }
    }

    #[inline(always)]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            left: Vec::with_capacity(capacity),
            right: Vec::with_capacity(capacity),
        }
    }

    fn insert(&mut self, pos: usize, new: &[u8]) {
        self.shift_at(pos);
        self.left.extend_from_slice(new);
    }

    fn push(&mut self, new: u8) {
        self.shift_at(self.len());
        self.left.push(new);
    }

    fn delete(&mut self, range: impl RangeBounds<usize>) -> usize {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&pos) => pos,
            std::ops::Bound::Excluded(&pos) => pos + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&pos) => pos + 1,
            std::ops::Bound::Excluded(&pos) => pos,
            std::ops::Bound::Unbounded => self.len(),
        };
        self.shift_at(end);
        let mut deleted = 0;
        let len = end - start;
        while deleted < len && !self.left.is_empty() {
            self.left.pop();
            deleted += 1;
        }
        deleted
    }

    fn drain(&mut self, range: Range<usize>) -> Vec<u8> {
        self.shift_at(range.start);
        let mut deleted = 0;
        let mut result = Vec::with_capacity(range.len());
        while deleted < range.len() && !self.right.is_empty() {
            result.push(self.right.pop().unwrap());
            deleted += 1;
        }
        result
    }

    fn split_right(&mut self, at: usize) -> Self {
        self.shift_at(at);
        let mut next_left = take(&mut self.right);
        next_left.reverse();
        Self {
            left: next_left,
            right: Vec::new(),
        }
    }

    fn split_left(&mut self, at: usize) -> Self {
        self.shift_at(at);
        let prev_left = take(&mut self.left);
        Self {
            left: prev_left,
            right: Vec::new(),
        }
    }

    fn shift_at(&mut self, pos: usize) {
        match pos.cmp(&self.left.len()) {
            std::cmp::Ordering::Less => {
                if self.right.capacity() < self.left.capacity() {
                    self.right
                        .reserve(self.left.capacity() - self.right.capacity());
                }
                while pos != self.left.len() {
                    self.right.push(self.left.pop().unwrap())
                }
            }
            std::cmp::Ordering::Greater => {
                if self.left.capacity() < self.right.capacity() {
                    self.left
                        .reserve(self.right.capacity() - self.left.capacity());
                }
                while pos != self.left.len() {
                    self.left.push(self.right.pop().unwrap())
                }
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    #[inline]
    fn get(&self, pos: usize) -> u8 {
        if pos < self.left.len() {
            self.left[pos]
        } else {
            self.right[self.right.len() - (pos - self.left.len()) - 1]
        }
    }

    #[inline]
    fn set(&mut self, pos: usize, value: u8) {
        if pos < self.left.len() {
            self.left[pos] = value;
        } else {
            let len = self.right.len();
            self.right[len - (pos - self.left.len())] = value;
        }
    }

    #[inline]
    fn append(&mut self, rhs: &RopeElem) {
        self.shift_at(self.len());
        for i in 0..rhs.len() {
            self.left.push(rhs.get(i));
        }
    }

    fn prepend(&mut self, lhs: &RopeElem) {
        self.shift_at(0);
        for i in (0..lhs.len()).rev() {
            self.right.push(lhs.get(i));
        }
    }

    fn prepend_bytes(&mut self, bytes: &[u8]) {
        self.shift_at(0);
        for i in (0..bytes.len()).rev() {
            self.right.push(bytes[i]);
        }
    }

    fn as_bytes(&mut self) -> &[u8] {
        self.shift_at(self.len());
        &self.left
    }

    fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.left
            .iter()
            .copied()
            .chain(self.right.iter().copied().rev())
    }

    fn to_string(&self) -> String {
        String::from_utf8(self.iter().collect::<Vec<u8>>()).unwrap()
    }

    fn push_bytes(&mut self, as_bytes: &[u8]) {
        self.shift_at(self.len());
        self.left.extend_from_slice(as_bytes);
    }
}

fn closest_2_power(x: usize) -> usize {
    let mut result = 8;
    while result < x {
        result <<= 1;
    }
    result
}

#[derive(Debug)]
pub struct Rope {
    tree: BTree<RopeTrait>,
}

impl UseLengthFinder<RopeTrait> for RopeTrait {
    #[inline(always)]
    fn get_len(cache: &<Self as BTreeTrait>::Cache) -> usize {
        *cache
    }

    fn find_element_by_offset(
        elements: &[<Self as BTreeTrait>::Elem],
        offset: usize,
    ) -> crate::FindResult {
        let mut left = offset;
        for (i, elem) in elements.iter().enumerate() {
            if left < elem.len() {
                return FindResult::new_found(i, left);
            }
            left -= elem.len();
        }

        FindResult::new_missing(elements.len(), left)
    }

    fn finder_drain_range(
        _: &mut HeapVec<<RopeTrait as BTreeTrait>::Elem>,
        _: Option<crate::QueryResult>,
        _: Option<crate::QueryResult>,
    ) -> Box<dyn Iterator<Item = <RopeTrait as BTreeTrait>::Elem> + '_> {
        unimplemented!()
    }

    fn finder_delete_range(
        elements: &mut HeapVec<<RopeTrait as BTreeTrait>::Elem>,
        start: Option<crate::QueryResult>,
        end: Option<crate::QueryResult>,
    ) {
        match (&start, &end) {
            (Some(from), Some(to)) if from.elem_index == to.elem_index => {
                if from.elem_index >= elements.len() {
                    return;
                }

                elements[from.elem_index].delete(from.offset..to.offset);
                return;
            }
            _ => {}
        }

        let start_index = match &start {
            Some(start) => {
                if start.offset == 0 {
                    // the whole element is included in the target range
                    start.elem_index
                } else if start.offset == elements[start.elem_index].len() {
                    // the start element is not included in the target range
                    start.elem_index + 1
                } else {
                    // partially included
                    let elem = &mut elements[start.elem_index];
                    elem.delete(start.offset..);
                    start.elem_index + 1
                }
            }
            None => 0,
        };
        match &end {
            Some(end) if end.elem_index < elements.len() => {
                if end.offset == elements[end.elem_index].len() {
                    // the whole element is included in the target range
                    elements.drain(start_index..end.elem_index + 1);
                } else if end.offset != 0 {
                    elements.drain(start_index..end.elem_index);
                    let elem = &mut elements[start_index];
                    elem.delete(..end.offset);
                } else {
                    elements.drain(start_index..end.elem_index);
                }
            }
            _ => {
                elements.truncate(start_index);
            }
        };
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
        let pos = self.tree.query::<LengthFinder>(&index);
        let tree = &mut self.tree;
        let index = pos.leaf;
        let leaf = tree.nodes.get_mut(index).unwrap();
        let elements = &mut leaf.elements;
        let mut index = pos.elem_index;
        let mut offset = pos.offset;
        if elements.is_empty() {
            let elem = RopeElem::from_str(elem);
            elements.push(elem);
        } else {
            if index >= elements.len() {
                index = elements.len() - 1;
                offset = elements[index].len();
            }
            let target = &mut elements[index];
            if target.len() < MAX_ELEM_SIZE || target.capacity() >= elem.len() + target.len() {
                elements[index].insert(offset, elem.as_bytes())
            } else {
                'insert_with_split: {
                    if elements.is_empty() {
                        let elem = RopeElem::from_str(elem);
                        elements.push(elem);
                        break 'insert_with_split;
                    }

                    if index == elements.len() {
                        debug_assert_eq!(offset, 0);
                        let last = elements.last_mut().unwrap();
                        let mut next = last.split_right(last.len() / 2);
                        next.push_bytes(elem.as_bytes());
                        elements.push(next);
                        break 'insert_with_split;
                    }

                    assert!(index < elements.len());
                    if offset == 0 {
                        let target = &mut elements[index];
                        let mut prev = target.split_left(target.len() / 2);
                        prev.prepend_bytes(elem.as_bytes());
                        elements.insert(index, prev);
                    } else if offset == elements[index].len() {
                        let target = &mut elements[index];
                        let mut next = target.split_right(target.len() / 2);
                        next.push_bytes(elem.as_bytes());
                        elements.insert(index + 1, next);
                    } else {
                        let left = elements.get_mut(index).unwrap();
                        let right = left.split_right(offset);
                        left.push_bytes(elem.as_bytes());
                        elements.insert(index + 1, right);
                    }
                }
            }
        }
        let is_full = leaf.is_full();
        tree.recursive_update_cache(pos.leaf, true, None);
        if is_full {
            tree.split(pos.leaf);
        }
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
        let end = end.min(self.len());
        let start = start.min(end);
        if start == end {
            return;
        }
        self.tree.drain::<LengthFinder>(start..end);
    }

    fn iter(&self) -> impl Iterator<Item = &[RopeElem]> {
        let mut node_iter = self
            .tree
            .first_path()
            .map(|first| crate::iter::Iter::new(&self.tree, first, self.tree.last_path().unwrap()));
        std::iter::from_fn(move || match &mut node_iter {
            Some(node_iter) => {
                if let Some(node) = node_iter.next() {
                    Some(node.1.elements.as_slice())
                } else {
                    None
                }
            }
            None => None,
        })
    }

    pub fn slice(&mut self, range: impl RangeBounds<usize>) {
        unimplemented!()
    }

    pub fn new() -> Self {
        Self { tree: BTree::new() }
    }

    #[allow(unused)]
    fn node_len(&self) -> usize {
        self.tree.node_len()
    }

    #[allow(unused)]
    fn update_in_place(&mut self, pos: usize, new: &str) {
        todo!()
    }

    pub fn clear(&mut self) {
        self.tree.clear();
    }

    #[allow(unused)]
    fn check(&self) {
        // dbg!(&self.tree);
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
        let mut ans = Vec::with_capacity(self.len());
        for elems in self.iter() {
            for elem in elems.iter() {
                for byte in elem.iter() {
                    ans.push(byte)
                }
            }
        }

        String::from_utf8(ans).unwrap()
    }
}

impl BTreeTrait for RopeTrait {
    type Elem = RopeElem;
    type WriteBuffer = ();
    type Cache = usize;

    const MAX_LEN: usize = 32;

    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[crate::Child<Self>],
        diff: Option<isize>,
    ) -> Option<isize> {
        match diff {
            Some(diff) => {
                *cache = (*cache as isize + diff) as usize;
                Some(diff)
            }
            None => {
                let new_cache = caches.iter().map(|x| x.cache).sum::<usize>();
                let diff = new_cache as isize - *cache as isize;
                *cache = new_cache;
                Some(diff)
            }
        }
    }

    fn calc_cache_leaf(
        cache: &mut Self::Cache,
        elements: &[Self::Elem],
        diff: Option<Self::CacheDiff>,
    ) -> isize {
        let new_cache = elements.iter().map(|x| x.len()).sum();
        let diff = new_cache as isize - *cache as isize;
        *cache = new_cache;
        diff
    }

    fn insert(_: &mut HeapVec<Self::Elem>, _: usize, _: usize, _: Self::Elem) {
        unreachable!()
    }

    type CacheDiff = isize;

    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff) {
        *diff1 += diff2;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    mod test_elem {
        use super::*;
        #[test]
        fn insert() {
            let mut e = RopeElem::new();
            e.insert(0, "abc".as_bytes());
            e.insert(2, "kkk".as_bytes());
            assert_eq!(e.to_string(), "abkkkc".to_string());

            let mut e = RopeElem::new();
            e.insert(0, "abc".as_bytes());
            e.insert(3, "kkk".as_bytes());
            assert_eq!(e.to_string(), "abckkk".to_string());
        }

        #[test]
        fn drain() {
            let mut e = RopeElem::new();
            e.insert(0, "0123456".as_bytes());
            e.drain(2..4);
            assert_eq!(e.to_string(), "01456".to_string());
        }

        #[test]
        fn delete() {
            let mut e = RopeElem::new();
            e.insert(0, "0123456".as_bytes());
            e.delete(2..4);
            assert_eq!(e.to_string(), "01456".to_string());
        }

        #[test]
        fn append() {
            let mut a = RopeElem::from_str("123");
            let mut b = RopeElem::from_str("456");
            a.append(&b);
            assert_eq!(a.to_string(), "123456".to_string());
            b.prepend(&a);
            assert_eq!(b.to_string(), "123456456".to_string());
        }

        #[test]
        fn set() {
            let mut a = RopeElem::from_str("0123");
            a.insert(1, "kk".as_bytes());
            assert_eq!(a.get(0), b'0');
            assert_eq!(a.get(1), b'k');
            assert_eq!(a.get(3), b'1');
            a.set(1, b'9');
            assert_eq!(a.get(1), b'9');
        }
    }

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
    fn test_delete_middle() {
        let mut rope = Rope::new();
        rope.insert(0, "135");
        rope.delete_range(1..2);
        assert_eq!(&rope.to_string(), "15");
    }

    #[test]
    fn test_insert_repeatedly() {
        let mut rope = Rope::new();
        rope.insert(0, "123");
        rope.insert(1, "x");
        rope.insert(2, "y");
        rope.insert(3, "z");
        assert_eq!(&rope.to_string(), "1xyz23");
    }

    #[test]
    #[ignore]
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
    #[ignore]
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
                    rope.check();
                }
                Action::Delete { pos, len } => {
                    let pos = pos as usize % (truth.len() + 1);
                    let mut len = len as usize % 10;
                    len = len.min(truth.len() - pos);
                    // dbg!(&rope);
                    // dbg!(rope.to_string(), pos, len);
                    // dbg!(&rope);
                    // dbg!(pos, len);
                    rope.delete_range(pos..(pos + len));
                    // dbg!(rope.to_string());
                    truth.drain(pos..pos + len);
                    rope.check();
                }
            }
        }

        assert_eq!(rope.to_string(), truth);
    }

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
    fn fuzz_3() {
        fuzz(vec![
            Insert {
                pos: 111,
                content: 140,
            },
            Insert {
                pos: 111,
                content: 107,
            },
            Insert {
                pos: 35,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 0,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 93,
                content: 93,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 102,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 111,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 101,
            },
            Insert {
                pos: 36,
                content: 146,
            },
            Delete { pos: 74, len: 102 },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 17,
                content: 17,
            },
            Insert {
                pos: 17,
                content: 17,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 102,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 111,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 3,
                content: 73,
            },
            Insert {
                pos: 146,
                content: 74,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 21,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 111,
                content: 111,
            },
            Insert { pos: 0, content: 8 },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 3,
            },
            Insert {
                pos: 36,
                content: 146,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Delete { pos: 111, len: 119 },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 73,
                content: 36,
            },
            Delete { pos: 74, len: 102 },
            Delete { pos: 255, len: 255 },
            Insert {
                pos: 42,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 0,
                content: 15,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 3,
            },
            Insert {
                pos: 36,
                content: 146,
            },
            Insert {
                pos: 255,
                content: 255,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 38,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 89,
                content: 89,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 42,
                content: 42,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 37,
            },
            Insert {
                pos: 101,
                content: 102,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 193, len: 63 },
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
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Delete { pos: 199, len: 199 },
            Delete { pos: 199, len: 199 },
            Delete { pos: 199, len: 199 },
            Delete { pos: 199, len: 199 },
            Delete { pos: 199, len: 199 },
            Delete { pos: 199, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Delete { pos: 187, len: 187 },
            Insert {
                pos: 3,
                content: 119,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Delete { pos: 163, len: 163 },
            Delete { pos: 163, len: 163 },
            Delete { pos: 163, len: 102 },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 108,
                content: 249,
            },
            Insert {
                pos: 135,
                content: 169,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 111, len: 255 },
            Insert {
                pos: 111,
                content: 111,
            },
            Insert {
                pos: 255,
                content: 255,
            },
        ])
    }

    #[test]
    fn fuzz_4() {
        fuzz(vec![
            Insert {
                pos: 0,
                content: 128,
            },
            Insert {
                pos: 0,
                content: 249,
            },
            Insert { pos: 8, content: 0 },
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
            Delete { pos: 255, len: 169 },
        ])
    }

    #[test]
    fn fuzz_5() {
        fuzz(vec![
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 0,
                content: 123,
            },
            Delete { pos: 108, len: 108 },
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
                pos: 12,
                content: 0,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 255, len: 246 },
            Delete { pos: 246, len: 246 },
            Delete { pos: 246, len: 246 },
            Delete { pos: 246, len: 246 },
            Insert {
                pos: 101,
                content: 101,
            },
            Insert {
                pos: 101,
                content: 101,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 169, len: 169 },
        ])
    }

    #[test]
    fn fuzz_6() {
        fuzz(vec![
            Insert {
                pos: 0,
                content: 128,
            },
            Insert { pos: 0, content: 0 },
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
                pos: 171,
                content: 171,
            },
            Delete { pos: 171, len: 0 },
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
                content: 171,
            },
            Delete { pos: 187, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
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
                pos: 171,
                content: 171,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 110,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 171,
            },
            Delete { pos: 187, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 8,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 50,
                content: 108,
            },
            Delete { pos: 108, len: 108 },
            Insert {
                pos: 108,
                content: 87,
            },
            Insert {
                pos: 249,
                content: 1,
            },
            Delete { pos: 169, len: 235 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 163, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 8, content: 0 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 41, len: 164 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert {
                pos: 254,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 0,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 238,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 86,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert {
                pos: 254,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 0,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 123,
            },
            Delete { pos: 123, len: 123 },
            Insert {
                pos: 86,
                content: 254,
            },
            Insert {
                pos: 33,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 123, len: 123 },
            Insert {
                pos: 0,
                content: 121,
            },
            Insert {
                pos: 26,
                content: 0,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 238, len: 254 },
            Insert {
                pos: 144,
                content: 238,
            },
            Delete { pos: 91, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 0, len: 51 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 123 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 86,
            },
            Delete { pos: 101, len: 144 },
            Delete { pos: 238, len: 91 },
            Delete { pos: 238, len: 238 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 3, content: 0 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 171,
                content: 63,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 8, content: 0 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 0, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 0,
                content: 171,
            },
            Delete { pos: 1, len: 126 },
            Delete { pos: 235, len: 154 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 84,
                content: 84,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 91, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 249,
                content: 1,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 8 },
            Insert {
                pos: 108,
                content: 32,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 235, len: 108 },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 255, len: 6 },
            Insert {
                pos: 135,
                content: 169,
            },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 171,
                content: 171,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 171,
                content: 171,
            },
            Insert {
                pos: 126,
                content: 111,
            },
            Delete { pos: 154, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 84,
                content: 171,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
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
                content: 235,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 87,
                content: 0,
            },
            Delete { pos: 1, len: 111 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 86,
                content: 254,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 86,
                content: 0,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 254, len: 193 },
            Delete { pos: 63, len: 64 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 8 },
            Insert {
                pos: 111,
                content: 127,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 0 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 8, len: 0 },
            Delete { pos: 249, len: 1 },
            Delete { pos: 169, len: 235 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 8,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 50,
                content: 108,
            },
            Delete { pos: 108, len: 108 },
            Insert {
                pos: 108,
                content: 8,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 169, len: 235 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 8, content: 0 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 41, len: 164 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 8, content: 0 },
            Insert {
                pos: 171,
                content: 171,
            },
            Insert { pos: 8, content: 0 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 41, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 41 },
            Insert {
                pos: 171,
                content: 171,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 165, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 170 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 235, len: 235 },
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
                pos: 0,
                content: 108,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 171,
                content: 171,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 8, content: 0 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert {
                pos: 254,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 8, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 238,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 91, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 18 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 86,
                content: 254,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 0,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 91, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 123 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 86,
                content: 86,
            },
            Insert {
                pos: 202,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert {
                pos: 254,
                content: 123,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 255, len: 101 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 123,
            },
            Delete { pos: 123, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 123,
            },
            Insert {
                pos: 238,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 1, len: 0 },
            Insert { pos: 0, content: 7 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 108 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 235,
                content: 235,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 111,
                content: 111,
            },
            Delete { pos: 154, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 171,
                content: 8,
            },
            Delete { pos: 171, len: 249 },
            Insert {
                pos: 135,
                content: 169,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 87,
                content: 84,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 11, len: 238 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 41, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 171, len: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 0,
                content: 108,
            },
            Delete { pos: 63, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 157, len: 157 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 108, len: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 0,
                content: 248,
            },
            Delete { pos: 154, len: 127 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 0 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 8, len: 0 },
            Delete { pos: 249, len: 1 },
            Delete { pos: 169, len: 235 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 84,
                content: 84,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 8 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 49,
            },
            Delete { pos: 235, len: 108 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 0,
                content: 249,
            },
            Insert {
                pos: 135,
                content: 169,
            },
            Delete { pos: 238, len: 123 },
            Insert { pos: 2, content: 0 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 1,
            },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 193,
                content: 192,
            },
            Delete { pos: 63, len: 127 },
            Insert {
                pos: 0,
                content: 235,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 87,
                content: 0,
            },
            Delete { pos: 1, len: 111 },
            Delete { pos: 235, len: 154 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 0, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 235, len: 235 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 127,
                content: 135,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 172 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 0 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 0,
                content: 171,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 108 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 235,
                content: 235,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 111,
                content: 111,
            },
            Delete { pos: 171, len: 0 },
            Insert {
                pos: 48,
                content: 111,
            },
            Delete { pos: 154, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Insert {
                pos: 84,
                content: 84,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 235 },
            Delete { pos: 254, len: 86 },
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
                content: 171,
            },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 20 },
            Delete { pos: 171, len: 108 },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert {
                pos: 235,
                content: 235,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 111,
                content: 111,
            },
            Delete { pos: 154, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 171, len: 171 },
            Delete { pos: 123, len: 123 },
            Insert {
                pos: 86,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 36,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 254,
                content: 255,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 123 },
            Insert { pos: 0, content: 0 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 123, len: 123 },
            Insert {
                pos: 238,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 238,
                content: 238,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 91, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert { pos: 0, content: 0 },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 238,
                content: 238,
            },
            Insert {
                pos: 108,
                content: 108,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 238, len: 238 },
            Insert {
                pos: 123,
                content: 2,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 238,
                content: 238,
            },
            Insert {
                pos: 0,
                content: 238,
            },
            Delete { pos: 238, len: 238 },
            Delete { pos: 0, len: 249 },
            Insert {
                pos: 135,
                content: 255,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 144, len: 255 },
            Delete { pos: 169, len: 169 },
        ])
    }

    #[test]
    fn ben() {
        use arbitrary::Arbitrary;
        #[derive(Arbitrary, Debug, Clone, Copy)]
        enum Action {
            Insert { pos: u8, content: u8 },
            Delete { pos: u8, len: u8 },
        }

        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut expected = String::new();
        let unstructured: Vec<u8> = (0..10_000).map(|_| rng.gen()).collect();
        let mut gen = arbitrary::Unstructured::new(&unstructured);
        let actions: [Action; 1_000] = gen.arbitrary().unwrap();
        let mut rope = Rope::new();
        for action in actions.iter() {
            match *action {
                Action::Insert { pos, content } => {
                    let pos = pos as usize % (rope.len() + 1);
                    let s = content.to_string();
                    expected.insert_str(pos, &s);
                    rope.insert(pos, &s);
                }
                Action::Delete { pos, len } => {
                    let pos = pos as usize % (rope.len() + 1);
                    let mut len = len as usize % 10;
                    len = len.min(rope.len() - pos);
                    expected.drain(pos..pos + len);
                    rope.delete_range(pos..(pos + len));
                }
            }
        }
        assert_eq!(rope.to_string(), expected);
    }

    #[test]
    fn fuzz_7() {
        fuzz(vec![
            Insert {
                pos: 111,
                content: 111,
            },
            Insert { pos: 0, content: 0 },
            Insert { pos: 0, content: 0 },
            Insert { pos: 0, content: 0 },
            Insert { pos: 0, content: 0 },
            Insert { pos: 0, content: 0 },
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
                content: 255,
            },
            Delete { pos: 255, len: 0 },
            Insert {
                pos: 140,
                content: 140,
            },
            Insert {
                pos: 102,
                content: 101,
            },
            Insert {
                pos: 36,
                content: 146,
            },
            Insert {
                pos: 102,
                content: 119,
            },
            Insert {
                pos: 118,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 21,
                content: 0,
            },
            Insert {
                pos: 140,
                content: 140,
            },
            Insert {
                pos: 107,
                content: 19,
            },
            Insert {
                pos: 102,
                content: 47,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 0,
                content: 102,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 0,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Delete { pos: 255, len: 136 },
            Delete { pos: 119, len: 111 },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 3,
                content: 73,
            },
            Insert {
                pos: 146,
                content: 74,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 0, len: 102 },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert { pos: 0, content: 0 },
            Delete { pos: 255, len: 255 },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 0,
                content: 255,
            },
            Delete { pos: 111, len: 108 },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Insert {
                pos: 83,
                content: 108,
            },
            Insert {
                pos: 111,
                content: 111,
            },
            Insert {
                pos: 119,
                content: 21,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 64,
                content: 64,
            },
            Insert {
                pos: 55,
                content: 119,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert { pos: 0, content: 0 },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Delete { pos: 130, len: 130 },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 96,
                content: 102,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 156,
                content: 111,
            },
            Insert {
                pos: 123,
                content: 37,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 37,
                content: 121,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 121,
                content: 86,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 123,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Delete { pos: 239, len: 239 },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 123,
                content: 0,
            },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 255, len: 255 },
            Delete { pos: 125, len: 125 },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 125,
                content: 125,
            },
            Insert {
                pos: 123,
                content: 123,
            },
            Insert {
                pos: 118,
                content: 118,
            },
            Insert {
                pos: 255,
                content: 255,
            },
            Insert {
                pos: 119,
                content: 119,
            },
            Insert {
                pos: 102,
                content: 102,
            },
            Delete { pos: 209, len: 255 },
            Delete { pos: 255, len: 255 },
        ])
    }

    #[test]
    fn fuzz_empty() {
        fuzz(vec![])
    }

    #[ctor::ctor]
    fn init_color_backtrace() {
        color_backtrace::install();
    }
}
