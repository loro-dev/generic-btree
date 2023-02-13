use std::{fmt::Display, ops::RangeBounds};

use crate::{BTree, BTreeTrait, FindResult, Query};

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

    pub fn insert(&mut self, index: usize, elem: String) {
        let result = self.tree.query::<Finder>(&index);
        self.tree.insert_by_query_result(result, elem);
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

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.tree.iter()
    }

    pub fn iter_range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = &String> {
        self.tree.iter_range::<Finder>(range)
    }

    pub(crate) fn new() -> Self {
        Self { tree: BTree::new() }
    }
}

impl ToString for Rope {
    fn to_string(&self) -> String {
        let mut ans = String::with_capacity(self.len());
        for s in self.iter() {
            ans.push_str(s);
        }

        ans
    }
}

impl BTreeTrait for RopeTrait {
    type Elem = String;

    type Cache = usize;

    const MAX_LEN: usize = 16;

    fn element_to_cache(element: &Self::Elem) -> Self::Cache {
        element.len()
    }

    fn calc_cache_internal(caches: &[crate::Child<Self::Cache>]) -> Self::Cache {
        caches.iter().map(|x| x.cache).sum::<usize>()
    }

    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache {
        elements.iter().map(|x| x.len()).sum()
    }

    fn insert(elements: &mut Vec<Self::Elem>, index: usize, offset: usize, elem: Self::Elem) {
        if index < elements.len() {
            if elements[index].capacity() > elements[index].len() + elem.len() {
                elements[index].insert_str(offset, &elem);
            } else if offset == 0 {
                elements.insert(index, elem);
            } else if offset == elements[index].len() {
                elements.insert(index + 1, elem);
            } else {
                let right = elements[index][offset..].to_owned();
                elements[index].drain(offset..);
                elements.splice(index + 1..index + 1, [elem, right]);
            }
        } else {
            elements.push(elem);
        }
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

    fn find_element(&mut self, _: &Self::QueryArg, elements: &[String]) -> crate::FindResult {
        for (i, elem) in elements.iter().enumerate() {
            if self.left >= elem.len() {
                self.left -= elem.len();
            } else {
                return FindResult::new_found(i, self.left);
            }
        }

        FindResult::new_missing(elements.len(), self.left)
    }

    fn init(target: &Self::QueryArg) -> Self {
        Self { left: *target }
    }

    fn delete(
        elements: &mut Vec<String>,
        _: &Self::QueryArg,
        elem_index: usize,
        offset: usize,
    ) -> Option<String> {
        if elem_index >= elements.len() {
            return None;
        }

        let text = &mut elements[elem_index];
        if offset >= text.len() {
            return None;
        }

        if offset == 0 && text.len() == 1 {
            return Some(elements.remove(elem_index));
        }

        Some(text.remove(offset).to_string())
    }

    fn delete_range<'x, 'b>(
        elements: &'x mut Vec<String>,
        _: &'b Self::QueryArg,
        _: &'b Self::QueryArg,
        start: Option<crate::QueryResult>,
        end: Option<crate::QueryResult>,
    ) -> Box<dyn Iterator<Item = String> + 'x> {
        fn drain_start(start: crate::QueryResult, elements: &mut [String]) -> usize {
            if start.offset == 0 || start.elem_index >= elements.len() {
                start.elem_index
            } else if start.offset == elements[start.elem_index].len() {
                start.elem_index + 1
            } else {
                elements[start.elem_index].drain(start.offset..);
                start.elem_index + 1
            }
        }

        fn drain_end(end: crate::QueryResult, elements: &mut [String]) -> usize {
            if end.elem_index >= elements.len() {
                end.elem_index
            } else if elements[end.elem_index].len() == end.offset {
                end.elem_index + 1
            } else if end.offset == 0 {
                end.elem_index
            } else {
                elements[end.elem_index].drain(..end.offset);
                end.elem_index
            }
        }

        match (start, end) {
            (None, None) => Box::new(elements.drain(..)),
            (None, Some(end)) => {
                let end = drain_end(end, elements);
                Box::new(elements.drain(..end))
            }
            (Some(start), None) => {
                let start = drain_start(start, elements);
                Box::new(elements.drain(start..))
            }
            (Some(start), Some(end)) => {
                if start.elem_index == end.elem_index {
                    let ans: String = elements[start.elem_index]
                        .drain(start.offset..end.offset)
                        .collect();
                    Box::new(Some(ans).into_iter())
                } else {
                    let start = drain_start(start, elements);
                    let end = drain_end(end, elements);
                    Box::new(elements.drain(start..end))
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
        rope.insert(0, "123".to_string());
        rope.insert(1, "x".to_string());
        assert_eq!(rope.len(), 4);
        rope.delete_range(2..4);
        assert_eq!(&rope.to_string(), "1x");
        rope.delete_range(..1);
        assert_eq!(&rope.to_string(), "x");
        rope.delete_range(..);
        assert_eq!(&rope.to_string(), "");
        assert_eq!(rope.len(), 0);
    }
}
