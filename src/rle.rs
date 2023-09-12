use core::ops::RangeBounds;

use crate::{BTree, BTreeTrait, HeapVec, MutElemArrSlice, NodePath, QueryResult};

pub trait Sliceable<T = usize>: HasLength<T> {
    #[must_use]
    fn slice(&self, range: impl RangeBounds<T>) -> Self;
    /// slice in-place
    fn slice_(&mut self, range: impl RangeBounds<T>)
    where
        Self: Sized,
    {
        *self = self.slice(range);
    }
}

pub trait Mergeable {
    /// this is not symmetric
    fn can_merge(&self, rhs: &Self) -> bool;
    fn merge_right(&mut self, rhs: &Self);
    fn merge_left(&mut self, left: &Self);
}

pub trait HasLength<T = usize> {
    fn rle_len(&self) -> T;
}

impl<T: Mergeable, B: BTreeTrait<Elem = T>> BTree<B> {
    /// This will return the number of mergeable elements. Ideally this should be zero.
    /// This is provided for debugging and optimization
    pub fn get_mergeable_num(&self) -> usize {
        let mut last: Option<&T> = None;
        let mut num = 0;
        for span in self.iter() {
            match &mut last {
                Some(last) => {
                    num += if last.can_merge(span) { 1 } else { 0 };
                }
                None => last = Some(span),
            }
        }

        num
    }

    /// Try to merge the elements at the given range.
    /// This operation will invalidate the path if succeed.
    pub fn try_merge_to_neighbors(&mut self, start: QueryResult, end: QueryResult) {
        todo!()
    }

    /// return merged
    fn try_merge_next(&mut self, path: NodePath) -> bool {
        let leaf_idx = path.last().unwrap();
        let leaf = self.get_node(leaf_idx.arena);
        if leaf.is_lack() {
            self.handle_lack(path.last().unwrap().arena);
            return true;
        }

        let mut sibling_path = path.clone();
        if !self.next_sibling(&mut sibling_path) {
            return false;
        }

        let next_idx = sibling_path.last().unwrap();
        let next = self.get_node(next_idx.arena);
        if next.is_lack() {
            self.handle_lack(sibling_path.last().unwrap().arena);
            return true;
        }

        if leaf
            .elements
            .last()
            .unwrap()
            .can_merge(next.elements.first().unwrap())
        {
            let (a, b) = self.get2_mut(leaf_idx.arena, next_idx.arena);
            while a
                .elements
                .last()
                .map(|x| x.can_merge(b.elements.first().unwrap()))
                .unwrap_or(false)
            {
                let last = a.elements.pop().unwrap();
                b.elements[0].merge_left(&last);
            }

            if a.is_lack() {
                self.handle_lack(path.last().unwrap().arena);
                return true;
            }
        }

        false
    }

    /// return merged, if true the path is invalidated
    fn try_merge_prev(&mut self, path: NodePath) -> bool {
        let mut sibling_path = path;
        if !self.prev_sibling(&mut sibling_path) {
            return false;
        }

        self.try_merge_next(sibling_path)
    }
}

pub fn delete_range_in_elements<T: Sliceable + Mergeable>(
    elements: &mut HeapVec<T>,
    start: Option<QueryResult>,
    end: Option<QueryResult>,
) -> Vec<T> {
    match (&start, &end) {
        (Some(from), Some(to)) if from.elem_index == to.elem_index => {
            if from.elem_index >= elements.len() {
                assert!(!from.found);
                return Vec::new();
            }

            let mut ans = Vec::new();
            let elem = &mut elements[from.elem_index];
            ans.push(elem.slice(from.offset..to.offset));
            if to.offset != elem.rle_len() {
                if from.offset == 0 {
                    elements[from.elem_index].slice_(to.offset..);
                } else {
                    let right = elements[from.elem_index].slice(to.offset..);
                    elements[from.elem_index].slice_(..from.offset);
                    if elements[from.elem_index].can_merge(&right) {
                        elements[from.elem_index].merge_right(&right);
                    } else {
                        elements.insert(from.elem_index + 1, right);
                    }
                }
            } else if from.offset == 0 {
                elements.remove(from.elem_index);
            } else {
                elements[from.elem_index].slice_(to.offset..);
            }

            return ans;
        }
        _ => {}
    }

    let mut ans: Vec<T> = Vec::new();
    let start_index = match &start {
        Some(start) => {
            if start.offset == 0 {
                // the whole element is included in the target range
                start.elem_index
            } else if start.offset == elements[start.elem_index].rle_len() {
                // the start element is not included in the target range
                start.elem_index + 1
            } else {
                // partially included
                let elem = &mut elements[start.elem_index];
                ans.push(elem.slice(start.offset..));
                elem.slice_(..start.offset);
                start.elem_index + 1
            }
        }
        None => 0,
    };
    match &end {
        Some(end) if end.elem_index < elements.len() => {
            if end.offset == elements[end.elem_index].rle_len() {
                // the whole element is included in the target range
                ans.extend(elements.drain(start_index..end.elem_index + 1));
            } else if end.offset != 0 {
                ans.extend(elements.drain(start_index..end.elem_index));
                let elem = &mut elements[start_index];
                ans.push(elem.slice(..end.offset));
                elem.slice_(end.offset..);
            } else {
                ans.extend(elements.drain(start_index..end.elem_index));
            }
        }
        _ => {
            ans.extend(elements.drain(start_index..));
        }
    };
    ans
}

/// Update the given sliced elements by f. If f returns true, the cache should be updated.
///
/// This method will split the elements if necessary.
pub fn update_slice<T: Sliceable, F>(slice: &mut MutElemArrSlice<T>, f: &mut F) -> bool
where
    F: FnMut(&mut T) -> bool,
{
    let mut should_update = false;

    // if the start and end are in the same element
    match (slice.start, slice.end) {
        (Some((start_index, start_offset)), Some((end_index, end_offset)))
            if start_index == end_index =>
        {
            if start_offset > 0 {
                if end_offset < slice.elements[start_index].rle_len() {
                    // need to insert two new elements, because the range is in the middle of the element
                    let mut elem = slice.elements[start_index].slice(start_offset..end_offset);
                    should_update = should_update || f(&mut elem);
                    let right = slice.elements[start_index].slice(end_offset..);
                    slice.elements[start_index].slice_(..start_offset);
                    slice
                        .elements
                        .splice(start_index + 1..start_index + 1, [elem, right]);
                } else {
                    // slice the elem into two part: ( ..start ), ( start.. )
                    let mut elem = slice.elements[start_index].slice(start_offset..end_offset);
                    should_update = should_update || f(&mut elem);
                    slice.elements[start_index].slice_(..start_offset);
                    slice.elements.insert(start_index + 1, elem);
                }
            } else if end_offset < slice.elements[start_index].rle_len() {
                // slice the elem into two part: ( ..end ), ( end.. )
                let mut elem = slice.elements[start_index].slice(..end_offset);
                should_update = should_update || f(&mut elem);
                slice.elements[start_index].slice_(end_offset..);
                slice.elements.insert(start_index, elem);
            } else {
                // no need to slice, update directly
                should_update = should_update || f(&mut slice.elements[start_index]);
            }

            return should_update;
        }
        _ => {}
    };

    let mut shift = 0;
    let start = match slice.start {
        Some((start_index, start_offset)) => {
            if start_offset == 0
                || start_index == slice.elements.len()
                || slice.elements[start_index].rle_len() == 0
            {
                start_index
            } else if start_offset == slice.elements[start_index].rle_len() {
                start_index + 1
            } else {
                let elem = slice.elements[start_index].slice(start_offset..);
                slice.elements[start_index].slice_(..start_offset);
                slice.elements.insert(start_index + 1, elem);
                shift = 1;
                start_index + 1
            }
        }
        None => 0,
    };
    let end = match slice.end {
        Some((end_index, end_offset)) if end_index + shift < slice.elements.len() => {
            let origin = &mut slice.elements[end_index + shift];
            if end_offset == origin.rle_len() || origin.rle_len() == 0 {
                end_index + 1 + shift
            } else if end_offset == 0 {
                end_index + shift
            } else {
                let elem = origin.slice(..end_offset);
                origin.slice_(end_offset..);
                slice.elements.insert(end_index + shift, elem);
                shift += 1;
                end_index + shift
            }
        }
        _ => slice.elements.len(),
    };

    let mut ans = false;
    for elem in slice.elements[start..end].iter_mut() {
        ans = f(elem) || ans;
    }
    ans
}

pub fn scan_and_merge<T: Mergeable>(elements: &mut HeapVec<T>, start: usize) {
    if start + 1 >= elements.len() {
        return;
    }

    let (left, right) = elements.split_at_mut(start + 1);
    let start_elem = left.last_mut().unwrap();
    let mut i = 0;
    while i < right.len() {
        if !start_elem.can_merge(&right[i]) {
            break;
        }

        start_elem.merge_right(&right[i]);
        i += 1;
    }

    if i > 0 {
        elements.drain(start + 1..start + 1 + i);
    }
}

pub fn insert_with_split<T: Sliceable + Mergeable>(
    elements: &mut HeapVec<T>,
    index: usize,
    offset: usize,
    elem: T,
) {
    if elements.is_empty() {
        elements.push(elem);
        return;
    }

    if index == elements.len() {
        debug_assert_eq!(offset, 0);
        let last = elements.last_mut().unwrap();
        if last.can_merge(&elem) {
            last.merge_right(&elem);
        } else {
            elements.push(elem);
        }

        return;
    }

    assert!(index < elements.len());
    if offset == 0 {
        let target = elements.get_mut(index).unwrap();
        if elem.can_merge(target) {
            target.merge_left(&elem);
        } else {
            elements.insert(index, elem);
        }
    } else if offset == elements[index].rle_len() {
        let target = elements.get_mut(index).unwrap();
        if target.can_merge(&elem) {
            target.merge_right(&elem);
        } else {
            elements.insert(index + 1, elem);
        }
    } else {
        let right = elements[index].slice(offset..);
        elements[index].slice_(..offset);
        let left = elements.get_mut(index).unwrap();
        if left.can_merge(&elem) {
            left.merge_right(&elem);
            elements.insert(index + 1, right);
        } else {
            elements.splice(index + 1..index + 1, [elem, right]);
        }
    }
}
