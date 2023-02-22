use core::ops::RangeBounds;

use crate::{HeapVec, MutElemArrSlice, QueryResult, SmallElemVec};

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
    fn can_merge(&self, rhs: &Self) -> bool;
    fn merge_right(&mut self, rhs: &Self);
    fn merge_left(&mut self, left: &Self);
}

pub fn delete_range_in_elements<T: Sliceable>(
    elements: &mut HeapVec<T>,
    start: Option<QueryResult>,
    end: Option<QueryResult>,
) -> SmallElemVec<T> {
    match (&start, &end) {
        (Some(from), Some(to)) if from.elem_index == to.elem_index => {
            let mut ans = SmallElemVec::new();
            let elem = &mut elements[from.elem_index];
            ans.push(elem.slice(from.offset..to.offset));
            if to.offset != elem.rle_len() {
                if from.offset == 0 {
                    elements[from.elem_index].slice_(to.offset..);
                } else {
                    elements[from.elem_index].slice_(..from.offset);
                    let right = elements[from.elem_index].slice(to.offset..);
                    elements.insert(from.elem_index + 1, right);
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

    let mut ans: SmallElemVec<T> = SmallElemVec::new();
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

/// F returns whether should update cache
pub fn update_slice<T: Sliceable, F>(slice: &mut MutElemArrSlice<T>, f: &mut F) -> bool
where
    F: FnMut(&mut T) -> bool,
{
    let mut should_update = false;
    match (slice.start, slice.end) {
        (Some((start_index, start_offset)), Some((end_index, end_offset)))
            if start_index == end_index =>
        {
            if start_offset > 0 {
                if end_offset < slice.elements[start_index].rle_len() {
                    let mut elem = slice.elements[start_index].slice(start_offset..end_offset);
                    should_update = should_update || f(&mut elem);
                    let right = slice.elements[start_index].slice(end_offset..);
                    slice.elements[start_index].slice_(..start_offset);
                    slice.elements.insert_many(start_index + 1, [elem, right]);
                } else {
                    let mut elem = slice.elements[start_index].slice(start_offset..end_offset);
                    should_update = should_update || f(&mut elem);
                    slice.elements[start_index].slice_(..start_offset);
                    slice.elements.insert(start_index + 1, elem);
                }
            } else if end_offset < slice.elements[start_index].rle_len() {
                let mut elem = slice.elements[start_index].slice(..end_offset);
                should_update = should_update || f(&mut elem);
                slice.elements[start_index].slice_(end_offset..);
                slice.elements.insert(start_index, elem);
            } else {
                should_update = should_update || f(&mut slice.elements[start_index]);
            }

            return should_update;
        }
        _ => {}
    };

    let mut shift = 0;
    let start = match slice.start {
        Some((start_index, start_offset)) => {
            if start_offset == 0 {
                start_index
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
        Some((end_index, end_offset)) if end_index < slice.elements.len() => {
            let origin = &mut slice.elements[end_index + shift];
            if end_offset == origin.rle_len() {
                end_index + 1 + shift
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

    if offset == 0 {
        let target = elements.get_mut(index).unwrap();
        if target.can_merge(&elem) {
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
            elements.insert(index, right);
        } else {
            elements.insert_many(index, [elem, right]);
        }
    }
}

pub trait HasLength<T = usize> {
    fn rle_len(&self) -> T;
}
