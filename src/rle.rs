use core::ops::RangeBounds;

use crate::{HeapVec, QueryResult, SmallElemVec};

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

pub trait HasLength<T = usize> {
    fn rle_len(&self) -> T;
}
