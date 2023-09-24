use std::ops::RangeBounds;

use crate::rle::{HasLength, Mergeable, Sliceable};

#[derive(Debug, Clone)]
pub(super) struct GapBuffer {
    buffer: Vec<u8>,
    gap_start: u16,
    gap_len: u16,
}

impl GapBuffer {
    pub fn new() -> Self {
        Self {
            buffer: vec![0; 32],
            gap_start: 0,
            gap_len: 32,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            gap_start: 0,
            gap_len: capacity as u16,
        }
    }

    pub fn shift_at(&mut self, index: usize) {
        if index > self.len() {
            panic!("index {} out of range len={}", index, self.len());
        }

        let gap_start = self.gap_start as usize;
        let gap_end = (self.gap_start + self.gap_len) as usize;
        match index.cmp(&gap_start) {
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Less => {
                let gap_move = gap_start - index;
                self.buffer
                    .copy_within(index..gap_start, gap_end - gap_move);
                self.gap_start -= gap_move as u16;
            }
            std::cmp::Ordering::Greater => {
                let gap_move = index - gap_start;
                let move_end = self.buffer.len().min(gap_end + gap_move);
                self.buffer
                    .copy_within(gap_end..move_end, gap_start);
                self.gap_start += gap_move as u16;
            }
        }
    }

    pub fn push(&mut self, value: u8) {
        self.reserve(1);
        self.buffer[self.gap_start as usize] = value;
        self.gap_start += 1;
        self.gap_len -= 1;
    }

    #[inline(always)]
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.insert_bytes(self.len(), bytes);
    }

    pub fn insert_bytes(&mut self, index: usize, bytes: &[u8]) {
        self.reserve(bytes.len());
        self.shift_at(index);
        self.buffer[index..index + bytes.len()].copy_from_slice(bytes);
        self.gap_start += bytes.len() as u16;
        self.gap_len -= bytes.len() as u16;
    }

    pub fn delete(&mut self, range: impl RangeBounds<usize>) {
        let mut start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let mut end = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len(),
        };

        end = end.min(self.len());
        start = start.min(self.len()).min(end);
        if start == end {
            return;
        }

        let len = end - start;
        self.shift_at(end);
        self.gap_start = start as u16;
        self.gap_len += len as u16;
    }

    fn reserve(&mut self, len: usize) {
        if self.gap_len >= len as u16 {
            return;
        }

        let gap_end = (self.gap_start + self.gap_len) as usize;
        let old_buffer_len = self.buffer.len();

        self.buffer.reserve(len - self.gap_len as usize);
        let len = self.len();
        let cap = self.buffer.capacity();
        let new_gap_len = cap as u16 - len as u16;
        let new_gap_end = self.gap_start as usize + new_gap_len as usize;
        for _ in 0..(cap - self.buffer.len()) {
            self.buffer.push(0);
        }

        self.buffer
            .copy_within(gap_end..old_buffer_len, new_gap_end);
        self.gap_len = new_gap_len;
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len() - self.gap_len as usize
    }

    pub fn as_bytes(&self) -> (&[u8], &[u8]) {
        (
            &self.buffer[..self.gap_start as usize],
            &self.buffer[(self.gap_start + self.gap_len) as usize..],
        )
    }

    pub fn to_vec(&self) -> Vec<u8> {
        let mut vec = Vec::with_capacity(self.len());
        let (left, right) = self.as_bytes();
        vec.extend_from_slice(left);
        vec.extend_from_slice(right);
        vec
    }

    pub(crate) fn from_str(elem: &str) -> GapBuffer {
        let mut gb = GapBuffer::with_capacity(elem.len().max(16));
        gb.push_bytes(elem.as_bytes());
        gb
    }
}

impl HasLength for GapBuffer {
    fn rle_len(&self) -> usize {
        self.len()
    }
}

impl Sliceable for GapBuffer {
    fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let mut gb = Self::new();
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len(),
        };

        let (l, r) = self.as_bytes();
        if start < l.len() {
            gb.push_bytes(&l[start..end.min(l.len())]);
        }
        if end > l.len() {
            gb.push_bytes(&r[start.saturating_sub(l.len())..end.saturating_sub(l.len())]);
        }

        debug_assert_eq!(gb.len(), end - start);
        gb
    }

    fn slice_(&mut self, range: impl RangeBounds<usize>)
        where
            Self: Sized,
    {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => self.len(),
        };

        self.delete(end..);
        self.delete(..start);
        debug_assert_eq!(self.len(), end - start);
    }
}

impl Mergeable for GapBuffer {
    fn can_merge(&self, rhs: &Self) -> bool {
        self.len() + rhs.len() < 128
    }

    fn merge_right(&mut self, rhs: &Self) {
        let (a, b) = rhs.as_bytes();
        self.push_bytes(a);
        self.push_bytes(b);
    }

    fn merge_left(&mut self, left: &Self) {
        let (a, b) = left.as_bytes();
        self.insert_bytes(0, a);
        self.insert_bytes(0, b);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let mut gb = GapBuffer::new();
        gb.insert_bytes(0, &[3, 8]);
        assert_eq!(gb.to_vec(), vec![3, 8]);
        gb.insert_bytes(1, &[4, 5, 6]);
        assert_eq!(gb.to_vec(), vec![3, 4, 5, 6, 8]);
        assert_eq!(gb.len(), 5);
        gb.insert_bytes(4, &[7]);
        assert_eq!(gb.to_vec(), vec![3, 4, 5, 6, 7, 8]);
        gb.insert_bytes(0, &[1, 2, 9, 9]);
        assert_eq!(gb.to_vec(), vec![1, 2, 9, 9, 3, 4, 5, 6, 7, 8]);
        gb.delete(2..4);
        assert_eq!(gb.len(), 8);
        let (left, right) = gb.as_bytes();
        assert_eq!(left, &[1, 2]);
        assert_eq!(right, &[3, 4, 5, 6, 7, 8]);
        assert_eq!(gb.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8])
    }

    #[test]
    fn slice() {
        let mut gb = GapBuffer::new();
        gb.push_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        gb.shift_at(5);
        let b = gb.slice(2..5);
        assert_eq!(b.to_vec(), vec![2, 3, 4]);

        gb.slice_(2..5);
        assert_eq!(gb.to_vec(), vec![2, 3, 4]);
    }
}
