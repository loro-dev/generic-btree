use std::{ops::Range, usize};

use generic_btree::{BTree, BTreeTrait, LengthFinder, MutElemArrSlice, UseLengthFinder};

/// This struct keep the mapping of ranges to numbers
pub struct RangeNumMap(BTree<RangeNumMapTrait>);
struct RangeNumMapTrait;

#[derive(Clone, Debug)]
enum Modifier {
    Add(isize),
    Set(isize),
}

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
        if self.len() < range.end {
            self.0.insert::<LengthFinder>(
                &range.end,
                Elem {
                    value: None,
                    len: range.end - self.len() + 10,
                },
            );
        }

        let range = self.0.range::<LengthFinder>(range);
        self.0.update_with_buffer(
            range,
            &mut |mut slice| {
                let before_len = if cfg!(debug_assert) {
                    slice.elements.iter().map(|x| x.len).sum()
                } else {
                    0
                };
                let ans = set_value(&mut slice, value);
                if cfg!(debug_assert) {
                    let after_len = slice.elements.iter().map(|x| x.len).sum();
                    assert_eq!(before_len, after_len);
                }
                ans
            },
            |buffer, _| {
                *buffer = Some(Modifier::Set(value));
                false
            },
        );
    }

    pub fn get(&mut self, index: usize) -> Option<isize> {
        let result = self.0.query::<LengthFinder>(&index);
        self.0.get_elem(result).and_then(|x| x.value)
    }

    pub fn iter(&mut self) -> impl Iterator<Item = (Range<usize>, isize)> + '_ {
        let mut index = 0;
        self.0.iter_with_buffer_unloaded().filter_map(move |elem| {
            let len = elem.len;
            let value = elem.value?;
            let range = index..index + len;
            index += len;
            Some((range, value))
        })
    }

    pub fn drain(
        &mut self,
        range: Range<usize>,
    ) -> impl Iterator<Item = (Range<usize>, isize)> + '_ {
        let mut index = 0;
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

fn set_value(slice: &mut MutElemArrSlice<Elem>, value: isize) -> bool {
    let mut len = 0;
    match (slice.start, slice.end) {
        (Some((start_index, start_offset)), Some((end_index, end_offset)))
            if start_index == end_index =>
        {
            len = end_offset - start_offset;
            if start_offset > 0 {
                if end_offset < slice.elements[start_index].len {
                    let right_len = slice.elements[start_index].len - end_offset;
                    let old_value = slice.elements[start_index].value;
                    slice.elements[start_index].len = start_offset;
                    slice.elements.insert_many(
                        start_index + 1,
                        [
                            Elem {
                                value: Some(value),
                                len,
                            },
                            Elem {
                                value: old_value,
                                len: right_len,
                            },
                        ],
                    );
                } else {
                    slice.elements[start_index].len = start_offset;
                    slice.elements.insert(
                        start_index + 1,
                        Elem {
                            value: Some(value),
                            len,
                        },
                    );
                }
            } else if end_offset < slice.elements[start_index].len {
                slice.elements[start_index].len -= len;
                slice.elements.insert(
                    start_index,
                    Elem {
                        value: Some(value),
                        len,
                    },
                );
            } else {
                slice.elements[start_index].value = Some(value);
            }

            return false;
        }
        _ => {}
    };
    let drain_start = match slice.start {
        Some((start_index, start_offset)) => {
            if start_offset == 0 {
                start_index
            } else {
                len += slice.elements[start_index].len - start_offset;
                slice.elements[start_index].len = start_offset;
                start_index + 1
            }
        }
        None => 0,
    };
    let drain_end = match slice.end {
        Some((end_index, end_offset)) if end_index < slice.elements.len() => {
            if end_offset == slice.elements[end_index].len {
                end_index + 1
            } else {
                len += end_offset;
                slice.elements[end_index].len -= end_offset;
                end_index
            }
        }
        _ => slice.elements.len(),
    };
    len += slice
        .elements
        .drain(drain_start..drain_end)
        .map(|x| x.len)
        .sum::<usize>();
    slice.elements.insert(
        drain_start,
        Elem {
            value: Some(value),
            len,
        },
    );

    false
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
}

impl BTreeTrait for RangeNumMapTrait {
    /// value
    type Elem = Elem;
    /// len
    type Cache = usize;

    type WriteBuffer = Modifier;

    const MAX_LEN: usize = 8;

    fn element_to_cache(element: &Self::Elem) -> Self::Cache {
        element.len
    }

    fn calc_cache_internal(caches: &[generic_btree::Child<Self>]) -> Self::Cache {
        caches.iter().map(|c| c.cache).sum()
    }

    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache {
        elements.iter().map(|c| c.len).sum()
    }

    fn apply_write_buffer_to_elements(
        elements: &mut generic_btree::HeapVec<Self::Elem>,
        write_buffer: &Self::WriteBuffer,
    ) {
        elements.iter_mut().for_each(|x| {
            x.value = match write_buffer {
                Modifier::Add(value) => x.value.map(|x| x + value),
                Modifier::Set(value) => Some(*value),
            }
        });
    }

    fn apply_write_buffer_to_nodes(
        children: &mut [generic_btree::Child<Self>],
        write_buffer: &Self::WriteBuffer,
    ) {
        children.iter_mut().for_each(|x| {
            let v = match write_buffer {
                Modifier::Add(value) => x.write_buffer.as_ref().map(|x| match x {
                    Modifier::Add(x) => Modifier::Add(x + value),
                    Modifier::Set(x) => Modifier::Set(x + value),
                }),
                Modifier::Set(value) => Some(Modifier::Set(*value)),
            };
            x.write_buffer = v;
        });
    }
}
