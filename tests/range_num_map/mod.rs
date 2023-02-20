use generic_btree::{BTree, BTreeTrait};

/// This struct keep the mapping of ranges to numbers
pub struct RangeNumMap(BTree<RangeNumMapTrait>);
struct RangeNumMapTrait;

#[derive(Clone)]
enum Modifier {
    Add(isize),
    Set(isize),
}

struct Elem {
    value: isize,
    len: usize,
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
}
