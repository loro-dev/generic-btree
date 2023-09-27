use crate::LeafIndex;

/// This event is fired when an element is created/deleted/spliced/merged.
///
/// It's used to track the which leaf node an element is in.
///
/// The delete events are not guaranteed to be called.
/// Deletion events are missing when:
///
/// - The tree is dropped
/// - Draining a range
#[derive(Debug, Clone)]
pub struct ElemUpdateEvent<'a, T> {
    /// If this is None, it means the element is deleted from the tree
    pub target_leaf: Option<LeafIndex>,
    pub elem: &'a T,
}

/// This is a event listener for element update event.
/// It's used to track the which leaf node an element is in.
pub type UpdateListener<T> = Box<dyn Fn(ElemUpdateEvent<'_, T>) + Send + Sync>;

impl<'a, T> From<(LeafIndex, &'a T)> for ElemUpdateEvent<'a, T> {
    #[inline]
    fn from(value: (LeafIndex, &'a T)) -> Self {
        Self {
            target_leaf: Some(value.0),
            elem: value.1,
        }
    }
}

impl<'a, T> ElemUpdateEvent<'a, T> {
    #[inline(always)]
    pub fn new_del(elem: &'a T) -> Self {
        Self {
            target_leaf: None,
            elem,
        }
    }

    #[inline(always)]
    pub fn new_move(to_leaf: LeafIndex, elem: &'a T) -> Self {
        Self {
            target_leaf: Some(to_leaf),
            elem,
        }
    }

    #[inline(always)]
    pub fn is_deleted(&self) -> bool {
        self.target_leaf.is_none()
    }
}
