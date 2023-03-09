use thunderdome::Index as ArenaIndex;

/// The move event of an element.
///
/// It's used to track the which leaf node an element is in.
///
/// The delete events are not guaranteed to be called.  The events
/// are missing when:
///
/// - The tree is dropped
/// - Or, when draining a range, the elements from the start/end
///   leaf node
pub struct MoveEvent<'a, T> {
    /// If this is None, it means the element is deleted from the tree
    pub target_leaf: Option<ArenaIndex>,
    pub elem: &'a T,
}

/// This is a event listener for element move event.
/// It's used to track the which leaf node an element is in.
pub type MoveListener<T> = Box<dyn Fn(MoveEvent<'_, T>)>;

impl<'a, T> From<(ArenaIndex, &'a T)> for MoveEvent<'a, T> {
    fn from(value: (ArenaIndex, &'a T)) -> Self {
        Self {
            target_leaf: Some(value.0),
            elem: value.1,
        }
    }
}

impl<'a, T> MoveEvent<'a, T> {
    pub fn new_del(elem: &'a T) -> Self {
        Self {
            target_leaf: None,
            elem,
        }
    }

    pub fn new_move(to_leaf: ArenaIndex, elem: &'a T) -> Self {
        Self {
            target_leaf: Some(to_leaf),
            elem,
        }
    }

    pub fn is_deleted(&self) -> bool {
        self.target_leaf.is_none()
    }
}
