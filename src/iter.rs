use crate::{
    delete_range, rle::HasLength, ArenaIndex, BTree, BTreeTrait, LeafIndex, LeafNode, NodePath,
    QueryResult,
};

/// iterate node (not element) from the start path to the **inclusive** end path
pub(super) struct Iter<'a, B: BTreeTrait> {
    tree: &'a BTree<B>,
    inclusive_end: LeafIndex,
    current_leaf: Option<LeafIndex>,
}

struct TempStore {
    start_path: NodePath,
    end_path: NodePath,
    leaf_before_drain_range: Option<ArenaIndex>,
    leaf_after_drain_range: Option<ArenaIndex>,
}

pub struct Drain<'a, B: BTreeTrait> {
    tree: &'a mut BTree<B>,
    current_leaf: Option<LeafIndex>,
    done: bool,
    end_leaf: Option<LeafIndex>,
    store: Option<TempStore>,
}

impl<'a, B: BTreeTrait> Drain<'a, B> {
    pub fn new(
        tree: &'a mut BTree<B>,
        start_result: Option<QueryResult>,
        end_result: Option<QueryResult>,
    ) -> Self {
        if start_result.is_none() || end_result.is_none() {
            return Self::none(tree);
        }

        let start_result = start_result.unwrap();
        let end_result = end_result.unwrap();
        let end_result = tree.split_leaf_if_needed(end_result.cursor).new_pos;
        let Some(start_result) = tree.split_leaf_if_needed(start_result.cursor).new_pos else {
            // if start from the right most leaf, the range is empty
            return Self::none(tree);
        };
        let start_path = tree.get_path(start_result.leaf.into());
        let end_path = tree.get_path(
            end_result
                .map(|x| x.leaf.into())
                .unwrap_or_else(|| tree.last_leaf().unwrap().into()),
        );
        let leaf_before_drain_range = {
            let leaf_idx = start_result.leaf;
            if start_result.offset == 0 {
                tree.get_leaf(leaf_idx.into())
                    .prev_leaf()
                    .map(ArenaIndex::from)
            } else {
                Some(leaf_idx.into())
            }
        };
        let leaf_after_drain_range = {
            if let Some(end) = end_result {
                let len = tree.leaf_nodes.get(end.leaf.0).unwrap().elem.rle_len();
                if len == end.offset {
                    tree.get_leaf(end.leaf.into())
                        .next_leaf()
                        .map(ArenaIndex::from)
                } else {
                    Some(end.leaf.into())
                }
            } else {
                None
            }
        };
        Self {
            current_leaf: Some(start_result.leaf),
            tree,
            done: false,
            end_leaf: end_result.map(|x| x.leaf),
            store: Some(TempStore {
                start_path,
                end_path,
                leaf_before_drain_range,
                leaf_after_drain_range,
            }),
        }
    }

    fn none(tree: &'a mut BTree<B>) -> Drain<'a, B> {
        Self {
            current_leaf: None,
            done: true,
            end_leaf: None,
            tree,
            store: None,
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Drain<'a, B> {
    type Item = B::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let leaf_idx = self.current_leaf?;
        if self.end_leaf == Some(leaf_idx) {
            self.done = true;
            return None;
        }

        let next_leaf = self.tree.get_leaf(leaf_idx.into()).next_leaf();
        self.current_leaf = next_leaf;
        if next_leaf.is_none() {
            self.done = true;
        }

        self.tree.unlink_leaf(leaf_idx);
        let node = self.tree.leaf_nodes.remove(leaf_idx.0).unwrap();
        Some(node.elem)
    }
}

impl<'a, B: BTreeTrait> Drain<'a, B> {
    fn ensure_finished(&mut self) {
        while self.next().is_some() {}
    }
}

impl<'a, B: BTreeTrait> Drop for Drain<'a, B> {
    fn drop(&mut self) {
        self.ensure_finished();
        let Some(store) = self.store.take() else {
            return;
        };
        let TempStore {
            start_path,
            end_path,
            leaf_before_drain_range,
            leaf_after_drain_range,
        } = store;
        // the deepest internal node level
        let mut level = start_path.len() - 2;
        let mut deleted = Vec::new();

        // The deepest internal node level, need to filter deleted children
        // to ensure is_empty() has correct result
        self.tree.filter_deleted_children(start_path[level].arena);
        self.tree.filter_deleted_children(end_path[level].arena);
        while start_path[level].arena != end_path[level].arena {
            let start_node = self.tree.get_internal(start_path[level].arena);
            let end_node = self.tree.get_internal(end_path[level].arena);
            let del_start = if start_node.is_empty() {
                start_path[level].arr
            } else {
                start_path[level].arr + 1
            };
            let del_end = if end_node.is_empty() {
                end_path[level].arr + 1
            } else {
                end_path[level].arr
            };

            // remove del_start.. in start_node's parent
            // remove   ..del_end in end_node's   parent
            let start_arena = start_path[level - 1].arena;
            let end_arena = end_path[level - 1].arena;
            if start_arena == end_arena {
                // parent is the same, delete start..end
                let parent = self.tree.get_internal_mut(start_arena);
                for x in &parent.children[del_start as usize..del_end as usize] {
                    deleted.push(x.arena);
                }

                delete_range(&mut parent.children, del_start as usize..del_end as usize);
                self.tree
                    .update_children_parent_slot_from(start_arena, del_start as usize);
            } else {
                // parent is different
                {
                    // delete start..
                    let start_parent = self.tree.get_internal_mut(start_arena);
                    for x in &start_parent.children[del_start as usize..] {
                        deleted.push(x.arena);
                    }
                    delete_range(&mut start_parent.children, del_start as usize..);
                }
                {
                    // delete ..end
                    let end_parent = self.tree.get_internal_mut(end_arena);
                    for x in &end_parent.children[..del_end as usize] {
                        deleted.push(x.arena);
                    }
                    delete_range(&mut end_parent.children, ..del_end as usize);
                    self.tree.update_children_parent_slot_from(end_arena, 0);
                }
            }

            level -= 1
            // this loop will abort before overflow, because level=0 is guaranteed to be the same
        }

        while level >= 1 {
            let (child, parent) = self
                .tree
                .get2_mut(start_path[level].arena, start_path[level - 1].arena);
            if child.is_empty() {
                assert_eq!(
                    parent.children[start_path[level].arr as usize].arena,
                    start_path[level].arena
                );
                deleted.push(parent.children.remove(start_path[level].arr as usize).arena);
                self.tree.update_children_parent_slot_from(
                    start_path[level - 1].arena,
                    start_path[level].arr as usize,
                );
            } else {
                break;
            }
            level -= 1;
        }

        // release memory
        for x in deleted {
            self.tree.purge(x);
        }

        if let Some(after) = leaf_after_drain_range {
            self.tree.recursive_update_cache(
                after,
                leaf_after_drain_range == leaf_before_drain_range,
                None,
            );
        }

        // otherwise the path is invalid (e.g. the tree is empty)
        if let Some(before) = leaf_before_drain_range {
            if leaf_before_drain_range == leaf_after_drain_range {
                self.tree.recursive_update_cache(before, B::USE_DIFF, None);
            } else {
                self.tree.recursive_update_cache(before, false, None);
                if let Some(after) = leaf_after_drain_range {
                    self.tree.recursive_update_cache(after, false, None);
                }
            }
            seal(self.tree, before);
        } else {
            self.tree.update_root_cache();
            self.tree.try_reduce_levels();
        }
    }
}

fn seal<B: BTreeTrait>(tree: &mut BTree<B>, leaf: ArenaIndex) {
    handle_lack_on_path_to_leaf(tree, leaf);
    if let Some(sibling) = tree.next_same_level_in_node(leaf) {
        handle_lack_on_path_to_leaf(tree, sibling);
    }
    tree.try_reduce_levels();
}

fn handle_lack_on_path_to_leaf<B: BTreeTrait>(tree: &mut BTree<B>, leaf: ArenaIndex) {
    let mut last_lack_count = 0;
    let mut lack_count;
    loop {
        lack_count = 0;
        let path = tree.get_path(leaf);
        for i in 1..path.len() - 1 {
            let Some(node) = tree.in_nodes.get(path[i].arena.unwrap_internal()) else {
                unreachable!()
            };
            let is_lack = node.is_lack();
            if is_lack {
                let lack_info = tree.handle_lack_single_layer(path[i].arena);
                if lack_info.parent_lack.is_some() {
                    lack_count += 1;
                }
            }
        }
        // parent may be lack after some children is merged
        if lack_count == 0 || lack_count == last_lack_count {
            break;
        }

        last_lack_count = lack_count;
    }
}

impl<'a, B: BTreeTrait> Iter<'a, B> {
    pub fn new(tree: &'a BTree<B>, start: LeafIndex, inclusive_end: LeafIndex) -> Self {
        Self {
            tree,
            inclusive_end,
            current_leaf: Some(start),
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Iter<'a, B> {
    type Item = (LeafIndex, &'a LeafNode<B::Elem>);

    fn next(&mut self) -> Option<Self::Item> {
        let leaf_idx = self.current_leaf?;
        let node = self.tree.leaf_nodes.get(leaf_idx.0).unwrap();
        self.current_leaf = if leaf_idx == self.inclusive_end {
            None
        } else {
            node.next_leaf()
        };
        Some((leaf_idx, node))
    }
}
