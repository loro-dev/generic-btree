use thunderdome::Index as ArenaIndex;

use crate::{BTree, BTreeTrait, MoveEvent, Node, NodePath, Query, QueryResult};

/// iterate node (not element) from the start path to the **inclusive** end path
pub(super) struct Iter<'a, B: BTreeTrait> {
    tree: &'a BTree<B>,
    inclusive_end: NodePath,
    path: NodePath,
    done: bool,
}

pub struct Drain<'a, B: BTreeTrait, Q: Query<B>> {
    tree: &'a mut BTree<B>,
    current_path: NodePath,
    done: bool,
    start_query: Q::QueryArg,
    start_result: QueryResult,
    end_query: Q::QueryArg,
    end_result: QueryResult,

    reversed_elements: Vec<B::Elem>,
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drain<'a, B, Q> {
    pub fn new(
        tree: &'a mut BTree<B>,
        start_query: Q::QueryArg,
        end_query: Q::QueryArg,
        start_result: QueryResult,
        end_result: QueryResult,
    ) -> Self {
        Self {
            current_path: tree.get_path(start_result.leaf),
            tree,
            done: false,
            start_query,
            end_query,
            start_result,
            end_result,
            reversed_elements: Vec::new(),
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Iterator for Drain<'a, B, Q> {
    type Item = B::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(next_elem) = self.reversed_elements.pop() {
                return Some(next_elem);
            }

            if self.done {
                return None;
            }

            if self.end_result.leaf == self.current_path.last().unwrap().arena {
                self.done = true;
            }

            let idx = *self.current_path.last().unwrap();
            if !self.tree.next_sibling(&mut self.current_path) {
                self.done = true;
            }

            let node = self.tree.get_mut(idx.arena);
            let start = if idx.arena == self.start_result.leaf {
                Some(self.start_result)
            } else {
                None
            };
            let end = if idx.arena == self.end_result.leaf {
                Some(self.end_result)
            } else {
                None
            };

            let iter = Q::drain_range(
                &mut node.elements,
                &self.start_query,
                &self.end_query,
                start,
                end,
            );
            for x in iter {
                self.reversed_elements.push(x);
            }

            if let Some(listener) = self.tree.element_move_listener.as_ref() {
                for elem in self.reversed_elements.iter() {
                    listener(MoveEvent::new_del(elem));
                }
            }
            self.reversed_elements.reverse();
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drain<'a, B, Q> {
    fn ensure_trim_start_and_end(&mut self) {
        if self.done {
            return;
        }

        let idx = *self.current_path.last().unwrap();
        let node = self.tree.get_mut(idx.arena);
        let start = if idx.arena == self.start_result.leaf {
            Some(self.start_result)
        } else {
            None
        };
        let is_last = idx.arena == self.end_result.leaf;
        let end = if is_last { Some(self.end_result) } else { None };

        Q::delete_range(
            &mut node.elements,
            &self.start_query,
            &self.end_query,
            start,
            end,
        );

        if !is_last {
            let last = self.tree.get_mut(self.end_result.leaf);
            Q::delete_range(
                &mut last.elements,
                &self.start_query,
                &self.end_query,
                None,
                Some(self.end_result),
            );
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drop for Drain<'a, B, Q> {
    fn drop(&mut self) {
        self.ensure_trim_start_and_end();
        let start_path = self.tree.get_path(self.start_result.leaf);
        let end_path = self.tree.get_path(self.end_result.leaf);
        let mut level = start_path.len() - 1;
        let mut deleted = Vec::new();
        let leaf_before_drain_range = {
            let node_idx = start_path[level].arena;
            let node = self.tree.get_node(node_idx);
            if node.is_empty() {
                self.tree.prev_same_level_node(node_idx)
            } else {
                Some(node_idx)
            }
        };
        let leaf_after_drain_range = {
            let node_idx = end_path[level].arena;
            let node = self.tree.get_node(node_idx);
            if node.is_empty() {
                self.tree.next_same_level_node(node_idx)
            } else {
                Some(node_idx)
            }
        };
        while start_path[level].arena != end_path[level].arena {
            let start_node = self.tree.get_node(start_path[level].arena);
            let end_node = self.tree.get_node(end_path[level].arena);
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
            let start_arena = start_path[level - 1].arena;
            let end_arena = end_path[level - 1].arena;
            if start_arena == end_arena {
                // parent is the same, delete start..end
                let parent = self.tree.get_mut(start_arena);
                for x in parent.children.drain(del_start..del_end) {
                    deleted.push(x.arena);
                }
                self.tree
                    .update_children_parent_slot_from(start_arena, del_start);
            } else {
                // parent is different
                {
                    // delete start..
                    let start_parent = self.tree.get_mut(start_arena);
                    for x in start_parent.children.drain(del_start..) {
                        deleted.push(x.arena);
                    }
                }
                {
                    // delete ..end
                    let end_parent = self.tree.get_mut(end_arena);
                    for x in end_parent.children.drain(..del_end) {
                        deleted.push(x.arena);
                    }
                    self.tree.update_children_parent_slot_from(end_arena, 0);
                }
            }

            level -= 1
            // this loop would break since level=0 is guaranteed to be the same
        }

        while level >= 1 {
            let (child, parent) = self
                .tree
                .get2_mut(start_path[level].arena, start_path[level - 1].arena);
            if child.is_empty() {
                assert_eq!(
                    parent.children[start_path[level].arr].arena,
                    start_path[level].arena
                );
                deleted.push(parent.children.remove(start_path[level].arr).arena);
                self.tree.update_children_parent_slot_from(
                    start_path[level - 1].arena,
                    start_path[level].arr,
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
            self.tree
                .recursive_update_cache(after, leaf_after_drain_range == leaf_before_drain_range);
        }

        // otherwise the path is invalid (e.g. the tree is empty)
        if let Some(before) = leaf_before_drain_range {
            self.tree
                .recursive_update_cache(before, leaf_after_drain_range == leaf_before_drain_range);
            seal(self.tree, before);
        } else {
            self.tree.update_root_cache();
            self.tree.try_reduce_levels();
        }
    }
}

fn seal<B: BTreeTrait>(tree: &mut BTree<B>, leaf: ArenaIndex) {
    handle_lack_on_path_to_leaf(tree, leaf);
    if let Some(sibling) = tree.next_same_level_node(leaf) {
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
        for i in 1..path.len() {
            let Some(node) = tree.nodes.get(path[i].arena) else { unreachable!() };
            let is_lack = node.is_lack();
            if is_lack {
                let lack_info = tree.handle_lack(path[i].arena);
                if lack_info.is_parent_lack {
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
    pub fn new(tree: &'a BTree<B>, start: NodePath, inclusive_end: NodePath) -> Self {
        Self {
            tree,
            inclusive_end,
            path: start,
            done: false,
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Iter<'a, B> {
    type Item = (NodePath, &'a Node<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.inclusive_end.last() == self.path.last() {
            self.done = true;
        }

        let last = *self.path.last().unwrap();
        let path = self.path.clone();
        if !self.tree.next_sibling(&mut self.path) {
            self.done = true;
        }

        let node = self.tree.get_node(last.arena);
        Some((path, node))
    }
}
