use smallvec::SmallVec;

use crate::{BTree, BTreeTrait, Idx, Node, Path, PathRef, Query, QueryResult, StackVec};

/// iterate node (not element) from the start path to the **inclusive** end path
pub(super) struct Iter<'a, B: BTreeTrait> {
    tree: &'a BTree<B>,
    inclusive_end: QueryResult,
    path: Path,
    done: bool,
}

pub struct Drain<'a, B: BTreeTrait, Q: Query<B>> {
    tree: &'a mut BTree<B>,
    current_path: Path,
    done: bool,
    start_query: Q::QueryArg,
    start_result: QueryResult,
    end_query: Q::QueryArg,
    end_result: QueryResult,

    reversed_elements: StackVec<B::Elem>,
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
            tree,
            done: false,
            start_query,
            end_query,
            current_path: start_result.node_path.clone(),
            start_result,
            end_result,
            reversed_elements: StackVec::with_capacity(B::MAX_LEN),
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

            if self.end_result.node_path.last() == self.current_path.last() {
                self.done = true;
            }

            let idx = *self.current_path.last().unwrap();
            if !self.tree.next_sibling(&mut self.current_path) {
                self.done = true;
            }

            let node = self.tree.get_mut(idx.arena);
            let start = if idx.arena == self.start_result.node_path.last().unwrap().arena {
                Some(self.start_result.clone())
            } else {
                None
            };
            let end = if idx.arena == self.end_result.node_path.last().unwrap().arena {
                Some(self.end_result.clone())
            } else {
                None
            };

            let iter = Q::delete_range(
                &mut node.elements,
                &self.start_query,
                &self.end_query,
                start,
                end,
            );
            for x in iter {
                self.reversed_elements.push(x);
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
        let start = if idx.arena == self.start_result.node_path.last().unwrap().arena {
            Some(self.start_result.clone())
        } else {
            None
        };
        let is_last = idx.arena == self.end_result.node_path.last().unwrap().arena;
        let end = if is_last {
            Some(self.end_result.clone())
        } else {
            None
        };

        for _ in Q::delete_range(
            &mut node.elements,
            &self.start_query,
            &self.end_query,
            start,
            end,
        ) {}

        if !is_last {
            let last = self
                .tree
                .get_mut(self.end_result.node_path.last().unwrap().arena);
            for _ in Q::delete_range(
                &mut last.elements,
                &self.start_query,
                &self.end_query,
                None,
                Some(self.end_result.clone()),
            ) {}
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drop for Drain<'a, B, Q> {
    fn drop(&mut self) {
        self.ensure_trim_start_and_end();
        let start_path = &self.start_result.node_path;
        let end_path = &self.end_result.node_path;
        let mut level = start_path.len() - 1;
        let mut deleted: SmallVec<[_; 32]> = SmallVec::new();
        let mut zipper_left: SmallVec<[_; 16]> = SmallVec::with_capacity(start_path.len());
        let mut zipper_right: SmallVec<[_; 16]> = SmallVec::with_capacity(start_path.len());
        // TODO: formalize zipper left and zipper right and document it
        while start_path[level].arena != end_path[level].arena {
            let start_node = self.tree.get(start_path[level].arena);
            let end_node = self.tree.get(end_path[level].arena);
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
            if start_path[level - 1].arena == end_path[level - 1].arena {
                // parent is the same, delete start..end
                let parent = self.tree.get_mut(start_path[level - 1].arena);
                zipper_right.push(Some(del_start));
                if del_start == 0 && del_end == parent.children.len() {
                    // if we are deleting the whole node, we need to delete the parent as well
                    // so the path index at this level would be the last child of the parent
                    zipper_left.push(None);
                } else {
                    zipper_left.push(Some(del_start.max(1) - 1));
                }

                for x in parent.children.drain(del_start..del_end) {
                    deleted.push(x.arena);
                }
            } else {
                zipper_right.push(Some(0));
                if del_start == 0 {
                    // if we are deleting the whole node, we need to delete the parent as well
                    // so the path index at this level would be the last child of the parent
                    zipper_left.push(None);
                } else {
                    zipper_left.push(Some(del_start - 1));
                }

                // parent is different
                {
                    // delete start..
                    let start_parent = self.tree.get_mut(start_path[level - 1].arena);
                    for x in start_parent.children.drain(del_start..) {
                        deleted.push(x.arena);
                    }
                }
                {
                    // delete ..end
                    let end_parent = self.tree.get_mut(end_path[level - 1].arena);
                    for x in end_parent.children.drain(..del_end) {
                        deleted.push(x.arena);
                    }
                }
            }

            level -= 1
            // this loop would break since level=0 is guaranteed to be the same
        }

        // release memory
        for x in deleted {
            self.tree.purge(x);
        }

        loop {
            zipper_left.push(Some(start_path[level].arr));
            zipper_right.push(Some(start_path[level].arr));
            if level == 0 {
                break;
            }
            level -= 1;
        }

        zipper_left.reverse(); // now in root to leaf order
        zipper_right.reverse(); // now in root to leaf order
        if let Some(path) = self.tree.try_get_path_from_indexes(&zipper_right) {
            self.tree.recursive_update_cache(path.as_ref().into());
        }

        if let Some(path) = self.tree.try_get_path_from_indexes(&zipper_left) {
            // otherwise the path is invalid (e.g. the tree is empty)
            seal(self.tree, path);
        }
    }
}

fn seal<B: BTreeTrait>(tree: &mut BTree<B>, path: Path) {
    // update cache
    let mut sibling_path = path.clone();
    let same = !tree.next_sibling(&mut sibling_path);
    tree.recursive_update_cache(path.as_ref().into());
    if !same {
        tree.recursive_update_cache(sibling_path.as_ref().into());
    }

    for i in 1..path.len() {
        let idx = path[i];
        let node = tree.get(idx.arena);
        let is_lack = node.is_lack();
        let path_ref: PathRef = path[0..=i].into();
        if is_lack {
            tree.handle_lack(&path_ref);
        }
    }

    for i in 1..path.len() {
        if sibling_path[i] == path[i] {
            continue;
        }
        let idx = sibling_path[i];
        let node = match tree.nodes.get(idx.arena) {
            Some(node) => node,
            None => {
                // it must be merged into path
                // println!("{} {:?} is merged into path", i, idx.arena); //DEBUG
                sibling_path[i] = path[i];
                continue;
            }
        };
        let is_lack = node.is_lack();
        let path_ref: PathRef = sibling_path[0..=i].into();
        if is_lack {
            tree.handle_lack(&path_ref);
        }
    }

    tree.try_shrink_levels();
}

impl<'a, B: BTreeTrait> Iter<'a, B> {
    pub fn new(tree: &'a BTree<B>, start: QueryResult, inclusive_end: QueryResult) -> Self {
        Self {
            tree,
            inclusive_end,
            path: start.node_path,
            done: false,
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Iter<'a, B> {
    type Item = (Path, &'a Node<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.inclusive_end.node_path.last() == self.path.last() {
            self.done = true;
        }

        let last = *self.path.last().unwrap();
        let path = self.path.clone();
        if !self.tree.next_sibling(&mut self.path) {
            self.done = true;
        }

        let node = self.tree.get(last.arena);
        Some((path, node))
    }
}
