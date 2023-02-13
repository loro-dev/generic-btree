#![no_main]

use std::collections::BTreeSet;

use arbitrary::Arbitrary;
use generic_btree::OrdTreeSet;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
enum Action {
    Insert(u8),
    Delete,
}

fuzz_target!(|actions: Vec<Action>| {
    let mut tree_a: BTreeSet<u8> = BTreeSet::new();
    let mut tree: OrdTreeSet<u8> = OrdTreeSet::new();
    for action in actions {
        match action {
            Action::Insert(value) => {
                tree_a.insert(value);
                tree.insert(value);
            }
            Action::Delete => {
                let value = tree_a.iter().nth(3).copied();
                if let Some(value) = value {
                    tree_a.remove(&value);
                    tree.delete(&value);
                }
            }
        }
    }

    assert_eq!(
        tree_a.iter().collect::<Vec<_>>(),
        tree.iter().collect::<Vec<_>>()
    );
});
