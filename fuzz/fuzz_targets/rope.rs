#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use generic_btree::Rope;

#[derive(Arbitrary, Debug)]
enum Action {
    Insert { pos: u8, content: u8 },
    Delete { pos: u8, len: u8 },
}

fuzz_target!(|data: Vec<Action>| {
    let mut rope = Rope::new();
    let mut truth = String::new();
    for action in data {
        match action {
            Action::Insert { pos, content } => {
                let pos = pos as usize % (truth.len() + 1);
                let s = content.to_string();
                truth.insert_str(pos, &s);
                rope.insert(pos, s);
            }
            Action::Delete { pos, len } => {
                let pos = pos as usize % (truth.len() + 1);
                let mut len = len as usize % 10;
                len = len.min(truth.len() - pos);

                rope.delete_range(pos..(pos + len));
                truth.drain(pos..pos + len);
            }
        }
    }

    assert_eq!(rope.to_string(), truth);
});
