use arbitrary::{Arbitrary, Unstructured};
use generic_btree::{HeapVec, Rope};
use rand::{Rng, SeedableRng};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Action {
    Insert { pos: u8, content: u8 },
    Delete { pos: u8, len: u8 },
}

pub fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let data: HeapVec<u8> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut gen = Unstructured::new(&data);
    let actions: [Action; 10_000] = gen.arbitrary().unwrap();

    let mut rope = Rope::new();
    for _ in 0..10000 {
        for action in actions.iter() {
            match *action {
                Action::Insert { pos, content } => {
                    let pos = pos as usize % (rope.len() + 1);
                    let s = content.to_string();
                    rope.insert(pos, &s);
                }
                Action::Delete { pos, len } => {
                    let pos = pos as usize % (rope.len() + 1);
                    let mut len = len as usize % 10;
                    len = len.min(rope.len() - pos);
                    rope.delete_range(pos..(pos + len));
                }
            }
        }
    }
}
