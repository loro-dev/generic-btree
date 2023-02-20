use arbitrary::{Arbitrary, Unstructured};
use criterion::{criterion_group, criterion_main, Criterion};
use generic_btree::{HeapVec, Rope};
use jumprope::JumpRope;
use rand::{Rng, SeedableRng};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Action {
    Insert { pos: u8, content: u8 },
    Delete { pos: u8, len: u8 },
}

pub fn bench(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let data: HeapVec<u8> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut gen = Unstructured::new(&data);
    let actions: [Action; 10_000] = gen.arbitrary().unwrap();
    c.bench_function("Rope 10K insert/delete", |b| {
        b.iter(|| {
            let mut rope = Rope::new();
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
        });
    });

    c.bench_function("RawString 10K insert/delete", |b| {
        b.iter(|| {
            let mut raw = String::new();
            for action in actions.iter() {
                match *action {
                    Action::Insert { pos, content } => {
                        let pos = pos as usize % (raw.len() + 1);
                        let s = content.to_string();
                        raw.insert_str(pos, &s);
                    }
                    Action::Delete { pos, len } => {
                        let pos = pos as usize % (raw.len() + 1);
                        let mut len = len as usize % 10;
                        len = len.min(raw.len() - pos);
                        raw.drain(pos..(pos + len));
                    }
                }
            }
        });
    });

    c.bench_function("JumpRope 10K insert/delete", |b| {
        b.iter(|| {
            let mut rope = JumpRope::new();
            for action in actions.iter() {
                match *action {
                    Action::Insert { pos, content } => {
                        let pos = pos as usize % (rope.len_bytes() + 1);
                        let s = content.to_string();
                        rope.insert(pos, &s);
                    }
                    Action::Delete { pos, len } => {
                        let pos = pos as usize % (rope.len_bytes() + 1);
                        let mut len = len as usize % 10;
                        len = len.min(rope.len_bytes() - pos);
                        rope.remove(pos..(pos + len));
                    }
                }
            }
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
