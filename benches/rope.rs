use arbitrary::{Arbitrary, Unstructured};
use criterion::{criterion_group, criterion_main, Criterion};
use generic_btree::{HeapVec, Rope};
use jumprope::JumpRope;
use rand::{Rng, SeedableRng};
mod automerge;
mod utils;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Action {
    Insert { pos: u8, content: u8 },
    Delete { pos: u8, len: u8 },
}

pub fn bench(c: &mut Criterion) {
    bench_random(c);
    bench_automerge(c)
}

fn bench_random(c: &mut Criterion) {
    let mut b = c.benchmark_group("10K random insert/delete");
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let data: HeapVec<u8> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut gen = Unstructured::new(&data);
    let actions: [Action; 10_000] = gen.arbitrary().unwrap();
    b.bench_function("Rope 10K insert/delete", |b| {
        let guard = utils::PProfGuard::new("target/rope.svg");
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
        drop(guard);
    });

    b.bench_function("RawString", |b| {
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

    b.bench_function("JumpRope", |b| {
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

fn bench_automerge(c: &mut Criterion) {
    let mut b = c.benchmark_group("Automerge paper");
    let actions = automerge::get_automerge_actions();
    b.bench_function("Rope", |b| {
        let guard = utils::PProfGuard::new("target/rope-automerge.svg");
        b.iter(|| {
            let mut rope = Rope::new();
            for action in actions.iter() {
                if action.del > 0 {
                    rope.delete_range(action.pos..action.pos + action.del);
                }
                if !action.ins.is_empty() {
                    rope.insert(action.pos, &action.ins)
                }
            }
        });
        drop(guard);
    });

    b.bench_function("JumpRope", |b| {
        b.iter(|| {
            let mut rope = JumpRope::new();
            for action in actions.iter() {
                if action.del > 0 {
                    rope.remove(action.pos..action.pos + action.del);
                }
                if !action.ins.is_empty() {
                    rope.insert(action.pos, &action.ins)
                }
            }
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
