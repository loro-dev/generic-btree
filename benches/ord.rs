use std::collections::BTreeSet;

use criterion::{criterion_group, criterion_main, Criterion};
use generic_btree::{HeapVec, OrdTreeSet};
use rand::{Rng, SeedableRng};

pub fn bench(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let data: HeapVec<u64> = (0..100_000).map(|_| rng.gen()).collect();
    c.bench_function("OrdTree 100K insert", |b| {
        b.iter(|| {
            let mut tree: OrdTreeSet<u64> = OrdTreeSet::new();
            for &value in data.iter() {
                tree.insert(value);
            }
        });
    });

    c.bench_function("std BTree 100K insert", |b| {
        b.iter(|| {
            let mut tree: BTreeSet<u64> = BTreeSet::new();
            for &value in data.iter() {
                tree.insert(value);
            }
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
