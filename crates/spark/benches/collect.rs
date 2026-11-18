// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion_spark::function::aggregate::{collect_list, collect_set};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Builder;

const ROWS: usize = 2_000_000;
const BATCH_SIZE: usize = 8192;

fn make_context(num_groups: i64) -> SessionContext {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let mut rng = StdRng::seed_from_u64(42);
    let batches = (0..ROWS.div_ceil(BATCH_SIZE))
        .map(|i| {
            let len = BATCH_SIZE.min(ROWS - i * BATCH_SIZE);
            let g = Int64Array::from_iter_values(
                (0..len).map(|_| rng.random_range(0..num_groups)),
            );
            // 10% NULLs, values drawn from a small domain so collect_set dedups.
            let v: Int32Array = (0..len)
                .map(|_| (rng.random::<f32>() >= 0.1).then(|| rng.random_range(0..100)))
                .collect();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(g), Arc::new(v)])
                .unwrap()
        })
        .collect::<Vec<_>>();

    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema, vec![batches]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();
    ctx.register_udaf((*collect_list()).clone());
    ctx.register_udaf((*collect_set()).clone());
    ctx
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Builder::new_current_thread().build().unwrap();
    let mut group = c.benchmark_group("collect");
    group.sample_size(10);

    for num_groups in [1_000_i64, 200_000] {
        let ctx = make_context(num_groups);
        for func in ["collect_list", "collect_set"] {
            let sql = format!("SELECT g, {func}(v) FROM t GROUP BY g");
            group.bench_function(format!("{func}/{num_groups}_groups"), |b| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(ctx.sql(&sql).await.unwrap().collect().await.unwrap())
                    })
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
