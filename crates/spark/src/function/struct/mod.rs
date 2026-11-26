// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark struct functions.

pub mod spark_struct;

use datafusion_expr::ScalarUDF;
use datafusion_functions::make_udf_function;
use std::sync::Arc;

make_udf_function!(spark_struct::SparkStruct, r#struct);

pub mod expr_fn {
    use datafusion_functions::export_functions;

    export_functions!((
        r#struct,
        "Returns a struct with the given expressions as fields named `col1`, `col2`, ... (Spark semantics).",
        args
    ));
}

pub fn functions() -> Vec<Arc<ScalarUDF>> {
    vec![r#struct()]
}
