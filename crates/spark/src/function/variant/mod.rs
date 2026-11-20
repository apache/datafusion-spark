// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark VARIANT functions, such as `is_variant_null`.

pub mod is_variant_null;

use datafusion_expr::ScalarUDF;
use datafusion_functions::make_udf_function;
use std::sync::Arc;

make_udf_function!(is_variant_null::SparkIsVariantNull, is_variant_null);

pub mod expr_fn {
    use datafusion_functions::export_functions;

    export_functions!((
        is_variant_null,
        "Returns true only if the VARIANT value is a variant null (a JSON `null`), and false otherwise, including for a SQL NULL.",
        arg1
    ));
}

pub fn functions() -> Vec<Arc<ScalarUDF>> {
    vec![is_variant_null()]
}
