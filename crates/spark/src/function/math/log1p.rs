// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};
use arrow::array::{ArrayRef, AsArray, Float64Array};
use arrow::datatypes::{DataType, Float64Type};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Spark-compatible `log1p` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#log1p>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `ln(1 + expr)` as a DOUBLE, computed accurately when `expr` is \
        close to 0. Values less than or equal to -1 return NULL. NULL input returns NULL.",
    syntax_example = "log1p(expr)",
    sql_example = r#"```sql
> SELECT log1p(0);
0
> SELECT log1p(-1);
NULL
```"#,
    argument(
        name = "expr",
        description = "Numeric expression evaluated as DOUBLE. Values <= -1 and NULL return NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLog1p {
    input_types: SparkInputTypes,
}

impl Default for SparkLog1p {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkLog1p {
    pub fn new() -> Self {
        Self {
            input_types: SparkInputTypes::new(
                vec![SparkInputType::Double],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkLog1p {
    fn name(&self) -> &str {
        "log1p"
    }

    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        self.input_types.coerce(self.name(), arg_types)
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let input_types = self.input_types.with_config(config)?;
        Some(ScalarUDF::from(Self { input_types }))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;

        match arg {
            ColumnarValue::Scalar(ScalarValue::Float64(value)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Float64(
                    value.and_then(log1p_or_null),
                )))
            }
            ColumnarValue::Array(array) => {
                let result: Float64Array = array
                    .as_primitive::<Float64Type>()
                    .iter()
                    .map(|value| value.and_then(log1p_or_null))
                    .collect();

                Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
            }
            other => internal_err!(
                "Unexpected argument type for log1p after coercion: {}",
                other.data_type()
            ),
        }
    }
}

fn log1p_or_null(value: f64) -> Option<f64> {
    if value <= -1.0 {
        None
    } else {
        Some(value.ln_1p())
    }
}
