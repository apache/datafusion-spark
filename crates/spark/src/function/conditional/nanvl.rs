// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Float32Type, Float64Type};
use datafusion_common::Result;
use datafusion_common::exec_err;
use datafusion_common::utils::take_function_args;
use datafusion_expr::expr_fn::when;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark-compatible `nanvl` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#nanvl>
#[user_doc(
    doc_section(label = "Conditional Functions"),
    description = "Returns `expr1` if it is not NaN, or `expr2` if `expr1` is NaN. Unlike \
        DataFusion's built-in `nanvl`, which evaluates both arguments eagerly, this function \
        only evaluates `expr2` when `expr1` is NaN, matching Spark's short-circuiting behavior. \
        If `expr1` is NULL the result is NULL, even when `expr2` is not NULL; a NULL `expr2` \
        only matters when `expr1` is NaN. Both arguments are implicitly cast to DOUBLE (integer, \
        DECIMAL and STRING arguments included); if both arguments are FLOAT the result is FLOAT, \
        otherwise the result is DOUBLE. +/-Infinity and -0.0 are not NaN and are returned \
        unchanged, with the sign of zero preserved.",
    syntax_example = "nanvl(expr1, expr2)",
    sql_example = r#"```sql
> SELECT nanvl(cast('NaN' as double), 123);
123
> SELECT nanvl(5.0, CAST(NULL AS DOUBLE));
5
> SELECT nanvl(CAST(NULL AS DOUBLE), 5.0);
NULL
> SELECT nanvl(CAST('NaN' AS DOUBLE), CAST('NaN' AS DOUBLE));
NaN
```"#,
    argument(
        name = "expr1",
        description = "Numeric expression, cast to DOUBLE (or FLOAT if both arguments are \
            FLOAT). NULL returns NULL regardless of expr2."
    ),
    argument(
        name = "expr2",
        description = "Numeric expression, cast to match expr1. Only evaluated, and only \
            matters, when expr1 is NaN."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkNanvl {
    signature: Signature,
}

impl Default for SparkNanvl {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkNanvl {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkNanvl {
    fn name(&self) -> &str {
        "nanvl"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [t1, t2] = take_function_args("nanvl", arg_types)?;

        // Spark coerces non-FLOAT/DOUBLE arguments (ints, decimals, strings) to DOUBLE.
        // FLOAT is only kept when *both* sides are FLOAT (or NULL, which adapts to the
        // other side); any other combination, including FLOAT mixed with DOUBLE,
        // promotes to DOUBLE.
        let is_float32_or_null =
            |dt: &DataType| matches!(dt, DataType::Float32 | DataType::Null);

        let target = if is_float32_or_null(t1)
            && is_float32_or_null(t2)
            && !matches!((t1, t2), (DataType::Null, DataType::Null))
        {
            DataType::Float32
        } else {
            DataType::Float64
        };

        Ok(vec![target.clone(), target])
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [t1, _t2] = take_function_args("nanvl", arg_types)?;
        Ok(t1.clone())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [expr1, expr2] = take_function_args("nanvl", args)?;
        // `expr1 != expr1` is true exactly when expr1 is NaN (standard IEEE 754
        // trick), and is NULL when expr1 is NULL. That NULL naturally takes the
        // ELSE branch below (a CASE WHEN only takes the THEN branch when the
        // condition is true), returning expr1 (NULL) without ever evaluating
        // expr2 — which reproduces Spark's NULL-propagation rule for free.
        // CASE WHEN's branches are evaluated lazily, so expr2 is only
        // evaluated on rows where expr1 actually is NaN.
        let is_nan = expr1.clone().not_eq(expr1.clone());
        let case_expr = when(is_nan, expr2).otherwise(expr1)?;
        Ok(ExprSimplifyResult::Simplified(case_expr))
    }

    // This eager path is used only when `simplify()` doesn't run (e.g. Comet's
    // physical execution, which bypasses DataFusion's logical optimizer).
    // DataFusion's query engine instead takes the lazy CASE-expression rewrite
    // from `simplify()` above.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match args.args[0].data_type() {
            DataType::Float32 => {
                make_scalar_function(spark_nanvl_f32, vec![])(&args.args)
            }
            DataType::Float64 => {
                make_scalar_function(spark_nanvl_f64, vec![])(&args.args)
            }
            other => exec_err!("Unsupported data type {other} for function nanvl"),
        }
    }
}

fn spark_nanvl_f64(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [expr1, expr2] = take_function_args("nanvl", args)?;
    let expr1 = expr1.as_primitive::<Float64Type>();
    let expr2 = expr2.as_primitive::<Float64Type>();

    let result: Float64Array = (0..expr1.len())
        .map(|i| {
            if expr1.is_null(i) {
                None
            } else {
                let v1 = expr1.value(i);
                if v1.is_nan() {
                    if expr2.is_null(i) {
                        None
                    } else {
                        Some(expr2.value(i))
                    }
                } else {
                    Some(v1)
                }
            }
        })
        .collect();
    Ok(Arc::new(result))
}

fn spark_nanvl_f32(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [expr1, expr2] = take_function_args("nanvl", args)?;
    let expr1 = expr1.as_primitive::<Float32Type>();
    let expr2 = expr2.as_primitive::<Float32Type>();

    let result: Float32Array = (0..expr1.len())
        .map(|i| {
            if expr1.is_null(i) {
                None
            } else {
                let v1 = expr1.value(i);
                if v1.is_nan() {
                    if expr2.is_null(i) {
                        None
                    } else {
                        Some(expr2.value(i))
                    }
                } else {
                    Some(v1)
                }
            }
        })
        .collect();
    Ok(Arc::new(result))
}
