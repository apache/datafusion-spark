// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::compute::kernels::cast::{CastOptions, cast_with_options};
use arrow::compute::kernels::zip::zip;
use arrow::datatypes::{DataType, Float32Type, Float64Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err, plan_err};
use datafusion_expr::expr_fn::when;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ExprSchemable, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_functions::math::expr_fn::isnan;
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

        for t in [t1, t2] {
            if !is_nanvl_coercible(t) {
                return plan_err!(
                    "nanvl does not support the input type {t}; expected a \
                     numeric, decimal, string or null type"
                );
            }
        }

        let is_string = |t: &DataType| {
            matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
        };

        if is_string(t1) || is_string(t2) {
            // A STRING argument's numeric parse can fail at runtime, and
            // whether that failure is NULL (legacy) or an error (ANSI)
            // depends on `enable_ansi_mode`. Letting the analyzer insert
            // its own implicit Cast here would always use a strict cast
            // regardless of ANSI mode, so a STRING argument is left as-is
            // and cast manually, ANSI-aware, in `invoke_with_args`. Spark
            // resolves a STRING argument against DOUBLE, never FLOAT, so
            // the other side is coerced to DOUBLE too (a numeric -> DOUBLE
            // cast never fails, so the analyzer can still insert that one
            // safely).
            let resolve = |t: &DataType| {
                if is_string(t) {
                    t.clone()
                } else {
                    DataType::Float64
                }
            };
            return Ok(vec![resolve(t1), resolve(t2)]);
        }

        // Spark coerces a NULL *left* argument straight to DOUBLE
        // (TypeCollection(DoubleType, FloatType)'s default for an untyped
        // NULL), but a NULL *right* argument instead adopts whatever type
        // the left argument resolves to. The rule is not symmetric:
        // nanvl(NULL, FLOAT) -> DOUBLE, but nanvl(FLOAT, NULL) -> FLOAT.
        let is_float32 = |dt: &DataType| matches!(dt, DataType::Float32);

        let target = match (t1, t2) {
            (DataType::Null, _) => DataType::Float64,
            (t1, DataType::Null) if is_float32(t1) => DataType::Float32,
            (t1, t2) if is_float32(t1) && is_float32(t2) => DataType::Float32,
            _ => DataType::Float64,
        };

        Ok(vec![target.clone(), target])
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [t1, _t2] = take_function_args("nanvl", arg_types)?;
        match t1 {
            // A STRING argument stays unresolved through coerce_types (see
            // above), but it always produces a DOUBLE result.
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                Ok(DataType::Float64)
            }
            other => Ok(other.clone()),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [expr1, expr2] = take_function_args("nanvl", args)?;

        // A volatile expr1 (e.g. `rand()`) must be evaluated exactly once.
        // The CASE rewrite below references expr1 twice — once in the
        // condition and once in the ELSE branch — which would evaluate it
        // separately in each place. Skip the rewrite for volatile input and
        // fall back to `invoke_with_args`, which evaluates each argument
        // array exactly once.
        if expr1.is_volatile() {
            return Ok(ExprSimplifyResult::Original(vec![expr1, expr2]));
        }

        // A string argument is deliberately left unresolved by
        // `coerce_types` (see its comment) so that `invoke_with_args` can
        // apply an ANSI-aware manual cast instead of the analyzer's own
        // strict one. That means a string argument's `DataType` here is
        // still Utf8/LargeUtf8/Utf8View, not Float64 — and a `CASE`
        // expression requires its THEN and ELSE branches to share a single
        // type, so building one with a Utf8 branch and a Float64 branch
        // fails during logical planning. Skip the rewrite in that case and
        // let `invoke_with_args` handle it eagerly instead.
        let is_string_expr = |e: &Expr| {
            matches!(
                e.get_type(info.schema()),
                Ok(DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
            )
        };
        if is_string_expr(&expr1) || is_string_expr(&expr2) {
            return Ok(ExprSimplifyResult::Original(vec![expr1, expr2]));
        }

        // Arrow/DataFusion comparison kernels use IEEE-754 totalOrder
        // semantics, under which NaN compares equal to itself, so
        // `expr1 != expr1` is never true for a NaN value coming from a
        // column — it only appears to work for literal NaNs that the
        // simplifier constant-folds ahead of time. Use `isnan` instead,
        // which inspects the actual bit pattern.
        let is_nan = isnan(expr1.clone());
        let case_expr = when(is_nan, expr2).otherwise(expr1)?;
        Ok(ExprSimplifyResult::Simplified(case_expr))
    }

    // This eager path is used both for any Comet physical execution that
    // bypasses the logical optimizer, and whenever either argument is a
    // STRING (see `coerce_types`), since that case needs an ANSI-aware
    // manual cast rather than the simplify() rewrite below.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let is_string_type = |dt: &DataType| {
            matches!(
                dt,
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
        };

        if args.args.iter().any(|a| is_string_type(&a.data_type())) {
            let cast_options = CastOptions {
                safe: !args.config_options.execution.enable_ansi_mode,
                ..Default::default()
            };
            let arrays = ColumnarValue::values_to_arrays(&args.args)?;
            let casted: Vec<ArrayRef> = arrays
                .iter()
                .map(|a| Ok(cast_with_options(a, &DataType::Float64, &cast_options)?))
                .collect::<Result<_>>()?;
            return Ok(ColumnarValue::Array(spark_nanvl_f64(&casted)?));
        }

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

/// Spark's `ImplicitCastInputTypes` for `nanvl` accepts
/// `TypeCollection(DoubleType, FloatType)`, which numeric types (integers,
/// FLOAT, DOUBLE, DECIMAL), STRING and NULL can all implicitly cast into.
/// Other types, such as BOOLEAN, cannot, and must be rejected rather than
/// silently accepted just because Arrow happens to support a cast.
fn is_nanvl_coercible(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Null
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
    )
}

fn spark_nanvl_f64(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [expr1, expr2] = take_function_args("nanvl", args)?;
    let expr1_prim = expr1.as_primitive::<Float64Type>();
    let mask: BooleanArray = (0..expr1_prim.len())
        .map(|i| !expr1_prim.is_null(i) && expr1_prim.value(i).is_nan())
        .collect();
    nanvl_from_mask(&mask, expr2, expr1)
}

fn spark_nanvl_f32(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [expr1, expr2] = take_function_args("nanvl", args)?;
    let expr1_prim = expr1.as_primitive::<Float32Type>();
    let mask: BooleanArray = (0..expr1_prim.len())
        .map(|i| !expr1_prim.is_null(i) && expr1_prim.value(i).is_nan())
        .collect();
    nanvl_from_mask(&mask, expr2, expr1)
}

/// Selects `expr2` where `mask` ("is expr1 NaN", with no nulls of its own)
/// is true, and `expr1` otherwise — vectorised via Arrow's `zip` kernel
/// instead of a per-row loop. `expr1`'s own nulls flow through unchanged on
/// the `false` branch, which is exactly what gives a NULL `expr1` priority
/// over everything else; a NULL `expr2` only surfaces where the mask is
/// actually true.
fn nanvl_from_mask(
    mask: &BooleanArray,
    expr2: &ArrayRef,
    expr1: &ArrayRef,
) -> Result<ArrayRef> {
    Ok(zip(mask, expr2, expr1)?)
}
