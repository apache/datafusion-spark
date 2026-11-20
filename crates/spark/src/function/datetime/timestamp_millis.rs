// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::TimestampMicrosecondArray;
use arrow::datatypes::{DataType, Field, FieldRef, TimeUnit};
use datafusion_common::cast::as_int64_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, internal_err, plan_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

/// Spark-compatible `timestamp_millis` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#timestamp_millis>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the timestamp that is `milliseconds` milliseconds after \
        1970-01-01 00:00:00 UTC. Negative values precede the epoch. Unlike DataFusion's \
        `to_timestamp_millis`, the result has microsecond precision, non-integer inputs \
        are rejected, and scaling to microseconds raises `long overflow` on overflow \
        in both ANSI and non-ANSI modes. Valid millisecond counts range from \
        -9223372036854775 through 9223372036854775, inclusive. The result uses the \
        session time zone (`datafusion.execution.time_zone`, default UTC), which \
        affects display but not the stored count. Returns NULL for NULL input.",
    syntax_example = "timestamp_millis(milliseconds)",
    sql_example = r#"```sql
> SELECT timestamp_millis(0::BIGINT);
1970-01-01T00:00:00Z
> SELECT timestamp_millis(-1::BIGINT);
1969-12-31T23:59:59.999Z
> SELECT timestamp_millis(NULL);
NULL
```"#,
    argument(
        name = "milliseconds",
        description = "TINYINT, SMALLINT, INT or BIGINT count of milliseconds since the UTC epoch. Arrow UInt8, UInt16 and UInt32 inputs are also accepted. UInt64 is rejected because its full range cannot fit in BIGINT. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTimestampMillis {
    input_types: SparkInputTypes,
    timezone: Arc<str>,
}

impl Default for SparkTimestampMillis {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTimestampMillis {
    pub fn new() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }

    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            input_types: SparkInputTypes::check_only(
                vec![SparkInputType::Integral],
                Volatility::Stable,
            ),
            timezone: config
                .execution
                .time_zone
                .as_ref()
                .map_or("UTC", |tz| tz.as_str())
                .into(),
        }
    }
}

impl ScalarUDFImpl for SparkTimestampMillis {
    fn name(&self) -> &str {
        "timestamp_millis"
    }

    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let types = self.input_types.coerce(self.name(), arg_types)?;
        if types[0] == DataType::UInt64 {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of \
                 `timestamp_millis` requires an integer type whose full range fits in \
                 BIGINT, however it has the type \"UInt64\". SQLSTATE: 42K09"
            );
        }
        Ok(vec![DataType::Int64])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = take_function_args(self.name(), args.arg_fields)?;
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&self.timezone))),
            input.is_nullable(),
        )))
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::from(Self::new_with_config(config)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [milliseconds] = take_function_args(self.name(), args.args)?;
        match milliseconds {
            ColumnarValue::Scalar(ScalarValue::Int64(value)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                    value.map(millis_to_micros).transpose()?,
                    Some(Arc::clone(&self.timezone)),
                )))
            }
            ColumnarValue::Array(array) => {
                let result: TimestampMicrosecondArray =
                    as_int64_array(array.as_ref())?.try_unary(millis_to_micros)?;
                Ok(ColumnarValue::Array(Arc::new(
                    result.with_timezone(Arc::clone(&self.timezone)),
                )))
            }
            other => {
                internal_err!(
                    "Unsupported arg {other:?} for Spark function `timestamp_millis`"
                )
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn millis_to_micros(milliseconds: i64) -> Result<i64> {
    milliseconds
        .checked_mul(1_000)
        .ok_or_else(|| exec_datafusion_err!("long overflow"))
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::TimestampMicrosecondType;

    use super::*;

    #[test]
    fn return_field_tracks_session_timezone_and_input_nullability() -> Result<()> {
        for timezone in [None, Some("America/Los_Angeles")] {
            let mut config = ConfigOptions::default();
            config.execution.time_zone = timezone.map(str::parse).transpose()?;
            let function = SparkTimestampMillis::new()
                .with_updated_config(&config)
                .unwrap();
            for nullable in [false, true] {
                let input = Arc::new(Field::new("millis", DataType::Int64, nullable));
                let field = function.inner().return_field_from_args(ReturnFieldArgs {
                    arg_fields: &[input],
                    scalar_arguments: &[None],
                })?;
                assert_eq!(field.is_nullable(), nullable);
                assert_eq!(
                    field.data_type(),
                    &DataType::Timestamp(
                        TimeUnit::Microsecond,
                        Some(timezone.unwrap_or("UTC").into()),
                    ),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn null_slots_do_not_overflow() -> Result<()> {
        // SQL cannot construct a null with an overflowing value in its storage.
        let input = Int64Array::new(
            vec![i64::MAX, 1, i64::MIN].into(),
            Some(vec![false, true, false].into()),
        );
        let input_field = Arc::new(Field::new("millis", DataType::Int64, true));
        let function = SparkTimestampMillis::new();
        let return_field = function.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[Arc::clone(&input_field)],
            scalar_arguments: &[None],
        })?;
        let result = function.invoke_with_args(ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(input))],
            arg_fields: vec![input_field],
            number_rows: 3,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        })?;
        let ColumnarValue::Array(result) = result else {
            panic!("expected array output");
        };
        assert_eq!(
            result.as_primitive::<TimestampMicrosecondType>(),
            &TimestampMicrosecondArray::from(vec![None, Some(1_000), None])
                .with_timezone("UTC"),
        );
        Ok(())
    }
}
