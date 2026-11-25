// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{AsArray, TimestampMicrosecondArray};
use arrow::datatypes::{
    DataType, Decimal128Type, Decimal256Type, Field, FieldRef, Float64Type, Int64Type,
    TimeUnit, i256,
};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, exec_err, internal_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

/// Spark-compatible `timestamp_seconds` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#timestamp_seconds>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Creates a timestamp from seconds since the UTC epoch, preserving fractions \
        to microsecond precision. Unlike DataFusion's `to_timestamp_seconds`, strings \
        are rejected and the result has microsecond precision. Integers raise \
        `long overflow` when scaling overflows. Decimals must convert exactly: fractional \
        microseconds raise `Rounding necessary`, and out-of-range values raise `Overflow`. \
        These errors occur in both ANSI and non-ANSI modes. FLOAT is widened to DOUBLE \
        before scaling; floating-point results truncate toward zero and saturate at the \
        BIGINT limits. NaN and infinity return NULL. NULL input returns NULL. The result \
        uses the session time zone (`datafusion.execution.time_zone`, default UTC), which \
        affects display but not the stored value.",
    syntax_example = "timestamp_seconds(seconds)",
    sql_example = r#"```sql
> SELECT timestamp_seconds(1.23::DECIMAL(3,2));
1970-01-01T00:00:01.230Z
> SELECT timestamp_seconds(-1::BIGINT);
1969-12-31T23:59:59Z
```"#,
    argument(
        name = "seconds",
        description = "Numeric seconds since the UTC epoch: TINYINT, SMALLINT, INT, BIGINT, DECIMAL, FLOAT or DOUBLE. Non-numeric inputs are rejected without implicit casts."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTimestampSeconds {
    input_types: SparkInputTypes,
    timezone: Arc<str>,
}

impl Default for SparkTimestampSeconds {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTimestampSeconds {
    pub fn new() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }

    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            input_types: SparkInputTypes::check_only(
                vec![SparkInputType::Numeric],
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

impl ScalarUDFImpl for SparkTimestampSeconds {
    fn name(&self) -> &str {
        "timestamp_seconds"
    }
    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let types = self.input_types.coerce(self.name(), arg_types)?;
        // Widen only after validating NumericType. Keep decimals exact and widen
        // FLOAT to DOUBLE before multiplying, as Spark does.
        Ok(vec![match types[0] {
            DataType::UInt64 => DataType::Decimal128(20, 0),
            DataType::Decimal32(p, s) | DataType::Decimal64(p, s) => {
                DataType::Decimal128(p, s)
            }
            DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => types[0].clone(),
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                DataType::Float64
            }
            _ => DataType::Int64,
        }])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = take_function_args(self.name(), args.arg_fields)?;
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&self.timezone))),
            input.is_nullable() || input.data_type().is_floating(),
        )))
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::from(Self::new_with_config(config)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [seconds] = take_function_args(self.name(), args.args)?;
        let scalar = matches!(seconds, ColumnarValue::Scalar(_));
        let array = seconds.into_array(1)?;
        let result: TimestampMicrosecondArray = match array.data_type() {
            DataType::Int64 => array.as_primitive::<Int64Type>().try_unary(|v| {
                v.checked_mul(1_000_000)
                    .ok_or_else(|| exec_datafusion_err!("long overflow"))
            })?,
            DataType::Float64 => array
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| {
                    v.filter(|v| v.is_finite())
                        .map(|v| (v * 1_000_000.0) as i64)
                })
                .collect(),
            DataType::Decimal128(_, scale) => array
                .as_primitive::<Decimal128Type>()
                .iter()
                .map(|v| {
                    v.map(|v| decimal_to_micros(i256::from(v), *scale))
                        .transpose()
                })
                .collect::<Result<_>>()?,
            DataType::Decimal256(_, scale) => array
                .as_primitive::<Decimal256Type>()
                .iter()
                .map(|v| v.map(|v| decimal_to_micros(v, *scale)).transpose())
                .collect::<Result<_>>()?,
            other => {
                return internal_err!(
                    "Unsupported arg {other:?} for Spark function `timestamp_seconds`"
                );
            }
        };
        let result = Arc::new(result.with_timezone(Arc::clone(&self.timezone)));
        if scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                result.as_ref(),
                0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn decimal_to_micros(value: i256, scale: i8) -> Result<i64> {
    if value == i256::ZERO {
        return Ok(0);
    }
    let exponent = 6 - i16::from(scale);
    let ten = i256::from(10);
    let micros = if exponent >= 0 {
        ten.checked_pow(exponent as u32)
            .and_then(|factor| value.checked_mul(factor))
            .ok_or_else(|| exec_datafusion_err!("Overflow"))?
    } else {
        let divisor = ten
            .checked_pow((-exponent) as u32)
            .ok_or_else(|| exec_datafusion_err!("Rounding necessary"))?;
        let quotient = value / divisor;
        // BigDecimal.longValueExact checks >19 integer digits before rounding.
        let limit = i256::from(10_000_000_000_000_000_000_i128);
        if quotient >= limit || quotient <= -limit {
            return exec_err!("Overflow");
        }
        if value % divisor != i256::ZERO {
            return exec_err!("Rounding necessary");
        }
        quotient
    };
    micros
        .to_i128()
        .and_then(|v| i64::try_from(v).ok())
        .ok_or_else(|| exec_datafusion_err!("Overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Decimal128Array, Int64Array};
    use arrow::datatypes::TimestampMicrosecondType;

    #[test]
    fn return_field_tracks_timezone_and_float_nullability() -> Result<()> {
        for timezone in [None, Some("America/Los_Angeles")] {
            let mut config = ConfigOptions::default();
            config.execution.time_zone = timezone.map(str::parse).transpose()?;
            let udf = SparkTimestampSeconds::new()
                .with_updated_config(&config)
                .unwrap();
            for data_type in [
                DataType::Int64,
                DataType::Decimal128(38, 7),
                DataType::Float32,
                DataType::Float64,
            ] {
                for nullable in [false, true] {
                    let field = udf.inner().return_field_from_args(ReturnFieldArgs {
                        arg_fields: &[Arc::new(Field::new(
                            "seconds",
                            data_type.clone(),
                            nullable,
                        ))],
                        scalar_arguments: &[None],
                    })?;
                    assert_eq!(field.is_nullable(), nullable || data_type.is_floating());
                    assert_eq!(
                        field.data_type(),
                        &DataType::Timestamp(
                            TimeUnit::Microsecond,
                            Some(timezone.unwrap_or("UTC").into())
                        )
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_values_in_null_slots_are_not_evaluated() -> Result<()> {
        // SQL cannot place overflowing/rounding values in a masked storage slot.
        let inputs: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::new(
                vec![i64::MAX, 1, i64::MIN].into(),
                Some(vec![false, true, false].into()),
            )),
            Arc::new(
                Decimal128Array::new(
                    vec![1, 10_000_000, i128::MAX].into(),
                    Some(vec![false, true, false].into()),
                )
                .with_precision_and_scale(38, 7)?,
            ),
        ];
        for input in inputs {
            let udf = SparkTimestampSeconds::new();
            let input_field =
                Arc::new(Field::new("seconds", input.data_type().clone(), true));
            let return_field = udf.return_field_from_args(ReturnFieldArgs {
                arg_fields: &[Arc::clone(&input_field)],
                scalar_arguments: &[None],
            })?;
            let result = udf.invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Array(input)],
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
                &TimestampMicrosecondArray::from(vec![None, Some(1_000_000), None])
                    .with_timezone("UTC")
            );
        }
        Ok(())
    }
}
