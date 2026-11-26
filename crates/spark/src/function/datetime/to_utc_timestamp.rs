// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use super::timezone::TimeZoneRules;
use arrow::array::{Array, ArrayRef, AsArray, PrimitiveBuilder, StringArrayType};
use arrow::datatypes::TimeUnit;
use arrow::datatypes::{
    ArrowTimestampType, DataType, Field, FieldRef, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use datafusion_common::types::{NativeType, logical_string};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_datafusion_err, exec_err, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark-compatible `to_utc_timestamp` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#to_utc_timestamp>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Treats `ts` as a wall-clock time in time zone `tz` and returns the UTC wall-clock time. \
        This is the inverse of `from_utc_timestamp`. DataFusion has no built-in equivalent. A string `ts` is cast to a timestamp, \
        so a string with an offset is first converted to UTC. The result has the same type as `ts`, including any time zone \
        it carries. Returns NULL if either argument is NULL. `tz` accepts the same spellings as `from_utc_timestamp`: \
        region IDs, offsets such as `+01:00`, `GMT+1`, `UTC+1:2` and `UT-010203`, Java short IDs such as `PST`, and legacy \
        `SystemV/` IDs. An offset beyond 18 hours or an unknown zone is an error. A time skipped when clocks go forward \
        moves forward by the length of the gap, and a time that happens twice when clocks go back takes the earlier offset. \
        Differences from Spark: `tz` is only parsed for rows where both arguments are non-NULL, so an invalid constant `tz` \
        can return NULL where Spark raises an error; conversion is limited to Chrono's calendar range, which is narrower \
        than the range Spark implements but covers Spark's documented range; and region rules from `chrono-tz` may differ \
        after 2099, because it does not extrapolate later daylight saving time transitions.",
    syntax_example = "to_utc_timestamp(ts, tz)",
    sql_example = r#"```sql
> SELECT to_utc_timestamp('2016-08-31'::timestamp, 'Asia/Seoul');
2016-08-30T15:00:00
> SELECT to_utc_timestamp('2018-03-13T06:18:23+02:00', 'America/New_York');
2018-03-13T08:18:23
> SELECT to_utc_timestamp('2024-07-01T00:00:00'::timestamp, 'GMT+1');
2024-06-30T23:00:00
> SELECT to_utc_timestamp('2016-08-31'::timestamp, NULL);
NULL
```"#,
    argument(
        name = "ts",
        description = "Timestamp expression, interpreted as a wall-clock time in `tz`. A string is cast to a timestamp. NULL returns NULL."
    ),
    argument(
        name = "tz",
        description = "String time zone that `ts` is in. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkToUtcTimestamp {
    signature: Signature,
}

impl Default for SparkToUtcTimestamp {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkToUtcTimestamp {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Timestamp,
                        vec![TypeSignatureClass::Native(logical_string())],
                        NativeType::Timestamp(TimeUnit::Microsecond, None),
                    ),
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkToUtcTimestamp {
    fn name(&self) -> &str {
        "to_utc_timestamp"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());

        Ok(Arc::new(Field::new(
            self.name(),
            args.arg_fields[0].data_type().clone(),
            nullable,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(to_utc_timestamp, vec![])(&args.args)
    }
}

fn to_utc_timestamp(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [timestamp, timezone] = take_function_args("to_utc_timestamp", args)?;

    match timestamp.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, tz_opt) => {
            process_timestamp_with_tz_array::<TimestampNanosecondType>(
                timestamp,
                timezone,
                tz_opt.clone(),
            )
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz_opt) => {
            process_timestamp_with_tz_array::<TimestampMicrosecondType>(
                timestamp,
                timezone,
                tz_opt.clone(),
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz_opt) => {
            process_timestamp_with_tz_array::<TimestampMillisecondType>(
                timestamp,
                timezone,
                tz_opt.clone(),
            )
        }
        DataType::Timestamp(TimeUnit::Second, tz_opt) => {
            process_timestamp_with_tz_array::<TimestampSecondType>(
                timestamp,
                timezone,
                tz_opt.clone(),
            )
        }
        ts_type => {
            exec_err!("`to_utc_timestamp`: unsupported argument types: {ts_type}")
        }
    }
}

fn process_timestamp_with_tz_array<T: ArrowTimestampType>(
    ts_array: &ArrayRef,
    tz_array: &ArrayRef,
    tz_opt: Option<Arc<str>>,
) -> Result<ArrayRef> {
    match tz_array.data_type() {
        DataType::Utf8 => {
            process_arrays::<T, _>(tz_opt, ts_array, tz_array.as_string::<i32>())
        }
        DataType::LargeUtf8 => {
            process_arrays::<T, _>(tz_opt, ts_array, tz_array.as_string::<i64>())
        }
        DataType::Utf8View => {
            process_arrays::<T, _>(tz_opt, ts_array, tz_array.as_string_view())
        }
        other => {
            exec_err!("`to_utc_timestamp`: timezone must be a string type, got {other}")
        }
    }
}

fn process_arrays<'a, T: ArrowTimestampType, S>(
    return_tz_opt: Option<Arc<str>>,
    ts_array: &ArrayRef,
    tz_array: &'a S,
) -> Result<ArrayRef>
where
    &'a S: StringArrayType<'a>,
{
    let ts_primitive = ts_array.as_primitive::<T>();
    let mut builder = PrimitiveBuilder::<T>::with_capacity(ts_array.len());

    // Parse a scalar zone once per batch and avoid re-parsing consecutive column values
    let mut last_timezone = None;

    for (ts_opt, tz_opt) in ts_primitive.iter().zip(tz_array.iter()) {
        match (ts_opt, tz_opt) {
            (Some(ts), Some(tz_str)) => {
                let timezone = match last_timezone {
                    Some((previous, timezone)) if previous == tz_str => timezone,
                    _ => {
                        let timezone = parse_timezone(tz_str)?;
                        last_timezone = Some((tz_str, timezone));
                        timezone
                    }
                };
                let val = timezone.local_to_utc::<T>(ts)?;
                builder.append_value(val);
            }
            _ => builder.append_null(),
        }
    }

    builder = builder.with_timezone_opt(return_tz_opt);
    Ok(Arc::new(builder.finish()))
}

fn parse_timezone(timezone: &str) -> Result<TimeZoneRules> {
    TimeZoneRules::parse(timezone).map_err(|e| {
        exec_datafusion_err!("`to_utc_timestamp`: invalid timezone '{timezone}': {e}")
    })
}
