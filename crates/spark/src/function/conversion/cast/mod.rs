// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible `CAST`.
//!
//! [`spark_cast`] is the kernel, for engines that build physical plans
//! themselves, such as Comet. The `spark_cast(expr, '<type>')` SQL function is
//! built on it.
//!
//! The kernel is being ported from Comet's
//! `native/spark-expr/src/conversion_funcs`, one source type family at a time
//! (<https://github.com/apache/datafusion-spark/issues/88>). A cast with no
//! Spark-compatible implementation yet returns a "not implemented" error, rather
//! than falling through to Arrow's cast and producing a result that differs
//! from Spark's.

mod udf;
mod utils;

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::{CastOptions, cast_with_options};
use arrow::datatypes::DataType;
use arrow::util::display::FormatOptions;
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err};
use datafusion_expr::ColumnarValue;

use crate::EvalMode;
use crate::function::timezone_utils::{
    array_with_timezone, cast_timestamp_to_ntz, timestamp_ntz_to_timestamp,
};
use utils::spark_cast_postprocess;

pub use udf::SparkCast;

/// The format Spark uses for a timestamp cast to a string. `%.f` writes the
/// fraction of a second only when it is not zero, and
/// [`spark_cast_postprocess`] then drops its trailing zeros.
const TIMESTAMP_FORMAT: Option<&str> = Some("%Y-%m-%d %H:%M:%S%.f");

/// Options for [`spark_cast`].
///
/// More options, such as the target Spark version, are added as the casts that
/// depend on them are ported.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub struct SparkCastOptions {
    /// How a value that cannot be cast is handled.
    pub eval_mode: EvalMode,
    /// The session time zone (`spark.sql.session.timeZone`), as an IANA name
    /// such as `America/New_York` or an offset such as `+05:00`. Casts between
    /// `TIMESTAMP` and `TIMESTAMP_NTZ`, `STRING` or `DATE` need it. It is empty
    /// when the caller has no time zone, and those casts then return an error.
    pub timezone: String,
}

impl SparkCastOptions {
    pub fn new(eval_mode: EvalMode, timezone: &str) -> Self {
        Self {
            eval_mode,
            timezone: timezone.to_string(),
        }
    }

    /// Options for a cast that does not involve a time zone.
    pub fn new_without_timezone(eval_mode: EvalMode) -> Self {
        Self::new(eval_mode, "")
    }
}

/// Casts `arg` to `data_type` the way Spark's `Cast` expression does.
///
/// An error is returned for a cast that Spark does not support, or that is not
/// implemented yet. Spark can never name a dictionary type, so a dictionary
/// `data_type` is an internal error.
pub fn spark_cast(
    arg: ColumnarValue,
    data_type: &DataType,
    cast_options: &SparkCastOptions,
) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(cast_array(
            array,
            data_type,
            cast_options,
        )?)),
        ColumnarValue::Scalar(scalar) => {
            let array = cast_array(scalar.to_array()?, data_type, cast_options)?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &array, 0,
            )?))
        }
    }
}

pub(crate) fn cast_array(
    array: ArrayRef,
    to_type: &DataType,
    cast_options: &SparkCastOptions,
) -> Result<ArrayRef> {
    use DataType::*;
    let from_type = array.data_type().clone();

    // Spark's SQL type grammar cannot express a dictionary as a cast target:
    // https://github.com/apache/spark/blob/v4.2.0/sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser/SqlBaseParser.g4#L1477-L1525
    if matches!(to_type, Dictionary(_, _)) {
        return internal_err!("Spark cannot specify dictionary types as cast targets");
    }

    if &from_type == to_type {
        return Ok(array);
    }

    let array = array_with_timezone(array, cast_options.timezone.clone(), Some(to_type))?;
    let eval_mode = cast_options.eval_mode;

    let native_cast_options = CastOptions {
        // Arrow returns NULL for a value it cannot cast when `safe` is set, and
        // an error otherwise.
        safe: eval_mode != EvalMode::Ansi,
        format_options: FormatOptions::new()
            .with_timestamp_tz_format(TIMESTAMP_FORMAT)
            .with_timestamp_format(TIMESTAMP_FORMAT),
    };

    // Engines such as Comet can read a string or binary Parquet column as a
    // dictionary, although Spark sees its value type. Cast the dictionary values
    // and unpack the result.
    if let Dictionary(_, value_type) = &from_type
        && matches!(value_type.as_ref(), Utf8 | LargeUtf8 | Binary | LargeBinary)
    {
        let dictionary = array.as_any_dictionary();
        let values = cast_array(Arc::clone(dictionary.values()), to_type, cast_options)?;
        let dictionary = dictionary.with_values(values);
        let unpacked = cast_with_options(&dictionary, to_type, &native_cast_options)?;
        return spark_cast_postprocess(unpacked, &from_type, to_type);
    }

    let result = match (&from_type, to_type) {
        // A NULL array has no values, so Arrow's cast only changes the type.
        (Null, _) => cast_with_options(&array, to_type, &native_cast_options)?,
        // `array_with_timezone` has already converted the local date-times to
        // UTC. This arm must come before the Arrow fallthrough, which would copy
        // the values without converting them.
        (Timestamp(_, None), Timestamp(_, Some(target_tz))) => {
            timestamp_ntz_to_timestamp(
                array,
                &cast_options.timezone,
                Some(target_tz.as_ref()),
            )?
        }
        (Timestamp(_, Some(_)), Timestamp(_, None)) => {
            cast_timestamp_to_ntz(array, &cast_options.timezone)?
        }
        _ if is_pending_port(&from_type, to_type, eval_mode) => {
            return not_impl_err!(
                "Spark-compatible cast from {from_type} to {to_type} in {eval_mode:?} mode \
                 is not implemented yet"
            );
        }
        _ if is_datafusion_spark_compatible(&from_type, to_type) => {
            cast_with_options(&array, to_type, &native_cast_options)?
        }
        _ => {
            return not_impl_err!(
                "Spark-compatible cast from {from_type} to {to_type} is not supported"
            );
        }
    };

    spark_cast_postprocess(result, &from_type, to_type)
}

/// Whether Spark's result for the cast differs from Arrow's, although
/// [`is_datafusion_spark_compatible`] accepts it, and the Spark-compatible
/// implementation has not been ported from Comet yet. Each family removes its
/// cases from here as it is ported.
fn is_pending_port(
    from_type: &DataType,
    to_type: &DataType,
    eval_mode: EvalMode,
) -> bool {
    use DataType::*;
    match (from_type, to_type) {
        // LEGACY wraps on overflow and ANSI raises CAST_OVERFLOW, while Arrow
        // returns NULL. TRY matches Arrow.
        (Int64, Int32 | Int16 | Int8) | (Int32, Int16 | Int8) | (Int16, Int8) => {
            eval_mode != EvalMode::Try
        }
        (Float32 | Float64 | Decimal128(_, _), Int8 | Int16 | Int32 | Int64) => {
            eval_mode != EvalMode::Try
        }
        // ANSI raises NUMERIC_VALUE_OUT_OF_RANGE, with Spark's message.
        (Decimal128(_, _), Decimal128(_, _)) => eval_mode == EvalMode::Ansi,
        // LEGACY uses Java's BigDecimal.toString, as in "0E-18".
        (Decimal128(_, _), Utf8) => eval_mode == EvalMode::Legacy,
        // Spark replaces invalid UTF-8, and Arrow raises an error.
        (Binary, Utf8) => true,
        _ => false,
    }
}

/// Whether Arrow's cast from `from_type` to `to_type` matches Spark's, for every
/// eval mode, once [`spark_cast_postprocess`] has run.
fn is_datafusion_spark_compatible(from_type: &DataType, to_type: &DataType) -> bool {
    use DataType::*;
    if from_type == to_type {
        return true;
    }
    match from_type {
        Null => matches!(to_type, List(_)),
        Boolean => matches!(
            to_type,
            Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Utf8
        ),
        Int8 | Int16 | Int32 | Int64 => matches!(
            to_type,
            Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Utf8
        ),
        Float32 | Float64 => matches!(
            to_type,
            Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64
        ),
        // Float32 and Float64 are absent: Arrow computes
        // `(unscaled as f64) / 10^scale`, which rounds twice and can differ from
        // Spark's correctly rounded BigDecimal.doubleValue by one ulp. Boolean is
        // absent because Arrow does not implement that cast.
        Decimal128(_, _) | Decimal256(_, _) => matches!(
            to_type,
            Int8 | Int16 | Int32 | Int64 | Decimal128(_, _) | Decimal256(_, _) | Utf8
        ),
        Utf8 => matches!(to_type, Binary),
        // Arrow reinterprets the days since the epoch, as Spark does.
        Date32 => matches!(to_type, Int32 | Utf8),
        Timestamp(_, _) => matches!(to_type, Int64 | Date32 | Utf8 | Timestamp(_, _)),
        Binary => matches!(to_type, Utf8),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, DictionaryArray, Int32Array, Int64Array, NullArray, StringArray,
        TimestampMicrosecondArray,
    };
    use arrow::datatypes::{Field, Int32Type, Int64Type, TimeUnit};
    use datafusion_common::DataFusionError;

    fn legacy() -> SparkCastOptions {
        SparkCastOptions::new(EvalMode::Legacy, "UTC")
    }

    fn timestamp(tz: Option<&str>) -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, tz.map(Into::into))
    }

    #[test]
    fn test_cast_to_dictionary_is_rejected() {
        let error = cast_array(
            Arc::new(StringArray::from(vec!["a"])),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            &legacy(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Spark cannot specify dictionary types as cast targets"),
            "{error}"
        );
    }

    #[test]
    fn test_cast_to_same_type_returns_input() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let result = cast_array(Arc::clone(&array), &DataType::Int32, &legacy()).unwrap();
        assert!(Arc::ptr_eq(&array, &result));
    }

    #[test]
    fn test_unsupported_cast_is_an_error() {
        let error = cast_array(
            Arc::new(StringArray::from(vec!["1"])),
            &DataType::Int32,
            &legacy(),
        )
        .unwrap_err();
        assert!(
            matches!(error, DataFusionError::NotImplemented(_)),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("cast from Utf8 to Int32 is not supported"),
            "{error}"
        );
    }

    /// A cast whose Spark-compatible implementation has not been ported must
    /// not fall through to Arrow's cast, whose result differs from Spark's.
    #[test]
    fn test_pending_cast_does_not_fall_through_to_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![i64::from(i32::MAX) + 1]));
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi] {
            let error = cast_array(
                Arc::clone(&array),
                &DataType::Int32,
                &SparkCastOptions::new_without_timezone(eval_mode),
            )
            .unwrap_err();
            assert!(
                matches!(error, DataFusionError::NotImplemented(_)),
                "{error}"
            );
            assert!(error.to_string().contains("not implemented yet"), "{error}");
        }

        // TRY matches Arrow's safe cast, which returns NULL on overflow.
        let result = cast_array(
            array,
            &DataType::Int32,
            &SparkCastOptions::new_without_timezone(EvalMode::Try),
        )
        .unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn test_arrow_fallthrough() {
        // A widening cast cannot fail in any mode.
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi, EvalMode::Try] {
            let result = cast_array(
                Arc::new(Int32Array::from(vec![Some(-1), None])),
                &DataType::Int64,
                &SparkCastOptions::new_without_timezone(eval_mode),
            )
            .unwrap();
            assert_eq!(
                result.as_primitive::<Int64Type>(),
                &Int64Array::from(vec![Some(-1), None])
            );
        }

        let result = cast_array(
            Arc::new(Int32Array::from(vec![Some(0), Some(7), None])),
            &DataType::Boolean,
            &legacy(),
        )
        .unwrap();
        let result = result.as_boolean();
        assert!(!result.value(0) && result.value(1) && result.is_null(2));
    }

    #[test]
    fn test_scalar() {
        let result = spark_cast(
            ColumnarValue::Scalar(ScalarValue::Int32(Some(42))),
            &DataType::Utf8,
            &legacy(),
        )
        .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert_eq!(scalar, ScalarValue::Utf8(Some("42".to_string())));

        let result = spark_cast(
            ColumnarValue::Scalar(ScalarValue::Int32(None)),
            &DataType::Int64,
            &legacy(),
        )
        .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert_eq!(scalar, ScalarValue::Int64(None));
    }

    #[test]
    fn test_cast_null_array() {
        for to_type in [
            DataType::Int32,
            DataType::Utf8,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
        ] {
            let result =
                cast_array(Arc::new(NullArray::new(3)), &to_type, &legacy()).unwrap();
            assert_eq!(result.data_type(), &to_type);
            assert_eq!(result.null_count(), 3);
        }
    }

    #[test]
    fn test_cast_timestamp_to_long_rounds_down() {
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![1_500_000, -1_500_000])
                .with_timezone("UTC"),
        );
        let result = cast_array(array, &DataType::Int64, &legacy()).unwrap();
        assert_eq!(
            result.as_primitive::<Int64Type>(),
            &Int64Array::from(vec![1, -2])
        );
    }

    #[test]
    fn test_cast_timestamp_to_string_in_session_timezone() {
        // 2024-01-15 15:30:00.1 UTC
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(1_705_332_600_100_000), None])
                .with_timezone("UTC"),
        );
        let options = SparkCastOptions::new(EvalMode::Legacy, "America/New_York");
        let result = cast_array(array, &DataType::Utf8, &options).unwrap();
        assert_eq!(
            result.as_string::<i32>(),
            &StringArray::from(vec![Some("2024-01-15 10:30:00.1"), None])
        );
    }

    #[test]
    fn test_cast_between_timestamp_and_timestamp_ntz() {
        // 2024-01-15 15:30:00 UTC is 10:30:00 in New York.
        let utc_micros = 1_705_332_600_000_000;
        let local_micros = 1_705_314_600_000_000;
        let options = SparkCastOptions::new(EvalMode::Legacy, "America/New_York");

        let ltz: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![utc_micros]).with_timezone("UTC"),
        );
        let ntz = cast_array(ltz, &timestamp(None), &options).unwrap();
        assert_eq!(ntz.data_type(), &timestamp(None));
        assert_eq!(
            ntz.as_primitive::<arrow::datatypes::TimestampMicrosecondType>()
                .value(0),
            local_micros
        );

        let ltz = cast_array(ntz, &timestamp(Some("UTC")), &options).unwrap();
        assert_eq!(ltz.data_type(), &timestamp(Some("UTC")));
        assert_eq!(
            ltz.as_primitive::<arrow::datatypes::TimestampMicrosecondType>()
                .value(0),
            utc_micros
        );
    }

    #[test]
    fn test_cast_timestamp_without_timezone_is_an_error() {
        let array: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC"));
        let options = SparkCastOptions::new_without_timezone(EvalMode::Legacy);
        let error = cast_array(array, &DataType::Utf8, &options).unwrap_err();
        assert!(error.to_string().contains("requires a timezone"), "{error}");
    }

    #[test]
    fn test_cast_invalid_timezone() {
        let array: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC"));
        let options = SparkCastOptions::new(EvalMode::Legacy, "Not a valid timezone");
        assert!(cast_array(array, &DataType::Date32, &options).is_err());
    }

    #[test]
    fn test_cast_unsupported_timestamp_to_date() {
        // chrono cannot represent every value of a microsecond timestamp.
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![i64::MAX])
                .with_timezone("Europe/Copenhagen"),
        );
        assert!(cast_array(array, &DataType::Date32, &legacy()).is_err());
    }

    #[test]
    fn test_cast_dictionary_string_unpacks_result() {
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![Some(1), None, Some(0), Some(1)]),
                Arc::new(StringArray::from(vec!["a", "bc"])),
            )
            .unwrap(),
        );
        let result = cast_array(dictionary, &DataType::Binary, &legacy()).unwrap();
        assert_eq!(
            result.as_binary::<i32>(),
            &BinaryArray::from(vec![
                Some(b"bc".as_slice()),
                None,
                Some(b"a".as_slice()),
                Some(b"bc".as_slice()),
            ])
        );
    }

    #[test]
    fn test_cast_dictionary_of_other_types_is_unsupported() {
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![0]),
                Arc::new(Int32Array::from(vec![1])),
            )
            .unwrap(),
        );
        assert!(cast_array(dictionary, &DataType::Int64, &legacy()).is_err());
    }
}
