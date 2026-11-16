// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Fix-ups applied after Arrow's cast kernel, where its result differs from
//! Spark's.
//!
//! Ported from Comet's `native/spark-expr/src/conversion_funcs/utils.rs`.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, GenericStringArray, PrimitiveArray,
};
use arrow::compute::unary;
use arrow::datatypes::{DataType, Int64Type};
use arrow::error::ArrowError;
use datafusion_common::Result;

pub(crate) const MICROS_PER_SECOND: i64 = 1_000_000;

/// Applies `op` to every value of a primitive array, or to the values of a
/// dictionary array of that type.
fn unary_dyn<F, T>(array: &ArrayRef, op: F) -> Result<ArrayRef, ArrowError>
where
    T: ArrowPrimitiveType,
    F: Fn(T::Native) -> T::Native,
{
    if let Some(d) = array.as_any_dictionary_opt() {
        let new_values = unary_dyn::<F, T>(d.values(), op)?;
        return Ok(Arc::new(d.with_values(Arc::new(new_values))));
    }

    match array.as_primitive_opt::<T>() {
        Some(a) if PrimitiveArray::<T>::is_compatible(a.data_type()) => {
            Ok(Arc::new(unary::<T, F, T>(a, op)))
        }
        _ => Err(ArrowError::NotYetImplemented(format!(
            "Cannot perform unary operation of type {} on array of type {}",
            T::DATA_TYPE,
            array.data_type()
        ))),
    }
}

/// Corrects the result of Arrow's cast from `from_type` to `to_type` where it
/// differs from Spark's:
///
/// - `TIMESTAMP` to `BIGINT`: Arrow reinterprets the microseconds, while Spark
///   returns whole seconds, rounded down.
/// - `TIMESTAMP` to `STRING`: Arrow pads the fraction of a second to 3, 6 or 9
///   digits, while Spark drops trailing zeros.
///
/// `from_type` is the type before the cast, so it can still be a dictionary.
pub(crate) fn spark_cast_postprocess(
    array: ArrayRef,
    from_type: &DataType,
    to_type: &DataType,
) -> Result<ArrayRef> {
    let is_timestamp = match from_type {
        DataType::Dictionary(_, value_type) => {
            matches!(value_type.as_ref(), DataType::Timestamp(_, _))
        }
        other => matches!(other, DataType::Timestamp(_, _)),
    };
    if !is_timestamp {
        return Ok(array);
    }
    match to_type {
        DataType::Int64 => Ok(unary_dyn::<_, Int64Type>(&array, |v| {
            v.div_euclid(MICROS_PER_SECOND)
        })?),
        DataType::Utf8 => Ok(remove_trailing_zeroes(&array)),
        _ => Ok(array),
    }
}

/// Removes trailing zeros from the fraction of a second, as Spark does:
///
/// | Arrow                     | Spark                   |
/// |---------------------------|-------------------------|
/// | `1970-01-01 05:29:59.900` | `1970-01-01 05:29:59.9` |
/// | `1970-01-01 05:29:59.990` | `1970-01-01 05:29:59.99`|
/// | `1970-01-01 05:30:00`     | `1970-01-01 05:30:00`   |
/// | `1970-01-01 05:30:00.001` | `1970-01-01 05:30:00.001`|
fn remove_trailing_zeroes(array: &ArrayRef) -> ArrayRef {
    let string_array = array.as_string::<i32>();
    let result = string_array
        .iter()
        .map(|s| s.map(trim_fraction_zeroes))
        .collect::<GenericStringArray<i32>>();
    Arc::new(result)
}

fn trim_fraction_zeroes(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        DictionaryArray, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
    };
    use arrow::datatypes::{Int32Type, TimeUnit};

    const TS: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);

    #[test]
    fn test_timestamp_to_long_rounds_down() {
        // Arrow's cast kernel reinterprets the microseconds as the BIGINT.
        let micros: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1_500_000),
            Some(-1_500_000),
            Some(-1),
            None,
        ]));
        let result = spark_cast_postprocess(micros, &TS, &DataType::Int64).unwrap();
        assert_eq!(
            result.as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(1), Some(-2), Some(-1), None])
        );
    }

    #[test]
    fn test_dictionary_timestamp_to_long_rounds_down() {
        let values = Arc::new(Int64Array::from(vec![-1_500_000, 2_000_000]));
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![1, 0, 1]),
                values,
            )
            .unwrap(),
        );
        let from = DataType::Dictionary(Box::new(DataType::Int32), Box::new(TS));
        let result = spark_cast_postprocess(dict, &from, &DataType::Int64).unwrap();
        let result = result.as_dictionary::<Int32Type>();
        assert_eq!(
            result.values().as_primitive::<Int64Type>(),
            &Int64Array::from(vec![-2, 2])
        );
    }

    #[test]
    fn test_timestamp_to_string_drops_trailing_zeroes() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![
            Some("1970-01-01 05:29:59.900"),
            Some("1970-01-01 05:29:59.990"),
            Some("1970-01-01 05:29:59.999"),
            Some("1970-01-01 05:30:00"),
            Some("1970-01-01 05:30:00.001"),
            Some("1970-01-01 05:30:00.123400"),
            None,
        ]));
        let result = spark_cast_postprocess(strings, &TS, &DataType::Utf8).unwrap();
        assert_eq!(
            result.as_string::<i32>(),
            &StringArray::from(vec![
                Some("1970-01-01 05:29:59.9"),
                Some("1970-01-01 05:29:59.99"),
                Some("1970-01-01 05:29:59.999"),
                Some("1970-01-01 05:30:00"),
                Some("1970-01-01 05:30:00.001"),
                Some("1970-01-01 05:30:00.1234"),
                None,
            ])
        );
    }

    #[test]
    fn test_other_casts_are_unchanged() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1_500_000]));
        let result = spark_cast_postprocess(
            Arc::clone(&array),
            &DataType::Int32,
            &DataType::Int64,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&array, &result));

        let ts = TimestampMicrosecondArray::from(vec![1_500_000]);
        let array: ArrayRef = Arc::new(ts);
        let result =
            spark_cast_postprocess(Arc::clone(&array), &TS, &DataType::Date32).unwrap();
        assert!(Arc::ptr_eq(&array, &result));
    }
}
