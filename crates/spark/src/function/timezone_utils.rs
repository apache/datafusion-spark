// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Helpers for applying Spark's session time zone to timestamp arrays.
//!
//! Spark stores a `TIMESTAMP` as microseconds since the epoch in UTC and a
//! `TIMESTAMP_NTZ` as microseconds since the epoch of a local date-time. Moving
//! between them, or rendering a `TIMESTAMP` as a local string or date, needs
//! the session time zone.
//!
//! Ported from Comet's `native/spark-expr/src/utils.rs`.

use std::sync::Arc;

use arrow::array::timezone::Tz;
use arrow::array::types::{
    Int32Type, TimestampMicrosecondType, TimestampMillisecondType,
};
use arrow::array::{
    Array, ArrayRef, PrimitiveArray, as_dictionary_array, as_primitive_array,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::error::ArrowError;
use arrow::temporal_conversions::as_datetime;
use chrono::{DateTime, LocalResult, NaiveDateTime, Offset, TimeZone};

/// Prepares a timestamp array for a cast to `to_type`, using the session
/// `timezone`.
///
/// | Conversion                         | Input array             | Output array                                    |
/// |------------------------------------|-------------------------|-------------------------------------------------|
/// | `TIMESTAMP` to string or date      | UTC                     | Shifted to local time, with the time zone removed |
/// | `TIMESTAMP` to `TIMESTAMP`         | UTC                     | The input, tagged with `timezone`               |
/// | `TIMESTAMP_NTZ` to string or date  | Local time              | The input                                       |
/// | `TIMESTAMP_NTZ` to `TIMESTAMP`     | Local time in `timezone`| UTC, tagged with the target time zone           |
/// | `TIMESTAMP_NTZ` to any other type  |                         | An error                                        |
///
/// Arrays of other types are returned unchanged.
pub(crate) fn array_with_timezone(
    array: ArrayRef,
    timezone: String,
    to_type: Option<&DataType>,
) -> Result<ArrayRef, ArrowError> {
    match array.data_type() {
        DataType::Timestamp(_, None) => {
            require_timezone(&timezone)?;
            match to_type {
                Some(DataType::Utf8) | Some(DataType::Date32) => Ok(array),
                Some(DataType::Timestamp(_, Some(target_tz))) => {
                    // Interpret NTZ as local time in the session time zone, and tag the
                    // output with the target time zone the caller expects.
                    timestamp_ntz_to_timestamp(
                        array,
                        timezone.as_str(),
                        Some(target_tz.as_ref()),
                    )
                }
                _ => Err(ArrowError::CastError(format!(
                    "Cannot convert from {:?} to {:?}",
                    array.data_type(),
                    to_type
                ))),
            }
        }
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            require_timezone(&timezone)?;
            let array = as_primitive_array::<TimestampMicrosecondType>(&array);
            let array =
                Arc::new(array.clone().with_timezone(timezone.clone())) as ArrayRef;
            match to_type {
                Some(DataType::Utf8) | Some(DataType::Date32) => {
                    pre_timestamp_cast(array, timezone)
                }
                _ => Ok(array),
            }
        }
        DataType::Timestamp(TimeUnit::Millisecond, Some(_)) => {
            require_timezone(&timezone)?;
            let array = as_primitive_array::<TimestampMillisecondType>(&array);
            let array =
                Arc::new(array.clone().with_timezone(timezone.clone())) as ArrayRef;
            match to_type {
                Some(DataType::Utf8) | Some(DataType::Date32) => {
                    pre_timestamp_cast(array, timezone)
                }
                _ => Ok(array),
            }
        }
        DataType::Dictionary(key_type, value_type)
            if key_type.as_ref() == &DataType::Int32
                && matches!(value_type.as_ref(), DataType::Timestamp(_, _)) =>
        {
            let dict = as_dictionary_array::<Int32Type>(&array);
            let values =
                array_with_timezone(Arc::clone(dict.values()), timezone, to_type)?;
            Ok(Arc::new(dict.with_values(values)))
        }
        _ => Ok(array),
    }
}

/// Converting a timestamp needs the session time zone. An empty one means the
/// caller built the cast without it, which is reported as an error rather than
/// a panic.
fn require_timezone(timezone: &str) -> Result<(), ArrowError> {
    if timezone.is_empty() {
        return Err(ArrowError::InvalidArgumentError(
            "Converting a timestamp requires a timezone, but none was given".to_string(),
        ));
    }
    Ok(())
}

fn datetime_cast_err(value: i64) -> ArrowError {
    ArrowError::CastError(format!(
        "Cannot convert timestamp {value} to a date-time. Only dates between \
         Jan 1, 262145 BCE and Dec 31, 262143 CE are supported",
    ))
}

/// Resolves a local date-time in `tz` to an instant, the way Spark's
/// `LocalDateTime.atZone` does.
///
/// An ambiguous time, in the hour repeated when clocks go back, resolves to the
/// earlier offset. A time in the gap when clocks go forward is shifted forward
/// by the length of the gap, by applying the offset in effect before it.
pub(crate) fn resolve_local_datetime(
    tz: &Tz,
    local_datetime: NaiveDateTime,
) -> DateTime<Tz> {
    match tz.from_local_datetime(&local_datetime) {
        LocalResult::Single(dt) => dt,
        LocalResult::Ambiguous(dt, _) => dt,
        LocalResult::None => {
            // Find the offset in effect before the transition.
            let probe = local_datetime - chrono::Duration::hours(3);
            let pre_offset = match tz.from_local_datetime(&probe) {
                LocalResult::Single(dt) => dt.offset().fix(),
                LocalResult::Ambiguous(dt, _) => dt.offset().fix(),
                LocalResult::None => {
                    // The offset cannot be determined, so fall back to UTC.
                    return local_datetime.and_utc().with_timezone(tz);
                }
            };
            let offset_secs = pre_offset.local_minus_utc() as i64;
            let utc_naive = local_datetime - chrono::Duration::seconds(offset_secs);
            utc_naive.and_utc().with_timezone(tz)
        }
    }
}

/// Converts a `TIMESTAMP_NTZ` array, whose values are local date-times in `tz`,
/// to a `TIMESTAMP` array in UTC tagged with `to_timezone`.
///
/// Arrays that are not `TIMESTAMP_NTZ` in milliseconds or microseconds are
/// returned unchanged.
pub(crate) fn timestamp_ntz_to_timestamp(
    array: ArrayRef,
    tz: &str,
    to_timezone: Option<&str>,
) -> Result<ArrayRef, ArrowError> {
    fn convert<T: arrow::datatypes::ArrowTimestampType>(
        array: &ArrayRef,
        tz: &str,
        to_timezone: Option<&str>,
        to_value: impl Fn(DateTime<Tz>) -> i64,
    ) -> Result<ArrayRef, ArrowError> {
        let array = as_primitive_array::<T>(array);
        let tz: Tz = tz.parse()?;
        let array: PrimitiveArray<T> = array.try_unary(|value| {
            as_datetime::<T>(value)
                .ok_or_else(|| datetime_cast_err(value))
                .map(|local_datetime| {
                    to_value(resolve_local_datetime(&tz, local_datetime))
                })
        })?;
        Ok(Arc::new(match to_timezone {
            Some(to_tz) => array.with_timezone(to_tz),
            None => array,
        }))
    }

    match array.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            convert::<TimestampMicrosecondType>(&array, tz, to_timezone, |dt| {
                dt.timestamp_micros()
            })
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            convert::<TimestampMillisecondType>(&array, tz, to_timezone, |dt| {
                dt.timestamp_millis()
            })
        }
        _ => Ok(array),
    }
}

/// Converts a `TIMESTAMP` array in UTC to a `TIMESTAMP_NTZ` array of local
/// date-times in the session `timezone`, like Spark's
/// `convertTz(ts, ZoneOffset.UTC, zoneId)`.
pub(crate) fn cast_timestamp_to_ntz(
    array: ArrayRef,
    timezone: &str,
) -> Result<ArrayRef, ArrowError> {
    let tz: Tz = timezone.parse()?;
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            let array = as_primitive_array::<TimestampMicrosecondType>(&array);
            let result: PrimitiveArray<TimestampMicrosecondType> =
                array.try_unary(|value| {
                    as_datetime::<TimestampMicrosecondType>(value)
                        .ok_or_else(|| datetime_cast_err(value))
                        .map(|utc_naive| {
                            // Store the local date-time as if it were UTC, which is how
                            // NTZ values are encoded.
                            let local_dt = tz.from_utc_datetime(&utc_naive);
                            local_dt.naive_local().and_utc().timestamp_micros()
                        })
                })?;
            // No time zone on the output type means TIMESTAMP_NTZ.
            Ok(Arc::new(result))
        }
        _ => Err(ArrowError::CastError(format!(
            "cast_timestamp_to_ntz: unexpected input type {:?}",
            array.data_type()
        ))),
    }
}

/// Shifts a `TIMESTAMP` array to local time in `timezone` and drops its time
/// zone, before a cast to string or date. Spark renders the local time without
/// an offset, while Arrow's cast kernel appends one when the type has a time
/// zone.
fn pre_timestamp_cast(array: ArrayRef, timezone: String) -> Result<ArrayRef, ArrowError> {
    fn shift<T: arrow::datatypes::ArrowTimestampType>(
        array: &ArrayRef,
        tz: &Tz,
        to_value: impl Fn(NaiveDateTime) -> i64,
    ) -> Result<ArrayRef, ArrowError> {
        let array = as_primitive_array::<T>(array);
        let array: PrimitiveArray<T> = array.try_unary(|value| {
            as_datetime::<T>(value)
                .ok_or_else(|| datetime_cast_err(value))
                .map(|datetime| {
                    let offset = tz.offset_from_utc_datetime(&datetime).fix();
                    to_value(datetime + offset)
                })
        })?;
        Ok(Arc::new(array))
    }

    let tz: Tz = timezone.parse()?;
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            shift::<TimestampMicrosecondType>(&array, &tz, |dt| {
                dt.and_utc().timestamp_micros()
            })
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            shift::<TimestampMillisecondType>(&array, &tz, |dt| {
                dt.and_utc().timestamp_millis()
            })
        }
        _ => Ok(array),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{TimestampMicrosecondArray, TimestampMillisecondArray};

    fn micros_for(datetime: &str) -> i64 {
        NaiveDateTime::parse_from_str(datetime, "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp_micros()
    }

    /// A TIMESTAMP_NTZ array holding one local date-time.
    fn ntz(local_datetime: &str) -> ArrayRef {
        Arc::new(TimestampMicrosecondArray::from(vec![micros_for(
            local_datetime,
        )]))
    }

    /// A TIMESTAMP array holding one UTC date-time, tagged with `tz`.
    fn ts_with_tz(utc_datetime: &str, tz: &str) -> ArrayRef {
        Arc::new(
            TimestampMicrosecondArray::from(vec![micros_for(utc_datetime)])
                .with_timezone(tz.to_string()),
        )
    }

    fn first_value(array: &ArrayRef) -> i64 {
        as_primitive_array::<TimestampMicrosecondType>(array).value(0)
    }

    #[test]
    fn test_timestamp_ntz_to_timestamp_handles_non_existent_time() {
        // 01:30 does not exist in London on 2024-03-31. Spark moves it forward an
        // hour, to 02:30 BST, which is 01:30 UTC.
        let output =
            timestamp_ntz_to_timestamp(ntz("2024-03-31 01:30:00"), "Europe/London", None)
                .unwrap();
        assert_eq!(first_value(&output), micros_for("2024-03-31 01:30:00"));
    }

    #[test]
    fn test_timestamp_ntz_to_timestamp_handles_ambiguous_time() {
        // 01:30 happens twice in London on 2024-10-27. Spark picks the earlier
        // offset, BST, which is 00:30 UTC.
        let output =
            timestamp_ntz_to_timestamp(ntz("2024-10-27 01:30:00"), "Europe/London", None)
                .unwrap();
        assert_eq!(first_value(&output), micros_for("2024-10-27 00:30:00"));
    }

    #[test]
    fn test_timestamp_ntz_to_timestamp_tags_target_timezone() {
        let output = timestamp_ntz_to_timestamp(
            ntz("2024-01-15 10:30:00"),
            "America/New_York",
            Some("UTC"),
        )
        .unwrap();
        assert_eq!(first_value(&output), micros_for("2024-01-15 15:30:00"));
        assert_eq!(
            output.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );
    }

    #[test]
    fn test_timestamp_ntz_to_timestamp_milliseconds() {
        let millis = micros_for("2024-01-15 10:30:00") / 1000;
        let input: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![millis]));
        let output = timestamp_ntz_to_timestamp(input, "America/New_York", None).unwrap();
        assert_eq!(
            as_primitive_array::<TimestampMillisecondType>(&output).value(0),
            micros_for("2024-01-15 15:30:00") / 1000
        );
    }

    #[test]
    fn test_array_with_timezone_requires_a_timezone() {
        let ltz = ts_with_tz("2024-01-15 10:30:00", "UTC");
        let err =
            array_with_timezone(ltz, String::new(), Some(&DataType::Utf8)).unwrap_err();
        assert!(err.to_string().contains("requires a timezone"), "{err}");

        let utc = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        assert!(
            array_with_timezone(ntz("2024-01-15 10:30:00"), String::new(), Some(&utc))
                .is_err()
        );
    }

    #[test]
    fn test_array_with_timezone_shifts_timestamp_for_string() {
        let input = ts_with_tz("2024-01-15 15:30:00", "UTC");
        let output = array_with_timezone(
            input,
            "America/New_York".to_string(),
            Some(&DataType::Utf8),
        )
        .unwrap();
        assert_eq!(first_value(&output), micros_for("2024-01-15 10:30:00"));
    }

    #[test]
    fn test_array_with_timezone_rejects_ntz_to_other_types() {
        let err = array_with_timezone(
            ntz("2024-01-15 10:30:00"),
            "UTC".to_string(),
            Some(&DataType::Int64),
        )
        .unwrap_err();
        assert!(err.to_string().contains("Cannot convert"), "{err}");
    }

    #[test]
    fn test_cast_timestamp_to_ntz_utc() {
        let result =
            cast_timestamp_to_ntz(ts_with_tz("2024-01-15 10:30:00", "UTC"), "UTC")
                .unwrap();
        assert_eq!(first_value(&result), micros_for("2024-01-15 10:30:00"));
        assert_eq!(
            result.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
    }

    #[test]
    fn test_cast_timestamp_to_ntz_offset_timezone() {
        // New York is UTC-5 in January.
        let result = cast_timestamp_to_ntz(
            ts_with_tz("2024-01-15 15:30:00", "UTC"),
            "America/New_York",
        )
        .unwrap();
        assert_eq!(first_value(&result), micros_for("2024-01-15 10:30:00"));
    }

    #[test]
    fn test_cast_timestamp_to_ntz_dst() {
        // New York is UTC-4 in July.
        let result = cast_timestamp_to_ntz(
            ts_with_tz("2024-07-04 16:30:00", "UTC"),
            "America/New_York",
        )
        .unwrap();
        assert_eq!(first_value(&result), micros_for("2024-07-04 12:30:00"));
    }

    #[test]
    fn test_out_of_range_timestamp_is_an_error() {
        let input: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![i64::MAX]).with_timezone("UTC"),
        );
        assert!(cast_timestamp_to_ntz(input, "Europe/Copenhagen").is_err());
    }
}
