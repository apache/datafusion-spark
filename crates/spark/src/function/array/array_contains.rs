// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::float_normalize::normalize_floats;
use crate::function::type_coercion::{
    is_orderable, spark_sql_type, struct_field_types_match, wider_type,
};
use arrow::array::{
    Array, AsArray, BooleanArray, BooleanBufferBuilder, GenericListArray, OffsetSizeTrait,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{DataType, Field};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions_nested::array_has::array_has_udf;
use std::sync::Arc;

/// Spark-compatible `array_contains` function.
///
/// Calls DataFusion's `array_has` and then applies Spark's null semantics:
/// - If the result from `array_has` is `true`, return `true`.
/// - If the result is `false` and the input array row contains any null elements,
///   return `null` (because the element might have been the null).
/// - If the result is `false` and the input array row has no null elements,
///   return `false`.
///
/// Spark compares floating point elements with `-0.0 == 0.0` and `NaN == NaN`,
/// while `array_has` compares them by their bits. Both arguments are therefore
/// normalized first, at any depth of list and struct elements.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayContains {
    signature: Signature,
}

impl Default for SparkArrayContains {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayContains {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayContains {
    fn name(&self) -> &str {
        "array_contains"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [array, element] = take_function_args(self.name(), arg_types)?;
        coerce_array_and_element(self.name(), array, element)
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        for arg in &mut args.args {
            *arg = normalize_floats(arg)?;
        }
        let haystack = args.args[0].clone();
        let array_has_result = array_has_udf().invoke_with_args(args)?;

        let result_array = array_has_result.to_array(1)?;
        let patched = apply_spark_null_semantics(result_array.as_boolean(), &haystack)?;
        Ok(ColumnarValue::Array(Arc::new(patched)))
    }
}

/// Casts the array element and the search element to their wider type, and rejects the
/// argument types that Spark's `ArrayContains.checkInputDataTypes` rejects.
fn coerce_array_and_element(
    name: &str,
    array: &DataType,
    element: &DataType,
) -> Result<Vec<DataType>> {
    if array.is_null() || element.is_null() {
        return null_type_err(name);
    }
    let (item, large) = match array {
        DataType::List(field) | DataType::FixedSizeList(field, _) => (field, false),
        DataType::LargeList(field) => (field, true),
        other => {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `{name}` \
                 requires the \"ARRAY\" type, however it has the type \"{}\". SQLSTATE: 42K09",
                spark_sql_type(other)
            );
        }
    };
    // Spark's `ArrayContains.inputTypes` calls `TypeCoercion.findWiderTypeWithoutStringPromotionForTwo`.
    let common = wider_type(item.data_type(), element, false)
        .filter(|_| struct_field_types_match(item.data_type(), element));
    let Some(common) = common else {
        return plan_err!(
            "[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES] Input to `{name}` should have \
             been \"ARRAY\" followed by a value with same element type, but it's \
             [\"{}\", \"{}\"]. SQLSTATE: 42K09",
            spark_sql_type(array),
            spark_sql_type(element)
        );
    };
    if common.is_null() {
        return null_type_err(name);
    }
    if !is_orderable(&common) {
        return plan_err!(
            "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{name}` does not support \
             ordering on type \"{}\". SQLSTATE: 42K09",
            spark_sql_type(&common)
        );
    }
    let item = Arc::new(Field::new(item.name(), common.clone(), true));
    let array = if large {
        DataType::LargeList(item)
    } else {
        DataType::List(item)
    };
    Ok(vec![array, common])
}

fn null_type_err<T>(name: &str) -> Result<T> {
    plan_err!(
        "[DATATYPE_MISMATCH.NULL_TYPE] Null typed values cannot be used as arguments of \
         `{name}`. SQLSTATE: 42K09"
    )
}

/// For each row where `array_has` returned `false`, set the output to null
/// if that row's input array contains any null elements.
fn apply_spark_null_semantics(
    result: &BooleanArray,
    haystack_arg: &ColumnarValue,
) -> Result<BooleanArray> {
    // happy path
    if haystack_arg.data_type() == DataType::Null || !result.has_false() {
        return Ok(result.clone());
    }

    let haystack = haystack_arg.to_array_of_size(result.len())?;

    let row_has_nulls = compute_row_has_nulls(&haystack)?;

    // A row keeps its validity when result is true OR the row has no nulls.
    let keep_mask = result.values() | &!&row_has_nulls;
    let new_validity = match result.nulls() {
        Some(n) => n.inner() & &keep_mask,
        None => keep_mask,
    };

    Ok(BooleanArray::new(
        result.values().clone(),
        Some(NullBuffer::new(new_validity)),
    ))
}

/// Returns a per-row bitmap where bit i is set if row i's list contains any null element.
fn compute_row_has_nulls(haystack: &dyn Array) -> Result<BooleanBuffer> {
    match haystack.data_type() {
        DataType::List(_) => generic_list_row_has_nulls(haystack.as_list::<i32>()),
        DataType::LargeList(_) => generic_list_row_has_nulls(haystack.as_list::<i64>()),
        DataType::FixedSizeList(_, _) => {
            let list = haystack.as_fixed_size_list();
            let buf = match list.values().nulls() {
                Some(nulls) => {
                    let validity = nulls.inner();
                    let vl = list.value_length() as usize;
                    let mut builder = BooleanBufferBuilder::new(list.len());
                    for i in 0..list.len() {
                        builder.append(validity.slice(i * vl, vl).count_set_bits() < vl);
                    }
                    builder.finish()
                }
                None => BooleanBuffer::new_unset(list.len()),
            };
            Ok(mask_with_list_nulls(buf, list.nulls()))
        }
        dt => exec_err!("compute_row_has_nulls: unsupported data type {dt}"),
    }
}

/// Computes per-row null presence for `List` and `LargeList` arrays.
fn generic_list_row_has_nulls<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
) -> Result<BooleanBuffer> {
    let buf = match list.values().nulls() {
        Some(nulls) => {
            let validity = nulls.inner();
            let offsets = list.offsets();
            let mut builder = BooleanBufferBuilder::new(list.len());
            for i in 0..list.len() {
                let s = offsets[i].as_usize();
                let len = offsets[i + 1].as_usize() - s;
                builder.append(validity.slice(s, len).count_set_bits() < len);
            }
            builder.finish()
        }
        None => BooleanBuffer::new_unset(list.len()),
    };
    Ok(mask_with_list_nulls(buf, list.nulls()))
}

/// Rows where the list itself is null should not be marked as "has nulls".
fn mask_with_list_nulls(
    buf: BooleanBuffer,
    list_nulls: Option<&NullBuffer>,
) -> BooleanBuffer {
    match list_nulls {
        Some(n) => &buf & n.inner(),
        None => buf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float64Array};
    use arrow::datatypes::{Field, Fields};
    use datafusion_common::ScalarValue;

    #[test]
    fn zero_width_fixed_size_list_keeps_its_row() {
        // One valid row of a zero-width list, without a null buffer to infer the length from.
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let haystack = FixedSizeListArray::try_new_with_length(
            Arc::clone(&field),
            0,
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            None,
            1,
        )
        .unwrap();
        assert_eq!(haystack.len(), 1);

        let result = SparkArrayContains::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(haystack)),
                    ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
                ],
                arg_fields: vec![],
                number_rows: 1,
                return_field: Arc::new(Field::new("result", DataType::Boolean, true)),
                config_options: Default::default(),
            })
            .unwrap()
            .to_array(1)
            .unwrap();
        assert_eq!(result.as_boolean(), &BooleanArray::from(vec![Some(false)]));
    }

    #[test]
    fn test_coerce_types_untyped_null_array() {
        let udf = SparkArrayContains::new();
        let err = udf
            .coerce_types(&[DataType::Null, DataType::Int32])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.NULL_TYPE] Null typed values cannot be used as arguments of `array_contains`. SQLSTATE: 42K09")
        );
    }

    #[test]
    fn test_coerce_types_untyped_null_element() {
        let udf = SparkArrayContains::new();
        let err = udf
            .coerce_types(&[
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                DataType::Null,
            ])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.NULL_TYPE] Null typed values cannot be used as arguments of `array_contains`. SQLSTATE: 42K09")
        );
    }

    #[test]
    fn test_coerce_types_both_untyped_null() {
        let udf = SparkArrayContains::new();
        let err = udf
            .coerce_types(&[DataType::Null, DataType::Null])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.NULL_TYPE] Null typed values cannot be used as arguments of `array_contains`. SQLSTATE: 42K09")
        );
    }

    #[test]
    fn test_coerce_types_same_type() {
        let udf = SparkArrayContains::new();
        let array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let coerced = udf
            .coerce_types(&[array_type.clone(), DataType::Int32])
            .unwrap();
        assert_eq!(coerced, vec![array_type, DataType::Int32]);
    }

    #[test]
    fn test_coerce_types_decimal_widening() {
        let udf = SparkArrayContains::new();
        let array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let coerced = udf
            .coerce_types(&[array_type, DataType::Decimal128(2, 1)])
            .unwrap();
        let expected_common = DataType::Decimal128(11, 1);
        let expected_array =
            DataType::List(Arc::new(Field::new("item", expected_common.clone(), true)));
        assert_eq!(coerced, vec![expected_array, expected_common]);
    }

    #[test]
    fn test_coerce_types_nested_decimal_widening() {
        let udf = SparkArrayContains::new();
        let inner_array =
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let outer_array = DataType::List(Arc::new(Field::new("item", inner_array, true)));
        let search_element = DataType::List(Arc::new(Field::new(
            "item",
            DataType::Decimal128(2, 1),
            true,
        )));
        let coerced = udf.coerce_types(&[outer_array, search_element]).unwrap();
        let expected_inner = DataType::List(Arc::new(Field::new(
            "item",
            DataType::Decimal128(21, 1),
            true,
        )));
        let expected_outer =
            DataType::List(Arc::new(Field::new("item", expected_inner.clone(), true)));
        assert_eq!(coerced, vec![expected_outer, expected_inner]);
    }

    #[test]
    fn test_coerce_types_widening() {
        let udf = SparkArrayContains::new();
        let array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let expected_array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        let coerced = udf.coerce_types(&[array_type, DataType::Int64]).unwrap();
        assert_eq!(coerced, vec![expected_array_type, DataType::Int64]);
    }

    #[test]
    fn test_coerce_types_type_mismatch() {
        let udf = SparkArrayContains::new();
        let array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let err = udf.coerce_types(&[array_type, DataType::Utf8]).unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES]")
        );
    }

    #[test]
    fn test_coerce_types_unexpected_input_type() {
        let udf = SparkArrayContains::new();
        let err = udf
            .coerce_types(&[DataType::Int32, DataType::Int32])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE]")
        );
    }

    #[test]
    fn test_coerce_types_invalid_ordering_type() {
        let udf = SparkArrayContains::new();
        let map_entries = DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));
        let map_type =
            DataType::Map(Arc::new(Field::new("entries", map_entries, false)), false);
        let array_type =
            DataType::List(Arc::new(Field::new("item", map_type.clone(), true)));
        let err = udf.coerce_types(&[array_type, map_type]).unwrap_err();
        assert!(
            err.to_string()
                .contains("[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE]")
        );
    }

    #[test]
    fn test_coerce_types_null_element_list_widening() {
        let udf = SparkArrayContains::new();
        let null_array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Null, true)));
        let expected_array_type =
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let coerced = udf
            .coerce_types(&[null_array_type, DataType::Int32])
            .unwrap();
        assert_eq!(coerced, vec![expected_array_type, DataType::Int32]);
    }

    #[test]
    fn test_coerce_types_wrong_arity() {
        let udf = SparkArrayContains::new();
        assert!(udf.coerce_types(&[DataType::Int32]).is_err());
        assert!(
            udf.coerce_types(&[DataType::Int32, DataType::Int32, DataType::Int32])
                .is_err()
        );
    }
}
