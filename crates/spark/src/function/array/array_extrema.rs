// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible `array_max` and `array_min`.

use super::array_position::contains_map;
use super::float_normalize::normalize_array;
use arrow::array::{
    Array, ArrayRef, AsArray, GenericListArray, OffsetSizeTrait, UInt64Array,
    make_comparator, new_null_array,
};
use arrow::compute::{SortOptions, take};
use arrow::datatypes::{DataType, Field};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use std::cmp::Ordering;
use std::sync::Arc;

/// Spark-compatible `array_max` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_max>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the maximum element of the array, using Spark's ordering. NULL \
        elements are skipped, and an empty array or an array of only NULL elements returns \
        NULL. For floating point values, also inside nested arrays and structs, NaN is \
        greater than every other value, including infinity, and -0.0 is equal to 0.0. On a \
        tie the first of the tied elements is returned as it is, so the sign of a zero and the \
        bits of a NaN are kept. Strings and binary values compare bytewise. Arrays compare \
        element by element and then by length, and structs compare field by field; NULLs \
        nested inside them sort before other values. Map elements, at any depth, are an \
        error, as in Spark.",
    syntax_example = "array_max(array)",
    sql_example = r#"```sql
> SELECT array_max(array(1, 20, NULL, 3));
20
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayMax {
    signature: Signature,
}

impl Default for SparkArrayMax {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayMax {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayMax {
    fn name(&self) -> &str {
        "array_max"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        element_type(self.name(), arg_types)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_array(self.name(), arg_types)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke(self.name(), args, Ordering::Greater)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Spark-compatible `array_min` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_min>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the minimum element of the array, using Spark's ordering. NULL \
        elements are skipped, and an empty array or an array of only NULL elements returns \
        NULL. For floating point values, also inside nested arrays and structs, NaN is \
        greater than every other value, including infinity, and -0.0 is equal to 0.0. On a \
        tie the first of the tied elements is returned as it is, so the sign of a zero and the \
        bits of a NaN are kept. Strings and binary values compare bytewise. Arrays compare \
        element by element and then by length, and structs compare field by field; NULLs \
        nested inside them sort before other values. Map elements, at any depth, are an \
        error, as in Spark.",
    syntax_example = "array_min(array)",
    sql_example = r#"```sql
> SELECT array_min(array(1, 20, NULL, 3));
1
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayMin {
    signature: Signature,
}

impl Default for SparkArrayMin {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayMin {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayMin {
    fn name(&self) -> &str {
        "array_min"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        element_type(self.name(), arg_types)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_array(self.name(), arg_types)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke(self.name(), args, Ordering::Less)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// The element type of the array argument. A NULL-typed argument, which Spark accepts,
/// returns the NULL type.
fn element_type(name: &str, arg_types: &[DataType]) -> Result<DataType> {
    let [array] = take_function_args(name, arg_types)?;
    match array {
        DataType::Null => Ok(DataType::Null),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => Ok(field.data_type().clone()),
        other => unexpected_input_type(name, other),
    }
}

/// Accepts the argument types that Spark's `ArrayMax` and `ArrayMin` accept: an array of an
/// orderable type, or NULL. Fixed-size lists are cast to lists.
fn coerce_array(name: &str, arg_types: &[DataType]) -> Result<Vec<DataType>> {
    let [array] = take_function_args(name, arg_types)?;
    let coerced = match array {
        DataType::Null | DataType::List(_) | DataType::LargeList(_) => array.clone(),
        DataType::FixedSizeList(field, _) => DataType::List(Arc::new(Field::new(
            field.name(),
            field.data_type().clone(),
            true,
        ))),
        other => return unexpected_input_type(name, other),
    };
    let element = element_type(name, std::slice::from_ref(&coerced))?;
    if contains_map(&element) {
        return plan_err!(
            "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{name}` does not support \
             ordering on type {element}"
        );
    }
    Ok(vec![coerced])
}

fn unexpected_input_type<T>(name: &str, data_type: &DataType) -> Result<T> {
    plan_err!(
        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `{name}` \
         requires the \"ARRAY\" type, however it has the type {data_type}"
    )
}

fn invoke(
    name: &str,
    args: ScalarFunctionArgs,
    target: Ordering,
) -> Result<ColumnarValue> {
    let [arg] = take_function_args(name, &args.args)?;
    let element_type = args.return_field.data_type();
    let array = match arg {
        ColumnarValue::Array(array) => Arc::clone(array),
        ColumnarValue::Scalar(scalar) => scalar.to_array()?,
    };

    let result = match array.data_type() {
        DataType::Null => new_null_array(element_type, array.len()),
        // All the elements are NULL, so every result is NULL.
        _ if element_type.is_null() => new_null_array(element_type, array.len()),
        DataType::List(_) => extrema(array.as_list::<i32>(), target)?,
        DataType::LargeList(_) => extrema(array.as_list::<i64>(), target)?,
        other => return exec_err!("{name} does not support type {other}"),
    };

    if matches!(arg, ColumnarValue::Scalar(_)) {
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
            &result, 0,
        )?))
    } else {
        Ok(ColumnarValue::Array(result))
    }
}

/// Returns, for each list, its first non-NULL element that no other element orders
/// `target` of, or NULL when the list is NULL, empty, or holds only NULL elements.
fn extrema<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    target: Ordering,
) -> Result<ArrayRef> {
    let offsets = list.offsets();
    // Only the values that this (possibly sliced) list refers to are compared.
    let base = offsets[0].as_usize();
    let len = offsets[offsets.len() - 1].as_usize() - base;
    let values = list.values().slice(base, len);
    // Compare canonicalized floats, so that NaN is the greatest value and -0.0 equals 0.0 at
    // any depth, but return the original element.
    let normalized = normalize_array(&values)?;
    // Spark orders NULLs nested inside arrays and structs before other values.
    let cmp = make_comparator(
        normalized.as_ref(),
        normalized.as_ref(),
        SortOptions::default(),
    )?;

    let indices: UInt64Array = offsets
        .windows(2)
        .enumerate()
        .map(|(row, w)| {
            if list.is_null(row) {
                return None;
            }
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            (start..end)
                .filter(|&i| values.is_valid(i))
                .reduce(|best, i| if cmp(i, best) == target { i } else { best })
                .map(|i| i as u64)
        })
        .collect();

    Ok(take(values.as_ref(), &indices, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int32Array, LargeListArray, ListArray,
        StringArray, StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Float32Type, Float64Type, Int32Type};
    use datafusion_common::config::ConfigOptions;

    fn run(udf: &dyn ScalarUDFImpl, input: ColumnarValue) -> ColumnarValue {
        let input_type = input.data_type();
        let return_type = udf.return_type(std::slice::from_ref(&input_type)).unwrap();
        let number_rows = match &input {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        udf.invoke_with_args(ScalarFunctionArgs {
            args: vec![input],
            arg_fields: vec![Arc::new(Field::new("input", input_type, true))],
            number_rows,
            return_field: Arc::new(Field::new("result", return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
        .unwrap()
    }

    fn max(input: &dyn Array) -> ArrayRef {
        run_array(&SparkArrayMax::new(), input)
    }

    fn min(input: &dyn Array) -> ArrayRef {
        run_array(&SparkArrayMin::new(), input)
    }

    fn run_array(udf: &dyn ScalarUDFImpl, input: &dyn Array) -> ArrayRef {
        let input = input.slice(0, input.len());
        match run(udf, ColumnarValue::Array(input)) {
            ColumnarValue::Array(result) => result,
            ColumnarValue::Scalar(_) => panic!("array input must produce array output"),
        }
    }

    fn list(values: ArrayRef, offsets: &[i32]) -> ListArray {
        ListArray::new(
            Arc::new(Field::new_list_field(values.data_type().clone(), true)),
            OffsetBuffer::new(offsets.to_vec().into()),
            values,
            None,
        )
    }

    fn f64_bits(array: &ArrayRef) -> Vec<Option<u64>> {
        array
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.map(f64::to_bits))
            .collect()
    }

    #[test]
    fn integers_skip_nulls() {
        let input = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(20), None, Some(3)]),
            Some(vec![]),
            Some(vec![None, None]),
            None,
            Some(vec![Some(-5)]),
        ]);
        assert_eq!(
            max(&input).as_ref(),
            &Int32Array::from(vec![Some(20), None, None, None, Some(-5)])
        );
        assert_eq!(
            min(&input).as_ref(),
            &Int32Array::from(vec![Some(1), None, None, None, Some(-5)])
        );
    }

    macro_rules! float_tests {
        ($name:ident, $arrow_type:ty, $array:ty, $native:ident, $positive:expr, $negative:expr) => {
            #[test]
            fn $name() {
                let positive = $native::from_bits($positive);
                let negative = $native::from_bits($negative);
                let mut long = vec![Some(1.0); 67];
                long[7] = Some(negative);
                long[8] = Some(positive);
                // (input, min, max)
                let cases = [
                    (vec![Some(0.0), Some(-0.0)], Some(0.0), Some(0.0)),
                    (vec![None, Some(-0.0), Some(0.0)], Some(-0.0), Some(-0.0)),
                    (
                        vec![Some(positive), Some(negative)],
                        Some(positive),
                        Some(positive),
                    ),
                    (
                        vec![Some(negative), Some(positive)],
                        Some(negative),
                        Some(negative),
                    ),
                    (
                        vec![
                            Some(negative),
                            Some($native::INFINITY),
                            Some($native::NEG_INFINITY),
                        ],
                        Some($native::NEG_INFINITY),
                        Some(negative),
                    ),
                    (vec![Some(3.0), None, Some(-2.0)], Some(-2.0), Some(3.0)),
                    (long, Some(1.0), Some(negative)),
                ];
                for (row, expected_min, expected_max) in cases {
                    let input =
                        ListArray::from_iter_primitive::<$arrow_type, _, _>([Some(row)]);
                    for (result, expected) in
                        [(min(&input), expected_min), (max(&input), expected_max)]
                    {
                        let result = result.as_any().downcast_ref::<$array>().unwrap();
                        assert_eq!(
                            result.iter().next().unwrap().map($native::to_bits),
                            expected.map($native::to_bits),
                        );
                    }
                }
            }
        };
    }

    float_tests!(
        float32_nan_and_signed_zero,
        Float32Type,
        Float32Array,
        f32,
        0x7fc0_0001,
        0xffc0_0002
    );
    float_tests!(
        float64_nan_and_signed_zero,
        Float64Type,
        Float64Array,
        f64,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_0002
    );

    #[test]
    fn scalar_and_sliced_input() {
        let input = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(99.0)]),
            Some(vec![Some(-0.0), None, Some(0.0)]),
            None,
        ])
        .slice(1, 2);
        for udf in [
            &SparkArrayMax::new() as &dyn ScalarUDFImpl,
            &SparkArrayMin::new(),
        ] {
            assert_eq!(
                f64_bits(&run_array(udf, &input)),
                vec![Some((-0.0f64).to_bits()), None]
            );
            let scalar = ScalarValue::try_from_array(&input, 0).unwrap();
            let ColumnarValue::Scalar(result) = run(udf, ColumnarValue::Scalar(scalar))
            else {
                panic!("scalar input must produce scalar output")
            };
            assert_eq!(
                f64_bits(&result.to_array().unwrap()),
                vec![Some((-0.0f64).to_bits())]
            );
        }
    }

    #[test]
    fn strings_and_binary_compare_bytewise() {
        let values: ArrayRef =
            Arc::new(StringArray::from(vec!["a", "b", "B", "", "é", "z"]));
        let input = list(values, &[0, 4, 6]);
        assert_eq!(max(&input).as_ref(), &StringArray::from(vec!["b", "é"]));
        assert_eq!(min(&input).as_ref(), &StringArray::from(vec!["", "z"]));

        let values: ArrayRef = Arc::new(BinaryArray::from(vec![
            &[0x01][..],
            &[0xff][..],
            &[0x01, 0x00][..],
        ]));
        let input = list(values, &[0, 3]);
        assert_eq!(max(&input).as_ref(), &BinaryArray::from(vec![&[0xff][..]]));
        assert_eq!(min(&input).as_ref(), &BinaryArray::from(vec![&[0x01][..]]));
    }

    #[test]
    fn large_list() {
        let input = LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(3), None, Some(7)]),
            None,
        ]);
        assert_eq!(max(&input).as_ref(), &Int32Array::from(vec![Some(7), None]));
        assert_eq!(min(&input).as_ref(), &Int32Array::from(vec![Some(3), None]));
    }

    #[test]
    fn nested_lists_order_lexicographically_with_nulls_first() {
        let children = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
            Some(vec![Some(2.0)]),
            Some(vec![]),
            Some(vec![Some(1.0), None]),
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(f64::NAN)]),
            Some(vec![Some(f64::INFINITY)]),
            None,
            Some(vec![Some(-0.0)]),
            Some(vec![Some(0.0)]),
        ]);
        let input = list(Arc::new(children.clone()), &[0, 4, 6, 9, 11]);
        let expected = |winners: [u64; 4]| -> ArrayRef {
            take(&children, &UInt64Array::from(winners.to_vec()), None).unwrap()
        };
        let result = max(&input);
        assert_eq!(&result, &expected([2, 5, 6, 9]));
        let result = min(&input);
        assert_eq!(&result, &expected([3, 4, 7, 9]));
        // The first tied winner keeps the sign of its zero.
        let last = result.as_list::<i32>().value(3);
        assert!(
            last.as_primitive::<Float64Type>()
                .value(0)
                .is_sign_negative()
        );
    }

    #[test]
    fn structs_compare_field_by_field() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![
                Some(-0.0),
                Some(0.0),
                Some(1.0),
                None,
                Some(5.0),
                Some(5.0),
            ])),
            Arc::new(Int32Array::from(vec![2, 1, 1, 100, 1, 2])),
        ];
        let fields = vec![
            Arc::new(Field::new("a", DataType::Float64, true)),
            Arc::new(Field::new("b", DataType::Int32, true)),
        ];
        let children = StructArray::new(
            fields.into(),
            columns,
            Some(NullBuffer::from(vec![true, true, true, true, false, true])),
        );
        let input = list(Arc::new(children.clone()), &[0, 2, 4, 6]);
        let expected = |winners: [u64; 3]| {
            take(&children, &UInt64Array::from(winners.to_vec()), None).unwrap()
        };
        // -0.0 equals 0.0, so the second field decides. A NULL field sorts first, and a NULL
        // struct is skipped.
        assert_eq!(&max(&input), &expected([0, 2, 5]));
        assert_eq!(&min(&input), &expected([1, 3, 5]));
    }

    #[test]
    fn null_input_and_null_elements() {
        let input = new_null_array(&DataType::Null, 2);
        assert_eq!(max(&input).as_ref(), &new_null_array(&DataType::Null, 2));
        let input = list(new_null_array(&DataType::Null, 3), &[0, 0, 3]);
        assert_eq!(min(&input).as_ref(), &new_null_array(&DataType::Null, 2));
    }

    #[test]
    fn rejects_unsupported_types() {
        let udf = SparkArrayMax::new();
        let err = udf.coerce_types(&[DataType::Int32]).unwrap_err();
        assert!(err.to_string().contains("UNEXPECTED_INPUT_TYPE"), "{err}");
        let map = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Int32, false),
                        Field::new("value", DataType::Int32, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let array = DataType::new_list(DataType::new_list(map, true), true);
        let err = udf.coerce_types(&[array]).unwrap_err();
        assert!(err.to_string().contains("INVALID_ORDERING_TYPE"), "{err}");
        let fixed = DataType::new_fixed_size_list(DataType::Int32, 2, true);
        assert_eq!(
            udf.coerce_types(&[fixed]).unwrap(),
            vec![DataType::new_list(DataType::Int32, true)]
        );
    }
}
