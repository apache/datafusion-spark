// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, StructArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `is_variant_null` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#is_variant_null>
#[user_doc(
    doc_section(label = "Variant Functions"),
    description = "Returns true only if the VARIANT value is a variant null, which is a JSON `null` stored in a \
        VARIANT, and false otherwise. A SQL NULL returns false, so the result is never NULL. \
        A VARIANT is the struct of the Arrow `arrow.parquet.variant` extension type, with a binary `metadata` field \
        and a binary `value` field, and the function reads the first byte of `value`. \
        An empty `value` raises `[MALFORMED_VARIANT]`. \
        Any other argument type is rejected at planning time, with no implicit casts, as in Spark. \
        Spark has had `is_variant_null` since 4.0.0 and has no VARIANT type before that.",
    syntax_example = "is_variant_null(variant)",
    sql_example = r#"```sql
> SELECT is_variant_null(named_struct('metadata', X'010000', 'value', X'00'));
true
> SELECT is_variant_null(named_struct('metadata', X'010000', 'value', X'0C0D'));
false
> SELECT is_variant_null(NULL);
false
```"#,
    argument(
        name = "variant",
        description = "VARIANT expression: a struct with a binary `metadata` field and a binary `value` field. \
            A NULL returns false."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkIsVariantNull {
    signature: Signature,
}

impl Default for SparkIsVariantNull {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkIsVariantNull {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkIsVariantNull {
    fn name(&self) -> &str {
        "is_variant_null"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        match arg_types {
            // Spark accepts a NULL literal as a SQL NULL VARIANT
            [data_type] if data_type == &DataType::Null || is_variant(data_type) => {
                Ok(arg_types.to_vec())
            }
            [other] => plan_err!(
                "is_variant_null expects a VARIANT argument, a struct with binary \
                 `metadata` and `value` fields, got {other}"
            ),
            _ => plan_err!(
                "is_variant_null expects exactly one argument, got {}",
                arg_types.len()
            ),
        }
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        // A SQL NULL input returns false, so the result is never NULL.
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, false)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [input] = take_function_args(self.name(), args.args)?;
        match input {
            ColumnarValue::Array(array) => {
                Ok(ColumnarValue::Array(Arc::new(is_variant_null(&array)?)))
            }
            ColumnarValue::Scalar(scalar) => {
                let result = is_variant_null(&scalar.to_array()?)?;
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(
                    result.value(0),
                ))))
            }
        }
    }
}

fn is_binary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    )
}

/// Whether `data_type` is the struct of a VARIANT: binary `metadata` and
/// `value` fields. A shredded VARIANT has more fields, which are not read.
fn is_variant(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    ["metadata", "value"].iter().all(|name| {
        fields
            .find(name)
            .is_some_and(|(_, field)| is_binary(field.data_type()))
    })
}

fn is_variant_null(array: &ArrayRef) -> Result<BooleanArray> {
    if array.data_type() == &DataType::Null {
        return Ok(BooleanArray::from(vec![false; array.len()]));
    }
    let Some(variants) = array.as_struct_opt() else {
        return exec_err!(
            "is_variant_null expects a VARIANT argument, got {}",
            array.data_type()
        );
    };
    let Some(values) = variants.column_by_name("value") else {
        return exec_err!("is_variant_null expects a VARIANT with a `value` field");
    };
    match values.data_type() {
        DataType::Binary => rows(variants, values.as_binary::<i32>().iter()),
        DataType::LargeBinary => rows(variants, values.as_binary::<i64>().iter()),
        DataType::BinaryView => rows(variants, values.as_binary_view().iter()),
        other => exec_err!("is_variant_null expects a binary `value` field, got {other}"),
    }
}

fn rows<'a>(
    variants: &StructArray,
    values: impl Iterator<Item = Option<&'a [u8]>>,
) -> Result<BooleanArray> {
    values
        .enumerate()
        .map(|(row, value)| match value {
            // A SQL NULL is not a variant null. A NULL `value` in a VARIANT that
            // is not NULL holds no value either, or its value is in a shredded
            // `typed_value` field.
            _ if variants.is_null(row) => Ok(false),
            None => Ok(false),
            Some([]) => exec_err!(
                "[MALFORMED_VARIANT] Variant binary is malformed. Please check the data source is valid. SQLSTATE: 22023"
            ),
            // The header byte of a variant null has basic type 0 (primitive) and
            // type id 0 (null).
            Some([header, ..]) => Ok(*header == 0),
        })
        .collect::<Result<Vec<_>>>()
        .map(BooleanArray::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, BinaryViewArray, LargeBinaryArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::Fields;

    const METADATA: &[u8] = &[0x01, 0x00, 0x00];

    fn struct_of(fields: Vec<(&str, ArrayRef)>, nulls: Option<Vec<bool>>) -> StructArray {
        let (names, arrays): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        let fields: Fields = names
            .iter()
            .zip(&arrays)
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect::<Vec<_>>()
            .into();
        StructArray::new(fields, arrays, nulls.map(NullBuffer::from))
    }

    /// A VARIANT array whose rows have the given `value` bytes.
    fn variants(values: &[Option<&[u8]>]) -> StructArray {
        let metadata: Vec<Option<&[u8]>> = vec![Some(METADATA); values.len()];
        struct_of(
            vec![
                ("metadata", Arc::new(BinaryArray::from(metadata))),
                ("value", Arc::new(BinaryArray::from(values.to_vec()))),
            ],
            None,
        )
    }

    fn evaluate(array: StructArray) -> Result<BooleanArray> {
        let result = SparkIsVariantNull::new().invoke_with_args(ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(array.clone()))],
            arg_fields: vec![Arc::new(Field::new("v", array.data_type().clone(), true))],
            number_rows: array.len(),
            return_field: Arc::new(Field::new("r", DataType::Boolean, false)),
            config_options: Arc::new(Default::default()),
        })?;
        match result {
            ColumnarValue::Array(array) => Ok(array.as_boolean().clone()),
            ColumnarValue::Scalar(_) => panic!("expected an array"),
        }
    }

    #[test]
    fn test_header_byte_decides() {
        let result = evaluate(variants(&[
            Some(&[0x00]),                         // null
            Some(&[0x04]),                         // true
            Some(&[0x0C, 0x0D]),                   // int8 13
            Some(&[0x11, b'n', b'u', b'l', b'l']), // the string "null"
            Some(&[0x03, 0x00, 0x00]),             // []
            Some(&[0x02, 0x00, 0x00]),             // {}
            Some(&[0x00, 0x01, 0x02]),             // only the first byte is read
        ]))
        .unwrap();
        assert_eq!(
            result,
            BooleanArray::from(vec![true, false, false, false, false, false, true])
        );
    }

    #[test]
    fn test_sql_null_is_false() {
        // A NULL struct row returns false, even with a `value` that would be a
        // variant null or that is empty.
        let array = struct_of(
            vec![
                (
                    "metadata",
                    Arc::new(BinaryArray::from(vec![Some(METADATA); 3])),
                ),
                (
                    "value",
                    Arc::new(BinaryArray::from(vec![
                        Some(&[0x00][..]),
                        Some(&[][..]),
                        Some(&[0x00][..]),
                    ])),
                ),
            ],
            Some(vec![false, false, true]),
        );
        assert_eq!(
            evaluate(array).unwrap(),
            BooleanArray::from(vec![false, false, true])
        );
    }

    #[test]
    fn test_null_literal_is_false() {
        let result = SparkIsVariantNull::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(ScalarValue::Null)],
                arg_fields: vec![Arc::new(Field::new("v", DataType::Null, true))],
                number_rows: 1,
                return_field: Arc::new(Field::new("r", DataType::Boolean, false)),
                config_options: Arc::new(Default::default()),
            })
            .unwrap();
        assert!(matches!(
            result,
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false)))
        ));
    }

    #[test]
    fn test_null_value_is_false() {
        let result = evaluate(variants(&[None, Some(&[0x00])])).unwrap();
        assert_eq!(result, BooleanArray::from(vec![false, true]));
    }

    #[test]
    fn test_sliced_struct_with_offset() {
        // Row 2 is a SQL NULL with an empty `value`, which must not be read.
        // Every row has a different answer than its neighbors, so a `value`
        // column and a validity buffer that are out of step would show.
        let metadata: Vec<Option<&[u8]>> = vec![Some(METADATA); 4];
        let values: Vec<Option<&[u8]>> =
            vec![Some(&[0x00]), Some(&[0x04]), Some(&[]), Some(&[0x00])];
        let array = struct_of(
            vec![
                ("metadata", Arc::new(BinaryArray::from(metadata))),
                ("value", Arc::new(BinaryArray::from(values))),
            ],
            Some(vec![true, true, false, true]),
        );
        assert_eq!(
            evaluate(array.clone()).unwrap(),
            BooleanArray::from(vec![true, false, false, true])
        );

        // The slice starts at a non-zero offset and keeps the NULL row.
        assert_eq!(
            evaluate(array.slice(1, 3)).unwrap(),
            BooleanArray::from(vec![false, false, true])
        );
        // The slice starts at the NULL row.
        assert_eq!(
            evaluate(array.slice(2, 2)).unwrap(),
            BooleanArray::from(vec![false, true])
        );
        // The slice ends before the NULL row.
        assert_eq!(
            evaluate(array.slice(0, 2)).unwrap(),
            BooleanArray::from(vec![true, false])
        );
    }

    #[test]
    fn test_empty_value_is_malformed() {
        let err = evaluate(variants(&[Some(&[0x00]), Some(&[])]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("[MALFORMED_VARIANT]"), "{err}");
        assert!(err.contains("SQLSTATE: 22023"), "{err}");
    }

    #[test]
    fn test_binary_layouts_and_field_order() {
        let values: Vec<Option<&[u8]>> = vec![Some(&[0x00]), Some(&[0x04])];
        let expected = BooleanArray::from(vec![true, false]);
        let metadata: Vec<Option<&[u8]>> = vec![Some(METADATA); 2];

        // `value` before `metadata`, as a large binary
        let large = struct_of(
            vec![
                ("value", Arc::new(LargeBinaryArray::from(values.clone()))),
                (
                    "metadata",
                    Arc::new(LargeBinaryArray::from(metadata.clone())),
                ),
            ],
            None,
        );
        assert_eq!(evaluate(large).unwrap(), expected);

        // a binary view, with a shredded `typed_value` field that is not read
        let shredded = struct_of(
            vec![
                ("metadata", Arc::new(BinaryViewArray::from(metadata))),
                ("value", Arc::new(BinaryViewArray::from(values))),
                (
                    "typed_value",
                    Arc::new(arrow::array::Int32Array::from(vec![None, Some(1)])),
                ),
            ],
            None,
        );
        assert_eq!(evaluate(shredded).unwrap(), expected);
    }

    #[test]
    fn test_scalar_input() {
        let array = variants(&[Some(&[0x00])]);
        let func = SparkIsVariantNull::new();
        let result = func
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                    array.clone(),
                )))],
                arg_fields: vec![Arc::new(Field::new(
                    "v",
                    array.data_type().clone(),
                    true,
                ))],
                number_rows: 1,
                return_field: Arc::new(Field::new("r", DataType::Boolean, false)),
                config_options: Arc::new(Default::default()),
            })
            .unwrap();
        assert!(matches!(
            result,
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))
        ));
    }

    #[test]
    fn test_result_is_not_nullable() {
        let func = SparkIsVariantNull::new();
        let variant = variants(&[None]);
        let arg_field = Arc::new(Field::new("v", variant.data_type().clone(), true));
        let field = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[arg_field],
                scalar_arguments: &[None],
            })
            .unwrap();
        assert_eq!(field.data_type(), &DataType::Boolean);
        assert!(!field.is_nullable());
    }

    #[test]
    fn test_only_variant_is_accepted() {
        let func = SparkIsVariantNull::new();
        let variant = variants(&[None]).data_type().clone();
        assert_eq!(
            func.coerce_types(std::slice::from_ref(&variant)).unwrap(),
            vec![variant.clone()]
        );

        let not_variants = [
            DataType::Binary,
            DataType::Utf8,
            DataType::Boolean,
            // a struct without a binary `value` field
            DataType::Struct(Fields::from(vec![
                Field::new("metadata", DataType::Binary, true),
                Field::new("value", DataType::Utf8, true),
            ])),
            DataType::Struct(Fields::from(vec![Field::new(
                "value",
                DataType::Binary,
                true,
            )])),
        ];
        for data_type in not_variants {
            let err = func
                .coerce_types(std::slice::from_ref(&data_type))
                .unwrap_err();
            assert!(
                err.to_string().contains("expects a VARIANT argument"),
                "{data_type}: {err}"
            );
        }
        // a NULL literal is a SQL NULL VARIANT, as in Spark
        assert_eq!(
            func.coerce_types(&[DataType::Null]).unwrap(),
            vec![DataType::Null]
        );
        assert!(func.coerce_types(&[]).is_err());
        assert!(func.coerce_types(&[variant.clone(), variant]).is_err());
    }
}
