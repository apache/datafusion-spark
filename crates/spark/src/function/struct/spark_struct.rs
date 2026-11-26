// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::StructArray;
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `struct` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#struct>
#[user_doc(
    doc_section(label = "Struct Functions"),
    description = "Returns a struct with the given expressions as its fields. \
        Unlike DataFusion's `struct`, the fields are named `col1`, `col2`, ... \
        as in Spark, rather than `c0`, `c1`, ...",
    syntax_example = "struct(expr, ...)",
    sql_example = r#"```sql
> SELECT struct(1, 2, 3);
{col1: 1, col2: 2, col3: 3}
> SELECT struct(1, 'a', NULL);
{col1: 1, col2: a, col3: NULL}
```"#,
    argument(
        name = "expr",
        description = "One or more expressions of any type, each becoming a field of the struct."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkStruct {
    signature: Signature,
}

impl Default for SparkStruct {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkStruct {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkStruct {
    fn name(&self) -> &str {
        "struct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return exec_err!("struct requires at least one argument, got 0 instead");
        }

        // Spark names the fields of a struct built from unnamed expressions
        // `col1`, `col2`, ... (1-based), whereas DataFusion uses `c0`, `c1`, ...
        let fields = arg_types
            .iter()
            .enumerate()
            .map(|(pos, dt)| Field::new(format!("col{}", pos + 1), dt.clone(), true))
            .collect::<Vec<Field>>();

        Ok(DataType::Struct(fields.into()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let arg_types = args
            .arg_fields
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        // The constructed struct row is never NULL: `invoke_with_args` builds
        // the `StructArray` without a null buffer.
        Ok(Field::new(self.name(), self.return_type(&arg_types)?, false).into())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let DataType::Struct(fields) = args.return_type() else {
            return internal_err!("incorrect struct return type");
        };

        if fields.len() != args.args.len() {
            return internal_err!(
                "struct return type has {} fields but {} arguments were given",
                fields.len(),
                args.args.len()
            );
        }

        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        Ok(ColumnarValue::Array(Arc::new(StructArray::new(
            fields.clone(),
            arrays,
            None,
        ))))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array, StringArray};
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;

    fn field(name: &str, data_type: DataType) -> FieldRef {
        Arc::new(Field::new(name, data_type, true))
    }

    #[test]
    fn test_struct_fields_are_named_col_n() -> Result<()> {
        let data_type = SparkStruct::new().return_type(&[
            DataType::Int32,
            DataType::Utf8,
            DataType::Boolean,
        ])?;

        let expected = DataType::Struct(
            vec![
                Field::new("col1", DataType::Int32, true),
                Field::new("col2", DataType::Utf8, true),
                Field::new("col3", DataType::Boolean, true),
            ]
            .into(),
        );
        assert_eq!(data_type, expected);
        Ok(())
    }

    #[test]
    fn test_struct_requires_an_argument() {
        let err = SparkStruct::new().return_type(&[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("struct requires at least one argument"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_struct_return_field_is_not_nullable() -> Result<()> {
        let return_field =
            SparkStruct::new().return_field_from_args(ReturnFieldArgs {
                arg_fields: &[field("a", DataType::Int32)],
                scalar_arguments: &[None],
            })?;

        assert!(!return_field.is_nullable());
        assert_eq!(
            return_field.data_type(),
            &DataType::Struct(vec![Field::new("col1", DataType::Int32, true)].into())
        );
        Ok(())
    }

    #[test]
    fn test_struct_invoke_with_arrays_and_scalars() -> Result<()> {
        let udf = SparkStruct::new();
        let args = vec![
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![
                Some(1),
                None,
                Some(3),
            ]))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("x".to_string()))),
        ];
        let arg_fields = vec![field("a", DataType::Int32), field("b", DataType::Utf8)];
        let return_field = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, None],
        })?;

        let result = udf.invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 3,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        })?;

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("a struct array");

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 0);
        assert_eq!(array.column_names(), vec!["col1", "col2"]);

        let col1 = array
            .column_by_name("col1")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
            .expect("col1 is Int32");
        assert_eq!(col1, &Int32Array::from(vec![Some(1), None, Some(3)]));

        let col2 = array
            .column_by_name("col2")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .expect("col2 is Utf8");
        assert_eq!(col2, &StringArray::from(vec!["x", "x", "x"]));
        Ok(())
    }
}
