// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::unicode::find_in_set as core_find_in_set;
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

/// Spark-compatible `find_in_set` expression.
/// <https://spark.apache.org/docs/latest/api/sql/index.html#find_in_set>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the 1-based index of the first matching element as an INT. \
        Uses the UTF8_BINARY collation: the search is case-sensitive and elements are split on commas without trimming. \
        Empty elements count, including the single empty element in an empty list. \
        Returns 0 when the string contains a comma or has no match and NULL if either argument is NULL.",
    syntax_example = "find_in_set(str, strlist)",
    sql_example = r#"```sql
> SELECT find_in_set('b', 'a,b,c');
2
> SELECT find_in_set('', 'a,,b');
2
```"#,
    argument(name = "str", description = "String expression to match."),
    argument(name = "strlist", description = "Comma-separated list of strings.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkFindInSet {
    input_types: SparkInputTypes,
}

impl Default for SparkFindInSet {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkFindInSet {
    pub fn new() -> Self {
        let input_types = SparkInputTypes::new(
            vec![SparkInputType::String, SparkInputType::String],
            Volatility::Immutable,
        );
        Self { input_types }
    }
}

impl ScalarUDFImpl for SparkFindInSet {
    fn name(&self) -> &str {
        "find_in_set"
    }

    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let types = self.input_types.coerce(self.name(), arg_types)?;
        // The core kernel expects both string arrays to use the same layout.
        let target = if types.contains(&DataType::LargeUtf8) {
            DataType::LargeUtf8
        } else if types.contains(&DataType::Utf8View) {
            DataType::Utf8View
        } else {
            DataType::Utf8
        };
        Ok(vec![target; types.len()])
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let input_types = self.input_types.with_config(config)?;
        Some(ScalarUDF::from(Self { input_types }))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|field| field.is_nullable());
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, nullable)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let core = core_find_in_set();
        let [needle, list] = take_function_args(self.name(), &args.args)?;
        let return_type = core.return_type(&[needle.data_type(), list.data_type()])?;
        if return_type == DataType::Int32 {
            return core.invoke_with_args(args);
        }

        // DataFusion returns Int64 for LargeUtf8. Spark always returns INT.
        args.return_field = Arc::new(
            args.return_field
                .as_ref()
                .clone()
                .with_data_type(return_type),
        );
        core.invoke_with_args(args)?.cast_to(&DataType::Int32, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_field_matches_spark_nullability() -> Result<()> {
        let function = SparkFindInSet::new();
        for (needle_nullable, list_nullable) in
            [(false, false), (true, false), (false, true)]
        {
            let fields = [
                Arc::new(Field::new("str", DataType::LargeUtf8, needle_nullable)),
                Arc::new(Field::new("strlist", DataType::LargeUtf8, list_nullable)),
            ];
            let result = function.return_field_from_args(ReturnFieldArgs {
                arg_fields: &fields,
                scalar_arguments: &[None, None],
            })?;
            assert_eq!(result.data_type(), &DataType::Int32);
            assert_eq!(result.is_nullable(), needle_nullable || list_nullable);
        }
        Ok(())
    }
}
