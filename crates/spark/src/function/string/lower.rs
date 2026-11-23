// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::string::lower::LowerFunc;
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use super::make_valid_utf8::spark_make_valid_utf8_inner;
use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

/// Spark-compatible `lower` / `lcase` with UTF8_BINARY mapping.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#lower>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Converts a string to lowercase using full, locale-independent Unicode \
        mapping, matching UTF8_BINARY with Spark 4.2's default case mappings. Greek \
        capital sigma is mapped using its word context, and one character may expand \
        (for example, İ becomes i plus a combining dot). Unlike DataFusion's `lower`, \
        Spark's `lcase` alias and implicit atomic-to-string input coercion are provided. \
        Binary input is decoded with invalid UTF-8 sequences replaced by U+FFFD before \
        lowercasing. NULL returns NULL; empty input returns an empty string. The source \
        string layout and nullability are preserved. ANSI mode does not change the \
        string mapping. Locale-specific collations and JVM default-locale mappings are \
        not supported. Remaining implicit CAST formatting differences are tracked by \
        https://github.com/apache/datafusion-spark/issues/88.",
    syntax_example = "lower(str)",
    alternative_syntax = "lcase(str)",
    sql_example = r#"```sql
> SELECT lower('SparkSql');
sparksql
> SELECT lcase('ΘΑΛΑΣΣΙΝΟΣ');
θαλασσινος
> SELECT lower('İ');
i̇
```"#,
    argument(
        name = "str",
        description = "String to lowercase, or an atomic expression implicitly cast to string. Binary input is decoded as UTF-8 with replacement. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLower {
    input_types: SparkInputTypes,
    inner: LowerFunc,
    aliases: Vec<String>,
}

impl Default for SparkLower {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkLower {
    pub fn new() -> Self {
        Self {
            input_types: SparkInputTypes::new(
                vec![SparkInputType::String],
                Volatility::Immutable,
            ),
            inner: LowerFunc::new(),
            aliases: vec!["lcase".to_string()],
        }
    }
}

impl ScalarUDFImpl for SparkLower {
    fn name(&self) -> &str {
        "lower"
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
    }
    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let targets = self.input_types.coerce(self.name(), arg_types)?;
        let mut source = &arg_types[0];
        while let DataType::Dictionary(_, value) = source {
            source = value;
        }
        // Keep binary bytes until execution: an Arrow STRING cast rejects invalid
        // UTF-8, whereas Spark's default lower path decodes with replacement.
        match source {
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                Ok(vec![source.clone()])
            }
            DataType::FixedSizeBinary(_) => Ok(vec![DataType::Binary]),
            _ => Ok(targets),
        }
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let input_types = self.input_types.with_config(config)?;
        Some(ScalarUDF::from(Self {
            input_types,
            ..Self::new()
        }))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = take_function_args(self.name(), args.arg_fields)?;
        let output_type = match input.data_type() {
            DataType::Binary | DataType::BinaryView => DataType::Utf8,
            DataType::LargeBinary => DataType::LargeUtf8,
            other => other.clone(),
        };
        Ok(Arc::new(Field::new(
            self.name(),
            output_type,
            input.is_nullable(),
        )))
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [input] = take_function_args(self.name(), &args.args)?;
        if input.data_type().is_binary() {
            let decoded =
                make_scalar_function(spark_make_valid_utf8_inner, vec![])(&args.args)?;
            let field = &args.arg_fields[0];
            args.arg_fields = vec![Arc::new(Field::new(
                field.name(),
                decoded.data_type(),
                field.is_nullable(),
            ))];
            args.args = vec![decoded];
        }
        // Reuse DataFusion's full Unicode/Final_Sigma mapping and ASCII fast paths.
        self.inner.invoke_with_args(args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_field_preserves_layout_and_nullability() -> Result<()> {
        let udf = SparkLower::new();
        for (input, output) in [
            (DataType::Utf8, DataType::Utf8),
            (DataType::LargeUtf8, DataType::LargeUtf8),
            (DataType::Utf8View, DataType::Utf8View),
            (DataType::Binary, DataType::Utf8),
            (DataType::LargeBinary, DataType::LargeUtf8),
            (DataType::BinaryView, DataType::Utf8),
        ] {
            for nullable in [false, true] {
                let field = udf.return_field_from_args(ReturnFieldArgs {
                    arg_fields: &[Arc::new(Field::new("input", input.clone(), nullable))],
                    scalar_arguments: &[None],
                })?;
                assert_eq!(field.data_type(), &output);
                assert_eq!(field.is_nullable(), nullable);
            }
        }
        Ok(())
    }
}
