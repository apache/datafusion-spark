// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, LargeStringArray, StringArray, StringViewArray};
use arrow::datatypes::{DataType, Field, FieldRef, Int32Type};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

/// Spark-compatible `substring_index` with UTF8_BINARY matching.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#substring_index>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the part of `str` before the `count`-th delimiter from the left \
        for positive counts, or after the |count|-th delimiter from the right for negative \
        counts. Unlike DataFusion's `substr_index`, delimiter matches may overlap: \
        `substring_index('||||||', '|||', 3)` returns `||`. Matching is case-sensitive \
        and byte-exact (UTF8_BINARY); other Spark collations are not supported. Fewer \
        matches returns the whole string. Zero count or an empty delimiter returns an \
        empty string. NULL in any argument returns NULL. Count uses Spark's INT type; \
        INT_MIN with a nonempty delimiter raises an error, as in Spark. Implicit cast \
        behavior still depends on the shared CAST work in \
        https://github.com/apache/datafusion-spark/issues/88.",
    syntax_example = "substring_index(str, delim, count)",
    sql_example = r#"```sql
> SELECT substring_index('www.apache.org', '.', 2);
www.apache
> SELECT substring_index('www.apache.org', '.', -2);
apache.org
> SELECT substring_index('||||||', '|||', 3);
||
```"#,
    argument(name = "str", description = "String to search. NULL returns NULL."),
    argument(
        name = "delim",
        description = "Case-sensitive delimiter. Matches may overlap; an empty delimiter returns an empty string."
    ),
    argument(
        name = "count",
        description = "Occurrence count, implicitly cast to INT. Positive searches from the left, negative from the right, and zero returns an empty string."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkSubstringIndex {
    input_types: SparkInputTypes,
}

impl Default for SparkSubstringIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkSubstringIndex {
    pub fn new() -> Self {
        Self {
            input_types: SparkInputTypes::new(
                vec![
                    SparkInputType::String,
                    SparkInputType::String,
                    SparkInputType::Integer,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkSubstringIndex {
    fn name(&self) -> &str {
        "substring_index"
    }
    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let mut types = self.input_types.coerce(self.name(), arg_types)?;
        // Align the delimiter's Arrow layout while preserving the source layout
        // as the return type, just as Spark preserves the source STRING type.
        types[1] = types[0].clone();
        Ok(types)
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        let input_types = self.input_types.with_config(config)?;
        Some(ScalarUDF::from(Self { input_types }))
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [source, _, _] = take_function_args(self.name(), args.arg_fields)?;
        Ok(Arc::new(Field::new(
            self.name(),
            source.data_type().clone(),
            args.arg_fields.iter().any(|f| f.is_nullable()),
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(substring_index, vec![])(&args.args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn substring_index(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [source, delimiter, count] = take_function_args("substring_index", args)?;
    let counts = count.as_primitive::<Int32Type>();
    macro_rules! evaluate {
        ($source:expr, $delimiter:expr, $output:ty) => {{
            let output = $source
                .iter()
                .zip($delimiter.iter())
                .zip(counts.iter())
                .map(|((s, d), c)| match (s, d, c) {
                    (Some(s), Some(d), Some(c)) => {
                        substring_index_value(s, d, c).map(Some)
                    }
                    _ => Ok(None),
                })
                .collect::<Result<$output>>()?;
            Ok(Arc::new(output) as ArrayRef)
        }};
    }
    match source.data_type() {
        DataType::Utf8 => evaluate!(
            source.as_string::<i32>(),
            delimiter.as_string::<i32>(),
            StringArray
        ),
        DataType::LargeUtf8 => evaluate!(
            source.as_string::<i64>(),
            delimiter.as_string::<i64>(),
            LargeStringArray
        ),
        DataType::Utf8View => evaluate!(
            source.as_string_view(),
            delimiter.as_string_view(),
            StringViewArray
        ),
        other => internal_err!("Unsupported string layout {other} for substring_index"),
    }
}

fn substring_index_value<'a>(
    source: &'a str,
    delimiter: &str,
    count: i32,
) -> Result<&'a str> {
    if delimiter.is_empty() || count == 0 {
        return Ok("");
    }
    // UTF8String.subStringIndex negates count as a Java int. INT_MIN stays
    // negative, skips its search loop and attempts an array of size -1.
    if count == i32::MIN {
        return exec_err!("NegativeArraySizeException: -1");
    }
    let mut matches = source
        .as_bytes()
        .windows(delimiter.len())
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == delimiter.as_bytes()).then_some(offset));
    let index = if count > 0 {
        matches.nth(count as usize - 1)
    } else {
        matches.rev().nth((-count) as usize - 1)
    };
    // A valid UTF-8 delimiter can only match at character boundaries, including
    // overlapping matches. Searching bytes avoids slicing inside a character.
    Ok(match index {
        Some(offset) if count > 0 => &source[..offset],
        Some(offset) => &source[offset + delimiter.len()..],
        None => source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_field_preserves_layout_and_nullability() -> Result<()> {
        let udf = SparkSubstringIndex::new();
        for layout in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            for mask in 0..8 {
                let fields = vec![
                    Arc::new(Field::new("str", layout.clone(), mask & 1 != 0)),
                    Arc::new(Field::new("delim", layout.clone(), mask & 2 != 0)),
                    Arc::new(Field::new("count", DataType::Int32, mask & 4 != 0)),
                ];
                let field = udf.return_field_from_args(ReturnFieldArgs {
                    arg_fields: &fields,
                    scalar_arguments: &[None, None, None],
                })?;
                assert_eq!(field.data_type(), &layout);
                assert_eq!(field.is_nullable(), mask != 0);
            }
        }
        Ok(())
    }
}
