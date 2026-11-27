// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::logical_date;
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, internal_err, plan_err,
    types::logical_string, utils::take_function_args,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Field names Spark accepts, one group per field, from `DatePart.parseExtractField`.
const SPARK_FIELDS: &[&str] = &[
    "year y years yr yrs",
    "yearofweek",
    "quarter qtr",
    "month mon mons months",
    "week w weeks",
    "day d days",
    "dayofweek dow",
    "dayofweek_iso dow_iso",
    "doy",
    "hour h hours hr hrs",
    "minute m min mins minutes",
    "second s sec seconds secs",
];

/// Spark-compatible `date_part` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_part>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Extracts a field from a date or timestamp, as an INT. \
        Unlike DataFusion's `date_part`, it accepts Spark's field names and abbreviations (case-insensitive), \
        and `DAYOFWEEK`/`DOW` counts from 1 (Sunday) to 7 (Saturday) where DataFusion's `dow` counts from 0 (Sunday). \
        `DAYOFWEEK_ISO`/`DOW_ISO` counts from 1 (Monday) to 7 (Sunday), and `YEAROFWEEK` is the ISO week-numbering year. \
        A field name Spark does not accept (such as `QUARTERS` or `DAYOFYEAR`) is an error. \
        A NULL field or source returns NULL. \
        `SECOND` returns whole seconds; Spark returns DECIMAL(8, 6) including the fraction of a second.",
    syntax_example = "date_part(field, source)",
    alternative_syntax = "datepart(field, source)",
    sql_example = r#"```sql
> SELECT date_part('YEAR', DATE '2000-01-01');
2000
> SELECT date_part('DOW', DATE '2000-01-01');
7
> SELECT date_part('MINS', TIMESTAMP '2000-01-01 12:30:45');
30
> SELECT date_part('year', NULL::TIMESTAMP);
NULL
```"#,
    argument(
        name = "field",
        description = "String literal naming the field to extract: `YEAR` (`Y`, `YEARS`, `YR`, `YRS`), `YEAROFWEEK`, \
            `QUARTER` (`QTR`), `MONTH` (`MON`, `MONS`, `MONTHS`), `WEEK` (`W`, `WEEKS`), `DAY` (`D`, `DAYS`), \
            `DAYOFWEEK` (`DOW`), `DAYOFWEEK_ISO` (`DOW_ISO`), `DOY`, `HOUR` (`H`, `HOURS`, `HR`, `HRS`), \
            `MINUTE` (`M`, `MIN`, `MINS`, `MINUTES`) or `SECOND` (`S`, `SEC`, `SECONDS`, `SECS`). NULL returns NULL."
    ),
    argument(
        name = "source",
        description = "DATE or TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDatePart {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkDatePart {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDatePart {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                        Coercion::new_exact(TypeSignatureClass::Timestamp),
                    ]),
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                        Coercion::new_exact(TypeSignatureClass::Native(logical_date())),
                    ]),
                ],
                Volatility::Immutable,
            ),
            aliases: vec![String::from("datepart")],
        }
    }
}

impl ScalarUDFImpl for SparkDatePart {
    fn name(&self) -> &str {
        "date_part"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("Use return_field_from_args in this case instead.")
    }

    // Nullable, like the integer parts of `date_part`, which this simplifies to,
    // because they can be NULL for a non-null input
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Reject an invalid literal field while the source expression's name is
        // still available. Physical argument fields no longer retain that name
        // for literals, and simplifying the source may fold a cast or function.
        if let Some(Some(field)) = args.scalar_arguments.first()
            && let Err(error @ FieldError::Invalid(_)) = check_field(Some(field))
        {
            let source =
                source_argument(args.scalar_arguments[1], args.arg_fields[1].name());
            return plan_err!("{}", error.message(&source));
        }
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let field = match &args.args[0] {
            ColumnarValue::Scalar(field) => Some(field),
            ColumnarValue::Array(_) => None,
        };
        let checked = check_field(field).map_err(|error| {
            // Direct physical callers have evaluated arguments rather than the
            // original logical expression. Report the scalar value or actual
            // argument field, without inventing an expression from array rows.
            let value = match &args.args[1] {
                ColumnarValue::Scalar(value) => Some(value),
                ColumnarValue::Array(_) => None,
            };
            let source = source_argument(value, args.arg_fields[1].name());
            exec_datafusion_err!("{}", error.message(&source))
        })?;
        match checked {
            ExtractField::Part(part) => {
                args.args[0] = ColumnarValue::Scalar(ScalarValue::new_utf8(part));
                datafusion_functions::datetime::date_part().invoke_with_args(args)
            }
            // A NULL field returns NULL, like Spark
            ExtractField::Null => Ok(ColumnarValue::Scalar(ScalarValue::Int32(None))),
        }
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [part_expr, date_expr] = take_function_args(self.name(), args)?;

        let part = match check_field(part_expr.as_literal()) {
            Ok(ExtractField::Part(part)) => part,
            // A NULL field returns NULL, like Spark
            Ok(ExtractField::Null) => {
                return Ok(ExprSimplifyResult::Simplified(Expr::Literal(
                    ScalarValue::Int32(None),
                    None,
                )));
            }
            Err(FieldError::Invalid(name)) => {
                let source =
                    source_argument(date_expr.as_literal(), &date_expr.to_string());
                let message = FieldError::Invalid(name).message(&source);
                return plan_err!("{message}");
            }
            // A parameter is not known at planning time, so `invoke_with_args` checks it
            Err(FieldError::NotConstant) if matches!(part_expr, Expr::Placeholder(_)) => {
                return Ok(ExprSimplifyResult::Original(vec![part_expr, date_expr]));
            }
            // Any other non-literal field, such as a column, is rejected like Spark does
            Err(FieldError::NotConstant) => return plan_err!("{NOT_CONSTANT}"),
        };

        let part_expr = Expr::Literal(ScalarValue::new_utf8(part), None);

        let date_part_expr = Expr::ScalarFunction(ScalarFunction::new_udf(
            datafusion_functions::datetime::date_part(),
            vec![part_expr, date_expr],
        ));

        Ok(ExprSimplifyResult::Simplified(date_part_expr))
    }
}

/// A checked `field` argument
enum ExtractField {
    /// The name of the field in DataFusion's `date_part`
    Part(&'static str),
    Null,
}

/// Why a `field` argument is rejected
enum FieldError<'a> {
    Invalid(&'a str),
    NotConstant,
}

impl FieldError<'_> {
    fn message(&self, source: &str) -> String {
        match self {
            FieldError::Invalid(name) => {
                format!(
                    "[INVALID_EXTRACT_FIELD] Cannot extract `{name}` from \"{source}\". SQLSTATE: 42601"
                )
            }
            FieldError::NotConstant => NOT_CONSTANT.to_string(),
        }
    }
}

const NOT_CONSTANT: &str = "First argument of `DATE_PART` must be a constant string";

fn source_argument(value: Option<&ScalarValue>, expression: &str) -> String {
    match value {
        Some(value) if value.is_null() => "NULL".to_string(),
        Some(value @ ScalarValue::Date32(_)) => format!("DATE '{value}'"),
        Some(
            value @ (ScalarValue::TimestampSecond(_, None)
            | ScalarValue::TimestampMillisecond(_, None)
            | ScalarValue::TimestampMicrosecond(_, None)
            | ScalarValue::TimestampNanosecond(_, None)),
        ) => match value.cast_to(&DataType::Utf8) {
            Ok(ScalarValue::Utf8(Some(value))) => {
                format!("TIMESTAMP '{}'", value.replace('T', " "))
            }
            _ => format!("{value:?}"),
        },
        Some(value) => format!("{value:?}"),
        None => expression.to_string(),
    }
}

/// Checks the `field` argument against Spark's field names. The caller formats the
/// error with the available source expression and picks the error kind.
fn check_field(
    field: Option<&ScalarValue>,
) -> std::result::Result<ExtractField, FieldError<'_>> {
    let Some(field) = field else {
        return Err(FieldError::NotConstant);
    };
    // `try_as_str` also handles dictionary and run-end encoded strings
    match field.try_as_str() {
        Some(Some(name)) => {
            let lower = name.to_lowercase();
            let Some(part) = SPARK_FIELDS
                .iter()
                .flat_map(|group| group.split(' '))
                .find(|f| *f == lower)
            else {
                return Err(FieldError::Invalid(name));
            };
            // Map Spark-specific names to DataFusion ones
            Ok(ExtractField::Part(match part {
                "yearofweek" => "isoyear",
                "dayofweek" | "dow" => "dow1",
                "dayofweek_iso" | "dow_iso" => "isodow",
                other => other,
            }))
        }
        Some(None) => Ok(ExtractField::Null),
        None if field.is_null() => Ok(ExtractField::Null),
        None => Err(FieldError::NotConstant),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Array, ArrayRef, Date32Array, Int32Array};
    use datafusion_common::config::ConfigOptions;

    /// 2000-01-01, a Saturday
    const DATE: i32 = 10_957;
    /// 2000-01-01 12:30:45 UTC, in seconds
    const TIMESTAMP_SECONDS: i64 = 946_729_845;

    /// Calls `invoke_with_args` directly, without `simplify`
    fn invoke(field: ScalarValue, source: ColumnarValue) -> Result<ColumnarValue> {
        let number_rows = match &source {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        SparkDatePart::new().invoke_with_args(ScalarFunctionArgs {
            arg_fields: vec![
                Arc::new(Field::new("field", field.data_type(), true)),
                Arc::new(Field::new("source", source.data_type(), true)),
            ],
            args: vec![ColumnarValue::Scalar(field), source],
            number_rows,
            return_field: Arc::new(Field::new("date_part", DataType::Int32, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn field(name: &str) -> ScalarValue {
        ScalarValue::Utf8(Some(name.to_string()))
    }

    fn extract(name: &str, source: ScalarValue) -> ScalarValue {
        match invoke(field(name), ColumnarValue::Scalar(source)).unwrap() {
            ColumnarValue::Scalar(result) => result,
            ColumnarValue::Array(array) => {
                ScalarValue::try_from_array(&array, 0).unwrap()
            }
        }
    }

    #[test]
    fn invoke_extracts_every_field_group() {
        let date = ScalarValue::Date32(Some(DATE));
        let timestamp = ScalarValue::TimestampSecond(Some(TIMESTAMP_SECONDS), None);
        for (name, source, expected) in [
            ("YEAR", &date, 2000),
            ("yrs", &date, 2000),
            ("YEAROFWEEK", &date, 1999),
            ("QTR", &date, 1),
            ("MONS", &date, 1),
            ("W", &date, 52),
            ("DAYS", &date, 1),
            // Spark counts DAYOFWEEK from 1 (Sunday) and DAYOFWEEK_ISO from 1 (Monday)
            ("DAYOFWEEK", &date, 7),
            ("dow", &date, 7),
            ("DAYOFWEEK_ISO", &date, 6),
            ("DOW_ISO", &date, 6),
            ("DOY", &date, 1),
            ("HRS", &timestamp, 12),
            ("MINS", &timestamp, 30),
            ("SECS", &timestamp, 45),
        ] {
            assert_eq!(
                extract(name, source.clone()),
                ScalarValue::Int32(Some(expected)),
                "{name}"
            );
        }
    }

    #[test]
    fn invoke_accepts_every_source_type() {
        let zone: Option<Arc<str>> = Some("Asia/Kolkata".into());
        for (source, hour) in [
            (ScalarValue::Date32(Some(DATE)), 0),
            (
                ScalarValue::TimestampSecond(Some(TIMESTAMP_SECONDS), None),
                12,
            ),
            (
                ScalarValue::TimestampMillisecond(Some(TIMESTAMP_SECONDS * 1_000), None),
                12,
            ),
            (
                ScalarValue::TimestampMicrosecond(
                    Some(TIMESTAMP_SECONDS * 1_000_000),
                    None,
                ),
                12,
            ),
            (
                ScalarValue::TimestampNanosecond(
                    Some(TIMESTAMP_SECONDS * 1_000_000_000),
                    None,
                ),
                12,
            ),
            // 12:30:45 UTC is 18:00:45 in Kolkata
            (
                ScalarValue::TimestampMicrosecond(
                    Some(TIMESTAMP_SECONDS * 1_000_000),
                    zone.clone(),
                ),
                18,
            ),
        ] {
            assert_eq!(
                extract("YEAR", source.clone()),
                ScalarValue::Int32(Some(2000))
            );
            assert_eq!(
                extract("HOUR", source.clone()),
                ScalarValue::Int32(Some(hour)),
                "{source:?}"
            );
        }
    }

    #[test]
    fn invoke_extracts_from_an_array() {
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![Some(DATE), None]));
        for name in [
            ScalarValue::Utf8(Some("year".to_string())),
            ScalarValue::LargeUtf8(Some("Year".to_string())),
            ScalarValue::Utf8View(Some("YEAR".to_string())),
        ] {
            let result = invoke(name, ColumnarValue::Array(Arc::clone(&dates)))
                .unwrap()
                .into_array(2)
                .unwrap();
            let expected: ArrayRef = Arc::new(Int32Array::from(vec![Some(2000), None]));
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn invoke_rejects_an_invalid_field() {
        // dow1, isodow and isoyear are DataFusion names, not Spark ones
        for name in ["QUARTERS", "DAYOFYEAR", "test", "dow1", "isodow", "isoyear"] {
            let err = invoke(
                field(name),
                ColumnarValue::Scalar(ScalarValue::Date32(Some(DATE))),
            )
            .unwrap_err();
            assert!(err.to_string().starts_with("Execution error:"), "{err}");
            assert!(err.to_string().contains("[INVALID_EXTRACT_FIELD]"), "{err}");
            assert!(
                err.to_string().contains(&format!(
                    "Cannot extract `{name}` from \"DATE '2000-01-01'\". SQLSTATE: 42601"
                )),
                "{err}"
            );
        }
    }

    #[test]
    fn invalid_field_uses_the_source_column_name() {
        let field = field("QUARTERS");
        let arg_fields = vec![
            Arc::new(Field::new("field", DataType::Utf8, false)),
            Arc::new(Field::new("event_date", DataType::Date32, true)),
        ];
        let error = SparkDatePart::new()
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &arg_fields,
                scalar_arguments: &[Some(&field), None],
            })
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Error during planning: [INVALID_EXTRACT_FIELD] Cannot extract `QUARTERS` from \"event_date\". SQLSTATE: 42601"
        );

        let source: ArrayRef = Arc::new(Date32Array::from(vec![Some(DATE), None]));
        let error = invoke(field, ColumnarValue::Array(source)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Execution error: [INVALID_EXTRACT_FIELD] Cannot extract `QUARTERS` from \"source\". SQLSTATE: 42601"
        );
    }

    #[test]
    fn invoke_returns_null_for_a_null_field() {
        let result = invoke(
            ScalarValue::Utf8(None),
            ColumnarValue::Scalar(ScalarValue::Date32(Some(DATE))),
        )
        .unwrap()
        .into_array(1)
        .unwrap();
        assert_eq!(result.data_type(), &DataType::Int32);
        assert!(result.is_null(0));
    }

    #[test]
    fn invoke_accepts_an_encoded_string_field() {
        let dictionary = ScalarValue::Dictionary(
            Box::new(DataType::Int32),
            Box::new(ScalarValue::Utf8(Some("year".to_string()))),
        );
        let result = invoke(
            dictionary,
            ColumnarValue::Scalar(ScalarValue::Date32(Some(DATE))),
        )
        .unwrap()
        .into_array(1)
        .unwrap();
        assert_eq!(
            ScalarValue::try_from_array(&result, 0).unwrap(),
            ScalarValue::Int32(Some(2000))
        );
    }

    #[test]
    fn invoke_rejects_a_field_that_is_not_constant() {
        let fields: ArrayRef = Arc::new(arrow::array::StringArray::from(vec!["year"]));
        let err = SparkDatePart::new()
            .invoke_with_args(ScalarFunctionArgs {
                arg_fields: vec![
                    Arc::new(Field::new("field", DataType::Utf8, true)),
                    Arc::new(Field::new("source", DataType::Date32, true)),
                ],
                args: vec![
                    ColumnarValue::Array(fields),
                    ColumnarValue::Scalar(ScalarValue::Date32(Some(DATE))),
                ],
                number_rows: 1,
                return_field: Arc::new(Field::new("date_part", DataType::Int32, true)),
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap_err();
        assert!(err.to_string().starts_with("Execution error:"), "{err}");
        assert!(
            err.to_string().contains("must be a constant string"),
            "{err}"
        );
    }

    #[test]
    fn invoke_rejects_an_invalid_field_for_a_timestamp() {
        let source =
            ScalarValue::TimestampMicrosecond(Some(TIMESTAMP_SECONDS * 1_000_000), None);
        let err = invoke(field("DAYOFYEAR"), ColumnarValue::Scalar(source)).unwrap_err();
        assert!(err.to_string().starts_with("Execution error:"), "{err}");
        assert!(err.to_string().contains("[INVALID_EXTRACT_FIELD]"), "{err}");
        assert!(
            err.to_string().contains(
                "Cannot extract `DAYOFYEAR` from \"TIMESTAMP '2000-01-01 12:30:45'\". SQLSTATE: 42601"
            ),
            "{err}"
        );
    }
}
