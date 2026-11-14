// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::LowerExp;
use std::sync::Arc;

use arrow::array::timezone::Tz;
use arrow::array::{
    Array, ArrayAccessor, ArrayRef, AsArray, MapArray, PrimitiveArray, StringArray,
    StringBuilder,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::cast;
use arrow::datatypes::{
    ArrowPrimitiveType, ArrowTimestampType, DataType, Decimal32Type, Decimal64Type,
    Decimal128Type, Decimal256Type, DecimalType, FieldRef, Float32Type, Float64Type,
    TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use arrow::error::ArrowError;
use arrow::json::writer::{
    Encoder, EncoderFactory, EncoderOptions, NullableEncoder, make_encoder,
};
use arrow::temporal_conversions::{as_datetime, as_datetime_with_timezone};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Column, Result, ScalarValue, exec_datafusion_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility, lit,
};
use datafusion_functions::core::expr_fn::named_struct;

/// Spark-compatible version of the [to_json] function.
///
/// Serializes a struct value to a JSON string, one JSON object per row.
/// Differs from arrow-json's default output in the Spark-specific rules
/// listed below, which are implemented by `spark_encoder_options`.
///
/// [to_json]: https://spark.apache.org/docs/latest/api/sql/index.html#to_json
///
/// to_json(struct<...>) -> Utf8
///
/// Semantics (matching Spark's defaults):
/// - A NULL struct row produces NULL
/// - NULL struct fields are omitted from the object (`ignoreNullFields = true`);
///   NULL map entries are kept as `"key":null`
/// - Map keys are written as strings whatever their type (`{"1":"v"}`)
/// - Binary values are Base64 strings
/// - Floats are written like Java's `Double.toString` / `Float.toString`:
///   plain decimal with at least one fraction digit for `1e-3 <= |x| < 1e7`
///   (`1.0`, `0.001`), scientific notation otherwise (`1.0E-5`, `1.23456789E8`);
///   non-finite values are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`
/// - Decimals are written like Java's `BigDecimal.toString`, keeping the
///   declared scale: `1.50`, `0.000001`, and scientific notation when the
///   adjusted exponent is below -6 (`0E-10`, `1.0E-7`). These are unquoted
///   JSON numbers, as Spark writes them.
/// - Timestamps are written as `yyyy-MM-dd'T'HH:mm:ss.SSS` (milliseconds,
///   truncated). A timestamp with a time zone is an instant and is written in
///   the session time zone (`datafusion.execution.time_zone`, UTC by default)
///   with its offset, `Z` for UTC; the time zone recorded on the Arrow type is
///   not used, as in Spark. A timestamp without a time zone (`TIMESTAMP_NTZ`)
///   is written as is, without an offset.
/// - Nested structs, lists and maps are serialized recursively
///
/// The serialization is done batch by batch with the arrow-json row encoder:
/// an encoder is built once per input array and each row is written straight
/// into a StringBuilder, so no intermediate RecordBatch or serde_json::Value
/// is created and nothing is collected.
///
/// Not yet supported (Spark accepts these): map/array top-level inputs and the
/// options map argument (`ignoreNullFields`, timestampFormat, ...).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkToJson {
    signature: Signature,
}

impl Default for SparkToJson {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkToJson {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkToJson {
    fn name(&self) -> &str {
        "to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::Struct(_)] => Ok(DataType::Utf8),
            [other] => plan_err!("to_json expects a struct argument, got {other}"),
            _ => plan_err!(
                "to_json expects exactly one argument, got {}",
                arg_types.len()
            ),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            arg_fields,
            config_options,
            ..
        } = args;
        let [input] = take_function_args(self.name(), args)?;
        let [field] = take_function_args(self.name(), arg_fields)?;

        match input {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(struct_to_json(
                &field,
                &array,
                &config_options,
            )?)),
            ColumnarValue::Scalar(scalar) => {
                let array = scalar.to_array()?;
                let result = struct_to_json(&field, &array, &config_options)?;
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &result, 0,
                )?))
            }
        }
    }
}

/// Serializes a struct array with Spark's output rules.
fn struct_to_json(
    field: &FieldRef,
    array: &ArrayRef,
    config: &ConfigOptions,
) -> Result<ArrayRef> {
    encode_struct_rows(field, array, &spark_encoder_options(config)?)
}

/// arrow-json options that reproduce Spark's `to_json` output.
///
/// This is the only Spark-specific input to the serialization: everything
/// else (`encode_struct_rows`, `IntoJsonStruct`, the UDF plumbing) is
/// independent of the output flavor and can be shared with other `to_json`
/// variants, e.g. a Postgres-style one with explicit nulls.
fn spark_encoder_options(config: &ConfigOptions) -> Result<EncoderOptions> {
    // Spark writes TIMESTAMP values in the session time zone
    let session_tz: Tz = config
        .execution
        .time_zone
        .as_ref()
        .map(|tz| tz.as_str())
        .unwrap_or("+00:00")
        .parse()?;

    Ok(EncoderOptions::default()
        // Spark default: ignoreNullFields = true -> NULL struct fields are omitted
        .with_explicit_nulls(false)
        // where Spark's output differs from arrow's defaults: NULL map entries
        // are kept, map keys are strings, binary is Base64, non-finite floats
        // are strings, timestamps use Spark's format (see `SparkJsonEncoderFactory`)
        .with_encoder_factory(Arc::new(SparkJsonEncoderFactory { session_tz })))
}

/// Encodes every row of a struct array as a JSON object string using the
/// given arrow-json options, one output string per input row.
///
/// `field` describes `array` (its name and the nested field names/types are
/// what the encoder uses for the JSON keys). A NULL struct row produces a
/// NULL output; everything else about the output is decided by `options`,
/// which makes this loop independent of the JSON flavor.
fn encode_struct_rows(
    field: &FieldRef,
    array: &ArrayRef,
    options: &EncoderOptions,
) -> Result<ArrayRef> {
    let mut encoder = make_encoder(field, array.as_ref(), options)?;
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * 32);
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    for row in 0..array.len() {
        if encoder.is_null(row) {
            builder.append_null();
            continue;
        }
        buf.clear();
        encoder.encode(row, &mut buf);
        let json = std::str::from_utf8(&buf)
            .map_err(|e| exec_datafusion_err!("to_json produced invalid UTF-8: {e}"))?;
        builder.append_value(json);
    }

    Ok(Arc::new(builder.finish()))
}

/// Overrides arrow-json's default encoders where Spark's `to_json` output
/// differs from arrow's. `make_encoder` consults the factory for every array at
/// every nesting level, so the overrides also apply inside lists, structs and
/// maps.
#[derive(Debug)]
struct SparkJsonEncoderFactory {
    /// `datafusion.execution.time_zone`; timestamps with a time zone are
    /// written in this zone
    session_tz: Tz,
}

impl EncoderFactory for SparkJsonEncoderFactory {
    fn make_default_encoder<'a>(
        &self,
        field: &'a FieldRef,
        array: &'a dyn Array,
        options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        match array.data_type() {
            DataType::Map(_, _) => {
                let array = array.as_map();
                let encoder = SparkMapEncoder::try_new(field, array, options)?;
                Ok(Some(NullableEncoder::new(
                    Box::new(encoder),
                    array.nulls().cloned(),
                )))
            }
            // Spark serializes binary values as Base64 strings, arrow-json as hex
            DataType::Binary => Ok(Some(base64_encoder(array.as_binary::<i32>()))),
            DataType::LargeBinary => Ok(Some(base64_encoder(array.as_binary::<i64>()))),
            DataType::BinaryView => Ok(Some(base64_encoder(array.as_binary_view()))),
            DataType::FixedSizeBinary(_) => {
                Ok(Some(base64_encoder(array.as_fixed_size_binary())))
            }
            // Spark writes floats like Java's Double.toString (1.0E-5, 1.0) and
            // non-finite values as the strings "NaN", "Infinity" and "-Infinity";
            // arrow-json writes 0.00001 and JSON null
            DataType::Float32 => Ok(Some(spark_float_encoder(
                array.as_primitive::<Float32Type>(),
            ))),
            DataType::Float64 => Ok(Some(spark_float_encoder(
                array.as_primitive::<Float64Type>(),
            ))),
            // Spark writes decimals like Java's BigDecimal.toString (0E-10, 1.0E-7);
            // arrow-json writes them padded to the scale (0.0000000000). Spark data
            // maps to Decimal128 (precision <= 38); the other Arrow decimal widths
            // are formatted the same way rather than rejected.
            DataType::Decimal32(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal32Type>(),
            ))),
            DataType::Decimal64(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal64Type>(),
            ))),
            DataType::Decimal128(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal128Type>(),
            ))),
            DataType::Decimal256(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal256Type>(),
            ))),
            // Spark writes timestamps as yyyy-MM-dd'T'HH:mm:ss.SSS, instants in
            // the session time zone with their offset; arrow-json keeps the full
            // precision and uses the time zone recorded on the type
            DataType::Timestamp(unit, tz) => {
                let session_tz = tz.as_ref().map(|_| self.session_tz);
                Ok(Some(match unit {
                    TimeUnit::Second => timestamp_encoder(
                        array.as_primitive::<TimestampSecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Millisecond => timestamp_encoder(
                        array.as_primitive::<TimestampMillisecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Microsecond => timestamp_encoder(
                        array.as_primitive::<TimestampMicrosecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Nanosecond => timestamp_encoder(
                        array.as_primitive::<TimestampNanosecondType>(),
                        session_tz,
                    )?,
                }))
            }
            // everything else uses arrow's default encoder
            _ => Ok(None),
        }
    }
}

/// Spark's default `timestampFormat`, `yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]`,
/// without the offset part (added separately so that UTC is written as `Z`).
const SPARK_TIMESTAMP_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.3f";

/// Builds an encoder that writes timestamps the way Spark does.
///
/// `session_tz` is `Some` for a timestamp with a time zone (a Spark
/// `TIMESTAMP`, an instant written in the session time zone with its offset)
/// and `None` for one without (`TIMESTAMP_NTZ`, written as is). The values
/// are formatted here, where an out-of-range value can be reported, so that
/// `encode` only has to copy them.
fn timestamp_encoder<'a, T: ArrowTimestampType>(
    array: &'a PrimitiveArray<T>,
    session_tz: Option<Tz>,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let mut texts = StringBuilder::with_capacity(array.len(), array.len() * 30);
    for value in array.iter() {
        match value {
            Some(v) => texts.append_value(spark_timestamp_text::<T>(v, session_tz)?),
            None => texts.append_null(),
        }
    }
    Ok(NullableEncoder::new(
        Box::new(QuotedTextEncoder(texts.finish())),
        array.nulls().cloned(),
    ))
}

/// Formats one timestamp value as Spark's `to_json` does.
fn spark_timestamp_text<T: ArrowTimestampType>(
    value: i64,
    session_tz: Option<Tz>,
) -> Result<String, ArrowError> {
    let out_of_range = || {
        ArrowError::JsonError(format!(
            "to_json: timestamp value {value} ({:?}) is out of range",
            T::UNIT
        ))
    };
    match session_tz {
        Some(tz) => {
            let dt =
                as_datetime_with_timezone::<T>(value, tz).ok_or_else(out_of_range)?;
            let offset = dt.format("%:z").to_string();
            // Java's `XXX` pattern writes a zero offset as `Z`
            let offset = if offset == "+00:00" { "Z" } else { &offset };
            Ok(format!("{}{offset}", dt.format(SPARK_TIMESTAMP_FORMAT)))
        }
        None => Ok(as_datetime::<T>(value)
            .ok_or_else(out_of_range)?
            .format(SPARK_TIMESTAMP_FORMAT)
            .to_string()),
    }
}

/// Writes pre-formatted text as a JSON string. The text must not need
/// escaping (timestamps are digits and `-T:.+Z` only).
struct QuotedTextEncoder(StringArray);

impl Encoder for QuotedTextEncoder {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        out.push(b'"');
        out.extend_from_slice(self.0.value(idx).as_bytes());
        out.push(b'"');
    }
}

/// Floating point types that Spark serializes like Java's `Double.toString`.
trait SparkFloat: Copy + LowerExp {
    /// The JSON text Spark emits for a non-finite value, or `None` if finite.
    fn non_finite_json(self) -> Option<&'static [u8]>;
}

macro_rules! spark_float {
    ($($t:ty),*) => {$(
        impl SparkFloat for $t {
            fn non_finite_json(self) -> Option<&'static [u8]> {
                if self.is_nan() {
                    Some(b"\"NaN\"")
                } else if self.is_infinite() {
                    Some(if self.is_sign_positive() {
                        b"\"Infinity\""
                    } else {
                        b"\"-Infinity\""
                    })
                } else {
                    None
                }
            }
        }
    )*};
}
spark_float!(f32, f64);

/// Formats a finite float the way Java's `Double.toString` / `Float.toString`
/// do, which is what Spark's `to_json` writes:
///
/// - `1e-3 <= |x| < 1e7`: plain decimal with at least one fraction digit
///   (`1.0`, `0.001`, `123.456`)
/// - otherwise: scientific notation `d.dddE<exp>` with at least one fraction
///   digit and no `+` in the exponent (`1.0E-5`, `1.23456789E8`, `1.0E10`)
///
/// The digits are the shortest that round-trip, taken from Rust's `{:e}`
/// formatting, which is also what Java produces (JDK 19+; older JDKs
/// occasionally emit one more digit).
fn java_float_text<F: LowerExp>(value: F) -> String {
    // e.g. "1.23456789e8", "1e-5", "-0e0"
    let exp_form = format!("{value:e}");
    let (sign, rest) = match exp_form.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", exp_form.as_str()),
    };
    // `{:e}` always has the form <digits>[.<digits>]e<exp>; if it ever did
    // not, the exponent form is still a valid JSON number
    let Some((mantissa, exp)) = rest.split_once('e') else {
        return exp_form;
    };
    let Ok(exp) = exp.parse::<i32>() else {
        return exp_form;
    };
    // all significant digits; `exp` is the decimal exponent of the first one
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut text = String::with_capacity(digits.len() + 8);
    text.push_str(sign);
    if (-3..7).contains(&exp) {
        if exp < 0 {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            text.push_str(&digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                text.push_str(&digits);
                text.extend(std::iter::repeat_n('0', int_len - digits.len()));
                text.push_str(".0");
            } else {
                text.push_str(&digits[..int_len]);
                text.push('.');
                text.push_str(&digits[int_len..]);
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        text.push_str(first);
        text.push('.');
        text.push_str(if rest.is_empty() { "0" } else { rest });
        text.push('E');
        text.push_str(&exp.to_string());
    }
    text
}

/// Builds a float encoder that writes values the way Spark does.
fn spark_float_encoder<'a, T>(array: &'a PrimitiveArray<T>) -> NullableEncoder<'a>
where
    T: ArrowPrimitiveType,
    T::Native: SparkFloat,
{
    let encoder = SparkFloatEncoder {
        values: array.values(),
    };
    NullableEncoder::new(Box::new(encoder), array.nulls().cloned())
}

struct SparkFloatEncoder<'a, N: SparkFloat> {
    values: &'a [N],
}

impl<N: SparkFloat> Encoder for SparkFloatEncoder<'_, N> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let value = self.values[idx];
        match value.non_finite_json() {
            Some(text) => out.extend_from_slice(text),
            None => out.extend_from_slice(java_float_text(value).as_bytes()),
        }
    }
}

/// Formats a decimal the way Java's `BigDecimal.toString` does, which is what
/// Spark's `to_json` writes. `unscaled` is the signed unscaled value as text.
///
/// With `adjusted = -scale + (digits - 1)`: plain notation when `scale >= 0`
/// and `adjusted >= -6` (`1.50`, `0.000001`), scientific notation otherwise
/// (`0E-10`, `1.0E-7`, `1.23E+4`; the exponent always carries a sign). Both
/// forms are valid JSON numbers, so they are written unquoted.
fn java_big_decimal_text(unscaled: &str, scale: i8) -> String {
    let (sign, coefficient) = match unscaled.strip_prefix('-') {
        Some(coefficient) => ("-", coefficient),
        None => ("", unscaled),
    };
    let adjusted = -(scale as i64) + (coefficient.len() as i64 - 1);

    let mut text = String::with_capacity(coefficient.len() + 8);
    text.push_str(sign);
    if scale >= 0 && adjusted >= -6 {
        let scale = scale as usize;
        if scale == 0 {
            text.push_str(coefficient);
        } else if coefficient.len() > scale {
            let (int, frac) = coefficient.split_at(coefficient.len() - scale);
            text.push_str(int);
            text.push('.');
            text.push_str(frac);
        } else {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', scale - coefficient.len()));
            text.push_str(coefficient);
        }
    } else {
        // the coefficient of an integer has at least one digit
        let (first, rest) = coefficient.split_at(1);
        text.push_str(first);
        if !rest.is_empty() {
            text.push('.');
            text.push_str(rest);
        }
        text.push('E');
        if adjusted >= 0 {
            text.push('+');
        }
        text.push_str(&adjusted.to_string());
    }
    text
}

/// Builds a decimal encoder that writes values like Java's `BigDecimal.toString`.
/// The scale is the one declared on the Arrow type, as Spark's `BigDecimal`
/// carries the column's scale.
fn spark_decimal_encoder<'a, T>(array: &'a PrimitiveArray<T>) -> NullableEncoder<'a>
where
    T: DecimalType,
    T::Native: std::fmt::Display,
{
    let encoder = SparkDecimalEncoder {
        values: array.values(),
        scale: array.scale(),
    };
    NullableEncoder::new(Box::new(encoder), array.nulls().cloned())
}

struct SparkDecimalEncoder<'a, N> {
    values: &'a [N],
    scale: i8,
}

impl<N: std::fmt::Display + Copy> Encoder for SparkDecimalEncoder<'_, N> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let text = java_big_decimal_text(&self.values[idx].to_string(), self.scale);
        out.extend_from_slice(text.as_bytes());
    }
}

/// Builds a Base64 encoder for any of arrow's binary array types.
fn base64_encoder<'a, B>(array: B) -> NullableEncoder<'a>
where
    B: ArrayAccessor<Item = &'a [u8]> + Array + 'a,
{
    let nulls = array.nulls().cloned();
    NullableEncoder::new(Box::new(Base64Encoder::new(array)), nulls)
}

/// Encodes binary values as Base64 strings (standard alphabet, padded), which
/// is what Spark's `to_json` produces; arrow-json's default is hexadecimal.
struct Base64Encoder<B> {
    array: B,
    scratch: String,
}

impl<B> Base64Encoder<B> {
    fn new(array: B) -> Self {
        Self {
            array,
            scratch: String::new(),
        }
    }
}

impl<'a, B> Encoder for Base64Encoder<B>
where
    B: ArrayAccessor<Item = &'a [u8]>,
{
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        self.scratch.clear();
        BASE64.encode_string(self.array.value(idx), &mut self.scratch);
        // Base64 output is [A-Za-z0-9+/=], so it needs no JSON escaping
        out.push(b'"');
        out.extend_from_slice(self.scratch.as_bytes());
        out.push(b'"');
    }
}

/// Encodes a map as a JSON object the way Spark does:
///
/// - NULL values are always written as `"key":null`. arrow-json's own map
///   encoder drops such entries when `explicit_nulls` is off, but Spark's
///   `ignoreNullFields` only applies to struct fields.
/// - keys are written as strings whatever their type (`{"1":"v"}`), see
///   `MapKeyEncoder`.
struct SparkMapEncoder<'a> {
    offsets: OffsetBuffer<i32>,
    keys: MapKeyEncoder,
    values: NullableEncoder<'a>,
}

impl<'a> SparkMapEncoder<'a> {
    fn try_new(
        field: &'a FieldRef,
        array: &'a MapArray,
        options: &'a EncoderOptions,
    ) -> Result<Self, ArrowError> {
        let keys = MapKeyEncoder::try_new(array.keys())?;
        // The value encoder is built with the same options, so nested structs
        // keep omitting NULL fields and nested maps use this encoder.
        let values = make_encoder(field, array.values(), options)?;

        Ok(Self {
            offsets: array.offsets().clone(),
            keys,
            values,
        })
    }
}

/// Encodes map keys as JSON strings.
///
/// Spark writes every map key as its string form (`{"1":"v"}`,
/// `{"true":"v"}`), so non-string keys are cast to `Utf8` first. Key types
/// without a string cast (lists, structs, ...) are rejected.
struct MapKeyEncoder(StringArray);

impl MapKeyEncoder {
    fn try_new(keys: &ArrayRef) -> Result<Self, ArrowError> {
        let strings = cast(keys, &DataType::Utf8).map_err(|e| {
            ArrowError::JsonError(format!(
                "to_json: unsupported map key type {}: {e}",
                keys.data_type()
            ))
        })?;
        if strings.null_count() > 0 {
            return Err(ArrowError::InvalidArgumentError(
                "to_json: encountered a NULL map key".to_string(),
            ));
        }
        Ok(Self(strings.as_string::<i32>().clone()))
    }
}

impl Encoder for MapKeyEncoder {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        // serde_json handles the quoting and escaping; writing to a Vec is
        // infallible
        serde_json::to_writer(&mut *out, self.0.value(idx)).unwrap();
    }
}

impl Encoder for SparkMapEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;

        out.push(b'{');
        for i in start..end {
            if i != start {
                out.push(b',');
            }
            self.keys.encode(i, out);
            out.push(b':');
            if self.values.is_null(i) {
                out.extend_from_slice(b"null");
            } else {
                self.values.encode(i, out);
            }
        }
        out.push(b'}');
    }
}

/// Input accepted by the to_json expression builder.
///
/// Implemented for an expression that evaluates to a struct, such as a struct column or a `named_struct` call,
/// which is used as-is; and for a slice or array of column names, which are packed into a struct first.
/// In the second form the JSON keys are the plain column names without any table qualifier.
pub trait IntoJsonStruct {
    /// Converts self into the struct-valued [`Expr`] to serialize.
    fn into_struct_expr(self) -> Expr;
}

impl IntoJsonStruct for Expr {
    fn into_struct_expr(self) -> Expr {
        self
    }
}

impl<S: AsRef<str>> IntoJsonStruct for &[S] {
    fn into_struct_expr(self) -> Expr {
        let args = self
            .iter()
            .flat_map(|name| {
                let column = Column::from_qualified_name(name.as_ref());
                [lit(column.name().to_owned()), Expr::Column(column)]
            })
            .collect();
        named_struct(args)
    }
}

impl<S: AsRef<str>, const N: usize> IntoJsonStruct for &[S; N] {
    fn into_struct_expr(self) -> Expr {
        self.as_slice().into_struct_expr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        BinaryArray, BinaryBuilder, BinaryViewArray, BooleanArray, FixedSizeBinaryArray,
        Float32Array, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array,
        LargeBinaryArray, ListBuilder, MapBuilder, StringArray, StructArray,
        StructBuilder, TimestampMicrosecondArray, TimestampSecondArray,
    };
    use arrow::array::{Decimal32Array, Decimal128Array, Decimal256Array};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{Field, Fields, i256};
    use datafusion_common::cast::as_string_array;
    use datafusion_common::config::ConfigOptions;

    /// Runs to_json over input (whose type is described by `field`) and
    /// returns the resulting string array.
    fn run(input: ColumnarValue, field: FieldRef) -> Result<ArrayRef> {
        run_with_config(input, field, ConfigOptions::default())
    }

    /// Like `run`, with the session configuration (e.g. the time zone).
    fn run_with_config(
        input: ColumnarValue,
        field: FieldRef,
        config: ConfigOptions,
    ) -> Result<ArrayRef> {
        let number_rows = match &input {
            ColumnarValue::Array(a) => a.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let args = ScalarFunctionArgs {
            args: vec![input],
            arg_fields: vec![field],
            number_rows,
            return_field: Field::new("to_json", DataType::Utf8, true).into(),
            config_options: Arc::new(config),
        };
        SparkToJson::new()
            .invoke_with_args(args)?
            .to_array(number_rows)
    }

    fn sample_struct() -> StructArray {
        let a = Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef;
        let b = Arc::new(StringArray::from(vec![Some("x"), None, Some("z")])) as ArrayRef;
        let c =
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Boolean, true),
        ]);
        let nulls = NullBuffer::from(vec![true, true, false]);
        StructArray::new(fields, vec![a, b, c], Some(nulls))
    }

    fn field_of(array: &StructArray) -> FieldRef {
        Field::new("s", array.data_type().clone(), true).into()
    }

    #[test]
    fn test_return_type() {
        let func = SparkToJson::new();
        let fields = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        assert_eq!(
            func.return_type(&[DataType::Struct(fields)]).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn test_return_type_rejects_non_struct() {
        let err = SparkToJson::new()
            .return_type(&[DataType::Utf8])
            .unwrap_err();
        assert!(err.to_string().contains("expects a struct argument"));

        let err = SparkToJson::new().return_type(&[]).unwrap_err();
        assert!(err.to_string().contains("exactly one argument"));
    }

    #[test]
    fn test_array_input() {
        let input = sample_struct();
        let field = field_of(&input);
        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0), r#"{"a":1,"b":"x","c":true}"#);
        // NULL field is omitted (Spark ignoreNullFields = true)
        assert_eq!(out.value(1), r#"{"a":2,"c":false}"#);
        // NULL struct row -> NULL
        assert!(out.is_null(2));
    }

    #[test]
    fn test_encode_struct_rows_is_independent_of_the_json_flavor() {
        // The output flavor is decided entirely by the options: the same loop
        // with explicit nulls (what a Postgres-style `to_json` would use)
        // keeps the NULL fields that the Spark options omit.
        let input = sample_struct();
        let field = field_of(&input);
        let array = Arc::new(input) as ArrayRef;

        let spark_options = spark_encoder_options(&ConfigOptions::default()).unwrap();
        let spark = encode_struct_rows(&field, &array, &spark_options).unwrap();
        let spark = as_string_array(&spark).unwrap();
        assert_eq!(spark.value(1), r#"{"a":2,"c":false}"#);

        let explicit = EncoderOptions::default().with_explicit_nulls(true);
        let other = encode_struct_rows(&field, &array, &explicit).unwrap();
        let other = as_string_array(&other).unwrap();
        assert_eq!(other.value(1), r#"{"a":2,"b":null,"c":false}"#);
        // a NULL struct row is NULL in both
        assert!(spark.is_null(2) && other.is_null(2));
    }

    #[test]
    fn test_scalar_input() {
        let input = sample_struct();
        let field = field_of(&input);
        let scalar = ScalarValue::Struct(Arc::new(input.slice(0, 1)));
        let out = run(ColumnarValue::Scalar(scalar), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), r#"{"a":1,"b":"x","c":true}"#);
    }

    #[test]
    fn test_scalar_null_struct() {
        let input = sample_struct();
        let field = field_of(&input);
        let scalar = ScalarValue::Struct(Arc::new(input.slice(2, 1)));
        let out = run(ColumnarValue::Scalar(scalar), field).unwrap();

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
    }

    #[test]
    fn test_nested_struct_and_list() {
        // {inner: {x: Int64}, tags: List<Utf8>}
        let x = Arc::new(Int64Array::from(vec![Some(10), None])) as ArrayRef;
        let inner_fields = Fields::from(vec![Field::new("x", DataType::Int64, true)]);
        let inner = Arc::new(StructArray::new(inner_fields, vec![x], None)) as ArrayRef;

        let mut tags = ListBuilder::new(StringBuilder::new());
        tags.values().append_value("p");
        tags.values().append_value("q");
        tags.append(true);
        tags.append(false);
        let tags = Arc::new(tags.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("inner", inner.data_type().clone(), true),
            Field::new("tags", tags.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![inner, tags], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"inner":{"x":10},"tags":["p","q"]}"#);
        // nested NULL field and NULL list are both omitted
        assert_eq!(out.value(1), r#"{"inner":{}}"#);
    }

    #[test]
    fn test_sliced_nested_struct_and_list() {
        // 4 rows of {inner: {x: Int64}, tags: List<Utf8>} with uneven list
        // lengths (non-trivial list offsets) and a NULL parent in row 2:
        //   0: {x:1}, []        1: {x:2}, [a,b]
        //   2: NULL             3: {x:4}, [d,e,f]
        let x = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let inner_fields = Fields::from(vec![Field::new("x", DataType::Int64, true)]);
        let inner = Arc::new(StructArray::new(inner_fields, vec![x], None)) as ArrayRef;

        let mut tags = ListBuilder::new(StringBuilder::new());
        tags.append(true); // []
        for t in ["a", "b"] {
            tags.values().append_value(t);
        }
        tags.append(true);
        tags.values().append_value("c");
        tags.append(true);
        for t in ["d", "e", "f"] {
            tags.values().append_value(t);
        }
        tags.append(true);
        let tags = Arc::new(tags.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("inner", inner.data_type().clone(), true),
            Field::new("tags", tags.data_type().clone(), true),
        ]);
        let nulls = NullBuffer::from(vec![true, true, false, true]);
        let full = StructArray::new(fields, vec![inner, tags], Some(nulls));
        let field = field_of(&full);

        // encode rows 1..4 through a non-zero-offset slice
        let sliced = full.slice(1, 3);
        let out = run(ColumnarValue::Array(Arc::new(sliced)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out.null_count(), 1);
        assert_eq!(out.value(0), r#"{"inner":{"x":2},"tags":["a","b"]}"#);
        assert!(
            out.is_null(1),
            "NULL parent row must stay NULL after slicing"
        );
        assert_eq!(out.value(2), r#"{"inner":{"x":4},"tags":["d","e","f"]}"#);
    }

    #[test]
    fn test_string_escaping() {
        let a = Arc::new(StringArray::from(vec![Some("he said \"hi\"\n")])) as ArrayRef;
        let fields = Fields::from(vec![Field::new("a", DataType::Utf8, true)]);
        let input = StructArray::new(fields, vec![a], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"a":"he said \"hi\"\n"}"#);
    }
    /// Unwraps a named_struct(...) call and returns its arguments.
    fn named_struct_args(expr: &Expr) -> &[Expr] {
        match expr {
            Expr::ScalarFunction(f) if f.func.name() == "named_struct" => &f.args,
            other => panic!("expected named_struct call, got {other:?}"),
        }
    }

    #[test]
    fn test_into_json_struct_passes_expr_through() {
        let expr = Expr::Column(Column::from_name("event"));
        assert_eq!(expr.clone().into_struct_expr(), expr);
    }

    #[test]
    fn test_into_json_struct_from_column_names() {
        let expr = ["foo", "bar"].into_struct_expr();
        let args = named_struct_args(&expr);

        assert_eq!(
            args,
            &[
                lit("foo"),
                Expr::Column(Column::from_name("foo")),
                lit("bar"),
                Expr::Column(Column::from_name("bar")),
            ]
        );
    }

    #[test]
    fn test_into_json_struct_accepts_slice_and_strings() {
        let names: Vec<String> = vec!["foo".to_string(), "bar".to_string()];
        let from_vec = names.as_slice().into_struct_expr();
        let from_array = ["foo", "bar"].into_struct_expr();
        assert_eq!(from_vec, from_array);
    }

    #[test]
    fn test_into_json_struct_uses_unqualified_key_for_qualified_column() {
        let expr = ["t.foo"].into_struct_expr();
        let args = named_struct_args(&expr);

        // JSON key is the bare column name...
        assert_eq!(args[0], lit("foo"));
        // ...while the column reference keeps its qualifier
        assert_eq!(args[1], Expr::Column(Column::from_qualified_name("t.foo")));
    }

    #[test]
    fn test_map_null_values_kept_but_null_struct_fields_omitted() {
        // m: Map<Utf8, Int32> with rows {"k":null}, {"a":1,"b":null}, NULL
        let mut mb = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        mb.keys().append_value("k");
        mb.values().append_null();
        mb.append(true).unwrap();
        mb.keys().append_value("a");
        mb.values().append_value(1);
        mb.keys().append_value("b");
        mb.values().append_null();
        mb.append(true).unwrap();
        mb.append(false).unwrap();
        let m = Arc::new(mb.finish()) as ArrayRef;

        // s: Int32 with a NULL in row 1
        let s = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("m", m.data_type().clone(), true),
            Field::new("s", DataType::Int32, true),
        ]);
        let input = StructArray::new(fields, vec![m, s], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        // a NULL map *value* is written explicitly (Spark keeps map entries)
        assert_eq!(out.value(0), r#"{"m":{"k":null},"s":1}"#);
        // ...while a NULL struct *field* (`s`) is still omitted
        assert_eq!(out.value(1), r#"{"m":{"a":1,"b":null}}"#);
        // a NULL map is a NULL struct field, so it is omitted as well
        assert_eq!(out.value(2), r#"{"s":3}"#);
    }

    #[test]
    fn test_binary_is_base64_encoded() {
        // b: Binary ["abc", "", NULL] -> "YWJj", "", omitted
        let b = Arc::new(BinaryArray::from(vec![
            Some(b"abc".as_slice()),
            Some(b"".as_slice()),
            None,
        ])) as ArrayRef;
        let abc = vec![Some(b"abc".as_slice()); 3];
        let lb = Arc::new(LargeBinaryArray::from(abc.clone())) as ArrayRef;
        let bv = Arc::new(BinaryViewArray::from(abc)) as ArrayRef;
        let fsb = Arc::new(
            FixedSizeBinaryArray::try_from_iter(vec![b"ab".as_slice(); 3].into_iter())
                .unwrap(),
        ) as ArrayRef;

        // l: List<Binary> [["abc"]] per row; n: Struct<b: Binary> {"abc"} per row
        let mut l = ListBuilder::new(BinaryBuilder::new());
        for _ in 0..3 {
            l.values().append_value(b"abc");
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("b", DataType::LargeBinary, true)]),
            vec![Arc::clone(&lb)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("b", DataType::Binary, true),
            Field::new("lb", DataType::LargeBinary, true),
            Field::new("bv", DataType::BinaryView, true),
            Field::new("fsb", DataType::FixedSizeBinary(2), true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![b, lb, bv, fsb, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"b":"YWJj","lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
        // empty binary is an empty string, not NULL
        assert_eq!(
            out.value(1),
            r#"{"b":"","lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
        // NULL binary is a NULL struct field -> omitted
        assert_eq!(
            out.value(2),
            r#"{"lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
    }

    #[test]
    fn test_non_finite_floats_are_spark_strings() {
        // d: Float64 [NaN, +inf, -inf, 1.5, NULL]; f: Float32 the same
        let d = Arc::new(Float64Array::from(vec![
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(1.5),
            None,
        ])) as ArrayRef;
        let f = Arc::new(Float32Array::from(vec![
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(1.5),
            None,
        ])) as ArrayRef;

        // l: List<Float64> [[NaN, 1.5]] per row; n: Struct<d: Float64>
        let mut l = ListBuilder::new(Float64Builder::new());
        for _ in 0..5 {
            l.values().append_value(f64::NAN);
            l.values().append_value(1.5);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("d", DataType::Float64, true)]),
            vec![Arc::clone(&d)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, f, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":"NaN","f":"NaN","l":["NaN",1.5],"n":{"d":"NaN"}}"#
        );
        assert_eq!(
            out.value(1),
            r#"{"d":"Infinity","f":"Infinity","l":["NaN",1.5],"n":{"d":"Infinity"}}"#
        );
        assert_eq!(
            out.value(2),
            r#"{"d":"-Infinity","f":"-Infinity","l":["NaN",1.5],"n":{"d":"-Infinity"}}"#
        );
        // finite values keep arrow's formatting
        assert_eq!(
            out.value(3),
            r#"{"d":1.5,"f":1.5,"l":["NaN",1.5],"n":{"d":1.5}}"#
        );
        // NULL is still a NULL struct field -> omitted
        assert_eq!(out.value(4), r#"{"l":["NaN",1.5],"n":{}}"#);
    }

    #[test]
    fn test_map_keys_are_stringified() {
        // i: Map<Int32, Utf8> {1:"a", 2:"b"}; f: Map<Float64, Utf8> {1.5:"c"}
        let mut i = MapBuilder::new(None, Int32Builder::new(), StringBuilder::new());
        i.keys().append_value(1);
        i.values().append_value("a");
        i.keys().append_value(2);
        i.values().append_value("b");
        i.append(true).unwrap();
        let i = Arc::new(i.finish()) as ArrayRef;

        let mut f = MapBuilder::new(None, Float64Builder::new(), StringBuilder::new());
        f.keys().append_value(1.5);
        f.values().append_value("c");
        f.append(true).unwrap();
        let f = Arc::new(f.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("i", i.data_type().clone(), true),
            Field::new("f", f.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![i, f], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"i":{"1":"a","2":"b"},"f":{"1.5":"c"}}"#);
    }

    #[test]
    fn test_map_with_unsupported_key_type_errors() {
        // Map<Struct<a: Int32>, Utf8>: structs have no string cast
        // (lists do: a List<Int32> key becomes the string "[1]")
        let key_fields = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        let key_builder = StructBuilder::from_fields(key_fields, 1);
        let mut m = MapBuilder::new(None, key_builder, StringBuilder::new());
        m.keys()
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_value(1);
        m.keys().append(true);
        m.values().append_value("v");
        m.append(true).unwrap();
        let m = Arc::new(m.finish()) as ArrayRef;

        let fields = Fields::from(vec![Field::new("m", m.data_type().clone(), true)]);
        let input = StructArray::new(fields, vec![m], None);
        let field = field_of(&input);

        let err = run(ColumnarValue::Array(Arc::new(input)), field).unwrap_err();
        assert!(
            err.to_string()
                .contains("to_json: unsupported map key type"),
            "unexpected error: {err}"
        );
    }

    /// 2024-01-01 01:02:03.123456 UTC, in microseconds since the epoch
    const TS_MICROS: i64 = 1_704_070_923_123_456;

    #[test]
    fn test_timestamps_use_spark_format_and_session_time_zone() {
        // ltz: Timestamp(us, "UTC") -> Spark TIMESTAMP (an instant)
        // ntz: Timestamp(us, None)  -> Spark TIMESTAMP_NTZ (a wall-clock time)
        let ltz = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(TS_MICROS), None])
                .with_timezone("UTC"),
        ) as ArrayRef;
        let ntz = Arc::new(TimestampMicrosecondArray::from(vec![Some(TS_MICROS), None]))
            as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("ltz", ltz.data_type().clone(), true),
            Field::new("ntz", ntz.data_type().clone(), true),
        ]);
        let input = Arc::new(StructArray::new(fields, vec![ltz, ntz], None));
        let field = field_of(&input);

        // default session time zone (UTC): milliseconds, offset written as `Z`
        let out = run(
            ColumnarValue::Array(Arc::clone(&input) as ArrayRef),
            Arc::clone(&field),
        )
        .unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"ltz":"2024-01-01T01:02:03.123Z","ntz":"2024-01-01T01:02:03.123"}"#
        );
        // NULL timestamps are NULL struct fields -> omitted
        assert_eq!(out.value(1), "{}");

        // the instant is written in the session time zone with its offset;
        // the wall-clock value is unchanged
        let mut config = ConfigOptions::default();
        config
            .set("datafusion.execution.time_zone", "+02:00")
            .unwrap();
        let out = run_with_config(ColumnarValue::Array(input), field, config).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"ltz":"2024-01-01T03:02:03.123+02:00","ntz":"2024-01-01T01:02:03.123"}"#
        );
    }

    #[test]
    fn test_timestamp_type_time_zone_is_ignored_and_precision_is_millis() {
        // Spark has no per-column time zone: the zone recorded on the Arrow
        // type does not change the output, only the session time zone does
        let plus5 = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(TS_MICROS)])
                .with_timezone("+05:00"),
        ) as ArrayRef;
        // whole seconds still get three fractional digits; a nested list too
        let secs = Arc::new(
            TimestampSecondArray::from(vec![Some(TS_MICROS / 1_000_000)])
                .with_timezone("UTC"),
        ) as ArrayRef;
        let mut l =
            ListBuilder::new(TimestampSecondArray::builder(1).with_timezone("UTC"));
        l.values().append_value(TS_MICROS / 1_000_000);
        l.append(true);
        let l = Arc::new(l.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("plus5", plus5.data_type().clone(), true),
            Field::new("secs", secs.data_type().clone(), true),
            Field::new("l", l.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![plus5, secs, l], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"plus5":"2024-01-01T01:02:03.123Z","secs":"2024-01-01T01:02:03.000Z","l":["2024-01-01T01:02:03.000Z"]}"#
        );
    }

    #[test]
    fn test_java_float_text() {
        // Java Double.toString: plain decimal for 1e-3 <= |x| < 1e7, else scientific
        let cases: [(f64, &str); 14] = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (123.456, "123.456"),
            (0.001, "0.001"),
            (0.0001, "1.0E-4"),
            (1e-5, "1.0E-5"),
            (9999999.0, "9999999.0"),
            (1e7, "1.0E7"),
            (123456789.0, "1.23456789E8"),
            (1e10, "1.0E10"),
            (-2.5e-8, "-2.5E-8"),
            (f64::MAX, "1.7976931348623157E308"),
        ];
        for (value, expected) in cases {
            assert_eq!(java_float_text(value), expected, "value {value:?}");
        }
        // Float.toString uses the shortest digits of the f32 value
        let cases: [(f32, &str); 4] = [
            (0.1, "0.1"),
            (1e10, "1.0E10"),
            (3.0, "3.0"),
            (f32::MAX, "3.4028235E38"),
        ];
        for (value, expected) in cases {
            assert_eq!(java_float_text(value), expected, "value {value:?}");
        }
    }

    #[test]
    fn test_finite_floats_use_java_to_string() {
        // d: Float64, f: Float32, l: List<Float64>, n: Struct<d: Float64>
        let d =
            Arc::new(Float64Array::from(vec![1e-5, 123456789.0, 1e10, 1.5])) as ArrayRef;
        let f = Arc::new(Float32Array::from(vec![0.1, 1e10, 3.0, 1.5])) as ArrayRef;
        let mut l = ListBuilder::new(Float64Builder::new());
        for _ in 0..4 {
            l.values().append_value(1e-5);
            l.values().append_value(1.0);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("d", DataType::Float64, true)]),
            vec![Arc::clone(&d)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, f, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":1.0E-5,"f":0.1,"l":[1.0E-5,1.0],"n":{"d":1.0E-5}}"#
        );
        assert_eq!(
            out.value(1),
            r#"{"d":1.23456789E8,"f":1.0E10,"l":[1.0E-5,1.0],"n":{"d":1.23456789E8}}"#
        );
        assert_eq!(
            out.value(2),
            r#"{"d":1.0E10,"f":3.0,"l":[1.0E-5,1.0],"n":{"d":1.0E10}}"#
        );
        assert_eq!(
            out.value(3),
            r#"{"d":1.5,"f":1.5,"l":[1.0E-5,1.0],"n":{"d":1.5}}"#
        );
    }

    #[test]
    fn test_java_big_decimal_text() {
        // Java BigDecimal.toString: plain when scale >= 0 and adjusted >= -6,
        // scientific (signed exponent) otherwise
        let cases: [(&str, i8, &str); 13] = [
            ("0", 10, "0E-10"),  // 0 as decimal(38,10)
            ("10", 8, "1.0E-7"), // 0.0000001 as decimal(10,8)
            ("150", 2, "1.50"),
            ("123", 0, "123"),
            ("1", 6, "0.000001"), // adjusted = -6: still plain
            ("1", 7, "1E-7"),     // adjusted = -7: scientific, single digit
            ("-5", 2, "-0.05"),
            ("0", 2, "0.00"),
            ("0", 0, "0"),
            ("-123", 8, "-0.00000123"), // adjusted = -6: still plain
            ("-123", 9, "-1.23E-7"),    // adjusted = -7: scientific
            ("123", -2, "1.23E+4"),     // negative scale: exponent gets a '+'
            (
                "12345678901234567890123456789012345678",
                10,
                "1234567890123456789012345678.9012345678",
            ),
        ];
        for (unscaled, scale, expected) in cases {
            assert_eq!(
                java_big_decimal_text(unscaled, scale),
                expected,
                "unscaled {unscaled} scale {scale}"
            );
        }
    }

    #[test]
    fn test_decimals_use_java_big_decimal_to_string() {
        // d: Decimal128(38,10) [0, 1.5], e: Decimal128(10,8) [0.0000001, NULL],
        // s: Decimal32(5,2) [1.50], w: Decimal256(50,3) [-0.005],
        // l: List<Decimal128(10,8)> [[0.0000001]]
        let d = Arc::new(
            Decimal128Array::from(vec![Some(0), Some(15_000_000_000)])
                .with_precision_and_scale(38, 10)
                .unwrap(),
        ) as ArrayRef;
        let e = Arc::new(
            Decimal128Array::from(vec![Some(10), None])
                .with_precision_and_scale(10, 8)
                .unwrap(),
        ) as ArrayRef;
        let s = Arc::new(
            Decimal32Array::from(vec![Some(150), Some(150)])
                .with_precision_and_scale(5, 2)
                .unwrap(),
        ) as ArrayRef;
        let w = Arc::new(
            Decimal256Array::from(vec![Some(i256::from(-5)), Some(i256::from(-5))])
                .with_precision_and_scale(50, 3)
                .unwrap(),
        ) as ArrayRef;
        let mut l = ListBuilder::new(
            Decimal128Array::builder(1)
                .with_precision_and_scale(10, 8)
                .unwrap(),
        );
        for _ in 0..2 {
            l.values().append_value(10);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", d.data_type().clone(), true),
            Field::new("e", e.data_type().clone(), true),
            Field::new("s", s.data_type().clone(), true),
            Field::new("w", w.data_type().clone(), true),
            Field::new("l", l.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, e, s, w, l], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":0E-10,"e":1.0E-7,"s":1.50,"w":-0.005,"l":[1.0E-7]}"#
        );
        // 1.5 keeps the declared scale; a NULL decimal is omitted
        assert_eq!(
            out.value(1),
            r#"{"d":1.5000000000,"s":1.50,"w":-0.005,"l":[1.0E-7]}"#
        );
    }
}

/// DataFrame-level tests: to_json used through DataFrame::with_column
/// and through SQL, the way a user of the crate calls it.
#[cfg(test)]
mod dataframe_tests {
    use super::*;

    use datafusion::dataframe;
    use datafusion::prelude::*;
    use datafusion_common::cast::as_string_array;
    use datafusion_common::test_util::batches_to_sort_string;
    use datafusion_common::{assert_batches_sorted_eq, assert_contains};
    use datafusion_functions_nested::expr_fn::make_array;

    use crate::function::json::expr_fn::to_json;

    /// id  foo  bar
    /// 10  1    NULL
    /// 20  2    bak
    /// 30  3    baz
    fn sample_frame() -> Result<DataFrame> {
        dataframe!(
            "id" => [10_i32, 20, 30],
            "foo" => [1_i32, 2, 3],
            "bar" => [None, Some("bak"), Some("baz")],
        )
    }

    /// The same rows as `sample_frame`, as a SQL table expression named `t`.
    const SAMPLE_TABLE_SQL: &str =
        "(VALUES (20, 2, 'bak'), (30, 3, 'baz'), (10, 1, NULL)) AS t(id, foo, bar)";

    #[tokio::test]
    async fn with_column_to_json() -> Result<()> {
        let df = sample_frame()?.with_column("json", to_json(&["foo", "bar"]))?;

        // check that the new column is a nullable string
        let json_field = df.schema().field_with_unqualified_name("json")?;
        assert_eq!(json_field.data_type(), &DataType::Utf8);

        // check that new column added; the NULL `bar` is omitted
        let df_results = df.clone().collect().await?;
        assert_batches_sorted_eq!(
            [
                "+----+-----+-----+-----------------------+",
                "| id | foo | bar | json                  |",
                "+----+-----+-----+-----------------------+",
                "| 10 | 1   |     | {\"foo\":1}             |",
                "| 20 | 2   | bak | {\"foo\":2,\"bar\":\"bak\"} |",
                "| 30 | 3   | baz | {\"foo\":3,\"bar\":\"baz\"} |",
                "+----+-----+-----+-----------------------+",
            ],
            &df_results
        );

        // check that a column with the same name is overwritten, which folds the
        // source columns into a single JSON column
        let df_results_overwrite = df
            .clone()
            .with_column("foo", to_json(&["foo", "bar"]))?
            .collect()
            .await?;
        assert_batches_sorted_eq!(
            [
                "+----+-----------------------+-----+-----------------------+",
                "| id | foo                   | bar | json                  |",
                "+----+-----------------------+-----+-----------------------+",
                "| 10 | {\"foo\":1}             |     | {\"foo\":1}             |",
                "| 20 | {\"foo\":2,\"bar\":\"bak\"} | bak | {\"foo\":2,\"bar\":\"bak\"} |",
                "| 30 | {\"foo\":3,\"bar\":\"baz\"} | baz | {\"foo\":3,\"bar\":\"baz\"} |",
                "+----+-----------------------+-----+-----------------------+",
            ],
            &df_results_overwrite
        );

        // check that an existing struct column can be serialized with col(..),
        // and that it gives the same result as packing the columns by name
        let df_struct_col = df
            .clone()
            .with_column(
                "s",
                named_struct(vec![lit("foo"), col("foo"), lit("bar"), col("bar")]),
            )?
            .with_column("json_from_struct", to_json(col("s")))?
            .select(vec![
                col("id"),
                col("foo"),
                col("bar"),
                col("json_from_struct"),
            ])?
            .with_column_renamed("json_from_struct", "json")?;
        assert_eq!(
            batches_to_sort_string(&df_struct_col.collect().await?),
            batches_to_sort_string(&df_results)
        );

        // check that a non-struct argument is rejected when the plan is built
        let err = df
            .with_column("bad", to_json(col("foo")))
            .unwrap_err()
            .to_string();
        assert_contains!(err, "to_json expects a struct argument");

        // check that the same result is produced through SQL: to_json is a
        // regular scalar UDF, so once registered it works on any struct
        // expression (`with_spark_features()` registers it together with all
        // other Spark functions; here only this one is registered)
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());

        let sql_results = ctx
            .sql(&format!(
                "SELECT id, foo, bar, to_json(named_struct('foo', foo, 'bar', bar)) AS json \
                 FROM {SAMPLE_TABLE_SQL}"
            ))
            .await?
            .collect()
            .await?;
        assert_eq!(
            batches_to_sort_string(&sql_results),
            batches_to_sort_string(&df_results)
        );

        // the SQL path rejects a non-struct argument at plan time as well
        let err = ctx
            .sql(&format!("SELECT to_json(foo) FROM {SAMPLE_TABLE_SQL}"))
            .await
            .unwrap_err()
            .to_string();
        assert_contains!(err, "to_json expects a struct argument");

        Ok(())
    }

    #[tokio::test]
    async fn with_column_to_json_edge_cases() -> Result<()> {
        // NULL fields are omitted from the object (Spark ignoreNullFields
        // default), so a row whose fields are all NULL becomes an empty object
        let df = dataframe!(
            "a" => [Some(1_i32), None],
            "b" => [None, None::<&str>],
        )?
        .with_column("json", to_json(&["a", "b"]))?;
        assert_batches_sorted_eq!(
            [
                "+---+---+---------+",
                "| a | b | json    |",
                "+---+---+---------+",
                "|   |   | {}      |",
                "| 1 |   | {\"a\":1} |",
                "+---+---+---------+",
            ],
            &df.collect().await?
        );

        // a constant struct is evaluated once at plan time (scalar path) and
        // broadcast to every row
        let df = dataframe!("id" => [1_i32, 2])?
            .with_column("json", to_json(named_struct(vec![lit("a"), lit(1)])))?;
        assert_batches_sorted_eq!(
            [
                "+----+---------+",
                "| id | json    |",
                "+----+---------+",
                "| 1  | {\"a\":1} |",
                "| 2  | {\"a\":1} |",
                "+----+---------+",
            ],
            &df.collect().await?
        );
        // nested structs and lists are serialized recursively
        let df = dataframe!("a" => [1_i32, 2])?.with_column(
            "json",
            to_json(named_struct(vec![
                lit("n"),
                named_struct(vec![lit("x"), col("a")]),
                lit("l"),
                make_array(vec![col("a"), lit(9)]),
            ])),
        )?;
        assert_batches_sorted_eq!(
            [
                "+---+-------------------------+",
                "| a | json                    |",
                "+---+-------------------------+",
                "| 1 | {\"n\":{\"x\":1},\"l\":[1,9]} |",
                "| 2 | {\"n\":{\"x\":2},\"l\":[2,9]} |",
                "+---+-------------------------+",
            ],
            &df.collect().await?
        );

        // string values are JSON-escaped; non-ASCII is emitted as-is
        let df = dataframe!("s" => [r#"he said "hi""#, r"a\b", "π"])?
            .with_column("json", to_json(&["s"]))?;
        assert_batches_sorted_eq!(
            [
                "+--------------+------------------------+",
                "| s            | json                   |",
                "+--------------+------------------------+",
                "| a\\b          | {\"s\":\"a\\\\b\"}           |",
                "| he said \"hi\" | {\"s\":\"he said \\\"hi\\\"\"} |",
                "| π            | {\"s\":\"π\"}              |",
                "+--------------+------------------------+",
            ],
            &df.collect().await?
        );

        // a qualified column name is resolved as such, but the JSON key is the
        // bare column name
        let ctx = SessionContext::new();
        let df = ctx
            .sql("SELECT * FROM (VALUES (1, 'x')) AS t(foo, bar)")
            .await?
            .with_column("json", to_json(&["t.foo", "bar"]))?
            .select(vec![col("json")])?;
        assert_batches_sorted_eq!(
            [
                "+---------------------+",
                "| json                |",
                "+---------------------+",
                "| {\"foo\":1,\"bar\":\"x\"} |",
                "+---------------------+",
            ],
            &df.collect().await?
        );

        // an empty input produces no rows but still carries the new column
        let df = sample_frame()?
            .filter(col("foo").gt(lit(100)))?
            .with_column("json", to_json(&["foo"]))?;
        assert_eq!(
            df.schema().field_with_unqualified_name("json")?.data_type(),
            &DataType::Utf8
        );
        assert_eq!(df.count().await?, 0);

        // an empty column list cannot build a struct and fails when planning
        // (rejected by the named_struct signature check)
        let err = dataframe!("a" => [1_i32])?
            .with_column("json", to_json(&[] as &[&str]))
            .unwrap_err()
            .to_string();
        assert_contains!(err, "'named_struct' does not support zero arguments");

        Ok(())
    }

    /// Evaluates `SELECT to_json(<expr>)` for each `(expr, expected)` pair and
    /// checks the JSON string.
    ///
    /// The `datafusion` dev-dependency is built without the
    /// `nested_expressions` feature, so the nested functions used to build
    /// test values (`make_array`, `map`, ...) are registered here explicitly.
    async fn assert_sql_json(cases: &[(&str, &str)]) -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());
        for udf in datafusion_functions_nested::all_default_nested_functions() {
            ctx.register_udf((*udf).clone());
        }

        for (expr, expected) in cases {
            let batches = ctx
                .sql(&format!("SELECT to_json({expr}) AS j"))
                .await?
                .collect()
                .await?;
            let j = as_string_array(batches[0].column(0))?;
            assert_eq!(j.value(0), *expected, "expression: {expr}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn to_json_keeps_null_map_values_sql() -> Result<()> {
        assert_sql_json(&[
            // NULL map value is kept
            (
                "named_struct('m', map(make_array('k'), make_array(CAST(NULL AS INT))))",
                r#"{"m":{"k":null}}"#,
            ),
            // nested map inside a map value
            (
                "named_struct('m', map(make_array('o'), make_array(map(make_array('k'), make_array(CAST(NULL AS INT))))))",
                r#"{"m":{"o":{"k":null}}}"#,
            ),
            // a struct inside a map value still omits its NULL fields
            (
                "named_struct('m', map(make_array('k'), make_array(named_struct('a', 1, 'b', CAST(NULL AS INT)))))",
                r#"{"m":{"k":{"a":1}}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_encodes_binary_as_base64_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('b', arrow_cast('abc', 'Binary'))",
                r#"{"b":"YWJj"}"#,
            ),
            (
                "named_struct('b', arrow_cast('abc', 'LargeBinary'))",
                r#"{"b":"YWJj"}"#,
            ),
            (
                "named_struct('b', arrow_cast('abc', 'BinaryView'))",
                r#"{"b":"YWJj"}"#,
            ),
            // nested in a list, in a struct and in a map value
            (
                "named_struct('l', make_array(arrow_cast('abc', 'Binary')))",
                r#"{"l":["YWJj"]}"#,
            ),
            (
                "named_struct('n', named_struct('b', arrow_cast('abc', 'Binary')))",
                r#"{"n":{"b":"YWJj"}}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(arrow_cast('abc', 'Binary'))))",
                r#"{"m":{"k":"YWJj"}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_non_finite_floats_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('x', CAST('NaN' AS DOUBLE))",
                r#"{"x":"NaN"}"#,
            ),
            (
                "named_struct('x', CAST('Infinity' AS DOUBLE))",
                r#"{"x":"Infinity"}"#,
            ),
            (
                "named_struct('x', CAST('-Infinity' AS DOUBLE))",
                r#"{"x":"-Infinity"}"#,
            ),
            (
                "named_struct('x', arrow_cast('NaN', 'Float32'))",
                r#"{"x":"NaN"}"#,
            ),
            // finite values are unchanged
            ("named_struct('x', 1.5)", r#"{"x":1.5}"#),
            // nested in a list, a struct and a map value
            (
                "named_struct('l', make_array(CAST('NaN' AS DOUBLE), 1.5))",
                r#"{"l":["NaN",1.5]}"#,
            ),
            (
                "named_struct('n', named_struct('x', CAST('Infinity' AS DOUBLE)))",
                r#"{"n":{"x":"Infinity"}}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(CAST('NaN' AS DOUBLE))))",
                r#"{"m":{"k":"NaN"}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_stringifies_map_keys_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('m', map(make_array(1), make_array('v')))",
                r#"{"m":{"1":"v"}}"#,
            ),
            (
                "named_struct('m', map(make_array(true), make_array('v')))",
                r#"{"m":{"true":"v"}}"#,
            ),
            (
                "named_struct('m', map(make_array(1.5), make_array('v')))",
                r#"{"m":{"1.5":"v"}}"#,
            ),
            // string keys are unchanged
            (
                "named_struct('m', map(make_array('k'), make_array('v')))",
                r#"{"m":{"k":"v"}}"#,
            ),
            // recursively: an int-keyed map inside a list, a struct and a map value
            (
                "named_struct('l', make_array(map(make_array(1), make_array('v'))))",
                r#"{"l":[{"1":"v"}]}"#,
            ),
            (
                "named_struct('n', named_struct('m', map(make_array(1), make_array('v'))))",
                r#"{"n":{"m":{"1":"v"}}}"#,
            ),
            (
                "named_struct('m', map(make_array('o'), make_array(map(make_array(2), make_array('v')))))",
                r#"{"m":{"o":{"2":"v"}}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_timestamps_sql() -> Result<()> {
        assert_sql_json(&[
            // TIMESTAMP (instant): milliseconds, session time zone offset (`Z` for UTC)
            (
                "named_struct('t', arrow_cast('2024-01-01T01:02:03.123456Z', 'Timestamp(Microsecond, Some(\"UTC\"))'))",
                r#"{"t":"2024-01-01T01:02:03.123Z"}"#,
            ),
            // TIMESTAMP_NTZ: no offset, no conversion
            (
                "named_struct('t', TIMESTAMP '2024-01-01 01:02:03.123456')",
                r#"{"t":"2024-01-01T01:02:03.123"}"#,
            ),
        ])
        .await?;

        // the session time zone decides how an instant is written
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());
        ctx.sql("SET datafusion.execution.time_zone = '+02:00'")
            .await?
            .collect()
            .await?;
        let batches = ctx
            .sql(
                "SELECT to_json(named_struct('t', arrow_cast('2024-01-01T01:02:03.123456Z', \
                 'Timestamp(Microsecond, Some(\"UTC\"))'))) AS j",
            )
            .await?
            .collect()
            .await?;
        let j = as_string_array(batches[0].column(0))?;
        assert_eq!(j.value(0), r#"{"t":"2024-01-01T03:02:03.123+02:00"}"#);
        Ok(())
    }

    #[tokio::test]
    async fn to_json_floats_use_java_to_string_sql() -> Result<()> {
        assert_sql_json(&[
            ("named_struct('x', 1e-5)", r#"{"x":1.0E-5}"#),
            (
                "named_struct('x', CAST(123456789 AS DOUBLE))",
                r#"{"x":1.23456789E8}"#,
            ),
            ("named_struct('x', 1e10)", r#"{"x":1.0E10}"#),
            ("named_struct('x', 1.0)", r#"{"x":1.0}"#),
            ("named_struct('x', 0.001)", r#"{"x":0.001}"#),
            (
                "named_struct('x', arrow_cast(0.1, 'Float32'))",
                r#"{"x":0.1}"#,
            ),
            // nested in a list and a map value
            (
                "named_struct('l', make_array(1e-5, 1.0))",
                r#"{"l":[1.0E-5,1.0]}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(1e10)))",
                r#"{"m":{"k":1.0E10}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_decimals_use_java_big_decimal_sql() -> Result<()> {
        assert_sql_json(&[
            ("named_struct('d', CAST(0 AS DECIMAL(38, 10)))", r#"{"d":0E-10}"#),
            (
                "named_struct('d', CAST(0.0000001 AS DECIMAL(10, 8)))",
                r#"{"d":1.0E-7}"#,
            ),
            ("named_struct('d', CAST(1.5 AS DECIMAL(10, 2)))", r#"{"d":1.50}"#),
            ("named_struct('d', CAST(123 AS DECIMAL(10, 0)))", r#"{"d":123}"#),
            (
                "named_struct('d', CAST(-0.05 AS DECIMAL(10, 2)))",
                r#"{"d":-0.05}"#,
            ),
            // nested in a list and a map value
            (
                "named_struct('l', make_array(CAST(0 AS DECIMAL(38, 10))))",
                r#"{"l":[0E-10]}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(CAST(1.5 AS DECIMAL(10, 2)))))",
                r#"{"m":{"k":1.50}}"#,
            ),
        ])
        .await
    }
}
