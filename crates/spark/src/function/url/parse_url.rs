// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, LargeStringArray, StringArray, StringArrayType,
    StringViewArray, new_null_array,
};
use arrow::datatypes::DataType;
use datafusion_common::cast::{
    as_large_string_array, as_string_array, as_string_view_array,
};
use datafusion_common::{DataFusionError, Result, exec_datafusion_err, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;
use url::{ParseError, Url};

use crate::EvalMode;

#[user_doc(
    doc_section(label = "URL Functions"),
    description = "Extracts a part from a URL string, such as the host, path, query, or authority.         Ports above 65535 are accepted to match Apache Spark (`java.net.URI`) behavior, differing from         the standard 16-bit port restriction. If the port is not a valid signed 32-bit integer,         `HOST` returns NULL while other valid components are preserved.",
    syntax_example = "parse_url(url, partToExtract[, key])",
    sql_example = r#"```sql
> SELECT parse_url('http://spark.apache.org:99999/path?query=1', 'HOST');
spark.apache.org
> SELECT parse_url('http://spark.apache.org:99999/path?query=1', 'AUTHORITY');
spark.apache.org:99999
> SELECT parse_url('http://spark.apache.org/path?query=1', 'QUERY', 'query');
1
```"#,
    argument(name = "url", description = "URL string expression to parse."),
    argument(
        name = "partToExtract",
        description = "The component to extract: 'HOST', 'PATH', 'QUERY', 'REF', 'PROTOCOL', 'FILE', 'AUTHORITY', or 'USERINFO'."
    ),
    argument(
        name = "key",
        description = "Optional key name when partToExtract is 'QUERY'. Extracts the specific parameter value."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ParseUrl {
    signature: Signature,
}

impl Default for ParseUrl {
    fn default() -> Self {
        Self::new()
    }
}

impl ParseUrl {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::String(2), TypeSignature::String(3)],
                Volatility::Immutable,
            ),
        }
    }
    /// Parses a URL and extracts the specified component.
    ///
    /// This function takes a URL string and extracts different parts of it based on the
    /// `part` parameter. For query parameters, an optional `key` can be specified to
    /// extract a specific query parameter value.
    ///
    /// # Arguments
    ///
    /// * `value` - The URL string to parse
    /// * `part` - The component of the URL to extract. Valid values are:
    ///   - `"HOST"` - The hostname (e.g., "example.com")
    ///   - `"PATH"` - The path portion (e.g., "/path/to/resource")
    ///   - `"QUERY"` - The query string or a specific query parameter
    ///   - `"REF"` - The fragment/anchor (the part after #)
    ///   - `"PROTOCOL"` - The URL scheme (e.g., "https", "http")
    ///   - `"FILE"` - The path with query string (e.g., "/path?query=value")
    ///   - `"AUTHORITY"` - The authority component (host:port)
    ///   - `"USERINFO"` - The user information (username:password)
    /// * `key` - Optional parameter used only with `"QUERY"`. When provided, extracts
    ///   the value of the specific query parameter with this key name.
    ///
    /// Ports above 65535 are accepted to match Apache Spark (`java.net.URI`) behavior,
    /// intentionally differing from the Rust `url` crate's 16-bit port restriction.
    /// For ports that are not valid signed 32-bit integers (e.g. non-numeric ports or
    /// ports > `i32::MAX`), `HOST` returns `None` (NULL) while other components (such as
    /// `AUTHORITY`, `PATH`, `QUERY`, `REF`, `PROTOCOL`) are preserved.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(String))` - The extracted URL component as a string
    /// * `Ok(None)` - If the requested component doesn't exist
    /// * `Err(DataFusionError)` - If the URL is malformed and cannot be parsed
    fn parse(value: &str, part: &str, key: Option<&str>) -> Result<Option<String>> {
        let url: std::result::Result<Url, ParseError> = Url::parse(value);
        if url == Err(ParseError::RelativeUrlWithoutBase) {
            return if !value.contains("://") {
                // Schemeless URLs are treated as relative URIs (like java.net.URI).
                // Manually parse path, query, and fragment components.
                let (without_fragment, fragment) = match value.split_once('#') {
                    Some((before, frag)) => (before, Some(frag)),
                    None => (value, None),
                };
                let (path, query) = match without_fragment.split_once('?') {
                    Some((p, q)) => (p, Some(q)),
                    None => (without_fragment, None),
                };
                Ok(match part {
                    "PATH" => Some(path.to_string()),
                    "QUERY" => match key {
                        None => query.map(String::from),
                        Some(key) => Self::query_value(query, key).map(String::from),
                    },
                    "REF" => fragment.map(String::from),
                    "FILE" => {
                        // FILE = path + query (without fragment)
                        Some(without_fragment.to_string())
                    }
                    // HOST, PROTOCOL, AUTHORITY, USERINFO → NULL
                    _ => None,
                })
            } else {
                Err(Self::invalid_url_err(value))
            };
        }
        match url {
            Ok(url) => Ok(Self::extract_url_part(value, &url, part, key)),
            Err(ParseError::InvalidPort) => {
                if let Some((temp_url, raw_authority, is_spark_host)) =
                    Self::normalize_invalid_port_url(value)
                {
                    if part == "AUTHORITY" {
                        return Ok(Some(raw_authority));
                    }
                    if part == "HOST" {
                        return Ok(if is_spark_host {
                            temp_url.host_str().map(String::from)
                        } else {
                            None
                        });
                    }
                    return Ok(Self::extract_url_part(value, &temp_url, part, key));
                }
                Err(Self::invalid_url_err(value))
            }
            Err(_) => Err(Self::invalid_url_err(value)),
        }
    }

    fn extract_url_part(
        value: &str,
        url: &Url,
        part: &str,
        key: Option<&str>,
    ) -> Option<String> {
        match part {
            "HOST" => url.host_str().map(String::from),
            "PATH" => {
                let path = Self::path(value, url);
                Some(path.to_string())
            }
            "QUERY" => match key {
                None => url.query().map(String::from),
                Some(key) => Self::query_value(url.query(), key).map(String::from),
            },
            "REF" => url.fragment().map(String::from),
            "PROTOCOL" => Some(url.scheme().to_string()),
            "FILE" => {
                let path = Self::path(value, url);
                match url.query() {
                    Some(query) => Some(format!("{path}?{query}")),
                    None => Some(path.to_string()),
                }
            }
            "AUTHORITY" => Some(url.authority().to_string()),
            "USERINFO" => {
                let username = url.username();
                if username.is_empty() {
                    return None;
                }
                match url.password() {
                    Some(password) => Some(format!("{username}:{password}")),
                    None => Some(username.to_string()),
                }
            }
            _ => None,
        }
    }

    /// When `Url::parse` returns `ParseError::InvalidPort`, attempt to isolate the
    /// authority component and substitute a valid port (`:80`) so that the standard
    /// parser can extract components with Spark compatibility (issue #365).
    ///
    /// Returns `Some((temp_url, raw_authority, is_spark_host))` if the URL
    /// structure conforms to an authority with a port specification, or `None` if
    /// the URL is structurally malformed.
    fn normalize_invalid_port_url(value: &str) -> Option<(Url, String, bool)> {
        // Strip scheme (e.g. "http://")
        let (scheme, rest) = value.split_once("://")?;
        if scheme.is_empty() {
            return None;
        }

        // Find end of authority (first '/', '?', '#', or end of string)
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let raw_authority = &rest[..auth_end];
        let path_and_beyond = &rest[auth_end..];

        // Java URI throws URISyntaxException for any whitespace or backslash in authority
        if raw_authority
            .chars()
            .any(|c| c.is_whitespace() || c == '\\')
        {
            return None;
        }

        // Separate userinfo if present
        let (userinfo_prefix, host_port) = match raw_authority.rsplit_once('@') {
            Some((userinfo, host_port)) => {
                if userinfo.is_empty() {
                    return None;
                }
                (format!("{userinfo}@"), host_port)
            }
            None => (String::new(), raw_authority),
        };

        // Find the port separator ':' - for IPv6 [::1]:port, find ':' after ']'
        let colon_pos = if host_port.starts_with('[') {
            let close_bracket = host_port.find(']')?;
            close_bracket + host_port[close_bracket..].find(':')?
        } else {
            host_port.rfind(':')?
        };

        let host_str = &host_port[..colon_pos];
        let port_str = &host_port[colon_pos + 1..];

        // Port cannot be empty
        if port_str.is_empty() || host_str.is_empty() {
            return None;
        }

        // For bracketed IPv6, non-numeric port is invalid in java.net.URI
        if host_port.starts_with('[') && !port_str.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }

        // Spark (via java.net.URI) extracts the host if the port can be parsed as a 32-bit
        // signed integer (e.g. 65536, 99999, 1000000). For non-numeric ports or ports > i32::MAX,
        // Spark treats the host as null, but still extracts authority, path, query, and ref.
        let is_spark_host = port_str.chars().all(|c| c.is_ascii_digit())
            && port_str.parse::<i32>().is_ok();

        // Construct a normalized URL with port omitted: scheme://[userinfo@]host[:path_and_beyond]
        let normalized =
            format!("{scheme}://{userinfo_prefix}{host_str}{path_and_beyond}");
        let temp_url = Url::parse(&normalized).ok()?;

        Some((temp_url, raw_authority.to_string(), is_spark_host))
    }

    /// The same error for every URL that can't be parsed
    fn invalid_url_err(value: &str) -> DataFusionError {
        exec_datafusion_err!(
            "The url is invalid: {value}. Use `try_parse_url` to tolerate invalid URL and return NULL instead. SQLSTATE: 22P02"
        )
    }

    fn path<'a>(value: &str, url: &'a Url) -> &'a str {
        let path = url.path();
        if path == "/" && Self::absolute_url_has_empty_path(value) {
            ""
        } else {
            path
        }
    }

    fn absolute_url_has_empty_path(value: &str) -> bool {
        let Some(authority_start) = value.find("://").map(|index| index + 3) else {
            return false;
        };
        let after_authority = &value[authority_start..];
        match after_authority.find(['/', '?', '#']) {
            None => true,
            Some(index) => matches!(after_authority.as_bytes()[index], b'?' | b'#'),
        }
    }

    fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
        query.and_then(|query| {
            query
                .split('&')
                .filter_map(|pair| pair.split_once('='))
                .find(|(query_key, _)| *query_key == key)
                .map(|(_, value)| value)
        })
    }
}

impl ScalarUDFImpl for ParseUrl {
    fn name(&self) -> &str {
        "parse_url"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let eval_mode = EvalMode::from_config(&args.config_options);
        make_scalar_function(
            move |arrays: &[ArrayRef]| spark_parse_url(arrays, eval_mode),
            vec![],
        )(&args.args)
    }
}

/// Core implementation of URL parsing function.
///
/// # Arguments
///
/// * `args` - A slice of ArrayRef containing the input arrays:
///   - `args[0]` - URL array: The URLs to parse
///   - `args[1]` - Part array: The URL components to extract (HOST, PATH, QUERY, etc.)
///   - `args[2]` - Key array (optional): For QUERY part, the specific parameter names to extract
///
/// # Return Value
///
/// Returns `Result<ArrayRef>` containing:
/// - A string array with extracted URL components
/// - `None` values where extraction failed or component doesn't exist
/// - The output array type (StringArray or LargeStringArray) is determined by input types
///
/// A URL that cannot be parsed produces `NULL` in [`EvalMode::Legacy`] (Spark's
/// `parse_url` with ANSI mode off) and [`EvalMode::Try`] (`try_parse_url`), and
/// an error in [`EvalMode::Ansi`]. An invalid number of arguments or a
/// non-string argument is an error in every mode.
pub fn spark_parse_url(args: &[ArrayRef], eval_mode: EvalMode) -> Result<ArrayRef> {
    spark_handled_parse_url(args, |x| match x {
        Err(_) if eval_mode != EvalMode::Ansi => Ok(None),
        result => result,
    })
}

pub fn spark_handled_parse_url(
    args: &[ArrayRef],
    handler_err: impl Fn(Result<Option<String>>) -> Result<Option<String>>,
) -> Result<ArrayRef> {
    if args.len() < 2 || args.len() > 3 {
        return exec_err!(
            "{} expects 2 or 3 arguments, but got {}",
            "`parse_url`",
            args.len()
        );
    }
    // Required arguments
    let url = &args[0];
    let part = &args[1];

    if args.len() == 3 {
        // In this case, the 'key' argument is passed
        let key = &args[2];

        match (url.data_type(), part.data_type(), key.data_type()) {
            (DataType::Utf8, DataType::Utf8, DataType::Utf8) => {
                process_parse_url::<_, _, _, StringArray>(
                    as_string_array(url)?,
                    as_string_array(part)?,
                    as_string_array(key)?,
                    handler_err,
                    true,
                )
            }
            (DataType::Utf8View, DataType::Utf8View, DataType::Utf8View) => {
                process_parse_url::<_, _, _, StringViewArray>(
                    as_string_view_array(url)?,
                    as_string_view_array(part)?,
                    as_string_view_array(key)?,
                    handler_err,
                    true,
                )
            }
            (DataType::LargeUtf8, DataType::LargeUtf8, DataType::LargeUtf8) => {
                process_parse_url::<_, _, _, LargeStringArray>(
                    as_large_string_array(url)?,
                    as_large_string_array(part)?,
                    as_large_string_array(key)?,
                    handler_err,
                    true,
                )
            }
            _ => exec_err!(
                "`parse_url` expects STRING arguments, got ({}, {}, {})",
                url.data_type(),
                part.data_type(),
                key.data_type()
            ),
        }
    } else {
        // The 'key' argument is omitted, assume all values are null.
        // `new_null_array` allocates the null array outright, rather than
        // appending one null per row through a builder.
        let key_array = new_null_array(&DataType::Utf8, args[0].len());
        let key = key_array.as_string::<i32>();

        match (url.data_type(), part.data_type()) {
            (DataType::Utf8, DataType::Utf8) => {
                process_parse_url::<_, _, _, StringArray>(
                    as_string_array(url)?,
                    as_string_array(part)?,
                    key,
                    handler_err,
                    false,
                )
            }
            (DataType::Utf8View, DataType::Utf8View) => {
                process_parse_url::<_, _, _, StringViewArray>(
                    as_string_view_array(url)?,
                    as_string_view_array(part)?,
                    key,
                    handler_err,
                    false,
                )
            }
            (DataType::LargeUtf8, DataType::LargeUtf8) => {
                process_parse_url::<_, _, _, LargeStringArray>(
                    as_large_string_array(url)?,
                    as_large_string_array(part)?,
                    key,
                    handler_err,
                    false,
                )
            }
            _ => exec_err!(
                "`parse_url` expects STRING arguments, got ({}, {})",
                url.data_type(),
                part.data_type()
            ),
        }
    }
}

fn process_parse_url<'a, A, B, C, T>(
    url_array: &'a A,
    part_array: &'a B,
    key_array: &'a C,
    handle: impl Fn(Result<Option<String>>) -> Result<Option<String>>,
    has_key_arg: bool,
) -> Result<ArrayRef>
where
    &'a A: StringArrayType<'a>,
    &'a B: StringArrayType<'a>,
    &'a C: StringArrayType<'a>,
    T: Array + FromIterator<Option<String>> + 'static,
{
    url_array
        .iter()
        .zip(part_array.iter())
        .zip(key_array.iter())
        .map(|((url, part), key)| {
            // Spark returns NULL when the third argument is explicitly NULL
            if has_key_arg && key.is_none() {
                return Ok(None);
            }
            if let (Some(url), Some(part)) = (url, part) {
                handle(ParseUrl::parse(url, part, key))
            } else {
                Ok(None)
            }
        })
        .collect::<Result<T>>()
        .map(|array| Arc::new(array) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use std::array::from_ref;

    fn sa(vals: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(vals.to_vec())) as ArrayRef
    }

    #[test]
    fn test_parse_host() -> Result<()> {
        let got = ParseUrl::parse("https://example.com/a?x=1", "HOST", None)?;
        assert_eq!(got, Some("example.com".to_string()));
        Ok(())
    }

    #[test]
    fn test_parse_query_no_key_vs_with_key() -> Result<()> {
        let got_all = ParseUrl::parse("https://ex.com/p?a=1&b=2", "QUERY", None)?;
        assert_eq!(got_all, Some("a=1&b=2".to_string()));

        let got_a = ParseUrl::parse("https://ex.com/p?a=1&b=2", "QUERY", Some("a"))?;
        assert_eq!(got_a, Some("1".to_string()));

        let got_c = ParseUrl::parse("https://ex.com/p?a=1&b=2", "QUERY", Some("c"))?;
        assert_eq!(got_c, None);
        Ok(())
    }

    #[test]
    fn test_parse_ref_protocol_userinfo_file_authority() -> Result<()> {
        let url = "ftp://user:pwd@ftp.example.com:21/files?x=1#frag";
        assert_eq!(ParseUrl::parse(url, "REF", None)?, Some("frag".to_string()));
        assert_eq!(
            ParseUrl::parse(url, "PROTOCOL", None)?,
            Some("ftp".to_string())
        );
        assert_eq!(
            ParseUrl::parse(url, "USERINFO", None)?,
            Some("user:pwd".to_string())
        );
        assert_eq!(
            ParseUrl::parse(url, "FILE", None)?,
            Some("/files?x=1".to_string())
        );
        assert_eq!(
            ParseUrl::parse(url, "AUTHORITY", None)?,
            Some("user:pwd@ftp.example.com".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_parse_path_empty_vs_root() -> Result<()> {
        assert_eq!(
            ParseUrl::parse("https://example.com", "PATH", None)?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse("https://example.com/", "PATH", None)?,
            Some("/".to_string())
        );
        assert_eq!(
            ParseUrl::parse("https://ex.com/dir%20/pa%20th.HTML", "PATH", None)?,
            Some("/dir%20/pa%20th.HTML".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_parse_query_key_is_raw() -> Result<()> {
        let url = "https://use%20r:pas%20s@example.com/dir%20/pa%20th.HTML?query=x%20y&q2=2#Ref%20two";
        assert_eq!(
            ParseUrl::parse(url, "QUERY", None)?,
            Some("query=x%20y&q2=2".to_string())
        );
        assert_eq!(
            ParseUrl::parse(url, "QUERY", Some("query"))?,
            Some("x%20y".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com?key=", "QUERY", Some("key"))?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com?keyonly", "QUERY", Some("keyonly"))?,
            None
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com?a=1&a=2", "QUERY", Some("a"))?,
            Some("1".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com?a%20b=1", "QUERY", Some("a b"))?,
            None
        );
        Ok(())
    }

    #[test]
    fn test_parse_empty_path_file() -> Result<()> {
        assert_eq!(ParseUrl::parse("", "PATH", None)?, Some("".to_string()));
        assert_eq!(
            ParseUrl::parse("http://example.com", "FILE", None)?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://example.com?foo=bar", "FILE", None)?,
            Some("?foo=bar".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://example.com#fragment", "FILE", None)?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://example.com/?foo=bar", "FILE", None)?,
            Some("/?foo=bar".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com/?", "FILE", None)?,
            Some("/?".to_string())
        );
        assert_eq!(
            ParseUrl::parse("http://ex.com?", "FILE", None)?,
            Some("?".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_parse_schemeless_url() -> Result<()> {
        // Spark's java.net.URI treats schemeless strings as relative URIs.
        // Simple schemeless string: no query, no fragment.
        assert_eq!(
            ParseUrl::parse("notaurl", "PATH", None)?,
            Some("notaurl".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl", "FILE", None)?,
            Some("notaurl".to_string())
        );
        assert_eq!(ParseUrl::parse("notaurl", "HOST", None)?, None);
        assert_eq!(ParseUrl::parse("notaurl", "PROTOCOL", None)?, None);
        assert_eq!(ParseUrl::parse("notaurl", "QUERY", None)?, None);
        assert_eq!(ParseUrl::parse("notaurl", "REF", None)?, None);
        assert_eq!(ParseUrl::parse("notaurl", "AUTHORITY", None)?, None);
        assert_eq!(ParseUrl::parse("notaurl", "USERINFO", None)?, None);

        // Schemeless URL with query string
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "PATH", None)?,
            Some("notaurl".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "FILE", None)?,
            Some("notaurl?key=value".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "QUERY", None)?,
            Some("key=value".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "QUERY", Some("key"))?,
            Some("value".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "QUERY", Some("missing"))?,
            None
        );
        assert_eq!(ParseUrl::parse("notaurl?key=value", "HOST", None)?, None);
        assert_eq!(
            ParseUrl::parse("notaurl?key=value", "PROTOCOL", None)?,
            None
        );

        // Schemeless URL with fragment
        assert_eq!(
            ParseUrl::parse("notaurl#reference", "REF", None)?,
            Some("reference".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl#reference", "PATH", None)?,
            Some("notaurl".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl#reference", "FILE", None)?,
            Some("notaurl".to_string())
        );

        // Schemeless URL with both query and fragment
        assert_eq!(
            ParseUrl::parse("notaurl?a=1&b=2#frag", "PATH", None)?,
            Some("notaurl".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?a=1&b=2#frag", "QUERY", None)?,
            Some("a=1&b=2".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?a=1&b=2#frag", "QUERY", Some("b"))?,
            Some("2".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?a=1&b=2#frag", "REF", None)?,
            Some("frag".to_string())
        );
        assert_eq!(
            ParseUrl::parse("notaurl?a=1&b=2#frag", "FILE", None)?,
            Some("notaurl?a=1&b=2".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_spark_utf8_two_args() -> Result<()> {
        let urls = sa(&[Some("https://example.com/a?x=1"), Some("https://ex.com/")]);
        let parts = sa(&[Some("HOST"), Some("PATH")]);

        let out = spark_handled_parse_url(&[urls, parts], |x| x)?;
        let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(out_sa.len(), 2);
        assert_eq!(out_sa.value(0), "example.com");
        assert_eq!(out_sa.value(1), "/");
        Ok(())
    }

    #[test]
    fn test_spark_utf8_three_args_query_key() -> Result<()> {
        let urls = sa(&[
            Some("https://example.com/a?x=1&y=2"),
            Some("https://ex.com/?a=1"),
        ]);
        let parts = sa(&[Some("QUERY"), Some("QUERY")]);
        let keys = sa(&[Some("y"), Some("b")]);

        let out = spark_handled_parse_url(&[urls, parts, keys], |x| x)?;
        let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(out_sa.len(), 2);
        assert_eq!(out_sa.value(0), "2");
        assert!(out_sa.is_null(1));
        Ok(())
    }

    #[test]
    fn test_spark_userinfo_and_nulls() -> Result<()> {
        let urls = sa(&[
            Some("ftp://user:pwd@ftp.example.com:21/files"),
            Some("https://example.com"),
            None,
        ]);
        let parts = sa(&[Some("USERINFO"), Some("USERINFO"), Some("USERINFO")]);

        let out = spark_handled_parse_url(&[urls, parts], |x| x)?;
        let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(out_sa.len(), 3);
        assert_eq!(out_sa.value(0), "user:pwd");
        assert!(out_sa.is_null(1));
        assert!(out_sa.is_null(2));
        Ok(())
    }

    #[test]
    fn test_invalid_arg_count() {
        let urls = sa(&[Some("https://example.com")]);
        let err = spark_handled_parse_url(from_ref(&urls), |x| x).unwrap_err();
        assert!(format!("{err}").contains("expects 2 or 3 arguments"));

        let parts = sa(&[Some("HOST")]);
        let keys = sa(&[Some("x")]);
        let err =
            spark_handled_parse_url(&[urls, parts, keys, sa(&[Some("extra")])], |x| x)
                .unwrap_err();
        assert!(format!("{err}").contains("expects 2 or 3 arguments"));
    }

    #[test]
    fn test_non_string_types_error() {
        let urls = sa(&[Some("https://example.com")]);
        let bad_part = Arc::new(Int32Array::from(vec![1])) as ArrayRef;

        let err = spark_handled_parse_url(&[urls, bad_part], |x| x).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expects STRING arguments"));
    }

    #[test]
    fn test_malformed_url_null_unless_ansi() -> Result<()> {
        let urls = sa(&[
            Some("https://example.com/a"),
            Some("inva lid://example.com"),
        ]);
        let parts = sa(&[Some("HOST"), Some("HOST")]);

        for eval_mode in [EvalMode::Legacy, EvalMode::Try] {
            let out =
                spark_parse_url(&[Arc::clone(&urls), Arc::clone(&parts)], eval_mode)?;
            let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(out_sa.value(0), "example.com");
            assert!(out_sa.is_null(1));
        }

        let err =
            spark_parse_url(&[Arc::clone(&urls), Arc::clone(&parts)], EvalMode::Ansi)
                .unwrap_err();
        assert!(format!("{err}").contains("The url is invalid"));

        // Argument errors are not turned into NULL in any mode
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi, EvalMode::Try] {
            let err = spark_parse_url(from_ref(&urls), eval_mode).unwrap_err();
            assert!(format!("{err}").contains("expects 2 or 3 arguments"));
        }
        Ok(())
    }

    #[test]
    fn test_parse_url_overflow_port_components() -> Result<()> {
        let u = "http://spark.apache.org:99999/path?query=1#ref";
        assert_eq!(
            ParseUrl::parse(u, "HOST", None)?,
            Some("spark.apache.org".to_string())
        );
        assert_eq!(ParseUrl::parse(u, "PATH", None)?, Some("/path".to_string()));
        assert_eq!(
            ParseUrl::parse(u, "QUERY", None)?,
            Some("query=1".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u, "QUERY", Some("query"))?,
            Some("1".to_string())
        );
        assert_eq!(ParseUrl::parse(u, "REF", None)?, Some("ref".to_string()));
        assert_eq!(
            ParseUrl::parse(u, "PROTOCOL", None)?,
            Some("http".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u, "FILE", None)?,
            Some("/path?query=1".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u, "AUTHORITY", None)?,
            Some("spark.apache.org:99999".to_string())
        );
        assert_eq!(ParseUrl::parse(u, "USERINFO", None)?, None);

        // Port exceeding signed 32-bit integer (> 2147483647): HOST is null, other parts preserved
        let u_huge = "http://spark.apache.org:9999999999/path?query=1#ref";
        assert_eq!(ParseUrl::parse(u_huge, "HOST", None)?, None);
        assert_eq!(
            ParseUrl::parse(u_huge, "AUTHORITY", None)?,
            Some("spark.apache.org:9999999999".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_huge, "PATH", None)?,
            Some("/path".to_string())
        );

        // Non-numeric port (:abc): Spark treats host as null, but extracts authority & path
        let u_abc = "http://spark.apache.org:abc/path?query=1#ref";
        assert_eq!(ParseUrl::parse(u_abc, "HOST", None)?, None);
        assert_eq!(
            ParseUrl::parse(u_abc, "AUTHORITY", None)?,
            Some("spark.apache.org:abc".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_abc, "PATH", None)?,
            Some("/path".to_string())
        );

        // Non-numeric port after bracketed IPv6 is rejected (java.net.URI throws URISyntaxException)
        assert!(ParseUrl::parse("http://[::1]:abc/path", "HOST", None).is_err());
        assert!(ParseUrl::parse("http://[::1]:abc/path", "AUTHORITY", None).is_err());
        assert!(ParseUrl::parse("http://[::1]:abc/path", "PATH", None).is_err());

        // Backslash in authority is rejected (java.net.URI throws URISyntaxException)
        assert!(
            ParseUrl::parse(r"http://spark.apache.org:99999\path", "HOST", None).is_err()
        );
        assert!(
            ParseUrl::parse(r"http://spark.apache.org:99999\path", "AUTHORITY", None)
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn test_parse_url_structures_with_overflow_port() -> Result<()> {
        // IPv4 host with port 99999
        let u_ipv4 = "http://127.0.0.1:99999/path";
        assert_eq!(
            ParseUrl::parse(u_ipv4, "HOST", None)?,
            Some("127.0.0.1".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_ipv4, "AUTHORITY", None)?,
            Some("127.0.0.1:99999".to_string())
        );

        // IPv6 host with port 99999
        let u_ipv6 = "http://[::1]:99999/path";
        assert_eq!(
            ParseUrl::parse(u_ipv6, "HOST", None)?,
            Some("[::1]".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_ipv6, "AUTHORITY", None)?,
            Some("[::1]:99999".to_string())
        );

        // Userinfo with port 99999
        let u_user = "https://user:pass@spark.apache.org:99999/path";
        assert_eq!(
            ParseUrl::parse(u_user, "HOST", None)?,
            Some("spark.apache.org".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_user, "USERINFO", None)?,
            Some("user:pass".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_user, "AUTHORITY", None)?,
            Some("user:pass@spark.apache.org:99999".to_string())
        );

        // Encoded characters in path, query, and ref
        let u_enc = "http://spark.apache.org:99999/pa%20th?q=hello%20world#frag%20ment";
        assert_eq!(
            ParseUrl::parse(u_enc, "HOST", None)?,
            Some("spark.apache.org".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_enc, "PATH", None)?,
            Some("/pa%20th".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_enc, "QUERY", None)?,
            Some("q=hello%20world".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_enc, "REF", None)?,
            Some("frag%20ment".to_string())
        );

        // Empty path with port 99999
        let u_empty_path = "http://spark.apache.org:99999";
        assert_eq!(
            ParseUrl::parse(u_empty_path, "HOST", None)?,
            Some("spark.apache.org".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_empty_path, "PATH", None)?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_empty_path, "FILE", None)?,
            Some("".to_string())
        );

        // Query only (empty path) with port 99999
        let u_query_only = "http://spark.apache.org:99999?query=1";
        assert_eq!(
            ParseUrl::parse(u_query_only, "HOST", None)?,
            Some("spark.apache.org".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_query_only, "PATH", None)?,
            Some("".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_query_only, "QUERY", None)?,
            Some("query=1".to_string())
        );
        assert_eq!(
            ParseUrl::parse(u_query_only, "FILE", None)?,
            Some("?query=1".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_parse_url_overflow_port_eval_modes() -> Result<()> {
        let urls = sa(&[
            Some("http://spark.apache.org:99999/path"),
            Some("inva lid://example.com"),
        ]);
        let parts = sa(&[Some("HOST"), Some("HOST")]);

        // Legacy and Try modes return host for 99999 and NULL for malformed URL
        for eval_mode in [EvalMode::Legacy, EvalMode::Try] {
            let out =
                spark_parse_url(&[Arc::clone(&urls), Arc::clone(&parts)], eval_mode)?;
            let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(out_sa.value(0), "spark.apache.org");
            assert!(out_sa.is_null(1));
        }

        // Ansi mode succeeds for port 99999
        let valid_99999 = sa(&[Some("http://spark.apache.org:99999/path")]);
        let valid_part = sa(&[Some("HOST")]);
        let out = spark_parse_url(&[valid_99999, valid_part], EvalMode::Ansi)?;
        let out_sa = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out_sa.value(0), "spark.apache.org");

        // Ansi mode continues to error on truly malformed URLs
        let err =
            spark_parse_url(&[Arc::clone(&urls), Arc::clone(&parts)], EvalMode::Ansi)
                .unwrap_err();
        assert!(format!("{err}").contains("The url is invalid: inva lid://example.com"));

        // Ansi mode errors on invalid IPv6 without closing bracket
        let bad_ipv6 = sa(&[Some("http://[::1/path")]);
        let bad_part = sa(&[Some("HOST")]);
        let err = spark_parse_url(&[bad_ipv6, bad_part], EvalMode::Ansi).unwrap_err();
        assert!(format!("{err}").contains("The url is invalid: http://[::1/path"));

        // Ansi mode errors on bracketed IPv6 with non-numeric port
        let bad_ipv6_port = sa(&[Some("http://[::1]:abc/path")]);
        let err = spark_parse_url(&[bad_ipv6_port, sa(&[Some("HOST")])], EvalMode::Ansi)
            .unwrap_err();
        assert!(format!("{err}").contains("The url is invalid: http://[::1]:abc/path"));

        // Ansi mode errors on whitespace in authority
        let space_auth = sa(&[Some("http://spark.apache.org: 99999/path")]);
        let err = spark_parse_url(&[space_auth, sa(&[Some("HOST")])], EvalMode::Ansi)
            .unwrap_err();
        assert!(
            format!("{err}")
                .contains("The url is invalid: http://spark.apache.org: 99999/path")
        );

        // Ansi mode errors on backslash in authority
        let backslash_auth = sa(&[Some(r"http://spark.apache.org:99999\path")]);
        let err = spark_parse_url(&[backslash_auth, sa(&[Some("HOST")])], EvalMode::Ansi)
            .unwrap_err();
        assert!(
            format!("{err}")
                .contains(r"The url is invalid: http://spark.apache.org:99999\path")
        );

        Ok(())
    }
}
