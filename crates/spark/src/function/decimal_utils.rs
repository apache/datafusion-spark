// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

/// Formats a decimal the way Java's `BigDecimal.toString` does, which is what
/// Spark prints for a DECIMAL in `to_json` and `format_string`'s `%s`.
/// `unscaled` is the signed unscaled value as text.
///
/// With `adjusted = -scale + (digits - 1)`: plain notation when `scale >= 0`
/// and `adjusted >= -6` (`1.50`, `0.000001`), scientific notation otherwise
/// (`0E-10`, `1.0E-7`, `1.23E+4`; the exponent always carries a sign).
pub(crate) fn java_big_decimal_text(unscaled: &str, scale: i8) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
