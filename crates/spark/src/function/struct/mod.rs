// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark struct functions.

use datafusion_expr::ScalarUDF;
use std::sync::Arc;

pub mod expr_fn {
    use datafusion_expr::{Expr, lit};

    /// Creates a struct with Spark's field names.
    ///
    /// Names are literal arguments to DataFusion's `named_struct`, so they stay
    /// in the expression when functions are reconstructed through the registry.
    /// Per-child nullability and empty structs are deferred to the Spark
    /// `named_struct` implementation: <https://github.com/apache/datafusion-spark/issues/224>.
    pub fn r#struct(args: Vec<Expr>) -> Expr {
        let mut named_args = Vec::with_capacity(args.len() * 2);
        for (i, expr) in args.into_iter().enumerate() {
            let name = match &expr {
                Expr::Column(column) => Some(column.name.clone()),
                Expr::Alias(alias) => Some(alias.name.clone()),
                Expr::ScalarFunction(function) if function.func.name() == "get_field" => {
                    match function.args.get(1) {
                        Some(Expr::Literal(value, _)) => {
                            value.try_as_str().flatten().map(str::to_owned)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
            .unwrap_or_else(|| format!("col{}", i + 1));
            named_args.push(lit(name));
            named_args.push(match expr {
                Expr::Alias(alias) => *alias.expr,
                expr => expr,
            });
        }
        datafusion_functions::core::named_struct().call(named_args)
    }
}

pub fn functions() -> Vec<Arc<ScalarUDF>> {
    vec![]
}

#[cfg(test)]
mod tests {
    use super::expr_fn;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::{DFSchema, Result};
    use datafusion_expr::{Expr, ExprSchemable, col, lit};

    #[test]
    fn names_are_literal_arguments() -> Result<()> {
        let schema = DFSchema::try_from(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]))?;
        let expr = expr_fn::r#struct(vec![col("a"), lit(2), lit(3).alias("c1")]);
        let Expr::ScalarFunction(function) = &expr else {
            panic!("expected named_struct")
        };
        assert_eq!(function.func.name(), "named_struct");
        assert_eq!(
            function.args,
            vec![lit("a"), col("a"), lit("col2"), lit(2), lit("c1"), lit(3)]
        );
        let DataType::Struct(fields) = expr.get_type(&schema)? else {
            panic!("expected struct")
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["a", "col2", "c1"]
        );
        Ok(())
    }
    #[cfg(feature = "core")]
    #[tokio::test]
    async fn registered_function_reconstruction_retains_schema() -> Result<()> {
        use datafusion::prelude::SessionContext;
        use datafusion_expr::registry::FunctionRegistry;

        let ctx = SessionContext::new();
        let input = ctx.sql("SELECT 1::INT AS a").await?;
        let expr = expr_fn::r#struct(vec![col("a"), lit(2)]);
        let Expr::ScalarFunction(function) = &expr else {
            panic!("expected named_struct")
        };
        let rebuilt = ctx.udf(function.func.name())?.call(function.args.clone());
        assert_eq!(
            expr.get_type(input.schema())?,
            rebuilt.get_type(input.schema())?
        );
        let batches = input.select(vec![rebuilt.alias("s")])?.collect().await?;
        let DataType::Struct(fields) = batches[0].schema().field(0).data_type().clone()
        else {
            panic!("expected struct")
        };
        assert_eq!(fields[0].name(), "a");
        assert_eq!(fields[1].name(), "col2");
        assert_eq!(batches[0].num_rows(), 1);
        Ok(())
    }
}
