// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{Array, ArrayRef, BooleanArray, ListArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::SingleRowListArrayBuilder;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateMetrics, AggregateUDFImpl, Documentation, EmitTo,
    GroupsAccumulator, Signature, Volatility,
};
use datafusion_functions_aggregate::array_agg::{
    ArrayAggAccumulator, DistinctArrayAggAccumulator, array_agg_udaf,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Create DataFusion's native `array_agg` grouped accumulator configured for
/// Spark `collect_list` (`is_distinct = false`) or `collect_set`
/// (`is_distinct = true`) semantics, wrapped so empty groups yield `[]`.
fn create_collect_groups_accumulator(
    args: &AccumulatorArgs,
    is_distinct: bool,
) -> Result<Box<dyn GroupsAccumulator>> {
    let mut args = args.clone();
    args.ignore_nulls = true;
    args.is_distinct = is_distinct;
    let inner = array_agg_udaf().create_groups_accumulator(args)?;
    Ok(Box::new(NullToEmptyListGroupsAccumulator::new(inner)))
}

/// Whether DataFusion's `array_agg` supports grouped accumulation for the
/// given arguments configured for `collect_list` / `collect_set` semantics.
fn collect_groups_accumulator_supported(
    args: &AccumulatorArgs,
    is_distinct: bool,
) -> bool {
    let mut args = args.clone();
    args.ignore_nulls = true;
    args.is_distinct = is_distinct;
    array_agg_udaf().groups_accumulator_supported(args)
}

/// Build an empty list `ScalarValue` for a `List(element_type)` data type.
/// Used as the result for empty window frames and for groups whose inputs
/// were all NULL, matching Spark's `collect_list` / `collect_set` semantics.
fn empty_list_scalar(list_type: &DataType) -> Result<ScalarValue> {
    let DataType::List(field) = list_type else {
        return internal_err!(
            "collect_list/collect_set expected List return type, got {list_type:?}"
        );
    };
    let empty = arrow::array::new_empty_array(field.data_type());
    Ok(SingleRowListArrayBuilder::new(empty).build_list_scalar())
}

/// Spark-compatible `collect_list` aggregate expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#collect_list>
#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns an array of the non-NULL values, including duplicates. \
        Unlike DataFusion's `array_agg`, NULL values are skipped, so a group or window frame that is empty \
        or holds only NULLs returns an empty array, where `array_agg` returns NULL or an array of NULLs. \
        The order of the elements depends on the order of the rows, which may not be deterministic. \
        `ORDER BY` inside the call is not supported and fails with an internal error ([#112](https://github.com/apache/datafusion-spark/issues/112)).",
    syntax_example = "collect_list(expr)",
    sql_example = r#"```sql
> SELECT collect_list(a) FROM (VALUES (1), (2), (NULL), (1)) AS t(a);
[1, 2, 1]
> SELECT collect_list(a) FROM (VALUES (CAST(NULL AS INT))) AS t(a);
[]
```"#,
    argument(
        name = "expr",
        description = "Expression of any type to collect. NULL values are skipped."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkCollectList {
    signature: Signature,
}

impl Default for SparkCollectList {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkCollectList {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SparkCollectList {
    fn name(&self) -> &str {
        "collect_list"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new_list_field(
            arg_types[0].clone(),
            true,
        ))))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new_list(
                format_state_name(args.name, "collect_list"),
                Field::new_list_field(args.input_fields[0].data_type().clone(), true),
                true,
            )
            .into(),
        ])
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let element_type = acc_args.expr_fields[0].data_type().clone();
        let ignore_nulls = true;
        Ok(Box::new(NullToEmptyListAccumulator::new(
            ArrayAggAccumulator::try_new(&element_type, ignore_nulls)?,
            acc_args.return_type().clone(),
        )))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        collect_groups_accumulator_supported(&args, false)
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        create_collect_groups_accumulator(&args, false)
    }

    fn default_value(&self, data_type: &DataType) -> Result<ScalarValue> {
        empty_list_scalar(data_type)
    }
}

/// Spark-compatible `collect_set` aggregate expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#collect_set>
#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns an array of the distinct non-NULL values, in no particular order. \
        Unlike DataFusion's `array_agg(DISTINCT ...)`, NULL values are skipped, so a group or window frame that is empty \
        or holds only NULLs returns an empty array, where `array_agg` returns NULL or `[NULL]`.",
    syntax_example = "collect_set(expr)",
    sql_example = r#"```sql
> SELECT array_sort(collect_set(a)) FROM (VALUES (1), (2), (NULL), (2)) AS t(a);
[1, 2]
> SELECT collect_set(a) FROM (VALUES (CAST(NULL AS INT))) AS t(a);
[]
```"#,
    argument(
        name = "expr",
        description = "Expression of any type to collect. NULL values are skipped."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkCollectSet {
    signature: Signature,
}

impl Default for SparkCollectSet {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkCollectSet {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SparkCollectSet {
    fn name(&self) -> &str {
        "collect_set"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new_list_field(
            arg_types[0].clone(),
            true,
        ))))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new_list(
                format_state_name(args.name, "collect_set"),
                Field::new_list_field(args.input_fields[0].data_type().clone(), true),
                true,
            )
            .into(),
        ])
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let element_type = acc_args.expr_fields[0].data_type().clone();
        let ignore_nulls = true;
        Ok(Box::new(NullToEmptyListAccumulator::new(
            DistinctArrayAggAccumulator::try_new(&element_type, None, ignore_nulls)?,
            acc_args.return_type().clone(),
        )))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        collect_groups_accumulator_supported(&args, true)
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        create_collect_groups_accumulator(&args, true)
    }

    fn default_value(&self, data_type: &DataType) -> Result<ScalarValue> {
        empty_list_scalar(data_type)
    }
}

/// Grouped counterpart of [`NullToEmptyListAccumulator`].
///
/// DataFusion's grouped `array_agg` emits NULL list entries for groups that
/// retained no values. Those entries have empty offset ranges, so dropping the
/// null buffer turns them into valid empty lists without copying values.
/// Only [`GroupsAccumulator::evaluate`] is altered; intermediate state is
/// passed through unchanged so partial aggregation and merging keep working.
struct NullToEmptyListGroupsAccumulator {
    inner: Box<dyn GroupsAccumulator>,
}

impl NullToEmptyListGroupsAccumulator {
    fn new(inner: Box<dyn GroupsAccumulator>) -> Self {
        Self { inner }
    }
}

impl GroupsAccumulator for NullToEmptyListGroupsAccumulator {
    fn set_metrics(&mut self, metrics: Arc<dyn AggregateMetrics>) {
        self.inner.set_metrics(metrics)
    }

    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.inner
            .update_batch(values, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let result = self.inner.evaluate(emit_to)?;
        let Some(list) = result.as_any().downcast_ref::<ListArray>() else {
            return internal_err!(
                "collect_list/collect_set expected ListArray result, got {:?}",
                result.data_type()
            );
        };
        // Only valid because DataFusion emits NULL for groups with no retained
        // values, and those entries have an empty offset range.
        debug_assert!(
            list.nulls().is_none_or(|nulls| {
                nulls
                    .iter()
                    .zip(list.offsets().lengths())
                    .all(|(valid, len)| valid || len == 0)
            }),
            "NULL list entries must be empty for the null buffer to be dropped"
        );
        let (field, offsets, values, _nulls) = list.clone().into_parts();
        Ok(Arc::new(ListArray::new(field, offsets, values, None)))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.inner.state(emit_to)
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        self.inner
            .merge_batch(values, group_indices, total_num_groups)
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        self.inner.convert_to_state(values, opt_filter)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }
}

/// Wrapper accumulator that returns an empty list instead of NULL when all inputs are NULL.
/// This implements Spark's behavior for collect_list and collect_set.
#[derive(Debug)]
struct NullToEmptyListAccumulator<T: Accumulator> {
    inner: T,
    list_type: DataType,
}

impl<T: Accumulator> NullToEmptyListAccumulator<T> {
    pub fn new(inner: T, list_type: DataType) -> Self {
        Self { inner, list_type }
    }
}

impl<T: Accumulator> Accumulator for NullToEmptyListAccumulator<T> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.update_batch(values)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.merge_batch(states)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let result = self.inner.evaluate()?;
        if result.is_null() {
            empty_list_scalar(&self.list_type)
        } else {
            Ok(result)
        }
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.retract_batch(values)
    }

    fn supports_retract_batch(&self) -> bool {
        self.inner.supports_retract_batch()
    }

    fn size(&self) -> usize {
        self.inner.size() + self.list_type.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Int32Array, StringArray};
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Int32Type, Schema};

    fn list_type(element: &DataType) -> DataType {
        DataType::List(Arc::new(Field::new_list_field(element.clone(), true)))
    }

    /// Create a grouped accumulator through the UDAF entry points, as the
    /// physical planner would for `collect_list` / `collect_set`.
    fn make_acc(
        udaf: &dyn AggregateUDFImpl,
        element_type: DataType,
        nullable: bool,
    ) -> Box<dyn GroupsAccumulator> {
        let return_field = Arc::new(Field::new("out", list_type(&element_type), true));
        let schema = Schema::empty();
        let expr_fields = [Arc::new(Field::new("a", element_type, nullable))];
        let args = AccumulatorArgs {
            return_field,
            schema: &schema,
            // Intentionally the opposite of what the Spark functions require:
            // they must override these before delegating.
            ignore_nulls: false,
            order_bys: &[],
            is_reversed: false,
            name: "collect",
            is_distinct: udaf.name() == "collect_list",
            exprs: &[],
            expr_fields: &expr_fields,
        };
        assert!(udaf.groups_accumulator_supported(args.clone()));
        udaf.create_groups_accumulator(args).unwrap()
    }

    fn int_acc(udaf: &dyn AggregateUDFImpl) -> Box<dyn GroupsAccumulator> {
        make_acc(udaf, DataType::Int32, true)
    }

    fn int_array(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    /// Rows: group 0 -> [1, 2, 1], group 1 -> [3, 3], group 2 -> only NULLs,
    /// group 3 -> no rows at all.
    fn update(acc: &mut dyn GroupsAccumulator) {
        let values = int_array(vec![
            Some(1),
            Some(3),
            None,
            Some(2),
            Some(3),
            None,
            Some(1),
        ]);
        acc.update_batch(&[values], &[0, 1, 2, 0, 1, 2, 0], None, 4)
            .unwrap();
    }

    /// Sorted per-group contents; asserts no group is NULL and that the result
    /// type matches what `return_type` declares.
    fn rows(result: &ArrayRef) -> Vec<Vec<i32>> {
        assert_eq!(result.data_type(), &list_type(&DataType::Int32));
        let list = result.as_list::<i32>();
        assert_eq!(list.null_count(), 0, "no group may evaluate to NULL");
        (0..list.len())
            .map(|i| {
                let mut v = list.value(i).as_primitive::<Int32Type>().values().to_vec();
                v.sort();
                v
            })
            .collect()
    }

    /// Runs `scenario` against both functions and compares the sorted rows.
    fn check(
        scenario: impl Fn(&mut dyn GroupsAccumulator) -> ArrayRef,
        list_expected: Vec<Vec<i32>>,
        set_expected: Vec<Vec<i32>>,
    ) {
        let mut acc = int_acc(&SparkCollectList::new());
        assert_eq!(rows(&scenario(acc.as_mut())), list_expected);
        let mut acc = int_acc(&SparkCollectSet::new());
        assert_eq!(rows(&scenario(acc.as_mut())), set_expected);
    }

    #[test]
    fn update_and_evaluate() {
        check(
            |acc| {
                update(acc);
                acc.evaluate(EmitTo::All).unwrap()
            },
            vec![vec![1, 1, 2], vec![3, 3], vec![], vec![]],
            vec![vec![1, 2], vec![3], vec![], vec![]],
        );
    }

    #[test]
    fn partial_state_then_merge() {
        for udaf in [
            &SparkCollectList::new() as &dyn AggregateUDFImpl,
            &SparkCollectSet::new(),
        ] {
            let mut partial = int_acc(udaf);
            update(partial.as_mut());
            // State is passed through untouched: groups without values stay
            // NULL (DataFusion's representation) rather than becoming `[]`.
            let state = partial.state(EmitTo::All).unwrap();
            assert_eq!(state.len(), 1);
            assert_eq!(state[0].as_list::<i32>().null_count(), 2);

            let mut merged = int_acc(udaf);
            merged.merge_batch(&state, &[0, 1, 2, 3], 4).unwrap();
            let result = merged.evaluate(EmitTo::All).unwrap();
            let expected = if udaf.name() == "collect_list" {
                vec![vec![1, 1, 2], vec![3, 3], vec![], vec![]]
            } else {
                vec![vec![1, 2], vec![3], vec![], vec![]]
            };
            assert_eq!(rows(&result), expected);
        }
    }

    #[test]
    fn convert_to_state_matches_state_representation() {
        for udaf in [
            &SparkCollectList::new() as &dyn AggregateUDFImpl,
            &SparkCollectSet::new(),
        ] {
            let acc = int_acc(udaf);
            let input = int_array(vec![Some(1), None, Some(2)]);
            let state = acc.convert_to_state(&[input], None).unwrap();
            let list = state[0].as_list::<i32>();
            assert_eq!(list.len(), 3);
            // The NULL input row becomes a NULL state entry (not `[NULL]`).
            // Its offset range is not empty, which is fine because only
            // `evaluate` drops the null buffer, never this state.
            assert!(!list.is_null(0));
            assert!(list.is_null(1));
            assert!(!list.is_null(2));

            // The converted state must be mergeable.
            let mut merged = int_acc(udaf);
            merged.merge_batch(&state, &[0, 0, 1], 2).unwrap();
            let result = merged.evaluate(EmitTo::All).unwrap();
            assert_eq!(rows(&result), vec![vec![1], vec![2]]);
        }
    }

    #[test]
    fn evaluate_emit_first_keeps_remaining_groups() {
        for (udaf, expected_first) in [
            (
                &SparkCollectList::new() as &dyn AggregateUDFImpl,
                vec![vec![1, 1, 2], vec![3, 3]],
            ),
            (&SparkCollectSet::new(), vec![vec![1, 2], vec![3]]),
        ] {
            let mut acc = int_acc(udaf);
            update(acc.as_mut());
            // Group 3 gets a value; group 2 stays all-NULL.
            acc.update_batch(&[int_array(vec![Some(9)])], &[3], None, 4)
                .unwrap();

            let first = acc.evaluate(EmitTo::First(2)).unwrap();
            assert_eq!(rows(&first), expected_first);

            // Old groups 2 and 3 are now indexed 0 and 1 and keep their state.
            let rest = acc.evaluate(EmitTo::All).unwrap();
            assert_eq!(rows(&rest), vec![vec![], vec![9]]);
        }
    }

    #[test]
    fn multiple_batches_with_growing_group_count() {
        check(
            |acc| {
                let b1 = int_array(vec![Some(1), Some(2)]);
                acc.update_batch(&[b1], &[0, 1], None, 2).unwrap();
                let b2 = int_array(vec![Some(1), Some(5), None]);
                acc.update_batch(&[b2], &[0, 3, 2], None, 4).unwrap();
                acc.evaluate(EmitTo::All).unwrap()
            },
            vec![vec![1, 1], vec![2], vec![], vec![5]],
            vec![vec![1], vec![2], vec![], vec![5]],
        );
    }

    #[test]
    fn update_with_filter() {
        check(
            |acc| {
                let values = int_array(vec![Some(1), Some(2), Some(3), Some(4)]);
                let filter = BooleanArray::from(vec![true, false, false, false]);
                acc.update_batch(&[values], &[0, 0, 1, 1], Some(&filter), 3)
                    .unwrap();
                acc.evaluate(EmitTo::All).unwrap()
            },
            vec![vec![1], vec![], vec![]],
            vec![vec![1], vec![], vec![]],
        );
    }

    #[test]
    fn string_values() {
        for (udaf, expected) in [
            (
                &SparkCollectList::new() as &dyn AggregateUDFImpl,
                vec![vec!["x", "x", "y"], vec![]],
            ),
            (&SparkCollectSet::new(), vec![vec!["x", "y"], vec![]]),
        ] {
            let mut acc = make_acc(udaf, DataType::Utf8, true);
            let values: ArrayRef = Arc::new(StringArray::from(vec![
                Some("x"),
                Some("y"),
                None,
                Some("x"),
            ]));
            acc.update_batch(&[values], &[0, 0, 1, 0], None, 2).unwrap();
            let result = acc.evaluate(EmitTo::All).unwrap();
            assert_eq!(result.data_type(), &list_type(&DataType::Utf8));
            let list = result.as_list::<i32>();
            assert_eq!(list.null_count(), 0);
            let actual: Vec<Vec<String>> = (0..list.len())
                .map(|i| {
                    let mut v: Vec<String> = list
                        .value(i)
                        .as_string::<i32>()
                        .iter()
                        .map(|s| s.unwrap().to_string())
                        .collect();
                    v.sort();
                    v
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn non_nullable_input() {
        for udaf in [
            &SparkCollectList::new() as &dyn AggregateUDFImpl,
            &SparkCollectSet::new(),
        ] {
            let mut acc = make_acc(udaf, DataType::Int32, false);
            let values = int_array(vec![Some(1), Some(1), Some(2)]);
            // Group 2 is never seen and must still evaluate to `[]`.
            acc.update_batch(&[values], &[0, 0, 1], None, 3).unwrap();
            let result = acc.evaluate(EmitTo::All).unwrap();
            // `rows` also checks the result type matches `return_type`.
            let expected = if udaf.name() == "collect_list" {
                vec![vec![1, 1], vec![2], vec![]]
            } else {
                vec![vec![1], vec![2], vec![]]
            };
            assert_eq!(rows(&result), expected);
        }
    }

    /// Inner accumulator that returns a NULL list entry with a non-empty
    /// offset range, which the wrapper must refuse to turn into a valid list.
    struct NonEmptyNullInner;

    impl GroupsAccumulator for NonEmptyNullInner {
        fn update_batch(
            &mut self,
            _: &[ArrayRef],
            _: &[usize],
            _: Option<&BooleanArray>,
            _: usize,
        ) -> Result<()> {
            unimplemented!()
        }

        fn evaluate(&mut self, _: EmitTo) -> Result<ArrayRef> {
            let values: ArrayRef = Arc::new(Int32Array::from(vec![1]));
            let field = Arc::new(Field::new_list_field(DataType::Int32, true));
            let offsets = OffsetBuffer::new(vec![0, 1].into());
            let nulls = Some(NullBuffer::from(vec![false]));
            Ok(Arc::new(ListArray::new(field, offsets, values, nulls)))
        }

        fn state(&mut self, _: EmitTo) -> Result<Vec<ArrayRef>> {
            unimplemented!()
        }

        fn merge_batch(&mut self, _: &[ArrayRef], _: &[usize], _: usize) -> Result<()> {
            unimplemented!()
        }

        fn convert_to_state(
            &self,
            _: &[ArrayRef],
            _: Option<&BooleanArray>,
        ) -> Result<Vec<ArrayRef>> {
            unimplemented!()
        }

        fn size(&self) -> usize {
            0
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "NULL list entries must be empty")]
    fn evaluate_rejects_non_empty_null_entries() {
        let mut acc = NullToEmptyListGroupsAccumulator::new(Box::new(NonEmptyNullInner));
        let _ = acc.evaluate(EmitTo::All);
    }
}
