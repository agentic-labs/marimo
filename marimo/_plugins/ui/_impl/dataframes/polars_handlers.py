from typing import TYPE_CHECKING, cast
from marimo._plugins.ui._impl.dataframes.handlers import TransformHandlersBase, _coerce_value, _assert_never
from marimo._plugins.ui._impl.dataframes.transforms import (
    ColumnConversionTransform,
    RenameColumnTransform,
    SortColumnTransform,
    FilterRowsTransform,
    GroupByTransform,
    AggregateTransform,
    SelectColumnsTransform,
    ShuffleRowsTransform,
    SampleRowsTransform,
    Transform,
    TransformType,
)

if TYPE_CHECKING:
    import polars as pl

class PolarsTransformHandlers(TransformHandlersBase):
    @staticmethod
    def handle(df: "pl.DataFrame", transform: Transform) -> "pl.DataFrame":
        transform_type: TransformType = transform.type

        if transform_type is TransformType.COLUMN_CONVERSION:
            return PolarsTransformHandlers.handle_column_conversion(
                df, cast(ColumnConversionTransform, transform)
            )
        elif transform_type is TransformType.RENAME_COLUMN:
            return PolarsTransformHandlers.handle_rename_column(
                df, cast(RenameColumnTransform, transform)
            )
        elif transform_type is TransformType.SORT_COLUMN:
            return PolarsTransformHandlers.handle_sort_column(
                df, cast(SortColumnTransform, transform)
            )
        elif transform_type is TransformType.FILTER_ROWS:
            return PolarsTransformHandlers.handle_filter_rows(
                df, cast(FilterRowsTransform, transform)
            )
        elif transform_type is TransformType.GROUP_BY:
            return PolarsTransformHandlers.handle_group_by(
                df, cast(GroupByTransform, transform)
            )
        elif transform_type is TransformType.AGGREGATE:
            return PolarsTransformHandlers.handle_aggregate(
                df, cast(AggregateTransform, transform)
            )
        elif transform_type is TransformType.SELECT_COLUMNS:
            return PolarsTransformHandlers.handle_select_columns(
                df, cast(SelectColumnsTransform, transform)
            )
        elif transform_type is TransformType.SHUFFLE_ROWS:
            return PolarsTransformHandlers.handle_shuffle_rows(
                df, cast(ShuffleRowsTransform, transform)
            )
        elif transform_type is TransformType.SAMPLE_ROWS:
            return PolarsTransformHandlers.handle_sample_rows(
                df, cast(SampleRowsTransform, transform)
            )
        else:
            _assert_never(transform_type)

    @staticmethod
    def handle_column_conversion(
        df: "pl.DataFrame", transform: ColumnConversionTransform
    ) -> "pl.DataFrame":
        return df.with_column(df[transform.column_id].cast(transform.data_type))

    @staticmethod
    def handle_rename_column(
        df: "pl.DataFrame", transform: RenameColumnTransform
    ) -> "pl.DataFrame":
        return df.rename({transform.column_id: transform.new_column_id})

    @staticmethod
    def handle_sort_column(
        df: "pl.DataFrame", transform: SortColumnTransform
    ) -> "pl.DataFrame":
        return df.sort(
            by_column=transform.column_id,
            descending=not transform.ascending,
            nulls_last=transform.na_position == "last",
        )

    @staticmethod
    def handle_filter_rows(
        df: "pl.DataFrame", transform: FilterRowsTransform
    ) -> "pl.DataFrame":
        for condition in transform.where:
            value = _coerce_value(
                df[condition.column_id].dtype, condition.value
            )
            if condition.operator == "==":
                df_filter = df[condition.column_id] == value
            elif condition.operator == "!=":
                df_filter = df[condition.column_id] != value
            elif condition.operator == ">":
                df_filter = df[condition.column_id] > value
            elif condition.operator == "<":
                df_filter = df[condition.column_id] < value
            elif condition.operator == ">=":
                df_filter = df[condition.column_id] >= value
            elif condition.operator == "<=":
                df_filter = df[condition.column_id] <= value
            elif condition.operator == "is_true":
                df_filter = df[condition.column_id].eq(True)
            elif condition.operator == "is_false":
                df_filter = df[condition.column_id].eq(False)
            elif condition.operator == "is_nan":
                df_filter = df[condition.column_id].is_null()
            elif condition.operator == "is_not_nan":
                df_filter = df[condition.column_id].is_not_null()
            elif condition.operator == "equals":
                df_filter = df[condition.column_id].eq(value)
            elif condition.operator == "does_not_equal":
                df_filter = df[condition.column_id].ne(value)
            elif condition.operator == "contains":
                df_filter = df[condition.column_id].str.contains(
                    value, literal=True
                )
            elif condition.operator == "regex":
                df_filter = df[condition.column_id].str.contains(
                    value, literal=False
                )
            elif condition.operator == "starts_with":
                df_filter = df[condition.column_id].str.starts_with(value)
            elif condition.operator == "ends_with":
                df_filter = df[condition.column_id].str.ends_with(value)
            elif condition.operator == "in":
                df_filter = df[condition.column_id].is_in(value)
            else:
                _assert_never(condition.operator)

            if transform.operation == "keep_rows":
                df = df.filter(df_filter)
            elif transform.operation == "remove_rows":
                df = df.filter(~df_filter)
            else:
                _assert_never(transform.operation)
        return df

    @staticmethod
    def handle_group_by(
        df: "pl.DataFrame", transform: GroupByTransform
    ) -> "pl.DataFrame":
        group = df.groupby(transform.column_ids)
        if transform.aggregation == "count":
            return group.count()
        elif transform.aggregation == "sum":
            return group.sum()
        elif transform.aggregation == "mean":
            return group.mean()
        elif transform.aggregation == "median":
            return group.median()
        elif transform.aggregation == "min":
            return group.min()
        elif transform.aggregation == "max":
            return group.max()
        else:
            _assert_never(transform.aggregation)

    @staticmethod
    def handle_aggregate(
        df: "pl.DataFrame", transform: AggregateTransform
    ) -> "pl.DataFrame":
        aggs = [df[col].agg(func) for col in transform.column_ids for func in transform.aggregations]
        return df.select(aggs)

    @staticmethod
    def handle_select_columns(
        df: "pl.DataFrame", transform: SelectColumnsTransform
    ) -> "pl.DataFrame":
        return df.select(transform.column_ids)

    @staticmethod
    def handle_shuffle_rows(
        df: "pl.DataFrame", transform: ShuffleRowsTransform
    ) -> "pl.DataFrame":
        return df.sample(frac=1.0, with_replacement=False, shuffle=True, seed=transform.seed)

    @staticmethod
    def handle_sample_rows(
        df: "pl.DataFrame", transform: SampleRowsTransform
    ) -> "pl.DataFrame":
        return df.sample(n=transform.n, with_replacement=transform.replace, shuffle=True, seed=transform.seed)
