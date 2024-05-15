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
    import pandas as pd

class PandasTransformHandlers(TransformHandlersBase):
    @staticmethod
    def handle(df: "pd.DataFrame", transform: Transform) -> "pd.DataFrame":
        transform_type: TransformType = transform.type

        if transform_type is TransformType.COLUMN_CONVERSION:
            return PandasTransformHandlers.handle_column_conversion(
                df, cast(ColumnConversionTransform, transform)
            )
        elif transform_type is TransformType.RENAME_COLUMN:
            return PandasTransformHandlers.handle_rename_column(
                df, cast(RenameColumnTransform, transform)
            )
        elif transform_type is TransformType.SORT_COLUMN:
            return PandasTransformHandlers.handle_sort_column(
                df, cast(SortColumnTransform, transform)
            )
        elif transform_type is TransformType.FILTER_ROWS:
            return PandasTransformHandlers.handle_filter_rows(
                df, cast(FilterRowsTransform, transform)
            )
        elif transform_type is TransformType.GROUP_BY:
            return PandasTransformHandlers.handle_group_by(
                df, cast(GroupByTransform, transform)
            )
        elif transform_type is TransformType.AGGREGATE:
            return PandasTransformHandlers.handle_aggregate(
                df, cast(AggregateTransform, transform)
            )
        elif transform_type is TransformType.SELECT_COLUMNS:
            return PandasTransformHandlers.handle_select_columns(
                df, cast(SelectColumnsTransform, transform)
            )
        elif transform_type is TransformType.SHUFFLE_ROWS:
            return PandasTransformHandlers.handle_shuffle_rows(
                df, cast(ShuffleRowsTransform, transform)
            )
        elif transform_type is TransformType.SAMPLE_ROWS:
            return PandasTransformHandlers.handle_sample_rows(
                df, cast(SampleRowsTransform, transform)
            )
        else:
            _assert_never(transform_type)

    @staticmethod
    def handle_column_conversion(
        df: "pd.DataFrame", transform: ColumnConversionTransform
    ) -> "pd.DataFrame":
        df[transform.column_id] = df[transform.column_id].astype(
            transform.data_type,
            errors=transform.errors,
        )  # type: ignore[call-overload]
        return df

    @staticmethod
    def handle_rename_column(
        df: "pd.DataFrame", transform: RenameColumnTransform
    ) -> "pd.DataFrame":
        return df.rename(
            columns={transform.column_id: transform.new_column_id}
        )

    @staticmethod
    def handle_sort_column(
        df: "pd.DataFrame", transform: SortColumnTransform
    ) -> "pd.DataFrame":
        return df.sort_values(
            by=cast(str, transform.column_id),
            ascending=transform.ascending,
            na_position=transform.na_position,
        )

    @staticmethod
    def handle_filter_rows(
        df: "pd.DataFrame", transform: FilterRowsTransform
    ) -> "pd.DataFrame":
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
                df_filter = df[condition.column_id].isna()
            elif condition.operator == "is_not_nan":
                df_filter = df[condition.column_id].notna()
            elif condition.operator == "equals":
                df_filter = df[condition.column_id].eq(value)
            elif condition.operator == "does_not_equal":
                df_filter = df[condition.column_id].ne(value)
            elif condition.operator == "contains":
                df_filter = df[condition.column_id].str.contains(
                    value, regex=False
                )
            elif condition.operator == "regex":
                df_filter = df[condition.column_id].str.contains(
                    value, regex=True
                )
            elif condition.operator == "starts_with":
                df_filter = df[condition.column_id].str.startswith(value)
            elif condition.operator == "ends_with":
                df_filter = df[condition.column_id].str.endswith(value)
            elif condition.operator == "in":
                df_filter = df[condition.column_id].isin(value)
            else:
                _assert_never(condition.operator)

            if transform.operation == "keep_rows":
                df = df[df_filter]
            elif transform.operation == "remove_rows":
                df = df[~df_filter]
            else:
                _assert_never(transform.operation)
        return df

    @staticmethod
    def handle_group_by(
        df: "pd.DataFrame", transform: GroupByTransform
    ) -> "pd.DataFrame":
        group = df.groupby(transform.column_ids, dropna=transform.drop_na)
        if transform.aggregation == "count":
            return group.count()
        elif transform.aggregation == "sum":
            return group.sum()
        elif transform.aggregation == "mean":
            return group.mean(numeric_only=True)
        elif transform.aggregation == "median":
            return group.median(numeric_only=True)
        elif transform.aggregation == "min":
            return group.min()
        elif transform.aggregation == "max":
            return group.max()
        else:
            _assert_never(transform.aggregation)

    @staticmethod
    def handle_aggregate(
        df: "pd.DataFrame", transform: AggregateTransform
    ) -> "pd.DataFrame":
        dict_of_aggs = {
            column_id: transform.aggregations
            for column_id in transform.column_ids
        }

        # Pandas type-checking doesn't like the fact that the values
        # are lists of strings (function names), even though the docs permit
        # such a value
        return cast("pd.DataFrame", df.agg(dict_of_aggs))  # type: ignore  # noqa: E501

    @staticmethod
    def handle_select_columns(
        df: "pd.DataFrame", transform: SelectColumnsTransform
    ) -> "pd.DataFrame":
        return df[transform.column_ids]

    @staticmethod
    def handle_shuffle_rows(
        df: "pd.DataFrame", transform: ShuffleRowsTransform
    ) -> "pd.DataFrame":
        return df.sample(frac=1, random_state=transform.seed)

    @staticmethod
    def handle_sample_rows(
        df: "pd.DataFrame", transform: SampleRowsTransform
    ) -> "pd.DataFrame":
        return df.sample(
            n=transform.n,
            random_state=transform.seed,
            replace=transform.replace,
        )
