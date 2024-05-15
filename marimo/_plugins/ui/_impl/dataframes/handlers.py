# Copyright 2024 Marimo. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, NoReturn, cast

from marimo._plugins.ui._impl.dataframes.transforms import (
    AggregateTransform,
    ColumnConversionTransform,
    FilterRowsTransform,
    GroupByTransform,
    RenameColumnTransform,
    SampleRowsTransform,
    SelectColumnsTransform,
    ShuffleRowsTransform,
    SortColumnTransform,
    Transform,
    Transformations,
    TransformType,
)

if TYPE_CHECKING:
    import pandas as pd

class TransformHandlersBase(ABC):
    @staticmethod
    @abstractmethod
    def handle(df: Any, transform: Transform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_column_conversion(df: Any, transform: ColumnConversionTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_rename_column(df: Any, transform: RenameColumnTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_sort_column(df: Any, transform: SortColumnTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_filter_rows(df: Any, transform: FilterRowsTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_group_by(df: Any, transform: GroupByTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_aggregate(df: Any, transform: AggregateTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_select_columns(df: Any, transform: SelectColumnsTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_shuffle_rows(df: Any, transform: ShuffleRowsTransform) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def handle_sample_rows(df: Any, transform: SampleRowsTransform) -> Any:
        pass

def apply_transforms(df: Any, transforms: Transformations) -> Any:
    if not transforms.transforms:
        return df
    for transform in transforms.transforms:
        df = TransformHandlersBase.handle(df, transform)
    return df

def _assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"Unhandled value: {value} ({type(value).__name__})")

def _coerce_value(dtype: Any, value: Any) -> Any:
    import numpy as np

    return np.array([value]).astype(dtype)[0]

class TransformsContainer:
    """
    Keeps internal state of the last transformation applied to the dataframe.
    So that we can incrementally apply transformations.
    """

    def __init__(self, df: Any) -> None:
        self._original_df = df
        # The dataframe for the given transform.
        self._snapshot_df = df
        self._transforms: List[Transform] = []

    def apply(self, transform: Transformations) -> Any:
        """
        Applies the given transformations to the dataframe.
        """
        # If the new transformations are a superset of the existing ones,
        # then we can just apply the new ones to the snapshot dataframe.
        if self._is_superset(transform):
            transforms_to_apply = self._get_next_transformations(transform)
            self._snapshot_df = apply_transforms(
                self._snapshot_df, transforms_to_apply
            )
            self._transforms = transform.transforms
            return self._snapshot_df

        # If the new transformations are not a superset of the existing ones,
        # then we need to start from the original dataframe.
        else:
            self._snapshot_df = apply_transforms(self._original_df, transform)
            self._transforms = transform.transforms
            return self._snapshot_df

    def _is_superset(self, transforms: Transformations) -> bool:
        """
        Checks if the new transformations are a superset of the existing ones.
        """
        if not self._transforms:
            return False

        # If the new transformations are smaller than the existing ones,
        # then it's not a superset.
        if len(self._transforms) > len(transforms.transforms):
            return False

        for i, transform in enumerate(self._transforms):
            if transform != transforms.transforms[i]:
                return False

        return True

    def _get_next_transformations(
        self, transforms: Transformations
    ) -> Transformations:
        """
        Gets the next transformations to apply.
        """
        if self._is_superset(transforms):
            return Transformations(
                transforms.transforms[len(self._transforms) :]
            )
        else:
            return transforms
