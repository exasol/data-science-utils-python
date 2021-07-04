from typing import Iterator

import numpy as np
import pandas as pd


class ReservoirShuffle:
    def __init__(self, iterator: Iterator[pd.DataFrame], buffer_size: int, batch_size: int):
        self._iterator = iterator
        self._fill_level = 0
        self._internal_buffer_size = buffer_size + batch_size
        self._free_bitset = np.ones(shape=self._internal_buffer_size, dtype=bool)
        self._buffer_df = None
        self._batch_size = batch_size
        self._buffer_size = buffer_size

    def _df_empty(self, columns, dtypes, size: int):
        assert len(columns) == len(dtypes)
        df = pd.DataFrame(index=range(size))
        for c, d in zip(columns, dtypes):
            df[c] = pd.Series(np.zeros(shape=size, dtype=d))
        return df

    def _copy_dataframe_selection_to_dataframe_selection(self, from_df, from_selection, to_df, to_selection):
        for c in from_df.columns:
            batch_rows_to_move_to_buffer = from_df[c].values[from_selection]
            to_df[c].values[to_selection] = batch_rows_to_move_to_buffer

    def _fill_buffer_with_batch(self, batch_df: pd.DataFrame):
        free = len(self._buffer_df) - self._fill_level
        elements_to_fill_count = min(free, len(batch_df))
        if free > 0:
            elements_to_fill_in_buffer = np.arange(len(self._buffer_df), dtype=int)[self._free_bitset][
                                         0:elements_to_fill_count]
            self._copy_dataframe_selection_to_dataframe_selection(batch_df, slice(0, elements_to_fill_count),
                                                                  self._buffer_df, elements_to_fill_in_buffer)
            self._free_bitset[elements_to_fill_in_buffer] = 0
            self._fill_level += elements_to_fill_count
            remaining_df = batch_df.iloc[elements_to_fill_count:]
            return remaining_df
        else:
            return batch_df

    def _generate_batch(self, remaining_df: pd.DataFrame):
        result_count = max(1,
                           len(remaining_df) // self._batch_size + 0 if len(
                               remaining_df) % self._batch_size == 0 else 1)
        result_size = min(result_count * self._batch_size, self._fill_level)
        choice_range, choice_range_for_remaining_start = self._compute_choice_ranges(remaining_df)

        choice_in_buffer, choice_in_remaining = self._make_random_choices(
            choice_range, choice_range_for_remaining_start, result_size)

        result_batch_df = self._create_result_batch_from_choices(
            choice_in_buffer, choice_in_remaining, remaining_df)

        elements_to_replace_in_buffer = self._move_remaining_rows_to_buffer(
            choice_in_buffer, choice_in_remaining, remaining_df)

        self._free_buffer(choice_in_buffer, elements_to_replace_in_buffer)

        return result_batch_df

    def _free_buffer(self, choice_in_buffer: np.array, elements_to_replace_in_buffer: np.array):
        elements_to_free = choice_in_buffer[np.logical_not(np.isin(choice_in_buffer, elements_to_replace_in_buffer))]
        self._free_bitset[elements_to_free] = 1
        self._fill_level -= len(elements_to_free)

    def _move_remaining_rows_to_buffer(self, choice_in_buffer: np.array, choice_in_remaining: np.array,
                                       remaining_df: pd.DataFrame):
        elements_to_replace_in_buffer = choice_in_buffer[:len(remaining_df) - len(choice_in_remaining)]
        remaining_range = np.arange(len(remaining_df), dtype=int)
        elements_to_move_to_buffer = remaining_range[np.logical_not(np.isin(remaining_range, choice_in_remaining))]
        self._copy_dataframe_selection_to_dataframe_selection(remaining_df, elements_to_move_to_buffer,
                                                              self._buffer_df, elements_to_replace_in_buffer)
        return elements_to_replace_in_buffer

    def _create_result_batch_from_choices(self, choice_in_buffer: np.array, choice_in_remaining: np.array,
                                          remaining_df: pd.DataFrame):
        result_in_buffer_df = self._buffer_df.iloc[choice_in_buffer]
        result_in_remaining_df = remaining_df.iloc[choice_in_remaining]
        result_df = pd.concat([result_in_buffer_df, result_in_remaining_df])
        return result_df

    def _make_random_choices(self, choice_range: np.array, choice_range_for_remaining_start: int, result_size: int):
        np.random.shuffle(choice_range)
        choice = choice_range[0:result_size]
        choice_in_buffer = choice[choice < choice_range_for_remaining_start]
        choice_in_remaining = choice[choice >= choice_range_for_remaining_start] - choice_range_for_remaining_start
        return choice_in_buffer, choice_in_remaining

    def _compute_choice_ranges(self, remaining_df: pd.DataFrame):
        choice_range_for_buffer = np.arange(len(self._buffer_df), dtype=int)[np.logical_not(self._free_bitset)]
        if len(choice_range_for_buffer) > 0:
            choice_range_for_remaining_start = np.max(choice_range_for_buffer) + 1
        else:
            choice_range_for_remaining_start = 0
        choice_range_for_remaining = np.arange(
            choice_range_for_remaining_start,
            choice_range_for_remaining_start + len(remaining_df), dtype=int)
        choice_range = np.concatenate([choice_range_for_buffer, choice_range_for_remaining])
        return choice_range, choice_range_for_remaining_start

    def shuffle(self):
        empty_df = None
        remaining_df = None
        for batch_df in self._iterator:
            if self._buffer_df is None:
                self._buffer_df = self._df_empty(batch_df.columns, batch_df.dtypes, self._internal_buffer_size)
                empty_df = self._df_empty(batch_df.columns, batch_df.dtypes, 0)
            remaining_df = self._fill_buffer_with_batch(batch_df)
            if len(remaining_df) > 0:
                result = self._generate_batch(remaining_df)
                yield result
        while self._fill_level > 0:
            result = self._generate_batch(empty_df)
            yield result
