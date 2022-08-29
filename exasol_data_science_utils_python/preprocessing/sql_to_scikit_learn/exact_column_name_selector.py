from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_selector import ColumnSelector
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class ExactColumnNameSelector(ColumnSelector):
    def __init__(self, column_name: str):
        self._column_name = column_name

    def column_accepted(self, column: Column):
        return column.name.name == self._column_name

    def __eq__(self, other):
        return isinstance(other, ExactColumnNameSelector) and \
               self._column_name == other._column_name

    def __repr__(self):
        generate_repr_for_object(self)
