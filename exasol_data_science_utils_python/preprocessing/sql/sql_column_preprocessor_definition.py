from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor import SQLColumnPreprocessor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class SQLColumnPreprocessorDefinition:
    @typechecked
    def __init__(self, column_name: str, column_preprocessor: SQLColumnPreprocessor):
        self.column_preprocessor = column_preprocessor
        self.column_name = column_name

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, SQLColumnPreprocessorDefinition) and \
               self.column_preprocessor == other.column_preprocessor and \
               self.column_name == other.column_name
