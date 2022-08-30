from typing import List

from typeguard import typechecked

from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.table_like import TableLike
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.view_name import ViewName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class View(TableLike):

    @typechecked
    def __init__(self, name: ViewName, columns: List[Column]):
        super().__init__(name, columns)
