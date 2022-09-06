import dataclasses

import typeguard

from exasol_data_science_utils_python.schema.column_name import ColumnName
from exasol_data_science_utils_python.schema.column_type import ColumnType


@dataclasses.dataclass(frozen=True, repr=True, eq=True)
class Column:
    name: ColumnName
    type: ColumnType

    def __post_init__(self):
        typeguard.check_type(value=self.name,
                             expected_type=ColumnName,
                             argname="name")
        typeguard.check_type(value=self.type,
                             expected_type=ColumnType,
                             argname="type")
