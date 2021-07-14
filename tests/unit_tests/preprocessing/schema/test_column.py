import pytest

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.column_type import ColumnType


def test_set_new_type_fail():
    column = Column(ColumnName("abc"), ColumnType("INTEGER"))
    with pytest.raises(AttributeError) as c:
        column.type = "edf"


def test_set_new_name_fail():
    column = Column(ColumnName("abc"), ColumnType("INTEGER"))
    with pytest.raises(AttributeError) as c:
        column.name = "edf"

def test_wrong_types_in_constructor():
    with pytest.raises(TypeError) as c:
        column = Column("abc", "INTEGER")
