import pytest

from exasol_data_science_utils_python.schema import ColumnBuilder
from exasol_data_science_utils_python.schema import ColumnName
from exasol_data_science_utils_python.schema import ColumnType


def test_create_column_with_name_only():
    with pytest.raises(TypeError):
        column = ColumnBuilder().with_name(ColumnName("column")).build()

def test_create_column_with_type_only():
    with pytest.raises(TypeError):
        column = ColumnBuilder().with_type(type=ColumnType("INTEGER")).build()

def test_create_column_with_name_and_type():
    column = ColumnBuilder().with_name(ColumnName("column")).with_type(type=ColumnType("INTEGER")).build()
    assert column.name.name == "column" and column.type.name == "INTEGER"


