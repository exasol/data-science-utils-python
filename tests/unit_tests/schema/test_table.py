import pytest

from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.table import Table
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


def test_set_new_name_fail():
    table = Table(TableNameImpl("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    with pytest.raises(AttributeError) as c:
        table.name = "edf"


def test_set_new_columns_fail():
    table = Table(TableNameImpl("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    with pytest.raises(AttributeError) as c:
        table.columns = [Column(ColumnName("column1"), ColumnType("INTEGER"))]


def test_wrong_types_in_constructor():
    with pytest.raises(TypeError) as c:
        column = Table("abc", "INTEGER")


def test_columns_list_is_immutable():
    table = Table(TableNameImpl("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    columns = table.columns
    columns.append(Column(ColumnName("column"), ColumnType("INTEGER")))
    assert len(columns) == 2 and len(table.columns) == 1
