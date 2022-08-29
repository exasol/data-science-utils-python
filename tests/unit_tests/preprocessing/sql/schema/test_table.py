import pytest

from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.table import Table
from exasol_data_science_utils_python.schema.table_name import TableName


def test_set_new_name_fail():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    with pytest.raises(AttributeError) as c:
        table.name = "edf"


def test_set_new_columns_fail():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    with pytest.raises(AttributeError) as c:
        table.columns = [Column(ColumnName("column1"), ColumnType("INTEGER"))]


def test_set_new_columns_fail():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    with pytest.raises(AttributeError) as c:
        table.is_view = True


def test_wrong_types_in_constructor():
    with pytest.raises(TypeError) as c:
        column = Table("abc", "INTEGER")

def test_columns_list_is_immutable():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    columns = table.columns
    columns.append(Column(ColumnName("column"), ColumnType("INTEGER")))
    assert len(columns) == 2 and len(table.columns) == 1

def test_properties_is_view_default():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))])
    assert table.name.name=="table" and table.columns[0].name.name=="column" and table.is_view == False

def test_properties_is_view_true():
    table = Table(TableName("table"), [Column(ColumnName("column"), ColumnType("INTEGER"))], is_view=True)
    assert table.name.name=="table" and table.columns[0].name.name=="column" and table.is_view == True