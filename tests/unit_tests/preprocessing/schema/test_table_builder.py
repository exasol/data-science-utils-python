import pytest

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.schema.table_builder import TableBuilder
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName


def test_create_table_with_name_only_fail():
    with pytest.raises(TypeError):
        column = TableBuilder().with_name(TableName("table")).build()


def test_create_table_with_columns_only_fail():
    with pytest.raises(TypeError):
        column = TableBuilder().with_columns([Column(ColumnName("abc"), ColumnType("INTEGER"))]).build()


def test_create_table_with_is_view_only_fail():
    with pytest.raises(TypeError):
        column = TableBuilder().with_is_view(True).build()


def test_create_table_with_name_and_columns():
    table = TableBuilder().with_name(TableName("table")).with_columns(
        [Column(ColumnName("column"), ColumnType("INTEGER"))]).build()
    assert table.name.name == "table" and table.columns[0].name.name == "column" and table.is_view == False


def test_create_table_with_all():
    table = TableBuilder().with_name(TableName("table")).with_columns(
        [Column(ColumnName("column"), ColumnType("INTEGER"))]).with_is_view(True).build()
    assert table.name.name == "table" and table.columns[0].name.name == "column" and table.is_view == True
