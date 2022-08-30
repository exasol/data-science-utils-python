import pytest

from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.table_builder import TableBuilder
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


def test_create_table_with_name_only_fail():
    with pytest.raises(TypeError):
        column = TableBuilder().with_name(TableNameImpl("table")).build()


def test_create_table_with_columns_only_fail():
    with pytest.raises(TypeError):
        column = TableBuilder().with_columns([Column(ColumnName("abc"), ColumnType("INTEGER"))]).build()


def test_create_table_with_name_and_columns():
    table = TableBuilder().with_name(TableNameImpl("table")).with_columns(
        [Column(ColumnName("column"), ColumnType("INTEGER"))]).build()
    assert table.name.name == "table" and table.columns[0].name.name == "column"
