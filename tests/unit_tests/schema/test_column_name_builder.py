import pytest

from exasol_data_science_utils_python.schema.column_name import ColumnName
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


def test_using_empty_constructor():
    with pytest.raises(TypeError):
        column_name = ColumnNameBuilder().build()


def test_using_constructor_name_only():
    column_name = ColumnNameBuilder(name="table").build()
    assert column_name.name == "table" and column_name.table_name is None


def test_using_constructor_table():
    column_name = ColumnNameBuilder(name="table", table_name=TableNameImpl("table")).build()
    assert column_name.name == "table" and column_name.table_name.name is "table"


def test_using_with_name_only():
    column_name = ColumnNameBuilder().with_name("table").build()
    assert column_name.name == "table" and column_name.table_name is None


def test_using_with_table():
    column_name = ColumnNameBuilder().with_name("table").with_table_name(TableNameImpl("table")).build()
    assert column_name.name == "table" and column_name.table_name.name == "table"


def test_from_existing_using_with_table():
    source_column_name = ColumnName("table")
    column_name = ColumnNameBuilder(column_name=source_column_name).with_table_name(TableNameImpl("table")).build()
    assert source_column_name.name == "table" \
           and source_column_name.table_name is None \
           and column_name.name == "table" \
           and column_name.table_name.name == "table"


def test_from_existing_using_with_name():
    source_column_name = ColumnName("column", TableNameImpl("table"))
    column_name = ColumnNameBuilder(column_name=source_column_name).with_name("column1").build()
    assert source_column_name.name == "column" \
           and source_column_name.table_name.name == "table" \
           and column_name.table_name.name == "table" \
           and column_name.name == "column1"


def test_from_existing_and_new_table_in_constructor():
    source_column_name = ColumnName("column")
    column_name = ColumnNameBuilder(table_name=TableNameImpl("table"),
                                    column_name=source_column_name).build()
    assert source_column_name.name == "column" \
           and source_column_name.table_name is None \
           and column_name.name == "column" \
           and column_name.table_name.name == "table"


def test_from_existing_and_new_name_in_constructor():
    source_column_name = ColumnName("column", TableNameImpl("table"))
    column_name = ColumnNameBuilder(name="column1",
                                    column_name=source_column_name).build()
    assert source_column_name.name == "column" \
           and source_column_name.table_name.name == "table" \
           and column_name.table_name.name == "table" \
           and column_name.name == "column1"
