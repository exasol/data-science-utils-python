from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName


def test_create_column_with_name_only():
    column_name = ColumnNameBuilder().with_name("column").build()
    assert column_name.name == "column" and column_name.table_name is None


def test_create_column_with_table():
    column_name = ColumnNameBuilder().with_name("column").with_table_name(TableName("table")).build()
    assert column_name.name == "column" and column_name.table_name.name == "table"


def test_create_column_from_existing_changing_table():
    source_column_name = ColumnName("column")
    column_name = ColumnNameBuilder(source_column_name).with_table_name(TableName("table")).build()
    assert source_column_name.name == "column" \
           and source_column_name.table_name is None \
           and column_name.name == "column" \
           and column_name.table_name.name == "table"


def test_create_column_from_existing_changing_name():
    source_column_name = ColumnName("column", TableName("table"))
    column_name = ColumnNameBuilder(source_column_name).with_name("column1").build()
    assert source_column_name.name == "column" \
           and source_column_name.table_name.name == "table" \
           and column_name.table_name.name == "table" \
           and column_name.name == "column1"
