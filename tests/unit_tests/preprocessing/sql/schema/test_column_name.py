import pytest

from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName


def test_fully_qualified():
    column = ColumnName("column")
    assert column.fully_qualified() == '"column"'


def test_fully_qualified_with_table():
    column = ColumnName("column", TableName("table"))
    assert column.fully_qualified() == '"table"."column"'


def test_fully_qualified_with_table_and_schema():
    column = ColumnName("column", TableName("table", schema=SchemaName("schema")))
    assert column.fully_qualified() == '"schema"."table"."column"'

def test_set_new_table_fail():
    column = ColumnName("abc")
    with pytest.raises(AttributeError) as c:
        column.table_name = "edf"