import pytest

from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName

def test_fully_qualified():
    column = ColumnName("column")
    assert column.fully_qualified() == '"column"'

def test_fully_qualified_with_table():
    column = ColumnName("column", TableName("table"))
    assert column.fully_qualified() == '"table"."column"'


def test_fully_qualified_with_table_and_schema():
    column = ColumnName("column", TableName("table", schema=SchemaName("schema")))
    assert column.fully_qualified() == '"schema"."table"."column"'
