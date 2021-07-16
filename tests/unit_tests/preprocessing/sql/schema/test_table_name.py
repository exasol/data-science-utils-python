import pytest

from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName


def test_fully_qualified():
    table = TableName("table")
    assert table.fully_qualified() == '"table"'


def test_fully_qualified_with_schema():
    table = TableName("table", schema=SchemaName("schema"))
    assert table.fully_qualified() == '"schema"."table"'


def test_set_new_schema_fail():
    table = TableName("abc")
    with pytest.raises(AttributeError) as c:
        table.schema_name = "edf"
