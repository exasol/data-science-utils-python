from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema
from exasol_data_science_utils_python.preprocessing.schema.table_name import Table


def test_fully_qualified():
    table = Table("table")
    assert table.fully_qualified() == '"table"'


def test_fully_qualified_with_schema():
    table = Table("table", schema=Schema("schema"))
    assert table.fully_qualified() == '"schema"."table"'
