import pytest

from exasol_data_science_utils_python.preprocessing.schema.column_name import Column
from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema
from exasol_data_science_utils_python.preprocessing.schema.table_name import Table

def test_fully_qualified():
    column = Column("column")
    assert column.fully_qualified() == '"column"'

def test_fully_qualified_with_table():
    column = Column("column", Table("table"))
    assert column.fully_qualified() == '"table"."column"'


def test_fully_qualified_with_table_and_schema():
    column = Column("column", Table("table", schema=Schema("schema")))
    assert column.fully_qualified() == '"schema"."table"."column"'
