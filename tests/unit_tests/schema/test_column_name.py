import pytest

from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder


def test_fully_qualified():
    column = ColumnName("column")
    assert column.fully_qualified == '"column"'


def test_fully_qualified_with_table():
    column = ColumnName("column", TableNameBuilder.create("table"))
    assert column.fully_qualified == '"table"."column"'


def test_fully_qualified_with_table_and_schema():
    column = ColumnName("column", TableNameBuilder.create("table", schema=SchemaName("schema")))
    assert column.fully_qualified == '"schema"."table"."column"'


def test_set_new_table_fail():
    column = ColumnName("abc")
    with pytest.raises(AttributeError) as c:
        column.table_name = "edf"


def test_equality():
    column1 = ColumnName("column", TableNameBuilder.create("table"))
    column2 = ColumnName("column", TableNameBuilder.create("table"))
    assert column1 == column2


def test_inequality_name():
    column1 = ColumnName("column1", TableNameBuilder.create("table"))
    column2 = ColumnName("column2", TableNameBuilder.create("table"))
    assert column1 != column2


def test_inequality_table():
    column1 = ColumnName("column", TableNameBuilder.create("table1"))
    column2 = ColumnName("column", TableNameBuilder.create("table2"))
    assert column1 != column2


def test_hash_equality():
    column1 = ColumnName("column", TableNameBuilder.create("table"))
    column2 = ColumnName("column", TableNameBuilder.create("table"))
    assert hash(column1) == hash(column2)


def test_hash_inequality_name():
    column1 = ColumnName("column1", TableNameBuilder.create("table"))
    column2 = ColumnName("column2", TableNameBuilder.create("table"))
    assert hash(column1) != hash(column2)


def test_hash_inequality_type():
    column1 = ColumnName("column", TableNameBuilder.create("table1"))
    column2 = ColumnName("column", TableNameBuilder.create("table2"))
    assert hash(column1) != hash(column2)
