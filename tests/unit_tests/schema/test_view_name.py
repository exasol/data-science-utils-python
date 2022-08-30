import pytest

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.view_name import ViewName


def test_fully_qualified():
    view = ViewName("view")
    assert view.fully_qualified() == '"view"'


def test_fully_qualified_with_schema():
    view = ViewName("view", schema=SchemaName("schema"))
    assert view.fully_qualified() == '"schema"."view"'


def test_set_new_schema_fail():
    view = ViewName("abc")
    with pytest.raises(AttributeError) as c:
        view.schema_name = "edf"


def test_equality_true():
    t1 = ViewName("view", schema=SchemaName("schema"))
    t2 = ViewName("view", schema=SchemaName("schema"))
    assert t1 == t2


def test_equality_name_not_equal():
    t1 = ViewName("view1", schema=SchemaName("schema"))
    t2 = ViewName("view2", schema=SchemaName("schema"))
    assert t1 != t2


def test_equality_schema_not_equal():
    t1 = ViewName("view", schema=SchemaName("schema1"))
    t2 = ViewName("view", schema=SchemaName("schema2"))
    assert t1 != t2


def test_not_equal_to_table():
    t1 = ViewName("view", schema=SchemaName("schema"))
    t2 = TableName("view", schema=SchemaName("schema"))
    assert t1 != t2
