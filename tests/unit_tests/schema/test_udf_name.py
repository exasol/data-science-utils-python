import pytest

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.udf_name_impl import UDFNameImpl


def test_fully_qualified():
    table = UDFNameImpl("udf")
    assert table.fully_qualified() == '"udf"'


def test_fully_qualified_with_schema():
    table = UDFNameImpl("udf", schema=SchemaName("schema"))
    assert table.fully_qualified() == '"schema"."udf"'


def test_set_new_schema_fail():
    table = UDFNameImpl("abc")
    with pytest.raises(AttributeError) as c:
        table.schema_name = "edf"


def test_equality_true():
    t1 = UDFNameImpl("udf", schema=SchemaName("schema"))
    t2 = UDFNameImpl("udf", schema=SchemaName("schema"))
    assert t1 == t2


def test_equality_name_not_equal():
    t1 = UDFNameImpl("table1", schema=SchemaName("schema"))
    t2 = UDFNameImpl("table2", schema=SchemaName("schema"))
    assert t1 != t2


def test_equality_schema_not_equal():
    t1 = UDFNameImpl("udf", schema=SchemaName("schema1"))
    t2 = UDFNameImpl("udf", schema=SchemaName("schema2"))
    assert t1 != t2
