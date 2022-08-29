from exasol_data_science_utils_python.schema import SchemaName
from exasol_data_science_utils_python.schema import TableName
from exasol_data_science_utils_python.schema import TableNameBuilder


def test_create_table_with_name_only():
    table_name = TableNameBuilder().with_name("column").build()
    assert table_name.name == "column" and table_name.schema_name is None


def test_create_table_with_schema():
    table_name = TableNameBuilder().with_name("table").with_schema_name(SchemaName("schema")).build()
    assert table_name.name == "table" and table_name.schema_name.name == "schema"


def test_create_table_from_existing_changing_schema():
    source_table_name = TableName("table")
    table_name = TableNameBuilder(source_table_name).with_schema_name(SchemaName("schema")).build()
    assert source_table_name.name == "table" \
           and source_table_name.schema_name is None \
           and table_name.name == "table" \
           and table_name.schema_name.name == "schema"


def test_create_column_from_existing_changing_name():
    source_table_name = TableName("table", SchemaName("schema"))
    table_name = TableNameBuilder(source_table_name).with_name("table1").build()
    assert source_table_name.name == "table" \
           and source_table_name.schema_name.name == "schema" \
           and table_name.schema_name.name == "schema" \
           and table_name.name == "table1"
