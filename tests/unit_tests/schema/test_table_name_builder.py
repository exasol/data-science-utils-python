from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


def test_create_table_with_name_only():
    table_name = TableNameBuilder().with_name("column").build()
    assert table_name.name == "column" and table_name.schema_name is None


def test_create_table_with_schema():
    table_name = TableNameBuilder().with_name("table").with_schema_name(SchemaName("schema")).build()
    assert table_name.name == "table" and table_name.schema_name.name == "schema"


def test_create_table_from_existing_changing_schema():
    source_table_name = TableNameImpl("table")
    table_name = TableNameBuilder(source_table_name).with_schema_name(SchemaName("schema")).build()
    assert source_table_name.name == "table" \
           and source_table_name.schema_name is None \
           and table_name.name == "table" \
           and table_name.schema_name.name == "schema"


def test_create_column_from_existing_changing_name():
    source_table_name = TableNameImpl("table", SchemaName("schema"))
    table_name = TableNameBuilder(source_table_name).with_name("table1").build()
    assert source_table_name.name == "table" \
           and source_table_name.schema_name.name == "schema" \
           and table_name.schema_name.name == "schema" \
           and table_name.name == "table1"
