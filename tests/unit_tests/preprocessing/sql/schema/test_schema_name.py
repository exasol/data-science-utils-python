from exasol_data_science_utils_python.schema.schema_name import SchemaName


def test_fully_qualified():
    schema = SchemaName("schema")
    assert schema.fully_qualified() == '"schema"'
