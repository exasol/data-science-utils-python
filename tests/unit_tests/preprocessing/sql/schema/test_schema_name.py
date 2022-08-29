from exasol_data_science_utils_python.schema import SchemaName


def test_fully_qualified():
    schema = SchemaName("schema")
    assert schema.fully_qualified() == '"schema"'
