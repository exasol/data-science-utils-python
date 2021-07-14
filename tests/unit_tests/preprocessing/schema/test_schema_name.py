from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema


def test_fully_qualified():
    schema = Schema("schema")
    assert schema.fully_qualified() == '"schema"'
