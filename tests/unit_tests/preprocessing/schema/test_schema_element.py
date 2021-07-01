import pytest

from exasol_data_science_utils_python.preprocessing.schema.schema_element import SchemaElement


@pytest.mark.parametrize("test_input",
                         [
                             "A",
                             "a",
                             "B_",
                             "Z1",
                             "Q\uFE33",
                             "Ü"
                         ])
def test_name_valid(test_input):
    assert SchemaElement.validate_name(test_input) == True


@pytest.mark.parametrize("test_input",
                         [
                             "_",
                             ",",
                             ";",
                             ":",
                             "1",
                             "\uFE33",
                             '"',
                             'A"'
                         ])
def test_name_invalid(test_input):
    assert SchemaElement.validate_name(test_input) == False

def test_quote():
    class TestSchemaElement(SchemaElement):
        def __init__(self, name: str):
            super().__init__(name)

        def fully_qualified(self) -> str:
            raise NotImplemented()

    quoted_name = TestSchemaElement("ABC").quoted_name()
    assert quoted_name == '"ABC"'
