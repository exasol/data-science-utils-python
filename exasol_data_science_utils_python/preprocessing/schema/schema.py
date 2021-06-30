from exasol_data_science_utils_python.preprocessing.schema.schema_element import SchemaElement


class Schema(SchemaElement):
    def __init__(self, schema_name: str):
        super().__init__(schema_name)

    def fully_qualified(self) -> str:
        return f'"{self.name}"'
