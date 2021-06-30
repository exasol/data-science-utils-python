from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.schema_element import SchemaElement


class Table(SchemaElement):
    def __init__(self, table_name: str, schema: Schema = None):
        super().__init__(table_name)
        self.schema = schema

    def fully_qualified(self) -> str:
        if self.schema is not None:
            return f'{self.schema.fully_qualified()}."{self.name}"'
        else:
            return f'"{self.name}"'
