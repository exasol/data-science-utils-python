from exasol_data_science_utils_python.preprocessing.schema.schema import Schema


class Table:
    def __init__(self, table_name: str, schema: Schema = None):
        self.name = table_name
        self.schema = schema

    def identifier(self) -> str:
        if self.schema is not None:
            return f'{self.schema.identifier()}."{self.name}"'
        else:
            return f'"{self.name}"'
