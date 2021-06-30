from exasol_data_science_utils_python.preprocessing.schema.schema_element import SchemaElement
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class Column(SchemaElement):
    def __init__(self, column_name: str, table: Table = None):
        super().__init__(column_name)
        self.table = table

    def fully_qualified(self) -> str:
        if self.table is not None:
            return f'{self.table.fully_qualified()}."{self.name}"'
        else:
            return f'"{self.name}"'
