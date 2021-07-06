from exasol_data_science_utils_python.preprocessing.schema.schema_element import SchemaElement
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class Column(SchemaElement):
    def __init__(self, column_name: str, table: Table = None):
        super().__init__(column_name)
        self.table = table

    def fully_qualified(self) -> str:
        if self.table is not None:
            return f'{self.table.fully_qualified()}.{self.quoted_name()}'
        else:
            return self.quoted_name()

    def __eq__(self, other):
        return isinstance(other, Column) and \
               self.name == other.name and \
               self.table == other.table
