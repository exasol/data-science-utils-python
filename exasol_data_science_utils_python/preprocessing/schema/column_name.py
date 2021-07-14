from exasol_data_science_utils_python.preprocessing.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName


class ColumnName(ExasolIdentifier):
    def __init__(self, column_name: str, table: TableName = None):
        super().__init__(column_name)
        self.table = table

    def fully_qualified(self) -> str:
        if self.table is not None:
            return f'{self.table.fully_qualified()}.{self.quoted_name()}'
        else:
            return self.quoted_name()

    def __eq__(self, other):
        return isinstance(other, ColumnName) and \
               self.name == other.name and \
               self.table == other.table
