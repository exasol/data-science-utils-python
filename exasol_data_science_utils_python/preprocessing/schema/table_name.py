from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.identifier import ExasolIdentifier


class TableName(ExasolIdentifier):
    def __init__(self, table_name: str, schema: SchemaName = None):
        super().__init__(table_name)
        self.schema = schema

    def fully_qualified(self) -> str:
        if self.schema is not None:
            return f'{self.schema.fully_qualified()}.{self.quoted_name()}'
        else:
            return self.quoted_name()

    def __eq__(self, other):
        return isinstance(other, TableName) and \
               self.name == other.name and \
               self.schema == other.schema
