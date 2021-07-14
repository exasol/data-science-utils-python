from typing import Union

from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName


class TableNameBuilder:
    def __init__(self, table_name: Union[TableName, None] = None):
        if table_name is not None:
            self._name = table_name.name
            self._schema_name = table_name.schema_name
        else:
            self._name = None
            self._schema_name = None

    def with_name(self, name: str) -> "TableNameBuilder":
        self._name = name
        return self

    def with_schema_name(self, schema_name: SchemaName) -> "TableNameBuilder":
        self._schema_name = schema_name
        return self

    def build(self) -> TableName:
        return TableName(self._name, self._schema_name)
