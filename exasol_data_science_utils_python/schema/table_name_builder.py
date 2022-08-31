from typing import Union, Optional

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


class TableNameBuilder:

    def __init__(self,
                 name: Optional[str] = None,
                 schema: Optional[SchemaName] = None,
                 table_name: Optional[TableName] = None):
        """
        Creates a builder for TableName objects,
        either by copying a TableName object or
        using the newly provided values.
        """
        self._name = None
        self._schema_name = None
        if table_name is not None:
            self._name = table_name.name
            self._schema_name = table_name.schema_name
        if name is not None:
            self._name = name
        if schema is not None:
            self._schema_name = schema

    def with_name(self, name: str) -> "TableNameBuilder":
        self._name = name
        return self

    def with_schema_name(self, schema_name: SchemaName) -> "TableNameBuilder":
        self._schema_name = schema_name
        return self

    def build(self) -> TableName:
        return TableNameImpl(self._name, self._schema_name)
