from typing import Union, Optional

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl
from exasol_data_science_utils_python.schema.view_name import ViewName
from exasol_data_science_utils_python.schema.view_name_impl import ViewNameImpl


class ViewNameBuilder:

    def __init__(self,
                 name: Optional[str] = None,
                 schema: Optional[SchemaName] = None,
                 view_name: Optional[ViewName] = None):
        """
        Creates a builder for ViewName objects,
        either by copying a ViewName object or
        using the newly provided values.
        """
        self._name = None
        self._schema_name = None
        if view_name is not None:
            self._name = view_name.name
            self._schema_name = view_name.schema_name
        if name is not None:
            self._name = name
        if schema is not None:
            self._schema_name = schema

    def with_name(self, name: str) -> "ViewNameBuilder":
        self._name = name
        return self

    def with_schema_name(self, schema_name: SchemaName) -> "ViewNameBuilder":
        self._schema_name = schema_name
        return self

    def build(self) -> ViewName:
        return ViewNameImpl(self._name, self._schema_name)
