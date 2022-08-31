from typing import Union, Optional

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl
from exasol_data_science_utils_python.schema.udf_name import UDFName
from exasol_data_science_utils_python.schema.udf_name_impl import UDFNameImpl
from exasol_data_science_utils_python.schema.view_name import ViewName
from exasol_data_science_utils_python.schema.view_name_impl import ViewNameImpl


class UDFNameBuilder:

    def __init__(self,
                 name: Optional[str] = None,
                 schema: Optional[SchemaName] = None,
                 udf_name: Optional[UDFName] = None):
        """
        Creates a builder for UDFName objects,
        either by copying a UDFName object or
        using the newly provided values.
        """
        self._name = None
        self._schema_name = None
        if udf_name is not None:
            self._name = udf_name.name
            self._schema_name = udf_name.schema_name
        if name is not None:
            self._name = name
        if schema is not None:
            self._schema_name = schema

    def with_name(self, name: str) -> "UDFNameBuilder":
        self._name = name
        return self

    def with_schema_name(self, schema_name: SchemaName) -> "UDFNameBuilder":
        self._schema_name = schema_name
        return self

    def build(self) -> UDFName:
        return UDFNameImpl(self._name, self._schema_name)