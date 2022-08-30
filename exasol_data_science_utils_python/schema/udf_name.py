from abc import ABC

from typeguard import typechecked

from exasol_data_science_utils_python.schema.dbobject_name_with_schema import DBObjectNameWithSchema
from exasol_data_science_utils_python.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class UDFName(DBObjectNameWithSchema):
    @typechecked
    def __init__(self, udf_name: str, schema: SchemaName = None):
        super().__init__(udf_name, schema)