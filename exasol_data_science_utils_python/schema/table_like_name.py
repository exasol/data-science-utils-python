from abc import ABC

from typeguard import typechecked

from exasol_data_science_utils_python.schema.dbobject_name_with_schema import DBObjectNameWithSchema
from exasol_data_science_utils_python.schema.schema_name import SchemaName


class TableLikeName(DBObjectNameWithSchema, ABC):
    @typechecked
    def __init__(self, table_like_name: str, schema: SchemaName = None):
        super().__init__(table_like_name, schema)
