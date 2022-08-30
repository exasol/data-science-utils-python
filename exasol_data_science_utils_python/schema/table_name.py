from typeguard import typechecked

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_like_name import TableLikeName


class TableName(TableLikeName):
    @typechecked
    def __init__(self, table_name: str, schema: SchemaName = None):
        super().__init__(table_name)
        self._schema_name = schema