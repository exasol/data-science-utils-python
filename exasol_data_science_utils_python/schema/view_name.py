from typeguard import typechecked

from exasol_data_science_utils_python.schema.identifier import ExasolIdentifier
from exasol_data_science_utils_python.schema.table_like_name import TableLikeName
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object
from exasol_data_science_utils_python.schema.schema_name import SchemaName


class ViewName(TableLikeName):
    @typechecked
    def __init__(self, view_name: str, schema: SchemaName = None):
        super().__init__(view_name)
        self._schema_name = schema