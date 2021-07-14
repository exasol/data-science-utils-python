from exasol_data_science_utils_python.preprocessing.schema.identifier import ExasolIdentifier


class SchemaName(ExasolIdentifier):
    def __init__(self, schema_name: str):
        super().__init__(schema_name)

    def fully_qualified(self) -> str:
        return self.quoted_name()

    def __eq__(self, other):
        return isinstance(other, SchemaName) and \
               self.name == other.name
