from exasol_data_science_utils_python.schema.dbobject_name import DBObjectName


class ConnectionObjectName(DBObjectName):

    def normalized_name_for_udfs(self) -> str:
        raise NotImplementedError()
