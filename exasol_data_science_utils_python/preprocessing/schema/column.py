from exasol_data_science_utils_python.preprocessing.schema.table import Table


class Column:
    def __init__(self, column_name: str, table: Table = None):
        self.name = column_name
        self.table = table

    def identifier(self) -> str:
        if self.table is not None:
            return f'{self.table.identifier()}."{self.name}"'
        else:
            return f'"{self.name}"'
