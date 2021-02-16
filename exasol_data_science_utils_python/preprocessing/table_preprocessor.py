import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor


class ColumnPreprocesserDefinition:
    def __init__(self, column_name: str, column_preprocessor: ColumnPreprocessor):
        self.column_preprocessor = column_preprocessor
        self.column_name = column_name


class TablePreprocessor():
    def __init__(self, target_schema: str, source_schema: str, source_table: str,
                 column_preprocessor_defintions: List[ColumnPreprocesserDefinition]):
        self.column_preprocessor_defintions = column_preprocessor_defintions
        self.source_table = source_table
        self.source_schema = source_schema
        self.target_schema = target_schema

    def create_fit_queries(self) -> List[str]:
        result = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            result.extend(column_preprocessor_defintion.column_preprocessor.create_fit_queries(
                self.source_schema, self.source_table, column_preprocessor_defintion.column_name,
                self.target_schema))
        return result

    def create_transform_query(self, input_schema: str, input_table: str) -> str:
        select_clause_parts = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            select_clause_parts.extend(
                column_preprocessor_defintion.column_preprocessor.create_select_clause_part(
                    self.source_schema, self.source_table, column_preprocessor_defintion.column_name,
                    input_schema, input_table,
                    self.target_schema))
        select_clause_parts_str = ",\n".join(select_clause_parts)

        from_clause_parts = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            from_clause_parts.extend(
                column_preprocessor_defintion.column_preprocessor.create_from_clause_part(
                    self.source_schema, self.source_table, column_preprocessor_defintion.column_name,
                    input_schema, input_table,
                    self.target_schema))
        from_clause_parts_str = "\n".join(from_clause_parts)

        query = textwrap.dedent(f"""
CREATE OR REPLACE TABLE "{self.target_schema}"."{input_schema}_{input_table}_TRANSFORMED" AS
SELECT
{select_clause_parts_str}
FROM "{input_schema}"."{input_table}"
{from_clause_parts_str}
""")
#        query="\n".join([line.strip() for line in query.split("\n") if line.strip() != ""])
        return query
