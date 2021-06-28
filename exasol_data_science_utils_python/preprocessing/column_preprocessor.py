from abc import ABC, abstractmethod
from typing import List

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class ColumnPreprocessor(ABC):

    def _get_table_alias(self, target_schema: Schema, source_column: Column, prefix: str):
        target_schema_name = target_schema.name
        source_schema_name = source_column.table.schema.name
        source_table_name = source_column.table.name
        alias = Table(f"{target_schema_name}_{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}")
        return alias

    def _get_target_table(self, target_schema: Schema, source_column: Column, prefix: str):
        source_schema_name = source_column.table.schema.name
        source_table_name = source_column.table.name
        target_table = Table(f"{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}", target_schema)
        return target_table

    @abstractmethod
    def create_fit_queries(self,
                           source_column: Column,
                           target_schema: Schema) -> List[str]:
        pass

    @abstractmethod
    def create_transform_from_clause_part(self,
                                          source_column: Column,
                                          input_table: Table,
                                          target_schema: Schema) -> List[str]:
        pass

    @abstractmethod
    def create_transform_select_clause_part(self,
                                            source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        pass
