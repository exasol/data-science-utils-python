from typing import List
from abc import ABC, abstractmethod


class ColumnPreprocessor(ABC):
    def _get_table_alias(self, target_schema: str, source_schema: str, source_table: str, source_column: str, prefix:str):
        alias = f'"{target_schema}_{source_schema}_{source_table}_{source_column}_{prefix}"'
        return alias


    def _get_target_table_name(self, target_schema:str, source_schema:str, source_table:str, source_column:str, prefix:str):
        return f'"{target_schema}"."{source_schema}_{source_table}_{source_column}_{prefix}"'

    def _get_table_qualified(self, source_schema:str, source_table:str):
        return f'"{source_schema}"."{source_table}"'

    @abstractmethod
    def create_fit_queries(self, source_schema:str, source_table:str, source_column:str, target_schema:str)->List[str]:
        pass

    @abstractmethod
    def create_from_clause_part(self, source_schema:str, source_table:str, source_column:str,
                                input_schema: str, input_table: str,
                                target_schema:str)->List[str]:
        pass

    @abstractmethod
    def create_select_clause_part(self, source_schema:str, source_table:str, source_column:str,
                                  input_schema: str, input_table: str,
                                  target_schema:str)->List[str]:
        pass