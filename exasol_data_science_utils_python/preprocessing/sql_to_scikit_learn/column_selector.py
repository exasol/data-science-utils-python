from abc import ABC, abstractmethod

from exasol_data_science_utils_python.schema.column import Column


class ColumnSelector(ABC):

    @abstractmethod
    def column_accepted(self, column: Column):
        pass
