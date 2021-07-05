from typing import Tuple, Any, List, Dict

from exasol_data_science_utils_python.preprocessing.sql_executor import ResultSet


class MockResultSet(ResultSet):
    def __iter__(self):
        raise NotImplemented()

    def __next__(self) -> Tuple[Any]:
        raise NotImplemented()

    def fetchone(self) -> Tuple[Any]:
        raise NotImplemented()

    def fetchmany(self, size=None) -> List[Tuple[Any]]:
        raise NotImplemented()

    def fetchall(self) -> List[Tuple[Any]]:
        raise NotImplemented()

    def rowcount(self):
        raise NotImplemented()

    def columns(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplemented()

    def column_names(self) -> List[str]:
        raise NotImplemented()

    def close(self):
        raise NotImplemented()