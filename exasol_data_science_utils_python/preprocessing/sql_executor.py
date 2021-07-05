from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class ResultSet(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self) -> Tuple[Any]:
        pass

    @abstractmethod
    def fetchone(self) -> Tuple[Any]:
        pass

    @abstractmethod
    def fetchmany(self, size=None) -> List[Tuple[Any]]:
        pass

    @abstractmethod
    def fetchall(self) -> List[Tuple[Any]]:
        pass

    @abstractmethod
    def rowcount(self):
        pass

    @abstractmethod
    def columns(self) -> Dict[str, Dict[str, Any]]:
        """
        dataType (object) => column metadata
            type (string) => column data type
            precision (number, optional) => column precision
            scale (number, optional) => column scale
            size (number, optional) => maximum size in bytes of a column value
            characterSet (string, optional) => character encoding of a text column
            withLocalTimeZone (true | false, optional) => specifies if a timestamp has a local time zone
            fraction (number, optional) => fractional part of number
            srid (number, optional) => spatial reference system identifier
        :return:
        """
        pass

    @abstractmethod
    def column_names(self) -> List[str]:
        pass

    @abstractmethod
    def close(self):
        pass


class SQLExecutor(ABC):
    @abstractmethod
    def execute(self, sql: str) -> ResultSet:
        pass
