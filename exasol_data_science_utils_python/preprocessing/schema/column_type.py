from typing import Union

from typeguard import typechecked


class ColumnType:
    @typechecked
    def __init__(
            self,
            name: str,
            precision: Union[int, None] = None,
            scale: Union[int, None] = None,
            size: Union[int, None] = None,
            characterSet: Union[str, None] = None,
            withLocalTimeZone: Union[bool, None] = None,
            fraction: Union[int, None] = None,
            srid: Union[int, None] = None
    ):
        self._srid = srid
        self._fraction = fraction
        self._withLocalTimeZone = withLocalTimeZone
        self._characterSet = characterSet
        self._size = size
        self._scale = scale
        self._precision = precision
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def srid(self) -> Union[int, None]:
        return self._srid

    @property
    def fraction(self) -> Union[int, None]:
        return self._fraction

    @property
    def withLocalTimeZone(self) -> Union[bool, None]:
        return self._withLocalTimeZone

    @property
    def characterSet(self) -> Union[str, None]:
        return self._characterSet

    @property
    def size(self) -> Union[int, None]:
        return self._size

    @property
    def scale(self) -> Union[int, None]:
        return self._scale

    @property
    def precision(self) -> Union[int, None]:
        return self._precision
