from typing import Union

from typeguard import typechecked

from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


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

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, ColumnType) and \
               self._name == other.name and \
               self._scale == other.scale and \
               self._precision == other.precision and \
               self._size == other.size and \
               self._characterSet == other.characterSet and \
               self._fraction == other.fraction and \
               self._withLocalTimeZone == other.withLocalTimeZone and \
               self._srid == other._srid
