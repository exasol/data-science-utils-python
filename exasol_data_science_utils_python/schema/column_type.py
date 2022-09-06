import dataclasses
from typing import Optional

import typeguard


@dataclasses.dataclass(frozen=True, repr=True, eq=True)
class ColumnType:
    name: str
    precision: Optional[int] = None
    scale: Optional[int] = None
    size: Optional[int] = None
    characterSet: Optional[str] = None
    withLocalTimeZone: Optional[bool] = None
    fraction: Optional[int] = None
    srid: Optional[int] = None

    def __post_init__(self):
        typeguard.check_type(value=self.name,
                             expected_type=str,
                             argname="name")
        typeguard.check_type(value=self.precision,
                             expected_type=Optional[int],
                             argname="precision")
        typeguard.check_type(value=self.scale,
                             expected_type=Optional[int],
                             argname="scale")
        typeguard.check_type(value=self.size,
                             expected_type=Optional[int],
                             argname="size")
        typeguard.check_type(value=self.characterSet,
                             expected_type=Optional[str],
                             argname="characterSet")
        typeguard.check_type(value=self.withLocalTimeZone,
                             expected_type=Optional[bool],
                             argname="withLocalTimeZone")
        typeguard.check_type(value=self.fraction,
                             expected_type=Optional[int],
                             argname="fraction")
        typeguard.check_type(value=self.srid,
                             expected_type=Optional[int],
                             argname="srid")
