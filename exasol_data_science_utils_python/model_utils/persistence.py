from base64 import b64encode, b64decode
from io import BytesIO
from typing import Any

import joblib


def dump_to_base64_string(value: Any):
    bytesIO = BytesIO()
    joblib.dump(value, bytesIO, compress=True)
    bytes = bytesIO.getvalue()
    b64_bytes = b64encode(bytes)
    b64_string = b64_bytes.decode('ascii')
    return b64_string


def load_from_base64_string(b64_string: str):
    b64_bytes = b64_string.encode("ascii")
    bytes = b64decode(b64_bytes)
    bytesIO = BytesIO(bytes)
    value = joblib.load(bytesIO)
    return value
