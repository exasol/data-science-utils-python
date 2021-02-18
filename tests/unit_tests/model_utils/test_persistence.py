from exasol_data_science_utils_python.model_utils.persistence import dump_to_base64_string, load_from_base64_string


class MyClass:
    def __init__(self, a, b, c):
        self.c = c
        self.b = b
        self.a = a


def test_persistence():
    obj = MyClass("a", 2, {"c": 4})
    b64_string = dump_to_base64_string(obj)
    new_obj = load_from_base64_string(b64_string)
    assert isinstance(b64_string, str) and \
           isinstance(new_obj, MyClass) and \
           new_obj.a == obj.a and new_obj.b == obj.b and new_obj.c == obj.c
