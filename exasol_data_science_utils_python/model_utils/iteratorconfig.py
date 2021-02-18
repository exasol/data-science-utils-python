from typing import List

import jsonpickle


class IteratorConfig():
    def __init__(self,
                 categorical_input_column_names: List[str],
                 numerical_input_column_names: List[str],
                 target_column_name: str,
                 input_column_category_counts: List[int],
                 target_classes: int):
        self.target_classes = target_classes
        self.input_column_category_counts = input_column_category_counts
        self.target_column_name = target_column_name
        self.numerical_input_column_names = numerical_input_column_names
        self.categorical_input_column_names = categorical_input_column_names

    def is_compatible(self, iterator_config: "IteratorConfig") -> bool:
        result = \
            self.categorical_input_column_names == iterator_config.categorical_input_column_names and \
            self.input_column_category_counts == iterator_config.input_column_category_counts and \
            self.numerical_input_column_names == iterator_config.numerical_input_column_names and \
            self.target_classes == iterator_config.target_classes and \
            self.target_column_name == iterator_config.target_column_name
        return result


    @classmethod
    def from_json(self, json_string: str) -> "IteratorConfig":
        loaded_object = jsonpickle.decode(json_string)
        if not isinstance(loaded_object, IteratorConfig):
            raise TypeError("Type %s of loaded object does not match %s" % (type(loaded_object), IteratorConfig))
        return loaded_object

    def to_json(self, indent=4):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=indent)
        return jsonpickle.encode(self)
