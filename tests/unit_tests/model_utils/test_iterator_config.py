from exasol_data_science_utils_python.model_utils.iteratorconfig import IteratorConfig


def test_iterator_config():
    config = IteratorConfig(numerical_input_column_names=["t1"],
                            categorical_input_column_names=["t2"],
                            target_column_name="t3",
                            target_classes=10,
                            input_column_category_counts=[10])
    json_string = config.to_json()
    print(json_string)
    new_config = IteratorConfig.from_json(json_string)
    assert config.target_classes == new_config.target_classes and \
           config.input_column_category_counts == new_config.input_column_category_counts and \
           config.target_column_name == new_config.target_column_name and \
           config.numerical_input_column_names == new_config.numerical_input_column_names and \
           config.categorical_input_column_names == new_config.categorical_input_column_names
