from collections import OrderedDict
from pathlib import PurePosixPath

from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.udf_context_wrapper import UDFContextWrapper


class PartialFitRegressorUDF:
    INIT_MODEL_FILE_NAME = "init_model.pkl"
    BASE_ESTIMATOR_DIRECTORY = "base_estimators"

    def __init__(self, exa):
        self.exa = exa

    def run(self, ctx):
        df = ctx.get_dataframe(1)
        model_connection_name = df["0"][0]
        column_name_list = df["1"][0].split(",")
        epochs = df["2"][0]
        batch_size = df["3"][0]
        shuffle_buffer_size = df["4"][0]
        model_connection = self.exa.get_connection(model_connection_name)
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=None
            )
        regressor_partial_fit_iterator = \
            model_bucketfs_location.download_object_from_bucketfs_via_joblib(self.INIT_MODEL_FILE_NAME)
        column_mapping = OrderedDict([(str(5 + index), column)
                                      for index, column in enumerate(column_name_list)])
        udf_conext_wrapper = UDFContextWrapper(ctx, column_mapping=column_mapping)
        for epoch in range(epochs):
            regressor_partial_fit_iterator.train(
                udf_conext_wrapper,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size)
        output_model_file_name = f"{str(self.exa.meta.session_id)}_{str(self.exa.meta.statement_id)}_{str(self.exa.meta.node_id)}_{str(self.exa.meta.vm_id)}.pkl"
        output_model_path = PurePosixPath(self.BASE_ESTIMATOR_DIRECTORY, output_model_file_name)
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     str(output_model_path))
        ctx.emit(str(output_model_path))
