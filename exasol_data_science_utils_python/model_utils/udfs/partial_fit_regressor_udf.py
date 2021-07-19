from collections import OrderedDict
from pathlib import PurePosixPath

from tenacity import Retrying, stop_after_delay, wait_fixed

from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.udf_context_wrapper import UDFContextWrapper

MODEL_CONNECTION_NAME_PARAMETER = "0"
PATH_UNDER_MODEL_CONNECTION_PARAMETER = "1"
DOWNLOAD_RETRY_SECONDS = "2"
COLUMN_NAME_LIST_PARAMETER = "3"
EPOCHS_PARAMETER = "4"
BATCH_SIZE_PARAMETER = "5"
SHUFFLE_BUFFER_SIZE_PARAMETER = "6"
FIRST_VARARG_PARAMETER = 7


class PartialFitRegressorUDF:
    INIT_MODEL_FILE_NAME = "init_model.pkl"
    BASE_MODEL_DIRECTORY = "base_models"

    def __init__(self, exa):
        self.exa = exa
        self.counter = 0

    def run(self, ctx):
        df = ctx.get_dataframe(1)
        model_connection_name = df[MODEL_CONNECTION_NAME_PARAMETER][0]
        path_under_model_connection = PurePosixPath(df[PATH_UNDER_MODEL_CONNECTION_PARAMETER][0])
        download_retry_seconds = df[DOWNLOAD_RETRY_SECONDS][0].item()
        column_name_list = df[COLUMN_NAME_LIST_PARAMETER][0].split(",")
        epochs = df[EPOCHS_PARAMETER][0].item()
        batch_size = df[BATCH_SIZE_PARAMETER][0].item()
        shuffle_buffer_size = df[SHUFFLE_BUFFER_SIZE_PARAMETER][0].item()
        model_connection = self.exa.get_connection(model_connection_name)
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=path_under_model_connection
            )

        retryer = Retrying(stop=stop_after_delay(download_retry_seconds), wait=wait_fixed(1), reraise=True)
        regressor_partial_fit_iterator = \
            retryer(model_bucketfs_location.download_object_from_bucketfs_via_joblib, self.INIT_MODEL_FILE_NAME)

        column_mapping = OrderedDict([(str(FIRST_VARARG_PARAMETER + index), column)
                                      for index, column in enumerate(column_name_list)])
        udf_conext_wrapper = UDFContextWrapper(ctx, column_mapping=column_mapping)
        for epoch in range(epochs):
            regressor_partial_fit_iterator.train(
                udf_conext_wrapper,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size)
        output_model_file_name = f"{str(self.exa.meta.session_id)}_{str(self.exa.meta.statement_id)}_{str(self.exa.meta.node_id)}_{str(self.exa.meta.vm_id)}_{str(self.counter)}.pkl"
        output_model_path = PurePosixPath(self.BASE_MODEL_DIRECTORY, output_model_file_name)
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     str(output_model_path))
        score_sum, score_count = regressor_partial_fit_iterator.compute_score(
            udf_conext_wrapper,
            batch_size=batch_size
        )
        score_sum = float(score_sum)
        ctx.emit(model_connection_name, str(path_under_model_connection), str(output_model_path),
                 score_sum, score_count)
        self.counter += 1
