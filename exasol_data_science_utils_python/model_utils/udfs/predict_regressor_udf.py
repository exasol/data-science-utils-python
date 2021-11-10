from collections import OrderedDict
from pathlib import PurePosixPath

from exasol_data_science_utils_python.model_utils.prediction_iterator import PredictionIterator
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.udf_context_wrapper import UDFContextWrapper

MODEL_CONNECTION_NAME_PARAMETER = "0"
PATH_UNDER_MODEL_CONNECTION_PARAMETER = "1"
COLUMN_NAME_LIST_PARAMETER = "2"
EPOCHS_PARAMETER = "3"
BATCH_SIZE_PARAMETER = "4"
SHUFFLE_BUFFER_SIZE_PARAMETER = "5"
FIRST_VARARG_PARAMETER = 6


class PredictRegressorUDF:

    def __init__(self, exa):
        self.exa = exa

    def run(self, ctx):
        df = ctx.get_dataframe(1)
        model_connection_name = df["0"][0]
        model_path = PurePosixPath(df["1"][0])
        column_name_list = df["2"][0].split(",")
        batch_size = df["3"][0].item()
        model_connection = self.exa.get_connection(model_connection_name)
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=None
            )
        prediction_iterator = \
            model_bucketfs_location.download_object_from_bucketfs_via_joblib(model_path)  # type: PredictionIterator
        column_mapping = OrderedDict([(str(4 + index), column)
                                      for index, column in enumerate(column_name_list)])
        udf_conext_wrapper = UDFContextWrapper(ctx, column_mapping=column_mapping)
        prediction_iterator.predict(
            udf_conext_wrapper,
            batch_size=batch_size,
            result_callback=lambda df: ctx.emit(df)
        )
