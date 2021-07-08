from collections import OrderedDict

from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.udf_context_wrapper import UDFContextWrapper


class ScoreRegressorUDF:

    def __init__(self, exa):
        self.exa = exa

    def run(self, ctx):
        df = ctx.get_dataframe(1)
        model_connection_name = df["0"][0]
        model_path = df["1"][0]
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
        score_iterator = \
            model_bucketfs_location.download_object_from_bucketfs_via_joblib(model_path) # type: ScoreIterator
        column_mapping = OrderedDict([(str(4 + index), column)
                                      for index, column in enumerate(column_name_list)])
        udf_conext_wrapper = UDFContextWrapper(ctx, column_mapping=column_mapping)
        score_sum, score_count = score_iterator.compute_score(
            udf_conext_wrapper,
            batch_size=batch_size
        )
        score_sum = float(score_sum)
        ctx.emit(score_sum, score_count)
