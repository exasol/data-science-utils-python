from pathlib import PurePosixPath

from tenacity import stop_after_delay, Retrying, wait_fixed

from exasol_data_science_utils_python.model_utils.model_aggregator import combine_to_voting_regressor
from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.score_iterator import ScoreIterator
from exasol_data_science_utils_python.udf_utils.abstract_bucketfs_location import AbstractBucketFSLocation
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory


class CombineToVotingRegressorUDF:
    COMBINED_MODEL_DIRECTORY = "combined_models"

    def __init__(self, exa):
        self.exa = exa
        self.counter = 0

    def run(self, ctx):
        model_connection_name = ctx.model_connection
        path_under_model_connection = ctx.path_under_model_connection
        model_connection = self.exa.get_connection(model_connection_name)
        download_retry_seconds = ctx.download_retry_seconds
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=path_under_model_connection
            )

        models = []
        table_preprocessor = None
        while True:
            input_model_path = ctx.input_model_path
            retryer = Retrying(stop=stop_after_delay(download_retry_seconds), wait=wait_fixed(1), reraise=True)
            try:
                iterator = retryer(self.load_base_model, input_model_path, model_bucketfs_location)
            except Exception as e:
                print("Retry didn't work",e)
                raise e

            if not isinstance(iterator, RegressorPartialFitIterator):
                raise Exception(
                    f"Model from {input_model_path} is not a RegressorPartialFitIterator, instead we got {iterator}")
            if len(models) == 0:
                table_preprocessor = iterator.table_preprocessor
            models.append(iterator.model)
            if not ctx.next():
                break
        combined_model = combine_to_voting_regressor(models)
        combined_model_iterator = ScoreIterator(model=combined_model,
                                                table_preprocessor=table_preprocessor)
        combined_model_file_name = f"{str(self.exa.meta.session_id)}_{str(self.exa.meta.statement_id)}_{str(self.exa.meta.node_id)}_{str(self.exa.meta.vm_id)}_{str(self.counter)}.pkl"
        combined_model_path = PurePosixPath(self.COMBINED_MODEL_DIRECTORY, combined_model_file_name)
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(combined_model_iterator,
                                                                     combined_model_path)
        ctx.emit(model_connection_name, path_under_model_connection, str(combined_model_path))
        self.counter += 1

    def load_base_model(self, input_model_path: str, model_bucketfs_location: AbstractBucketFSLocation):
        iterator = model_bucketfs_location.download_object_from_bucketfs_via_joblib(input_model_path)
        return iterator
