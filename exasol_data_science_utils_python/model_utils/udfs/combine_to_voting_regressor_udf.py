from pathlib import PurePosixPath

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
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
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=path_under_model_connection
            )

        regressor_partial_fit_iterators = []
        while True:
            input_model_path = ctx.input_model_path
            model = model_bucketfs_location.download_object_from_bucketfs_via_joblib(input_model_path)
            regressor_partial_fit_iterators.append(model)
            if not isinstance(model, RegressorPartialFitIterator):
                raise Exception(
                    f"Model from {input_model_path} is not a RegressorPartialFitIterator, instead we got {model}")
            if not ctx.next():
                break
        combined_model = RegressorPartialFitIterator.combine_to_voting_regressor(regressor_partial_fit_iterators)
        combined_model_file_name = f"{str(self.exa.meta.session_id)}_{str(self.exa.meta.statement_id)}_{str(self.exa.meta.node_id)}_{str(self.exa.meta.vm_id)}_{str(self.counter)}.pkl"
        combined_model_path = PurePosixPath(self.COMBINED_MODEL_DIRECTORY, combined_model_file_name)
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(combined_model,
                                                                     combined_model_path)
        ctx.emit(model_connection_name, path_under_model_connection, str(combined_model_path))
        self.counter += 1
