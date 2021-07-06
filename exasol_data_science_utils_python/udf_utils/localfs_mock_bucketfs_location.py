from pathlib import PurePosixPath
from typing import Any

import joblib

from exasol_data_science_utils_python.udf_utils.abstract_bucketfs_location import AbstractBucketFSLocation


class LocalFSMockBucketFSLocation(AbstractBucketFSLocation):

    def __init__(self, base_path: PurePosixPath):
        self.base_path = base_path

    def get_complete_file_path_in_bucket(self, bucket_file_path) -> str:
        return str(PurePosixPath(self.base_path, bucket_file_path))

    def download_from_bucketfs_to_string(self, bucket_file_path: str) -> str:
        with open(self.get_complete_file_path_in_bucket(bucket_file_path), "rt") as f:
            result = f.read()
            return result

    def download_object_from_bucketfs_via_joblib(self, bucket_file_path: str) -> Any:
        result = joblib.load(self.get_complete_file_path_in_bucket(bucket_file_path))
        return result

    def upload_string_to_bucketfs(self, bucket_file_path: str, string: str):
        with open(self.get_complete_file_path_in_bucket(bucket_file_path), "wt") as f:
            f.write(string)

    def upload_object_to_bucketfs_via_joblib(self, object: Any,
                                             bucket_file_path: str,
                                             **kwargs):
        joblib.dump(object, self.get_complete_file_path_in_bucket(bucket_file_path), **kwargs)
