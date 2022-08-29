import textwrap

from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


def create_combine_to_voting_regressor_udf(sql_executor: SQLExecutor, target_schema: SchemaName):
    udf = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3_DSUP SET SCRIPT {target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
    model_connection VARCHAR(2000000),
    path_under_model_connection VARCHAR(2000000),
    input_model_path VARCHAR(2000000),
    download_retry_seconds INTEGER
    ) 
    EMITS (
        model_connection_name VARCHAR(2000000),
        path_under_model_connection VARCHAR(2000000),
        combined_model_path VARCHAR(2000000)
        ) AS
    from exasol_data_science_utils_python.model_utils.udfs.combine_to_voting_regressor_udf import \
        CombineToVotingRegressorUDF

    udf = CombineToVotingRegressorUDF(exa)

    def run(ctx):
        udf.run(ctx)
    """)
    sql_executor.execute(udf)


def create_partial_fit_regressor_udf(sql_executor: SQLExecutor, target_schema: SchemaName):
    udf = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3_DSUP SET SCRIPT {target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(...) 
    EMITS (
        model_connection_name VARCHAR(2000000),
        path_under_model_connection VARCHAR(2000000),
        output_model_path VARCHAR(2000000), 
        training_score_sum DOUBLE, 
        training_score_count INTEGER) AS
    from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF

    udf = PartialFitRegressorUDF(exa)

    def run(ctx):
        udf.run(ctx)
    """)
    sql_executor.execute(udf)
