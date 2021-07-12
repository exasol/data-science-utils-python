import textwrap

from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


def create_combine_to_voting_regressor_udf(sql_executor: SQLExecutor, target_schema: Schema):
    udf = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3 SET SCRIPT {target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
    model_connection varchar(10000),
    input_model_path varchar(10000)
    ) 
    EMITS (output_model_path varchar(10000)) AS
    from exasol_data_science_utils_python.model_utils.udfs.combine_to_voting_regressor_udf import \
        CombineToVotingRegressorUDF

    udf = CombineToVotingRegressorUDF(exa)

    def run(ctx):
        udf.run(ctx)
    """)
    sql_executor.execute(udf)


def create_partial_fit_regressor_udf(sql_executor: SQLExecutor, target_schema: Schema):
    udf = textwrap.dedent(f"""
    CREATE OR REPLACE PYTHON3 SET SCRIPT {target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(...) 
    EMITS (output_model_path varchar(10000), SCORE_SUM DOUBLE, SCORE_COUNT INTEGER) AS
    from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF

    udf = PartialFitRegressorUDF(exa)

    def run(ctx):
        udf.run(ctx)
    """)
    sql_executor.execute(udf)
