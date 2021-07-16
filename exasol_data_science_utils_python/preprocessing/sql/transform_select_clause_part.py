from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class TransformSelectClausePart:
    @typechecked
    def __init__(self,
                 tranformation_column: TransformationColumn,
                 select_clause_part_expression: str):
        self.tranformation_column = tranformation_column
        self.select_clause_part_expression = select_clause_part_expression

    def __repr__(self):
        return generate_repr_for_object(self)

    def __eq__(self, other):
        return isinstance(other, TransformSelectClausePart) and \
               self.tranformation_column == other.tranformation_column and \
               self.select_clause_part_expression == other.select_clause_part_expression
