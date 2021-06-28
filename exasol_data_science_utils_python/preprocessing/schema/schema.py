class Schema:
    def __init__(self, schema_name: str):
        self.name = schema_name

    def identifier(self) -> str:
        return f'"{self.name}"'
