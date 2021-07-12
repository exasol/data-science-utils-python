
class ConnectionObject:
    def __init__(self, name:str, address:str, user:str, password:str):
        self.password = password
        self.user = user
        self.address = address
        self.name = name
