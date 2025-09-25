class REGISTER:
    def __init__(self, name):
        self.name = name
        self.dict = dict()

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        return key in self.dict.keys()

    def register_module(self, fn, name=None):
        name = name if name is not None else fn.__name__
        self.dict[name] = fn

    def get_module(self, name):
        return self.dict[name] if self.__contains__(name) else None
