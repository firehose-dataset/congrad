import abc

class Algorithm(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, online_batch, stats, skip_optim=False):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return "{}()".format(self.__class__.__name__)
