import punq


class ResultQueue:
    def __init__(self, deps: punq.Container):

        self.storage = deps.resolve(instance="storage")
        self.rabbitmq = rabbitmq
