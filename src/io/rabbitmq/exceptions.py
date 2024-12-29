from src.exceptions import AppError


class RabbitMQClientError(AppError):
    ...


class RabbitMQConnectionError(RabbitMQClientError):
    ...
