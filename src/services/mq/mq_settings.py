from pydantic import BaseModel


class MQSettings(BaseModel):
    request_queue_name: str | None = "recognition.request"
    response_routing_base: str | None = "recognition.response"
