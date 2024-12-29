from pydantic import BaseModel


class DispatcherSettings(BaseModel):
    request_queue_name: str | None = "recognition.requests"
    result_queue_base: str | None = "recognition.results"
