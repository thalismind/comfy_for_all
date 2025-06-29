from pydantic import BaseModel

class ImageJob(BaseModel):
    id: int
    requested_at: str
    started_at: str | None = None
    request_type: str
    requester: str
    requested_prompt: str
    negative_prompt: str | None = None
    model: str
    steps: int
    channel: str
    image_link: str | None = None
    resolution: str
    batch_size: int
    config_scale: int