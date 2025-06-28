from pydantic import BaseModel

class ImageJob(BaseModel):
    id: int
    created_at: str
    updated_at: str
    job_type: str
    user: int
    prompt: str
    negative_prompt: str
    model: str
    steps: int
    seed: int
    size: str
    batch_size: int
    cfg: float
