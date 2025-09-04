from pydantic import BaseModel


class Config(BaseModel):
    api_key: str
    api_url: str
    timeout: int
