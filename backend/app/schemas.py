from pydantic import BaseModel

class AskRequest(BaseModel):
    text: str
