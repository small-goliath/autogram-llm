from pydantic import BaseModel

class AskRequest(BaseModel):
    text: str
    amount: int = 0
