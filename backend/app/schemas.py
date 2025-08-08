from typing import List, Optional
from pydantic import BaseModel

class AskRequest(BaseModel):
    text: str
    pre_comments: Optional[List[str]] = None
