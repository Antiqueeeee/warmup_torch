from pydantic import BaseModel
from typing import List, Optional


class paramsRunCommand(BaseModel):
    task : str = None
    instruction : str = None