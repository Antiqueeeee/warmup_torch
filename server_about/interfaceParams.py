from pydantic import BaseModel
from typing import List, Optional


class paramsRunCommand(BaseModel):
    task : str 
    instruction : str
    selected_model: Optional[str] = None
    inference_data: Optional[str] = None
    
    
    def dict(self, *args, **kwargs):
        original_dict = super().dict(*args, **kwargs)
        # Merge extra_params into the main dictionary
        return {**original_dict}