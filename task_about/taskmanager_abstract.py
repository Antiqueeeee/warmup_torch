import os, sys
from abc import ABC, abstractmethod
from utils import recorder

current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)
class abstract_task_manager(ABC):
    def __init__(self, task_name) -> None:
        self.task_name = task_name
        self.recorder = recorder(task_name = task_name)
        self._check_neccery_prop()
        
        self.model_base = os.path.join(
            project_path
            , "model_about"
            , "models"
            , task_name
        )
        
        self.datasets_base = (
            project_path
            , "model_about"
            , "datasets"
            , task_name
        )
        
        self.results_base = (
            project_path
            , "model_about"
            , "datasets"
            , task_name
        )
        
        
    @abstractmethod
    def run_command(self, instruction):
        pass
    
    def _check_neccery_prop(self):
        """检查子类是否定义了 supported_instruction 属性"""
        if not hasattr(self, 'supported_instruction'):
            raise NotImplementedError("子类必须在super()之前定义 supported_instruction 属性")
        
        if not hasattr(self, 'supported_model'):
            raise NotImplementedError("子类必须在super()之前定义 supported_instruction 属性")
        
    def check_instruction_valid(self, instruction):
        if instruction in self.supported_instruction:
            return True
        else:
            return False