import os, sys
current_path = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)

import logging
from datetime import datetime


class BasicPathCreator(object):
    def __init__(self, task_name) -> None:
        self.model_base = os.path.join(
            project_path
            , 'tasks'
            , task_name
            , "models"
        )
        
        self.datasets_base = os.path.join(
            project_path
            , "tasks"
            , task_name
            , "datasets"
        )
        
        self.results_base = os.path.join(
            project_path
            , "tasks"
            , task_name
            , "results"
        )
        
        self.checkpoints_base = os.path.join(
            project_path
            , "tasks"
            , task_name
            , "checkpoints_trained"
        )
        
        self._paths = [self.model_base, self.datasets_base, self.results_base ,self.checkpoints_base]
        for _path in self._paths:
            os.makedirs(_path, exist_ok=True)
    
    def get_path(self, target):
        path_recorder = {
            "model" : self.model_base,
            "datasets" : self.datasets_base,
            "results" : self.results_base,
            "checkpoints" : self.checkpoints_base,
        }
        return path_recorder[target]
    
class recorder:
    def __init__(self, task_name) -> None:
        self.task_name = task_name
        self.path_mananager = BasicPathCreator(task_name=task_name)
        self.log_file = os.path.join(self.path_mananager.get_path("results"), f"{self.task_name}.log")
        self.level = logging.INFO
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(self.level)
        
        # Prevent adding multiple handlers to the logger
        if not self.logger.handlers:
            # Create file handler with append mode
            file_handler = logging.FileHandler(self.log_file, mode='a')
            file_handler.setLevel(self.level)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            
            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
    def info(self, message):
        self.logger.info(message)
        
    def error(self, message):
        self.logger.error(message)
        
    def read_logs(self):
        if os.path.exists(self.logger.handlers[0].baseFilename):
            with open(self.logger.handlers[0].baseFilename, 'r') as log_file:
                return log_file.read()
        return None
    
    def time_now(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    

