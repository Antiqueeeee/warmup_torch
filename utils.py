import os, sys
current_path = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)

import logging
from datetime import datetime


class recorder:
    def __init__(self, task_name) -> None:
        self.task_name = task_name
        self.log_file = os.path.join(project_path, f"{self.task_name}.log")
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
    

