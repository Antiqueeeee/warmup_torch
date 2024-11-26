from fastapi import FastAPI
import os
import uvicorn


from config import *
from server_about.interfaceParams import *
from task_about import taskmanager_drug_tox

app = FastAPI()

managers = {
    "药物毒性预测" : taskmanager_drug_tox
}

class Server(FastAPI):
    def __init__(self):   
        super().__init__()
        self.post("/runCommand")(self.run_command)
    
    def run_command(self, parameters : paramsRunCommand):
        task, instruction = parameters.task, parameters.instruction
        if task not in managers.keys():
            return {"data" : "不支持当前任务"}
        taskmanager = managers[task](task_name = task)
        result = taskmanager.run_command(instruction)
        return result
        
    def run(self):
        uvicorn.run(self, host=SERVER_ADDRESS, port=SERVER_PORT)
        


if __name__ == "__main__":
    server = Server()
    server.run()