from fastapi import FastAPI
import os
import uvicorn


from config import *
from server_about.interfaceParams import *
from tasks import taskmanager_drug_efficacy

app = FastAPI()

managers = {
    "药物有效性" : taskmanager_drug_efficacy
}

class Server(FastAPI):
    def __init__(self):   
        super().__init__()
        self.post("/runCommand")(self.run_command)
    
    def run_command(self, parameters : paramsRunCommand):
        task = parameters.task
        if task not in managers.keys():
            return {"data" : "不支持当前任务"}
        taskmanager = managers[task](task_name = task)
        result = taskmanager.run_command(**parameters.dict())
        return result
        
    def run(self):
        uvicorn.run(self, host=SERVER_ADDRESS, port=SERVER_PORT)
        


if __name__ == "__main__":
    server = Server()
    server.run()