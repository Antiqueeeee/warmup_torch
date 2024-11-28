from fastapi import FastAPI, BackgroundTasks
import os
import uuid
import uvicorn


from config import *
from server_about.interfaceParams import *
from tasks import taskmanager_drug_efficacy
from threading import Thread

app = FastAPI()

managers = {
    "药物有效性" : taskmanager_drug_efficacy
}


class Server(FastAPI):
    def __init__(self):   
        super().__init__()
        self.processing_task = dict()
        self.post("/runCommand")(self.run_command)
        self.get("/taskStatus/{task_id}")(self.get_task_status)
        
    def run_command(self, parameters : paramsRunCommand):
        task = parameters.task
        if task not in managers.keys():
            return {"data" : "不支持当前任务"}
        task_id = str(uuid.uuid4())
        self.processing_task[task_id] = "pending"
        Thread(target=self.execute_task, args=(task_id, parameters)).start()
        return {"task_id": task_id, "status": "pending"}
        
    def execute_task(self, task_id, parameters):
        try:
            task = parameters.task
            taskmanager = managers[task](task_name = task)
            result = taskmanager.run_command(**parameters.dict())
            self.processing_task[task_id] = result["data"]
        except Exception as e:
            self.processing_task[task_id] = "Failed" + f" {str(e)}"

    def get_task_status(self, task_id: str):
        task_status = self.processing_task.get(task_id, "Task not found")
        return {
            "task_id": task_id,
            "status": task_status,
        }

    def run(self):
        uvicorn.run(self, host=SERVER_ADDRESS, port=SERVER_PORT)
        


if __name__ == "__main__":
    server = Server()
    server.run()