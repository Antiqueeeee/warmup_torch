from fastapi import FastAPI, BackgroundTasks
import os
import uuid
import uvicorn
import requests

from config import *
import schedule
from server_about.interfaceParams import *
from tasks import taskmanager_drug_efficacy
from tasks import task_manager_drug_long_tox
from tasks import task_manager_drug_shot_tox

from threading import Thread
from tools_about.EmailParser import Process
import time

app = FastAPI()

managers = {
    "药物有效性" : taskmanager_drug_efficacy,
    "药物短期肝毒性" : task_manager_drug_shot_tox,
    "药物长期肝毒性" : task_manager_drug_long_tox
}

# 创建相关路径
for key,values in managers.items():
    print(key)
    managers[key](task_name=key)

class Server(FastAPI):
    def __init__(self):   
        super().__init__()
        self.processing_task = dict()
        self.email_processor = Process(EMAIL_PROCESSOR, EMAIL_PROCESSOR_PWD)
        self.email_processed_record = list()
        self.post("/runCommand")(self.run_command)
        self.get("/taskStatus/{task_id}")(self.get_task_status)
        self.start_scheduled_tasks()
        
    def run_command(self, parameters : paramsRunCommand):
        task = parameters.task
        if task not in managers.keys():
            return {"data" : "不支持当前任务"}
        task_id = str(uuid.uuid4())
        self.processing_task[task_id] = {"task_id": task_id, "status": "pending"}
        Thread(target=self.execute_task, args=(task_id, parameters)).start()
        return {"task_id": task_id, "status": "pending"}
        
    def execute_task(self, task_id, parameters):
        try:
            task = parameters.task
            taskmanager = managers[task](task_name = task)
            result = taskmanager.run_command(**parameters.dict())
            self.processing_task[task_id] = result
            # 推理结束后手动执行一次process_emails_task，尽量及时更新email处理记录
            self.update_emails_record()
        except Exception as e:
            self.processing_task[task_id] = {"task_id": task_id, "status": "Failed" + f" {str(e)}"}

    def get_task_status(self, task_id: str):
        return self.processing_task.get(
            task_id
            , {"task_id":task_id,"status" : "Not Found"}
        )

    def update_emails_record(self):
        to_remove = list()
        for record in self.email_processed_record:
            _url = f"http://127.0.0.1:{SERVER_PORT}/taskStatus/{record}"
            _response = requests.get(_url).json()
            status = _response.get("data", "Not Found")
            if "任务已完成" in status:
                self.email_processor.processed_record.append(_response["UID"])
                self.email_processor.update_record()
                to_remove.append(record)
                
        for record in to_remove:
            self.email_processed_record.remove(record)
            
    def process_emails_task(self):
        # to_remove = list()
        # for record in self.email_processed_record:
        #     _url = f"http://127.0.0.1:{SERVER_PORT}/taskStatus/{record}"
        #     _response = requests.get(_url).json()
        #     status = _response.get("data", "Not Found")
        #     if "任务已完成" in status:
        #         self.email_processor.processed_record.append(_response["UID"])
        #         self.email_processor.update_record()
        #         to_remove.append(record)
                
        # for record in to_remove:
        #     self.email_processed_record.remove(record)
        self.update_emails_record()
        
        to_be_processed = self.email_processor.process_emails(managers.keys())
        for _process in to_be_processed:
            _tasks_id = requests.post(f"http://127.0.0.1:{SERVER_PORT}/runCommand", json=_process).json().get("task_id", str())
            self.email_processed_record.append(_tasks_id)
    
    def start_scheduled_tasks(self):
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        # 设置定时任务，每小时执行一次
        # schedule.every(1).hours.do(self.process_emails_task)
        schedule.every(15).seconds.do(self.process_emails_task)
        
        # 在后台线程中运行调度器
        Thread(target=run_schedule, daemon=True).start()
        

    def run(self):
        uvicorn.run(self, host=SERVER_ADDRESS, port=SERVER_PORT)

if __name__ == "__main__":
    server = Server()
    server.run()
