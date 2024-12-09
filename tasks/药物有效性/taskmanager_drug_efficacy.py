import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_path)

import pandas as pd
import json
import time
import requests
from tasks.taskmanager_abstract import abstract_task_manager
from tasks.药物有效性.models.model_logistic_regression import LogisiticRegression, LogisiticRegressionDataProcessor
from config import *

class taskmanager_drug_efficacy(abstract_task_manager):
    def __init__(self, task_name) -> None:
        self.supported_instruction = ["模型推理"]
        self.supported_model = [
            "药物有效性_逻辑回归_20241126_1.pkl"
        ]
        super().__init__(task_name)
        self.model_mapping = {
            "药物有效性_逻辑回归_20241126_1.pkl"  : (LogisiticRegression, LogisiticRegressionDataProcessor)
        }
        
    def run_command(self, **kwargs):
        instruction = kwargs["instruction"]
        if self.check_instruction_valid(instruction) is False:
            return {"data" : f'当前任务中支持:{",".join(self.supported_instruction)},{instruction}不在其中。'}
        
        if instruction == "模型推理":
            selected_model, inference_data = kwargs["selected_model"], kwargs["inference_data"]
            if selected_model not in self.supported_model:
                raise
            (manager, precessor) = self.model_mapping[selected_model]
            data_processor = precessor(task_name=self.task_name)
            model_manager = manager(task_name=self.task_name, data_processor=data_processor)
            model_manager.model_load(selected_model)
            results = model_manager.model_inference(inference_data)
            self.recorder.info("模型推理完成")
            response = self.data_upload(results)
            if response.get("msg", "操作失败") == "操作成功" :
                self.recorder.info(f"数据同步完成：{self.task_name}-{instruction}-{results}")
                return {
                    "data" : f"任务已完成：{self.task_name}-{instruction}-{results}"
                    ,"UID" : os.path.splitext(kwargs["inference_data"])[0]
                }
            else:
                return {
                    "data" : f"任务异常：{self.task_name}-{instruction}-{response.get('msg', '服务器未返回Msg')}"
                    ,"UID" : os.path.splitext(kwargs["inference_data"])[0]
                }
                
    def data_upload(self, result_path):
        frame = pd.read_excel(result_path)
        params = list()
        for param in frame.to_dict("records"):
            params.append({
                "checkTime": param["日期"],
                "sampleSerialNumber": param["样品编号"],
                "methotrexateEffectivePercentage": float(param["有效性类别1概率"]),
                "shortTermHepatotoxicityProbability": None,
                "longTermHepatotoxicityProbability": None,
                "timestamp": str(int(time.time()))
            })
        secret = INTERFACE_SECRET.format(
            json_data = json.dumps(params, ensure_ascii=False)
            ,timestamp = str(int(time.time()))
            ,nonce = 9527
            ,key = SECRETKEY
        )
        request_body = {
            "data" : json.dumps(params, ensure_ascii=False),
            "nonce" : 9527,
            "timestamp" : str(int(time.time())),
            "signStr" : self.md5_encoding(secret)
        }
        response = requests.post(INTERFACE_GENE_DATA_UPLOAD, json=request_body).json()
        return response
        
if __name__ == "__main__":
    # task_name = "药物有效性"
    # tm = taskmanager_drug_efficacy(task_name)
    # kwargs = {
    #     "task_name" : task_name
    #     ,"instruction" : "模型推理"
    #     ,"selected_model" : "药物有效性_逻辑回归_20241126_1.pkl"
    #     ,"inference_data" : "20241126_检测结果.xlsx"
    # }
    # tm.run_command(**kwargs)
    load_dotenv()
    task_name = "药物有效性"
    tm = taskmanager_drug_efficacy(task_name)
    tm.data_upload(r"E:\feynmindPyhton\warmup_torch\tasks\药物有效性\results\测试上传.xlsx")