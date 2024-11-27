import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_path)

import pandas as pd
from tasks.taskmanager_abstract import abstract_task_manager
from tasks.药物有效性.models.model_logistic_regression import LogisiticRegression, LogisiticRegressionDataProcessor
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
            return {"data" : f"已开始{self.task_name}-{instruction}"}

    def data_upload(self, result_path):
        frame = pd.read_excel(result_path)
        print(f"我拿到了数据:{frame.shape}")

if __name__ == "__main__":
    task_name = "药物有效性"
    tm = taskmanager_drug_efficacy(task_name)
    kwargs = {
        "task_name" : task_name
        ,"instruction" : "模型推理"
        ,"selected_model" : "药物有效性_逻辑回归_20241126_1.pkl"
        ,"inference_data" : "20241126_检测结果.xlsx"
    }
    tm.run_command(**kwargs)