import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_path)


from task_about.taskmanager_abstract import abstract_task_manager

class taskmanager_drug_efficacy(abstract_task_manager):
    def __init__(self, task_name) -> None:
        self.supported_instruction = ["模型推理"]
        self.supported_model = ["逻辑回归"]
        super().__init__(task_name)
        
    def run_command(self, instruction, **kwargs):
        if self.check_instruction_valid(instruction) is False:
            return {"data" : f'当前任务中支持:{",".join(self.supported_instruction)},{instruction}不在其中。'}
        
        if instruction == "模型推理":
            kwargs = {
                "selected_model" : "逻辑回归"
                ,"inference_data" : ""
            }
            
            return {"data" : f"已开始{self.task_name}-{instruction}"}


if __name__ == "__main__":
    task_name = "药物毒性预测"
    tm = taskmanager_drug_efficacy(task_name)
