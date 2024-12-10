import os,sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../../"))
sys.path.append(project_path)
import pickle
import pandas as pd

from tasks.model_abstract import abstract_model_factory, abstract_data_processor
from sklearn.impute import SimpleImputer


class LogisiticRegressionDataProcessor(abstract_data_processor):
    def __init__(self, task_name) -> None:
        super().__init__(task_name)

class LogisiticRegression(abstract_model_factory):
    def __init__(self, task_name, data_processor) -> None:
        super().__init__(task_name, data_processor)
        self.model = None
        self.required_features = ['rs2298294', 'rs10937405', 'rs1247456', 'rs11785942', 'rs13051672', 'rs4965177', 'rs4867081', 'rs788970', 'rs9324014', 'rs12513652', 'rs1962292', 'rs2761239', 'rs3888613', 'rs1359168']
        self.basic_info = [
            "样品/SNP", "样品编号", "日期"
        ]
        
    def model_load(self, saved_checkpoint):
        saved_checkpoint_path = os.path.join(
            self.path_mananager.get_path("checkpoints")
            , saved_checkpoint
        )
        with open(saved_checkpoint_path, "rb") as f:
            self.model = pickle.load(f)
        return self.model
    
    def model_save(self, model):
        return super().model_save(model)
    
    def model_inference(self, test_file):
        self.model_load('药物长期肝毒性_逻辑回归_20241126_1.pkl')
        test_file_name, test_file_fix = os.path.splitext(test_file)
        _path = os.path.join(self.path_mananager.get_path("datasets"), test_file)
        self.recorder.info(f"{self.task_name} - 读取数据: {_path}")
        results_data = pd.read_excel(_path)
        results_data = results_data.astype(str)
        results_data = results_data[self.basic_info + self.required_features]

        # 映射
        for column in results_data.columns:
            if column in self.required_features:
                mapping = self.data_processor.simple_feature_mapping(column)
                results_data[column] = results_data[column].apply(lambda x: self.data_processor.map_value(x, mapping))
        _path = os.path.join(
                self.path_mananager.get_path("results")
                , f"_{test_file_name}.映射结果{test_file_fix}"
            )
        self.recorder.info(f"{self.task_name} - 映射结果存储位置为: {_path}")
        results_data.to_excel(_path, index=False)
        
        # 开始推理
        _data = results_data[self.required_features]
        imputer = SimpleImputer(strategy='most_frequent')
        _data = pd.DataFrame(imputer.fit_transform(_data), columns=_data.columns)
        _data = _data.astype(float)
        predictions = self.model.predict(_data)
        probabilities = self.model.predict_proba(_data)[:, 1]
        prediction_results = pd.DataFrame({
            '样品编号': results_data['样品编号'],  # 使用共同的键
            f'长期肝毒性类别': predictions,
            f'长期肝毒性类别1概率': probabilities
        })
        results_with_info = pd.merge(prediction_results, results_data, on='样品编号')
        _path = os.path.join(
                self.path_mananager.get_path("results")
                , f"_{test_file_name}.预测结果{test_file_fix}"
            )
        results_with_info.to_excel(_path, index=False)
        self.recorder.info(f"{self.task_name} - 预测结果存储位置为: {_path}")
        return _path
        
    def model_training(self, model):
        return super().model_training(model)
    
if __name__ == "__main__":
    task_name = "药物长期肝毒性"
    data_prcessor = LogisiticRegressionDataProcessor(task_name=task_name)
    test_file = "1730796867-20241129.xlsx"
    lr = LogisiticRegression(task_name=task_name,data_processor=data_prcessor)
    lr.model_inference(test_file=test_file)