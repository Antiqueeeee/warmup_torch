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

    # def simple_feature_mapping(self, column):
    #     origin_data = pd.read_excel(
    #         os.path.join(
    #             self.path_mananager.get_path("datasets")
    #             , "2024.9.12 300人的43SNP基因型疗效肝毒性.xlsx"
    #         )
    #     )
    #     mapping = dict(origin_data[column].value_counts())
    #     mapping = {"".join(k.split()): v for k, v in mapping.items()}
    #     mapping.pop("00", None)
    #     keys_with_set_size_2 = [k for k in mapping if len(set(k)) == 2]
    #     filtered_keys = {k: v for k, v in mapping.items() if k not in keys_with_set_size_2}
    #     max_value_key = max(filtered_keys, key=filtered_keys.get)
    #     min_value_key = min(filtered_keys, key=filtered_keys.get)
    #     # Update mapping values
    #     for key in keys_with_set_size_2:
    #         mapping[key] = 1
    #     mapping[max_value_key] = 0
    #     mapping[min_value_key] = 2
    #     return mapping

    def simple_feature_mapping(self, column):
        origin_data = pd.read_excel(
            os.path.join(
                self.path_mananager.get_path("datasets")
                , "2024.9.12 300人的43SNP基因型疗效肝毒性.xlsx"
            )
        )
        mapping = dict(origin_data[column].value_counts())
        mapping = {"".join(k.split()): v for k, v in mapping.items()}
        mapping.pop("00", None)
        
        # 找出所有set大小为2的键
        keys_with_set_size_2 = [k for k in mapping if len(set(k)) == 2]
        
        # 过滤掉set大小为2的键
        filtered_keys = {k: v for k, v in mapping.items() if k not in keys_with_set_size_2}
        
        # 获取最大值和最小值对应的键
        max_value_key = max(filtered_keys, key=filtered_keys.get)
        min_value_key = min(filtered_keys, key=filtered_keys.get)
        
        # 先处理set大小为2的键
        for key in keys_with_set_size_2:
            mapping[key] = 1
            
        # 处理最大值和最小值键
        if max_value_key == min_value_key:
            # 如果最大值键和最小值键相同，将其设为2
            mapping[max_value_key] = 2
            # 将剩余的键（除了set大小为2的键和max_value_key）设为0
            for key in filtered_keys:
                if key != max_value_key:
                    mapping[key] = 0
        else:
            # 原来的逻辑
            mapping[max_value_key] = 0
            mapping[min_value_key] = 2
            
        return mapping


    # Apply new mapping to results data
    def map_value(self, x, mapping):
        x_key = "".join(x.split())
        for k, v in mapping.items():
            if set(x_key) == set(k):
                return v
        return 1

class LogisiticRegression(abstract_model_factory):
    def __init__(self, task_name, data_processor) -> None:
        super().__init__(task_name, data_processor)
        self.model = None
        self.required_features = ['rs8008009', 'rs2842512', 'rs4961080', 'rs243195', 'rs3794338', 'rs1963088', 'rs2515284', 'rs6486979', 'rs19026', 'rs7013741', 'rs12146924', 'rs10845199', 'rs12513652', 'rs4956949', 'rs2761239']
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
        self.model_load("药物短期肝毒性_逻辑回归_20241126_1.pkl")
        test_file_name, test_file_fix = os.path.splitext(test_file)
        _path = os.path.join(self.path_mananager.get_path("datasets"), test_file)
        self.recorder.info(f"{self.task_name} - 读取数据: {_path}")
        results_data = pd.read_excel(_path)
        results_data = results_data.astype(str)
        results_data = results_data[self.basic_info + self.required_features]

        # 映射
        for column in results_data.columns:
            if column in self.required_features:
                if "rs243195" in column:
                    print()
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
            f'短期肝毒性类别': predictions,
            f'短期肝毒性类别1概率': probabilities
        })
        results_with_info = pd.merge(results_data, prediction_results, on='样品编号')
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
    task_name = "药物短期肝毒性"
    data_prcessor = LogisiticRegressionDataProcessor(task_name=task_name)
    test_file = "1730796867-20241129.xlsx"
    lr = LogisiticRegression(task_name=task_name,data_processor=data_prcessor)
    lr.model_inference(test_file=test_file)