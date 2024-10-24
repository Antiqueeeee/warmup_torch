import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)

from abc import ABC
from util_base import recorder
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd


class abstract_model_factory(ABC):
    def __init__(self, task_name, data_processor) -> None:
        self.task_name = task_name
        self.data_processor = data_processor
        self.recorder = recorder(task_name)
        self.data_dir = os.path.join(project_path, "model_about", "datasets")
    
    def make_dataloader(self):
        pass
    
    def model_load(self, model_path):
        model = "torch.load"
        return model
    
    def model_save(self, model) -> bool:
        save_path = "save_path"
        model_save = "save_model"
        flag = True if model_save == "save_model" else False
        return flag
    
    def inference(self, model):
        pass
    
    def training(self, model):
        pass
    
    def feature_select(self):
        pass
    

class abstract_data_processor(ABC):
    def __init__(self, task_name) -> None:
        self.recorder = recorder(task_name)
        self.data_dir = os.path.join(project_path, "model_about", "datasets")
        
    def pre_processing(self, data_path, data_mode="single"):
        pass
    
    def normalize_data_byall(self, dataframe, columns_to_normalize):
        scaler = MinMaxScaler()
        scaler.fit(dataframe.loc[ : , columns_to_normalize])
        dataframe.loc[:, columns_to_normalize] = scaler.transform(dataframe.loc[:, columns_to_normalize])
        return dataframe
    
    def split_dataframe_by_column(self, dataframe, column_name, train_ratio=0.67):
        """
        按照指定列的值分割数据集为训练集和测试集。
        
        :param data: DataFrame类型，需要分割的数据集。
        :param column_name: str类型，用作分割依据的列名。
        :param train_ratio: float类型，训练集所占比例，默认值为0.67。
        :return: 分割后的训练集和测试集。
        """
        train = pd.DataFrame()
        valid = pd.DataFrame()

        for value in dataframe[column_name].unique():
            subset = dataframe[dataframe[column_name] == value]
            train_size = int(len(subset) * train_ratio)
            train = pd.concat([train, subset[:train_size]])
            valid = pd.concat([valid, subset[train_size:]])
        #由于我们是按照指定列将train和valid分开
        #因此现在train和valid的索引被切断了
        #需要为train、valid恢复索引
        
        train.index = range(train.shape[0])
        valid.index = range(valid.shape[0])
        
        return train, valid
    


