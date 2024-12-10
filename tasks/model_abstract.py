import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)

from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from utils import recorder, BasicPathCreator
import json

class abstract_model_factory(torch.nn.Module):
    def __init__(self, task_name, data_processor) -> None:
        super(abstract_model_factory, self).__init__()
        self.task_name = task_name
        self.data_processor = data_processor
        self.recorder = recorder(task_name = task_name)
        self.path_mananager = BasicPathCreator(task_name=task_name)
        self._check_necessary_property()
        
    @abstractmethod
    def model_load(self, saved_checkpoint_path):
        pass
    
    @abstractmethod
    def model_save(self, model):
        pass

    @abstractmethod
    def model_inference(self, test_file):
        pass

    @abstractmethod
    def model_training(self, model):
        pass
    
    def _check_necessary_property(self):
        # """检查子类是否定义了 supported_instruction 属性"""
        # if not hasattr(self, 'supported_instruction'):
        #     raise NotImplementedError("子类必须在super()之前定义 supported_instruction 属性")
        
        # if not hasattr(self, 'supported_model'):
        #     raise NotImplementedError("子类必须在super()之前定义 supported_instruction 属性")
        pass

class abstract_data_processor(ABC):
    def __init__(self, task_name) -> None:
        self.task_name = task_name
        self.recorder = recorder(task_name)
        self.path_mananager = BasicPathCreator(task_name=task_name)
    
    def make_dataloader(self, dataframe, window_size, pred_len, shuffle_flag, batch_size, drop_last_flag=True):
        X, y, y_indices = [], [], []
        for i in range(len(dataframe) - window_size - pred_len + 1):
            # 检查窗口内的Securities Code是否唯一
            # 选取从第4列到最后一列的特征和标签
            feature_and_label = dataframe.iloc[i:i + window_size, 3:].copy().values
            # 长度pred_len的标签作为目标
            target = dataframe.iloc[(i + window_size):(i + window_size + pred_len), -1]
            
            # 记录本窗口中要预测的标签的时间点
            target_indices = list(range(i + window_size, i + window_size + pred_len))

            X.append(feature_and_label)
            y.append(target)
            #将每个标签的索引添加到y_indices列表中
            y_indices.extend(target_indices)
                
        X = torch.FloatTensor(np.array(X, dtype=np.float32))
        y = torch.FloatTensor(np.array(y, dtype=np.float32))            
        dataloader = data.DataLoader(data.TensorDataset(X, y)
                         #每个表单内部是保持时间顺序的即可，表单与表单之间可以shuffle
                         , shuffle=shuffle_flag
                         , batch_size = batch_size
                         , drop_last = drop_last_flag) 
        return dataloader
    
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
    
    def split_dataframe_randomly(self, dataframe, train_ratio=0.67, random_state=None):
        """
        随机分割数据集为训练集和测试集。
        
        :param dataframe: DataFrame类型，需要分割的数据集。
        :param train_ratio: float类型，训练集所占比例，默认值为0.67。
        :param random_state: int类型，随机种子，默认为None。
        :return: 分割后的训练集和测试集。
        """
        train, valid = train_test_split(dataframe, train_size=train_ratio, random_state=random_state)
        return train, valid

    def simple_feature_mapping(self, column):
        with open(os.path.join(project_path, "mapping.json"),encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping[column]

    # Apply new mapping to results data
    def map_value(self, x, mapping):
        x_key = "".join(x.split())
        for k, v in mapping.items():
            if set(x_key) == set(k):
                return v
        return 1

