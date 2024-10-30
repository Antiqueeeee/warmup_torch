import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_path)

import pandas as pd
import numpy as np

from decimal import ROUND_HALF_UP, Decimal
from model_about.model_abstract import abstract_model_factory, abstract_data_processor
from config import *
class model_price_data_processor(abstract_data_processor):
    def __init__(self, task_name) -> None:
        super().__init__(task_name)
    
    def pre_processing(self, data_path, data_mode="single"):
        if data_mode == "single":
            stock_data = pd.read_csv(os.path.join(self.data_dir, data_path))
            processed_data = stock_data.copy()
            processed_data.rename(columns={'Target': 'Sharpe Ratio'}, inplace=True)
            processed_data["ExpectedDividend"] = processed_data["ExpectedDividend"].fillna(0)
            processed_data.dropna(inplace=True)
            # 调整股价
            processed_data.loc[: ,"Date"] = pd.to_datetime(processed_data.loc[: ,"Date"], format="%Y-%m-%d")
            processed_data = processed_data.sort_values(["SecuritiesCode", "Date"])
            processed_data = processed_data.sort_values("Date", ascending=False)
            processed_data.loc[:, "CumulativeAdjustmentFactor"] = processed_data["AdjustmentFactor"].cumprod()#累乘
            processed_data.loc[:, "AdjustedClose"] = (
                processed_data["CumulativeAdjustmentFactor"] * processed_data["Close"]
                    ).map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)))
            processed_data = processed_data.sort_values("Date")
            processed_data.loc[processed_data["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
            processed_data.loc[:, "AdjustedClose"] = processed_data.loc[:, "AdjustedClose"].ffill()
            train_data, valid_data = self.split_dataframe_randomly(processed_data, random_state=RANDOM_SEED)
            return train_data, valid_data
            
class model_price(abstract_model_factory):
    def __init__(self, task_name, data_processor) -> None:
        super().__init__(task_name, data_processor)
        
# import pandas as pd
# stock = pd.read_csv(os.path.join(project_path, "model_about", "datasets", "股票价格", "stock_prices.csv"))
# stock_list = pd.read_csv(os.path.join(project_path, "model_about", "datasets", "股票价格", "stock_list.csv"))
# data_ = stock.merge(stock_list[['SecuritiesCode','17SectorName']],on='SecuritiesCode')
# # 从 SecuritiesCode 中随机选择5个不同的股票代码
# selected_codes = data_['SecuritiesCode'].drop_duplicates().sample(n=5,random_state=1412)

# # 根据选中的股票代码筛选出所有对应的数据行
# data_ = data_[data_['SecuritiesCode'].isin(selected_codes)]
# data_.rename(columns={'Target': 'Sharpe Ratio'}, inplace=True)
# #恢复索引
# data_.index = range(data_.shape[0])
# data_.to_csv(os.path.join(project_path, "model_about", "datasets", "股票价格", "stock.csv"))