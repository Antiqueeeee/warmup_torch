import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_path)

import pandas as pd
import numpy as np
import math
from decimal import ROUND_HALF_UP, Decimal
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from model_about.models.model_abstract import abstract_model_factory, abstract_data_processor
from config import *
from torch import nn
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
            processed_data['17SectorName'] = LabelEncoder().fit_transform(processed_data['17SectorName'])
            train_data, valid_data = self.split_dataframe_randomly(processed_data, random_state=RANDOM_SEED)
            return train_data, valid_data

      
# 定义位置编码模块
class PositionalEncoding(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码张量，全零初始化
        pe = torch.zeros(max_len, d_model)
        # 生成位置序列
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项，用于调整正弦和余弦函数的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 使用正弦函数生成位置编码的偶数部分
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数生成位置编码的奇数部分
        pe[:, 1::2] = torch.cos(position * div_term)
        # 修改pe形状并固定维度顺序
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册位置编码为模型的缓冲区
        self.register_buffer('pe', pe)

    # 前向传播函数，将位置编码添加到输入张量上
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

      
class model_price(abstract_model_factory):
    def __init__(self, task_name, data_processor) -> None:
        super().__init__(task_name, data_processor)
        self.model_type = 'Transformer'
        self.src_mask = None  # 源数据掩码，用于遮蔽未来信息
        # 实例化位置编码模块，注意此时因为数据本来就是浮点数，所以无需再次embedding
        self.pos_encoder = PositionalEncoding(FEATURE_SIZE)
        # 创建Transformer编码层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=FEATURE_SIZE, nhead=N_HEAD, dropout=DROPOUT)
        # 根据编码层构建Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=NUM_LAYERS)
        # 初始化解码器，将Transformer输出转换为预测长度
        self.decoder = nn.Linear(FEATURE_SIZE, PRED_LEN)
        # 初始化模型权重
        self.init_weights()
        
    # 初始化权重函数，用于设置解码器的初始权重
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # 前向传播函数
    def forward(self, src):
        # 检查并更新掩码
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        # 应用位置编码
        src = self.pos_encoder(src)
        # 通过编码器处理数据
        output = self.transformer_encoder(src, self.src_mask)
        # 选择最后一个时间步的输出进行解码
        output = output[:, -1, :]
        # 通过解码器生成最终预测
        output = self.decoder(output)
        return output

    # 生成后续掩码，用于遮蔽未来信息
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
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