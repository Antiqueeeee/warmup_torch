from dotenv import load_dotenv
from config import *
load_dotenv()

from model_about.models.price_model import model_price_data_processor

task_name = "股票价格"
data_file  = "stock.csv"
data_processor = model_price_data_processor(task_name)
train_set, valid_set = data_processor.pre_processing(data_file)
train_dataloader = data_processor.make_dataloader(train_set
    , shuffle_flag = True
    , window_size = WINDOW_SIZE
    , pred_len = 1
    ,batch_size = TRAINSET_BATCH)

valid_dataloader = data_processor.make_dataloader(valid_set
    , shuffle_flag = True
    , window_size = WINDOW_SIZE
    , pred_len = 1
    ,batch_size = VALIDSET_BATCH)

task_name = "股票价格" 
from model_about.models import price_model
model = price_model(task_name, data_processor)
optimizer = optim.Adam(model.parameters(),lr=learning_rate) #定义优化器
criterion = nn.MSELoss() #定义损失函数
# 初始化早停参数
early_stopping_patience = 3  # 设置容忍的epoch数，即在这么多epoch后如果没有改进就停止
early_stopping_counter = 0  # 用于跟踪没有改进的epoch数
best_train_rmse = float('inf')  # 初始化最佳的训练RMSE