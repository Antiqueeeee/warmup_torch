from dotenv import load_dotenv
from config import *
load_dotenv()

from model_about.models.股票价格.price_model import model_price_data_processor
from torch import optim
from torch import nn
from model_about.models.股票价格.price_model import model_price


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
model = model_price(task_name, data_processor)
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE) #定义优化器
criterion = nn.MSELoss() #定义损失函数
# 初始化早停参数
early_stopping_patience = 3  # 设置容忍的epoch数，即在这么多epoch后如果没有改进就停止
early_stopping_counter = 0  # 用于跟踪没有改进的epoch数
best_train_rmse = float('inf')  # 初始化最佳的训练RMSE

for epoch in range(EPOCHS):
    model.train()
    
    # 前向传播
    for X_batch, y_batch in train_dataloader:
        print(X_batch.shape, y_batch.shape)
        optimizer.zero_grad()
        y_pred = model(X_batch.to(DEVICE))
        loss = criterion(y_pred, y_batch.to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
    
    # if epoch % 50 == 0 :
    #     model.eval()
    #     with torch.no_grad():
    #         y_pred = model()
    
        