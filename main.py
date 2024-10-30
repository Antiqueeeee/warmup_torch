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