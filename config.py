import os 
import numpy as np
import random
import torch
from dotenv import load_dotenv
load_dotenv()
# Train About
RANDOM_SEED = int(os.environ.get("DEFAULT_RANDOM_SEED", None))
LEARNING_RATE = 0.01 #学习率
TRAINSET_BATCH=int(os.environ.get("DEFAULT_TRAINSET_BATCH", None))
VALIDSET_BATCH=int(os.environ.get("DEFAULT_VALIDSET_BATCH", None))
EPOCHS = 1000 #迭代epoch
# Model About
WINDOW_SIZE=int(os.environ.get("DEFAULT_WINDOW_SIZE", None))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_SIZE=13
PRED_LEN=1
N_HEAD=1
NUM_LAYERS=1
DROPOUT=0.1

# Server About
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8000))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "127.0.0.1")

# Basement
def set_seed():
    seed_value = RANDOM_SEED
    random.seed(seed_value)  # Python内置的随机库
    np.random.seed(seed_value)  # Numpy库
    torch.manual_seed(seed_value)  # 为CPU设置种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置种子
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法将是确定的
    torch.backends.cudnn.benchmark = False


# 设定随机数种子
set_seed()