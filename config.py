"""
ImmunoBind项目的配置文件
包含数据路径、模型参数和训练参数
"""

import os
import torch

# 项目根目录 - 使用绝对路径确保准确性
ROOT_DIR = "C:/Users/21636/Desktop/ImmunoBind"

# 数据相关路径
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# 数据文件路径 - 使用绝对路径
BINDING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "bindingdata_neg_ratio_1.tsv")

# 模型保存路径
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)  # 确保目录存在
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "teim_model.pth")

# 预训练模型配置
TCR_MODEL_NAME = "wukevin/tcr-bert"
PROT_MODEL_NAME = "Rostlab/prot_bert_bfd"

# TEIM模型参数
TEIM_CONFIG = {
    "dim_hidden": 128,        # 隐藏层维度
    "layers_inter": 3,        # 交互层数量
    "dim_seqlevel": 64,       # 序列级特征维度
    "dim_emb_cdr3": 768,      # CDR3嵌入维度
    "dim_emb_epi": 1024,      # Epitope嵌入维度
    "dropout_rate": 0.2,      # Dropout比率
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 32,         # 批次大小
    "num_epochs": 10,         # 训练轮数
    "learning_rate": 1e-4,    # 学习率
    "weight_decay": 1e-5,     # 权重衰减
    "test_size": 0.2,         # 测试集比例
    "random_seed": 42,        # 随机种子
    "early_stopping": 5,      # 早停轮数
}

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
