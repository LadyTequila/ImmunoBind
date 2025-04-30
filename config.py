"""
ImmunoBind项目的配置文件
包含数据路径、模型参数和训练参数
"""
import os
import torch

# 项目根目录
# ROOT_DIR = "C:/Users/21636/Desktop/ImmunoBind"
# ROOT_DIR = "C:/Users/薛卜元/Desktop/ImmunoBind"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)

# 数据相关路径
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# 数据文件路径 - 使用相对路径
BINDING_DATA_PATH = "../data/raw/process_neg_ratio_50.tsv"

# 模型保存路径
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)  # 确保目录存在
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "teim_model.pth")

# 预训练模型配置
TCR_MODEL_NAME = "wukevin/tcr-bert"
PROT_MODEL_NAME = "Rostlab/prot_bert_bfd"

# TEIM模型参数
TEIM_CONFIG = {
    "dim_hidden": 256,        # 隐藏层维度
    "layers_inter": 4,        # 交互层数量
    "dim_seqlevel": 128,      # 序列级特征维度
    "dim_emb_cdr3": 768,      # CDR3嵌入维度
    "dim_emb_epi": 1024,      # Epitope嵌入维度
    "dropout_rate": 0.3,      # 增加Dropout比率以增强正则化
}

# 训练参数
TRAIN_CONFIG = {
    "batch_size": 64,         # 减小批次大小以提高稳定性
    "num_epochs": 50,         # 增加训练轮数
    "learning_rate": 1e-4,    # 调整学习率
    "weight_decay": 5e-4,     # 增加权重衰减以增强正则化
    "test_size": 0.2,         # 测试集比例
    "random_seed": 42,        # 随机种子
    "early_stopping": 10,     # 增加早停耐心值
    
    # 学习率调度器参数
    "lr_scheduler": {
        "type": "cosine_warmup",  # 使用带预热的余弦退火调度器
        "warmup_epochs": 3,       # 预热轮数
        "T_max": 27,              # 总轮数减去预热轮数
        "eta_min": 1e-7,          # 最小学习率
    },
    
    # 梯度裁剪
    "grad_clip": {
        "enabled": True,
        "max_norm": 0.5,      # 减小梯度裁剪阈值以增强稳定性
    },
    
    # 标签平滑
    "label_smoothing": 0.1,   # 添加标签平滑
}

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
