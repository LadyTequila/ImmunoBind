# config.py
class Config:
    # 超参数设置
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATE_INTERVAL = 5

    # 数据与模型路径
    DATA_PATH = "data/dataset/"
    MODEL_SAVE_PATH = "checkpoints/best_model.pth"

    # 损失权重
    LOSS_WEIGHTS = {
        "seq_loss": 1.0,    # 序列级损失权重
        "res_loss": 0.5,    # 残基级损失权重
        "struct_loss": 0.3  # 结构融合分支损失权重
    }

    # 其他可能的设置（如网络结构参数、dropout比例、设备等）
    DEVICE = "cuda"  # 或 "cpu"
