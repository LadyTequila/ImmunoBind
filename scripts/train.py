# train.py
import torch
from config import Config
from data_loader import get_dataloader
from models import MultiTaskModel
from utils import compute_loss, log_metrics, parse_batch, save_model

def train_model(config):
    # 获取数据加载器（训练数据）
    dataloader = get_dataloader(config, mode="train")
    device = config.DEVICE
    # 初始化模型并移动到设备上
    model = MultiTaskModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            # parse_batch() 解析 batch 数据
            x_seq, x_epitope, structure_data, labels_seq, labels_res, labels_struct = parse_batch(batch, device)
            optimizer.zero_grad()
            # 前向传播
            seq_pred, res_pred, struct_pred = model(x_seq, x_epitope, structure_data)
            # 计算各分支损失并加权求和
            loss = compute_loss(seq_pred, res_pred, struct_pred,
                                labels_seq, labels_res, labels_struct,
                                config.LOSS_WEIGHTS)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log_metrics(epoch, epoch_loss)
        
        # 每隔一定 epoch 进行验证
        if epoch % config.VALIDATE_INTERVAL == 0:
            validate_model(model, config)  # 这里假设有一个验证函数，可调用 evaluate.py 中的方法

    # 保存训练好的模型
    save_model(model, config.MODEL_SAVE_PATH)

# 注意：validate_model() 及 parse_batch() 等函数可放在 utils.py 中，或在此处简单实现。
