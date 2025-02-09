# utils.py
import torch.nn.functional as F
import torch

def compute_loss(seq_pred, res_pred, struct_pred,
                 labels_seq, labels_res, labels_struct, loss_weights):
    # 序列级损失：例如使用二分类交叉熵
    loss_seq = F.binary_cross_entropy(seq_pred, labels_seq)
    # 残基级损失：例如均方误差
    loss_res = F.mse_loss(res_pred, labels_res)
    # 结构分支损失：当有预测时计算，否则置 0
    loss_struct = F.mse_loss(struct_pred, labels_struct) if struct_pred is not None else 0
    total_loss = (loss_weights["seq_loss"] * loss_seq +
                  loss_weights["res_loss"] * loss_res +
                  loss_weights["struct_loss"] * loss_struct)
    return total_loss

def compute_metrics(seq_pred, res_pred, struct_pred,
                    labels_seq, labels_res, labels_struct):
    # 伪代码：计算各项指标（如 AUC、F1 分数、MSE 等）
    metrics = {}
    metrics["seq_auc"] = calculate_auc(seq_pred, labels_seq)
    metrics["res_mse"] = calculate_mse(res_pred, labels_res)
    if struct_pred is not None:
        metrics["struct_mse"] = calculate_mse(struct_pred, labels_struct)
    return metrics

def log_metrics(epoch, loss):
    print(f"Epoch {epoch} - Loss: {loss}")

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def parse_batch(batch, device):
    # 解析 batch 数据，并将数据移动到指定设备上
    # 假设 batch 为字典或元组，包含所有输入和标签
    x_seq = batch["tcr_seq"].to(device)
    x_epitope = batch["epitope"].to(device)
    structure_data = batch.get("structure")
    if structure_data is not None:
        structure_data = structure_data.to(device)
    labels_seq = batch["label_seq"].to(device)
    labels_res = batch["label_res"].to(device)
    labels_struct = batch.get("label_struct")
    if labels_struct is not None:
        labels_struct = labels_struct.to(device)
    return x_seq, x_epitope, structure_data, labels_seq, labels_res, labels_struct

# 下面定义 calculate_auc()、calculate_mse()、initialize_metrics()、update_metrics() 等辅助函数
def calculate_auc(pred, target):
    # 伪代码：计算 AUC
    return 0.95

def calculate_mse(pred, target):
    return ((pred - target) ** 2).mean().item()

def initialize_metrics():
    return {"seq_auc": 0, "res_mse": 0, "struct_mse": 0, "count": 0}

def update_metrics(total_metrics, batch_metrics):
    # 伪代码：累加指标，更新计数
    total_metrics["seq_auc"] += batch_metrics["seq_auc"]
    total_metrics["res_mse"] += batch_metrics["res_mse"]
    if "struct_mse" in batch_metrics:
        total_metrics["struct_mse"] += batch_metrics["struct_mse"]
    total_metrics["count"] += 1
