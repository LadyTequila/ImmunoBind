# evaluate.py
import torch
from data_loader import get_dataloader
from utils import parse_batch, compute_metrics

def evaluate_model(model, config):
    # 加载验证或测试数据，若数据路径不同可在 config 中指定
    dataloader = get_dataloader(config, mode="eval")
    model.eval()
    metrics_total = initialize_metrics()  # 伪函数，初始化指标容器
    with torch.no_grad():
        for batch in dataloader:
            x_seq, x_epitope, structure_data, labels_seq, labels_res, labels_struct = parse_batch(batch, config.DEVICE)
            seq_pred, res_pred, struct_pred = model(x_seq, x_epitope, structure_data)
            batch_metrics = compute_metrics(seq_pred, res_pred, struct_pred,
                                            labels_seq, labels_res, labels_struct)
            update_metrics(metrics_total, batch_metrics)  # 伪函数，更新累计指标
    print("Evaluation metrics:", metrics_total)

# initialize_metrics() 和 update_metrics() 均可在 utils.py 中定义
