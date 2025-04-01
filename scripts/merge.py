'''
可能存在梯度消失情况
'''

import os
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from model import TEIM
from encode_sequences import encode_sequences # 导入函数

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

'''
模型
'''

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEIM_CONFIG, TCR_MODEL_NAME, PROT_MODEL_NAME, DEVICE

# 定义残差网络模块，用于构建交互层中的残差块
class ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, data):
        tmp_data = self.cnn(data)
        out = tmp_data + data
        return out

# 定义 TEIM 模型类
class TEIM(nn.Module):
    def __init__(self, file_path=None):
        super().__init__()
        
        # 从配置中获取参数
        self.dim_hidden = TEIM_CONFIG["dim_hidden"]
        self.dim_emb_cdr3 = TEIM_CONFIG["dim_emb_cdr3"]
        self.dim_emb_epi = TEIM_CONFIG["dim_emb_epi"]
        self.dropout_rate = TEIM_CONFIG["dropout_rate"]
        
        # 初始化tokenizer和预训练模型
        self.tcr_tokenizer = AutoTokenizer.from_pretrained(TCR_MODEL_NAME)
        self.tcr_model = AutoModel.from_pretrained(TCR_MODEL_NAME)
        self.prot_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
        self.prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME)
        
        # 序列特征提取层
        self.seq_cdr3 = nn.Linear(self.dim_emb_cdr3, self.dim_hidden)
        self.seq_epi = nn.Linear(self.dim_emb_epi, self.dim_hidden)
        
        # # UNet
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(self.dim_hidden, self.dim_hidden*2, 3, padding=1),
        #     nn.BatchNorm2d(self.dim_hidden*2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1,2))
        # )
        
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(self.dim_hidden*2, self.dim_hidden*4, 3, padding=1),
        #     nn.BatchNorm2d(self.dim_hidden*4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1,1))
        # )
        
        # self.bridge = nn.Sequential(
        #     nn.Conv2d(self.dim_hidden*4, self.dim_hidden*8, 3, padding=1),
        #     nn.BatchNorm2d(self.dim_hidden*8),
        #     nn.ReLU()
        # )
        
        # self.up1 = nn.Sequential(
        #     nn.ConvTranspose2d(self.dim_hidden*8, self.dim_hidden*4, 2, stride=2),
        #     nn.BatchNorm2d(self.dim_hidden*4),
        #     nn.ReLU()
        # )
        
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose2d(self.dim_hidden*4, self.dim_hidden*2, 2, stride=2),
        #     nn.BatchNorm2d(self.dim_hidden*2),
        #     nn.ReLU()
        # )
        
        # self.final = nn.Sequential(
        #     nn.Conv2d(self.dim_hidden*2, self.dim_hidden, 1),
        #     nn.BatchNorm2d(self.dim_hidden),
        #     nn.ReLU()
        # )

        # # 保持输出层不变
        # self.seqlevel_outlyer = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Flatten(),
        #     nn.Dropout(self.dropout_rate),
        #     nn.Linear(self.dim_hidden, 1),
        #     # nn.Sigmoid()
        # )

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden, 
            nhead=4, 
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=2
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_hidden, 1)
            # nn.Sigmoid()
        )

    def forward(self, cdr3_sequences, epitope_sequences):
        """
        前向传播方法，接受CDR3和Epitope序列作为输入
        
        参数:
        cdr3_sequences: 批次的CDR3序列
        epitope_sequences: 批次的Epitope序列
        
        返回:
        包含序列级预测和交互映射的字典
        """
        device = next(self.parameters()).device  # 获取模型所在设备
        
        # 编码CDR3序列
        batch_size = len(cdr3_sequences)
        cdr3_embeddings = []
        for cdr3_seq in cdr3_sequences:
            inputs = self.tcr_tokenizer(cdr3_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.tcr_model(**inputs)
            # outputs = self.tcr_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cdr3_embeddings.append(cls_embedding)
        
        # 编码Epitope序列
        epitope_embeddings = []
        for epitope_seq in epitope_sequences:
            inputs = self.prot_tokenizer(epitope_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.prot_model(**inputs)
            # outputs = self.prot_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            epitope_embeddings.append(cls_embedding)
        
        # 将嵌入向量堆叠成批次
        cdr3_emb = torch.cat(cdr3_embeddings, dim=0)  # [batch_size, 768]
        epi_emb = torch.cat(epitope_embeddings, dim=0)  # [batch_size, 1024]
        
        # 先通过线性层转换维度
        cdr3_feat = self.seq_cdr3(cdr3_emb)  # [batch_size, dim_hidden]
        epi_feat = self.seq_epi(epi_emb)     # [batch_size, dim_hidden]
        
        # 计算注意力权重
        attention_weights = torch.matmul(cdr3_feat, epi_feat.transpose(1, 0))  # [batch_size, batch_size]
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # 计算注意力加权的特征
        attended_features = torch.matmul(attention_weights, epi_feat)  # [batch_size, dim_hidden]
        
        # # 将特征转换为适合UNet的形状
        # combined_features = torch.cat([cdr3_feat, attended_features], dim=1)  # [batch_size, dim_hidden*2]
        # unet_input = combined_features.view(batch_size, self.dim_hidden, 1, 2)  # 重塑为图像形式
        # # 继续UNet处理
        # down1 = self.down1(unet_input)
        # down2 = self.down2(down1)
        # bridge = self.bridge(down2)
        # up1 = self.up1(bridge)
        # up2 = self.up2(up1)
        # inter_map = self.final(up2)
        # # 保持输出部分不变
        # seqlevel_out = self.seqlevel_outlyer(inter_map)

        # transformer
        combined_features = torch.stack([cdr3_feat, attended_features], dim=1)  # [batch_size, 2, dim_hidden]
        # transformer编码器处理
        transformer_out = self.transformer_encoder(combined_features)  # [batch_size, 2, dim_hidden]
        # 池化 可以使用平均池化或仅取首token
        pooled_features = transformer_out.mean(dim=1)  # [batch_size, dim_hidden]
        # 分类输出（输出 logits，使用 BCEWithLogitsLoss）
        logits = self.classifier(pooled_features)  # [batch_size, 1]
        # 若想输出概率，则可以加上 sigmoid：probs = torch.sigmoid(logits)

        return {
            # 'seqlevel_out': seqlevel_out, # unet
            'seqlevel_out': logits, # transformer
            # 'inter_map': inter_map, # 暂时注释交互图输出
        }


'''
训练&验证
'''

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BINDING_DATA_PATH, 
    DEVICE, 
    TRAIN_CONFIG, 
    MODEL_DIR, 
    DEFAULT_MODEL_PATH
)

# 使用配置中的设备
device = DEVICE

# 假设数据集的CSV文件包含 'CDR3' 和 'Epitope' 列，以及标签列 'label'
class TEIMDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')
        print(f"数据集加载完成，共 {len(self.df)} 条记录")
        
        self.cdr3_sequences = self.df["CDR3"].tolist()
        self.epitope_sequences = self.df["Epitope"].tolist()
        self.labels = self.df["label"].tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cdr3_seq = self.cdr3_sequences[idx]
        epitope_seq = self.epitope_sequences[idx]
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return cdr3_seq, epitope_seq, label

# 创建训练和验证数据集
print(f"加载数据文件: {BINDING_DATA_PATH}")
dataset = TEIMDataset(BINDING_DATA_PATH)

# 创建训练和验证数据集
train_data, val_data = train_test_split(
    dataset, 
    test_size=TRAIN_CONFIG["test_size"], 
    random_state=TRAIN_CONFIG["random_seed"]
)
train_loader = DataLoader(train_data, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

# 定义模型、损失函数和优化器
print("初始化模型...")
model = TEIM()
model.to(device)
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=TRAIN_CONFIG["learning_rate"], 
    weight_decay=TRAIN_CONFIG["weight_decay"]
)

# 初始化学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TRAIN_CONFIG["lr_scheduler"]["T_max"],
    eta_min=TRAIN_CONFIG["lr_scheduler"]["eta_min"]
)

# 创建模型保存目录
os.makedirs(MODEL_DIR, exist_ok=True)

# 记录训练开始时间
start_time = time.time()

# 早停相关变量
best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
early_stopping_patience = TRAIN_CONFIG.get("early_stopping", 5)
use_early_stopping = early_stopping_patience > 0

# 训练过程
print("开始训练...")
print(f"早停设置: {'启用' if use_early_stopping else '禁用'}")
if use_early_stopping:
    print(f"早停耐心值: {early_stopping_patience}")
print(f"初始学习率: {TRAIN_CONFIG['learning_rate']}")
print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
print(f"总训练轮数: {TRAIN_CONFIG['num_epochs']}")

num_epochs = TRAIN_CONFIG["num_epochs"]
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    total_preds, total_labels = [], []

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        cdr3_seqs, epitope_seqs, labels = batch
        labels = labels.to(device)
        
        with torch.autograd.detect_anomaly(): # 调试 包裹前向 后向 梯度裁剪
            # 前向传播
            outputs = model(cdr3_seqs, epitope_seqs)
            seqlevel_out = outputs['seqlevel_out']
            
            # 确保形状匹配
            if seqlevel_out.shape == torch.Size([len(labels), 1]):
                seqlevel_out = seqlevel_out.view(-1)

            # 调试 输出均值和标准差
            # mean_val = seqlevel_out.mean().item()
            # std_val = seqlevel_out.std().item()
            # print(f"\nseqlevel_out 均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
            
            # 计算损失
            loss = criterion(seqlevel_out, labels)
            running_loss += loss.item()
            
            # 计算预测值
            # preds = seqlevel_out > 0.5
            # total_preds.extend(preds.cpu().numpy())
            total_preds.extend(seqlevel_out.detach().cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            
            # 反向传播前清空梯度
            optimizer.zero_grad()
            # loss.backward()
            # 调试 使用detect_anomaly包裹反向传播过程
            # with torch.autograd.detect_anomaly():
            #     loss.backward()
            
            # 调试 输出每个参数的梯度均值和标准差（仅打印非 None 的梯度）
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_mean = param.grad.mean().item()
            #         grad_std = param.grad.std().item()
            #         print(f"Layer: {name} | grad mean: {grad_mean:.6f} | grad std: {grad_std:.6f}")

            # 梯度裁剪
            if TRAIN_CONFIG["grad_clip"]["enabled"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    TRAIN_CONFIG["grad_clip"]["max_norm"]
                )
            
            optimizer.step()

    avg_train_loss = running_loss / len(train_loader)
    train_auc = roc_auc_score(total_labels, total_preds)
    
    # 验证过程
    model.eval()
    val_loss = 0.0
    total_preds, total_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}'):
            cdr3_seqs, epitope_seqs, labels = batch
            labels = labels.to(device)

            outputs = model(cdr3_seqs, epitope_seqs)
            seqlevel_out = outputs['seqlevel_out']
            
            # 确保形状匹配
            if seqlevel_out.shape == torch.Size([len(labels), 1]):
                seqlevel_out = seqlevel_out.view(-1)
            
            # 计算损失
            loss = criterion(seqlevel_out, labels)
            val_loss += loss.item()
            
            # 计算预测值
            # preds = seqlevel_out > 0.5
            # total_preds.extend(preds.cpu().numpy())
            total_preds.extend(seqlevel_out.detach().cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(total_labels, total_preds)
    
    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 计算本轮训练时间
    epoch_time = time.time() - epoch_start_time
    
    # 打印训练信息
    log_message = f"Epoch {epoch+1}/{num_epochs} | "
    log_message += f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f} | "
    log_message += f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | "
    log_message += f"LR: {current_lr:.6f} | "
    log_message += f"Time: {epoch_time:.2f}s"
    print(log_message)
    
    # 检查是否需要保存模型
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1
        patience_counter = 0
        
        # 保存最佳模型
        best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"保存最佳模型到: {best_model_path} (Val AUC: {best_val_auc:.4f})")
    else:
        patience_counter += 1
        if use_early_stopping:
            print(f"验证AUC未提升，当前耐心值: {patience_counter}/{early_stopping_patience}")
    
    # 早停检查
    if use_early_stopping and patience_counter >= early_stopping_patience:
        print(f"早停触发！验证AUC已连续 {early_stopping_patience} 轮未提升。")
        print(f"最佳验证AUC: {best_val_auc:.4f}，出现在第 {best_epoch} 轮。")
        break

# 训练结束
total_time = time.time() - start_time
print(f"\n=== 训练完成 ===")
print(f"总训练时间: {total_time:.2f}秒")
print(f"最佳验证AUC: {best_val_auc:.4f}，出现在第 {best_epoch} 轮。")

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))

# 保存最终模型
torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
print(f"最终模型已保存到: {DEFAULT_MODEL_PATH}")
