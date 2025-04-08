import math
import os
import pandas as pd
import random
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import time
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from model import TEIM
from encode_sequences import encode_sequences

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
        
        # 序列特征提取层 - 使用更复杂的特征提取
        self.seq_cdr3 = nn.Sequential(
            nn.Linear(self.dim_emb_cdr3, self.dim_hidden),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU()
        )
        self.seq_epi = nn.Sequential(
            nn.Linear(self.dim_emb_epi, self.dim_hidden),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU()
        )
        
        # 添加自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.dim_hidden,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden, 
            nhead=4, 
            dropout=self.dropout_rate,
            dim_feedforward=self.dim_hidden * 4,  # 增加前馈网络维度
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=3  # 增加层数
        )

        # 分类层 - 使用更复杂的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_hidden, self.dim_hidden // 2),
            nn.BatchNorm1d(self.dim_hidden // 2), 
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),  # 减小第二层的dropout
            nn.Linear(self.dim_hidden // 2, 1)
        )

        # 添加权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重以改善梯度流动
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        
        # 批量编码CDR3序列，而不是逐个处理
        batch_size = len(cdr3_sequences)
        
        # 编码CDR3序列 - 批量处理
        inputs = self.tcr_tokenizer(cdr3_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.tcr_model(**inputs)
        cdr3_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 编码Epitope序列 - 批量处理
        inputs = self.prot_tokenizer(epitope_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.prot_model(**inputs)
        epi_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, 1024]
        
        # 先通过线性层转换维度
        cdr3_feat = self.seq_cdr3(cdr3_emb)  # [batch_size, dim_hidden]
        epi_feat = self.seq_epi(epi_emb)     # [batch_size, dim_hidden]
        
        # 改进的注意力机制
        # 1. 计算点积注意力
        attention_scores = torch.matmul(cdr3_feat, epi_feat.transpose(1, 0)) / (self.dim_hidden ** 0.5)  # 缩放点积
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 2. 应用自注意力
        cdr3_feat_reshaped = cdr3_feat.unsqueeze(1)  # [batch_size, 1, dim_hidden]
        attended_cdr3, _ = self.self_attention(cdr3_feat_reshaped, cdr3_feat_reshaped, cdr3_feat_reshaped)
        attended_cdr3 = attended_cdr3.squeeze(1)  # [batch_size, dim_hidden]
        
        # 3. 计算注意力加权的特征
        attended_features = torch.matmul(attention_weights, epi_feat)  # [batch_size, dim_hidden]
        
        # 4. 特征融合 - 添加残差连接
        fused_features = attended_features + attended_cdr3
        
        # Transformer处理
        combined_features = torch.stack([fused_features, epi_feat], dim=1)  # [batch_size, 2, dim_hidden]
        transformer_out = self.transformer_encoder(combined_features)  # [batch_size, 2, dim_hidden]
        
        # 使用更复杂的池化策略
        # 1. 平均池化
        mean_pooled = transformer_out.mean(dim=1)  # [batch_size, dim_hidden]
        # 2. 最大池化
        max_pooled, _ = torch.max(transformer_out, dim=1)  # [batch_size, dim_hidden]
        # 3. 组合池化结果
        pooled_features = mean_pooled + max_pooled  # 简单相加
        
        # 分类输出
        logits = self.classifier(pooled_features)  # [batch_size, 1]
        
        return {
            'seqlevel_out': logits,
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
    def __init__(self, file_path, augment=False):
        self.df = pd.read_csv(file_path, sep='\t')
        print(f"数据集加载完成: 共 {len(self.df)} 条记录")
        
        self.cdr3_sequences = self.df["CDR3"].tolist()
        self.epitope_sequences = self.df["Epitope"].tolist()
        self.labels = self.df["label"].tolist()
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def _augment_sequence(self, seq, p=0.1):
        """
        简单的序列增强：随机替换、删除或插入氨基酸
        """
        if not self.augment or random.random() > p:
            return seq
            
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        seq_list = list(seq)
        
        # 随机选择一种操作
        op = random.choice(["replace", "delete", "insert"])
        
        if op == "replace" and len(seq) > 0:
            # 随机替换一个氨基酸
            idx = random.randint(0, len(seq) - 1)
            seq_list[idx] = random.choice(amino_acids)
        elif op == "delete" and len(seq) > 3:
            # 随机删除一个氨基酸
            idx = random.randint(0, len(seq) - 1)
            seq_list.pop(idx)
        elif op == "insert" and len(seq) > 0:
            # 随机插入一个氨基酸
            idx = random.randint(0, len(seq))
            seq_list.insert(idx, random.choice(amino_acids))
            
        return "".join(seq_list)
    
    def __getitem__(self, idx):
        cdr3_seq = self.cdr3_sequences[idx]
        epitope_seq = self.epitope_sequences[idx]
        
        # 对训练集应用数据增强
        if self.augment:
            cdr3_seq = self._augment_sequence(cdr3_seq)
            epitope_seq = self._augment_sequence(epitope_seq)
            
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return cdr3_seq, epitope_seq, label

# 创建训练和验证数据集
print(f"加载数据文件: {BINDING_DATA_PATH}")
dataset = TEIMDataset(BINDING_DATA_PATH)

# 创建训练和验证数据集
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=TRAIN_CONFIG["test_size"], 
    random_state=TRAIN_CONFIG["random_seed"],
    stratify=dataset.labels
)

# 创建训练集和验证集
train_data = TEIMDataset(BINDING_DATA_PATH, augment=True)
train_data = torch.utils.data.Subset(train_data, train_indices)
val_data = torch.utils.data.Subset(dataset, val_indices)

# 检查数据集标签分布
labels = dataset.labels
positive_count = sum(labels)
negative_count = len(labels) - positive_count
print(f"数据集标签分布: 正样本 {positive_count} ({positive_count/len(labels):.2%}), 负样本 {negative_count} ({negative_count/len(labels):.2%})")

# 创建训练和验证数据集
train_data, val_data = train_test_split(
    dataset, 
    test_size=TRAIN_CONFIG["test_size"], 
    random_state=TRAIN_CONFIG["random_seed"],
    stratify=dataset.labels  # 确保训练集和验证集有相同的标签分布
)
train_loader = DataLoader(train_data, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

# 定义模型、损失函数和优化器
print("初始化模型...")
model = TEIM()
model.to(device)

# 检查是否需要标签平滑
label_smoothing = TRAIN_CONFIG.get("label_smoothing", 0.0)

# 使用BCE损失，移除不支持的label_smoothing参数
if label_smoothing > 0:
    # 自定义实现标签平滑
    print(f"使用自定义标签平滑: {label_smoothing}")
    class BCEWithLogitsLossLS(nn.Module):
        def __init__(self, reduction='mean', pos_weight=None, smoothing=0.1):
            super().__init__()
            self.reduction = reduction
            self.pos_weight = pos_weight
            self.smoothing = smoothing
            self.bce = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
            
        def forward(self, pred, target):
            # 应用标签平滑: 将目标值从0/1调整为smoothing/(1-smoothing)
            smooth_target = target * (1 - self.smoothing) + self.smoothing * 0.5
            return self.bce(pred, smooth_target)
    
    criterion = BCEWithLogitsLossLS(
        reduction='mean',
        pos_weight=torch.tensor([1.0]).to(device),
        smoothing=label_smoothing
    )
else:
    # 使用标准BCE损失
    criterion = nn.BCEWithLogitsLoss(
        reduction='mean',
        pos_weight=torch.tensor([1.0]).to(device)
    )

# 使用AdamW优化器
optimizer = optim.AdamW(
    model.parameters(), 
    lr=TRAIN_CONFIG["learning_rate"], 
    weight_decay=TRAIN_CONFIG["weight_decay"],
    betas=(0.9, 0.999),
    eps=1e-8
)

# 实现带预热的学习率调度器
if TRAIN_CONFIG["lr_scheduler"]["type"] == "cosine_warmup":
    # 定义预热轮数和总轮数
    warmup_epochs = TRAIN_CONFIG["lr_scheduler"]["warmup_epochs"]
    t_max = TRAIN_CONFIG["lr_scheduler"]["T_max"]
    
    # 创建学习率调度器
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # 余弦退火
            progress = float(epoch - warmup_epochs) / float(max(1, t_max))
            return max(TRAIN_CONFIG["lr_scheduler"]["eta_min"] / TRAIN_CONFIG["learning_rate"], 
                      0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
else:
    # 使用普通的余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CONFIG["lr_scheduler"]["T_max"],
        eta_min=TRAIN_CONFIG["lr_scheduler"]["eta_min"]
    )

# 创建模型保存目录
os.makedirs(MODEL_DIR, exist_ok=True)

# 初始化梯度缩放器
scaler = GradScaler()

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
print(f"当前学习率: {optimizer.param_groups[0]['lr']:.10f}")
print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
print(f"总训练轮数: {TRAIN_CONFIG['num_epochs']}")

# 添加梯度累积步数
accumulation_steps = 4  # 每4个批次更新一次参数

num_epochs = TRAIN_CONFIG["num_epochs"]
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    total_preds, total_labels = [], []

    # 添加批次索引计数器
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
        cdr3_seqs, epitope_seqs, labels = batch
        labels = labels.to(device)
        
        # 使用新的 autocast 方法
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # 前向传播
            outputs = model(cdr3_seqs, epitope_seqs)
            seqlevel_out = outputs['seqlevel_out']

            # 确保形状匹配
            if seqlevel_out.shape == torch.Size([len(labels), 1]):
                seqlevel_out = seqlevel_out.view(-1)

            # 计算损失
            loss = criterion(seqlevel_out, labels)
            loss = loss / accumulation_steps  # 梯度累积
        
        # 记录原始损失
        running_loss += loss.item() * accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 每accumulation_steps步或最后一个批次时更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
            # 检查梯度是否包含NaN
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    # print(f"警告: 参数 {name} 的梯度包含NaN值")
            
            # 梯度裁剪和优化器步骤
            if not has_nan:
                if TRAIN_CONFIG["grad_clip"]["enabled"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        TRAIN_CONFIG["grad_clip"]["max_norm"]
                    )
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            # else:
            #     print("检测到NaN梯度，跳过此批次更新")
            
            # 清空梯度
            optimizer.zero_grad()
        
        # 计算预测值用于AUC计算
        total_preds.extend(seqlevel_out.detach().cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

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
