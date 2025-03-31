import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
from model import TEIM
import sys
import time
import datetime
from encode_sequences import encode_sequences  # 导入函数
from transformers import AutoTokenizer, AutoModel

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
        # 从配置文件中获取参数
        self.dim_hidden = TEIM_CONFIG["dim_hidden"]
        self.layers_inter = TEIM_CONFIG["layers_inter"]
        self.dim_seqlevel = TEIM_CONFIG["dim_seqlevel"]
        self.dropout_rate = TEIM_CONFIG.get("dropout_rate", 0.2)  # 使用get方法，如果不存在则使用默认值
        
        # 分别定义 CDR3 和 Epitope 的嵌入维度
        self.dim_emb_cdr3 = TEIM_CONFIG.get("dim_emb_cdr3", 768)  # 使用配置中的值或默认值
        self.dim_emb_epi = TEIM_CONFIG.get("dim_emb_epi", 1024)   # 使用配置中的值或默认值
        
        # 初始化tokenizer和模型
        self.tcr_tokenizer = AutoTokenizer.from_pretrained(TCR_MODEL_NAME)
        self.tcr_model = AutoModel.from_pretrained(TCR_MODEL_NAME)

        self.prot_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
        self.prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME)

        # 定义 CDR3 序列的旋转位置编码模块
        # self.rope_cdr3 = RotaryPositionEmbedding(self.dim_emb_cdr3)
        # 定义抗原表位序列的旋转位置编码模块
        # self.rope_epi = RotaryPositionEmbedding(self.dim_emb_epi)

        # 定义 CDR3 序列的特征提取模块
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(self.dim_emb_cdr3, self.dim_hidden, 1),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
        )
        # 定义抗原表位序列的特征提取模块
        self.seq_epi = nn.Sequential(
            nn.Conv1d(self.dim_emb_epi, self.dim_hidden, 1),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
        )

        self.inter_layers = nn.ModuleList([
            nn.Sequential(
                ResNet(nn.Conv2d(self.dim_hidden, self.dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(self.dim_hidden),
                nn.ReLU(),
            ) for _ in range(self.layers_inter)
        ])

        self.seqlevel_outlyer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),  # 使用配置中的dropout_rate
            nn.Linear(self.dim_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, cdr3_sequences, epitope_sequences):
        device = next(self.parameters()).device  # 获取模型所在设备
        
        # 编码CDR3序列
        batch_size = len(cdr3_sequences)
        cdr3_embeddings = []
        for cdr3_seq in cdr3_sequences:
            inputs = self.tcr_tokenizer(cdr3_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.tcr_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cdr3_embeddings.append(cls_embedding)
        
        # 编码Epitope序列
        epitope_embeddings = []
        for epitope_seq in epitope_sequences:
            inputs = self.prot_tokenizer(epitope_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.prot_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            epitope_embeddings.append(cls_embedding)
        
        # 将嵌入向量堆叠成批次
        cdr3_emb = torch.cat(cdr3_embeddings, dim=0)
        epi_emb = torch.cat(epitope_embeddings, dim=0)

        # 调整 cdr3_emb 的维度
        cdr3_emb = cdr3_emb.unsqueeze(1)
        cdr3_emb = cdr3_emb.transpose(1, 2)

        # 调整 epi_emb 的维度
        epi_emb = epi_emb.unsqueeze(1)
        epi_emb = epi_emb.transpose(1, 2)

        # 直接使用原始嵌入，不应用旋转位置编码
        cdr3_emb_RoPE = cdr3_emb
        epi_emb_RoPE = epi_emb

        ## sequence features
        cdr3_feat = self.seq_cdr3(cdr3_emb_RoPE)  # batch_size, dim_hidden, seq_len
        epi_feat = self.seq_epi(epi_emb_RoPE)

        len_cdr3 = cdr3_feat.shape[2]
        len_epi = epi_feat.shape[2]

        ## get init inter map
        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, len_cdr3, 1])

        inter_map = cdr3_feat_mat * epi_feat_mat

        ## inter layers features
        for layer in self.inter_layers:
            inter_map = layer(inter_map)

        ## output layers
        seqlevel_out = self.seqlevel_outlyer(inter_map)

        return {
            'seqlevel_out': seqlevel_out,
            'inter_map': torch.sigmoid(inter_map),
        }

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
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
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
# 训练过程
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    total_preds, total_labels = [], []

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        cdr3_seqs, epitope_seqs, labels = batch
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(cdr3_seqs, epitope_seqs)
        seqlevel_out = outputs['seqlevel_out']
        
        # 确保形状匹配
        if seqlevel_out.shape == torch.Size([len(labels), 1]):
            seqlevel_out = seqlevel_out.view(-1)
        
        # 计算损失
        loss = criterion(seqlevel_out, labels)
        running_loss += loss.item()
        
        # 直接使用模型输出的连续概率作为预测分数
        total_preds.extend(seqlevel_out.cpu().detach().numpy())
        total_labels.extend(labels.cpu().numpy())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
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
            
            # 直接使用连续概率作为预测分数
            total_preds.extend(seqlevel_out.cpu().detach().numpy())
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
