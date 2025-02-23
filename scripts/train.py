import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
from model import

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 假设数据集的CSV文件包含 'CDR3' 和 'Epitope' 列，以及标签列 'label'
class TEIMDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')
        self.cdr3_sequences = self.df["CDR3"].tolist()
        self.epitope_sequences = self.df["Epitope"].tolist()
        self.labels = self.df["label"].tolist()  # 假设目标标签列为 'label'
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cdr3_seq = self.cdr3_sequences[idx]
        epitope_seq = self.epitope_sequences[idx]
        label = self.labels[idx]
        return cdr3_seq, epitope_seq, label

# 创建训练和验证数据集
file_path = "../data/processed/bindingdata_neg_ratio_1.tsv"
dataset = TEIMDataset(file_path)

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 定义模型、损失函数和优化器
model = TEIM(file_path)
model.to(device)  # 将模型迁移到GPU

criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_preds, total_labels = [], []

    for cdr3_seq, epitope_seq, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # 将数据迁移到GPU
        cdr3_seq, epitope_seq, label = cdr3_seq.to(device), epitope_seq.to(device), label.to(device)
        
        # 前向传播
        outputs = model()  # 模型输出字典
        seqlevel_out = outputs['seqlevel_out'].squeeze()  # 获取序列层的输出
        
        # 计算损失
        loss = criterion(seqlevel_out, label)
        running_loss += loss.item()

        # 计算预测值
        preds = seqlevel_out > 0.5
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(label.cpu().numpy())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = running_loss / len(train_loader)
    train_auc = roc_auc_score(total_labels, total_preds)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")

    # 验证过程
    model.eval()
    val_loss = 0.0
    total_preds, total_labels = [], []

    with torch.no_grad():
        for cdr3_seq, epitope_seq, label in tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}'):
            # 将数据迁移到GPU
            cdr3_seq, epitope_seq, label = cdr3_seq.to(device), epitope_seq.to(device), label.to(device)

            # 前向传播
            outputs = model()
            seqlevel_out = outputs['seqlevel_out'].squeeze()
            
            # 计算损失
            loss = criterion(seqlevel_out, label)
            val_loss += loss.item()

            # 计算预测值
            preds = seqlevel_out > 0.5
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(label.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(total_labels, total_preds)
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

# 保存模型
torch.save(model.state_dict(), "../saved_models/try.pth")
