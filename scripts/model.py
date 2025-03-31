import torch
import torch.nn as nn
from encode_sequences import encode_sequences  # 导入函数
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import sys
import os

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

# 旋转位置编码实现
'''
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 确保最后一个维度是偶数
        if x.shape[-1] % 2 != 0:
            # 如果不是偶数，添加一个零列使其成为偶数
            pad = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        
        seq_len = x.shape[2]
        # 确保dim是偶数
        dim = self.dim if self.dim % 2 == 0 else self.dim - 1
        
        # 生成频率
        freqs = 10000 ** (-2 * torch.arange(0, dim // 2, device=x.device, dtype=torch.float32) / dim)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', position, freqs)
        
        # 确保freqs的形状与x兼容
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 使用更安全的方式重塑张量
        try:
            # 尝试使用view_as_complex
            x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
            x_complex = torch.view_as_complex(x_reshaped)
            emb_complex = torch.polar(torch.ones_like(x_complex), emb.to(x.device))
            x_rotated = torch.view_as_real(x_complex * emb_complex).flatten(-2)
        except RuntimeError as e:
            # 如果失败，使用替代方法
            x_rotated = x  # 暂时不应用旋转，只返回原始输入
        
        return x_rotated.type_as(x)
'''

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
        
        # UNet结构保持不变
        self.down1 = nn.Sequential(
            nn.Conv2d(self.dim_hidden, self.dim_hidden*2, 3, padding=1),
            nn.BatchNorm2d(self.dim_hidden*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(self.dim_hidden*2, self.dim_hidden*4, 3, padding=1),
            nn.BatchNorm2d(self.dim_hidden*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.bridge = nn.Sequential(
            nn.Conv2d(self.dim_hidden*4, self.dim_hidden*8, 3, padding=1),
            nn.BatchNorm2d(self.dim_hidden*8),
            nn.ReLU()
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.dim_hidden*8, self.dim_hidden*4, 2, stride=2),
            nn.BatchNorm2d(self.dim_hidden*4),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.dim_hidden*4, self.dim_hidden*2, 2, stride=2),
            nn.BatchNorm2d(self.dim_hidden*2),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(self.dim_hidden*2, self.dim_hidden, 1),
            nn.BatchNorm2d(self.dim_hidden),
            nn.ReLU()
        )

        # 保持输出层不变
        self.seqlevel_outlyer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_hidden, 1),
            nn.Sigmoid()
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
        
        # 将特征转换为适合UNet的形状
        combined_features = torch.cat([cdr3_feat, attended_features], dim=1)  # [batch_size, dim_hidden*2]
        unet_input = combined_features.view(batch_size, self.dim_hidden, 2, -1)  # 重塑为图像形式
        
        # 继续UNet处理
        down1 = self.down1(unet_input)
        down2 = self.down2(down1)
        
        bridge = self.bridge(down2)
        
        up1 = self.up1(bridge)
        up2 = self.up2(up1)
        
        inter_map = self.final(up2)

        # 保持输出部分不变
        seqlevel_out = self.seqlevel_outlyer(inter_map)

        return {
            'seqlevel_out': seqlevel_out,
            # 'inter_map': inter_map,  # 暂时注释交互图输出
        }
