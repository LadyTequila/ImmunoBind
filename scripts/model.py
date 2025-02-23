import torch
import torch.nn as nn
from data_loader import encode_sequences  # 导入函数
from transformers import AutoTokenizer, AutoModel
import pandas as pd

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
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        seq_len = x.shape[2]
        freqs = 10000 ** (-2 * torch.arange(0, self.dim // 2, 2) / self.dim)
        position = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', position, freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        emb_complex = torch.polar(torch.ones_like(x_complex), emb)
        x_rotated = torch.view_as_real(x_complex * emb_complex).flatten(-2)
        return x_rotated.type_as(x)

# 定义 TEIM 模型类
class TEIM(nn.Module):
    def __init__(self, file_path):
        super().__init__()
        self.dim_hidden = 128
        self.layers_inter = 3
        self.dim_seqlevel = 64
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path, sep='\t')
        self.cdr3_sequences = self.df["CDR3"].tolist()
        self.epitope_sequences = self.df["Epitope"].tolist()

        self.tcr_tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
        self.tcr_model = AutoModel.from_pretrained("wukevin/tcr-bert")

        self.prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

        # 分别定义 CDR3 和 Epitope 的嵌入维度
        dim_emb_cdr3 = 768
        dim_emb_epi = 1024

        # 定义 CDR3 序列的旋转位置编码模块
        self.rope_cdr3 = RotaryPositionEmbedding(dim_emb_cdr3)
        # 定义抗原表位序列的旋转位置编码模块
        self.rope_epi = RotaryPositionEmbedding(dim_emb_epi)

        # 定义 CDR3 序列的特征提取模块
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(dim_emb_cdr3, self.dim_hidden, 1),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
        )
        # 定义抗原表位序列的特征提取模块
        self.seq_epi = nn.Sequential(
            nn.Conv1d(dim_emb_epi, self.dim_hidden, 1),
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
            nn.Dropout(0.2),
            nn.Linear(self.dim_seqlevel, 1),
            nn.Sigmoid()
        )

        self.reslevel_outlyer = nn.Conv2d(
            in_channels=self.dim_hidden,
            out_channels=2,
            kernel_size=2 * self.layers_inter + 1,
            padding=self.layers_inter
        )

    def forward(self):
        encoded_df = encode_sequences(
            self.cdr3_sequences, self.epitope_sequences, self.tcr_tokenizer, self.tcr_model,
            self.prot_tokenizer, self.prot_model, None, self.df
        )

        # 从 encoded_df 中提取 CDR3 和 Epitope 的嵌入向量
        cdr3_emb = torch.tensor(encoded_df["CDR3_embedding"].tolist(), dtype=torch.float32)
        epi_emb = torch.tensor(encoded_df["Epitope_embedding"].tolist(), dtype=torch.float32)

        # 调整 cdr3_emb 的维度
        cdr3_emb = cdr3_emb.unsqueeze(1)
        cdr3_emb = cdr3_emb.transpose(1, 2)

        # 调整 epi_emb 的维度
        epi_emb = epi_emb.unsqueeze(1)
        epi_emb = epi_emb.transpose(1, 2)

        # 应用旋转位置编码
        cdr3_emb_RoPE = self.rope_cdr3(cdr3_emb)
        epi_emb_RoPE = self.rope_epi(epi_emb)

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
        reslevel_out = self.reslevel_outlyer(inter_map)
        out_dist = torch.relu(reslevel_out[:, 0, :, :])
        out_bd = torch.sigmoid(reslevel_out[:, 1, :, :])
        reslevel_out = torch.cat([out_dist.unsqueeze(-1), out_bd.unsqueeze(-1)], axis=-1)

        return {
            'seqlevel_out': seqlevel_out,
            'reslevel_out': reslevel_out,
            'inter_map': inter_map,
        }
