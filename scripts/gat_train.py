import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
from data_loader import encode_sequences  # 导入函数
from GAT_model import GAT
from adj_matrix_generator import full_connected_adj_matrix, similarity_based_adj_matrix, knn_adj_matrix

# 读取 TSV 文件
file_path = "C:/Users/21636/Desktop/ImmunoBind/data/processed/bindingdata_neg_ratio_5.tsv"
df = pd.read_csv(file_path, sep='\t')

# 获取 CDR3 序列和 epitope 序列
cdr3_sequences = df["CDR3"].tolist()
epitope_sequences = df["Epitope"].tolist()

# 加载 TCR-BERT 和 ProtBERT 模型和 tokenizer
tcr_tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
tcr_model = AutoModel.from_pretrained("wukevin/tcr-bert")

prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

# 判断是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcr_model.to(device)
prot_model.to(device)

# 调用 encode_sequences 函数对序列进行编码
encoded_df = encode_sequences(
    cdr3_sequences, epitope_sequences, tcr_tokenizer, tcr_model, prot_tokenizer, prot_model, device, df
)

# 从 DataFrame 中提取 CDR3 和 Epitope 编码后的向量
cdr3_vectors = torch.tensor([vec for vec in encoded_df["CDR3_embedding"]], dtype=torch.float32).to(device)
epitope_vectors = torch.tensor([vec for vec in encoded_df["Epitope_embedding"]], dtype=torch.float32).to(device)

# 定义 CDR3 的 GAT 模型和重构器
nfeat_cdr3 = cdr3_vectors.shape[1]  # 输入特征维度
nhid = 64  # 隐藏层维度
nclass = 512  # 输出特征维度（降维后的维度）
dropout = 0.5
alpha = 0.2
nheads = 3

gat_model_cdr3 = GAT(nfeat_cdr3, nhid, nclass, dropout, alpha, nheads).to(device)
reconstructor_cdr3 = nn.Linear(nclass, nfeat_cdr3).to(device)

# 定义 Epitope 的 GAT 模型和重构器
nfeat_epitope = epitope_vectors.shape[1]
gat_model_epitope = GAT(nfeat_epitope, nhid, nclass, dropout, alpha, nheads).to(device)
reconstructor_epitope = nn.Linear(nclass, nfeat_epitope).to(device)

# 定义优化器和损失函数
optimizer_cdr3 = torch.optim.Adam(list(gat_model_cdr3.parameters()) + list(reconstructor_cdr3.parameters()), lr=0.001)
optimizer_epitope = torch.optim.Adam(list(gat_model_epitope.parameters()) + list(reconstructor_epitope.parameters()), lr=0.001)
criterion = nn.MSELoss()

# 构建邻接矩阵
adj_matrix_cdr3 = similarity_based_adj_matrix(cdr3_vectors.cpu()).to(device)
adj_matrix_epitope = similarity_based_adj_matrix(epitope_vectors.cpu()).to(device)

# 训练参数
num_epochs = 100
for epoch in range(num_epochs):
    # 训练 CDR3 模型
    gat_model_cdr3.train()
    reconstructor_cdr3.train()
    optimizer_cdr3.zero_grad()

    # 前向传播：降维
    cdr3_reduced = gat_model_cdr3(cdr3_vectors, adj_matrix_cdr3)

    # 重构
    cdr3_reconstructed = reconstructor_cdr3(cdr3_reduced)

    # 计算损失
    loss_cdr3 = criterion(cdr3_reconstructed, cdr3_vectors)

    # 反向传播和优化
    loss_cdr3.backward()
    optimizer_cdr3.step()

    # 训练 Epitope 模型
    gat_model_epitope.train()
    reconstructor_epitope.train()
    optimizer_epitope.zero_grad()

    # 前向传播：降维
    epitope_reduced = gat_model_epitope(epitope_vectors, adj_matrix_epitope)

    # 重构
    epitope_reconstructed = reconstructor_epitope(epitope_reduced)

    # 计算损失
    loss_epitope = criterion(epitope_reconstructed, epitope_vectors)

    # 反向传播和优化
    loss_epitope.backward()
    optimizer_epitope.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], CDR3 Loss: {loss_cdr3.item():.4f}, Epitope Loss: {loss_epitope.item():.4f}')

# 保存训练好的 GAT 模型
torch.save(gat_model_cdr3.state_dict(), 'C:/Users/21636/Desktop/ImmunoBind/saved_models/gat_model_for_cdr3.pth')
torch.save(gat_model_epitope.state_dict(), 'C:/Users/21636/Desktop/ImmunoBind/saved_models/gat_model_for_epitope.pth')