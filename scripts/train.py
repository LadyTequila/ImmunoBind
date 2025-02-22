import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from data_loader import encode_sequences  # 导入函数
from GAT_model import GAT
from adj_matrix_generator import full_connected_adj_matrix, similarity_based_adj_matrix, knn_adj_matrix


# 读取 TSV 文件
file_path = "C:/Users/21636/Desktop/ImmunoBind/data/processed/bindingdata_neg_ratio_1.tsv"
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

# 定义 GAT 模型
nfeat = cdr3_vectors.shape[1]  # 输入特征维度
nhid = 64  # 隐藏层维度
nclass = 512  # 输出特征维度（降维后的维度）
dropout = 0.5
alpha = 0.2
nheads = 3

gat_model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads).to(device)

# 假设这里有一个构建邻接矩阵的函数
# 为了简化，这里省略具体实现
adj_matrix = similarity_based_adj_matrix(cdr3_vectors)

# 使用 GAT 进行降维
cdr3_vectors = cdr3_vectors.to(device)
adj_matrix = adj_matrix.to(device)
cdr3_reduced = gat_model(cdr3_vectors, adj_matrix)

# 后续可以继续添加位置编码等步骤

print(cdr3_reduced.shape)