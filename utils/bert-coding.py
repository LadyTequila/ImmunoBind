import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# 读取 TSV 文件
file_path = "C:/Users/21636/Desktop/ImmunoBind/data/tcr.tsv"  # 修改为你的文件路径
df = pd.read_csv(file_path, sep="\t", encoding='ISO-8859-1')  # 尝试使用 ISO-8859-1 编码

# 获取 CDR3 序列和 epitope 序列
cdr3_sequences = df["CDR3"].tolist()
epitope_sequences = df["Epitope"].tolist()

# 加载 TCR-BERT 和 ProtBERT 模型和 tokenizer
tcr_tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
tcr_model = AutoModel.from_pretrained("wukevin/tcr-bert")

prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

# 编码 CDR3 序列的函数
def encode_cdr3_sequence(sequence):
    inputs = tcr_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tcr_model(**inputs)
    # 获取 [CLS] token 对应的嵌入向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 编码 epitope 序列的函数
def encode_epitope_sequence(sequence):
    inputs = prot_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = prot_model(**inputs)
    # 获取 [CLS] token 对应的嵌入向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 初始化列表来存储嵌入向量
cdr3_embeddings = []
epitope_embeddings = []

# 遍历所有序列并进行编码
for cdr3_seq, epitope_seq in zip(cdr3_sequences, epitope_sequences):
    cdr3_embedding = encode_cdr3_sequence(cdr3_seq)
    epitope_embedding = encode_epitope_sequence(epitope_seq)
    
    cdr3_embeddings.append(cdr3_embedding)
    epitope_embeddings.append(epitope_embedding)

# 将嵌入向量转为 NumPy 数组
cdr3_embeddings_numpy = torch.cat(cdr3_embeddings, dim=0).numpy()
epitope_embeddings_numpy = torch.cat(epitope_embeddings, dim=0).numpy()

# 直接替换原 DataFrame 中的 CDR3 和 epitope 序列列
df["CDR3"] = list(cdr3_embeddings_numpy)
df["Epitope"] = list(epitope_embeddings_numpy)

# 查看修改后的 DataFrame
print(df.head())

