import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# 读取 TSV 文件
# file_path = 'data/processed/blosum62_encoded_with_negatives_neg_ratio_5.tsv'
# file_path = 'data/processed/blosum62_encoded_with_negatives_neg_ratio_10.tsv'
# file_path = 'data/processed/blosum62_encoded_with_negatives_neg_ratio_20.tsv'
# file_path = 'data/processed/blosum62_encoded_with_negatives_neg_ratio_50.tsv'
file_path = 'data/processed/blosum62_encoded_with_negatives_neg_ratio_100.tsv'
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

# 编码序列的函数（封装）
def encode_sequence(sequence, tokenizer, model):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取 [CLS] token 对应的嵌入向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 初始化列表来存储嵌入向量
cdr3_embeddings = []
epitope_embeddings = []

# 使用 tqdm 创建进度条，遍历所有序列并进行编码
for cdr3_seq, epitope_seq in tqdm(zip(cdr3_sequences, epitope_sequences), total=len(cdr3_sequences), desc="Encoding CDR3 & Epitope sequences"):
    cdr3_embedding = encode_sequence(cdr3_seq, tcr_tokenizer, tcr_model)
    epitope_embedding = encode_sequence(epitope_seq, prot_tokenizer, prot_model)

    cdr3_embeddings.append(cdr3_embedding)
    epitope_embeddings.append(epitope_embedding)

# 将嵌入向量转为 NumPy 数组
cdr3_embeddings_numpy = torch.cat(cdr3_embeddings, dim=0).cpu().numpy()
epitope_embeddings_numpy = torch.cat(epitope_embeddings, dim=0).cpu().numpy()

# 创建一个新的 DataFrame 来存储编码后的嵌入向量
encoded_df = pd.DataFrame({
    "CDR3_embedding": [list(emb) for emb in cdr3_embeddings_numpy],
    "Epitope_embedding": [list(emb) for emb in epitope_embeddings_numpy]
})

# 可以选择将原 DataFrame 中的其他列也添加到新的 DataFrame 中
for col in df.columns:
    if col not in ["CDR3", "Epitope"]:
        encoded_df[col] = df[col]

# 处理文件名，添加后缀
file_dir, file_name = os.path.split(file_path)
file_dir = 'data/embedding'
file_base, file_ext = os.path.splitext(file_name)
new_file_name = f"{file_base}_bert_encoded{file_ext}"
output_file_path = os.path.join(file_dir, new_file_name)

# 保存新的 DataFrame 为 TSV 文件
encoded_df.to_csv(output_file_path, sep="\t", na_rep='nan')
print(f"编码后的文件已保存到 {output_file_path}")