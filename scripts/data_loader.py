import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# 编码序列的函数
def encode_sequences(cdr3_sequences, epitope_sequences, tcr_tokenizer, tcr_model, prot_tokenizer, prot_model, device, df):
    """
    对输入的 CDR3 和 Epitope 序列进行编码，返回编码后的嵌入向量的 DataFrame。
    
    cdr3_sequences: list，CDR3 序列列表
    epitope_sequences: list，Epitope 序列列表
    tcr_tokenizer: TCR-BERT tokenizer
    tcr_model: TCR-BERT 模型
    prot_tokenizer: ProtBERT tokenizer
    prot_model: ProtBERT 模型
    device: 计算设备（cuda 或 cpu）
    df: 原始的 DataFrame
    
    返回值: DataFrame，其中包含编码后的 CDR3 和 Epitope 嵌入向量
    """
    # 初始化列表来存储嵌入向量
    cdr3_embeddings = []
    epitope_embeddings = []

    # 编码序列的函数（封装）
    def encode_sequence(sequence, tokenizer, model):
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取 [CLS] token 对应的嵌入向量
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

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

    # 将原 DataFrame 中的其他列也添加到新的 DataFrame 中
    for col in df.columns:
        if col not in ["CDR3", "Epitope"]:
            encoded_df[col] = df[col]

    return encoded_df

