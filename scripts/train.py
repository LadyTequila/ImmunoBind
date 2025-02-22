import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from data_loader import encode_sequences  # 导入函数

# 读取 TSV 文件
file_path = "C:/Users/21636/Desktop/ImmunoBind/data/processed/bindingdata_neg_ratio_10.tsv"
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
    cdr3_sequences, epitope_sequences, tcr_tokenizer, tcr_model, prot_tokenizer, prot_model, device
)

# 你可以在此之后继续将 encoded_df 传递给模型进行训练
print(encoded_df.head())
