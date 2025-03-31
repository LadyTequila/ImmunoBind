import pandas as pd
from Bio.SubsMat import MatrixInfo

df = pd.read_csv('data/raw/SearchTable-2024-12-15 09_08_53.829.tsv', sep='\t')

blosum62 = MatrixInfo.blosum62

def calculate_blosum_score(seq1, seq2, blosum_matrix):
    score = 0
    for a, b in zip(seq1, seq2):
        # 如果某个氨基酸在BLOSUM62矩阵中没有对应的得分，默认为0
        score += blosum_matrix.get((a, b), 0)
    return score

df['Blosum_Score'] = df.apply(lambda row: calculate_blosum_score(row['CDR3'], row['Epitope'], blosum62), axis=1)

#print(df)

df.to_csv('data/raw/blosum62_encoded.tsv', sep='\t', index=False)