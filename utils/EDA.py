import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("data/raw/SearchTable-2024-12-15 09_08_53.829.tsv", sep="\t")

# 移除 label 字段
df = df.drop(columns=["label"])

# 设置风格
sns.set(style="whitegrid")

# ============================
# 基本信息输出
# ============================
print("基本信息：\n")
print(df.info())
print("\n基本统计：\n")
print(df.describe(include="all"))

# ============================
# 唯一值统计
# ============================
print("\n每列的唯一值数量：\n")
print(df.nunique())

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================
# CDR3 序列长度分布
# ============================
df["CDR3_length"] = df["CDR3"].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df["CDR3_length"], bins=20, kde=True, color="skyblue")
plt.title("CDR3 序列长度分布")
plt.xlabel("CDR3 长度")
plt.ylabel("频数")
plt.tight_layout()
plt.show()

# ============================
# Epitope 出现频率前20
# ============================
plt.figure(figsize=(10, 6))
top_epitopes = df["Epitope"].value_counts().nlargest(20)
sns.barplot(x=top_epitopes.values, y=top_epitopes.index, palette="viridis")
plt.title("Epitope 出现频率 Top 20")
plt.xlabel("频数")
plt.ylabel("Epitope")
plt.tight_layout()
plt.show()

# ============================
# V区基因频率前20
# ============================
plt.figure(figsize=(10, 6))
top_v = df["V"].value_counts().nlargest(20)
sns.barplot(x=top_v.values, y=top_v.index, palette="magma")
plt.title("V 区基因使用频率 Top 20")
plt.xlabel("频数")
plt.ylabel("V 区")
plt.tight_layout()
plt.show()

# ============================
# J区基因频率前20
# ============================
plt.figure(figsize=(10, 6))
top_j = df["J"].value_counts().nlargest(20)
sns.barplot(x=top_j.values, y=top_j.index, palette="coolwarm")
plt.title("J 区基因使用频率 Top 20")
plt.xlabel("频数")
plt.ylabel("J 区")
plt.tight_layout()
plt.show()

# ============================
# MHC A 与 MHC B 频率分析
# ============================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# MHC A
top_mhc_a = df["MHC A"].value_counts().nlargest(10)
sns.barplot(y=top_mhc_a.index, x=top_mhc_a.values, ax=axes[0], palette="Blues_d")
axes[0].set_title("MHC A 频率 Top 10")
axes[0].set_xlabel("频数")

# MHC B
top_mhc_b = df["MHC B"].value_counts().nlargest(10)
sns.barplot(y=top_mhc_b.index, x=top_mhc_b.values, ax=axes[1], palette="Greens_d")
axes[1].set_title("MHC B 频率 Top 10")
axes[1].set_xlabel("频数")

plt.tight_layout()
plt.show()

# ============================
# V-J 组合热力图（前30）
# ============================
vj_pairs = df.groupby(["V", "J"]).size().reset_index(name="count")
top_v = df["V"].value_counts().nlargest(15).index
top_j = df["J"].value_counts().nlargest(15).index
vj_filtered = vj_pairs[vj_pairs["V"].isin(top_v) & vj_pairs["J"].isin(top_j)]

vj_pivot = vj_filtered.pivot("V", "J", "count").fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(vj_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("V-J 组合热力图（Top 15）")
plt.tight_layout()
plt.show()
