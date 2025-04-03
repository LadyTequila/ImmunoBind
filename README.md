# data\processed\bindingdata_neg_ratio_1.tsv文件的生成过程
## 原始文件
data\raw\SearchTable-2024-12-15 09_08_53.829.tsv
## 第一步
运行utils\blosum62.py
## 第二步
运行utils\shuffle.py
注意修改line 53: negative_ratio