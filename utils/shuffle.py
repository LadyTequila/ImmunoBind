import pandas as pd
import numpy as np

def generate_negative_samples(file_path, feature_columns, label_column, negative_ratio=1):
    # 读取 TSV 文件
    df = pd.read_csv(file_path, sep='\t')

    # 提取特征和标签
    features = df[feature_columns].values
    labels = df[label_column].values

    # 确保正样本标签为 1
    positive_labels = np.ones_like(labels)

    all_negative_features = []
    all_negative_labels = []
    for _ in range(negative_ratio):
        feature1 = features[:, 0]
        feature2 = features[:, 1]
        shuffled_feature2 = np.random.permutation(feature2)
        negative_features = np.column_stack((feature1, shuffled_feature2))
        negative_labels = np.zeros(len(feature1))
        all_negative_features.append(negative_features)
        all_negative_labels.append(negative_labels)

    all_negative_features = np.vstack(all_negative_features)
    all_negative_labels = np.hstack(all_negative_labels)

    # 合并正负样本
    all_features = np.vstack((features, all_negative_features))
    all_labels = np.hstack((positive_labels, all_negative_labels))

    return all_features, all_labels

def save_to_new_file(features, labels, output_file_path, feature_columns, label_column, negative_ratio):
    # 创建一个包含合并后特征和标签的 DataFrame
    data = {**dict(zip(feature_columns, features.T)), label_column: labels}
    result_df = pd.DataFrame(data)

    # 处理文件名，添加负样本比例信息
    file_name, file_ext = output_file_path.rsplit('.', 1)
    new_file_name = f"{file_name}_neg_ratio_{negative_ratio}.{file_ext}"

    # 将 DataFrame 保存为新的 TSV 文件
    result_df.to_csv(new_file_name, sep='\t', na_rep='nan', index=False)
    print(f"合并后的特征和标签已保存到 {new_file_name}")


file_path = 'data/raw/blosum62_encoded.tsv'
feature_columns = ['CDR3', 'Epitope']
label_column = 'label'
output_file_path = 'data/processed/blosum62_encoded_with_negatives.tsv'
negative_ratio = 2 # 负样本数量是正样本数量的倍数

features, labels = generate_negative_samples(file_path, feature_columns, label_column, negative_ratio)

# 保存到新文件
save_to_new_file(features, labels, output_file_path, feature_columns, label_column, negative_ratio)