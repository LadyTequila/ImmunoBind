import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class EncodedDataLoader(Dataset):
    def __init__(self, file_path):
        # 读取数据文件，这里假设是 TSV 文件
        self.data = pd.read_csv(file_path, sep='\t')

        # 解析编码后的特征列
        self.features = []
        for feature_str in self.data['encoded_feature']:  # 假设特征列名为 'encoded_feature'
            feature_list = json.loads(feature_str)
            self.features.append(feature_list)

        # 假设最后一列是标签
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 获取特征和标签
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature, label

# 使用示例
if __name__ == "__main__":
    file_path = "C:/Users/21636/Desktop/ImmunoBind/data/embedding/bindingdata__neg_ratio_5_bert_encoded.tsv"  # 替换为你的数据文件路径
    dataset = EncodedDataLoader(file_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for features, labels in dataloader:
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

