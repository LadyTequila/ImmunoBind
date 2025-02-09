# data_loader.py
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 从文件中加载数据（如：TCR序列、Epitope、结构数据及对应标签）
        self.data = self.load_data(data_path)
        self.transform = transform

    def load_data(self, path):
        # 伪代码：加载数据并返回列表或字典形式的数据样本
        data_list = []  
        # 例如：遍历目录、读取 CSV、JSON 或其他格式的数据
        # data_list.append({"tcr_seq": ..., "epitope": ..., "structure": ..., 
        #                    "label_seq": ..., "label_res": ..., "label_struct": ...})
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample  # 返回一个样本的字典

def get_dataloader(config, mode="train"):
    # 若有区分训练集和验证集，可在此处根据 mode 加载不同数据
    dataset = CustomDataset(config.DATA_PATH, transform=custom_transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    return dataloader

# 自定义变换函数（伪代码）
def custom_transform(sample):
    # 对 sample 中的 TCR 序列、Epitope 或结构信息进行预处理与编码
    # 例如：字符串转换为数值张量、归一化等操作
    return sample
