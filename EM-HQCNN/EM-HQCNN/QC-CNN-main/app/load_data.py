import os
from torch.utils.data import Dataset
import sys
import numpy as np
import torch
from sklearn.preprocessing import normalize

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


class MyCSVDatasetReader(Dataset):
    def __init__(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} 文件不存在")
        print(f"加载数据文件: {csv_path}")

        data = np.genfromtxt(csv_path, delimiter=',')
        self.features = (np.pi * data[:, :-1]) / 255  # 归一化并缩放到 [0, π]
        self.labels = data[:, -1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = self.labels[idx]
        return {'feature': feature, 'label': label}

    def get_labels(self):
        return self.labels


if __name__ == "__main__":
    dataset = MyCSVDatasetReader('./app/digits.csv')
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sample = dataset[index]
    print(sample)
