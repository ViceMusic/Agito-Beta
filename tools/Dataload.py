from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
class DnaDatasetRaw(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                seq, label = line.strip().split(',')
                self.samples.append((seq, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # 返回 (str, int)



def get_batch_data(path, batch_size=10,shuffle=True):
    dataset = DnaDatasetRaw(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader  # 返回DataLoader对象