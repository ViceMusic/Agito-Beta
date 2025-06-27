'''
此文件为主程序入口文件，负责启动整个应用。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Agito_test import Agito  # 假设agito.py在同一目录下
from Embedding import DNAEmbedding  # 假设DNAEmbedding在同一目录下
from Extract import Encoder  # 假设Encoder在同一目录下





from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = DNAEmbedding(vocab_dim=4, window_size=1)
        self.encoder = Encoder(input_dim=4, state_dim=4, final_dim=1)

    def forward(self, x):
        u = self.embedding(x)  # (batch_size, seq_len, 4)
        y, gamma, final = self.encoder(u)  # (batch_size, seq_len)
        # 简单平均后作为二分类预测
        return final.mean(dim=1)
    
model = Agito(pglobal_mi=0.5, pglobal_mr=0.5, alpha_mi=1.0, beta_mi=1.0, alpha_mr=1.0, beta_mr=1.0)

dataset = DnaDatasetRaw('test2.csv')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)



for epoch in range(10):  # 假设训练10个epoch
    for seq_batch, label_batch in dataloader:
        print("Before step:", model.encoder.A.weight[0,0].item())
        print("更新前参数:", model.alpha_mi, model.beta_mi, model.alpha_mr, model.beta_mr)


        result = model(list(seq_batch))  # 假设模型可以直接处理字符串序列
        loss = criterion(result, label_batch.float())     # batch_y shape: [batch]
        print("计算结果",result)
        print("标签结果",label_batch)
        optimizer.zero_grad()
        # 二重优化
        loss.backward()
        optimizer.step()
        model.update_ab(10,0,result,label_batch,0.99,0.1)

        print("After step:", model.encoder.A.weight[0,0].item())
        print("更新后参数:", model.alpha_mi, model.beta_mi, model.alpha_mr, model.beta_mr)

    print(f" Loss: {loss.item():.4f}")
    
print("训练完成！",model(["AGCTAGC"]))