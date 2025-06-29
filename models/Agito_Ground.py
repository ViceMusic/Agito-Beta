


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from components.Embedding import DNAEmbedding  # 假设DNAEmbedding在同一目录下
from components.Extract import Encoder  # 假设Encoder在同一目录下

class Agito_Ground(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = DNAEmbedding(vocab_dim=4, window_size=1)
        self.encoder = Encoder(input_dim=4, state_dim=4)

    def forward(self, x):
        u = self.embedding(x)  # (batch_size, seq_len, 4)
        y, gamma, final = self.encoder(u)  # (batch_size, seq_len)
        # 简单平均后作为二分类预测
        return final.mean(dim=1)