


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from components.Embedding import DNAEmbedding  # 假设DNAEmbedding在同一目录下
from components.Extract import Encoder  # 假设Encoder在同一目录下
from components.setAtten import SetAttention  # 假设SetAttention在同一目录下

class Agito_Ground(nn.Module):
    def __init__(self, mapping_dim=4):
        super().__init__()
        m=mapping_dim
        self.embedding = DNAEmbedding(vocab_dim=m, window_size=1)
        self.encoder = Encoder(input_dim=m, state_dim=m)
        self.set_attention = SetAttention(state_size=m, num_heads=m, lambda_min=0.7, gamma_max=0.3)

        # 临时区域
        self.final_fc = nn.Linear(m, 1)  # 映射成一个数字

    def forward(self, x):
        u = self.embedding(x)  # (batch_size, seq_len, 4)
        encoder_output, gamma, final = self.encoder(u)  # y为(batch_size,seq_len,state_size)final为(batch_size, seq_len)
        out = self.set_attention(encoder_output)  # (batch_size, seq_len+1, state_size)

        #临时区域----这部分是临时给注意力机制用的
        # 方法一：使用平均池化
        pooled = out.mean(dim=1)  # [batch_size, state_size]
        # 方法二（可选）：只取 [CLS] token
        # pooled = out[:, 0, :]  # [batch_size, state_size]

        # 映射为一个数值
        logits = self.final_fc(pooled).squeeze(-1)  # [batch_size]
        
        return logits.sigmoid()