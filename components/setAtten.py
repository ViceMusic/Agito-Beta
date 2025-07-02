# 

import torch
import torch.nn as nn
import torch.nn.functional as F

"""SetAttention 模块
该模块实现了一个多头注意力机制，结合全局池化向量和, 主要目的是确保平移不变性, 保证每个元素不再存在相互关系
"""
class SetAttention(nn.Module):
    def __init__(self, state_size, num_heads=4, lambda_min=0.7, gamma_max=0.3):
        super().__init__()
        self.state_size = state_size
        self.num_heads = num_heads
        self.lambda_min = lambda_min  # λ下限（保证原始信息）
        self.gamma_max = gamma_max    # γ上限（防止池化主导）
        
        # 可学习参数 (扩展到batch维度)
        self.lambda_params = nn.Parameter(torch.ones(1, 1, state_size) * 0.85)  # 初始值0.85
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 初始值0.1
        
        # 多头注意力模块
        self.mha = nn.MultiheadAttention(
            embed_dim=state_size,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        """ 
        x形状: (batch_size, seq_len, state_size)
        返回: (batch_size, seq_len+1, state_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算全局池化向量 [batch_size, 1, state_size]
        pooled = x.mean(dim=1, keepdim=True)  # 平均池化
        
        # 2. 拼接池化向量到序列末尾 [batch_size, seq_len+1, state_size]
        x_pooled = torch.cat([x, pooled], dim=1)
        
        # 3. 计算约束后的参数（确保数值范围）
        lambda_val = self.lambda_min + (1 - self.lambda_min) * torch.sigmoid(self.lambda_params)
        gamma_val = self.gamma_max * torch.sigmoid(self.gamma)  # [1]
        
        # 4. 信息融合：x_i' = λ * x_i + γ * pooled
        #    注意：不修改最后一个位置（池化向量本身）
        x_fused = x_pooled.clone()  # 创建副本
        x_fused[:, :-1, :] = (
            lambda_val * x_pooled[:, :-1, :] + 
            gamma_val * pooled.expand(-1, seq_len, -1)
        )
        
        # 5. 多头注意力处理 [batch_size, seq_len+1, state_size]
        attn_output, _ = self.mha(x_fused, x_fused, x_fused)
        return attn_output


'''
batch_size, seq_len, state_size, num_heads = 2, 10, 4, 4
model = SetAttention(state_size, num_heads=4)

# 创建随机输入 (batch_size, seq_len, state_size)
x = torch.randn(batch_size, seq_len, state_size)
output = model(x)

print("输入形状:", x.shape)         # [2, 10, 64]
print("池化向量形状:", x)  # [2, 64] 最后一行是池化向量
print("输出形状:", output.shape)    # [2, 11, 64]
print("输出内容:", output)          # 输出融合后的结果
'''