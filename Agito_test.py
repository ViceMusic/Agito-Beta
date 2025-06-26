

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np

'''
在这个文件中存在命名规范
mask=0	非掩盖区域	_mi	mask-ignore    代表无需进行掩盖的区域，可信度较高
mask=1	掩盖区域	_mr	mask-required 需要进行掩盖的区域，可信度较低
'''

# 这部分是用来计算整个似然结果的
class Agito(nn.Module):
    def __init__(self, alpha_mi=1.0, beta_mi=1.0, alpha_mr=1.0, beta_mr=1.0):
        super(Agito, self).__init__()
        # 约束参数范围（避免梯度NaN）
        self.alpha_mi = nn.Parameter(torch.tensor(alpha_mi).clamp(min=0.1))
        self.beta_mi = nn.Parameter(torch.tensor(beta_mi).clamp(min=0.1))
        self.alpha_mr = nn.Parameter(torch.tensor(alpha_mr).clamp(min=0.1))
        self.beta_mr = nn.Parameter(torch.tensor(beta_mr).clamp(min=0.1))
        
        # 预设概率权重（需sigmoid约束到(0,1)）
        self.P_mi = nn.Parameter(torch.tensor(0.5))
        self.P_mr = nn.Parameter(torch.tensor(0.5))

    def test(self):
        print("Agito model is ready for testing.")
        print("Alpha_mi:", self.alpha_mi.item())
        print("Beta_mi:", self.beta_mi.item())  
        return "This is a test message from Agito model."

    #==============tools========================
    # 计算KL散度的工具函数
    # 计算欧式距离的工具函数

    #KL 散度计算,越大表示源分布和目标分布越不相似
    def kl_divergence(self,pbatch, pglobal, epsilon=1e-12, log_base='e'):
        """
        计算 KL 散度 KL(pbatch || pglobal) 的 PyTorch 实现

        参数:
        - pbatch: torch.Tensor，源概率分布（P）
        - pglobal: torch.Tensor，目标概率分布（Q）
        - epsilon: float，为了避免 log(0) 和除以0 引入的小常数
        - log_base: str，'e' 表示自然对数，'2' 表示以 2 为底的对数

        返回:
        - KL 散度（标量 float 或 torch.Tensor）
        """
        pbatch = torch.clamp(pbatch, min=epsilon, max=1.0)
        pglobal = torch.clamp(pglobal, min=epsilon, max=1.0)

        if log_base == '2':
            log_fn = lambda x: torch.log2(x)
        else:
            log_fn = lambda x: torch.log(x)

        kl = torch.sum(pbatch * log_fn(pbatch / pglobal))
        return kl
    # 计算欧氏距离的工具函数,越小表示两个向量越相似，极限为0，因此需要在分母+1
    def euclidean_distance(self,tensor1, tensor2):
        return torch.norm(tensor1 - tensor2, p=2)
    # 计算不确定性(熵)，结果越可靠熵越小，e^-entropy越大,越接近1
    def binary_entropy(self,probs, epsilon=1e-12, reduction='mean'):
        probs = torch.clamp(probs, epsilon, 1 - epsilon)  # 保证数值稳定
        entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)

        if reduction == 'mean':
            return entropy.mean()
        elif reduction == 'sum':
            return entropy.sum()
        else:
            return entropy  # 每个样本的不确定性



    #==========计算核心部分========================

    # 根据当前批次的正确和错误数值，计算并且迭代参数
    # S这里设定为“需要掩盖”的样本数目， F为“无需掩盖的样本”
    def update_ab(self, num_mask_ignore, num_mask_required):
        lambda_mi=0.9
        lambda_mr=0.9
        omega_mi=0.1
        omega_mr=0.1

        self.alpha_mi.data = self.alpha_mi.data * lambda_mi + omega_mi * num_mask_ignore
        self.beta_mi.data  = self.beta_mi.data  * lambda_mi + omega_mi * num_mask_required

        self.alpha_mr.data = self.alpha_mr.data * lambda_mr + omega_mr * num_mask_required
        self.beta_mr.data  = self.beta_mr.data  * lambda_mr + omega_mr * num_mask_ignore

        
    # 前向传播函数，计算似然概率
    def forward(self, x):
         # 1. 计算对数概率（数值稳定版）
        log_beta_mi = dist.Beta(self.alpha_mi, self.beta_mi).log_prob(x).sum(dim=1)  # [batch_size]
        log_beta_mr = dist.Beta(self.alpha_mr, self.beta_mr).log_prob(x).sum(dim=1)
        
        # 2. 加权求和（使用logsumexp防溢出）
        log_numerator = log_beta_mr + torch.sigmoid(self.P_mr).log()
        log_denominator = torch.logsumexp(
            torch.stack([
                log_beta_mr + torch.sigmoid(self.P_mr).log(),
                log_beta_mi + torch.sigmoid(self.P_mi).log()
            ]), dim=0)
        
        # 3. 最终概率（反向传播友好）
        return (log_numerator - log_denominator).exp()  # [batch_size]

# 暂时假设输入是一个一维向量，长度为13
# 善, 还真能跑起来啊, 就是输入要求是0,1之间的浮点数, 这个到时候需要在神经网络上限制一下

input=torch.randn(10, 15)  # 输入特征数值是任意的
input= torch.sigmoid(input)  # 映射到(0,1)数据的输入特征必须是0.1之间


model = Agito()
print(model(input))

print("Model parameters:",model.test())



#这些权重计算都只是最基础的部分，什么都没加上，包括控制参数


# 测试KL散度计算
pbatch =  torch.tensor([0.8, 0.2])
pglobal =  torch.tensor([0.2, 0.8])
kl_result = model.kl_divergence(pbatch, pglobal, log_base='e')
print("KL Divergence:", kl_result)
# 测试欧氏距离计算
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([1.0, 2.0, 3.0])
euclidean_result = model.euclidean_distance(tensor1, tensor2)
print("Euclidean Distance:", euclidean_result)
# 测试二元熵计算，熵要加上一个参数控制
probs = torch.tensor([0.8, 0.21])
entropy_result = model.binary_entropy(probs, reduction='mean')
print("Binary Entropy:", 0.7*entropy_result)