

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from collections import defaultdict

class DNAEmbedding(nn.Module):
    def __init__(self, vocab_dim=4, window_size=1, padding_char='N'):
        """
        DNA序列嵌入层，支持滑动窗口组合映射
        参数:
        - vocab_dim: 嵌入维度 (默认64)
        - window_size: 滑动窗口大小 (奇数，默认1)
        - padding_char: 边界填充字符 (默认'N')
        """
        super().__init__()
        # 确保窗口大小为奇数
        self.window_size = max(1, window_size if window_size % 2 == 1 else window_size - 1)
        self.padding_char = padding_char
        
        # 基础词汇表 (单碱基)
        self.base_vocab = ['A', 'C', 'G', 'T', padding_char]
        
        # 构建k-mer词汇表 (窗口组合)
        self.kmer_vocab = self._generate_kmer_vocab()
        self.vocab_size = len(self.kmer_vocab)
        
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=vocab_dim
        )
        # 初始化嵌入权重
        self._init_embedding_weights()

    def _generate_kmer_vocab(self):
        """
            生成所有可能的k-mer组合词汇表，然后形成一个唯一id的字典
            返回: k-mer词汇表字典 {k-mer: id}
        """
        vocab = defaultdict(int)
        idx = 0
        
        # 单碱基词汇
        for base in self.base_vocab:
            vocab[base] = idx
            idx += 1
        
        # 多碱基组合 (k>1时)
        if self.window_size > 1:
            # 递归生成所有可能的组合
            prev_kmers = self.base_vocab.copy()
            for _ in range(1, self.window_size):
                new_kmers = []
                for kmer in prev_kmers:
                    for base in self.base_vocab:
                        new_kmer = kmer + base
                        vocab[new_kmer] = idx
                        new_kmers.append(new_kmer)
                        idx += 1
                prev_kmers = new_kmers
        return vocab
    # 获取权重
    # 因为词向量的核心就是查表，其实就相当于索引
    def _init_embedding_weights(self):
        """初始化嵌入权重 (相似k-mer具有相近向量)，第一个参数为词汇表大小，第二个参数为嵌入维度，形成一个矩阵"""
        weight = np.zeros((self.vocab_size, self.embedding.embedding_dim))
        
        # 构建k-mer相似性映射
        kmer_similarity = {}
        for kmer in self.kmer_vocab:
            if len(kmer) == 1:
                kmer_similarity[kmer] = [kmer]
            else:
                # 共享相同前缀的k-mer视为相似
                prefix = kmer[:-1]
                if prefix not in kmer_similarity:
                    kmer_similarity[prefix] = []
                kmer_similarity[prefix].append(kmer)

        '''
         kmer_similarity 最后是这种东西
        {
            "A": ["A"],          # 单字符独立组（len=1）
            "T": ["T"],          # 单字符独立组（len=1）
            "A": ["AT", "AA"],   # 前缀"A"组：包含所有以"A"开头的2-mer
            "AT": ["ATG", "ATC"] # 前缀"AT"组：包含所有以"AT"开头的3-mer
        }
        '''
        
        # 为相似k-mer分配相近向量
        for group in kmer_similarity.values():#遍历每个前缀组组合，相似的前缀组合有相同的内容
            # 生成一个基础向量，然后为每个k-mer添加随机扰动
            # 这里的0.05是扰动幅度，可以根据需要调整
            base_vec = np.random.normal(size=self.embedding.embedding_dim)
            for i, kmer in enumerate(group):
                weight[self.kmer_vocab[kmer]] = base_vec + 0.05 * np.random.randn(
                    self.embedding.embedding_dim
                )
        
        # 加载权重
        self.embedding.weight.data.copy_(torch.from_numpy(weight))

    #截取长度并且将其转化为对应的索引
    def _get_kmer_index(self, seq, center_pos):
        """获取以center_pos为中心的k-mer索引"""
        half_window = self.window_size // 2
        start = center_pos - half_window
        end = center_pos + half_window + 1
        
        # 边界填充
        if start < 0:
            prefix = self.padding_char * abs(start)
            start = 0
        else:
            prefix = ""
        
        if end > len(seq):
            suffix = self.padding_char * (end - len(seq))
            end = len(seq)
        else:
            suffix = ""
        
        kmer = prefix + seq[start:end] + suffix
        return self.kmer_vocab.get(kmer, self.kmer_vocab[self.padding_char])#self.kmer_vocab[self.padding_char]的作用是如果kmer不在词汇表中，则返回填充字符的索引

    def forward(self, batch_seqs):
        """
        前向传播
        输入: DNA序列列表, 如 ["ACGT", "TTGCA", ...]
        输出: 嵌入张量 [batch_size, seq_len, vocab_dim]
        """
        # 统一序列长度 (动态填充)
        max_len = max(len(seq) for seq in batch_seqs)
        padded_seqs = [seq + self.padding_char * (max_len - len(seq)) for seq in batch_seqs]
        
        # 为每个序列的每个位置生成k-mer索引
        batch_indices = []
        for seq in padded_seqs:
            seq_indices = []
            for center_pos in range(len(seq)):
                idx = self._get_kmer_index(seq, center_pos)
                seq_indices.append(idx)
            batch_indices.append(seq_indices)
        
        # 转换为张量
        indices_tensor = torch.tensor(batch_indices, dtype=torch.long)
        
        # 嵌入映射
        embeddings = self.embedding(indices_tensor)
        return embeddings

    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.vocab_size
#print("DNAEmbedding ", DNAEmbedding(vocab_dim=1)(["ATAC","AA","ACTG"]))


# 基于双问状态方程和门控机制的空间信息编码器

class Encoder(nn.Module):
    def __init__(self, input_dim, state_dim, final_dim=1):
        """
        基于双问状态方程和门控机制的空间信息编码器
        
        参数:
        - input_dim: 输入维度 (u_t的维度)，这里默认每个时间步骤都是张量而不是单纯的数字
        - state_dim: 状态维度 (x_t的维度，推荐与input_dim相同)，默认是一样的
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        # 状态转移参数 (公式中的A, B)
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        self.B = nn.Linear(input_dim, state_dim, bias=False)

        # 反向状态转移参数 (公式中的A, B)
        self.Af = nn.Linear(state_dim, state_dim, bias=False)
        self.Bf = nn.Linear(input_dim, state_dim, bias=False)
        
        # 门控参数 (公式中的w1, w2, b)
        self.w1 = nn.Linear(state_dim, 1, bias=False)
        self.w2 = nn.Linear(state_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

        # 最后的分类器
        self.classifier = nn.Linear(state_dim, final_dim, bias=False)
        
    def forward(self, u):
        """
        前向传播计算
        
        参数:
        - u: 输入序列张量 (batch_size, seq_len, input_dim)
        
        返回:
        - y: 输出序列 (batch_size, seq_len, state_dim)
        - gamma: 开度系数序列 (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = u.shape
        
        # 0. 初始化状态容器
        x_forward = torch.zeros(batch_size, seq_len, self.state_dim, device=u.device)
        x_backward = torch.zeros(batch_size, seq_len, self.state_dim, device=u.device)
        gamma = torch.zeros(batch_size, seq_len, 1, device=u.device)
        y = torch.zeros(batch_size, seq_len, self.state_dim, device=u.device)

        x_forward_new = x_forward.clone()
        x_backward_new = x_backward.clone()
        
        # 1. 正向状态计算 (从左到右)
        # x_t = A*x_{t-1} + B*u_t, x_0 = u_0
        for t in range(seq_len):
            if t == 0:
                x_forward_new[:, t] = u[:, 0]  # x_0 = u_0
            else:
                x_forward_new[:, t] = self.A(x_forward[:, t-1]) + self.B(u[:, t])
        
        # 2. 反向状态计算 (从右到左)
        # x_t^ = A*x_{t+1}^ + B*u_{t+1}, x_{L-1}^ = u_{L-1}

        for t in range(seq_len-1, -1, -1):
            if t == seq_len - 1:
                x_backward_new[:, t] = u[:, -1]  # x_{L-1}^ = u_{L-1}
            else:
                x_backward_new[:, t] = self.Af(x_backward[:, t+1]) + self.Bf(u[:, t+1])
        
        x_forward = x_forward_new
        x_backward = x_backward_new

        # 3. 计算开度系数gamma_t
        gamma = torch.sigmoid(
            self.w1(x_forward) + 
            self.w2(x_backward) + 
            self.b
        )
        
        # 4. 计算最终输出y_t
        y = gamma * (x_forward + x_backward) + (1 - gamma) * u

        # 5. 分类器输出(这里为了适应agito， 进行了最粗糙的reshape操作)
        final_output = nn.functional.sigmoid(self.classifier(y)).reshape(batch_size, seq_len)
        
        '''
        y为原始预测结果，输出为[batch_size, seq_len, state_dim],对每个时间步的结果都保留特征
        gamma为开度系数，输出为[batch_size, seq_len, 1],表示每个时间步的开度系数
        final_output为最终的输出结果，输出为[batch_size, seq_len]，并且已经经过sigmoid激活函数
        '''
        return y, gamma, final_output



'''
在这个文件中存在命名规范
mask=0	非掩盖区域	_mi	mask-ignore    代表无需进行掩盖的区域，可信度较高
mask=1	掩盖区域	_mr	mask-required 需要进行掩盖的区域，可信度较低
'''

# 这部分是用来计算整个似然结果的
class Agito(nn.Module):
    #===============启动参数部分============================
    def __init__(self, pglobal_mi, pglobal_mr, alpha_mi=1.0, beta_mi=1.0, alpha_mr=1.0, beta_mr=1.0, window_size=1):
        super(Agito, self).__init__()
        # embeding层
        self.embedding = DNAEmbedding(vocab_dim=4, window_size=window_size, padding_char='N')  # 假设嵌入维度为15，窗口大小为3
        # 状态编码器
        self.encoder = Encoder(input_dim=4, state_dim=4, final_dim=1)  # 假设输入维度和状态维度都是15

        # 约束参数范围（避免梯度NaN）
        self.alpha_mi = nn.Parameter(torch.tensor(alpha_mi).clamp(min=0.1), requires_grad=False)
        self.beta_mi = nn.Parameter(torch.tensor(beta_mi).clamp(min=0.1), requires_grad=False)
        self.alpha_mr = nn.Parameter(torch.tensor(alpha_mr).clamp(min=0.1), requires_grad=False)
        self.beta_mr = nn.Parameter(torch.tensor(beta_mr).clamp(min=0.1), requires_grad=False)
        
        # 预设概率权重（需sigmoid约束到(0,1)）
        self.P_mi = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.P_mr = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # 更新频率，假设一开始更新频率都是0
        self.update_mi = 0
        self.update_mr = 0

        # 计算记忆参数,一开始的记忆参数均为1
        self.lambda_mi=nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.lambda_mr=nn.Parameter(torch.tensor(1.), requires_grad=False)

        # 计算全局的无掩盖概率和掩盖概率
        self.pglobal_mi = pglobal_mi  # 全
        self.pglobal_mr = pglobal_mr  # 全局掩盖概率


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
    def binary_entropy(self, probs, epsilon=1e-3, reduction='mean'):
        probs = torch.clamp(probs, epsilon, 1 - epsilon)  # 保证数值稳定
        print("计算置信度的时候检查",probs)
        entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)

        if reduction == 'mean':
            return entropy.mean()
        elif reduction == 'sum':
            return entropy.sum()
        else:
            return entropy  # 每个样本的不确定性

    # 手动计算分布
    def beta_log_prob(self,x, alpha, beta, eps=1e-7):
        x = torch.clamp(x, eps, 1 - eps)
        return (
            (alpha - 1) * torch.log(x)
            + (beta - 1) * torch.log(1 - x)
            - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
        )


    #==========计算核心部分========================

    # 根据当前批次的正确和错误数值，计算并且迭代参数
    # S这里设定为“需要掩盖”的样本数目， F为“无需掩盖的样本”
    '''
    参数格式：
    num_mask_ignore: int, 无需掩盖的样本数目
    num_mask_required: int, 需要掩盖的样本数目
    model_output: torch.Tensor, 模型输出的预测结果
    labels: torch.Tensor, 真实标签 --------------------------》这两个标签都需要[batch_size, feature_size]的格式
    pglobal_mi: float, 全局的无掩盖概率
    pglobal_mr: float, 全局的掩盖概率-------------------------》这两个是全局计算的指标
    '''
    def update_ab(self, num_mask_ignore, num_mask_required, model_output, labels,lam=0.1,sen_w=0.7):\
        
        '''
        计算散度：使用pglobal和num_mask_ignore, num_mask_required计算KL散度
        计算欧氏距离：使用model_output和labels计算欧氏距离
        计算二元熵：使用model_output计算二元熵
        更新参数：使用上述计算结果更新alpha_mi, beta_mi, alpha_mr, beta_mr
        '''
        # 先更新速率
        self.update_mi = self.update_mi  +  num_mask_ignore
        self.update_mr = self.update_mr  + num_mask_required
        # 先对记忆参数进行更新， 这个计算lambda的很快就爆炸了--------------------------
        # 设置一个最低的记忆参数
        self.lambda_mi.data = self.lambda_mi.data * max(min(abs(num_mask_required/(num_mask_ignore+1)),1),0.1)
        self.lambda_mr.data = self.lambda_mr.data * max(min(abs(num_mask_ignore/(num_mask_required+1)),1),0.1)
        # 然后在计算全局和普通的p值
        P_mi= num_mask_ignore / (num_mask_ignore + num_mask_required)
        P_mr= num_mask_required / (num_mask_ignore + num_mask_required)
        # 再分别计算omega
        pbatch =  torch.tensor([P_mi, P_mr])
        pglobal =  torch.tensor([self.pglobal_mi, self.pglobal_mr])

        omega=((sen_w * self.binary_entropy(model_output))+math.exp(-1*lam*self.kl_divergence(pbatch,pglobal)))/(1+self.euclidean_distance(model_output,labels))

        print("置信度权重",(sen_w * self.binary_entropy(model_output)))
        print("置信度权重的问题？")
        print("分布权重",math.exp(-1*lam*self.kl_divergence(pbatch,pglobal)))
        print("距离权重",(1+self.euclidean_distance(model_output,labels)))
        print("omega",omega)
        print("更新标签数值：",num_mask_ignore, num_mask_required)
        print("记忆参数：",self.lambda_mi, self.lambda_mr)

        self.alpha_mi.data = self.alpha_mi.data * self.lambda_mi + omega * num_mask_ignore
        self.beta_mi.data  = self.beta_mi.data  * self.lambda_mi + omega * num_mask_required

        self.alpha_mr.data = self.alpha_mr.data * self.lambda_mr + omega * num_mask_required
        self.beta_mr.data  = self.beta_mr.data  * self.lambda_mr + omega * num_mask_ignore

        '''
        print("更新信息A_mi",self.alpha_mi)
        print("更新信息A_mr",self.alpha_mr)
        print("更新成功")
        '''

        
    # 前向传播函数，计算似然概率
    def forward(self, x):
        vocal_features= self.embedding(x)  # [batch_size, seq_len, vocab_dim]
        encoded_features, _, x = self.encoder(vocal_features)  # [batch_size, seq_len] 其中x为final_output
        print("输出x",x)
         # 1. 计算对数概率（数值稳定版）
        log_beta_mi = self.beta_log_prob(x, self.alpha_mi, self.beta_mi).sum(dim=1)
        log_beta_mr = self.beta_log_prob(x, self.alpha_mr, self.beta_mr).sum(dim=1)

        
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

# 整个模型是可以这样子跑起来的，
input=['ACGT', 'TTGCA', 'GCGTAC', 'NNNNN']  # 假设输入是一个包含DNA序列的列表


model = Agito(pglobal_mi=0.6, pglobal_mr=0.4, alpha_mi=1.0, beta_mi=1.0, alpha_mr=1.0, beta_mr=1.0)
print(model(input))





#=========测试区域===============

'''
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

# 测试更新参数
num_mask_ignore = 5
num_mask_required = 3
pglobal_mi = 0.6
pglobal_mr = 0.4    
model.update_ab(num_mask_ignore, num_mask_required, input, input+1, pglobal_mi, pglobal_mr)
print("Updated parameters:")
'''