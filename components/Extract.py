# 此文件夹主要是提取特征用的东西, 目前的想法包括以下几种
'''
1. attention
2. cnn
3. LSTM以及衍生机制
4. Hyena

上面都先算了吧，改用隐式卷积的方式提取信息
'''

# Hyena算子
'''
事先声明， 不是我不想用hyena-dna， 而是这东西实在是太复杂了
手动实现了。
需要实现的点包括: 动态卷积核生成， FFT快速傅里叶加速， 门控残差链接
'''


# 先直接使用卷积进行读取
'''
无论是CHIP和ATAC，都包含两种类型的文件, 首先是这一段基因的序列信息，
然后是该基因的某些信号，比如signal，可以用作引导信号的是比如在这一区域上的peak值
peak数值需要进行具体分析， 暂定规划为在某段基因上的富集程度，包括信号强度，峰值位置

涉及到BAW之类的技能点

不过有wj2背书， 我们就只对序列进行处理
'''


# 最后一次进行更新
'''
使用双向时空建模处理这个问题，目前默认是把数据的输出锁死为， 其中state_dim是状态维度
输出数据包括多种形式
[batch_size, seq_len, state_dim]
以及
[batch_size, seq_len] 强行进行了一个分类器和reshape操作，也因此数据被固定在1的维度上， 不建议动
'''



# 基于双问状态方程和门控机制的空间信息编码器
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, state_dim,final_dim=1):
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

        # 5. 分类器输出
        final_output = nn.functional.sigmoid(self.classifier(y)).reshape(batch_size, seq_len)
        
        '''
        y为原始预测结果，输出为[batch_size, seq_len, state_dim],对每个时间步的结果都保留特征
        gamma为开度系数，输出为[batch_size, seq_len, 1],表示每个时间步的开度系数
        final_output为最终的输出结果，输出为[batch_size, seq_len]，并且已经经过sigmoid激活函数
        '''
        return y, gamma, final_output


















'''
# 测试Encoder模块,因为初始状态的要求，我们让input_dim和state_dim相同
e= Encoder(input_dim=4, state_dim=4)

input=torch.randn(10, 15, 4)  # 输入特征数值是任意的
input= torch.sigmoid(input)  # 映射到(0,1)数据的输入特征必须是0.1之间

y, gamma,final_out = e(input)  # 添加最后一个维度以匹配输入维度
print("y shape:", y.shape)  # 应该是 (batch_size, seq_len, state_dim)
print("Gamma shape:", gamma.shape)  # 应该是 (batch_size, seq_len, 1)
print("Output:", final_out)

'''