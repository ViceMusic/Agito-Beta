import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# 其中vocab_dim是嵌入维度，window_size是滑动窗口大小，padding_char是边界填充字符
#         - 输出: 编码后的序列张量 (batch_size, seq_len, vocab_dim)
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