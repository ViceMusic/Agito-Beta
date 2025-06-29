'''
此文件为主程序入口文件，负责启动整个应用。
虽然叫入口文件，不过实际是一个训练脚本。
它会加载模型，准备数据，进行训练，并绘制损失曲线图。
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.Agito_Ground import Agito_Ground  # 假设agito.py在同一目录下
from components.Embedding import DNAEmbedding  # 假设DNAEmbedding在同一目录下
from components.Extract import Encoder  # 假设Encoder在同一目录下
from tools.Dataload import get_batch_data # 引入获取数据的方法
from tools.graph import plot_list_as_line_chart  # 假设graph.py在tools目录下



model = Agito_Ground()

dataloader = get_batch_data("asset/test.csv", batch_size=10, shuffle=True)  # 假设数据在data目录下
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

arr=[]

for epoch in range(10):  # 假设训练10个epoch
    for seq_batch, label_batch in dataloader:
        result= model(list(seq_batch))  # 假设模型可以直接处理字符串序列
        loss = criterion(result, label_batch.float())     # batch_y shape: [batch]
        optimizer.zero_grad()
        # 二重优化
        loss.backward()
        optimizer.step()
    print(f" Loss: {loss.item():.4f}")
    arr.append(loss.item())  # 保存每个epoch的损失值
    
print("训练完成！",model(["AGCTAGC"]))
#plot_list_as_line_chart(arr, title="Training Loss Curve", xlabel="Epoch", ylabel="Loss")  # 绘制损失曲线图