# 处理数据的方法主要写在这里,不过这个方法是测试
def test(i):
    if(i%2==0):
        return 1
    else:
        return 0


def expose_data(filepath):
    i=0
    arr=[]
    with open(filepath, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break  # 文件结束
            seq = f.readline().strip()
            plus = f.readline().strip()
            qual = f.readline().strip()
            arr.append([seq,0])  # 将序列和标签存入列表
            i += 1
    return arr

data_list = expose_data('test.fastq')
print("读取的Fastq文件内容:",data_list)  # 打出来的长度我们姑且可以认为是一样的
    # 能读取, 但是还得是神经网络

'''
处理成这种数据格式就欧克了
['CCTTGGGTTTCAGCTCCATCAGCTCCTTTAAGCACTTCTCTTTACTGGTTATTCTAGTTATACATTCTTCTAAATT', 1]
'''

import csv

with open('test2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_list)
