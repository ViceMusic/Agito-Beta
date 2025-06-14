


'''
对于fastq文件来说,每四行代表一个序列的相关信息:
1. 第一行是序列的标识符，以'@'开头
2. 第二行是实际的序列
3. 第三行是一个加号'+'，有时后面会跟着序列标识符
4. 第四行是质量分数，表示序列中每个碱基的质量

@NB501804:1305:HJFKMBGXL:4:11401:10000:6839 1:N:0:TAATGAAC
ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT   //长度为76
+
AAAAAEAEAEEE6EEEEEEEEEEEAEEEEEEA/EEEEEEEAEAEEEEEAEEE6EEEEEEEEEEEEEEEEEAE/E/E

第一行是一些信息, 不知道该怎么用:
测序仪编号,本次测序序号, 这条所在的flowcell编号,流动槽编号, title, x坐标, y坐标, 第几次读, 是否通过质量过滤, 保留字段, index序列


第四行为质量分数, 也不知道该怎么用, 应该是用ACSII表示分数, 转化成char的形式了(太抽象了, 恁用数字不管吗)


'''


# 测试读取Fastq文件
def read_fastq(filename):
    arr=[]
    with open(filename, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break  # 文件结束
            seq = f.readline().strip()
            plus = f.readline().strip()
            qual = f.readline().strip()
            arr.append([header, seq, qual])
    return arr

# 示例使用
arr=read_fastq('test.fastq')

print("读取的Fastq文件内容:") # 打出来的长度我们姑且可以认为是一样的
for item in arr[:5]:  # 只打印前5个序列
    print(item)
