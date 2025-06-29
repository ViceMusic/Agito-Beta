# 绘制折线图的方法
def plot_list_as_line_chart(data, title="Line Chart", xlabel="Index", ylabel="Value"):
    import matplotlib.pyplot as plt  # 延迟导入，避免 DLL 冲突

    x = list(range(len(data)))  # X轴为索引
    y = data                    # Y轴为数据

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Data')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ 先保存再展示，避免 show() 清空图形
    plt.savefig(f"{title}.svg", format='svg')  # 避免触发图形系统崩溃
    plt.show()
    plt.close()  # 关闭图形，释放内存