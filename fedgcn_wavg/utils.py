# import numpy as np
# from scipy import stats
# data = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
# # 使用Box-Cox变换
# normalized_data, lambda_value = stats.boxcox(data)
#
# print("归一化后的数据：", normalized_data)
# print("Box-Cox变换的Lambda值：", lambda_value)


import numpy as np

# 假设您有一个长尾分布的数据集 data
data = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500])

# 对数据进行对数变换
normalized_data = np.log1p(data)  # 使用np.log1p来处理数据，以避免对0值取对数时出错

# 最小值不为0，但数据已归一化
print("归一化后的数据：", normalized_data)


