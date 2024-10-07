import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 载入数据集
def loadData(filepath):
    """
    :param filepath: csv
    :return: X, y
    """
    data_list = pd.read_csv(filepath)
    median_values = data_list.median()  
    data_list = data_list.fillna(median_values)  
    # 使用Z-score对数据进行归一化处理
    data_list = (data_list - data_list.mean()) / data_list.std()
    return data_list


# 划分训练集与测试集
def splitData(data_list, ratio):
    train_size = int(len(data_list) * ratio)
    # 生成一个随机排列的整数数组
    random_indices = np.random.permutation(len(data_list))
    # 使用随机排列的索引列表重新设定 DataFrame 的行顺序
    data_list = data_list.iloc[random_indices]
    trainset = data_list[:train_size]
    testset = data_list[train_size:]
    X_train = trainset.drop("MEDV", axis=1)#训练集的特征
    y_train = trainset["MEDV"]#训练集的目标变量
    X_test = testset.drop("MEDV", axis=1)#测试集的特征
    y_test = testset["MEDV"]#测试集的目标变量
    return X_train, X_test, y_train, y_test


# 定义损失函数
def loss_function(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner)/(2*len(X))


# 定义正则化代价函数,防止过拟合
def regularized_loss(X, y, theta, l):
    reg = (l / (2 * len(X))) * (np.power(theta[1:], 2).sum())
    return loss_function(X, y, theta) + reg


# 定义梯度下降方法
def gradient_descent(X, y, theta, l, alpha, epoch):
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m
    for i in range(epoch):
        # 利用向量化一步求解
        theta = theta - (alpha / m) * (X * theta.T - y).T * X - (alpha * l / m) * theta# 添加了正则项
        cost[i] = regularized_loss(X, y, theta, l)  # 记录每次迭代后的代价函数值
    return theta, cost


if __name__ == '__main__':
    alpha = 0.01  # 学习率
    epoch = 1000  # 迭代次数
    l = 50  # 正则化参数
    data_list = loadData('housing.csv')
    X_train, X_test, y_train, y_test = splitData(data_list, 0.8)
    # 添加偏置列，同时初始化theta矩阵
    X_train = np.matrix(X_train.values)
    y_train = np.matrix(y_train.values)
    y_train = y_train.reshape(y_train.shape[1], 1)
    X_test = np.matrix(X_test.values)
    y_test = np.matrix(y_test.values)
    y_test = y_test.reshape(y_test.shape[1], 1)
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)
    theta = np.matrix(np.zeros((1, 14)))  # x的第二维维度为14，所以初始化theta为（1,14）
    final_theta, cost = gradient_descent(X_train, y_train, theta, l, alpha, epoch)
    print(final_theta)

    # 模型评估
    y_pred = X_test * final_theta.T
    mse = np.sum(np.power(y_pred - y_test, 2)) / (len(X_test))#均方误差
    rmse = np.sqrt(mse)#均方根误差
    R2_test = 1 - np.sum(np.power(y_pred - y_test, 2)) / np.sum(np.power(np.mean(y_test) - y_test, 2))#决定系数
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('R2_test = ', R2_test)
    
    # 房价分类
    price_bins = [-np.inf, 1, 2, np.inf]  # 设置边界  
    price_labels = ['Low', 'Medium', 'High']  
    y_pred_array = np.array(y_pred).flatten()  # 将预测值转换为NumPy数组以便使用np.digitize  
    price_categories = np.digitize(y_pred_array, price_bins) # 使用np.digitize进行分类  
    price_categories_labels = [price_labels[i-1] for i in price_categories]  # 将分类结果转换为对应的标签 # np.digitize返回的索引从1开始，需要减1对应到labels 
    
    # 绘制迭代曲线
    plt.plot(np.arange(epoch), cost, 'r')#横坐标次数#纵坐标差值#红色曲线
    plt.title('Error vs. Training Epoch')#标题
    plt.xlabel('Cost')#横坐标
    plt.ylabel('Iterations')#纵坐标
    plt.show()

    # 图例展示预测值与真实值的变化趋势
    t = np.arange(len(X_test))  # 创建等差数组
    plt.plot(t, y_test, 'r-', label='target value')#红色tagert value
    plt.plot(t, y_pred, 'b-', label='predict value')#蓝色predict value
    plt.legend(loc='upper right')#右上角图例
    plt.title('Linear Regression', fontsize=18)#标题和字体大小
    plt.grid(linestyle='--')#加格子
    plt.show()