
## 模型结构

1. **数据加载**：使用`pd.read_csv`函数加载数据集，存储在`data_list`变量中。
2. **数据预处理**：使用Z-score标准化方法对数据进行归一化处理，确保所有特征的均值为0，标准差为1。
3. **数据分割**：使用`splitData`函数将数据集分割为训练集和测试集，比例为8:2。
4. **模型定义**：定义线性回归模型的损失函数、正则化代价函数和梯度下降方法。
5. **模型训练**：使用梯度下降方法训练模型，迭代次数为1000，学习率为0.01，正则化参数为50。
6. **模型评估**：使用测试集评估模型的性能，计算MSE、RMSE和R2值。
7. **可视化**：绘制迭代曲线和预测值与真实值的变化趋势图。

## 代码详解

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
导入必要的库：pandas用于数据处理，numpy用于数值计算，matplotlib用于绘图。

```python
def loadData(filepath):
    data_list = pd.read_csv(filepath)
    # 使用Z-score对数据进行归一化处理
    data_list = (data_list - data_list.mean()) / data_list.std()
    return data_list
```
loadData是从CSV文件加载数据的函数。
```python
data_list = pd.read_csv(filepath)
```
使用 pandas 库的 read_csv 函数读取csv文件的数据，并将其内容存储在一个名为 data_list 中
```python
    median_values = data_list.median()  
    data_list = data_list.fillna(median_values)  
```
用中位数填充缺失值
```python
    data_list = (data_list - data_list.mean()) / data_list.std()
```
将每个数值特征的值减去其均值，然后除以其标准差，得到标准化的数值(**Z-score**)
```python
    return data_list
```
返回标准化后的数据。

```python
def splitData(data_list, ratio):
```
定义一个函数，用于将数据集分割为训练集和测试集。

```python
    train_size = int(len(data_list) * ratio)
```
计算训练集的大小，基于给定的比例ratio。

```python
    random_indices = np.random.permutation(len(data_list))
```
使用 NumPy 的 permutation 函数生成一个随机排列的整数数组，长度等于数据集的行数。

```python
    data_list = data_list.iloc[random_indices]
```
使用随机索引重新排列数据，用`.iloc`生成一个打乱的数据集。

```python
    trainset = data_list[:train_size]
    testset = data_list[train_size:]
```
data_list[:train_size] 表示从 data_list 的开头开始，取前 train_size 行的数据，并将其赋值给 trainset。
data_list[train_size:] 表示从 data_list 的第 train_size 行开始，取所有剩余的行，并将其赋值给 testset。
代码将数据集按照 train_size 分割成两部分，前半部分是训练集，后半部分是测试集。

```python
    X_train = trainset.drop("MEDV", axis=1)
    y_train = trainset["MEDV"]
    X_test = testset.drop("MEDV", axis=1)
    y_test = testset["MEDV"]
```
分离特征（X）和目标变量（y）。"MEDV"是目标变量。axis=1表示删除的是行

```python
    return X_train, X_test, y_train, y_test
```
返回分割后的数据集。

```python
def loss_function(X, y, theta):
```
定义损失函数（均方误差）
```python
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner)/(2*len(X))
```
`np.power`是 NumPy 库中的一个函数，用于计算数组中元素的幂。

基本语法：
```python
numpy.power(x1, x2)
```

其中：
- `x1` 是底数（可以是标量、数组或矩阵）
- `x2` 是指数（可以是标量、数组或矩阵）


除以 2 是为了在后续计算梯度时简化公式

不除以 2 时： ∂J/∂θj = (1/m) * Σ(h(x^(i)) - y^(i)) * x^(i)_j

除以 2 时： ∂J/∂θj = (1/m) * Σ(h(x^(i)) - y^(i)) * x^(i)_j

注意到，除以 2 后，在计算偏导数时，2 和 1/2 相互抵消，得到一个更简洁的形式。。


```python
def regularized_loss(X, y, theta, l):
```
定义L2正则化项的损失函数,防止过拟合


```python
np.power(theta[1:], 2).sum()
```
这部分计算从第二个元素（索引为1）开始的所有模型参数的平方和。这里假设第一个元素是偏置项（不被正则化）。
```python
    reg = (l / (2 * len(X))) * (np.power(theta[1:], 2).sum())
    return loss_function(X, y, theta) + reg
```
2 * len(X) 的出现是为了确保正则化项与损失函数中的平均损失保持一致的缩放比例。

在原损失函数基础上添加L2正则化项。

```python
def gradient_descent(X, y, theta, l, alpha, epoch):
```
定义梯度下降函数。

```python
    cost = np.zeros(epoch)
    m = X.shape[0]
```
`cost`用于储存每次的损失值
`X.shape`用于读取某个维度几行几列（一共多少数据），这里m表示第一维的样本数量
初始化存储每次迭代的损失值的数组，并获取样本数量。

```python
    for i in range(epoch):
        theta = theta - (alpha / m) * (X * theta.T - y).T * X - (alpha * l / m) * theta
        cost[i] = regularized_loss(X, y, theta, l)
```
- `(X * theta.T - y).T * X`：这部分计算的是损失函数关于`theta`的梯度（不考虑正则化项）。首先，`X * theta.T`计算的是模型的预测值，`X * theta.T - y`计算的是预测值与实际值之间的误差。然后，`.T`表示转置操作，`(X * theta.T - y).T`将误差向量转换为行向量，以便与特征矩阵`X`相乘，得到梯度向量。
- `(alpha / m) * (X * theta.T - y).T * X`：这部分是根据梯度下降算法更新`theta`的公式的一部分，乘以`(alpha / m)`是为了平均每个样本对梯度更新的贡献。
- `(alpha * l / m) * theta`：这是L2正则化项对`theta`更新的贡献。
- `theta = theta - (...) - (...)`：最后，将上述两部分（不考虑正则化的梯度更新和正则化项的更新）从`theta`中减去，得到新的`theta`值。这个更新步骤是梯度下降算法的核心。

执行梯度下降，更新theta并记录每次迭代的损失。


```python
    return theta, cost
```
返回最终的theta和损失历史。

```python
if __name__ == '__main__':
```
主程序入口。

```python
    alpha = 0.01
    epoch = 1000
    l = 50
```
设置学习率、迭代次数和正则化参数。

```python
    data_list = loadData('housing.csv')
    X_train, X_test, y_train, y_test = splitData(data_list, 0.8)
```
加载数据并分割为训练集和测试集。

```python
    X_train = np.matrix(X_train.values)
    y_train = np.matrix(y_train.values)
    y_train = y_train.reshape(y_train.shape[1], 1)
    X_test = np.matrix(X_test.values)
    y_test = np.matrix(y_test.values)
    y_test = y_test.reshape(y_test.shape[1], 1)
```
将数据转换为矩阵格式，并重塑为向量。

```python
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)
```
在特征矩阵的开始添加一列1，用于偏置项。

```python
    theta = np.matrix(np.zeros((1, 14)))
```
初始化theta为零矩阵。

```python
    final_theta, cost = gradient_descent(X_train, y_train, theta, l, alpha, epoch)
    print(final_theta)
```
执行梯度下降，并打印最终的theta值。

```python
    y_pred = X_test * final_theta.T
    mse = np.sum(np.power(y_pred - y_test, 2)) / (len(X_test))
    rmse = np.sqrt(mse)
    R2_test = 1 - np.sum(np.power(y_pred - y_test, 2)) / np.sum(np.power(np.mean(y_test) - y_test, 2))
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('R2_test = ', R2_test)
```
使用训练好的模型进行预测，并计算各种评估指标（MSE, RMSE, R2）。
```python
price_bins = [-np.inf, 20, 30, np.inf]  # 设置边界  
    price_labels = ['Low', 'Medium', 'High']  
    y_pred_array = np.array(y_pred).flatten()  # 将预测值转换为NumPy数组以便使用np.digitize  
    price_categories = np.digitize(y_pred_array, price_bins) # 使用np.digitize进行分类  
    price_categories_labels = [price_labels[i-1] for i in price_categories]  # 将分类结果转换为对应的标签 
```
```python
    plt.plot(np.arange(epoch), cost, 'r')
    plt.title('Error vs. Training Epoch')
    plt.xlabel('Cost')
    plt.ylabel('Iterations')
    plt.show()
```
绘制损失函数随迭代次数的变化曲线。

```python
    t = np.arange(len(X_test))
    plt.plot(t, y_test, 'r-', label='target value')
    plt.plot(t, y_pred, 'b-', label='predict value')
    plt.legend(loc='upper right')
    plt.title('Linear Regression', fontsize=18)
    plt.grid(linestyle='--')
    plt.show()
```
绘制预测值与真实值的对比图。

## 实验报告
损失值在随着迭代次数增加不断减少,200次之后变化就不明显了
学习率太大（如0.1）损失值降低更快，但是预测值的准确度会有所下降
目标变量在[-1,1]区间时，预测的准确率较高，对于极端值的预测准确率最低
对于一般值预测准确度高，学习率可以设成逐渐减小的动态值来提高运行速度的同时保证准确率