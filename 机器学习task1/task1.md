
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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
```
导入必要的库：pandas用于数据处理，numpy用于数值计算，matplotlib用于绘图,sklearn用于导入机器学习的各种函数，torch用于构建神经网络

```python
def loadData(filepath):
    """
    :param filepath: csv
    :return: list
    """
    data_list = pd.read_csv(filepath)
    median_values = data_list.median()  
    data_list = data_list.fillna(median_values)  
    X = data_list.drop("MEDV", axis=1)
    y = data_list["MEDV"]
    return X, y
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
用中位数替代缺失值
```python
scaler = StandardScaler()# 实例化StandardScaler对象  
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8)
```
对数据进行标准化处理，train_size=0.8表示训练集占80%，测试集占20% 
将数据集分割为训练集和测试集。
```python
# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
```
 **转换训练集特征**：
   ```python
   X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
   ```
   - `X_train` 是一个 NumPy 数组，包含了训练数据的特征。
   - `torch.tensor()` 函数用于创建一个新的 PyTorch 张量。
   - `dtype=torch.float32` 指定了张量的数据类型为 32 位浮点数，这是深度学习模型训练时常用的数据类型。

 **转换训练集标签**：
   ```python
   y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
   ```
   - `y_train` 包含了训练数据的目标值。
   - `.values` 将其转换为 NumPy 的一维数组。
   - 转换后的张量通过 `.view(-1, 1)` 方法被重塑为二维张量，其中 `-1` 表示自动计算该维度的大小（即保持元素总数不变），`1` 表示每个样本的标签是一个单独的列。
 **测试集同上**

```python
# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
**网络层**：
   - `super(Net, self).__init__()`：调用父类 `nn.Module` 的初始化方法，这是PyTorch中定义新模块时的标准做法。
   - `self.fc1 = nn.Linear(input_size, 64)`：定义第一个线性层，它将输入特征从 `input_size` 维映射到 64 维。。
   - `self.fc2 = nn.Linear(64, 32)`：定义第二个线性层，它将前一层的 64 维输出映射到 32 维。
   - `self.fc3 = nn.Linear(32, 1)`：定义第三个线性层，也是输出层，它将 32 维输入映射到单个输出值。

**前向传播方法**：
   - `x = torch.relu(self.fc1(x))`：首先，输入 `x` 通过第一个线性层 `self.fc1`，然后应用ReLU激活函数。ReLU是一个非线性函数，它将所有负值置为零，保留所有正值。
   - `x = torch.relu(self.fc2(x))`：接着，经过ReLU激活的输出通过第二个线性层 `self.fc2`，然后再次应用ReLU激活函数。
   - `x = self.fc3(x)`：最后，经过第二个ReLU激活的输出通过第三个线性层 `self.fc3`，得到最终的输出。
```python
# 实例化模型
model = Net(input_size)
```
实例化模型
```python
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
选择均方误差损失（MSE）作为损失函数
选择 Adam 优化器
`model.parameters()` 是一个方法，它返回模型中所有可训练的参数（通常是权重和偏置）。
```python
# 训练模型
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```
 ```python
num_epochs = 1000
```
训练1000次
```python   
losses = []
```
  记录每个epoch结束时的损失值。

 **训练循环**：
   ```python
   for epoch in range(num_epochs):
       model.train()
       optimizer.zero_grad()
       outputs = model(X_train_tensor)
       loss = criterion(outputs, y_train_tensor)
       loss.backward()
       optimizer.step()
       losses.append(loss.item())
       if (epoch+1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
   ```
   - `model.train()`：将模型设置为训练模式。
   - `optimizer.zero_grad()`：在每次迭代开始时，将优化器中所有参数的梯度清零。因为PyTorch默认会累积梯度。
   - `outputs = model(X_train_tensor)`：将训练数据（已转换为张量）传递给模型，并获得预测输出。
   - `loss = criterion(outputs, y_train_tensor)`：计算预测输出和真实标签之间的损失。
   - `loss.backward()`：执行反向传播，计算损失函数关于模型参数的梯度。
   - `optimizer.step()`：根据计算出的梯度更新模型参数。
   - `losses.append(loss.item())`：将当前损失值添加到损失记录列表中。
   - `if (epoch+1) % 10 == 0`：每10个epoch打印一次当前的epoch数和损失值。
```python
# 评估模型
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()
```

```python
model.eval()
```
模型设置为评估模式。


   ```python
   with torch.no_grad():
   ```
   使用`torch.no_grad()`禁用梯度计算，在评估模式下，我们不需要计算梯度，因为我们不更新模型的权重。


```python
y_pred_tensor = model(X_test_tensor)
```
获取模型的预测输出（`y_pred_tensor`）。`X_test_tensor`为张量格式。


   ```python
   y_pred = y_pred_tensor.numpy().flatten()
   ```
   将张量（`y_pred_tensor`）转换为NumPy数组，然后使用`.flatten()`方法将数组展平为一维数组。
```python
# 计算评估指标
mse = metrics.mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

mae = metrics.mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

r2 = metrics.r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')
```
计算：均方误差（MSE）均方根误差（RMSE）平均绝对误差（MAE）R²分数（R² Score）
```python
price_bins = [-np.inf, 10, 20, np.inf] # 设置边界
price_labels = ['Low', 'Medium', 'High']

price_categories = np.digitize(y_pred, price_bins) # 使用 np.digitize 进行分类
price_categories_labels = [price_labels[i-1] for i in price_categories] # 将分类结果转换为对应的标签
```
房价分类
```python
plt.plot(range(1, num_epochs+1), losses, 'r')  # 使用记录的损失值
plt.title('Error vs. Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
绘制损失函数随迭代次数的变化曲线。

```python
# 图例展示预测值与真实值的变化趋势     
t = np.arange(len(X_test))#创建等差数组
plt.plot(t, y_test.values, 'r-', label='target value')  # 使用 .values 而不是 .numpy()
plt.plot(t, y_pred, 'b-', label='predict value')
plt.legend(loc='upper right')
plt.title('Neural Network Regression', fontsize=18)# 标题和字体大小
plt.grid(linestyle='--') # 加格子
plt.show()
```
绘制预测值与真实值的对比图。

## 实验报告
损失值在随着迭代次数增加不断减少,200次之后变化就不明显了
学习率太大损失值降低更快，但是预测值的准确度会有所下降，太小（会使训练速度太慢
SGD的收敛速度比Adam快一点，但是准确率没Adam那没高