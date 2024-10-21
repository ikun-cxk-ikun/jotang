import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# 载入数据集
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

X, y = loadData('housing.csv')

# 数据预处理
scaler = StandardScaler()# 实例化StandardScaler对象    
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

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

# 实例化模型
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# 评估模型
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()

# 计算评估指标
mse = metrics.mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

mae = metrics.mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

r2 = metrics.r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# 房价分类
price_bins = [-np.inf, 20, 30, np.inf] # 设置边界
price_labels = ['Low', 'Medium', 'High']

price_categories = np.digitize(y_pred, price_bins) # 使用 np.digitize 进行分类
price_categories_labels = [price_labels[i-1] for i in price_categories] # 将分类结果转换为对应的标签

# 绘制迭代曲线
plt.plot(range(1, num_epochs+1), losses, 'r')  # 使用记录的损失值
plt.title('Error vs. Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 图例展示预测值与真实值的变化趋势     
t = np.arange(len(X_test))#创建等差数组
plt.plot(t, y_test.values, 'r-', label='target value')  # 使用 .values 而不是 .numpy()
plt.plot(t, y_pred, 'b-', label='predict value')
plt.legend(loc='upper right')
plt.title('Neural Network Regression', fontsize=18)# 标题和字体大小
plt.grid(linestyle='--') # 加格子
plt.show()