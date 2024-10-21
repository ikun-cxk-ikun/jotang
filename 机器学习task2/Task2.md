# Task2

### **导入必要的库**：
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```
   - `import torch`：导入PyTorch库，这是进行深度学习计算的基础库。
   - `import torch.nn as nn`：从PyTorch中导入神经网络模块（`nn`），这个模块包含了构建神经网络所需的所有组件，如层（layers）、激活函数（activation functions）等。
   - `import torch.optim as optim`：从PyTorch中导入优化算法模块（`optim`），这个模块包含了多种优化算法，如SGD（随机梯度下降）、Adam等，用于训练神经网络时更新模型参数。
   - `import torchvision`：导入`torchvision`库，这是一个处理图像数据的库，它提供了预训练的模型、数据加载和图像转换工具。
   - `import torchvision.transforms as transforms`：从`torchvision`中导入图像转换模块（`transforms`），这个模块提供了多种图像预处理操作，如裁剪、缩放、归一化等。
   - `from torch.utils.data import DataLoader`：从PyTorch的`utils.data`模块中导入`DataLoader`类，这个类用于包装数据集，使其可以迭代地提供给模型进行训练或评估。

### 指定运行位置 
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### 数据预处理
```python
transform = transforms.Compose([
    transforms.ToTensor()，
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
```
对数据进行格式转换和归一化
### 载入数据
```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
```
将数据分为训练集和测试集
### 定义卷积神经网络
#### 定义卷积层，池化层，全连接层
```python
self.conv1 = nn.Conv2d(3, 32, 3, 1)  
self.conv2 = nn.Conv2d(32, 64, 3, 1)   
self.pool = nn.MaxPool2d(2, 2)   
self.fc1 = nn.Linear(64*6*6, 128)      
self.fc2 = nn.Linear(128, 10) 
```

```python
def forward(self,x):
        # TODO:这里补全你的前向传播 
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6) # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
```
#### 前向传播过程

1. 输入图像首先通过第一个卷积层（`conv1`），接着应用ReLU激活函数增加非线性，然后通过池化层降低维度。
2. 经过第一个卷积块（卷积+ReLU+池化）处理后的特征图进入第二个卷积层（`conv2`），同样应用ReLU激活函数和池化层。
3. 经过第二个卷积块处理后的特征图被展平为一维向量，并输入到第一个全连接层（`fc1`），应用ReLU激活函数。
4. 最后，特征向量通过第二个全连接层（`fc2`）得到最终的分类结果。

### 实例化模型

1. **模型迁移到设备**：

```python
model = Network().to(device)
```

- `Network()` 是一个神经网络模型的实例化，这里假设 `Network` 是一个自定义的或来自某个库的神经网络类。
- `.to(device)` 方法是将模型迁移到指定的设备上，这个设备可以是CPU或GPU。`device` 通常是一个字符串，如 `'cpu'` 或 `'cuda'`（对于NVIDIA GPU）。这个步骤是为了确保模型能够在具有适当计算能力的硬件上运行。

2. **交叉熵损失函数**：

```python
criterion = nn.CrossEntropyLoss()
```

 `nn.CrossEntropyLoss()` 是PyTorch中的交叉熵损失函数，用于多分类任务。


1. **随机梯度下降（SGD）优化器**：

```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

- `optim.SGD` 是PyTorch的优化器之一，用于通过随机梯度下降算法更新模型的权重。
- `model.parameters()` 返回一个包含模型所有可训练参数的迭代器。这些参数是优化器需要更新的对象。
- `lr=learning_rate` 设置了学习率。

### 训练

```python
    model.train()
```
将模型设置为“训练模式”。在PyTorch中，一些模型组件（如`Dropout`和`BatchNorm`）在训练和评估阶段的行为是不同的。通过调用`.train()`，确保这些组件在训练期间按预期工作。

这行代码开始了一个循环，`num_epochs`是我们要训练模型的总轮数。每一轮都称为一个“epoch”。

#### 初始化变量

```python
        running_loss = 0.0
        correct = 0
        total = 0
```

在每个epoch开始时，初始化三个变量为0。

#### 数据加载循环

```python
        for i, data in enumerate(trainloader, 0):
```

这行代码遍历训练数据加载器`trainloader`，按批次（batch）提供数据和标签。`enumerate`函数还提供了当前批次的索引`i`。

#### 数据准备

```python
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
```

从`data`中提取输入和标签，并将它们发送到之前指定的设备

#### 优化器梯度清零

```python
            optimizer.zero_grad()
```

在开始计算新批次的梯度之前，需要清零之前批次累积的梯度。

#### 前向传播

```python
            outputs = model(inputs)
```

将输入数据传递给模型，并获取预测输出。

#### 计算损失

```python
            loss = criterion(outputs, labels)
```

使用损失函数`criterion`计算损失。

#### 反向传播和优化

```python
            loss.backward()
            optimizer.step()
```

调用`.backward()`来计算损失对所有模型参数的梯度，然后`optimizer.step()`根据这些梯度更新模型的参数。

#### 跟踪损失和准确率

```python
            running_loss += loss.item()
```

将当前批次的损失加到`running_loss`上。

```python
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
```

使用`torch.max`找到每个样本预测概率最高的类别作为预测结果。然后，更新总样本数和正确预测的数量。

#### 输出训练进度

```python
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
```

计算整个epoch的平均损失和准确率，并打印出来。
### 测试

```python
model.eval()
```
将模型设置为评估模式（evaluation mode）。
在评估模式下，模型不会更新参数，且会关闭 dropout 和 batch normalization 等正则化技术。

```python
correct = 0
total = 0
```
初始化两个变量：`correct` 用于记录模型正确预测的样本数量，`total` 用于记录测试集的总样本数量。

```python
with torch.no_grad():
```
使用torch.no_grad()上下文管理器来关闭PyTorch的自动求导功能（autograd）。
在测试阶段，不需要计算梯度，所以关闭自动求导功能可以节省计算资源。

```python
for data in testloader:
```
遍历测试集的数据加载器（testloader）。

```python
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
```

从`data`中提取输入和标签，并将它们发送到之前指定的设备

```pyhon
outputs = model(images)
```
将图像数据输入模型，获取输出结果。

```python
_, predicted = torch.max(outputs.data, 1)
```
获取输出结果中每个样本的预测类别。
```python
torch.max()
``` 
函数返回两个值：最大值和最大值的索引。我们只需要索引，所以使用 `_` 忽略最大值。

```python
total += labels.size(0)
```
更新总样本数量。

```python
correct += (predicted == labels).sum().item()
```
更新正确预测的样本数量。

  `.item()` 会将结果转换为 Python 的整数类型。

```python
accuracy = 100 * correct / total
```
计算模型在测试集上的准确率。

```python
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
```
打印模型在测试集上的准确率。
# Task 2 extra
### 绘制图像
```python
import matplotlib.pyplot as plt  
```
导入`matplotlib.pyplot`用于绘图
```python
train_losses = []  
train_accuracies = []  
```
创建列表存储每次的损失和准确率
```python
train_losses.append(running_loss / len(trainloader))  
train_accuracies.append(accuracy)  
```
将每次的损失和准确率追加到对应的列表中
```python
    plt.figure(figsize=(12, 5))  
    plt.subplot(1, 2, 1)  
    plt.plot(train_losses, label='Training Loss')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.title('Training Loss per Epoch')  
    plt.legend()  
  
    plt.subplot(1, 2, 2)  
    plt.plot(train_accuracies, label='Training Accuracy')  
    plt.xlabel('Epoch')  
    plt.ylabel('Accuracy')  
    plt.title('Training Accuracy per Epoch')  
    plt.legend()  
  
    plt.show()
```

`plt.figure(figsize=(12, 5))`：创建一个新的图形（figure），并设置其大小为宽12英寸、高5英寸。

`plt.subplot(1, 2, 1)`：这行代码将图形分割成1行2列的子图（subplot），并激活第一个子图（从左到右计数）作为当前绘图区域。这意味着接下来的绘图命令将应用于这个子图。

`plt.plot(train_losses, label='Training Loss')`：在当前激活的子图上绘制`train_losses`数组或列表的数据。`label`参数用于给这条线一个标签，这个标签稍后会显示在图例中。

`plt.xlabel('Epoch')`、`plt.ylabel('Loss')`：分别为当前子图的x轴和y轴设置标签，这里x轴表示训练轮次（Epoch），y轴表示损失（Loss）。

`plt.title('Training Loss per Epoch')`：为当前子图设置标题。

`plt.legend()`：显示图例。由于之前为绘图线设置了`label`，这里会基于这些标签生成图例。

`plt.subplot(1, 2, 2)`：再次使用`subplot`函数，但这次激活的是第二个子图。这意味着接下来的绘图命令将应用于这个新的子图。

`plt.plot(train_accuracies, label='Training Accuracy')`：在第二个子图上绘制`train_accuracies`数组或列表的数据。`label`参数同样用于给这条线一个标签，这个标签稍后会显示在第二个子图的图例中。

`plt.xlabel('Epoch')`、`plt.ylabel('Accuracy')`：分别为第二个子图的x轴和y轴设置标签，x轴示训练轮次（Epoch），但y轴表示准确率（Accuracy）。

`plt.title('Training Accuracy per Epoch')`：为第二个子图设置标题。

`plt.legend()`：同样在第二个子图上显示图例。

`plt.show()`：显示最终的图形。这行代码会打开一个窗口，显示包含两个子图的图形，一个子图展示了训练损失随训练轮次的变化，另一个子图展示了训练准确率随训练轮次的变化。
### 提高准确率
```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
这里换为了Adam优化器有了明显提高从20%+提高到了70%+，后续又尝试了RMSprop，Adagrad但是效果不如Adam
```python
self.fc1 = nn.Linear(64*6*6, 1024)      
self.fc2 = nn.Linear(1024, 10)   
```
这里把特征数直接拉到了1024，对于训练集预测的准确率提高大，但是对于测试集的提高不大。
```python
num_epochs = 20
```
把训练次数调到了20，训练集的准确率接近了100%（在18次的时候准确率突然下降），但是测试集和10次的时候基本一样
```python
        self.conv1 = nn.Conv2d(3, 64, 3, 1)  
        self.conv2 = nn.Conv2d(64, 128, 3, 1) 
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(128*6*6, 1024)   
        self.fc2 = nn.Linear(1024, 10)   
```
这里将输出通道调高了，准确率提升了一点
```python
self.dropout = nn.Dropout(p=0.5) 
```
```python
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
```
这里添加了dropout以减小过拟合，提高了2%
```python
class ResNet18CIFAR10(nn.Module):  
    def __init__(self):  
        super(ResNet18CIFAR10, self).__init__()  
        self.model = resnet18(pretrained=True)  
        # 修改最后的全连接层以适应CIFAR-10的10个类别  
        num_ftrs = self.model.fc.in_features  
        self.model.fc = nn.Linear(num_ftrs, 10)  
  
    def forward(self, x):  
        x = self.model(x)
        return x  
```
预训练模型yyds，白嫖就是香
`pretrained=True `表示加载在 ImageNet 数据集上预训练的权重。
`num_ftrs = self.model.fc.in_features:` 这行代码获取了预训练模型最后的全连接层`（fc）`的输入特征数量`（in_features）`。
`self.model.fc = nn.Linear(num_ftrs, 10):` 这行代码替换了预训练模型的最后一个全连接层。新的全连接层有 num_ftrs 个输入特征，和 10 个输出特征，对应于 CIFAR-10 数据集的 10 个类别。
# 回答
### 计算机存储、处理图像的数据结构

计算机采用特定的数据结构来存储和处理图像。常见的图像格式如JPEG、PNG、BMP等，这些格式将图像的像素数据和其他信息（如颜色深度、分辨率、元数据等）以特定的编码方式存储。以BMP（位图）文件格式为例，它直接存储图像的像素数据，每个像素的颜色值以二进制数据表示。文件头包含文件类型、文件大小等信息；信息头包含图像的宽度、高度、颜色位数等信息；调色板（可选）用于8位或以下位图，定义颜色表；像素数据则是每个像素的RGB值，按行存储。

### 神经网络的设计及其组成部分

设计神经网络涉及多个方面，包括网络结构的构建、激活函数的选择、优化算法的应用以及正则化技术的引入等。一个神经网络通常包括以下几个部分：

* **输入层**：接收原始数据输入，每个输入节点对应数据的一个特征。
* **隐藏层**：位于输入层和输出层之间的一层或多层神经元，对输入进行非线性变换和特征提取，使得网络能够学习到更复杂的模式和表示。隐藏层的非线性变换是通过激活函数实现的。
* **输出层**：输出层的节点产生最终的模型输出，通常对应任务的预测结果。

### 欠拟合与过拟合的定义

* **欠拟合**：是指模型在训练数据和测试数据上的性能都较差。这通常是因为模型的结构或参数过于简单，无法捕捉到数据中的复杂关系和特征。
* **过拟合**：是指模型在训练数据上表现很好，但在测试数据或新数据上表现较差的情况。这通常是因为模型的结构或参数过于复杂，导致它过度拟合了训练数据的噪声和细节，而非学习到数据的真正内在规律。