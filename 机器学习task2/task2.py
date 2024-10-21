#导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
 
 
#指定运行位置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# TODO:解释参数含义，在?处填入合适的参数
batch_size = 256#批量大小
learning_rate = 0.001#学习率
num_epochs = 10#次数

#数据预处理 
transform = transforms.Compose([
        transforms.ToTensor(),#格式转化
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        
])
# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
 
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 卷积层1：输入通道3，输出通道32，卷积核大小3x3，步长1  
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 卷积层2：输入通道32，输出通道64，卷积核大小3x3，步长1  
        self.pool = nn.MaxPool2d(2, 2)  # 2x2池化窗口，步长为2  
        self.fc1 = nn.Linear(64*6*6, 128)      # 全连接层1：输入特征数，输出特征数128  
        self.fc2 = nn.Linear(128, 10)   
    def forward(self,x):
        # TODO:这里补全你的前向传播 
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6) # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

        
# 实例化模型 
model = Network().to(device)  
 
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数  
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD优化器  
  
 
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
  
            optimizer.zero_grad()
 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
 
def test():
    model.eval ()
    correct = 0#正确数
    total = 0#总数
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
 
if __name__ == "__main__":
    train()
    test()