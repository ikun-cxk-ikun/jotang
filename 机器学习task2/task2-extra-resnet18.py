import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt  
from torchvision.models import resnet18  
  
# 指定运行位置  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# TODO: 解释参数含义，在?处填入合适的参数  
batch_size = 256  # 批量大小  
learning_rate = 0.001  # 学习率  
num_epochs = 10  # 训练次数  
  
# 数据预处理  
transform = transforms.Compose([  
    
    transforms.ToTensor(),  # 格式转化  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化  
])  
  
# root可以换为你自己的路径  
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)  
  
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)  
  
# 使用ResNet-18预训练模型，并修改最后的全连接层  
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
  
# 实例化模型  
model = ResNet18CIFAR10().to(device)  
  
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器  
  
train_losses = []  
train_accuracies = []  
  
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
        train_losses.append(running_loss / len(trainloader))  
        train_accuracies.append(accuracy)  
  
def test():  
    model.eval()  
    correct = 0  
    total = 0  
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
    # 绘制损失和准确率曲线  
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