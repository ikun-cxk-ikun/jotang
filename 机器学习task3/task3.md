# Task3
### Vision Transformer (ViT)学习笔记 

#### 1、架构分析
Vision Transformer模型主要由三部分组成：Embedding层、Transformer Encoder和MLP Head。

1. **Embedding层**：
   - **Patch Embedding**：将图像分割成若干个小块，并将每个小块嵌入到一个固定维度的向量中
   - **Positional Encoding**：为每个小块添加位置编码，以保留其在图像中的位置信息。
   - **Class Token**：在输入序列的开头添加一个特殊的“类别”标记（class token），用于代表整个图像的特征向量。

2. **Transformer Encoder**：由多个Transformer编码器层堆叠而成，每个编码器层包括多头自注意力机制和前馈神经网络。这些层对输入的patch序列进行编码，提取图像中的特征。
   - **自注意力机制**：能够捕捉到输入序列中的长距离依赖关系。
   - **前馈神经网络**：进一步增强模型的表示能力。

3. **MLP Head**：
   - **输出层**：通常由一个全连接层和一个softmax层组成，用于对输入的表示向量进行分类或回归。使用一个多层感知器（MLP）头来处理 class token 的输出，以进行类别预测

1. **过程**：首先是**Patch Embedding**，因为transformer的输入是一个序列，而图片是一个三维的，所以先得把图像给转化为序列数据。将`H*W*C`的图片切分成`N`个`P*P*C`的图像块，其中序列长度$$N=H*W/P^2$$
对每个patch进行embedding，通过一个线性变换层将二维的patch嵌入表示长度为D的一维向量，然后通过**position embedding**添加位置嵌入信息（0，1，2，3，...），即每个patch在原始图像中的位置信息。即采用**position embedding + patch embedding**方式来结合position信息，即公式中的Epos。接下来进入**Transformer Encoder**，将前面得到的初始的Z0作为transformer的初始输入，transformer encoder 是有L个**MSA**（多头自注意力机制）和**MLP**（多层感知机，MLP的主要功能是对输入数据进行非线性变换，以提取更高层次的特征。它通常位于MSA之后，用于进一步处理MSA的输出，这里的MLP就是一个两层的全连接层，采用GELU激活函数）块交替组成的，每次在MSA和MLP前都要进行LN归一化，在每个MSA和MLP后都加了残差链接。（防止梯度弥散），虽然最终Transformer Encoder会输出n+1个，但是最终只会选择第0个作为MLP Head的输入，接下来就是MLP Head
```python
if representation_size and not distilled:
    self.has_logits = True 
    self.num_features = representation_size
    self.pre_logits = nn.Sequential(OrderedDict([
        ("fc",nn.Linear(embed_dim,representation_size)),
        ("act",nn.Tanh())
    ]))
else:
    self.has_logits = False
    self.pre_logits = nn.Identity()
```
当MLP在训练image net-21k时是有一个全连接层和一个激活函数组成的，但是21k以上或者是自己数据的时候只有一个全连接层
**混合模型**：将transformer和CNN结合，即将Resnet的中间层的feature map作为transformer的输入

#### 2. 关键技术创新点

- **自注意力机制**: ViT 利用自注意力机制来捕捉图像中的长距离依赖关系，而不是仅仅依赖局部特征。
- **Patch Embedding**: 通过将图像分割成小块并嵌入到向量中，ViT 能够处理整个图像的全局信息。
- **Position Encoding**: 为每个小块添加位置编码，以保留其在图像中的位置信息，从而在自注意力机制中利用位置信息。

#### 3. ViT 和 CNN 的比较

##### 相同点

- **图像处理任务**: 两者都用于图像分类等任务。
- **输入数据**: 都接受图像作为输入数据。
- **预训练和微调**: 都可以在大规模数据集上预训练，然后在特定任务上微调。

##### 不同点

  - **CNN**: 通过卷积操作逐步提取局部特征，然后通过池化操作汇总。
  - **ViT**: 基于 Transformer 架构，将图像分割成小块，并通过自注意力机制捕捉长距离依赖关系。
  - **总结**：自注意力机制允许ViT整合信息在整个图像，甚至在最低层，换句话说，ViT在一开始就可以放眼全局，而CNN只能随着层数度的增加才可以注意到更多细节。
##### 优缺点

- **CNN**:
  - **优点**:
    - 在局部特征提取方面表现优异。
    - 结构简单。
  - **缺点**:
    - 对长距离依赖关系的捕捉能力有限。
    - 需要大量的层和参数来处理复杂的任务。

- **ViT**:
  - **优点**:
    - 能够捕捉图像中的长距离依赖关系。
    - 在大规模数据集上表现优异。
  - **缺点**:
    - 对小规模数据集的性能可能不如 CNN。
    - 需要较大的计算资源和内存。

### ViT复现
#### 1.代码 
```python
import timm
```
导入timm库
```python
    transform = transforms.Compose([  
    transforms.Resize(224),#ViT通常需要224*224的输入
    transforms.ToTensor(),  # 格式转化  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化  
])  
```
新增
```python
transforms.Resize(224)
```
将输入的图像大小调整为224x224像素ViT通常需要224*224的输入
```python
class ViTCIFAR10(nn.Module):  
    def __init__(self):  
        super(ViTCIFAR10, self).__init__()  
        self.model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=10)  
        
  
    def forward(self, x):  
        x = self.model(x)
        return x  
```
载入patch为16的模型，设置10个分类
#### 实验报告
训练时间：差不多10分钟才一轮
准确率不是很高，只有40%+的样子
用CNN模型的话运行的速度很快，准确率也比ViT高不少，ViT模型的表现没有预期的那样🐂，甚至不如CNN，原因应该是因为CIFAR10太小，而ViT更适合处理大量数据集，CNN在这种小规模的数据集上表现更出色。
**较大的Patch Size**：能够捕捉更多的全局信息，但可能丢失细节。较大的patch size意味着模型在每一步处理的区域更大，因此产生的特征图会更为粗糙，分辨率较低。通常会导致较低的计算复杂度和资源消耗，因为模型需要处理的patches数量较少。
**较小的Patch Size**：能够保留更多的细节，但可能无法捕捉到全局特征。较小的patch size可以生成分辨率较高的特征图，包含更多的局部细节信息。会增加模型的计算复杂度和资源消耗，因为模型需要处理更多的patches。