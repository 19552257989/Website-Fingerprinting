import torch
from torch import nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, dropout_p, activation):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  # 计算padding以保持输出大小与输入大小相同
        # 定义一个由两个卷积层组成的卷积块，每个卷积层后跟批量归一化和激活
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),  # First convolutional layer
            nn.BatchNorm1d(out_channels),  # 批量归一化层
            activation(inplace=True),  # 激活函数(e.g., ELU or ReLU)
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),  # Second convolutional layer
            nn.BatchNorm1d(out_channels),  # 批量归一化层
            activation(inplace=True),  # Activation function
            nn.MaxPool1d(pool_size, pool_stride, padding=0),  # 用于对输入进行下采样的最大池化层
            nn.Dropout(p=dropout_p)  # 正则化用的 dropout 层
        )

    def forward(self, x):
        # 通过卷积块传递输入
        return self.block(x)

class DF(nn.Module):
    def __init__(self, num_classes, num_tab=1):
        super(DF, self).__init__()
        
        # Configuration parameters for the convolutional blocks
        filter_num = [32, 64, 128, 256]  # 每个模块的过滤器数量
        kernel_size = 8  # 卷积层核大小
        conv_stride_size = 1  # 卷积层的步长
        pool_stride_size = 4  # Stride size for max pooling layers
        pool_size = 8  # Kernel(核) size for max pooling layers
        length_after_extraction = 18  # 特征提取部分之后的特征图长度
        
        # 使用序列容器定义网络的特征提取部分，包含 ConvBlock 实例
        self.feature_extraction = nn.Sequential(
            ConvBlock(1, filter_num[0], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ELU),  # Block 1
            ConvBlock(filter_num[0], filter_num[1], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU),  # Block 2
            ConvBlock(filter_num[1], filter_num[2], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU),  # Block 3
            ConvBlock(filter_num[2], filter_num[3], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ReLU)  # Block 4
        )
        
        # 定义网络的分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将张量展平为向量
            nn.Linear(filter_num[3] * length_after_extraction, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # 批标准化层
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=0.7),  # 正则化用的 dropout 层
            nn.Linear(512, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        # Pass the input through the feature extraction part
        x = self.feature_extraction(x)
        
        # Pass the output through the classifier part
        x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    net = DF(num_classes=100)  # 创建一个包含 100 个类别的模型实例
    x = np.random.rand(32, 1, 5000)  # 生成一个形状为（32，1，5000）的随机输入张量
    x = torch.tensor(x, dtype=torch.float32)  #将输入转换为 float32 类型的 torch 张量
    out = net(x)  # Perform a forward pass through the network指在神经网络中执行一次前向传播的过程。在深度学习和神经网络中，前向传播是指数据从输入层经过网络中的每一层，直到输出层的过程。
    print(f"Input shape: {x.shape} --> Output shape: {out.shape}")  #将输入张量（tensor）和输出张量的形状（shape）打印出来
