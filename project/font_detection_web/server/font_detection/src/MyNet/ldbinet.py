from typing import Tuple
import torch
from torch import nn
from torch import Tensor


# 基础卷积层（卷积 + ReLU）
class BasicConv2d(nn.Module):
	# init()：进行初始化，申明模型中各层的定义
	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
		# ReLU(inplace=True)：将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
		self.relu = nn.ReLU(inplace=True)

	# 前向传播过程
	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		return x


# Inception结构
class Inception(nn.Module):
	# init()：进行初始化
	def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
		super(Inception, self).__init__()
		# 分支1，单1x1卷积层
		# num_features, 4, 6, 8, 1, 2, 2
		self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
		# 分支2，1x1卷积层后接3x3卷积层
		self.branch2 = nn.Sequential(
			BasicConv2d(in_channels, ch3x3red, kernel_size=1),
			# 保证输出大小等于输入大小
			BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
		)
		# 分支3，1x1卷积层后接5x5卷积层
		self.branch3 = nn.Sequential(
			BasicConv2d(in_channels, ch5x5red, kernel_size=1),
			# 保证输出大小等于输入大小
			BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
		)
		# 分支4，3x3最大池化层后接1x1卷积层
		self.branch4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			BasicConv2d(in_channels, pool_proj, kernel_size=1)
		)

	# forward()：定义前向传播过程,描述了各层之间的连接关系
	def forward(self, x):
		branch1 = self.branch1(x)
		branch2 = self.branch2(x)
		branch3 = self.branch3(x)
		branch4 = self.branch4(x)

		# 在通道维上连结输出
		outputs = [branch1, branch2, branch3, branch4]
		# cat()：在给定维度上对输入的张量序列进行连接操作
		return torch.cat(outputs, 1)


class DenseLayer(nn.Module):
	def __init__(self, in_channels, growth_rate, bn_size, drop_rate: float = 0.0):
		super(DenseLayer, self).__init__()
		self.drop_rate = drop_rate
		self.bottleneck = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, bn_size * growth_rate, 1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(bn_size * growth_rate),
			nn.ReLU(True),
			nn.Conv2d(bn_size * growth_rate, growth_rate, 3, stride=1, padding=1, bias=False)
		)
		self.dropout = nn.Dropout(self.drop_rate)

	def forward(self, x):
		y = self.bottleneck(x)
		if self.drop_rate > 0:
			# 如果需要dropout的话
			y = self.dropout(y)
		# 特征融合
		output = torch.cat([x, y], 1)
		return output


class DenseBlock(nn.Module):
	# 组合多个DenseLayer
	def __init__(self, layers_num, in_channels, growth_rate, bn_size, drop_rate: float = 0.0):
		super(DenseBlock, self).__init__()
		layers = []
		for i in range(layers_num):
			# 按growth rate叠加输入的通道数
			layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
		# 将列表中的每一层按序传给Sequential
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		output = self.layers(x)
		return output


class Transition(nn.Module):
	def __init__(self, in_channels, channels):
		super(Transition, self).__init__()
		self.transition = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(True),
			nn.Conv2d(in_channels, channels, 1, stride=1, padding=0, bias=False),
			nn.AvgPool2d(2, stride=2)
		)

	def forward(self, x):
		output = self.transition(x)
		return output


class Ldbinet(nn.Module):
	def __init__(
			self,
			growth_rate: int = 18,
			block_config: Tuple[int, int] = (15, 9),
			num_init_features: int = 12,
			bn_size: int = 4,
			drop_rate: float = 0.2,
			num_classes: int = 5,
	) -> None:
		super().__init__()
		# First convolution
		self.first_convolution = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=num_init_features, kernel_size=5),
			nn.BatchNorm2d(num_init_features),
			nn.ReLU(inplace=True)
			# nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
		)
		num_features = num_init_features
		# in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
		self.inception1 = Inception(num_features, 4, 6, 8, 1, 2, 2)
		num_features = 4 + 8 + 2 + 2  # 16
		self.inception2 = Inception(num_features, 8, 8, 12, 2, 6, 4)
		num_features = 8 + 12 + 6 + 4  # 30
		self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

		self.denseblock1 = DenseBlock(layers_num=block_config[0], in_channels=num_features, bn_size=bn_size,
									  growth_rate=growth_rate, drop_rate=drop_rate)
		num_features = num_features + block_config[0] * growth_rate
		self.transition = Transition(in_channels=num_features, channels=num_features // 2)
		num_features //= 2
		self.denseblock2 = DenseBlock(layers_num=block_config[1], in_channels=num_features, bn_size=bn_size,
									  growth_rate=growth_rate, drop_rate=drop_rate)
		num_features = num_features + block_config[1] * growth_rate
		# Linear layer
		self.classification = nn.Sequential(
			nn.AvgPool2d(kernel_size=7, stride=1),
			nn.Flatten(),
			nn.Linear(num_features, num_classes)
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.first_convolution(x)
		x = self.inception1(x)
		x = self.inception2(x)
		x = self.maxpool(x)
		x = self.denseblock1(x)
		x = self.transition(x)
		x = self.denseblock2(x)
		x = self.classification(x)
		return x
