import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pylab import mpl


# 重构Dataset类
class Mydatasetpro(torch.utils.data.Dataset):
	# 初始化函数，得到数据
	def __init__(self, data_root, data_label, transform):
		self.data = data_root
		self.label = data_label
		self.transforms = transform
		# index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回

	def __getitem__(self, index):
		data = self.data[index]
		labels = self.label[index]
		pil_img = Image.open(data)
		data = self.transforms(pil_img)
		return data, labels
		# 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼

	def __len__(self):
		return len(self.data)


def data_get(train_imgs_path, test_imgs_path, classes, batchsz, transform):
	train_labels = []
	test_labels = []  # 用于生成标签
	# 对所有图片路径进行迭代
	for img in train_imgs_path:
		# 区分出每个img，应该属于什么类别
		img_class = os.path.basename(os.path.dirname(img))
		for i, c in enumerate(classes):
			if c == img_class:
				train_labels.append(i)
	for img in test_imgs_path:
		# 区分出每个img，应该属于什么类别
		img_class = os.path.basename(os.path.dirname(img))
		for i, c in enumerate(classes):
			if c == img_class:
				test_labels.append(i)  # 为对应的数据集增加标签
	data_train = Mydatasetpro(train_imgs_path, train_labels, transform)  # 训练数据读取
	data_train_loader = DataLoader(data_train, batch_size=batchsz, shuffle=True)  # 训练数据载入，训练时数据标签要打乱

	data_test = Mydatasetpro(test_imgs_path, test_labels, transform)  # 测试数据读取
	data_test_loader = DataLoader(data_test, batch_size=batchsz, shuffle=False)  # 测试数据载入
	return data_train, data_train_loader, data_test, data_test_loader


def data_get_normal(imgs_path, classes, batchsz, transform):
	labels = []
	# 对所有图片路径进行迭代
	for img in imgs_path:
		# 区分出每个img，应该属于什么类别
		for i, c in enumerate(classes):
			if c in img:
				labels.append(i)
	data = Mydatasetpro(imgs_path, labels, transform)  # 数据读取
	data_loader = DataLoader(data, batch_size=batchsz, shuffle=False)  # 数据载入，训练时数据标签要打乱
	return data, data_loader


def main():
	mpl.rcParams["font.sans-serif"] = ["SimHei"]
	# 测试下数据集里的图片
	# plt.figure(figsize=(12, 8))
	# for i, (img, label) in enumerate(zip(imgs_batch[:32], labels_batch[:8])):
	#     img = img.permute(1, 2, 0).numpy()
	#     plt.subplot(2, 4, i + 1)
	#     plt.xlabel(classes[label.numpy()])
	#     plt.imshow(img)
	# plt.show()


if __name__ == '__main__':
	main()
