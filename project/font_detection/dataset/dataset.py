import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pylab import mpl


# Refactor dataset class
class Mydatasetpro(torch.utils.data.Dataset):
	# Init function
	def __init__(self, data_root, data_label, transform):
		self.data = data_root
		self.label = data_label
		self.transforms = transform

	def __getitem__(self, index):
		data = self.data[index]
		labels = self.label[index]
		pil_img = Image.open(data)
		data = self.transforms(pil_img)
		return data, labels

	def __len__(self):
		return len(self.data)


def data_get(train_imgs_path, test_imgs_path, classes, batch_size, transform):
	train_labels = []
	test_labels = []
	for img in train_imgs_path:
		# Get the corresponding font class of the image
		font_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
		img_class = os.path.basename(font_dir)
		for i, c in enumerate(classes):
			if c == img_class:
				train_labels.append(i)
	for img in test_imgs_path:
		# Get the corresponding font class of the image
		font_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
		img_class = os.path.basename(font_dir)
		for i, c in enumerate(classes):
			if c == img_class:
				test_labels.append(i)
	data_train = Mydatasetpro(train_imgs_path, train_labels, transform)
	data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
	data_test = Mydatasetpro(test_imgs_path, test_labels, transform)
	data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
	return data_train, data_train_loader, data_test, data_test_loader


def main():
	print("Hello Dataset!")


if __name__ == '__main__':
	main()
