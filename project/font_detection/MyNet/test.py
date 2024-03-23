import cv2
import numpy as np
import torch
import os
import glob
from torchvision.transforms import Resize
from torch.nn.functional import softmax

from MyNet.ldbinet import Ldbinet
from modules import filereader

global_data_path = "data"
global_model_path = "MyNet/checkpoint/font_recognize_200.pth"
global_test_path = "MyNet/test"


# Opencv version: 4.7.0.72
def recognize_font(img, model, classes_dict):
	height, weight, channel = img.shape
	# Convert (H, W, C) to (C, H, W)
	_img = np.zeros([3, img.shape[0], img.shape[1]])
	for c in range(channel):
		for h in range(height):
			for w in range(weight):
				_img[c][h][w] = img[h][w][c]
	# Change numpy to tensor
	img = torch.from_numpy(_img)
	img = img.float()
	# RGB normalization
	img = img / 255
	# Extend dimension to [N, C, H, W]
	img = torch.unsqueeze(img, 0)
	# Define Resize slass
	torch_resize = Resize([32, 32])
	# Resize to 32 * 32
	img = torch_resize(img)
	img = img.cpu()
	output = model(img).data
	sf = softmax(output, dim=1)
	_, predicted = torch.max(sf, 1)
	# print(sf,predicted.item(), classes_dict[predicted.item()])
	# print(output, classes_dict[predicted.item()])
	return classes_dict[predicted.item()]


def recognize(data_path, model_path, test_path):
	fonts_path = os.path.join(data_path, "Fonts.xml")
	font_labels, font_classes, font_list = filereader.read_fonts(fonts_path)
	# Number of font classes
	num_classes = len(font_classes)
	classes_dict = dict(zip(list(range(num_classes)), font_labels))
	model = Ldbinet(num_classes=num_classes)
	model.cpu()
	model.load_state_dict(torch.load(model_path, map_location="cpu"))
	# Set model to eval mode
	model.eval()
	for img_path in glob.glob(os.path.join(test_path, "**/*.*"), recursive=True):
		img = cv2.imread(img_path, cv2.COLOR_GRAY2BGR)
		font = recognize_font(img, model, classes_dict)
		print(f"Font of {img_path}: {font}")


def main():
	recognize(global_data_path, global_model_path, global_test_path)


if __name__ == '__main__':
	main()
