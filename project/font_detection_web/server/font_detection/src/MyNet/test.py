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
global_model_path = "MyNet/checkpoint/font_recognize_200.pth"  # 模型路径，改成你自己的
global_test_path = "MyNet/test"


# 若有标红，说明有缺少库，自己安装一下
# opencv版本是4.7.0.72
def recognize_font(img, model, classes_dict):
	height, weight, channel = img.shape
	_img = np.zeros([3, img.shape[0], img.shape[1]])  # 格式为(高度,宽度,通道数)，需要转化成(通道数,高度,宽度)
	for c in range(channel):
		for h in range(height):
			for w in range(weight):
				_img[c][h][w] = img[h][w][c]
	img = torch.from_numpy(_img)  # numpy转换为tensor
	img = img.float()
	img = img / 255  # RGB归一化
	img = torch.unsqueeze(img, 0)  # 扩张维度变为[N,C,H,W]
	torch_resize = Resize([32, 32])  # 定义Resize类对象
	img = torch_resize(img)  # 变更大小为32*32
	img = img.cpu()
	output = model(img).data
	sf = softmax(output, dim=1)
	_, predicted = torch.max(sf, 1)
	# print(sf,predicted.item(), classes_dict[predicted.item()])
	# print(output, classes_dict[predicted.item()])
	return classes_dict[predicted.item()]


def recognize(data_path, model_path, test_path):
	fonts_path = os.path.join(data_path, "Fonts.txt")
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_path)

	# 字体种类，根据需要修改，数量要对得上，目前是20种
	num_classes = len(classes)
	classes_dict = dict(zip(list(range(num_classes)), font_label))
	model = Ldbinet(num_classes=num_classes)  # 如果标红，自己import一下
	model.cpu()
	model.load_state_dict(torch.load(model_path, map_location="cpu"))
	model.eval()  # 重要，必须加上，否则预测不准确
	for img_path in glob.glob(os.path.join(test_path, "**/*.*"), recursive=True):
		img = cv2.imread(img_path, cv2.COLOR_GRAY2BGR)
		font = recognize_font(img, model, classes_dict)
		print(f"Font of {img_path}: {font}")


def main():
	recognize(global_data_path, global_model_path, global_test_path)


if __name__ == '__main__':
	main()
