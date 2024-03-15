import cv2
import numpy as np
import torch
import os
import glob
from torchvision.transforms import Resize
from torch.nn.functional import softmax

from MyNet.ldbinet import Ldbinet
from modules import filereader, cutimg


# 若有标红，说明有缺少库，自己安装一下
# opencv版本是4.7.0.72
def predict(img, model):
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
	img.cpu()
	output = model(img).data
	sf = softmax(output, dim=1)
	_, predicted = torch.max(sf, 1)
	return predicted.item()


def init(data_path, model_file):
	fonts_path = os.path.join(data_path, "Fonts.txt")
	model_path = os.path.join(data_path, model_file)
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_path)

	# 字体种类，根据需要修改，数量要对得上，目前是20种
	num_classes = len(classes)
	classes_dict = dict(zip(list(range(num_classes)), font_label))
	model = Ldbinet(num_classes=num_classes)  # 如果标红，自己import一下
	model.load_state_dict(torch.load(model_path, map_location="cpu"))
	model.cpu()
	model.eval()  # 重要，必须加上，否则预测不准确
	return model, classes_dict


def find_majority(predict_list):
	predict_map = {}
	for class_id in predict_list:
		if class_id in predict_map:
			predict_map[class_id] += 1
		else:
			predict_map[class_id] = 1
	max_id, max_cnt = 0, 0
	flag = False
	for k, v in predict_map.items():
		if flag == False:
			max_id, max_cnt = k, v
			flag = True
		elif v > max_cnt:
			max_id, max_cnt = k, v
	return max_id


def recognize_font(input_img_path, model, classes_dict):
	print(f"Reading image {input_img_path}...")
	input_img = cv2.imread(input_img_path, cv2.COLOR_GRAY2BGR)
	segments = cutimg.segment_img(input_img)
	predict_list = []
	for i, img in enumerate(segments):
		class_id = predict(img, model)
		predict_list.append(class_id)
		print(f"Font of image {i}: {classes_dict[class_id]}")
	max_id = find_majority(predict_list)
	font_type = classes_dict[max_id]
	print(f"Font of {input_img_path}: {font_type}")
	return font_type


def main():
	print("训练6个字体的结果：")
	model, classes_dict = init("checkpoint", "font_recognize_200.pth")
	recognize_font("input/SimSun_Large.png", model, classes_dict)


if __name__ == '__main__':
	main()