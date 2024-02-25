import os
import random
import shutil
import numpy
import cv2

from PIL import Image, ImageFont, ImageDraw
from modules import filereader

global_data_path = "data"


def generate_train_dataset(data_path):
	words_path = os.path.join(data_path, "Words.txt")
	fonts_path = os.path.join(data_path, "Fonts.txt")
	file_root_dir = os.path.join(data_path, "train")
	f_words = filereader.read_words(words_path)
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_path)
	if os.path.exists(file_root_dir):
		shutil.rmtree(file_root_dir)
	os.makedirs(file_root_dir, exist_ok=True)
	for _, (label, font, bias) in enumerate(zip(classes, font_list, font_bias)):
		font_dir = os.path.join(file_root_dir, label)
		if not os.path.exists(font_dir):
			os.mkdir(font_dir)
		for i, word in enumerate(f_words):
			img = Image.new("RGB", (32, 32), (255, 255, 255))
			draw = ImageDraw.Draw(img)
			# 在图片上添加文字
			# fill用来设置绘制文字的颜色,(R,G,B)
			draw.text((0, bias), word, fill=(0, 0, 0), font=ImageFont.truetype(font, 32, encoding="utf-8"))
			img_cv2 = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
			_, output_img = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
			output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
			# 保存图片
			img_path = os.path.join(font_dir, f"{label}_{i + 1}.png")
			cv2.imwrite(img_path, output_img)


def generate_test_dataset(data_path):
	words_path = os.path.join(data_path, "Words.txt")
	fonts_path = os.path.join(data_path, "Fonts.txt")
	file_root_dir = os.path.join(data_path, "test")
	f_words = filereader.read_words(words_path)
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_path)
	if os.path.exists(file_root_dir):
		shutil.rmtree(file_root_dir)
	os.makedirs(file_root_dir, exist_ok=True)
	indexes = random.sample(range(0, len(f_words)), len(f_words) // 5)
	for _, (label, font, bias) in enumerate(zip(classes, font_list, font_bias)):
		font_dir = os.path.join(file_root_dir, label)
		if not os.path.exists(font_dir):
			os.mkdir(font_dir)
		for i in indexes:
			word = f_words[i]
			img = Image.new("RGB", (32, 32), (255, 255, 255))
			draw = ImageDraw.Draw(img)
			# 在图片上添加文字
			# fill用来设置绘制文字的颜色,(R,G,B)
			draw.text((0, bias), word, fill=(0, 0, 0), font=ImageFont.truetype(font, 32, encoding="utf-8"))
			img_cv2 = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
			_, output_img = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
			output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
			# 保存图片
			img_path = os.path.join(font_dir, f"{label}_{i + 1}.png")
			cv2.imwrite(img_path, output_img)


def main():
	generate_train_dataset(global_data_path)
	generate_test_dataset(global_data_path)


if __name__ == '__main__':
	main()
