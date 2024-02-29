import os
import random
import shutil
import numpy
import cv2

from PIL import Image, ImageFont, ImageDraw
from modules import filereader

global_data_path = "font_generate_data"


def font_generate(data_path):
	print(f"Generating fonts...")
	words_path = os.path.join(data_path, "Words.txt")
	fonts_info_path = os.path.join(data_path, "Fonts.txt")
	fonts_lib_path = os.path.join(data_path, "fonts")
	file_root_dir = os.path.join(data_path, "output")
	f_words = filereader.read_words(words_path)
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_info_path)
	if os.path.exists(file_root_dir):
		shutil.rmtree(file_root_dir)
	os.makedirs(file_root_dir, exist_ok=True)
	for _, (label, font, bias) in enumerate(zip(classes, font_list, font_bias)):
		font_file = os.path.join(fonts_lib_path, font)
		font_dir = os.path.join(file_root_dir, label)
		print(f"Generating {label} from {font_file}...")
		if not os.path.exists(font_dir):
			os.mkdir(font_dir)
		for i, word in enumerate(f_words):
			img = Image.new("RGB", (32, 32), (255, 255, 255))
			draw = ImageDraw.Draw(img)
			# 在图片上添加文字
			# fill用来设置绘制文字的颜色,(R,G,B)
			draw.text((0, bias), word, fill=(0, 0, 0), font=ImageFont.truetype(font_file, 32, encoding="utf-8"))
			img_cv2 = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
			_, output_img = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
			output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
			# 保存图片
			img_path = os.path.join(font_dir, f"{label}_{i + 1}.png")
			cv2.imwrite(img_path, output_img)



def main():
	font_generate(global_data_path)


if __name__ == '__main__':
	main()
