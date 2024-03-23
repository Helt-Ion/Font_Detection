import os
import random
import shutil
import numpy
import cv2

from PIL import Image, ImageFont, ImageDraw
from modules import filereader

global_data_path = "data"


def generate(file_root_dir, fonts_lib_path, fonts_info_path, indexes):
	if os.path.exists(file_root_dir):
		shutil.rmtree(file_root_dir)
	os.makedirs(file_root_dir, exist_ok=True)
	font_labels, font_classes, font_list = filereader.read_fonts(fonts_info_path)
	for (font_label, font_class, sample_list) in zip(font_labels, font_classes, font_list):
		print(f"Generating {font_class}...")
		for sample_index, (font, bias) in enumerate(sample_list):
			font_file = os.path.join(fonts_lib_path, font)
			sample_name = f"sample_{sample_index}"
			font_dir = os.path.join(file_root_dir, font_class, sample_name)
			print(f"Generating {font_class}/{sample_name} from {font_file}...")
			os.makedirs(font_dir, exist_ok=True)
			for i, word in indexes:
				img = Image.new("RGB", (32, 32), (255, 255, 255))
				draw = ImageDraw.Draw(img)
				# Add text to image
				# fill controls the color of text, (R, G, B)
				draw.text((0, bias), word, fill=(0, 0, 0), font=ImageFont.truetype(font_file, 32, encoding="utf-8"))
				img_cv2 = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
				_, output_img = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
				output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
				# Save image
				img_path = os.path.join(font_dir, f"{font_class}_{sample_name}_{i + 1}.png")
				cv2.imwrite(img_path, output_img)


def generate_train_dataset(data_path):
	print(f"Generating train dataset...")
	file_root_dir = os.path.join(data_path, "train")
	fonts_lib_path = os.path.join(data_path, "fonts")
	words_path = os.path.join(data_path, "Words.txt")
	fonts_info_path = os.path.join(data_path, "Fonts.xml")
	f_words = filereader.read_words(words_path)
	indexes = list(enumerate(f_words))
	generate(file_root_dir, fonts_lib_path, fonts_info_path, indexes)


def generate_test_dataset(data_path):
	print(f"Generating test dataset...")
	file_root_dir = os.path.join(data_path, "test")
	fonts_lib_path = os.path.join(data_path, "fonts")
	words_path = os.path.join(data_path, "Words.txt")
	fonts_info_path = os.path.join(data_path, "Fonts.xml")
	f_words = filereader.read_words(words_path)
	sample_siz = len(f_words) // 5
	indexes = random.sample(list(enumerate(f_words)), sample_siz)
	generate(file_root_dir, fonts_lib_path, fonts_info_path, indexes)


def main():
	generate_train_dataset(global_data_path)
	generate_test_dataset(global_data_path)


if __name__ == '__main__':
	main()
