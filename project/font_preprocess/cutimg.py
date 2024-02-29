import os
import shutil
import cv2
import glob

global_data_path = r"cutimg_data"
global_input_path = os.path.join(global_data_path, "input/test")
global_output_path = os.path.join(global_data_path, "output")
global_font_size = 32
global_accept_ratio = 1.4


def segment_word(input_img):
	height, width, channels = input_img.shape
	# Convert the input image into gray scale
	grayImg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	# Binarize the gray image with OTSU algorithm
	_, thresh = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
	# Create a Structuring Element size of width*1 for the horizontal contouring
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width * 2, 1))
	# apply Dilation for once only
	dilation = cv2.dilate(thresh, horizontal_kernel, iterations=1)
	# Find the contours
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	word_img = ~thresh
	preview = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)
	# Run through each contour and extract the bounding box
	segments = []
	for cnt in contours:
		# Computes the minimum rectangle
		x, y, w, h = cv2.boundingRect(cnt)
		# print(f"x, y, w, h = {x}, {y}, {w}, {h}")
		output_img = word_img[y: y + h, x: x + w]
		output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
		segments.append(output_img)
		# Draw a rectangle from the top left to the bottom right with the
		# Given Coordinates x, y and height and width
		cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 0)
	# Apply a Character Segmentation and return the output Image
	return preview, segments


def central_crop(character_img, font_size, accept_ratio):
	x, y, w, h = cv2.boundingRect(~character_img)
	ratio = h / w
	if ratio > accept_ratio:
		return None
	cropped_img = character_img[y: y + h, x: x + w]
	if h >= w:
		diff = h - w
		left = diff // 2
		right = diff - left
		cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=255)
	else:
		diff = w - h
		top = diff // 2
		bottom = diff - top
		cropped_img = cv2.copyMakeBorder(cropped_img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)
	cropped_img = cv2.copyMakeBorder(cropped_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
	cropped_img = cv2.resize(cropped_img, (font_size, font_size), cv2.INTER_NEAREST)
	_, thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
	return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def segment_character(input_img, font_size, accept_ratio):
	height, width, channels = input_img.shape
	# Convert the input image into gray scale
	grayImg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	# Binarize the gray image with OTSU algorithm
	_, thresh = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
	# Create a Structuring Element size of 1*height for the vertical contouring
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height * 2))
	# apply Dilation for once only
	dilation = cv2.dilate(thresh, vertical_kernel, iterations=1)
	# Find the contours
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	word_img = ~thresh
	preview = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)
	# Run through each contour and extract the bounding box
	segments = []
	for cnt in contours:
		# Computes the minimum rectangle
		x, y, w, h = cv2.boundingRect(cnt)
		# print(f"x, y, w, h = {x}, {y}, {w}, {h}")
		output_img = word_img[y: y + h, x: x + w]
		output_img = central_crop(output_img, font_size, accept_ratio)
		if output_img is not None:
			segments.append(output_img)
			# Draw a rectangle from the top left to the bottom right with the
			# Given Coordinates x, y and height and width
			cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 0)
	# Apply a Character Segmentation and return the output Image
	return preview, segments


def main():
	for input_img_path in glob.glob(os.path.join(global_input_path, "**/*.*"), recursive=True):
		print(f"Reading image from \"{input_img_path}\"...")
		input_img = cv2.imread(input_img_path, cv2.COLOR_GRAY2BGR)
		words_preview, words = segment_word(input_img)
		output_img_path = os.path.join(global_output_path, os.path.splitext(os.path.basename(input_img_path))[0])
		if os.path.exists(output_img_path):
			shutil.rmtree(output_img_path)
		os.makedirs(output_img_path, exist_ok=True)
		# cv2.imshow('Words Preview', words_preview)
		# cv2.waitKey()
		for i, word_img in enumerate(words):
			word_dir = os.path.join(output_img_path, f"word_{i:04}")
			print(f"Creating dir \"{word_dir}\"...")
			if os.path.exists(word_dir):
				shutil.rmtree(word_dir)
			os.makedirs(word_dir, exist_ok=True)
			characters_preview, characters = segment_character(word_img, global_font_size, global_accept_ratio)
			# cv2.imshow('Characters Preview', characters_preview)
			# cv2.waitKey()
			for i, character_img in enumerate(characters):
				character_img_file = os.path.join(word_dir, f"{i:04}.png")
				print(f"Writing image to \"{character_img_file}\"...")
				cv2.imwrite(character_img_file, character_img)


if __name__ == '__main__':
	main()
