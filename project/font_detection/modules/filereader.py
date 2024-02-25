def read_words(file_path: str):
	with open(file_path, "r", encoding="utf-8") as f:
		ret = f.read()
	return ret


def read_fonts(file_path: str):
	font_label, classes, font_list, font_bias = [], [], [], []
	with open(file_path, "r", encoding="utf-8") as f:
		lines = [line.rstrip("\n").split(" ") for line in f]
		for label, class_name, path, bias in lines:
			font_label.append(label), classes.append(class_name), font_list.append(path), font_bias.append(int(bias))
	return font_label, classes, font_list, font_bias


def main():
	print("Filereader Test:")
	words = read_words("../data/Words.txt")
	print("Words:")
	print(words)
	font_label, classes, font_list, font_bias = read_fonts("../data/Fonts.txt")
	print("font_label:")
	print(font_label)
	print("classes:")
	print(classes)
	print("font_list:")
	print(font_list)
	print("font_bias:")
	print(font_bias)


if __name__ == "__main__":
	main()
