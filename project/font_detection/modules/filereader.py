import xml.etree.ElementTree as ElementTree


def read_words(file_path):
	with open(file_path, "r", encoding="utf-8") as f:
		ret = f.read()
	return ret


def read_fonts(file_path):
	font_list = []
	tree = ElementTree.parse(file_path)
	for font in tree.findall('font'):
		font_name = font.find('name').text
		font_class = font.find('class').text
		sample_list = []
		for sample in font.findall('sample'):
			sample_list.append((sample.find('file').text, int(sample.find('bias').text)))
		font_list.append((font_name, font_class, sample_list))
	return font_list


def main():
	print("Filereader Test:")
	words = read_words("../data/Words.txt")
	print("Words:")
	print(words)
	font_list = read_fonts("../data/Fonts.xml")
	print("font_list:")
	print(font_list)


if __name__ == "__main__":
	main()
