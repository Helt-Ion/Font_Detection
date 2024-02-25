import docx
import re
import os


def get_pictures(docx_file, output_path):
	"""
	图片提取
	:param docx_file: 输入路径
	:param output_path: 输出路径
	"""
	try:
		doc = docx.Document(docx_file)
		dict_rel = doc.part.rels
		for rel in dict_rel:
			rel = dict_rel[rel]
			if "image" in rel.target_ref:
				if not os.path.exists(output_path):
					os.makedirs(output_path)
				img_name = re.findall("/(.*)", rel.target_ref)[0]
				word_name = os.path.splitext(docx_file)[0]
				if os.sep in word_name:
					new_name = word_name.split('\\')[-1]
				else:
					new_name = word_name.split('/')[-1]
				img_name = f"{new_name}_{img_name}"
				img_path = os.path.join(output_path, img_name)
				with open(img_path, "wb") as f:
					f.write(rel.target_part.blob)
	except Exception as e:
		print(e)


def main():
	data_dir = r"data"
	docx_file = os.path.join(data_dir, r"docx", r"test.docx")
	output_path = os.path.join(data_dir, r"font_input")

	get_pictures(docx_file, output_path)


if __name__ == '__main__':
	main()
