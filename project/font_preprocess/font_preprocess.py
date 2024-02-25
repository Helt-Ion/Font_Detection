import os
import cv2

data_path = r"data"
input_img_path = os.path.join(data_path, "input", "SimFang.png")
output_img_path = os.path.join(data_path, "output", "SimFang.png")


def main():
	print(f"cv2.IMREAD_GRAYSCALE={cv2.IMREAD_GRAYSCALE}")
	print(f"Reading image from \"{input_img_path}\"...")
	input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
	_, output_img = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)
	print("Showing image...")
	cv2.imshow("Showing image", output_img)
	cv2.waitKey(0)
	print(f"Writing image to \"{output_img_path}\"...")
	cv2.imwrite(output_img_path, output_img)


if __name__ == '__main__':
	main()
