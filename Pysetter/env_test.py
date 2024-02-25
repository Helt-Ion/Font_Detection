import torch


def env_test():
	import cv2
	import matplotlib
	import torchsummary
	import imageio
	import docx
	import django
	print("Success")


def cuda_test():
	cuda_is_available = torch.cuda.is_available()
	print(f"Cuda is available: {cuda_is_available}")
	return cuda_is_available


if __name__ == "__main__":
	print("Environment test:")
	env_test()
	cuda_test()
