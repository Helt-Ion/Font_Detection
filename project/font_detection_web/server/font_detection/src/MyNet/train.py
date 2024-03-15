import glob
import os
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import time

from dataset.dataset import data_get
from MyNet.ldbinet import Ldbinet
from modules import filereader

global_data_path = "data"


def run(data_path):
	fonts_path = os.path.join(data_path, "Fonts.txt")
	font_label, classes, font_list, font_bias = filereader.read_fonts(fonts_path)
	train_imgs_path = glob.glob(os.path.join(data_path, "train/*/*.*"))
	test_imgs_path = glob.glob(os.path.join(data_path, "test/*/*.*"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定采用cpu还是gpu计算
	model_load_dir = "MyNet/checkpoint"
	model_save_dir = "MyNet/models"
	input_model = ""
	output_model = "font_recognize_200.pth"
	model_load_path = None
	if input_model != "":
		model_load_path = os.path.join(model_load_dir, input_model)

	transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
	])
	batchsz = 32  # 定义批处理量
	data_train, data_train_loader, data_test, data_test_loader \
		= data_get(train_imgs_path, test_imgs_path, classes, batchsz, transform)

	model = Ldbinet(num_classes=len(classes))
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path))
	model.eval()
	model.to(device)
	# 4.训练目标：损失函数
	creation = CrossEntropyLoss()  # 损失函数：分类任务常用交叉熵函数，二分类任务可用nn.BCELoss()
	creation = creation.to(device)
	# 5.优化器
	learning_rate = 1e-4
	beta1 = 0.9
	beta2 = 0.999
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))  # 优化器
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)  # 学习率衰减：余弦退火策略

	os.makedirs(model_save_dir, exist_ok=True)
	epochs = 200
	train_steps = 0
	test_steps = 0
	print(len(data_train))
	# 用数组保存每一轮迭代中，训练的损失值和精确度，也是为了通过画图展示出来。
	train_losses = []
	train_acces = []
	# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
	eval_losses = []
	eval_acces = []

	start_time, p_time, c_time = time.time(), time.time(), 0.0
	for epoch in range(epochs):  # 0-99
		print(f"第{epoch + 1}轮训练过程：")
		# 训练步骤
		model.train()  # 训练模式
		epoch_train_loss = 0.0
		epoch_train_acc = 0.0
		for train_batch, (train_image, train_label) in enumerate(data_train_loader):
			# print(f"第{train_batch + 1}批数据进行训练，当前迭代第{epoch + 1}次")
			train_image, train_label = train_image.to(device), train_label.to(device)  # 将数据送到GPU
			train_predictions = model(train_image)

			batch_train_loss = creation(train_predictions, train_label)
			optimizer.zero_grad()  # 梯度清零
			batch_train_loss.backward()
			optimizer.step()

			epoch_train_loss += batch_train_loss.item()
			# 计算每个批次数据训练时的准确率
			_, predicted = torch.max(train_predictions.data, 1)
			batch_train_acc = (predicted == train_label).sum()
			epoch_train_acc += batch_train_acc

			train_steps += 1

			if train_steps % 50 == 0:
				if train_steps > 50:
					c_time += time.time() - p_time
				avg_time = c_time / (train_steps / 50 - 1) if train_steps > 50 else 0.0
				print(
					f"第{train_steps}次训练，训练损失为{batch_train_loss.item()}，耗时{time.time() - p_time}s，平均耗时{avg_time}s")
				p_time = time.time()
		# 结果存入列表
		train_losses.append(epoch_train_loss)
		train_acces.append(epoch_train_acc.cpu().numpy() / len(data_train))
		# 测试步骤
		model.eval()
		epoch_test_loss = 0.0
		epoch_test_acc = 0.0
		# 禁用自动求导
		with torch.no_grad():
			for test_batch, (test_image, test_label) in enumerate(data_test_loader):
				print(f"第{test_batch + 1}批数据进行测试")
				test_image, test_label = test_image.to(device), test_label.to(device)
				predictions = model(test_image)
				test_loss = creation(predictions, test_label)

				epoch_test_loss += test_loss.item()
				# 计算每个批次数据测试时的准确率
				_, predicted = torch.max(predictions.data, 1)
				batch_test_acc = (predicted == test_label).sum()
				epoch_test_acc += batch_test_acc

				test_steps += 1
				if test_steps % 50 == 0:
					print(f"第{test_steps}次测试，测试损失为{test_loss.item()}")

		print(
			f"第{epoch + 1}轮训练结束，训练损失为{epoch_train_loss}，测试损失为{epoch_test_loss}，测试准确率为{epoch_test_acc / len(data_test)}")

		model_save_path = os.path.join(model_save_dir, f"model_{epoch + 1}_{(epoch_test_acc / len(data_test)):.4f}.pth")
		torch.save(model.state_dict(), model_save_path)
		eval_losses.append(epoch_test_loss)
		eval_acces.append(epoch_test_acc.cpu().numpy() / len(data_test))

	# 7.保存模型
	model_save_path = os.path.join(model_save_dir, output_model)
	torch.save(model.state_dict(), model_save_path)
	end_time = time.time()
	print(f"训练结束！耗时{end_time - start_time}s")

	eval_save_path = os.path.join(model_save_dir, "eval.txt")
	with open(eval_save_path, 'w') as f:
		f.write(str(train_losses)[1: -1] + "\n")
		f.write(str(train_acces)[1: -1] + "\n")
		f.write(str(eval_losses)[1: -1] + "\n")
		f.write(str(eval_acces)[1: -1] + "\n")


# 绘图代码
# plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
# plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
# plt.plot(np.arange(len(eval_losses)), eval_losses, label="test loss")
# plt.plot(np.arange(len(eval_acces)), eval_acces, label="test acc")
# plt.legend()  # 显示图例
# plt.xlabel('epoches')
# plt.title('Model accuracy&loss')
# plt.show()


def main():
	run(global_data_path)


if __name__ == "__main__":
	main()
