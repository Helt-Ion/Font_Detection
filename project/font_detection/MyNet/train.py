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
	fonts_path = os.path.join(data_path, "Fonts.xml")
	font_labels, font_classes, font_list = filereader.read_fonts(fonts_path)
	train_imgs_path = glob.glob(os.path.join(data_path, "train/**/*.*"), recursive=True)
	test_imgs_path = glob.glob(os.path.join(data_path, "test/**/*.*"), recursive=True)
	# Define model path
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_load_dir = "MyNet/checkpoint"
	model_save_dir = "MyNet/models"
	input_model = ""
	output_model = "font_recognize_200.pth"
	model_load_path = None
	if input_model != "":
		model_load_path = os.path.join(model_load_dir, input_model)
	os.makedirs(model_save_dir, exist_ok=True)

	transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
	])
	# Define batch size
	batch_size = 32
	data_train, data_train_loader, data_test, data_test_loader \
		= data_get(train_imgs_path, test_imgs_path, font_classes, batch_size, transform)
	# Define model
	model = Ldbinet(num_classes=len(font_classes))
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path))
	model.eval()
	model.to(device)
	# Define loss function
	creation = CrossEntropyLoss()
	creation = creation.to(device)
	# Define optimizer
	learning_rate = 1e-4
	beta1 = 0.9
	beta2 = 0.999
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
	epochs = 200
	train_steps = 0
	test_steps = 0
	# Show number of training images
	print(f"Number of training images: {len(data_train)}")
	# Saving training losses and accuracy for plotting
	train_losses = []
	train_acces = []
	# Saving testing losses and accuracy for plotting
	eval_losses = []
	eval_acces = []
	# Setting time
	start_time, p_time, c_time = time.time(), time.time(), 0.0
	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}:")
		# Set model to training mode
		model.train()
		epoch_train_loss = 0.0
		epoch_train_acc = 0.0
		for train_batch, (train_image, train_label) in enumerate(data_train_loader):
			# print(f"Training batch {train_batch + 1}, Epoch {epoch + 1}")
			# Send data to GPU
			train_image, train_label = train_image.to(device), train_label.to(device)
			train_predictions = model(train_image)
			# Calculate batch train loss
			batch_train_loss = creation(train_predictions, train_label)
			# Zero the gradient
			optimizer.zero_grad()
			batch_train_loss.backward()
			optimizer.step()
			# Calculate training loss for one epoch
			epoch_train_loss += batch_train_loss.item()
			# Calculate accuracy for training data
			_, predicted = torch.max(train_predictions.data, 1)
			batch_train_acc = (predicted == train_label).sum()
			epoch_train_acc += batch_train_acc
			# Finished one training step
			train_steps += 1
			# Show progress
			if train_steps % 50 == 0:
				if train_steps > 50:
					c_time += time.time() - p_time
				avg_time = c_time / (train_steps / 50 - 1) if train_steps > 50 else 0.0
				print(
					f"Training step: {train_steps}, Training loss: {batch_train_loss.item()}, "
					f"Time cost: {time.time() - p_time}s, Average time: {avg_time}s"
				)
				p_time = time.time()
		# Save results to list
		train_losses.append(epoch_train_loss)
		train_acces.append(epoch_train_acc.cpu().numpy() / len(data_train))
		# Set model to eval mode
		model.eval()
		epoch_test_loss = 0.0
		epoch_test_acc = 0.0
		# Disable automatic differentiation
		with torch.no_grad():
			for test_batch, (test_image, test_label) in enumerate(data_test_loader):
				print(f"Test batch {test_batch + 1}")
				test_image, test_label = test_image.to(device), test_label.to(device)
				predictions = model(test_image)
				test_loss = creation(predictions, test_label)
				# Calculate test loss
				epoch_test_loss += test_loss.item()
				# Calculate accuracy for testing data
				_, predicted = torch.max(predictions.data, 1)
				batch_test_acc = (predicted == test_label).sum()
				epoch_test_acc += batch_test_acc
				# Finished one testing step
				test_steps += 1
				if test_steps % 50 == 0:
					print(f"Test step: {test_steps}, Testing loss: {test_loss.item()}")
		# Show data for one epoch
		print(
			f"Epoch {epoch + 1}: Training loss: {epoch_train_loss}, "
			f"Testing loss: {epoch_test_loss}, Accuracy: {epoch_test_acc / len(data_test)}"
		)
		model_save_path = os.path.join(model_save_dir, f"model_{epoch + 1}_{(epoch_test_acc / len(data_test)):.4f}.pth")
		torch.save(model.state_dict(), model_save_path)
		eval_losses.append(epoch_test_loss)
		eval_acces.append(epoch_test_acc.cpu().numpy() / len(data_test))
	# Save model
	model_save_path = os.path.join(model_save_dir, output_model)
	torch.save(model.state_dict(), model_save_path)
	end_time = time.time()
	print(f"Training complete! Time cost: {end_time - start_time}s")
	# Save evaluation
	eval_save_path = os.path.join(model_save_dir, "eval.txt")
	with open(eval_save_path, 'w') as f:
		f.write(str(train_losses)[1: -1] + "\n")
		f.write(str(train_acces)[1: -1] + "\n")
		f.write(str(eval_losses)[1: -1] + "\n")
		f.write(str(eval_acces)[1: -1] + "\n")


# Code for plotting
# plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
# plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
# plt.plot(np.arange(len(eval_losses)), eval_losses, label="test loss")
# plt.plot(np.arange(len(eval_acces)), eval_acces, label="test acc")
# plt.legend()
# plt.xlabel('epoches')
# plt.title('Model accuracy&loss')
# plt.show()


def main():
	run(global_data_path)


if __name__ == "__main__":
	main()
