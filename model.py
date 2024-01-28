import torch.nn as nn
import torch.nn.functional as F

#architecture (out_channel, kernel_size, stride)
#일단 padding값을 적용하지 않음.
in_channel = 3

# PreTrain Model Architecture
#논문에 나와있는 구조에서 20번째 conv Layer 까지만 반영함
architecture = [
	(64, 7, 2),
	"M",
	(192, 3, 1),
	"M",
	(128, 1, 1),
	(256, 3, 1),
	(256, 1, 1),
	(1024, 3, 1),
	"M",
	[(256, 1, 1),(512, 3, 1),4],
	(512, 1, 1),
	(1024, 3, 1),
	"M",
	[(512, 1, 1),(1024, 3, 1),2],
]

# Yolo v1 Model Architecture
architecture_YOLO = [
	(1024, 3, 1),
	(1024, 3, 2),
	(1024, 3, 1),
	(1024, 3, 1),
]

class PreTrained(nn.Module):
	def __init__(self):
		super(PreTrained, self).__init__()
		self.architecture = architecture
		self.in_channel = in_channel
		self.conv = self.createModel(self.architecture)

	def forward(self, x):
		out = self.conv(x)
		return out

	def createModel(self, architecture): #모델 설계
		in_channel = self.in_channel
		model = nn.Sequential()
		convN = 'conv' #Conv Layer 이름 설정용
		maxPN = 'maxP' #Maxpool Layer 이름 설정용
		for x in architecture:
			if type(x) == tuple:
				model.add_module(convN, nn.Conv2d(in_channel, x[0], kernel_size=x[1], stride=x[2]))
				model.add_module('leakyRelu', nn.LeakyReLU(0.1))
			elif type(x) == list:
				for order in range(x[2]):
					model.add_module(convN, nn.Conv2d(in_channel, x[0][0], kernel_size=x[0][1], stride=x[0][2]))
					model.add_module('leakyRelu', nn.LeakyReLU(0.1))
					model.add_module(convN, nn.Conv2d(in_channel, x[1][0], kernel_size=x[1][1], stride=x[1][2]))
					model.add_module('leakyRelu', nn.LeakyReLU(0.1))
			elif type(x) == str:
				model.add_module(maxPN, nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
		return model

class YOLOv1(nn.Module):
	def __init__(self, PreTrainedModel, **kwargs):
		super(YOLOv1, self).__init__()
		self.backbone = PreTrainedModel
		self.architecture = architecture_YOLO
		self.in_channel = in_channel
		self.conv = self.create_model(self.architecture)
		self.fc = self.create_fc(**kwargs)

	def forward(self, x):
		out = self.backbone(x)
		out = self.conv(out)
		return self.fc(out)
	
	def create_model(self):
		model = nn.Sequential()
		convN = 'conv' # Conv Layer 이름 설정용

		for x in self.architecture:
			if type(x) == tuple:
				model.add_module(convN, nn.Conv2d(in_channel, x[0], kernel_size=x[1], stride=x[2]))
				model.add_module('leakyRelu', nn.LeakyReLU(0.1))
			elif type(x) == list:
				for _ in range(x[2]):
					model.add_module(convN, nn.Conv2d(in_channel, x[0][0], kernel_size=x[0][1], stride=x[0][2]))
					model.add_module('leakyRelu', nn.LeakyReLU(0.1))
					model.add_module(convN, nn.Conv2d(in_channel, x[1][0], kernel_size=x[1][1], stride=x[1][2]))
					model.add_module('leakyRelu', nn.LeakyReLU(0.1))
		return model	

	def create_fc(self, split_size, num_boxes, num_classes):
		S, B, C = split_size, num_boxes, num_classes
		linear = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=1024 * S * S, out_features=4096),
			nn.Dropout(),
			nn.LeakyReLU(),
			nn.Linear(in_features=4096, out_features=S * S * (B * 5 + C))
		)
		return linear
