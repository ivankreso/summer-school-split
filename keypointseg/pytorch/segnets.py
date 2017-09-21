import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TinySegNet(nn.Module):
	def __init__(self, nchannels, nlabels):
		#
		super(TinySegNet, self).__init__()
		#
		S = 24
		# downsampling layers
		self.dn1 = nn.Sequential(
			nn.Conv2d(nchannels, S, kernel_size=3, padding=1),
			nn.BatchNorm2d(S),
			nn.ReLU(),
			nn.Conv2d(S, S, kernel_size=3, padding=1),
			nn.BatchNorm2d(S),
			nn.ReLU()
		)
		self.dn2 = nn.Sequential(
			nn.Conv2d(S, 2*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(2*S),
			nn.ReLU(),
			nn.Conv2d(2*S, 2*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(2*S),
			nn.ReLU()
		)
		self.dn3 = nn.Sequential(
			nn.Conv2d(2*S, 3*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(3*S),
			nn.ReLU(),
			nn.Conv2d(3*S, 3*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(3*S),
			nn.ReLU()
		)
		self.dn4 = nn.Sequential(
			nn.Conv2d(3*S, 4*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(4*S),
			nn.ReLU(),
			nn.Conv2d(4*S, 4*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(4*S),
			nn.ReLU()
		)
		#
		self.centralconv = nn.Conv2d(4*S, 4*S, kernel_size=3, padding=1)
		# upsampling layers
		self.up1 = nn.Sequential(
			nn.Conv2d(4*S, 3*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(3*S),
			nn.ReLU(),
			nn.Conv2d(3*S, 3*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(3*S),
			nn.ReLU()
		)
		self.up2 = nn.Sequential(
			nn.Conv2d(3*S, 2*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(2*S),
			nn.ReLU(),
			nn.Conv2d(2*S, 2*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(2*S),
			nn.ReLU()
		)
		self.up3 = nn.Sequential(
			nn.Conv2d(2*S, 1*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(1*S),
			nn.ReLU(),
			nn.Conv2d(1*S, 1*S, kernel_size=3, padding=1),
			nn.BatchNorm2d(1*S),
			nn.ReLU()
		)
		self.up4 = nn.Sequential(
			nn.Conv2d(S, S, kernel_size=3, padding=1),
			nn.BatchNorm2d(S),
			nn.ReLU(),
			nn.Conv2d(S, nlabels, kernel_size=3, padding=1),
		)

	def forward(self, x):
		#
		sizes = []
		#
		sizes.append( (x.size(2), x.size(3)) )
		x = self.dn1(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		sizes.append( (x.size(2), x.size(3)) )
		x = self.dn2(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		sizes.append( (x.size(2), x.size(3)) )
		x = self.dn3(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		sizes.append( (x.size(2), x.size(3)) )
		x = self.dn4(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		#
		x = self.centralconv(x)
		#
		x = F.upsample(x, sizes[3], mode='bilinear')
		x = self.up1(x)
		x = F.upsample(x, sizes[2], mode='bilinear')
		x = self.up2(x)
		x = F.upsample(x, sizes[1], mode='bilinear')
		x = self.up3(x)
		x = F.upsample(x, sizes[0], mode='bilinear')
		x = self.up4(x)
		#
		return F.sigmoid(x)

class ResNet18(nn.Module):
	def __init__(self, nchannels, nlabels, pretrained=False):
		super(ResNet18, self).__init__()
		resnet = torchvision.models.resnet18(pretrained)
		#
		self.conv1 = resnet.conv1
		self.bn1 = resnet.bn1
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		#
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		#
		self.conv2 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(192, nlabels, kernel_size=3, padding=1)

	def forward(self, x):
		#
		nrows = x.size(2)
		ncols = x.size(3)
		#
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		#
		a1 = self.layer1(x)
		a2 = self.layer2(a1)
		a3 = self.layer3(a2)
		a4 = self.layer4(a3)
		#
		x = a4
		x = torch.nn.functional.upsample(x, (a3.size(2), a3.size(3)), mode='bilinear')
		x = self.conv2(torch.cat([x, a3], 1))
		x = torch.nn.functional.upsample(x, (a2.size(2), a2.size(3)), mode='bilinear')
		x = self.conv3(torch.cat([x, a2], 1))
		x = torch.nn.functional.upsample(x, (a1.size(2), a1.size(3)), mode='bilinear')
		x = self.conv4(torch.cat([x, a1], 1))
		#
		return F.sigmoid(torch.nn.functional.upsample(x, (nrows, ncols), mode='bilinear'))
#
def init():
	return TinySegNet(3, 3)

#
'''
net = init()
x = torch.autograd.Variable(torch.randn(4, 3, 63, 138))
y = net.forward(x)
print(y.size())
'''