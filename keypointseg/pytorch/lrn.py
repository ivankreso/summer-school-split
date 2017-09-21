import torch
import numpy
import importlib
import time
import os

#
# parse command line options
#

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
parser.add_argument('dataloader', type=str, help='a script that loads training and validation samples')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned model weights')
parser.add_argument('--learnrate', type=float, default=1e-4, help='RMSprop learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')

args = parser.parse_args()

#
#
#

MODEL = importlib.import_module(args.modeldef).init()
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
MODEL.cuda()

#
#
#

print('* data loader: ' + args.dataloader)
load_batch = importlib.import_module(args.dataloader).get_loader(160)

def loss_forward(inputs, targets):
	#
	thr = 0.1
	return torch.nn.functional.relu(0.5*(inputs-targets).pow(2) - 0.5*thr**2).mean()

optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=args.learnrate)

for i in range(16384+1):
	#
	start = time.time()
	imgs, tgts = load_batch(args.batchsize)
	#print('* batch loaded in %f [s]' % (time.time() - start))
	#
	start = time.time()
	avgloss = 0
	optimizer.zero_grad()
	for j in range(0, len(imgs)):
		#
		img = torch.autograd.Variable(imgs[j].cuda())
		tgt = torch.autograd.Variable(tgts[j].cuda())
		#
		outputs = MODEL(img)
		loss = loss_forward(outputs, tgt)
		loss.backward()
		avgloss = avgloss + loss.data[0]
	optimizer.step()
	avgloss = avgloss/args.batchsize
	#print('* batch processed in %f [s]' % (time.time() - start))
	#
	print('* batch ' + str(i) + ' (average loss: ' + str(avgloss) + ')')
	if i%256 == 0 and i!=0 and args.writepath:
		#
		os.system('mkdir -p ' + args.writepath)
		#
		path = args.writepath + '/' + str(i) + '.pth'
		#
		print('* saving model weights to ' + path)
		torch.save(MODEL.state_dict(), path)