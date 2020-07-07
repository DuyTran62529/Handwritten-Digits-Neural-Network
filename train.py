from mnist import MNIST
from disp import display_im
import cupy as cp
from math import sqrt, log
import time

mndata = MNIST('./data')
train_im, train_lab = mndata.load_training()

print('Train: im=%s, lab=%s' % (len(train_im), len(train_lab)))

alph = 0.1
ipn = len(train_im[0])
l1n = 64
l2n = 64
l3n = 64
l4n = 64
opn = 10

# display_im(train_im, train_lab,5)

ip = cp.ones((len(train_im),ipn+1))
ip[:,:-1] = cp.array(train_im)
ip = ip /255

layer1 = cp.ones((len(train_im),l1n+1))
layer2 = cp.ones((len(train_im),l2n+1))
layer3 = cp.ones((len(train_im),l3n+1))
layer4 = cp.ones((len(train_im),l4n+1))
res = cp.zeros((len(train_lab),opn))

# w0 = cp.random.normal(0.5, 0.15, (ipn,opn))
w1 = cp.random.random((ipn,l1n))/(ipn+1)
w2 = cp.random.random((l1n,l2n))/(l1n+1)
w3 = cp.random.random((l2n,l3n))/(l2n+1)
w4 = cp.random.random((l3n,l4n))/(l3n+1)
w5 = cp.random.random((l4n,opn))/(l4n+1)

# wnb0 = cp.zeros((ipn+1,opn))
wnb1 = cp.zeros((ipn+1,l1n))
wnb2 = cp.zeros((l1n+1,l2n))
wnb3 = cp.zeros((l2n+1,l3n))
wnb4 = cp.zeros((l3n+1,l4n))
wnb5 = cp.zeros((l4n+1,opn))

# wnb0[:-1,:] = w0
wnb1[:-1,:] = w1
wnb2[:-1,:] = w2
wnb3[:-1,:] = w3
wnb4[:-1,:] = w4
wnb5[:-1,:] = w5

#Populate results
for i in range(len(train_lab)):
	res[i][train_lab[i]] = 1

newCost = 2
limit = 0.06

while (newCost > limit):

	#layer1
	layer1_l = cp.matmul(ip,wnb1)
	layer1_val = cp.maximum(0,layer1_l)
	layer1[:,:-1] = layer1_val

	#layer2
	layer2_l = cp.matmul(layer1,wnb2)
	layer2_val =  cp.maximum(0,layer2_l)
	layer2[:,:-1] = layer2_val

	#layer3
	layer3_l = cp.matmul(layer2,wnb3)
	layer3_val = cp.maximum(0,layer3_l)
	layer3[:,:-1] = layer3_val

	#layer4
	layer4_l = cp.matmul(layer3,wnb4)
	layer4_val = cp.maximum(0,layer4_l)
	layer4[:,:-1] = layer4_val

	#Output
	op_l = cp.matmul(layer4,wnb5)
	opmax = cp.amax(op_l,axis = 1, keepdims = True)
	op = cp.exp(op_l - opmax) / cp.sum(cp.exp(op_l - opmax), axis = 1, keepdims = True)

	# op_l = cp.matmul(ip,wnb0)
	# op = 1 / (1 + cp.exp(-op_l))

	#Cost calculation

	tmpop = op
	tmpop[tmpop == 0] = 0 + 0.000000001
	tmpop[tmpop == 1] = 1 - 0.000000001

	newCost = -(cp.sum (cp.multiply(res, cp.log(tmpop)))) / len(train_lab)
	
	if (newCost < 3 and newCost > 2.8):
		print(newCost)
	if (newCost < 1 and newCost > 0.8):
		print(newCost)
	if (newCost < 0.3):
		print(newCost)

	if (newCost <= limit):
		cp.save('wb1relu.npy', wnb1)
		cp.save('wb2relu.npy', wnb2)
		cp.save('wb3relu.npy', wnb3)
		cp.save('wb4relu.npy', wnb4)
		cp.save('wb5relu.npy', wnb5)

	# tmp1 = (op - res) / len(train_lab)
	# iptrans = cp.transpose(ip)
	# dCdw0 = cp.matmul(iptrans,tmp1)

	# O-4 Gradient descent
	tmp1 = (op - res) / len(train_lab)
	layer4trans = cp.transpose(layer4)
	dCdw5 = cp.matmul(layer4trans,tmp1) 

	#4-3 Gradient descent
	wnb5trans = cp.transpose(wnb5)
	dCdz4 = cp.matmul(tmp1,wnb5trans)[:,:-1]
	dz4dl4 = cp.sign(layer4)[:,:-1]
	dCdl4 = cp.multiply(dCdz4,dz4dl4)
	layer3trans = cp.transpose(layer3)
	dCdw4 = cp.matmul(layer3trans,dCdl4)

	#3-2 Gradient descent
	wnb4trans = cp.transpose(wnb4)
	dCdz3 = cp.matmul(dCdl4,wnb4trans)[:,:-1]
	dz3dl3 = cp.sign(layer3)[:,:-1]
	dCdl3 = cp.multiply(dCdz3,dz3dl3)
	layer2trans = cp.transpose(layer2)
	dCdw3 = cp.matmul(layer2trans,dCdl3)

	#2-1 Gradient descent
	wnb3trans = cp.transpose(wnb3)
	dCdz2 = cp.matmul(dCdl3,wnb3trans)[:,:-1]
	dz2dl2 = cp.sign(layer2)[:,:-1]
	dCdl2 = cp.multiply(dCdz2,dz2dl2)
	layer1trans = cp.transpose(layer1)
	dCdw2 = cp.matmul(layer1trans,dCdl2)

	#1-I Gradient desent
	wnb2trans = cp.transpose(wnb2)
	dCdz1 = cp.matmul(dCdl2,wnb2trans)[:,:-1]
	dz1dl1 = cp.sign(layer1)[:,:-1]
	dCdl1 = cp.multiply(dCdz1,dz1dl1)
	iptrans = cp.transpose(ip)
	dCdw1 = cp.matmul(iptrans,dCdl1)

	# wnb0 = wnb0 - alph*dCdw0
	wnb1 = wnb1 - alph*dCdw1
	wnb2 = wnb2 - alph*dCdw2
	wnb3 = wnb3 - alph*dCdw3
	wnb4 = wnb4 - alph*dCdw4
	wnb5 = wnb5 - alph*dCdw5

print(100 * cp.sum(cp.argmax(op,axis = 1) == cp.argmax(res,axis = 1)) / len(train_lab))