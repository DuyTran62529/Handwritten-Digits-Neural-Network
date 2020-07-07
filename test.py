from mnist import MNIST
from disp import display_im
import cupy as cp
from math import sqrt, log
import time

mndata = MNIST('./data')
test_im, test_lab = mndata.load_testing()

print('Test: im=%s, lab=%s' % (len(test_im), len(test_lab)))

ipn = len(test_im[0])
l1n = 32
l2n = 32
l3n = 32
l4n = 32
opn = 10

display_im(test_im, test_lab,5)

ip = cp.ones((len(test_im),ipn+1))
ip[:,:-1] = cp.array(test_im)
ip = ip / 255

layer1 = cp.ones((len(test_im),l1n+1))
layer2 = cp.ones((len(test_im),l2n+1))
layer3 = cp.ones((len(test_im),l3n+1))
layer4 = cp.ones((len(test_im),l4n+1))
res = cp.zeros((len(test_lab),opn))

wnb1 = cp.load('wb1relu.npy')
wnb2 = cp.load('wb2relu.npy')
wnb3 = cp.load('wb3relu.npy')
wnb4 = cp.load('wb4relu.npy')
wnb5 = cp.load('wb5relu.npy')

#Populate results
for i in range(len(test_lab)):
	res[i][test_lab[i]] = 1

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

print(100 * cp.sum(cp.argmax(op,axis = 1) == cp.argmax(res,axis = 1)) / len(test_lab))