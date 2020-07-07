from mnist import MNIST
from disp import display_im
import cupy as cp
from math import sqrt, log
import time

mndata = MNIST('./data')
test_im, test_lab = mndata.load_testing()

print('Test: im=%s, lab=%s' % (len(test_im), len(test_lab)))

ipn = len(test_im[0])
l1n = 100
l2n = 100
l3n = 100
opn = 10

display_im(test_im, test_lab,5)

ip = cp.ones((len(test_im),ipn+1))
ip[:,:-1] = cp.array(test_im)
ip = ip

layer1 = cp.ones((len(test_im),l1n+1))
layer2 = cp.ones((len(test_im),l2n+1))
layer3 = cp.ones((len(test_im),l3n+1))
res = cp.zeros((len(test_lab),opn))

wnb1 = cp.load('wb1.npy')
wnb2 = cp.load('wb2.npy')
wnb3 = cp.load('wb3.npy')
wnb4 = cp.load('wb4.npy')

#Populate results
for i in range(len(test_lab)):
	res[i][test_lab[i]] = 1

#layer1
layer1_l = cp.matmul(ip,wnb1)
layer1_val = 1/(1+cp.exp(-(layer1_l)))
layer1[:,:-1] = layer1_val

#layer2
layer2_l = cp.matmul(layer1,wnb2)
layer2_val =  1/(1+cp.exp(-(layer2_l)))
layer2[:,:-1] = layer2_val

#layer3
layer3_l = cp.matmul(layer2,wnb3)
layer3_val = 1/(1+cp.exp(-(layer3_l)))
layer3[:,:-1] = layer3_val

#Output
op_l = cp.matmul(layer3,wnb4)
op = 1/(1+cp.exp(-(op_l)))

print(100 * cp.sum(cp.argmax(op,axis = 1) == cp.argmax(res,axis = 1)) / len(test_lab))