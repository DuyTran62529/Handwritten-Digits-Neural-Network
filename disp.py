from matplotlib import pyplot as plt
import numpy as np

def display_im(im_ar, lab_ar, im):

	a = np.zeros((28,28))

	print('%s' % lab_ar[im])

	for i in range(783):
		train_tmp = im_ar[im][i]/255
		a[i//28][i%28] = train_tmp
		
	plt.imshow(a, cmap='gray', interpolation = 'hamming')
	plt.show()