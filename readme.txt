Simple neural net to recognize hand written digits

Library required:

	Cupy: https://docs-cupy.chainer.org/en/stable/install.html

	python-mnist: https://pypi.org/project/python-mnist/

2 Verison:
	Neural network model using the Sigmoid activation function: trainsig.py; testsig.py
		Training set accuracy: 96%
		Testing set accuracy: 93%
	Neural network model using the ReLu activation function and Sotmax for output: train.py; test.py
		Training set accuracy: 97%
		Testing set accuracy: 94%