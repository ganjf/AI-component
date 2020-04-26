import torch
import torch.nn as nn

"""
no bias decay

The weight deacy is often applied to all learnable parameters including both weights and bias.
It's recommended to only apply the regularization to weights to avoid overfitting.
So that only apply the weight decay to the weights in convolution and fully-connected layers. 
There is either no decay with bias in BN layers. 

Usage:
	params = add_weight_decay(model, 5e-4)
	optimizer = torch.optim.SGD(params, lr=0.1)
"""

def add_weight_decay(net, weight_decay):
	decay = list()
	no_decay = list()
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			decay.append(m.weight)
			if m.bias is not None:
				no_decay.append(m.bias)
		else:
			if hashattr(m, 'weight'):
				decay.append(m.weight)
			if hashattr(m, 'bias'):
				no_decay.append(m.bias)
	assert(len(list(net.parameters())) == len(decay) + len(no_decay))
	return [{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': weight_decay}]


