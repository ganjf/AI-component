import torch
import numpy as np

""" data operation"""

def label_smoothing(label, K, epsilon):
	"""
	one-hot label（分类任务）:
	对于给定的数据集的label,将正例设为1，负例设为0。
	会产生模型对于标签过于依赖，当出现噪声的时候，会导致训练结果出现偏差。
	同时，在数据量分布不均衡时，过度依赖某一类数量多的标签，容易导致过拟合。
	label smooth是将label的真实标签q(y|x)和独立分布u(y)做一些混合.
	q'(y|x) = (1 - ε) * q(y|x) + ε * u(y), ε = 0.1.
	"""
	return (1 - epsilon) * label + epsilon * torch.full_like(label.size(), K)


def mix_training(x1, y1, x2, y2):
	"""
	（分类任务中）数据增强的一种方法，感觉和label smoothing相似，
	λ是从Beta(α，α)分布中随机产生，合成两个样本，论文中α=0.2，收敛所需训练epoch增大，
	只用到新的样本；
	在工业类缺陷检测或者违禁物品检测中，常常会给出一些不含有待检测目标的正常图像，
	可以将含有目标的图像和随机选取的正常图像进行 mixup（随机意味着更多的组合~），
	这样数据量又上来了。就是跟 coco 的数据集进行 mixup
	"""
	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


