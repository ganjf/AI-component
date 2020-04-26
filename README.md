# Learning-Trick

This repo was inspired by Paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)


## tricks

tricks I've tested, some of them were from the Paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) :

|trick|referenced paper|
|:---:|:---:|
|xavier init|[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)|
|warmup training|[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677v2)|
|no bias decay|[Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205vx)|
|label smoothing|[Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567v3))|
|random erasing|[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2)|
|cutout|[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2)|
|linear scaling learning rate|[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677v2)|
|cosine learning rate decay|[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)|

**and more to come......**

## The scenario in which the technique is applied

### xvaier initiazation and He initialization
1. Xavier Glorot et al., Understanding the Difficult of Training Deep Feedforward Neural Networks 

2. Kaiming He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classfication 

KeyPoint:  
The variance of the activation value decreases layer by layer, which leads to the gradient in the back propagation also decreases layer by layer. To solve the gradient disappearance, we should avoid the attenuation of the variance of the activation value, and ideally, the output value of each layer (the activation value) should keep the gaussian distribution.  
$y = Wx+b, E(x_i)=0 and E(W_i)=0$, in order to make Var(y) = Var(x)=N*Var(W_i)*Var(x_i), so that Var(w_i) = 1/N. Because of $std = \sqrt {Var}$, std(W_i)=1/\sqrt N$.  
**The problem with Xavier initialization is that it only applies to linear activation functions **
