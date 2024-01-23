# Knowledge_Distillation

This repository is my reproduction of this article 'Distilling the Knowledge in a Neural Network', which is divided into two parts:

## 1.mnist.py

This file achieves a knowledge distillation structure with MNIST database.

The test accuracy is up to 91%.

## 2.teacher.py, student.py and student_noise.py

It is based primarily on CIFAR10 data sets.

### (1) teacher.py

You can set 'download=False' in 'torchvision.datasets.CIFAR10()' to 'True' in order to download this datasets. And when you start to train this network, you can restore it.

You can uncomment the line 'torch.save(net, './saved_pkl/teacher.pkl')' to save the trained parameters so you can use them next time or student network.

### (2) student.py

It will directly use the trained parameters by teeacher networ. You can arbitrarily set some hyper-parameter, such as temperature T and weight $\alpha$, which will be used as follows:

$$
q_i=\frac{exp(z_i/T)}{\sum_j exp(z_j/T)}
$$

Where, $q_i$ denotes the final output with softmax, $z_j$ denotes the network node result.

And the final train result will obtain by the following method:

$$
r_i = \alpha q_i + (1-\alpha) p_i
$$

where, $p_i$ denotes one-hot, such as [0 1 0 0 0 ] etc.

### (3) student_noise.py

In order to improve the training and prediction accuracy of the student network, different degrees of = noise, including random Gaussian noise and random flip, are added to the image when the training set is input, and the others are exactly the same as the 'student.py' file.

The framework of these files is as follows:

![image](https://github.com/Yaepiii/Knowledge_Distillation/assets/75295024/5f69fc9e-6559-4e60-b25b-bb4848ffd40d)





