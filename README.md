# Class2Str: End to End Latent Hierarchy Learning

This repository contains the code for our [ICPR 2018 (Beijing, China)](http://www.icpr2018.org/) Paper:

[Class2Str: End to End Latent Hierarchy Learning](https://arxiv.org/abs/1808.06675)

## Citation

Please consider citing our work, if you find it useful in your research:

```
@article{saha2018class2str,
  title={Class2Str: End to End Latent Hierarchy Learning},
  author={Saha, Soham and Varma, Girish and Jawahar, CV},
  journal={arXiv preprint arXiv:1808.06675},
  year={2018}
}
```
## Introduction

Deep neural networks for image classification typically consists of a convolutional feature extractor followed by a fully connected classifier network. The predicted and the ground truth labels are represented as one hot vectors. Such a representation assumes that all classes are equally dissimilar. However, classes have visual similarities and often form a hierarchy. Learning this latent hierarchy explicitly in the architecture could provide invaluable insights. We propose an alternate architecture to the classifier network called the Latent Hierarchy (LH) Classifier and an end to end learned Class2Str mapping which discovers a latent hierarchy of the classes. We show that for some of the best performing architectures on CIFAR and Imagenet datasets, the proposed replacement and training by LH classifier recovers the accuracy, with a fraction of the number of parameters in the classifier part. 

## Dependencies

- [Python 2.7](https://www.python.org/)
- [Tensorflow 0.1.0](https://www.tensorflow.org/install/pip?lang=python2)

## Usage

- `lib.py` is a wrapper for tensorflow functions and makes them easy to use.

- There are separate files for preparing the datasets in the required format namely `cifar10.py, cifar100.py, mnist.py` etc.

- For each dataset, the hierarchy is generated in 2 stages. First the conventional classifier is trained using LeNet or VGG16 with Batch Normalization. The the latent hierarchy classifier is used to fine tune the model after removing the fully connected layer.


For running on CIFAR 10, first create a directory name `cifar_10` and dump the CIFAR 10 data in batches of 10000 in the following format:

```
--- cifar_10
  --- data_batch_1
  --- data_batch_2
  --- ...
  --- test_batch

```

First the FC Classifier needs to trained. An example is as under:

```
python train_fc_classifier.py --B=256 --dataset cifar10 --name exp1 --model vgg16
```
Here B is the batch size and name can be any name given to the run.

This is followed by training the LH Classifier along with the other parts of the network detailed in our paper.

```
python train_rnn_classifier_cifar.py --B=200 --dataset cifar10 --name exp1 --model vgg16 --w_c=4 --w_i=100 --w_l2=0.1
```

Here the weights for the different components in our proposed loss are the most sensitive. Above is an example of the parameters to train the LH Classifier. 

- `w_c` is the weight given to cross entropy
- `w_i` is the weight given for inverse hash
- `w_l2` is the weight given for l2 regularization of the weights

### Contact

Please contact us at :

- sohamsaha\[dot]cs\[at]gmail.com
- soham\[dot]saha\[at]research\[dot]iiit\[dot]ac\[dot]in


