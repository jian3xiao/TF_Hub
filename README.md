# TensorFlow的使用下

内容主要来自书《简明的TensorFlow 2》，[TensorFlow Hub 模型复用](https://tf.wiki/zh_hans/appendix/tfhub.html)

## 使用说明
1.  test_tfhub.py 使用的就是书中提供的风格迁移的例子，可以正常运行。一点小疑问：为什么在输出时使用tf.constant() ？使用和不使用都能得到一样的结果。实验结果可以参考上面的网站

2. test_tfhub_v1.py 在TF-13.1的版本下使用Hub，以GANs生成图像为例。使用的Hub网站上提供的GANs模型是[ResNet CIFAR](https://hub.tensorflow.google.cn/google/compare_gan/model_13_cifar10_resnet_cifar/1)。但是运行TF-13.1的版本要注意更换CUDA和cuDNN版本。

下面展示生成的图像和真实的图像

 <div align="center">![img](fake_10000.png) 生成图像 </div>

 <div align="center">![img](real_10000.png) 真实图像 </div>


