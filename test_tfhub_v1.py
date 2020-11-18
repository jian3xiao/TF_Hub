# 此脚本在环境tf131py36下，测试tfhub v1
# 同时要切换环境变量，使用CUDA-10.0版本
# ResNet architecture with non-saturating loss and spectral normalization on CIFAR-10
# 从hub.tensorflow上下载的官方训练模型  https://hub.tensorflow.google.cn/google/compare_gan/model_13_cifar10_resnet_cifar/1
# Details
#
#     Dataset: CIFAR-10
#     Model: Non-saturating GAN
#     Architecture: ResNet CIFAR
#     Optimizer: Adam (lr=2.000e-04, beta1=0.500, beta2=0.999)
#     Discriminator iterations per generator iteration: 5
#     Discriminator normalization: Spectral normalization
#     Discriminator regularization: none
#
# Scores
#
#     FID: 22.91
#     Inception: 7.74
#     MS-SSIM: N/A

import tensorflow as tf  # 13.1
import tensorflow_hub as hub  # 0.1.1
from PIL import Image
import numpy as np
import os


def imwrite(filename, np_image, count):
    # numpy, float32
    num = np_image.shape[0]
    channel = np_image.shape[3]
    if channel == 1:  
        np_image = np.concatenate([np_image, np_image, np_image], 2)  # 在通道上连接
    for j in range(num):
        img_ = np_image[j, :, :, :].astype(np.float64)
        im = (img_ * 255).astype(np.uint8)
        img = Image.fromarray(im)
        all_count = 64*count + j
        img_name = '{:07d}.png'.format(all_count)
        img.save(os.path.join(filename, img_name))


with tf.Graph().as_default():
    sess = tf.Session()
    gan = hub.Module("1")  # gan = hub.Module("https://hub.tensorflow.google.cn/google/compare_gan/model_15_cifar10_resnet_cifar/1")
    init = tf.global_variables_initializer()
    sess.run(init)
    save_dir = "tfhub_v1_test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_num = 10000
    for i in range(int(test_num // 64)):
        z_values = tf.random.uniform(minval=-1, maxval=1, shape=[64, 128])
        images = gan(z_values, signature="generator")

        z = np.random.uniform(-1, 1, [64, 128])
        fake = sess.run([images], feed_dict={z_values: z})
        img = fake[0]  # [64, 32, 32, 3]   (0, 1)
        imwrite(save_dir, img, i)
