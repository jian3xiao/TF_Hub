# 来自 书《简明的TensorFlow 2》
# tensorflow2使用tfhub，tfhub版本大于等于0.5.0，使用的形式是hub.load()
# tensorflow13.1及以下可以使用，tfhub版本0.5.0以下，使用的形式是hub.Module()
# 从tfhub网站上下载模型时要注意是v1版本还是v2版本的
# 此脚本在环境py36tf20下，测试tfhub v2
# 同时要切换环境变量，使用CUDA-10.1版本
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # 2.1
import tensorflow_hub as hub  # 0.7.0

# gpus = tf.config.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)  # 设置按需要分配GPU资源，不设置就会报错！重启电脑错误消失！


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image_local(image_path, image_size=(512, 512), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_image(image, title, save=False, fig_dpi=300):
    plt.imshow(image, aspect='equal')
    plt.axis('off')
    if save:
        plt.savefig(title + '.png', bbox_inches='tight', dpi=fig_dpi, pad_inches=0.0)
    else:
        plt.show()


content_image_path = "contentimg.jpeg"
style_image_path = "styleimg.jpeg"

content_image = load_image_local(content_image_path)
style_image = load_image_local(style_image_path)

show_image(content_image[0], "Content Image")
show_image(style_image[0], "Style Image")

# Load image stylization module.
hub_module = hub.load('2')  # hub_module = hub.load('https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/2')

# Stylize image.
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))  # content_image与tf.constant()之后的都是tensor，且都能正常运行得到结果
stylized_image = outputs[0]

show_image(stylized_image[0], "Stylized Image", True)
