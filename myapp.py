#!/usr/bin/env python3

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.reduce_sum(tf.random.normal([1000, 1000])))

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

batch_size = 32
img_height = 180
img_width = 180


