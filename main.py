import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Paths
ANC_PATH = os.path.join('data', 'anchor')
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')

anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(300)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data) * .7))
test_data = test_data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def make_embedding():
    input = Input(shape=(100, 100, 3), name='input_image')

    x = Conv2D(64, (10,10), activation='relu')(input)
    x = MaxPooling2D(64, (2,2), padding='same')(x)

    x = Conv2D(128, (7,7), activation='relu')(x)
    x = MaxPooling2D(64, (2,2), padding='same')(x)

    x = Conv2D(128, (4,4), activation='relu')(x)
    x = MaxPooling2D(64, (2,2), padding='same')(x)

    x = Conv2D(256, (4,4), activation='relu')(x)
    x = Flatten()(x)

    x = Dense(4096, activation='sigmoid')(x)

    return Model(inputs=[input], outputs=[x], name='embedding')

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    embedding = make_embedding()

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')