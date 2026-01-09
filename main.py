import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
        return tf.math.abs(input_embedding[0] - validation_embedding[0])

def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    embedding = make_embedding()

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

binary_cross_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]

        y_hat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, y_hat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

def train(data, epochs):
    for epoch in range(1, epochs + 1):
        for _, batch in enumerate(data):
            loss = train_step(batch)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} | Loss: {loss.numpy():.4f}')
            checkpoint.save(file_prefix=checkpoint_prefix)

train(train_data, epochs=50)

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    y_hat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, y_hat)
    p.update_state(y_true, y_hat)

print(r.result().numpy(), p.result().numpy())

plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.imshow(test_input[0])
plt.subplot(1, 2, 2)
plt.imshow(test_val[0])
plt.show()

siamese_model.save('siamese_model.h5')

# model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})