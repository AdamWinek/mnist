import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
import os


def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False,
                           input_shape=(100,), activation="relu"))

    model.add(layers.Reshape((7, 7, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation="relu"))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation="relu"))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def pixel_difference(true, prediction):
    subbed = tf.math.abs(tf.math.subtract(true, prediction))
    return tf.math.log_sigmoid(tf.math.reduce_sum(subbed))


def main():

    # Construct a tf.data.Dataset
    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape training and testing image
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    model = make_generator_model()

    checkpoint_path = f"./checkpoints/generator" + "/{epoch:02d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    noise = tf.random.normal([1, 100])

    train_noise = []
    for i in range(0, len(train_images)):
        train_noise.append(tf.random.normal([1, 100]))

    test_noise = []
    for i in range(0, len(test_images)):
        test_noise.append(tf.random.normal([1, 100]))

    generated_image = model(noise, training=False)

    #plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    # plt.show()
    print("here")
    model.compile(optimizer='adam',
                  loss=pixel_difference,
                  metrics=['accuracy'])

    print(model.summary())
    print("here1")

    history = model.fit(np.array(train_noise), train_images,
                        epochs=10, verbose=1, validation_data=(np.array(test_noise), test_images), callbacks=[cp_callback])

    generated_image = model(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()
