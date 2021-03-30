import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np


def make_classifier_model():

    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    # flatten takes our layers and creates a long vector  from it
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation="softmax"))
    return model


def classifier_loss(real_output, fake_output):
    # difference in distributions between the predictions and the actual output
    real_loss = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def main():

    # Construct a tf.data.Dataset
    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(len(test_labels))
    print(len(test_images))

    print(len(train_labels))
    print(len(train_images))

    # Reshape training and testing image
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    # flatten takes our layers and creates a long vector  from it
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(train_images, train_labels,
                        epochs=10, verbose=1, validation_data=(test_images,  test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
