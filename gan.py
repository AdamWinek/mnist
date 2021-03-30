from generator import make_generator_model, generator_loss
from classifier import classifier_loss, make_classifier_model
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import imageio
import numpy as np
import os
import PIL
import time

from IPython import display


# Construct a tf.data.Dataset
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Reshape training and testing image
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


# get the generator and classifier
classifier = make_classifier_model()
generator = make_generator_model()

class_optimizer = tf.keras.optimizers.Adam(1e-4)
gen_optimizer = tf.keras.optimizers.Adam(1e-4)

BATCH_SIZE = 256
BUFFER_SIZE = 100
num_examples_to_generate = 16
noise_dim = 100
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train_data = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./class_first/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # apply the classifier to the
        real_output = classifier(images, training=True)
        fake_output = classifier(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        class_loss = classifier_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_classifier = disc_tape.gradient(
        class_loss, classifier.trainable_variables)

    gen_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    class_optimizer.apply_gradients(
        zip(gradients_of_classifier, classifier.trainable_variables))


@tf.function
def train_step_only_classifier(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # apply the classifier to the
        real_output = classifier(images, training=True)
        fake_output = classifier(generated_images, training=True)

        class_loss = classifier_loss(real_output, fake_output)

    gradients_of_classifier = disc_tape.gradient(
        class_loss, classifier.trainable_variables)

    class_optimizer.apply_gradients(
        zip(gradients_of_classifier, classifier.trainable_variables))


def train(dataset, epochs):

    for epoch in range(0, epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


train(train_data, 300)
