import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import numpy as np


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(80*35*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization()) # Normalize and scale inputs or activations. See remark bellow
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((35, 80, 256)))

    model.add(layers.Conv2DTranspose(256, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, 1, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, 3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(8, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 1, strides=1, padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(8, 1, strides=1, padding='same', input_shape=[140, 320, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(16, 3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(32, 5, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, 3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 5, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, 3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())

    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.random.uniform(real_output.shape, 0.9, 1), real_output)
    fake_loss = cross_entropy(tf.random.uniform(fake_output.shape, 0, 0.1), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.random.uniform(fake_output.shape, 0.9, 1), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # noise = seed
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, gen_loss


def train(epochs):
    for epoch in range(epochs):
        start = time.time()
        current_disc_loss = 0
        current_gen_loss = 0
        print("Starting epoch number %d" % (epoch + 1))
        for i in range(len(data_it)):
            start_batch = time.time()
            image_batch = data_it.next()
            disc_loss, gen_loss = train_step(image_batch)
            current_disc_loss += disc_loss
            current_gen_loss += gen_loss
            print(f'epoch {epoch + 1}, batch [{i + 1}/{len(data_it)}] took {time.time() - start_batch}s')
            print("disc_loss: " + str(disc_loss))
            print("gen_loss: " + str(gen_loss))
            print('---------------------------')

        # Produce images for the GIF as we go

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        disc_losses.append(current_disc_loss / len(data_it))
        gen_losses.append(current_gen_loss / len(data_it))
        print('Generator loss for epoch {} is {}'.format(epoch + 1, gen_losses[-1]))
        print('Discriminator loss for epoch {} is {}'.format(epoch + 1, disc_losses[-1]))
        generate_and_save_images(generator, epoch + 1, seed)
        print('Saved Images :)')

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed)
    plot_loss(disc_losses, gen_losses)


def plot_loss(d_loss, g_loss):
    try:
        plt.plot(range(1, len(d_loss) + 1), d_loss, c='blue')
        plt.plot(range(1, len(g_loss) + 1), g_loss, c='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Discriminator', 'Generator'], loc='upper right')
        plt.show()
    except:
        pass


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(32, 16))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) * 0.5)
        plt.axis('off')
    try:
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    except:
        pass
    plt.close(fig)


BATCH_SIZE = 32
EPOCHS = 400
noise_dim = 100
num_examples_to_generate = 16
datagen = ImageDataGenerator(preprocessing_function=(lambda img: (img / 127.5) - 1))
data_it = datagen.flow_from_directory('../fishDataSets/', target_size=(140, 320), class_mode=None, batch_size=BATCH_SIZE, )
disc_losses = []
gen_losses = []

if os.path.exists('./seed.npy'):
    seed = np.load('./seed.npy')
else:
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    np.save('./seed.npy', seed)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-7)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-7)

generator = make_generator_model()
discriminator = make_discriminator_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(EPOCHS)
# generate_and_save_images(generator,
#                          0000,
#                          seed)
