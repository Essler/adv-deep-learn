import numpy as np
import tensorflow as tf
import datetime
import time
import imageio
import glob
import atexit

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from keras.datasets import mnist
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, Reshape, LeakyReLU, ReLU, BatchNormalization, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


np.random.seed(42)  # Ensure consistent results between runs
noise_dim = 100  # Length of random noise vector
batch_size = 16
steps = 5000 // batch_size
epochs = 200
img_rows, img_cols, channels = 28, 28, 1
img_dir = './images/gan'
date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # Datetime string used in output file names.
cross_entropy = BinaryCrossentropy()  # Loss function helper method.
optimizer = Adam()
discriminator_optimizer = Adam()
generator_optimizer = Adam()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0
train_images = train_images[np.where(train_labels == 0)[0]].reshape((-1, img_rows, img_cols, channels))


def get_my_discriminator():
    input_shape = (28, 28, 1)
    kernel_size = (5, 5)

    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=kernel_size, strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(64, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(128, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(256, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.4)(x)

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='gan-discriminator')
    model.summary()
    # plot_model(model, img_folder + 'Discriminator-' + date_str + '.png', show_shapes=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


def get_my_generator():
    input_shape = 100
    kernel_size = (5, 5)

    inputs = Input(shape=input_shape)
    x = Dense(7 * 7 * 192)(inputs)

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = ReLU()(x)
    x = Reshape((7, 7, 192))(x)
    x = Dropout(rate=0.4)(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(96, kernel_size=kernel_size, strides=1, padding='same')(x)

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = ReLU()(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(48, kernel_size=kernel_size, strides=1, padding='same')(x)

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = ReLU()(x)
    x = Conv2DTranspose(24, kernel_size=kernel_size, strides=1, padding='same')(x)

    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = ReLU()(x)
    outputs = Conv2DTranspose(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')(x)

    # outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='gan-generator')
    model.summary()
    # plot_model(model, to_file=img_folder+'Generator' + date_str + '.png', show_shapes=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# Unused (poor generation)
def get_generator():
    gen = Sequential()

    d = 7
    gen.add(Dense(d * d * 192, kernel_initializer=RandomNormal(0, 0.02), input_dim=noise_dim))
    # gen.add(LeakyReLU(0.2))  # 4xLeakyReLU(0.2) = Squiggles
    gen.add(LeakyReLU())  # 4xLeakyReLU(0.2) = similar Squiggles
    # gen.add(ReLU(0.2))  # 4xReLU() gives black boxes, same with 4xRelu(0.2)
    gen.add(Reshape((d, d, 192)))
    # gen.add(Dropout(0.4))  # This and LeakyRelu(0.2) gives black boxes, without this = squiggles.

    gen.add(Conv2DTranspose(96, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    gen.add(LeakyReLU())

    gen.add(Conv2DTranspose(48, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    gen.add(LeakyReLU())

    gen.add(Conv2DTranspose(24, (5, 5), strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    gen.add(LeakyReLU())

    gen.add(Conv2DTranspose(channels, (5, 5), padding='same', activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02)))

    gen.compile(loss='binary_crossentropy', optimizer=optimizer)
    # gen.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
    # gen.compile(loss=generator_loss, optimizer=generator_optimizer)
    return gen


# Unused (poor generation)
def get_discriminator():
    disc = Sequential()

    disc.add(Conv2D(32, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02), input_shape=(img_cols, img_rows, channels)))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))

    disc.add(Conv2D(64, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))

    disc.add(Conv2D(128, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))

    disc.add(Conv2D(256, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))

    disc.add(Flatten())
    # disc.add(Dropout(0.4))  # TODO: Remove this if adding Dropouts between each Conv layer.
    disc.add(Dense(1, activation='sigmoid', input_shape=(img_cols, img_rows, channels)))

    disc.compile(loss='binary_crossentropy', optimizer=optimizer)
    # disc.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)
    # disc.compile(loss=discriminator_loss, optimizer=discriminator_optimizer)
    return disc


# Unused (and non-working)
def get_fc_generator():
    gen = Sequential()

    gen.add(Dense(256, input_dim=noise_dim))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(512))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(1024))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(img_rows * img_cols * channels, activation='tanh'))

    gen.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gen


# Unused (and non-working)
def get_fc_discriminator():
    disc = Sequential()

    disc.add(Dense(1024, input_dim=img_rows * img_cols * channels))
    disc.add(LeakyReLU(0.2))

    disc.add(Dense(512))
    disc.add(LeakyReLU(0.2))

    disc.add(Dense(256))
    disc.add(LeakyReLU(0.2))

    disc.add(Dense(1, activation='sigmoid'))

    disc.compile(loss='binary_crossentropy', optimizer=optimizer)
    return disc


# Unused
def discriminator_loss(real_output, fake_output):
    # Compare discriminator's predictions on real images to an array of all 1s (1 being considered real).
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Compare discriminator's predictions on fake images to an array of all 0s (0 being most fake).
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # The total loss is the summation of these two loss values.
    total_loss = real_loss + fake_loss

    return total_loss


# Unused
def generator_loss(fake):
    # Compare discriminator's decisions on fake images to an array of all 1s (1 being considered real).
    return cross_entropy(tf.ones_like(fake), fake)


def save_image(epoch, test_input):
    plt.figure(figsize=(4, 4))  # 4 x 4 images, for a total of 16.

    for i in range(test_input.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_input[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        # plt.imshow(test_input[i, :, :, 0] * 127.5 + 127.5)  # rgb (purple, yellow, green)
        # plt.imshow(test_input[i, :, :, 0], cmap='gray')  # No real difference in image.
        plt.axis('off')

    plt.savefig(f'{img_dir}/epoch_{epoch+1:04d}.png')
    plt.show()


def create_gif():
    anim_file = f'{img_dir}/gan-{date_str}.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{img_dir}/epoch_*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    import tensorflow_docs.vis.embed as embed
    embed.embed_file(anim_file)


def plot_history(name, hx):
    d_loss = hx.history['disc_loss']
    g_loss = hx.history['gan_loss']

    fig, axs = plt.subplots(1)
    fig.suptitle(name)
    plt.xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.set_xlim(1, epochs)
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.plot(range(1, epochs + 1), d_loss, 'r', label='Discriminator')
    axs.plot(range(1, epochs + 1), g_loss, 'b', label='GAN')
    axs.legend()

    plt.savefig(f'{img_dir}/{name}-hx-{date_str}.png')
    plt.show()


# Plot loss and create gif, even if program terminates early.
def exit_handler():
    print('Exiting program early...')
    plot_history('GAN', plot_data)
    create_gif()


atexit.register(exit_handler)

discriminator = get_my_discriminator()
generator = get_my_generator()
discriminator.trainable = False

input_layer = Input(shape=(noise_dim,))
fake_image = generator(input_layer)
output = discriminator(fake_image)

gan = Model(input_layer, output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

class PlotData:
    history = {'disc_loss':[], 'gan_loss':[]}

plot_data = PlotData()

for epoch in range(epochs):
    start = time.time()

    for batch in range(steps):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        real_x = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]
        fake_x = generator.predict(noise, verbose=0)
        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2 * batch_size)
        disc_y[:batch_size] = 0.9  # one-sided label smoothing
        disc_loss = discriminator.train_on_batch(x, disc_y)

        gan_y = np.ones(batch_size)
        gan_loss = gan.train_on_batch(noise, gan_y)

    save_image(epoch, fake_x)

    plot_data.history['disc_loss'].append(disc_loss)
    plot_data.history['gan_loss'].append(gan_loss)

    print(f'[{time.time()-start:.2f}s] Epoch {epoch+1:03}\tDisc Loss: {disc_loss:.5f}\tGan Loss: {gan_loss:.5f}')

print('Doing usual plot and gif...')
plot_history('GAN', plot_data)
create_gif()
