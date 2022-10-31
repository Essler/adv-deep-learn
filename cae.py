def cae_implement():
    import keras
    from keras import layers

    input_img = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mae')
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


def cae_train(autoencoder):
    from keras.datasets import cifar10
    import numpy as np

    (x_train, _), (x_test, _) = cifar10.load_data()

    x_train = x_train[:10000]  # First 10% of train data set
    x_test = x_test[:1000]  # First 10% of test data set

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    from keras.callbacks import TensorBoard

    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    return x_train, x_test


def cae_plot(autoencoder, x_test):
    import matplotlib.pyplot as plt

    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    offset = 400
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i+offset].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i+offset].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


autoencoder = cae_implement()
(x_train, x_test) = cae_train(autoencoder)
cae_plot(autoencoder, x_test)
