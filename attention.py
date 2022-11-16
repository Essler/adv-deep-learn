import datetime

import keras.layers
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img


date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # Datetime string used in output file names.

vgg16_model = VGG16(weights='imagenet',  # Use pre-trained weights from ImageNet's 14 million images and 1000 object classes.
              include_top=True)  # Include the classification top as we want to classify images.

vgg16_model.summary()

# Freeze weights
for layer in vgg16_model.layers:
    layer.trainable = False

quit()  # TODO Crashes on inputs/outputs!
# between input and conv 1-1
base_model = tf.keras.Model(inputs=vgg16_model.inputs,
                   outputs=vgg16_model.get_layer("block1_conv1").outputs)
base_out = base_model.output
senet = senet_layers(base_out)

sen_model = Model(inputs=base_model.input, outputs=senet)

sen_model.summary()
new_top_layer = base_model

merged = keras.layers.Concatenate([])

# between pooling and conv 2-1
base_model = tf.keras.Model(input=vgg16_model.input,
                   output=vgg16_model.get_layer("block2_conv1").output)

# between pooling and conv 3-1
base_model = tf.keras.Model(input=vgg16_model.input,
                   output=vgg16_model.get_layer("block3_conv1").output)

# between pooling and conv 4-1
base_model = tf.keras.Model(input=vgg16_model.input,
                   output=vgg16_model.get_layer("block4_conv1").output)

# between pooling and 5-1
base_model = tf.keras.Model(input=vgg16_model.input,
                   output=vgg16_model.get_layer("block5_conv1").output)

# between pooling and dense
base_model = tf.keras.Model(input=vgg16_model.input,
                   output=vgg16_model.get_layer("block2_conv1").output)


# ImageNet classes 134, 322, 736, and 986.
image_titles = ['Crane', 'Ringlet Butterfly', 'Pool Table', 'Yellow Lady-slipper']
# image_titles = ['Bittern', 'Admiral', 'Poncho', 'Daisy']

# Load images and convert to a Numpy array
img1 = load_img('images/134_Crane/0.jpg', target_size=(224, 224))
img2 = load_img('images/134_Crane/1.jpg', target_size=(224, 224))
img3 = load_img('images/134_Crane/2.jpg', target_size=(224, 224))
img4 = load_img('images/134_Crane/3.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3), np.array(img4)])

# Prepare input data for VGG16.
X = preprocess_input(images)

# Render original input images.
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('images/VGG16-'+date_str+'.png')
plt.show()

def senet_layers(module_input):
    channel = module_input._keras_shape[-1]  # Channel-last order.

    x = tf.keras.layers.GlobalAveragePooling2D()(module_input)
    x = tf.keras.layers.Reshape((1,1,channel))(x)
    x = tf.keras.layers.Dense(channel // 8,
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=True,
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.Dense(channel,
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=True,
                              bias_initializer='zeros')(x)
    x = multiply([module_input, x])

    return x
