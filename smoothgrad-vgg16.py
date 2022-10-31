import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency


date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # Datetime string used in output file names.

model = VGG16(weights='imagenet',  # Use pre-trained weights from ImageNet's 14 million images and 1000 object classes.
              include_top=True)  # Include the classification top, as we want to classify images.

# ImageNet classes 280, 320, and 674.
image_titles = ['Gray Fox', 'Damselfly', 'Mousetrap']

# Load images and convert to a Numpy array
img1 = load_img('images/280-grayfox-contrast.jpg', target_size=(224, 224))
img2 = load_img('images/320-damselfly-contrast.jpg', target_size=(224, 224))
img3 = load_img('images/674-mousetrap.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Prepare input data for VGG16.
X = preprocess_input(images)

# Render original input images.
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('images/VGG16-'+date_str+'.png')
plt.show()

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=ReplaceToLinear(),
                    clone=True)

# Generate saliency map with smoothing that reduces noise, by adding noise to input images.
saliency_map = saliency(CategoricalScore([280, 320, 674]),  # ImageNet indices for Gray Fox, Damselfly, and Mousetrap.
                        X,
                        smooth_samples=20,  # The number of calculating gradients iterations.
                        smooth_noise=0.20)  # noise spread level.

# Render saliency maps.
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('images/SmoothGrad-'+date_str+'.png')
plt.show()