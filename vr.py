import gc

import sys
sys.path.append('./yolov7')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))  # Datetime string used in output file names.

# YOLOv7 is implemented with PyTorch
import torch
from torchvision import transforms

# YOLOv7 modules
from utils.datasets import letterbox  # Pad image
from utils.general import non_max_suppression_kpt  # Run Non-Max Suppression algorithm to clean initial output for interpretation.
from utils.plots import output_to_keypoint, plot_skeleton_kpts  # Add keypoints to an image once predicted.

from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load('./yolov7/yolov7-w6-pose.pt', map_location=device)
    model = weights['model']  # Load model from weight file.
    _ = model.float().eval()  # Put in interface mode.

    if torch.cuda.is_available():
        model.half().to(device)  # float16 instead of float32 for faster inference.

    return model, device


def run_inference(url, model, device):
    image = cv2.imread(url)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    # image = image.unsqueeze(0)

    if torch.cuda.is_available():
        image = image.half().to(device)

    output, _ = model(image)
    return output, image


def visualize_output(output, image, model, img_path):
    # Clean output with Non-Max Suppression algorithm.
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence threshold
                                     0.65,  # Intersection over Union (IoU) threshold.
                                     nc=model.yaml['nc'],  # Number of classes
                                     nkpt=model.yaml['nkpt'],  # Number of keypoints
                                     kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    # nimg = doResize(nimg, 320, 408)
    nimg = doPerspectiveWarp(nimg)
    # nimg = doPolarWarp(nimg)
    # nimg = doAffineWarp(nimg)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.savefig(f'{img_path}-{date_str}.png', bbox_inches='tight')


def doResize(img, new_height, new_width):
    h, w, _ = img.shape
    new_size = (new_height * 2, w)
    img = img.resize(new_size, Image.LANCZOS)
    new_size = (new_height, new_width * 2)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def doPerspectiveWarp(img, flags=cv2.INTER_LINEAR):
    h, w, _ = img.shape
    input_pts = np.float32([[0, 0],
                            [w, 0],
                            [0, h],
                            [w, h]])
    output_pts = np.float32([[0, 0],
                             [w, 0],
                             [0, h*2],
                             [w, h*2]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=flags)

    return img


def doPolarWarp(img, flags=cv2.WARP_POLAR_LINEAR):
    h, w, _ = img.shape
    radius = h // (2 * np.pi)
    center = (w / 2, h)
    polar_img = cv2.warpPolar(img,
                              center=center,
                              maxRadius=radius,
                              dsize=(h, w),
                              flags=flags)
    return polar_img


def doAffineWarp(img, flags=cv2.INTER_LINEAR):
    h, w, _ = img.shape
    M = np.float32([[w, 0, 0],
                    [0, h*2, 0]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=flags)
    return img


def run(img_path):
    model, device = load_model()
    output, image = run_inference(img_path, model, device)
    visualize_output(output, image, model, img_path)


# Run on batches of the 50 example images, so as not to run out of memory.
i_rng, j_rng = 4,5
for i in range(i_rng, i_rng+1):
    for j in range(j_rng, j_rng+5):
        print(f'{str(i)} {str(j)}')
        run('./images/skele/ego/'+str(i)+str(j)+'teaser_hejkLDN.jpg')

# Run on individual images for quick testing.
# run('./images/skele/jasmin-chew-rhD1h1wUfNc-unsplash.jpg')
# run('./images/skele/crop-jasmin-chew-rhD1h1wUfNc-unsplash.jpg')
# run('./images/skele/joel-muniz-c_UEKZRvSU0-unsplash.jpg')
# run('./images/skele/luise-and-nic-WfB-32ng990-unsplash.jpg')
