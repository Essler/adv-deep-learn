import sys
sys.path.append('./yolov7')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))  # Datetime string used in output file names.

# YOLOv7 implemented with PyTorch
import torch
from torchvision import transforms

# git clone https://github.com/WongKinYiu/yolov7.git
# curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt

# YOLOv7 modules
from utils.datasets import letterbox  # Pad image
from utils.general import non_max_suppression_kpt  # Run Non-Max Suppression algorithm to clean initial output for interpretation.
from utils.plots import output_to_keypoint, plot_skeleton_kpts  # Add keypoints to an image once predicted.

from matplotlib import pyplot as plt
import cv2
import numpy as np


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model from weight file.
    model = torch.load('yolov7/yolov7-w6-pose.pt', map_location=device)['model']
    model.float().eval()  # Put in interface mode.

    if torch.cuda.is_available():
        model.half().to(device)  # float16 instead of float32 for faster inference.

    return model


def run_inference(url, model):
    image = cv2.imread(url)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    if torch.cuda.is_available():
        image = image.half().to("cuda:0")

    output, _ = model(image)
    return output, image


def visualize_output(output, image, model, img_path):
    output = non_max_suppression_kpt(output,
                                    0.25,  # Confidence
                                    0.65,
                                    nc=model.yaml['nc'],
                                    nkpt=model.yaml['nkpt'],
                                    kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.savefig(f'{img_path}-{date_str}.png')


def run(img_path):
    model = load_model()
    output, image = run_inference(img_path, model)
    visualize_output(output, image, model, img_path)


for i in range(4, 5):
    for j in range(5):
        run('./images/skele/ego/'+str(i)+str(j)+'teaser_hejkLDN.jpg')

print("DONE!")
