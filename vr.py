import sys
sys.path.append('./yolov7')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
date_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))  # Datetime string used in output file names.

# YOLOv7 implemented with PyTorch
import torch
from torchvision import transforms

# YOLOv7 modules
from utils.datasets import letterbox  # Pad image
from utils.general import non_max_suppression_kpt  # Run Non-Max Suppression algorithm to clean initial output for interpretation.
from utils.plots import output_to_keypoint, plot_skeleton_kpts  # Add keypoints to an image once predicted.

# import matplotlib  # vr.py:69: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
# matplotlib.use('TkAgg')  # Blank and unresponsive plots in pop-out windows.
# matplotlib.use('wxAgg')  # ModuleNotFoundError: No module named 'wx'
# matplotlib.use('Qt5Agg')  # Blank and unresponsive plots in pop-out windows.
# matplotlib.use('Qt5Agg')  # Blank and unresponsive plots in pop-out windows.

# import matplotlib.pyplot as plt
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
        # model.cuda()

    return model


def run_inference(url, model):
    image = cv2.imread(url)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)  # TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
    image = image.unsqueeze(0)  # RuntimeError: Given groups=1, weight of size [64, 12, 3, 3], expected input[1, 3, 1280, 480] to have 12 channels, but got 3 channels instead

    # image = image.half().to("cuda:0")
    if torch.cuda.is_available():
        image = image.half().to("cuda:0")

    output, _ = model(image)  # RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
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
    # img_dir = './images/skele'
    plt.savefig(f'{img_path}-{date_str}.png')
    # plt.show()


def run(img_path):
    model = load_model()
    output, image = run_inference(img_path, model)
    visualize_output(output, image, model, img_path)


# git clone https://github.com/WongKinYiu/yolov7.git
# curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt

# run('./images/skele/karate.jpg')  # Mondo Generator on Unsplash
# run('./images/skele/basketball.jpg')  # Bryan Reyes on Unsplash
