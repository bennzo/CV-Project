import os
import ast

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def show_annotations_file(img_path, annot_file):
    _, filename = os.path.split(img_path)

    # Extract annotations
    with open(annot_file, 'r') as file:
        annots = list(filter(lambda x: filename in x, file))
    annotations = ast.literal_eval(annots[0][len(filename)+1:-1])

    # Plot image
    img = np.array(Image.open(img_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Plot rectangles
    for xmin, ymin, width, height, _ in annotations:
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    return img, annotations

def show_annotations(img, annotations, format='xywh'):
    if format == 'xyxy':
        annotations[:,2:4] -= annotations[:, :2]

    # Plot image
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Plot rectangles
    for annot in annotations:
        rect = patches.Rectangle((annot[0], annot[1]), annot[2], annot[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def xyxy2xywh(boxes):
    boxes[:, 2:4] -= boxes[:, :2]
    boxes[:, :2]  += boxes[:, 2:4] * 0.5

def xywh2xyxy(boxes):
    boxes[:, :2]  -= boxes[:, 2:4] * 0.5
    boxes[:, 2:4] += boxes[:, :2]