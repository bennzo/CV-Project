import os
import ast

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def show_annotation(img_path, annot_file):
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
    return 0

show_annotation('data\DSCF1015.JPG', 'annotationsTrain.txt')