import sys
import os
import ast
import torch
import numpy as np
import pandas as pd
import random

from operator import methodcaller

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from skimage import io, transform

class BusDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        # Initializes a basic Bus dataset
        #
        # Args:
        #   data_dir: (string) path to the folder with the bus images
        #   annotations_file: (string) path to the file with the annotations
        #                      lines in the file should be of the form:
        #                      filename:[xmin1, ymin1, width1, height1, class1],...,[xmink, ymink, widthk, heightk, classk]
        #   transform: (torchvision.transforms) a transformation to be applied on the images
        super(BusDataset, self).__init__()
        self.files = []
        self.annotations = []
        self.classes = 7        # 0 is background
        self.transform = transform

        with open(annotations_file) as annot_file:
            name_annotations = annot_file.readlines()
            for filename, annots in map(methodcaller("split", ":"), name_annotations):
                # Append images
                self.files.append(os.path.join(data_dir, filename))

                # Append annotations
                # Convert classes to one hot vectors
                annot_np = np.array(ast.literal_eval(annots))
                one_hot_classes = np.zeros((annot_np.shape[0], self.classes))
                one_hot_classes[range(annot_np.shape[0]), annot_np[:, -1]] = 1
                self.annotations.append(np.column_stack((annot_np[:,:4], one_hot_classes)))


    def __len__(self):
        # Return the size of the dataset
        return len(self.files)

    def __getitem__(self, idx):
        # Fetches indexed idx image and annotation. Performs transform on image.
        #
        # Args:
        #   idx: (int) index of the image in [0...len(dataset)-1]
        #
        # Returns:
        #   img: (Tensor) Transoformed image in index idx
        #   annot: (Tensor[]) List of annotations of the form [xmin, ymin, width, height, color]

        image = io.imread(self.files[idx]).astype(np.float32) / 255
        annots = self.annotations[idx]
        sample = {'image':image, 'annots':annots}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    test = BusDataset('data', '..\\annotationsTrain.txt')