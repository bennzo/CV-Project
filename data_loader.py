import random
import ast
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BusDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform):

        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, filename) for filename in self.filenames]

        self.annotations = pd.read_csv('annotationsTrain.txt', sep=" ", header=None)
        self.annotations.columns = ['name', 'annot']
        self.annotations = [list(annot) for annot in map(ast.literal_eval, list(self.annotations['annot']))]


    def __len__(self):
        # Return the size of the dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        # Fetches indexed idx image and annotation. Performs transform on image.
        #
        # Args:
        #   idx: (int) index of the image in [0...len(dataset)-1]
        #
        # Returns:
        #   img: (Tensor) Transoformed image in index idx
        #   annot: (int[][]) List of annotations of the form [xmin, ymin, width, height, color]
        pass
