import sys
import os
import ast
import torch
import numpy as np
import pandas as pd
import cv2
import random

import skimage.io
import skimage.transform

from operator import methodcaller
from torch.utils.data import Dataset
from torchvision import transforms

from model.boxes import Anchors

import utils

class BusDataset(Dataset):
    def __init__(self, data_dir, annotations_file, num_classes, transform=None, eval=False):
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
        self.classes = num_classes
        self.transform = transform
        self.anchors = Anchors()

        if not eval:
            with open(annotations_file) as annot_file:
                name_annotations = annot_file.readlines()
                for filename, annots in map(methodcaller("split", ":"), name_annotations):
                    # Append images
                    self.files.append(os.path.join(data_dir, filename))

                    # Append annotations
                    annot_np = np.array(ast.literal_eval(annots))
                    if len(annot_np.shape) == 1:
                        annot_np = annot_np[np.newaxis, :]
                    # Convert xywh -> xyxy
                    annot_np[:,2:4] += annot_np[:, :2]
                    annot_np[:, 4] -= 1

                    # Encode one hot embeddings for classification
                    one_hot_classes = np.zeros((annot_np.shape[0], self.classes))
                    one_hot_classes[range(annot_np.shape[0]), annot_np[:, -1]] = 1
                    self.annotations.append(np.column_stack((annot_np[:,:4], one_hot_classes)))
        else:
            self.files = [os.path.join(data_dir, im) for im in os.listdir(data_dir)]
            self.annotations = [np.ones((1,4+self.classes)) for i in range(len(self.files))]



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

        # img = cv2.imread(self.files[idx]).astype(np.float32) / 255
        img = skimage.io.imread(self.files[idx]).astype(np.float32) / 255
        annots = self.annotations[idx].copy()
        sample = {'img':img, 'annots':annots}

        if self.transform:
            sample = self.transform(sample)

        # return sample['img'], sample['annots'].cuda(), sample['scales']
        return sample['img'], sample['annots'], sample['scales']

    def get_filename(self, idx):
        return os.path.basename(self.files[idx]).upper()

    # TODO: Change back to old collate and move anchor creation to end of pipeline
    def collate(self, batch):
        # A Collater function that enables work with batches that contain
        # images of different sizes and contains different number of objects
        #
        # Args:
        #     batch: list of (Image, Annotation)
        # Return:
        #     padded_batch: list of (Image, Annotation) where all of the images and annotations
        #                   are of the same shape
        N = len(batch)
        _, classes = batch[0][1].shape

        maxH = max(batch, key=lambda sample: sample[0].shape[1])[0].shape[1]
        maxW = max(batch, key=lambda sample: sample[0].shape[2])[0].shape[2]

        padded_images = torch.zeros((N, 3, maxH, maxW))
        padded_scales = torch.zeros((N,2)).float()
        param_annots = []

        for i, (img, annots, scale) in enumerate(batch):
            D, H, W = img.shape
            padded_images[i, :, :H, :W] = img
            param_annots.append(self.anchors.parameterize(annots, (W, H)))
            padded_scales[i] = scale

        param_annots = torch.stack(param_annots)
        return padded_images, param_annots, padded_scales


class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, annots = sample.values()
            H, W, D = img.shape

            # TODO: Change to skimage flip
            img = cv2.flip(img, 1)
            # img = img[:, ::-1, :]
            annots[:, 0], annots[:, 2] = W - annots[:, 2], W - annots[:, 0]

            sample['img'] = img
            sample['annots'] = annots

        return sample


class VerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, annots = sample.values()
            H, W, D = img.shape

            # TODO: Change to skimage flip
            img = cv2.flip(img, 0)
            annots[:, 1], annots[:, 3] = H - annots[:, 3], H - annots[:, 1]

            sample['img'] = img
            sample['annots'] = annots

        return sample


class Resize(object):
    def __init__(self, minside=256):
        self.minside = minside

    def __call__(self, sample):
        img, annots = sample.values()

        H, W, D = img.shape
        scale = self.minside / min(H,W)
        scale_W = (round(scale*W) - round(scale*W)%32) / W
        scale_H = (round(scale*H) - round(scale*H)%32) / H

        # img = cv2.resize(img, None, fx=scale_W, fy=scale_H)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = skimage.transform.resize(img, (int(round(H * scale_H)), int(round((W * scale_W)))), mode = 'constant', anti_aliasing = False)

        annots[:, [0, 2]] = (annots[:, [0, 2]] * scale_W)
        annots[:, [1, 3]] = (annots[:, [1, 3]] * scale_H)

        sample['img'] = torch.from_numpy(img).permute(2,0,1)
        sample['annots'] = torch.from_numpy(annots).type(torch.FloatTensor)
        sample['scales'] = torch.FloatTensor([scale_W, scale_H])

        return sample


class Normalize(object):
    # Normalize according to ResNet if using pretrained weights
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['img'] = self.normalize(sample['img'])
        return sample


def collate(batch):
    # A Collater function that enables work with batches that contain
    # images of different sizes and contains different number of objects
    #
    # Args:
    #     batch: list of (Image, Annotation)
    # Return:
    #     padded_batch: list of (Image, Annotation) where all of the images and annotations
    #                   are of the same shape
    N = len(batch)
    _, classes = batch[0][1].shape

    maxH = max(batch, key=lambda sample: sample[0].shape[1])[0].shape[1]
    maxW = max(batch, key=lambda sample: sample[0].shape[2])[0].shape[2]
    maxAnnot = max(batch, key=lambda sample: sample[1].shape[0])[1].shape[0]

    padded_images = torch.zeros((N, 3, maxH, maxW))
    # One hot embedding
    padded_annots = -1*torch.ones((N, maxAnnot, classes))

    param_annots = []

    for i, (img, annots) in enumerate(batch):
        D, H, W  = img.shape
        C, L = annots.shape

        # if H < maxH or W < maxW:
        padded_images[i, :, :H, :W] = img

        # if C < maxAnnot:
        padded_annots[i, :C, :L] = annots

    return padded_images, padded_annots

if __name__ == '__main__':
    train_data = BusDataset(
        '../data',
        '../annotationsTrain.txt',
        6,
        transforms.Compose([
            VerticalFlip(0),
            HorizontalFlip(0),
            Resize(minside=512),
        ]))
    sample_test = train_data[0]
    utils.show_annotations(sample_test[0].numpy().transpose(1,2,0), sample_test[1].numpy(), format='xyxy')
