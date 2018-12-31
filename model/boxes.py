import torch
import numpy as np

import utils


# Test
from model.data_loader import *

class Boxes:
    def __init__(self):
        self.areas  = np.array([32**2, 64**2, 128**2, 256**2, 512**2])    # Anchor box areas for each feature map as specified in FPN
        self.scales = np.array([2**0, 2**(1/3), 2**(2/3)])                # Scales for more Anchor boxes
        self.ratios = np.array([1/2, 1, 2/1])                             # W/H Ratio of Anchor boxes

        # self.areas  = torch.Tensor([32**2, 64**2, 128**2, 256**2, 512**2])    # Anchor box areas for each feature map as specified in FPN
        # self.scales = torch.Tensor([2**0, 2**(1/3), 2**(2/3)])                # Scales for more Anchor boxes
        # self.ratios = torch.Tensor([1/2, 1, 2/1])                             # W/H Ratio of Anchor boxes

    def create_anchors(self, img_shape):
        # Creating anchors according to the shape given and
        # the shape of the feature maps created by the FPN
        #
        # Args:
        #       img_shape: (Tensor) of shape (2,) representing (W,H) of the image
        # Return:
        #       anchors: (Tensor) of shape (#Anchors, 4) that holds the anchors of a given image
        #               each anchor is represented as (x_center, y_center, width, height)
        w, h = img_shape
        edges = np.sqrt(self.areas)
        n_fmaps = len(self.areas)
        n_anchors = len(self.scales)*len(self.ratios)

        # Compute width and height of boxes across all areas,scales,ratios
        boxW = np.sqrt(self.areas.reshape(-1,1,1) * (self.scales.reshape(-1,1) / self.ratios))          # shape:(areas,scales,ratios)
        boxH = boxW * self.ratios                                                                       # shape:(areas,scales,ratios)
        boxWH = np.stack([boxW.reshape(n_fmaps,-1), boxH.reshape(n_fmaps,-1)], axis=-1)

        # Compute number of cells in each row and column for every feature map
        cellW = np.ceil(w / edges).astype(int)                                                          # shape:(areas,)
        cellH = np.ceil(h / edges).astype(int)

        # TODO: Replace concatenate with pre-allocating memory for anchors
        anchors = []
        for i in range(n_fmaps):
            # Create coordinate grid
            grid = np.array(np.meshgrid(range(cellW[i]), range(cellH[i]))).T.reshape(-1,2) + 0.5        # shape:(cellW*cellH,2)

            # Multiply grid by edge lengths
            boxCenters = np.tile(edges[i], n_anchors).reshape(n_anchors,1,1) * grid                     # shape:(scales*ratios,cellW*cellH,2)
            boxCenters = boxCenters.transpose(1,0,2).reshape(-1,2)                                      # shape:(scales*ratios*cellW*cellH,2)

            # Stitch grid of centers and dimensions of anchors
            anchors.append(np.concatenate((boxCenters, np.tile(boxWH[i], (cellW[i]*cellH[i],1))), axis=1))

        anchors = np.concatenate(anchors)

        return anchors

    def create_anchors_torch(self, img_shape):
        # Creating anchors according to the shape given and
        # the shape of the feature maps created by the FPN
        #
        # Args:
        #     img_shape: (Tensor) of shape (2,) representing (W,H) of the image
        # Return:
        #     anchors: (Tensor) of shape (#Anchors, 4) that holds the anchors of a given image
        #               each anchor is represented as (x1,y1,x2,y2)
        w, h = img_shape
        edges = torch.sqrt(self.areas)
        n_fmaps = len(self.areas)
        n_anchors = len(self.scales) * len(self.ratios)

        # Compute width and height of boxes across all areas,scales,ratios
        boxW = torch.sqrt(self.areas.view(-1, 1, 1) * (self.scales.view(-1, 1) / self.ratios))  # shape:(areas,scales,ratios)
        boxH = boxW * self.ratios  # shape:(areas,scales,ratios)
        boxWH = torch.cat([boxW.view(n_fmaps, -1), boxH.view(n_fmaps, -1)], axis=-1)

        # Compute number of cells in each row and column for every feature map
        cellW = torch.ceil(w / edges).astype(int)  # shape:(areas,)
        cellH = torch.ceil(h / edges).astype(int)

        # TODO: Replace concatenate with pre-allocating memory for anchors
        anchors = []
        for i in range(n_fmaps):
            # Create coordinate grid
            grid = torch.meshgrid([range(cellW[i]), range(cellH[i])]).transpose().view(-1,2) + 0.5  # shape:(cellW*cellH,2)

            # Multiply grid by edge lengths
            boxCenters = np.tile(edges[i], n_anchors).reshape(n_anchors, 1,
                                                              1) * grid  # shape:(scales*ratios,cellW*cellH,2)
            boxCenters = boxCenters.transpose(1, 0, 2).reshape(-1, 2)  # shape:(scales*ratios*cellW*cellH,2)

            # Stitch grid of centers and dimensions of anchors
            anchors.append(np.concatenate((boxCenters, np.tile(boxWH[i], (cellW[i] * cellH[i], 1))), axis=1))

        anchors = np.concatenate(anchors)

        # Convert xywh->xyxy
        anchors[:, :2] -= anchors[:, 2:] * 0.5
        anchors[:, 2:] += anchors[:, :2]

        return anchors

    def parameterize(self, gt_annots, img_shape):
        # Parameterize ground truth annotations (x_center,y_center,w,h) to bounding box regression
        # offset (tx,ty,tw,th) according to the FasterRCNN paper.
        #               tx = (x-x_a)/w_a, ty = (y-y_a)/h_a
        #               tw = log(w/w_a), th = log(h/h_a)
        # Args:
        #       gt_annots: (Tensor) of shape (#objects, 4+c) representing the bounding boxes and classifications
        #                   of the objects in the image
        #       img_shape: (Tensor) of shape (2,) representing (W,H) of the image
        # Return:
        #       para_annots: (Tensor) of shape (#Anchors, 4+c) that parametrized bounding boxes and classes
        POS_THRESH = 0.5
        NEG_THRESH = 0.4

        anchors = torch.Tensor(self.create_anchors(img_shape))
        para_annots = torch.zeros((anchors.shape[0], gt_annots.shape[1]))

        # convert anchors to xyxy before iou calculation
        utils.xywh2xyxy(anchors)

        # calculate IoU
        ious = self.iou(anchors, gt_annots)
        iou_max, iou_argmax = torch.max(ious, dim=1)

        # convert boxes and anchors to xywh for parameterization
        utils.xyxy2xywh(gt_annots)
        utils.xyxy2xywh(anchors)

        obj_idx = iou_max > POS_THRESH
        bg_idx = iou_max < NEG_THRESH
        ignore_idx = 1 - (obj_idx | bg_idx)

        para_annots = gt_annots[iou_argmax]
        return

    def deparameterize(self):
        pass

    def iou(self, boxes1, boxes2):
        # Calculate IoU for each anchor and bounding box combination
        #
        # Args:
        #       boxes1,boxes2: (Tensor) of shape (#boxes, 4) representing a box (x_tl, y_tl, x_br, y_br)
        # Return:
        #       ious: (Tensor) of shape (#boxes1, #boxes2) representing the IoUs between box in row i and box in col j


        m = boxes1.shape[0]
        n = boxes2.shape[0]

        # calculate the top left and bottom right of the intersection box
        topleft = torch.max(boxes1.unsqueeze(1)[:,:,:2], boxes2[:,:2])          # shape:(m,n,2)
        bottomright = torch.min(boxes1.unsqueeze(1)[:,:,2:4], boxes2[:,2:4])      # shape:(m,n,2)

        # calculate intersection
        edges = torch.clamp(bottomright - topleft + 1, 0)
        intersection = edges[:,:,0] * edges[:,:,1]

        # calculate union
        edges1 = boxes1[:,2:4] - boxes1[:,:2] + 1
        edges2 = boxes2[:,2:4] - boxes2[:,:2] + 1
        areas1 = edges1[:,0] * edges1[:,1]
        areas2 = edges2[:,0] * edges2[:,1]
        union = (areas1.unsqueeze(1) + areas2.unsqueeze(0)) - intersection

        ious = intersection / union
        return ious



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
    img, annots = train_data[0]
    box = Boxes()
    box.parameterize(annots, img.shape[1:][::-1])

