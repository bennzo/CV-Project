import numpy as np
import torch
from torchvision import transforms

import utils

# Test
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Anchors:
    def __init__(self):
        self.areas  = np.array([32**2, 64**2, 128**2, 256**2, 512**2])    # Anchor box areas for each feature map as specified in FPN
        self.scales = np.array([2**0, 2**(1/3), 2**(2/3)])                # Scales for more Anchor boxes
        self.ratios = np.array([1/2, 1, 2/1])                             # W/H Ratio of Anchor boxes

        self.areas_t  = torch.Tensor([32**2, 64**2, 128**2, 256**2, 512**2])    # Anchor box areas for each feature map as specified in FPN
        self.scales_t = torch.Tensor([2**0, 2**(1/3), 2**(2/3)])                # Scales for more Anchor boxes
        self.ratios_t = torch.Tensor([1/2, 1, 2/1])                             # W/H Ratio of Anchor boxes

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

        # Steps according to the FPN feature maps receptive field
        steps = np.array([2**i for i in range(3,7+1)])

        n_fmaps = len(self.areas)
        n_anchors = len(self.scales)*len(self.ratios)

        # Compute width and height of boxes across all areas,scales,ratios
        boxW = np.sqrt(self.areas.reshape(-1,1,1) * (self.scales.reshape(-1,1) / self.ratios))          # shape:(areas,scales,ratios)
        boxH = boxW * self.ratios                                                                       # shape:(areas,scales,ratios)
        boxWH = np.stack([boxW.reshape(n_fmaps,-1), boxH.reshape(n_fmaps,-1)], axis=-1)

        # Compute number of cells in each row and column for every feature map
        cellsCol = np.ceil(w / steps).astype(int)
        cellsRow = np.ceil(h / steps).astype(int)

        # TODO: Replace concatenate with pre-allocating memory for anchors
        # TODO: Check other method of scaling sizes of anchors with ratios
        anchors = []
        for i in range(n_fmaps):
            # Create coordinate grid
            # grid = np.array(np.meshgrid(range(cellsCol[i]), range(cellsRow[i]))).T.reshape(-1,2) + 0.5        # shape:(cellW*cellH,2)
            grid = np.array(np.meshgrid(range(cellsRow[i]), range(cellsCol[i]))).T.reshape(-1, 2) + 0.5

            # Multiply grid by edge lengths
            boxCenters = np.tile(steps[i], n_anchors).reshape(n_anchors, 1, 1) * grid                       # shape:(scales*ratios,cellW*cellH,2)
            boxCenters = np.flip(boxCenters.transpose(1,0,2).reshape(-1,2), axis=1)                         # shape:(scales*ratios*cellW*cellH,2)

            # Stitch grid of centers and dimensions of anchors
            anchors.append(np.concatenate((boxCenters, np.tile(boxWH[i], (cellsCol[i]*cellsRow[i],1))), axis=1))

        anchors = np.concatenate(anchors)
        return torch.Tensor(anchors)

    def create_anchors_tensor(self, img_shape):
        # Creating anchors according to the shape given and
        # the shape of the feature maps created by the FPN
        #
        # Args:
        #       img_shape: (Tensor) of shape (2,) representing (W,H) of the image
        # Return:
        #       anchors: (Tensor) of shape (#Anchors, 4) that holds the anchors of a given image
        #               each anchor is represented as (x_center, y_center, width, height)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        w, h = img_shape
        # Steps according to the FPN feature maps receptive field
        steps = torch.Tensor([2 ** i for i in range(3, 7 + 1)])

        n_fmaps = len(self.areas)
        n_anchors = len(self.scales)*len(self.ratios)

        # Compute width and height of boxes across all areas,scales,ratios
        boxW = torch.sqrt(self.areas_t.view(-1,1,1) * (self.scales_t.view(-1,1) / self.ratios_t))             # shape:(areas,scales,ratios)
        boxH = boxW * self.ratios_t                                                                       # shape:(areas,scales,ratios)
        boxWH = torch.cat([boxW.view(n_fmaps, -1).unsqueeze(-1), boxH.view(n_fmaps, -1).unsqueeze(-1)], dim=2)

        # Compute number of cells in each row and column for every feature map
        cellsCol = torch.ceil(w / steps).type(torch.int)
        cellsRow = torch.ceil(h / steps).type(torch.int)

        # TODO: Replace concatenate with pre-allocating memory for anchors
        # TODO: Check other method of scaling sizes of anchors with ratios
        anchors = []
        for i in range(n_fmaps):
            # Create coordinate grid
            # grid = np.array(np.meshgrid(range(cellsCol[i]), range(cellsRow[i]))).T.reshape(-1,2) + 0.5        # shape:(cellW*cellH,2)
            grid_x, grid_y = torch.meshgrid(torch.Tensor(range(cellsRow[i])), torch.Tensor(range(cellsCol[i])))
            grid = torch.cat([grid_x.unsqueeze(2), grid_y.unsqueeze(2)], dim=2).view(-1,2) + 0.5

            # Multiply grid by edge lengths
            boxCenters = steps[i].repeat(n_anchors).view(n_anchors, 1, 1) * grid                       # shape:(scales*ratios,cellW*cellH,2)
            boxCenters = torch.flip(boxCenters.permute(1,0,2).reshape(-1,2), [1])                         # shape:(scales*ratios*cellW*cellH,2)

            # Stitch grid of centers and dimensions of anchors
            anchors.append(torch.cat([boxCenters, boxWH[i].repeat(cellsCol[i]*cellsRow[i],1)], dim=1))

        anchors = torch.cat(anchors, dim=0)
        return anchors


    def parameterize(self, gt_annots, img_shape):
        # Parameterize ground truth annotations (x_center,y_center,w,h) to bounding box regression
        # offset (tx,ty,tw,th) according to the FasterRCNN paper.
        #               tx = (x-x_a)/w_a, ty = (y-y_a)/h_a
        #               tw = log(w/w_a), th = log(h/h_a)
        # Args:
        #       gt_annots: (Tensor) shaped (#objects, 4+c) representing the bounding boxes and classifications
        #                   of the objects in the image
        #       img_shape: (Tensor) shaped (2,) representing (W,H) of the image
        # Return:
        #       param_annots: (Tensor) shaped (#Anchors, 4+c), parametrized anchor box offsets
        POS_THRESH = 0.5
        NEG_THRESH = 0.4

        # anchors = self.create_anchors(img_shape)
        anchors = self.create_anchors_tensor(img_shape)

        # convert anchors to xyxy before iou calculation
        utils.xywh2xyxy(anchors)

        # calculate IoU
        ious = self.iou(anchors, gt_annots)
        iou_max, iou_argmax = torch.max(ious, dim=1)

        obj_idx = iou_max > POS_THRESH
        bg_idx = iou_max < NEG_THRESH
        ignore_idx = 1 - (obj_idx | bg_idx)

        # convert boxes and anchors to xywh for parameterization
        utils.xyxy2xywh(anchors)
        utils.xyxy2xywh(gt_annots)

        # couple anchor box with object that shares largest IoU
        param_annots = gt_annots[iou_argmax]

        # compute parameters tx,ty,tw,th
        param_annots[:, :2] = (param_annots[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        param_annots[:, 2:4] = torch.log(param_annots[:, 2:4] / anchors[:, 2:])

        # set background class to anchors with iou < NEG_THRESH
        param_annots[bg_idx, 4:] = 0

        # ignore anchors with NEG_THRESH < iou < POS_THRESH
        param_annots[ignore_idx, 4:] = -1
        return param_annots


    def deparameterize(self, param_annots, img_shape):
        # Compute boxes from parameterized predicted offsets and infer related classes.
        # offset (tx,ty,tw,th) according to the FasterRCNN paper.
        #               tx = (x-x_a)/w_a, ty = (y-y_a)/h_a
        #               tw = log(w/w_a), th = log(h/h_a)
        # Args:
        #       para_annots: (Tensor) shaped (#anchors, 4+c), parameterized offsets and class prob of each anchor.
        #       img_shape: (Tensor) shaped (2,) representing (W,H) of the image
        # Return:
        #       annots: (Tensor) predicted annotations (x,y,w,h,class)

        CLASS_THRESH = 0.5
        NMS_THRESH = 0.5

        pred_offsets, pred_classes = param_annots

        # Infer classes
        prob, colors = pred_classes[0].max(dim=1)
        obj_idxs = prob > CLASS_THRESH

        # Infer bounding boxes
        anchors = torch.Tensor(self.create_anchors(img_shape)).cuda()
        bboxes = torch.zeros(anchors.shape).cuda()
        bboxes[:, :2] = (pred_offsets[0][:, :2] * anchors[:, 2:]) + anchors[:, :2]
        bboxes[:, 2:4] = torch.exp(pred_offsets[0][:, 2:4]) * anchors[:, 2:]
        utils.xywh2xyxy(bboxes)

        candidate_boxes = bboxes[obj_idxs]
        candidate_colors = colors[obj_idxs]

        keep = self.nms(candidate_boxes, candidate_colors, NMS_THRESH)

        return candidate_boxes[keep], candidate_colors[keep]


    def iou(self, boxes1, boxes2):
        # Calculate IoU for each anchor and bounding box combination
        #
        # Args:
        #       boxes1,boxes2: (Tensor) of shape (m,4),(n,4) representing boxes (x_tl, y_tl, x_br, y_br)
        # Return:
        #       ious: (Tensor) of shape (#boxes1, #boxes2) representing the IoUs between box in row i and box in col j


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


    def nms(self, dets, scores, thresh):
        # Non Maximum Suppression for tensors
        # Reference: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.argsort(scores, descending=True)

        keep = []
        while order.numel() > 0 and len(order.shape) != 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i].data)
            yy1 = y1[order[1:]].clamp(min=y1[i].data)
            xx2 = x2[order[1:]].clamp(max=x2[i].data)
            yy2 = y2[order[1:]].clamp(max=y2[i].data)

            w = (xx2 - xx1 + 1).clamp(min=0)
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            ids = (ovr <= thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        return torch.LongTensor(keep)

if __name__ == '__main__':
    # train_data = BusDataset(
    #     '../data',
    #     '../annotationsTrain.txt',
    #     6,
    #     transforms.Compose([
    #         VerticalFlip(0),
    #         HorizontalFlip(0),
    #         Resize(minside=512),
    #     ]))
    # img, annots = train_data[0]
    # box = Anchors()
    # # box.parameterize(annots, img.shape[1:][::-1])
    # box.create_anchors_tensor(img.shape[1:][::-1])

    # Anchors TEST
    # anchors = box.create_anchors(img.shape[1:][::-1])
    # utils.xywh2xyxy(anchors)
    # utils.show_annotations(img.numpy().transpose(1,2,0), anchors, format='xyxy')
    pass

