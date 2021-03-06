import numpy as np
import ast
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.net import resnet152, resnet101, resnet50
from model.data_loader import BusDataset, Resize, Normalize, collate



def run(myAnnFileName, buses):
    annFileEstimations = open(myAnnFileName, 'w+')

    # Initialize and load model
    # net = resnet50(6, pretrained=False, pre_retnet=True)
    # net = resnet101(6, pretrained=False, pre_retnet=True)
    net = resnet152(6, pretrained=False, pre_retnet=True)
    net.load_state_dict(torch.load('saved_models/RetinaNet_final_resnet152_annotationsTrain.txt_525.pt'))
    net = net.cuda()
    net.eval()
    net.freeze_bn()

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Initialize dataset
    test_data = BusDataset(buses, '', 6,
                            transforms.Compose([
                                Resize(minside=512),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]),
                            eval=True
                           )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_data.collate)

    for i, (images, gt_annots, scales) in enumerate(test_loader):
        with torch.no_grad():
            images = images.cuda()
            # gt_annots = gt_annots.cuda()
            scales = scales.cuda()


            pred_annots = net(images)

            boxes, colors = test_data.anchors.deparameterize(pred_annots, images[0].shape[1:][::-1])

            if len(boxes) > 0:
                # Rescale Boxes
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (1/scales[0,0])
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (1/scales[0,1])

                # Convert to XYWH
                boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, :2]

                # Restore Compliant Colors
                colors = colors + 1

                # Stitch annotations
                annotations = torch.cat([boxes.round().long(), colors.unsqueeze(1)], dim=1).tolist()
            else:
                annotations = []

            aline = test_data.get_filename(i) + ':' + ','.join(map(str,annotations[::-1])) + '\n'
            annFileEstimations.write(aline)
    annFileEstimations.close()



