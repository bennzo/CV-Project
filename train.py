import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision import transforms

from model.net import resnet101, resnet50
from model.data_loader import BusDataset, HorizontalFlip, VerticalFlip, Resize, Normalize, collate
from model.loss import  RetinaLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='Data directory path')
    parser.add_argument('--annotations', type=str, default='annotationsTrain.txt', help='Annotations text file path')

    parser.add_argument('--checkpointdir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--name', type=str, default='RetinaNet_time', help='Name of the model')
    parser.add_argument('--checkpoint', type=int, default='10', help='Number of epochs before saving a model')
    parser.add_argument('--pretrained', action='store_true', help='Continues training on checkpoint-dir/name model')

    parser.add_argument('--resnet', type=int, default=50, help='Depth of the resnet model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on')
    parser.add_argument('--lr', type=int, default=0.00001, help='Number of epochs to train on')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs to train on')

    parser.add_argument('--no_gpu', action='store_true', help='Disable gpu usage')
    opts = parser.parse_args()

    cuda = not opts.no_gpu

    if opts.resnet == 101:
        net = resnet101(6, pretrained=True)
    if opts.resnet == 50:
        net = resnet50(6, pretrained=True)

    if cuda:
        net = net.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net.freeze_bn()

    train_data = BusDataset(opts.data, opts.annotations, 6,
                            transforms.Compose([
                                #VerticalFlip(0.5),
                                # HorizontalFlip(0.5),
                                Resize(minside=512),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]))
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=1, collate_fn=train_data.collate)

    optimizer = Adam(net.parameters(), lr=opts.lr)
    criterion = RetinaLoss()

    train_loss = []
    for epoch in range(opts.epochs):
        epoch_class_loss = []
        epoch_loc_loss = []
        epoch_val_loss = []
        for i, (images, gt_annots, scales) in enumerate(train_loader):
            optimizer.zero_grad()

            if cuda:
                images = images.cuda()
                gt_annots = gt_annots.cuda()

            pred_annots = net(images)
            local_loss, class_loss = criterion(pred_annots, gt_annots)
            (local_loss + class_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()

            train_loss.append(float(local_loss+class_loss))
            epoch_class_loss.append(float(class_loss))
            epoch_loc_loss.append(float(local_loss))
            del class_loss
            del local_loss

        print(
            'epoch: {} | cls loss: {:1.15f} | loc loss: {:1.15f} | training loss: {:1.5f}'.format(
                epoch,  np.mean(epoch_class_loss), np.mean(epoch_loc_loss), np.mean(train_loss)))


        if epoch % opts.checkpoint == 0:
            torch.save(net.state_dict(), os.path.join(opts.checkpointdir, opts.name + '_resnet{}_{}_{}.pt'.format(str(opts.resnet), opts.annotations, str(epoch))))




if __name__ == '__main__':
    main()