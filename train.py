import argparse
import os

from torch.utils.data import DataLoader
from torch.optim import Adam

from model.net import resnet101, resnet50
from model.data_loader import BusDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='Data directory path')
    parser.add_argument('--annotations', type=str, default='annotationsTrain.txt', help='Annotations text file path')

    parser.add_argument('--checkpointdir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--name', type=str, default='RetinaNet', help='Name of the model')
    parser.add_argument('--checkpoint', type=int, default='10', help='Number of epochs before saving a model')
    parser.add_argument('--pretrained', action='store_true', help='Continues training on checkpoint-dir/name model')

    parser.add_argument('--resnet', type=int, default=50, help='Depth of the resnet model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train on')
    parser.add_argument('--lr', type=int, default=0.0001, help='Number of epochs to train on')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of epochs to train on')

    parser.add_argument('--no_gpu', action='store_true', help='Disable gpu usage')
    opts = parser.parse_args()

    if opts.resnet == 101:
        RetinaNet = resnet101(7, pretrained=False)
    if opts.resnet == 50:
        RetinaNet = resnet50(7, pretrained=False)

    if not opts.no_gpu:
        RetinaNet.cuda()

    train_data = DataLoader(BusDataset(opts.data, opts.annotations))
    optimizer = Adam(RetinaNet.parameters(), lr=opts.lr)

    cumulative_stats = {}
    for epoch in range(opts.epochs):
        stats = {}
        for i, data in enumerate(train_data):
            optimizer.zero_grad()

            img_batch, annots_gt = data
            annots_prop = RetinaNet(img_batch)

            # loss.backward()
            optimizer.step()




if __name__ == '__main__':
    main()