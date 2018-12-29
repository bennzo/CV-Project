import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class FeaturePyramidNetwork(nn.Module):
    # Building the FPN according to: https://arxiv.org/abs/1708.02002
    # Architecture details on page 4 (including footnotes)
    # Each c_i corresponds to a feature layer at the output of the ResNet layers
    # Each p_i corresponds to a feature layer at a different scale at the output of each pyramid layer
    # Each m_i corresponds to the mid-layer to be upsampled in the top-down path
    # A nice illustration by @jonathan_hui at https://cdn-images-1.medium.com/max/1600/1*ffxP_rL8-jMvipLhMJrVeA.png
    def __init__(self, channel_depth, fpn_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        c3_d, c4_d, c5_d = channel_depth

        self.p6_conv = nn.Conv2d(c5_d, fpn_channels, kernel_size=3, stride=2, padding=1)

        self.p7_activation = nn.ReLU()
        self.p7_conv = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)

        self.c5_conv1 = nn.Conv2d(c5_d, fpn_channels, kernel_size=1, stride=1)
        self.p5_conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)
        self.m5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c4_conv1 = nn.Conv2d(c4_d, fpn_channels, kernel_size=1, stride=1)
        self.p4_conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)
        self.m4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c3_conv1 = nn.Conv2d(c3_d, fpn_channels, kernel_size=1, stride=1)
        self.p3_conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, c3, c4, c5):
        p6 = self.p6_conv(c5)

        p7 = self.p7_activation(p6)
        p7 = self.p7_conv(p7)

        m5 = self.c5_conv1(c5)
        p5 = self.p5_conv3(m5)

        m4 = self.m5_upsample(m5) + self.c4_conv1(c4)
        p4 = self.p4_conv3(m4)

        m3 = self.m4_upsample(m4) + self.c3_conv1(c3)
        p3 = self.p3_conv3(m3)

        return [p3, p4, p5, p6, p7]


class ClassificationSubnet(nn.Module):
    # Building the Classification subnet according to: https://arxiv.org/abs/1708.02002
    # Architecture details on page 5
    def __init__(self, fpn_channels, n_anchors, n_classes):
        super(ClassificationSubnet, self).__init__()
        self.conv1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.class_conv = nn.Conv2d(fpn_channels, n_anchors * n_classes, kernel_size=3, stride=1, padding=0)

        self.relu = nn.ReLU
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = self.class_conv(x)
        output = self.sigmoid(x)

        return output


class BoxSubnet(nn.Module):
    # Building the Box Regression subnet according to: https://arxiv.org/abs/1708.02002
    # Architecture details on page 5
    def __init__(self, fpn_channels, n_anchors):
        super(BoxSubnet, self).__init__()
        self.conv1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=0)
        self.box_conv = nn.Conv2d(fpn_channels, n_anchors * 4, kernel_size=3, stride=1, padding=0)

        self.relu = nn.ReLU
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        output = self.class_conv(x)

        return output


class ResNet(nn.Module):
    def __init__(self, n_classes, block, layers):
        # ----------- Initialize ResNet layers ----------- #
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize ResNet layer's weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # ----------------------------------------- #

        # ----------- Initialize Feature Pyramid Network layers ----------- #
        # Channel depths according to Bottleneck block
        channel_depth = [self.layer2[-1].conv3.out_channels,
                         self.layer3[-1].conv3.out_channels,
                         self.layer4[-1].conv3.out_channels]
        self.fpn_channels = 256
        self.FPN = FeaturePyramidNetwork(channel_depth, self.fpn_channels)
        # ----------------------------------------------------------------- #

        # ----------- Initialize Regression Network ----------- #
        self.BoxRegression = BoxSubnet(self.fpn_channels, 9)
        # ----------------------------------------------------- #

        # ----------- Initialize Classification Network ----------- #
        self.BoxClassification = ClassificationSubnet(self.fpn_channels, 9, n_classes)
        # --------------------------------------------------------- #

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, images, annots_gt):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p3, p4, p5, p6, p7 = self.FPN(c3,c4,c5)

        return x


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out