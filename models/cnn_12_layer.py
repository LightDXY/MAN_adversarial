
import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class cnn_12_layer(nn.Module):
    def __init__(self):
        super(cnn_12_layer, self).__init__()
        self.conv1_1 = conv3x3(3, 64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = conv3x3(64, 64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2_1 = conv3x3(64, 128)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = conv3x3(128,128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.maxpool_2 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv3_1 = conv3x3(128,196)
        self.bn3_1 = nn.BatchNorm2d(196)
        self.conv3_2 = conv3x3(196,196)
        self.bn3_2 = nn.BatchNorm2d(196)
        self.maxpool_3 = nn.MaxPool2d((2,2), stride=(2,2))

        self.linear_1 = nn.Linear(196*4*4 , 256)
        self.bn4 = nn.BatchNorm2d(256)
        self.linear_2 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.maxpool_1(x)
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.maxpool_2(x)
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.maxpool_3(x)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.bn4(x)
        x = self.relu(x)
        feature = x
        x = self.linear_2(x)
        return x 
