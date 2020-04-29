import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        
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

def weights_init(m, act_type='relu'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_factor)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    

class Man_Concatenate(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6,
                 padding_type='reflect' , addition_res=False):
        assert (n_blocks >= 0)
        super(Man_Concatenate, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)


        En = [nn.ReflectionPad2d(3),
              nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                        bias=use_bias),
              norm_layer(ngf),
              self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if addition_res:
                for j in range(2):
                    En += [ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                             stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   self.act]  # 75 * 75

        mult = 2 ** n_downsampling
        self.extra_conv1 = nn.Conv2d(ngf * mult + 10 ,ngf *mult ,kernel_size=3, stride=1, padding=1)
        self.extra_bn1 = nn.BatchNorm2d(ngf * mult)

        self.res1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)

        Dn = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout,
                                       use_bias=use_bias)]
        Dn += [nn.ReflectionPad2d(3)]
        
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Tanh()]

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)
    def forward(self, input ,label):
        mid = self.En(input)
        t = torch.zeros(mid.size(0),10,8,8)
        label = label.view(mid.size(0), 10, 1, 1).expand_as(t)
        mid = torch.cat([mid ,label] ,1)
        mid = self.extra_conv1(mid)
        mid = self.extra_bn1(mid)
        mid = self.act(mid)
        mid = self.res1(mid)
        mid = self.res2(mid)
        mid = self.res3(mid)
        mid = self.res4(mid)
        mid = self.res5(mid)
        mid = self.res6(mid)
        per = self.Dn(mid)
        return per

        
class Man_Recalibrate(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6,
                 padding_type='reflect' , addition_res=False):
        assert (n_blocks >= 0)
        super(Man_Recalibrate, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        En = [nn.ReflectionPad2d(3),
              nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                        bias=use_bias),
              norm_layer(ngf),
              self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if addition_res:
                for j in range(2):
                    En += [ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                       use_bias=use_bias)]
            En += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                             stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   self.act]  # 75 * 75

        mult = 2 ** n_downsampling

        self.res1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)
        self.fc1 = nn.Linear(10, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(96, 64)
        self.fc5 = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        Dn = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Dn += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   self.act]
            if addition_res:
                for j in range(2):
                    Dn += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout,
                                       use_bias=use_bias)]
        Dn += [nn.ReflectionPad2d(3)]
        
        Dn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        Dn += [nn.Tanh()]

        self.En = nn.Sequential(*En)
        self.Dn = nn.Sequential(*Dn)

    def forward(self, input, label):
        weight = self.act(self.fc1(label))
        weight1 = self.fc2(weight)
        weight1_1 = self.sigmoid(weight1)
        weight1_2 = self.sigmoid(self.fc3(self.act(weight1)))
        weight1_1 = weight1_1.view(input.size(0), -1, 1, 1)
        weight1_2 = weight1_2.view(input.size(0), -1, 1, 1)
        weight2 = self.fc4(weight)
        weight2_1 = self.sigmoid(weight2)
        weight2_2 = self.sigmoid(self.fc5(self.act(weight2)))
        weight2_1 = weight2_1.view(input.size(0), -1, 1, 1)
        weight2_2 = weight2_2.view(input.size(0), -1, 1, 1)
        mid = self.En(input)
        mid = self.res1(mid * weight1_1)
        mid = self.res2(mid * weight1_2)
        mid = self.res3(mid)
        mid = self.res4(mid)
        mid = self.res5(mid * weight2_1)
        mid = self.res6(mid * weight2_2)
        
        per = self.Dn(mid)
        return per

