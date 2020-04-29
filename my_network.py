import torch
from torchvision import models

from models import identity_map_resnet_for_cifar as ident_cifar
from models import resnet_cifar as res_cifar
from models import cifar_acm as res_acm
from models import cnn_12_layer as cnn_12
from models import vgg_cifar as vgg
def load_pretrained_model(network):
    if network == 'resnet_50':
        net = models.resnet50(pretrained=True)
    elif network == 'resnet_18':
        net = models.resnet18(pretrained=True)
    elif network == 'alexnet':
        net = models.alexnet(pretrained=True)
    else:
        raise ValueError ('invalid network name')
    return net

def load_scratch_model(network):
    if network == 'resnet_50':
        net = models.resnet50(pretrained=False)
    elif network == 'resnet_18':
        net = models.resnet18(pretrained=False)
    elif network == 'alexnet':
        net = models.alexnet(pretrained=False)
    elif network == 'cnn_12':
        net = cnn_12.cnn_12_layer()
    elif network == 'vgg16':
        net = vgg.vgg16()
    elif network == 'vgg19':
        net = vgg.vgg19()
    elif network == 'vgg19_bn':
        net = vgg.vgg19_bn()
    elif network == 'pre_resnet_32_cifar_10':
        net = res_cifar.preact_resnet32_cifar()
    elif network == 'acm_resnet_32_cifar_10':
        net = res_acm.resnet32()
    elif network == 'acm_resnet_14_cifar_10':
        net = res_acm.resnet14()
    elif network == 'iden_resnet_32_cifar_10':
        net = ident_cifar.resnet32()
    elif network == 'iden_resnet_14_cifar_10':
        net = ident_cifar.resnet14()
    elif network == 'iden_resnet_110_cifar_10':
        net = ident_cifar.resnet110()
    elif network == 'iden_resnet_164_cifar_10':
        net = ident_cifar.resnet164()
    elif network == 'iden_resnet_44_cifar_100':
        net = res_cifar.preact_resnet44_cifar(num_classes=100)
    else:
        raise ValueError('invalid network name')
    return net