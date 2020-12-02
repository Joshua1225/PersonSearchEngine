from __future__ import absolute_import

from .resnet import *
from .resnext import *
from .seresnet import *
from .densenet import *
from .mudeep import *
from .hacnn import *
from .squeeze import *
from .mobilenetv2 import *
from .shufflenet import *
from .xception import *
from .inceptionv4 import *
from .nasnet import *
from .inceptionresnetv2 import *
from .vmgn_hgnn import *
from .resnet_ibn_a_hgnn import *
from .efficientnet.efficientnet import efficientnet
from .efficientnet.efficientnet_hgnn import efficientnet_hgnn


__model_factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    # 'resnet152': ResNet152,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnext50': SEResNeXt50,
    'seresnext101': SEResNeXt101,
    'resnext101': ResNeXt101_32x4d,
    'resnet50m': ResNet50M,
    'densenet121': DenseNet121,
    'squeezenet': SqueezeNet,
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
    'inceptionv4': InceptionV4,
    'nasnsetmobile': NASNetAMobile,
    'inceptionresnetv2': InceptionResNetV2,
    'mudeep': MuDeep,
    'hacnn': HACNN,
    'vmgn_hgnn': vmgn_hgnn,
    'resnet_ibn_a_hgnn': resnet101_ibn_a,
    'efficientnet': efficientnet,
    'efficientnet_hgnn': efficientnet_hgnn
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)