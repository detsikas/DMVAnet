import pickle
from enum import Enum, auto
import sys
import common.models as models


class ModelType(Enum):
    BASE_UNET = auto()
    RES_UNET = auto()
    VIS_ATTN_UNET = auto()
    MULTIRES_UNET = auto()
    UNET_PP = auto()
    DENSE_UNET = auto()
    DILATED_RESNET = auto()
    DEEPLABV3PLUS = auto()
    DILATED_VIS_ATTN_UNET = auto()
    DILATED_MULTIRES = auto()
    PRETRAINED_MOBILENET_V2_ENCODER_RESNET_UNET = auto()
    PRETRAINED_VGG19_ENCODER_RESNET_UNET = auto()
    PRETRAINED_VGG19_ENCODER_MULTIRESNET_UNET = auto()


class Info:
    def __init__(self, shape, type):
        self.shape = shape
        self.type = type


def write_info(info, filename):
    with open(filename, 'wb') as file:
        pickle.dump(info, file)


def read_info(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def restore_model(model_info):
    if model_info.type == ModelType.BASE_UNET:
        model = models.base_unet(model_info.shape)
    elif model_info.type == ModelType.RES_UNET:
        model = models.residual_unet(model_info.shape)
    elif model_info.type == ModelType.VIS_ATTN_UNET:
        model = models.visual_attention_residual_unet(model_info.shape)
    elif model_info.type == ModelType.MULTIRES_UNET:
        model = models.multires_visual_attention(model_info.shape, 16, True)
    elif model_info.type == ModelType.UNET_PP:
        model = models.unet_pp(model_info.shape, True)
    elif model_info.type == ModelType.DENSE_UNET:
        model = models.dense_unet(model_info.shape)
    elif model_info.type == ModelType.DILATED_RESNET:
        model = models.dilated_resnet(model_info.shape)
    elif model_info.type == ModelType.DEEPLABV3PLUS:
        model = models.deeplabv3plus(model_info.shape)
    elif model_info.type == ModelType.DILATED_MULTIRES:
        model = models.dilated_multires_visual_attention(model_info.shape, 16, True)
    elif model_info.type == ModelType.DILATED_VIS_ATTN_UNET:
        model = models.dilated_visual_attention_residual_unet(model_info.shape)
    elif model_info.type == ModelType.PRETRAINED_MOBILENET_V2_ENCODER_RESNET_UNET:
        model, _ = models.residual_unet_with_mobilenet_v2_pretrained_input(model_info.shape)
    elif model_info.type == ModelType.PRETRAINED_VGG19_ENCODER_RESNET_UNET:
        model, _ = models.residual_unet_with_vgg19_pretrained_input(model_info.shape)
    elif model_info.type == ModelType.PRETRAINED_VGG19_ENCODER_MULTIRESNET_UNET:
        model, _ = models.multires_unet_with_vgg19_pretrained_input(model_info.shape)
    else:
        print('Bad model type')
        sys.exit(0)
    return model

