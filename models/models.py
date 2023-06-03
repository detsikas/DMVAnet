import sys
from base_unet import base_unet
from res_unet import residual_unet
from visual_attention_residual_unet import visual_attention_residual_unet
from multires_visual_attention_unet import multires_visual_attention
from unet_pp import unet_pp
from dense_unet import dense_unet
from dilated_resnet import dilated_resnet
from deeplabv3plus import deeplabv3plus
from dilated_multires_visual_attention_unet import dilated_multires_visual_attention
from dilated_visual_attention_res_unet import dilated_visual_attention_residual_unet
from res_unet_mobilenet_v2_encoder import residual_unet_with_mobilenet_v2_pretrained_input
from res_unet_vgg19_encoder import residual_unet_with_vgg19_pretrained_input
from multires_unet_vgg19_encoder import multires_unet_with_vgg19_pretrained_input

BASE_UNET = 'base_uent'
RES_UNET = 'res_unet'
VIS_ATTN_UNET = 'visual_attention_unet'
MULTIRES_UNET = 'multires_unet'
UNET_PP = 'unet_++'
DENSE_UNET = 'dense_unet'
DILATED_RESNET = 'dilated_resnet'
DEEPLABV3PLUS = 'deeplabv3+'
DILATED_VIS_ATTN_UNET = 'dilated_visual_attention_unet'
DILATED_MULTIRES = 'dilated_multires_unet'
PRETRAINED_MOBILENET_V2_ENCODER_RES_UNET = 'pretrained_mobilenet_v2_encoder_residual_unet'
PRETRAINED_VGG19_ENCODER_RES_UNET = 'pretrained_vgg19_encoder_res_unet'
PRETRAINED_VGG19_ENCODER_MULTIRES_UNET = 'pretrained_vgg19_encoder_multires_unet'


def build_model(model_type, model_shape):
    if model_type == BASE_UNET:
        model = base_unet(model_shape)
    elif model_type == RES_UNET:
        model = residual_unet(model_shape)
    elif model_type == VIS_ATTN_UNET:
        model = visual_attention_residual_unet(model_shape)
    elif model_type == MULTIRES_UNET:
        model = multires_visual_attention(model_shape, 16, True)
    elif model_type == UNET_PP:
        model = unet_pp(model_shape, True)
    elif model_type == DENSE_UNET:
        model = dense_unet(model_shape)
    elif model_type == DILATED_RESNET:
        model = dilated_resnet(model_shape)
    elif model_type == DEEPLABV3PLUS:
        model = deeplabv3plus(model_shape)
    elif model_type == DILATED_MULTIRES:
        model = dilated_multires_visual_attention(
            shape, 16, True)
    elif model_type == DILATED_VIS_ATTN_UNET:
        model = dilated_visual_attention_residual_unet(model_shape)
    elif model_type == PRETRAINED_MOBILENET_V2_ENCODER_RES_UNET:
        model, _ = residual_unet_with_mobilenet_v2_pretrained_input(
            model_shape)
    elif model_type == PRETRAINED_VGG19_ENCODER_RES_UNET:
        model, _ = residual_unet_with_vgg19_pretrained_input(
            model_shape)
    elif model_type == PRETRAINED_VGG19_ENCODER_MULTIRES_UNET:
        model, _ = multires_unet_with_vgg19_pretrained_input(
            model_shape)
    else:
        print('Bad model type')
        sys.exit(0)
    return model
