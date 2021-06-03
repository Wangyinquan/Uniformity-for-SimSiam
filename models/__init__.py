from .simsiam import SimSiam
from .simsiam_f import SimSiam_f
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg,outdim):    

    if model_cfg.name == 'simsiam':
        model =  SimSiam(model_cfg, get_backbone(model_cfg.backbone),outdim)
#         if model_cfg.proj_layer is not None:
#             model.projector.set_layer(model_cfg.proj_layer)
    elif model_cfg.name =='simsiam_f':
        model =  SimSiam_f(model_cfg, get_backbone(model_cfg.backbone),outdim,model_cfg.rho)
    else:
        raise NotImplementedError
    return model






