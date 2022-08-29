import torch.nn as nn
import torchvision.models as models


class ResNet101(nn.Module):
    """ ResNet101 having the last FC layer either removed (to extract features) or modified by: FC(dim_embedding)
    to finetune. Dimension of embedding is the number of known classes (parents + leaves)"""
    def __init__(self, num_classes=1000, freeze_backbone=False, return_feat=True, finetune=False):
        """
        Args:
            return_feat (bool): If True, the output of the network are the features after the average pooling layer,
                before the fc layer, which is removed.
            finetune (bool): If True, last fc layer is modified so that its output size matches num_classes
        """
        super(ResNet101, self).__init__()

        resnet = models.resnet101(pretrained=True)
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        self.feat_dim = resnet.fc.in_features

        if return_feat:
            resnet.fc = Identity()
        elif finetune:
            resnet.fc = nn.Linear(self.feat_dim, num_classes)  # requires_grad is now True
            if freeze_backbone:
                self.unfrozen_params = ['module.backbone.fc.weight', 'module.backbone.fc.bias']  # these are not trained
            else:
                self.unfrozen_params = ['module.'+p for p in self.state_dict()]

        self.backbone = resnet

    def forward(self, x):
        x = self.backbone(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FeatureExtractor:
    """
    Registers a forward hook on layer so that it stores its output on self.features after a forward pass
    """
    def __init__(self, layer):
        self.features = None
        layer.register_forward_hook(self.hook)

    def hook(self, model, input, output):
        self.features = output.detach()
