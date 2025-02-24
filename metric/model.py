import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, generation_model=models.efficientnet_b0, weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,in_channels=24, num_classes=2):
        super(EfficientNet, self).__init__()
        self.model = generation_model(weights= weights)
        first_layer = self.model.features[0][0]
        new_first_layer = nn.Conv2d(in_channels=12, out_channels=first_layer.out_channels,
                      kernel_size=first_layer.kernel_size,
                      stride=first_layer.stride,
                      padding=first_layer.padding,
                      bias=False)
        assert (in_channels % 3 == 0)
        # TODO : For the moment just repeat original weight, might change later
        repeat = in_channels // 3
        new_first_layer.weight.data = first_layer.weight.data.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
        self.model.features[0][0] = new_first_layer
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
