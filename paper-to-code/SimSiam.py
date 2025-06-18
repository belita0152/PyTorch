import torch
import torch.nn as nn
import timm

class SimSiam(nn.Module):
    def __init__(self, hidden_features):
        super(SimSiam, self).__init__()

        self.backbone_encoder = timm.create_model('resnet34', pretrained=True)
        self.backbone_encoder.fc = nn.Identity()

        self.in_features = 512
        self.hidden_features = hidden_features

        self.predictor = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),  # Linear
            nn.BatchNorm1d(self.hidden_features),  # BatchNorm
            nn.GELU(),  # Activation
            nn.Linear(self.hidden_features, self.in_features)
        )

    def forward(self, x1, x2):
        p1 = self.backbone_encoder(x1)
        p2 = self.backbone_encoder(x2)

        z1 = self.predictor(p1)
        z2 = self.predictor(p2)

        return p1, p2, z1.detach(), z2.detach()


if __name__ == '__main__':
    model = SimSiam(256)
    p1, p2, z1, z2 = model(
        torch.randn(64, 3, 32, 32),
        torch.randn(64, 3, 32, 32),
    )
    print(p1.shape)
    print(p2.shape)
    print(z1.shape)
    print(z2.shape)
