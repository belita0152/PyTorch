import torch
import torch.nn as nn


# 1. frame -> space encoder 통과 -> average pooling -> xm (기본 modality token) 만들기
# full modality -> 각 modality에 대한 modality token 만들기

class BasicBlock(nn.Module):
    """ Space Encoder - ResNet34 (Baseline) """
    expansion = 1

    def __init__(self, in_dim, out_dim, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if downsample is None and (stride != 1 or in_dim != out_dim * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv1d(in_dim, out_dim * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_dim * self.expansion),
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, layers, n_channels=1, out_dim=512, block=BasicBlock):
        super(ResNet34, self).__init__()
        self.in_planes = 64

        # --- initial conv + pool ---
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # --- residual layers ---
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, out_dim)


    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        # if input and output dims differ or stride != 1, we need a projection
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride=1, downsample=None))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x : [batch_size, in_channels, seq_len] : torch.Size([16, 3, 3000])
        x = self.conv1(x)  # torch.Size([16, 64, 1500])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([16, 64, 750])

        x = self.layer1(x)  # torch.Size([16, 64, 750])
        x = self.layer2(x)  # torch.Size([16, 128, 750])
        x = self.layer3(x)  # torch.Size([16, 256, 750])
        x = self.layer4(x)  # torch.Size([16, 512, 750])

        x = self.avgpool(x)  # torch.Size([16, 512, 1])
        x = torch.flatten(x, 1)  # torch.Size([16, 512])
        x = self.fc(x)  # torch.Size([16, 5])

        return x


def resnet34():
    model = ResNet34(n_channels=1, out_dim=512,
                     block=BasicBlock,
                     layers=[3, 4, 6, 3])

    return model


if __name__ == '__main__':
    from utils.dataloader import TorchDataset
    from torch.utils.data import DataLoader
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    dataset = TorchDataset(mode='train')
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
    baseline_model = resnet34().to(device)

    modal_idx = {
        'EEG Fpz-Cz' : 0,
        'EEG Pz-Oz' : 1,
        'EOG horizontal' : 2
    }

    for batch in train_dataloader:
        x_, y_ = batch  # torch.Size([64, 3, 3000])
        x_, y_ = x_.to(device), y_.to(device)

        x_1, x_2, x_3 = x_[:, 0, :].unsqueeze(1), x_[:, 1, :].unsqueeze(1), x_[:, 2, :].unsqueeze(1)  # torch.Size([16, 1, 3000])

        out1, out2, out3 = baseline_model(x_1), baseline_model(x_2), baseline_model(x_3)  # torch.Size([64, 512]) torch.Size([64, 512]) torch.Size([64, 512])
        print(out1.shape, out2.shape, out3.shape)

        modalities = torch.stack([out1, out2, out3], dim=1)  # torch.Size([64, 3, 512])
        print(modalities.shape)

