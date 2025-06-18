from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block, VisionTransformer

class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            num_classes = 0,
            in_chans = 3,
            patch_size = 4,
            img_size=32,
        )

        self.patch_size = 4
        self.in_chans = 3
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)  # tensor 안에 값을 정규화

    def forward(self, x, mask):
        x = self.patch_embed(x)  # x -> pxp patches

        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)  # torch.Size([128, 64, 768])

        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)  # (B, L) -> (B, L, 1)
        x = x * (1 - w) + mask_token * w  # 마스크되지 않은 패치는 embedding x 유지 + 마스크된 패치는 mask_token으로 대체

        cls_tokens = self.cls_token.expand(B, -1, -1)  # ViT에서는 cls_token을 유지 / (1, 1, C) -> (B, 1, C)로 변경
        x = torch.cat((cls_tokens, x), dim=1)  # 기존 패치 앞에 cls token 연결 => (B, L+1, C)

        x = x + self.pos_embed
        x = self.pos_drop(x)  # Transformer 입력 전에 dropout -> 과적합 방지

        for blk in self.blocks:
            x = blk(x)  # Transformer block 만들기
        x = self.norm(x)

        x = x[:, 1:]  # cls token 제거
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride  # 인코더 입출력 해상도 비율 (ViT = 16)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),  # upsampling + shuffle 차원 순서 변경
        )  # SimMIM의 특징 ***** : Decoder = Conv + PixelShuffle => 매우 얇다

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        B, C, H, W = x_rec.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        mask = mask.view(B, ph, pw)  # torch.Size([128, 64, 768])

        mask = (mask.repeat_interleave(self.patch_size, 1)  # dim=1 방향으로 각 행을 patch_size 번 복제: (B, H/ps, W/ps) -> (B, H, W/ps)
                .repeat_interleave(self.patch_size, 2))  # dim=2 방향으로 각 행을 patch_size 번 복제: (B, H, W/ps) -> (B, H, W)
                                                         # idx=1 자리에 dim 추가: (B, H, W) -> (B, 1, H, W)
        mask = mask.unsqueeze(1).contiguous()

        loss_recon = F.l1_loss(x, x_rec, reduction='none')  # x와 복원시킨 x 사이의 L1 loss 계산
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans  # 마스크된 픽셀만 손실 계산 + 채널 수로 나누어 채널별 정규화
        return loss  # masking 영역의 평균 pixel L1 loss 반환


if __name__ == '__main__':
    from torch import optim as optim
    import torch
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = './checkpoint_simmim'

    transform = transforms.Compose(
        [transforms.ToTensor(),  # tensor + (C, H, W) shape으로 변환, 범위 [0, 1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 범위 [-1, 1]로 재조정

    batch_size = 128
    trainset = CIFAR10(root='./data', train=True, download=False, transform=transform)  # (10000, 32, 32, 3)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = CIFAR10(root='./data', train=False, download=False, transform=transform)  # (10000, 32, 32, 3)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 10


    encoder = VisionTransformerForSimMIM(in_chans=3).to(device)
    encoder_stride = 16
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    train_batch_accumulation = 1
    eff_batch_size = batch_size * train_batch_accumulation
    print_point = 20
    patch_size_ = 4
    mask_ratio = 0.6

    total_step = 0
    epoch, n_epochs = 0, 50
    for epoch in range(n_epochs):
        step = 0
        model.train()
        optimizer.zero_grad()
        for img, mask in trainloader:
            img, mask = img.to(device), mask.to(device)  # (B, C, H, W): [2, 3, 32, 32] / (B, ): [2]
            B, C, H, W = img.shape
            num_patches_ = (H // patch_size_) * (W // patch_size_)
            patch_mask = (torch.rand(B, num_patches_, device=device) < mask_ratio).long()
            patch_mask = (torch.rand(B, num_patches_, device=device) < mask_ratio).long()

            recon_loss = model(img, patch_mask)  # torch.Size([2, 17, 1024]) torch.Size([2, 64, 48]) torch.Size([2, 64])
            recon_loss.backward()

            if (step + 1) % train_batch_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (total_step + 1) % print_point == 0:
                print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                      '[Reconstruction Loss] : {2:02.4f}  '.format(
                    epoch, total_step + 1, recon_loss))

        model.eval()
        total_x, total_y = [], []
        for img, mask in testloader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            total_x.append(out)
            total_y.append(mask)
        total_x = torch.stack(total_x, dim=0)
        total_y = torch.stack(total_y, dim=0)
        total_y = total_y.detach.cpu.numpy()

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(total_x, total_y)
        out = knn.predict(total_x)
        from sklearn.metrics import classification_report
        print(classification_report(total_y, out))

    torch.save(model.state_dict(), f'{ckpt_dir}/MAE2_Epoch{epoch + 1}.pth')
