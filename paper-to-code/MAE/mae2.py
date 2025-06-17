import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
import matplotlib.pyplot as plt


# MAE => 전체 class로 묶기
from timm.models.vision_transformer import PatchEmbed, Block
from MAE.pos_embed2 import get_2d_sincos_pos_embed

class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 encoder_embed_dim=1024, encoder_depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super(MaskedAutoEncoderViT, self).__init__()
        self.mask_ratio = 0.75
        self.mlp_ratio = 4.

        self.encoder_embed_dim = encoder_embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, self.encoder_embed_dim)
        num_patches = self.patch_embed.num_patches  # 64

        # MAE encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        self.encoder_depth = encoder_depth

        # MAE decoder
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 1. image -> patch
    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]  # 패치 사이즈. p=16 -> 16x16 patch
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0  # assert: 디버깅용 검증 굼군. False 일 때 AssertionError

        h = w = imgs.shape[2] // p  # h=w 인 정사각형 패치. 가로/세로 방향이 몇 개의 패치로 나뉘는지 계산
        #    예: H=32, p=16 이면 h=w=2 → 총 2×2=4개의 패치.

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))  # H = h*p, W = w*p => patch 단위로 차원도 쪼갬
        x = torch.einsum('nchpwq->nhwpqc', x)  # 축 순서 ㅓ바꾸기
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  # h*w = 패치 개수, '픽셀수x채널'을 펴서 하나의 특징 벡터로
        # 결과: x.shape == (N, num_patches, patch_size² × 3)
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]  # 패치 사이즈. p=16 -> 16x16 patch
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # 2. 75% masking / 25% token
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 전체 길이 L의 25% 만큼은 keep (keep할 토큰 수)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]. sort noise 기준으로 정렬

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 원래 순서로 복원할 때 사용할 인덱스 매핑

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # 유지할 len_keep 개의 토큰 잘라내기
        latent = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # 1로 binary mask 초기화 (1 = masking)
        mask[:, :len_keep] = 0  # 앞에서 유지한 len_keep 개의 토큰 -> 0으로 변경
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 원래 토큰 순서대로 마스크 배치

        return latent, mask, ids_restore  # 유지할 토큰, mask, 원래 순서로 복원할 때 쓸 인덱스

    # 3. cls token + 25% token -> encoder -> latent vector
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # 그냥 붙이는 거임

        # Transformer encoder block (여기서 block 구성)
        for block in self.encoder_block:
            x = block(x)

        latent = self.encoder_norm(x)

        return latent, mask, ids_restore

    def forward_latent(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # 그냥 붙이는 거임

        # Transformer encoder block (여기서 block 구성)
        for block in self.encoder_block:
            x = block(x)
        latent = self.encoder_norm(x)
        latent = latent[:, 0, :].squeeze()
        return latent

    # 4. cls token 버리기
    # 5. 75% mask token + 25% latent vector => decoder
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])  # torch.Size([1, 16, 512])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print(x.shape)  # torch.Size([1, 64, 512])
        # print(self.decoder_pos_embed.shape)  # torch.Size([1, 64, 512])

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    # 6. MSE loss => 25% latent vector <-> 25% original patch
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return latent, pred, mask


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    ckpt_dir = './checkpoint_mae2'

    transform = transforms.Compose(
        [transforms.ToTensor(),  # tensor + (C, H, W) shape으로 변환, 범위 [0, 1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 범위 [-1, 1]로 재조정

    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)  # (10000, 32, 32, 3)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)  # (10000, 32, 32, 3)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 10

    model = MaskedAutoEncoderViT().to(device)
    lr = 1e-4
    optimizer = opt.AdamW(model.parameters(), lr=lr)
    train_batch_accumulation = 1
    eff_batch_size = batch_size * train_batch_accumulation
    print_point = 20

    total_step = 0
    epoch, n_epochs = 0, 100
    for epoch in range(n_epochs):
        step = 0
        model.train()
        optimizer.zero_grad()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # (B, C, H, W): [2, 3, 32, 32] / (B, ): [2]

            latent, pred, mask = model(inputs)  # torch.Size([2, 17, 1024]) torch.Size([2, 64, 48]) torch.Size([2, 64])

            recon_loss = model.forward_loss(inputs, pred, mask)
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
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model.forward_latent(inputs)
            total_x.append(out)
            total_y.append(labels)
        total_x = torch.stack(total_x, dim=0).detach.cpu.numpy()
        total_y = torch.stack(total_y, dim=0).detach.cpu.numpy()
        # total_x = np.concatenate(total_x, axis=0)
        # total_y = np.concatenate(total_y, axis=0)

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(total_x, total_y)
        out = knn.predict(total_x)
        from sklearn.metrics import classification_report
        print(classification_report(total_y, out))

    torch.save(model.state_dict(), f'{ckpt_dir}/MAE2_Epoch{epoch + 1}.pth')
