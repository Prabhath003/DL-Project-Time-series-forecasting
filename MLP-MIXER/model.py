'''
    Copyright (c) 2023 Prabhath Chellingi (CS20BTECH11038@iith.ac.in)
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size[0] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):


        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
    

class Net(nn.Module):
    def __init__(self, mixerModel, loss=nn.MSELoss()):
        super(Net, self).__init__()
        
        self.mixerModel = mixerModel
        self.loss = loss

    def calc_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)

        return self.loss(input, target)

    def forward(self, input, target):
        
        output = self.mixerModel(input)

        loss = self.calc_loss(output, target)

        return loss




if __name__ == "__main__":
    img = torch.ones([1, 3, 8, 8])

    model = MLPMixer(in_channels=3, dim=512, num_classes=1, patch_size=4, image_size=(8, 8), depth=16, token_dim=256, channel_dim=2048)
    
    model = model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img.to(device))

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
