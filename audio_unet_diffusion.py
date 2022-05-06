import torch
import torch.nn as nn
import math

from functools import partial


# based on Modulation2d from crowsonkb / RiversHaveWinds / alstroemeria 's v-diffusion-pytorch

class Modulation1d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['timestep_embed']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None], input, scales[..., None] + 1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)

    def forward(self, input):
        f = math.tau * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([
            nn.Linear(f_in, f_mid),
            nn.ReLU(inplace=True),
            nn.Linear(f_mid, f_out),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


# from this paper: https://arxiv.org/abs/2011.02421
# one-shot conditional audio filtering of arbitrary sounds
# using the film layers instead to condition based on diffusion timestep instead of sound embedding

class ResidualUnit(nn.Module):

    def __init__(self, num_channels, dilation):
        super().__init__()

        self.layers = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(num_channels, num_channels, kernel_size=(3,), dilation=dilation, padding=dilation,
                          padding_mode='replicate')),
            # nn.GroupNorm(16, num_channels),
            nn.ELU(inplace=True),
            nn.utils.weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=(1,))),
            # nn.GroupNorm(16, num_channels),
            nn.ELU(inplace=True),
        )

        self.skip = SkipConnection(num_channels)

    def forward(self, x):
        return self.layers(x) + self.skip(x)


class SkipConnection(nn.Module):

    def __init__(self, num_channels, output_channels=None):
        super().__init__()

        if output_channels is None:
            output_channels = num_channels

        layers = [
            nn.utils.weight_norm(nn.Conv1d(num_channels, output_channels, kernel_size=(1,))),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels, stride, mod_block, n_res_units=3, ):
        super().__init__()

        layers = []

        for i in range(n_res_units):
            layers.append(ResidualUnit(channels, 3 ** i))

        layers.append(mod_block(channels))

        layers.append(
            nn.utils.weight_norm(
                nn.Conv1d(channels, 2 * channels, kernel_size=(2 * stride,), stride=(stride,), padding=stride // 2,
                          padding_mode='replicate')))

        # layers.append(nn.GroupNorm(16, 2 * channels))
        layers.append(nn.ELU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels, stride, mod_block, n_res_units=3):
        super().__init__()

        layers = []

        layers.append(mod_block(channels))

        layers.append(
            nn.utils.weight_norm(
                nn.ConvTranspose1d(channels, channels // 2, kernel_size=(2 * stride,), stride=(stride,),
                                   padding=stride))
        )

        layers.append(nn.ReplicationPad1d(stride // 2))

        # layers.append(nn.GroupNorm(16, channels // 2))
        layers.append(nn.ELU(inplace=True))

        for i in range(n_res_units):
            if (i + 1) == n_res_units:
                layers.append(mod_block(channels // 2))
            layers.append(ResidualUnit(channels // 2, 3 ** i))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioUNet(nn.Module):

    def __init__(self, base_channels=32, input_channels=1, output_channels=1, n_res_units=3, cond_width=128,
                 cond_layers=2):
        super().__init__()

        self.state = {}

        self.modulation = partial(Modulation1d, self.state, cond_width)

        self.timestep_embed = FourierFeatures(1, cond_width)

        self.mapping = nn.Sequential(*[ResLinearBlock(cond_width, cond_width, cond_width) for _ in range(cond_layers)])

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(input_channels, base_channels, kernel_size=(7,), padding=3, padding_mode='replicate')),
            # nn.GroupNorm(16, base_channels),
            nn.ELU(inplace=True)
        )

        self.input_skip = SkipConnection(input_channels, output_channels)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(base_channels, stride=2, n_res_units=n_res_units, mod_block=self.modulation),
            EncoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units, mod_block=self.modulation),
            EncoderBlock(base_channels * 4, stride=8, n_res_units=n_res_units,  mod_block=self.modulation),
            EncoderBlock(base_channels * 8, stride=8, n_res_units=n_res_units, mod_block=self.modulation)
        ])

        self.middle_layers = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=(7,), padding=3,
                                           padding_mode='replicate')),
            # nn.GroupNorm(16, base_channels * 16),
            nn.ELU(inplace=True),
            self.modulation(base_channels * 16),
            nn.utils.weight_norm(nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=(7,), padding=3,
                                           padding_mode='replicate')),
            # nn.GroupNorm(16, base_channels * 16),
            nn.ELU(inplace=True),
        )

        self.middle_skip = SkipConnection(base_channels * 16)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(base_channels * 16, stride=8, n_res_units=n_res_units, mod_block=self.modulation),
            DecoderBlock(base_channels * 8, stride=8, n_res_units=n_res_units, mod_block=self.modulation),
            DecoderBlock(base_channels * 4, stride=2, n_res_units=n_res_units, mod_block=self.modulation),
            DecoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units, mod_block=self.modulation),
        ])

        self.decoder_skips = nn.ModuleList([
            SkipConnection(base_channels * 8),
            SkipConnection(base_channels * 4),
            SkipConnection(base_channels * 2),
            SkipConnection(base_channels),
        ])

        self.out_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(base_channels, output_channels, kernel_size=(7,), padding=3, padding_mode='replicate'))
        )

    def forward(self, input, timestep, condition_audio):
        # compute timestep embedding
        timestep_embed = self.mapping(self.timestep_embed(timestep))
        self.state['timestep_embed'] = timestep_embed

        # concatenate on channel dimension
        input = torch.cat([input, condition_audio], dim=1)

        # skip connections
        xs = []
        x = self.input_conv(input)

        for block in self.encoder_blocks:
            xs.append(x)
            x = block(x)

        xs.append(x)
        x = self.middle_layers(x)
        x = x + self.middle_skip(xs.pop())

        for block, skip in zip(self.decoder_blocks, self.decoder_skips):
            x = block(x)
            x = x + skip(xs.pop())

        x = self.out_conv(x)

        return x + self.input_skip(input)


def get_model(width=16, input_channels=2):
    return AudioUNet(width, input_channels=input_channels, output_channels=1, n_res_units=3)


if __name__ == '__main__':
    model = get_model()

    print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    a = torch.zeros(3, 1, 16384)
    b = torch.zeros(3, 1, 16384)
    ts = torch.rand((3, 1))
    print(ts.shape)
    with torch.no_grad():
        y = model(a, ts, b)
