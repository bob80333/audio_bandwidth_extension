import torch
import torch.nn as nn
import mup


# from this paper: https://arxiv.org/abs/2011.02421
# one-shot conditional audio filtering of arbitrary sounds
# skipping the FiLM layers, as we're just filtering everything but speech out

class ResidualUnit(nn.Module):

    def __init__(self, num_channels, dilation):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=(3,), dilation=dilation, padding=dilation,
                      padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=(1,)),
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

        self.skip = nn.Conv1d(num_channels, output_channels, kernel_size=(1,))

    def forward(self, x):
        return self.skip(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels, stride, n_res_units=3):
        super().__init__()

        layers = []

        for i in range(n_res_units):
            layers.append(ResidualUnit(channels, 3 ** i))

        layers.append(
            nn.Conv1d(channels, 2 * channels, kernel_size=(2 * stride,), stride=(stride,), padding=stride // 2,
                      padding_mode='replicate'))

        layers.append(nn.ELU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels, stride, n_res_units=3):
        super().__init__()

        layers = [
            nn.ConvTranspose1d(channels, channels // 2, kernel_size=(2 * stride,), stride=(stride,), padding=stride),
            nn.ReplicationPad1d(stride // 2), nn.ELU(inplace=True)
        ]

        for i in range(n_res_units):
            layers.append(ResidualUnit(channels // 2, 3 ** i))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MupAudioUNet(nn.Module):

    def __init__(self, base_channels=32, input_channels=1, output_channels=1, n_res_units=3):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=(7,), padding=3, padding_mode='replicate'),
            nn.ELU(inplace=True)
        )

        self.input_skip = SkipConnection(input_channels, output_channels)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(base_channels, stride=2, n_res_units=n_res_units),
            EncoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units),
            EncoderBlock(base_channels * 4, stride=8, n_res_units=n_res_units),
            EncoderBlock(base_channels * 8, stride=8, n_res_units=n_res_units),
        ])

        self.middle_layers = nn.Sequential(
            nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=(7,), padding=3, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=(7,), padding=3, padding_mode='replicate'),
            nn.ELU(inplace=True),
        )

        self.middle_skip = SkipConnection(base_channels * 16)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(base_channels * 16, stride=8, n_res_units=n_res_units),
            DecoderBlock(base_channels * 8, stride=8, n_res_units=n_res_units),
            DecoderBlock(base_channels * 4, stride=2, n_res_units=n_res_units),
            DecoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units),
        ])

        self.decoder_skips = nn.ModuleList([
            SkipConnection(base_channels * 8),
            SkipConnection(base_channels * 4),
            SkipConnection(base_channels * 2),
            SkipConnection(base_channels),
        ])

        self.out_conv = nn.Sequential(
            mup.MuOutConv1d(in_channels=base_channels, out_channels=output_channels, kernel_size=7, padding=3,
                            padding_mode='replicate'),
        )

    def forward(self, input):
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


# testing purposes, mup coord checks, less layers on the chart
class VeryShallowMupAudioUNet(nn.Module):

    def __init__(self, base_channels=2, input_channels=1, output_channels=1, n_res_units=1, ismup=True):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=(7,), padding=3, padding_mode='replicate'),
            nn.ELU(inplace=True)
        )

        self.input_skip = SkipConnection(input_channels, output_channels)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(base_channels, stride=2, n_res_units=n_res_units),
        ])

        self.middle_layers = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=(7,), padding=3, padding_mode='replicate'),
            # nn.GroupNorm(16, base_channels * 16),
            nn.ELU(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=(7,), padding=3, padding_mode='replicate'),
            nn.ELU(inplace=True),
        )

        self.middle_skip = SkipConnection(base_channels * 2)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units),
        ])

        self.decoder_skips = nn.ModuleList([
            SkipConnection(base_channels),
        ])

        if ismup:
            self.out_conv = nn.Sequential(
                mup.MuReadout(in_features=base_channels, out_features=output_channels),
            )

        else:
            self.out_conv = nn.Sequential(
                nn.Linear(in_features=base_channels, out_features=output_channels),
            )

    def forward(self, input):
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

        # shape of x is (batch_size, channels, time)
        # out is linear layer so needs input shape of (batch_size, time, channels)
        # return to (batch_size, channels, time) after out_conv
        x = self.out_conv(x.transpose(1, 2)).transpose(1, 2)

        return x + self.input_skip(input)

    def remove_all_wn(self):
        for m in self.modules():
            if ('weight_g' in m._parameters) or ('weight_v' in m._parameters):
                torch.nn.utils.remove_weight_norm(m)


def get_model(width=16, input_channels=1, n_res_units=3):
    return MupAudioUNet(width, input_channels=input_channels, output_channels=1, n_res_units=n_res_units)


if __name__ == '__main__':
    model = get_model()

    print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    a = torch.zeros(1, 1, 16384)
    with torch.no_grad():
        y = model(a)
