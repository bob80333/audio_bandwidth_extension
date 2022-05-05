import torch
import torch.nn as nn


# from this paper: https://arxiv.org/abs/2011.02421
# one-shot conditional audio filtering of arbitrary sounds
# skipping the FiLM layers, as we're just filtering everything but speech out

class ResidualUnit(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1, padding_mode='replicate')),
            nn.ELU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(num_channels, num_channels, kernel_size=(1, 1))),
            nn.ELU(inplace=True),
        )

        self.skip = SkipConnection2d(num_channels)

    def forward(self, x):
        return self.layers(x) + self.skip(x)


class SkipConnection2d(nn.Module):

    def __init__(self, num_channels, output_channels=None):
        super().__init__()

        if output_channels is None:
            output_channels = num_channels

        layers = [
            nn.utils.weight_norm(nn.Conv2d(num_channels, output_channels, kernel_size=(1, 1))),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels, stride, n_res_units=3):
        super().__init__()

        layers = []

        for i in range(n_res_units):
            layers.append(ResidualUnit(channels))

        layers.append(
            nn.utils.weight_norm(
                nn.Conv2d(channels, 2 * channels, kernel_size=(2 * stride, 2 * stride), stride=(stride, stride),
                          padding=stride // 2, padding_mode='replicate')))

        layers.append(nn.ELU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels, stride, n_res_units=3):
        super().__init__()

        layers = []

        layers.append(
            nn.utils.weight_norm(
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=(2 * stride, 2 * stride),
                                   stride=(stride, stride), padding=stride)
            ))
        layers.append(nn.ReplicationPad2d(stride // 2))
        layers.append(nn.ELU(inplace=True))

        for i in range(n_res_units):
            layers.append(ResidualUnit(channels // 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpecUNet(nn.Module):

    def __init__(self, base_channels=32, input_channels=1, output_channels=1, n_res_units=3, device="cpu"):
        super().__init__()

        self.window_size = 2400
        self.hop_size = 600
        self.n_fft = 2400
        self.window = torch.hann_window(self.window_size, device=device)

        # due to magnitue and phase components of the STFT we need to muliply input and output channels by 2
        self.input_channels = input_channels * 2
        self.output_channels = output_channels * 2

        # 16 is max downsampling of model
        self.chunk_multiple = self.hop_size * 16

        # chunk size is multiple, except it's hop_size * (n-1) due to stft
        self.last_chunk_piece = self.hop_size * 15

        # needs to be at least 2x chunk multiple # of samples for things to work correctly, so add last piece
        self.min_chunk_size = self.chunk_multiple + self.last_chunk_piece

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv2d(self.input_channels, base_channels, kernel_size=(7, 7), padding=3,
                          padding_mode='replicate')),
            nn.ELU(inplace=True)
        )

        self.input_skip = SkipConnection2d(self.input_channels,  self.output_channels)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(base_channels, stride=2, n_res_units=n_res_units),
            EncoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units),
            EncoderBlock(base_channels * 4, stride=4, n_res_units=n_res_units),
        ])

        self.middle_layers = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=(7, 7), padding=3,
                                           padding_mode='replicate')),
            nn.ELU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=(7, 7), padding=3,
                                           padding_mode='replicate')),
            nn.ELU(inplace=True),
        )

        self.middle_skip = SkipConnection2d(base_channels * 8)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(base_channels * 8, stride=4, n_res_units=n_res_units),
            DecoderBlock(base_channels * 4, stride=2, n_res_units=n_res_units),
            DecoderBlock(base_channels * 2, stride=2, n_res_units=n_res_units),
        ])

        self.decoder_skips = nn.ModuleList([
            SkipConnection2d(base_channels * 4),
            SkipConnection2d(base_channels * 2),
            SkipConnection2d(base_channels),
        ])

        self.out_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv2d(base_channels,  self.output_channels, kernel_size=(7, 7), padding=3,
                          padding_mode='replicate'))
        )

    def forward(self, input):
        n_batch, n_channel, input, leftover, padding = self.stft(input)

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

        output = x + self.input_skip(input)
        return self.istft(n_batch, n_channel, output, leftover, padding)

    def stft(self, input):
        n_batch, n_channel, samples = input.shape

        # calculate padding from # of samples

        # if # of samples below minimum:
        if samples < self.min_chunk_size:
            padding = self.min_chunk_size - samples

        else:
            # calculate samples to add by doing remainder on min chunk sub-piece

            remainder = samples % self.chunk_multiple

            # remove last chunk piece (since chunks must be hop_size * (n-1) where n is a multiple of 16)
            # so did remainder with hop_size * 16

            # now handling last piece hop_size * 15 (which is 16-1)
            padding = -remainder + self.last_chunk_piece

            # if padding is negative, then increment by min chunk sub-piece to find amount of padding to add
            # negative is # of samples to remove, and we're not removing any samples
            if padding < 0:
                padding += self.chunk_multiple

        zeros = torch.zeros(n_batch, n_channel, padding).to(input.device)

        # pad samples dim with zeros
        input = torch.cat([input, zeros], dim=-1)

        x = input.view(n_batch * n_channel, -1)

        spec = torch.stft(x, self.n_fft, self.hop_size, self.window_size, self.window)

        # shape of spec is: batch*channels, n_fft//2 + 1, length//hop_size + 1, 2
        # example input: batch [8, 1, 18600] 18600 samples (31 * 600), 600 is hop size, window/fft is 2400
        # example spec shape: [8, 1201, 32, 2]

        # take last 1088 (1024 + 64) channels instead of first 1088 channels
        # this modification makes more sense for audio bandwidth extension, where we need to work with higher frequencies
        spec_input = spec[:, 113:]  # spec input shape [8, 1088, 32, 2]
        spec_input = spec_input.permute(0, 3, 1, 2)  # [8, 2, 1088, 32]
        spec_leftover = spec[:, :113]  # [8, 113, 32, 2]

        return n_batch, n_channel, spec_input, spec_leftover, padding

    def istft(self, n_batch, n_channel, spec_output, spec_leftover, padding):
        # spec_output shape: [8, 2, 1088, 32]
        spec_output = spec_output.permute(0, 2, 3, 1)  # shape of [8, 1088, 32, 2]

        spec = torch.cat([spec_leftover, spec_output], dim=1)  # shape of [8, 1201, 32, 2]

        audio = torch.istft(spec, self.n_fft, self.hop_size, self.window_size, self.window)

        # audio shape is [8, 18600]
        # split channels and batch

        audio = audio.view(n_batch, n_channel, -1)
        # now [8, 1, 18600]

        # remove extra padding
        audio = audio[:, :, :-padding]
        return audio


def get_model(width=16, device="cpu", output_channels=1):
    return SpecUNet(width, input_channels=1, output_channels=output_channels, n_res_units=3, device=device)


if __name__ == '__main__':
    model = get_model()

    print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    a = torch.zeros(1, 1, 27000)
    with torch.no_grad():
        y = model(a)

    print(y.shape)
