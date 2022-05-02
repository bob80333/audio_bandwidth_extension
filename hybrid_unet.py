import torch
import torch.nn as nn

from spec_unet import get_model as get_spec_unet
from audio_unet import get_model as get_audio_unet


class HybridUnet(nn.Module):

    def __init__(self, spec_unet, wav_unet):
        super().__init__()
        self.spec_unet = spec_unet
        self.wav_unet = wav_unet

    def forward(self, input):
        return self.wav_unet(self.spec_unet(input))


def get_model(width=16, device='cpu'):
    return HybridUnet(get_spec_unet(width=width, device=device),
                      get_audio_unet(width=width))


if __name__ == '__main__':
    model = get_model()

    print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    a = torch.zeros(1, 1, 16384)
    with torch.no_grad():
        y = model(a)

    print(y.shape)
